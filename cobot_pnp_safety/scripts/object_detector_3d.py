#!/usr/bin/env python3
"""
Pub
  - /target_object_pose        첫 번째 검출 물체의 중심 위치
  - /detected_objects_markers  모든 검출 물체의 Bounding Box + 좌표 라벨
  - /object_pointcloud         RANSAC으로 테이블을 제거한 물체 점군

좌표계:
  MuJoCo ceiling_camera 위치:
      world = (0, 0.8, 2.5)

  ROS optical frame:
      X = 오른쪽
      Y = 아래
      Z = 카메라 전방

  ceiling_camera_optical_frame -> link0:
      Xopt -> +Xworld
      Yopt -> -Yworld
      Zopt -> -Zworld
"""
import math
import argparse
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped,PointStamped
from visualization_msgs.msg import Marker,MarkerArray
from sensor_msgs.msg import PointCloud2,PointField
from tf2_ros import Buffer,TransformListener
import tf2_geometry_msgs

import mujoco

PROJECT_ROOT=Path(__file__).resolve().parent.parent
SCENE_XML=PROJECT_ROOT/"scene/panda_test.xml"
MODEL_DIR=PROJECT_ROOT/"model/franka_emika_panda"


def create_pointcloud2_msg(stamp,frame_id,points):
    msg=PointCloud2()
    msg.header.stamp=stamp
    msg.header.frame_id=frame_id
    msg.height=1
    msg.width=len(points)
    msg.fields=[
        PointField(name="x",offset=0,datatype=PointField.FLOAT32,count=1),
        PointField(name="y",offset=4,datatype=PointField.FLOAT32,count=1),
        PointField(name="z",offset=8,datatype=PointField.FLOAT32,count=1)
    ]
    msg.is_bigendian=False
    msg.point_step=12
    msg.row_step=12*len(points)
    msg.is_dense=True
    msg.data=points.astype(np.float32).tobytes()
    return msg


def ransac_plane_segmentation(points,distance_threshold=0.015,max_iterations=150):
    if len(points)<50:
        return None,None,None

    best_inliers=[]
    best_plane=None

    for _ in range(max_iterations):
        idx=np.random.choice(len(points),3,replace=False)
        p1,p2,p3=points[idx]

        v1=p2-p1
        v2=p3-p1
        normal=np.cross(v1,v2)
        norm_len=np.linalg.norm(normal)

        if norm_len<1e-6:
            continue

        normal=normal/norm_len
        d=-np.dot(normal,p1)

        distances=np.abs(np.dot(points,normal)+d)
        inliers=np.where(distances<distance_threshold)[0]

        if len(inliers)>len(best_inliers):
            best_inliers=inliers
            best_plane=(normal,d)

    if best_plane is None or len(best_inliers)<50:
        return None,None,None

    normal,d=best_plane

    # 카메라가 위에서 아래를 바라보므로
    # ROS optical Z가 카메라 전방(아래쪽)이다.
    #
    # 물체는 테이블보다 카메라 방향에 있으므로
    # object point가 plane normal의 +방향에 오도록 설정한다.
    if normal[2]>0:
        normal=-normal
        d=-d

    signed_distances=np.dot(points,normal)+d

    object_indices=np.where(
        (signed_distances>distance_threshold)&
        (signed_distances<0.40)
    )[0]

    object_points=points[object_indices]

    return best_plane,best_inliers,object_points


def build_vfs():
    vfs_assets={}

    world_xml=PROJECT_ROOT/"world"/"test.xml"
    if world_xml.exists():
        vfs_assets["../world/test.xml"]=world_xml.read_bytes()

    panda_xml=MODEL_DIR/"panda.xml"
    if panda_xml.exists():
        vfs_assets["../model/franka_emika_panda/panda.xml"]=panda_xml.read_bytes()

    assets_dir=MODEL_DIR/"assets"

    if assets_dir.exists():
        for file_path in assets_dir.rglob("*"):
            if not file_path.is_file():
                continue

            rel_path=file_path.relative_to(assets_dir)
            vfs_path=f"assets/{str(rel_path).replace(chr(92),'/')}"

            if vfs_path not in vfs_assets:
                vfs_assets[vfs_path]=file_path.read_bytes()

    return vfs_assets


def load_mujoco_scene():
    print(f"[INFO] Loading MJCF Scene: {SCENE_XML}")

    try:
        model=mujoco.MjModel.from_xml_path(str(SCENE_XML))
        print("[INFO] MJCF loaded with from_xml_path.")
        return model
    except Exception as e:
        print(f"[WARN] from_xml_path failed: {e}")

    print("[INFO] Trying VFS fallback...")

    xml_string=SCENE_XML.read_text(encoding="utf-8")
    vfs_assets=build_vfs()

    print(f"[INFO] VFS files: {len(vfs_assets)}")

    try:
        model=mujoco.MjModel.from_xml_string(
            xml_string,
            assets=vfs_assets
        )
        print("[INFO] MJCF loaded with VFS.")
        return model
    except Exception as e:
        print(f"[ERROR] MuJoCo VFS scene load failed: {e}")

        for name in sorted(vfs_assets.keys()):
            print(f"  - {name}")

        raise


class Ransac3DObjectDetector(Node):
    def __init__(self,camera_name="ceiling_camera",img_width=640,img_height=480):
        super().__init__("ransac_3d_object_detector")

        self.camera_name=camera_name
        self.width=img_width
        self.height=img_height

        self.tf_buffer=Buffer()
        self.tf_listener=TransformListener(self.tf_buffer,self)

        self.pose_pub=self.create_publisher(
            PoseStamped,
            "/target_object_pose",
            10
        )

        self.marker_pub=self.create_publisher(
            MarkerArray,
            "/detected_objects_markers",
            10
        )

        self.cloud_pub=self.create_publisher(
            PointCloud2,
            "/object_pointcloud",
            10
        )

        self.model=load_mujoco_scene()
        self.data=mujoco.MjData(self.model)

        self.renderer=mujoco.Renderer(
            self.model,
            height=self.height,
            width=self.width
        )

        cam_id=mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.camera_name
        )

        if cam_id==-1:
            raise RuntimeError(
                f"Camera '{self.camera_name}' not found in MJCF."
            )

        self.cam_id=cam_id

        fovy=float(self.model.cam_fovy[cam_id])

        self.fy=(self.height/2.0)/math.tan(
            math.radians(fovy/2.0)
        )

        self.fx=self.fy*(self.width/self.height)

        self.cx=(self.width-1)/2.0
        self.cy=(self.height-1)/2.0

        self.get_logger().info(
            f"Camera: {self.camera_name}, "
            f"fovy={fovy:.2f}, "
            f"fx={self.fx:.2f}, "
            f"fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, "
            f"cy={self.cy:.2f}"
        )

        # MuJoCo camera depth buffer -> metric depth
        #
        # MuJoCo renderer의 depth는 0~1의 비선형 depth buffer이다.
        # 따라서 그대로 z로 사용하면 안 된다.
        self.znear=float(self.model.vis.map.znear)
        self.zfar=float(self.model.vis.map.zfar)

        self.get_logger().info(
            f"Depth range: znear={self.znear:.4f}, "
            f"zfar={self.zfar:.4f}"
        )

        self.stride=2

        u_coords,v_coords=np.meshgrid(
            np.arange(0,self.width,self.stride),
            np.arange(0,self.height,self.stride)
        )

        self.u_flat=u_coords.flatten().astype(np.float64)
        self.v_flat=v_coords.flatten().astype(np.float64)

        self.u_factor=(self.u_flat-self.cx)/self.fx
        self.v_factor=(self.v_flat-self.cy)/self.fy

        self.timer=self.create_timer(
            0.1,
            self.process_detection
        )

        self.get_logger().info(
            "3D RANSAC + DBSCAN Object Detector started."
        )

    def depth_buffer_to_metric(self,depth):
        """
        MuJoCo depth buffer [0,1]을 실제 카메라 거리(m)로 변환한다.

        MuJoCo depth buffer:
            0 -> near
            1 -> far

        변환:
            z = near * far /
                (far - depth * (far - near))
        """

        depth=np.asarray(depth,dtype=np.float64)

        denominator=(
            self.zfar-
            depth*(self.zfar-self.znear)
        )

        metric=np.zeros_like(depth)

        valid=denominator>1e-12

        metric[valid]=(
            self.znear*self.zfar/
            denominator[valid]
        )

        return metric

    def generate_point_cloud(self,depth):
        """
        MuJoCo depth image -> ROS camera optical frame point cloud.

        ROS optical frame:
            X = right
            Y = down
            Z = forward

        천장 카메라:
            forward = World -Z

        여기서 z는 '카메라 전방 거리'이다.
        """

        depth_metric=self.depth_buffer_to_metric(depth)

        depth_sampled=depth_metric[
            ::self.stride,
            ::self.stride
        ].flatten()

        valid_mask=(
            np.isfinite(depth_sampled)&
            (depth_sampled>0.3)&
            (depth_sampled<2.3)
        )

        z=depth_sampled[valid_mask]

        x=self.u_factor[valid_mask]*z
        y=self.v_factor[valid_mask]*z

        points=np.column_stack((x,y,z))

        return points

    def process_detection(self):
        self.get_logger().info(
            "process_detection called",
            throttle_duration_sec=1.0
        )

        try:
            mujoco.mj_forward(
                self.model,
                self.data
            )

            self.renderer.update_scene(
                self.data,
                camera=self.camera_name
            )

            self.renderer.enable_depth_rendering()

            depth_buffer=self.renderer.render()

            self.renderer.disable_depth_rendering()

            depth_metric=self.depth_buffer_to_metric(
                depth_buffer
            )

            valid_depth=(
                (depth_metric>0.3)&
                (depth_metric<3.5)&
                np.isfinite(depth_metric)
            )

            if np.any(valid_depth):
                self.get_logger().info(
                    f"depth buffer min={np.min(depth_buffer):.5f},"
                    f"max={np.max(depth_buffer):.5f},"
                    f"metric min={np.min(depth_metric[valid_depth]):.3f},"
                    f"max={np.max(depth_metric[valid_depth]):.3f},"
                    f"valid={np.count_nonzero(valid_depth)}",
                    throttle_duration_sec=1.0
                )

            points=self.generate_point_cloud(
                depth_buffer
            )

            if len(points)<100:
                return

            plane,inliers,object_points=(
                ransac_plane_segmentation(
                    points,
                    distance_threshold=0.015,
                    max_iterations=100
                )
            )

            num_obj_pts=(
                len(object_points)
                if object_points is not None
                else 0
            )

            self.get_logger().info(
                f"[RANSAC] Total points: {len(points)}, "
                f"Inliers: {len(inliers) if inliers is not None else 0}, "
                f"Object points: {num_obj_pts}",
                throttle_duration_sec=1.0
            )

            if object_points is None or num_obj_pts<10:
                return

            stamp=self.get_clock().now().to_msg()

            camera_optical_frame=(
                f"{self.camera_name}_optical_frame"
            )

            cloud_msg=create_pointcloud2_msg(
                stamp,
                camera_optical_frame,
                object_points
            )

            self.cloud_pub.publish(
                cloud_msg
            )

            clustering=DBSCAN(
                eps=0.06,
                min_samples=8
            ).fit(object_points)

            labels=clustering.labels_

            unique_labels=set(labels)

            if -1 in unique_labels:
                unique_labels.remove(-1)

            if not unique_labels:
                return

            marker_array=MarkerArray()

            target_frame="link0"

            valid_cluster_idx=0

            for label in sorted(unique_labels):

                cluster_pts=object_points[
                    labels==label
                ]

                if len(cluster_pts)<8:
                    continue

                min_bound=np.min(
                    cluster_pts,
                    axis=0
                )

                max_bound=np.max(
                    cluster_pts,
                    axis=0
                )

                center=np.mean(
                    cluster_pts,
                    axis=0
                )

                size=max_bound-min_bound

                pt_cam=PointStamped()

                # 최신 TF 사용
                pt_cam.header.stamp=(
                    rclpy.time.Time().to_msg()
                )

                pt_cam.header.frame_id=(
                    camera_optical_frame
                )

                pt_cam.point.x=float(center[0])
                pt_cam.point.y=float(center[1])
                pt_cam.point.z=float(center[2])

                try:
                    pt_robot=self.tf_buffer.transform(
                        pt_cam,
                        target_frame,
                        timeout=rclpy.duration.Duration(
                            seconds=0.1
                        )
                    )

                except Exception as e:
                    self.get_logger().warn(
                        f"TF Transform failed "
                        f"({camera_optical_frame} -> "
                        f"{target_frame}): {e}",
                        throttle_duration_sec=2.0
                    )
                    continue

                if valid_cluster_idx==0:

                    pose_msg=PoseStamped()

                    pose_msg.header.frame_id=(
                        target_frame
                    )

                    pose_msg.header.stamp=stamp

                    pose_msg.pose.position=(
                        pt_robot.point
                    )

                    pose_msg.pose.orientation.w=1.0

                    self.pose_pub.publish(
                        pose_msg
                    )

                    self.get_logger().info(
                        f"[3D DETECT] Target Object -> "
                        f"X:{pt_robot.point.x:.3f} "
                        f"Y:{pt_robot.point.y:.3f} "
                        f"Z:{pt_robot.point.z:.3f} "
                        f"Size:{size[0]:.2f}x"
                        f"{size[1]:.2f}x"
                        f"{size[2]:.2f}",
                        throttle_duration_sec=1.0
                    )

                box_marker=Marker()

                box_marker.header.frame_id=(
                    target_frame
                )

                box_marker.header.stamp=stamp

                box_marker.ns="object_bbox"

                box_marker.id=(
                    valid_cluster_idx*2
                )

                box_marker.type=Marker.CUBE

                box_marker.action=Marker.ADD

                box_marker.pose.position=(
                    pt_robot.point
                )

                box_marker.pose.orientation.w=1.0

                box_marker.scale.x=max(
                    float(size[0]),
                    0.04
                )

                box_marker.scale.y=max(
                    float(size[1]),
                    0.04
                )

                box_marker.scale.z=max(
                    float(size[2]),
                    0.04
                )

                box_marker.color.r=0.1
                box_marker.color.g=0.8
                box_marker.color.b=0.2
                box_marker.color.a=0.6

                marker_array.markers.append(
                    box_marker
                )

                text_marker=Marker()

                text_marker.header.frame_id=(
                    target_frame
                )

                text_marker.header.stamp=stamp

                text_marker.ns="object_label"

                text_marker.id=(
                    valid_cluster_idx*2+1
                )

                text_marker.type=(
                    Marker.TEXT_VIEW_FACING
                )

                text_marker.action=Marker.ADD

                text_marker.pose.position.x=(
                    pt_robot.point.x
                )

                text_marker.pose.position.y=(
                    pt_robot.point.y
                )

                text_marker.pose.position.z=(
                    pt_robot.point.z+0.1
                )

                text_marker.pose.orientation.w=1.0

                text_marker.scale.z=0.04

                text_marker.color.r=1.0
                text_marker.color.g=1.0
                text_marker.color.b=1.0
                text_marker.color.a=1.0

                text_marker.text=(
                    f"Obj_{valid_cluster_idx} "
                    f"({pt_robot.point.x:.2f},"
                    f"{pt_robot.point.y:.2f},"
                    f"{pt_robot.point.z:.2f})"
                )

                marker_array.markers.append(
                    text_marker
                )

                valid_cluster_idx+=1

            delete_marker=Marker()
            delete_marker.header.frame_id=target_frame
            delete_marker.header.stamp=stamp
            delete_marker.action=(
                Marker.DELETEALL
            )

            marker_array.markers.insert(
                0,
                delete_marker
            )

            if len(marker_array.markers)>1:
                self.marker_pub.publish(
                    marker_array
                )

        except Exception as e:
            self.get_logger().error(
                f"Detection processing failed: {e}",
                throttle_duration_sec=2.0
            )


def main(args=None):
    parser=argparse.ArgumentParser(
        description="3D RANSAC + DBSCAN Object Detector"
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="ceiling_camera",
        help="MuJoCo camera name"
    )

    cli_args,ros_args=parser.parse_known_args()

    rclpy.init(args=ros_args)

    try:
        node=Ransac3DObjectDetector(
            camera_name=cli_args.camera
        )

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(f"[FATAL] {e}")

    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__=="__main__":
    main()