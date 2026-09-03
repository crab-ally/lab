#!/usr/bin/env python3
"""
3D RANSAC + DBSCAN Object Detector for Franka Emika Panda

Input:
  - /camera/depth/image_raw
  - /camera/depth/camera_info
  - /camera/segmentation/image_raw

TF:
  - ceiling_camera_optical_frame -> link0

Output:
  - /target_object_pose
  - /object_height
  - /detected_objects_markers
  - /object_pointcloud
"""

import math
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

POSE_LOG_INTERVAL=1.0


class Ransac3DObjectDetector(Node):
    def __init__(self):
        super().__init__("ransac_3d_object_detector")

        self.camera_frame="ceiling_camera_optical_frame"
        self.target_frame="link0"
        self.depth=None
        self.seg=None
        self.depth_stamp=None
        self.fx=self.fy=self.cx=self.cy=None
        self.stride=2

        self.tf_buffer=Buffer()
        self.tf_listener=TransformListener(self.tf_buffer,self)

        self.pose_pub=self.create_publisher(
            PoseStamped,"/target_object_pose",10)
        self.height_pub=self.create_publisher(
            Float32,"/object_height",10)
        self.marker_pub=self.create_publisher(
            MarkerArray,"/detected_objects_markers",10)
        self.cloud_pub=self.create_publisher(
            PointCloud2,"/object_pointcloud",10)

        self.create_subscription(
            Image,"/camera/depth/image_raw",
            self.depth_callback,10)
        self.create_subscription(
            CameraInfo,"/camera/depth/camera_info",
            self.camera_info_callback,10)
        self.create_subscription(
            Image,"/camera/segmentation/image_raw",
            self.seg_callback,10)

        self.last_pose_log_time=0.0
        self.create_timer(0.1,self.process)

        self.get_logger().info(
            "3D RANSAC + DBSCAN Object Detector started.")


    def camera_info_callback(self,msg):
        if len(msg.k)<9:
            self.get_logger().error("Invalid CameraInfo K matrix.")
            return

        self.fx=float(msg.k[0])
        self.fy=float(msg.k[4])
        self.cx=float(msg.k[2])
        self.cy=float(msg.k[5])


    def depth_callback(self,msg):
        if msg.encoding=="32FC1":
            self.depth=np.frombuffer(
                msg.data,dtype=np.float32
            ).reshape(msg.height,msg.width).copy()

        elif msg.encoding=="16UC1":
            self.depth=np.frombuffer(
                msg.data,dtype=np.uint16
            ).reshape(msg.height,msg.width).astype(
                np.float32
            )/1000.0
        else:
            return

        self.depth_stamp=msg.header.stamp


    def seg_callback(self,msg):
        if msg.encoding!="32SC1":
            return

        self.seg=np.frombuffer(
            msg.data,dtype=np.int32
        ).reshape(msg.height,msg.width).copy()


    def depth_to_points(self):
        d=self.depth[::self.stride,::self.stride]
        s=self.seg[::self.stride,::self.stride]

        # 2D Segmentation Mask Erosion
        # 외곽 노이즈와 배경/테이블 픽셀 혼입 제거
        s_mask=(s!=0).astype(np.uint8)
        kernel=np.ones((5,5),np.uint8)
        s_eroded=cv2.erode(s_mask,kernel,iterations=1)

        h,w=d.shape
        u,v=np.meshgrid(
            np.arange(0,w*self.stride,self.stride),
            np.arange(0,h*self.stride,self.stride))

        d=d.reshape(-1)
        s_valid=s_eroded.reshape(-1)
        u=u.reshape(-1)
        v=v.reshape(-1)

        valid=(
            np.isfinite(d)
            &(d>0.01)
            &(d<50.0)
            &(s_valid>0)
        )

        z=d[valid]
        x=(u[valid]-self.cx)*z/self.fx
        y=(v[valid]-self.cy)*z/self.fy

        return np.column_stack((x,y,z))


    def ransac_plane(self,points,threshold=0.015,iterations=100):
        if len(points)<50:
            return None,None

        best_mask=None
        best_count=0

        for _ in range(iterations):
            p1,p2,p3=points[
                np.random.choice(len(points),3,replace=False)]

            n=np.cross(p2-p1,p3-p1)
            norm=np.linalg.norm(n)

            if norm<1e-6:
                continue

            n/=norm
            d=-np.dot(n,p1)

            mask=np.abs(points@n+d)<threshold
            count=np.sum(mask)

            if count>best_count:
                best_count=count
                best_mask=mask

        if best_mask is None:
            return None,None

        # RANSAC으로 찾은 테이블 평면과 나머지 물체 점군 분리
        plane_points=points[best_mask]
        object_points=points[~best_mask]

        return object_points,plane_points


    def estimate_table_z(self,plane_points):
        # RANSAC 테이블 평면에서 자동으로 테이블 Z 계산
        if plane_points is None or len(plane_points)<20:
            return None

        return float(np.median(plane_points[:,2]))


    def compute_object_center_and_pose(self,cluster_points,table_z):
        """
        전체 클러스터 점군을 이용한 Object Pose 추정

        X/Y:
          클러스터 점군 평균

        Z:
          테이블과 물체 상단 사이의 실제 높이를 이용한 중심

        object_height:
          테이블과 물체 상단 사이의 높이
        """

        # 전체 점군 크기
        size=cluster_points.max(axis=0)-cluster_points.min(axis=0)

        # X, Y 중심
        final_cx=float(np.mean(cluster_points[:,0]))
        final_cy=float(np.mean(cluster_points[:,1]))

        # 천장 카메라에서는 Z가 작을수록 카메라에 가까운 물체 상단
        object_top_z=float(
            np.percentile(cluster_points[:,2],5))

        # 테이블에서 물체 상단까지의 실제 높이
        object_height=table_z-object_top_z

        # 비정상적인 높이 방지
        object_height=float(
            np.clip(object_height,0.005,1.0))

        # 물체의 높이 방향 중심
        final_cz=object_top_z+object_height/2.0

        center=np.array([
            final_cx,
            final_cy,
            final_cz])

        # Orientation 계산
        pts_2d=cluster_points[:,:2].astype(np.float32)
        rect=cv2.minAreaRect(pts_2d)
        angle=rect[2]

        if rect[1][0]<rect[1][1]:
            angle+=90.0

        yaw_rad=math.radians(angle)

        q_cam=[
            0.0,
            0.0,
            math.sin(yaw_rad/2.0),
            math.cos(yaw_rad/2.0)]

        # Marker용 크기
        size[2]=object_height

        return center,size,q_cam,object_height,object_top_z


    def transform_pose(self,camera_pose_stamped):
        try:
            tf=self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))

            return tf2_geometry_msgs.do_transform_pose_stamped(
                camera_pose_stamped,tf)

        except Exception as e:
            self.get_logger().error(
                f"TF Transform failed: {e}")
            return None


    def publish_pointcloud(self,points):
        msg=PointCloud2()

        msg.header.stamp=(
            self.depth_stamp
            if self.depth_stamp is not None
            else self.get_clock().now().to_msg())

        msg.header.frame_id=self.camera_frame
        msg.height=1
        msg.width=len(points)

        msg.fields=[
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1)]

        msg.is_bigendian=False
        msg.point_step=12
        msg.row_step=12*len(points)
        msg.is_dense=True
        msg.data=points.astype(np.float32).tobytes()

        self.cloud_pub.publish(msg)


    def process(self):
        if self.depth is None or self.seg is None:
            return

        if (
            self.fx is None
            or self.fy is None
            or self.cx is None
            or self.cy is None
        ):
            return

        if self.depth.shape!=self.seg.shape:
            return

        # Depth -> 3D Point Cloud
        points=self.depth_to_points()

        if len(points)<100:
            return

        # RANSAC으로 테이블 평면 검출
        object_points,plane_points=self.ransac_plane(points)

        if object_points is None or plane_points is None:
            return

        if len(object_points)<20 or len(plane_points)<20:
            return

        # 테이블 높이 자동 계산
        table_z=self.estimate_table_z(plane_points)

        if table_z is None:
            return

        # 물체 점군 Publish
        self.publish_pointcloud(object_points)

        # DBSCAN으로 물체별 클러스터링
        labels=DBSCAN(
            eps=0.06,
            min_samples=8
        ).fit_predict(object_points)

        clusters=[
            object_points[labels==i]
            for i in set(labels)
            if i>=0 and np.sum(labels==i)>=20
        ]

        if not clusters:
            return

        markers=MarkerArray()
        stamp=self.depth_stamp

        # 이전 Marker 삭제
        delete=Marker()
        delete.header.frame_id=self.target_frame
        delete.header.stamp=stamp
        delete.action=Marker.DELETEALL
        markers.markers.append(delete)

        published_count=0

        for i,cluster in enumerate(clusters):

            center,size,q_cam,object_height,object_top_z=(
                self.compute_object_center_and_pose(
                    cluster,table_z))

            # 물체 높이 Publish
            height_msg=Float32()
            height_msg.data=float(object_height)
            self.height_pub.publish(height_msg)

            # Camera frame Pose
            cam_pose=PoseStamped()
            cam_pose.header.frame_id=self.camera_frame
            cam_pose.header.stamp=stamp

            cam_pose.pose.position.x=float(center[0])
            cam_pose.pose.position.y=float(center[1])
            cam_pose.pose.position.z=float(center[2])

            cam_pose.pose.orientation.x=q_cam[0]
            cam_pose.pose.orientation.y=q_cam[1]
            cam_pose.pose.orientation.z=q_cam[2]
            cam_pose.pose.orientation.w=q_cam[3]

            # Camera -> link0
            target_pose=self.transform_pose(cam_pose)

            if target_pose is None:
                continue

            px=target_pose.pose.position.x
            py=target_pose.pose.position.y
            pz=target_pose.pose.position.z

            # 첫 번째 클러스터를 Target으로 Publish
            if published_count==0:

                self.pose_pub.publish(target_pose)

                now=(
                    self.get_clock().now().nanoseconds
                    *1e-9)

                if now-self.last_pose_log_time>=POSE_LOG_INTERVAL:

                    self.get_logger().info(
                        f"Target center(link0): "
                        f"({px:.3f}, {py:.3f}, {pz:.3f}) | "
                        f"Object height: {object_height:.3f}")

                    self.last_pose_log_time=now

            # Marker
            marker=Marker()
            marker.header.frame_id=self.target_frame
            marker.header.stamp=stamp
            marker.ns="objects"
            marker.id=i
            marker.type=Marker.CUBE
            marker.action=Marker.ADD
            marker.pose=target_pose.pose

            marker.scale.x=max(
                float(size[0]),0.04)
            marker.scale.y=max(
                float(size[1]),0.04)
            marker.scale.z=max(
                float(object_height),0.005)

            marker.color.r=1.0
            marker.color.g=0.0
            marker.color.b=0.0
            marker.color.a=0.6

            marker.lifetime.sec=0
            marker.lifetime.nanosec=0

            markers.markers.append(marker)
            published_count+=1

        if published_count>0:
            self.marker_pub.publish(markers)


def main():
    rclpy.init()
    node=Ransac3DObjectDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=="__main__":
    main()