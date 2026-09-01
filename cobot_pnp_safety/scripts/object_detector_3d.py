#!/usr/bin/env python3
"""
3D RANSAC + DBSCAN Object Detector for Franka Emika Panda

Subscribed / Used:
  - MuJoCo camera rendering: ceiling_camera (world: 0, 0.8, 2.5)
  - TF: ceiling_camera_optical_frame -> link0

Published:
  - /object_pointcloud         (sensor_msgs/PointCloud2)    : RANSAC으로 테이블을 분리한 물체 점군 (optical frame)
  - /target_object_pose        (geometry_msgs/PoseStamped)  : 첫 번째 검출 물체의 중심 위치 (link0 frame)
  - /detected_objects_markers  (visualization_msgs/MarkerArray) : Bounding Box 및 좌표 텍스트 마커 (link0 frame)
"""
import math
import argparse
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = PROJECT_ROOT / "scene" / "panda_test.xml"
MODEL_DIR = PROJECT_ROOT / "model" / "franka_emika_panda"


def create_pointcloud2_msg(stamp, frame_id, points):
    """numpy array (N, 3)를 ROS2 PointCloud2 메시지로 변환"""
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(points)
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    return msg


def ransac_plane_segmentation(points, distance_threshold=0.015, max_iterations=150):
    """
    RANSAC 알고리즘으로 테이블 평면을 추정하고 평면 위의 물체 포인트들을 분리
    """
    if len(points) < 50:
        return None, None, None

    best_inliers = []
    best_plane = None

    for _ in range(max_iterations):
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)

        if norm_len < 1e-6:
            continue

        normal = normal / norm_len
        d = -np.dot(normal, p1)

        distances = np.abs(np.dot(points, normal) + d)
        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    if best_plane is None or len(best_inliers) < 50:
        return None, None, None

    normal, d = best_plane

    # 카메라가 위에서 아래를 바라보므로
    # ROS optical Z가 카메라 전방(아래쪽)이다.
    # 물체는 테이블보다 카메라 쪽(작은 Z)에 있으므로
    # object point가 plane normal 기준 카메라 쪽으로 오도록 부호 정렬
    if normal[2] > 0:
        normal = -normal
        d = -d

    signed_distances = np.dot(points, normal) + d

    # 테이블 상판 위 1.5cm ~ 40cm 사이의 점들을 물체 후보로 추출
    object_indices = np.where(
        (signed_distances > distance_threshold) &
        (signed_distances < 0.40)
    )[0]

    object_points = points[object_indices]

    return (normal, d), best_inliers, object_points


def build_vfs():
    """VFS 에셋 빌드"""
    vfs_assets = {}

    world_xml = PROJECT_ROOT / "world" / "test.xml"
    if world_xml.exists():
        vfs_assets["../world/test.xml"] = world_xml.read_bytes()

    panda_xml = MODEL_DIR / "panda.xml"
    if panda_xml.exists():
        vfs_assets["../model/franka_emika_panda/panda.xml"] = panda_xml.read_bytes()

    assets_dir = MODEL_DIR / "assets"
    if assets_dir.exists():
        for file_path in assets_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(assets_dir)
            vfs_path = f"assets/{str(rel_path).replace(chr(92),'/')}"
            if vfs_path not in vfs_assets:
                vfs_assets[vfs_path] = file_path.read_bytes()

    return vfs_assets


def load_mujoco_scene():
    """MuJoCo 모델 로드"""
    print(f"[INFO] Loading MJCF Scene: {SCENE_XML}")
    try:
        model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        print("[INFO] MJCF loaded with from_xml_path.")
        return model
    except Exception as e:
        print(f"[WARN] from_xml_path failed: {e}")

    print("[INFO] Trying VFS fallback...")
    xml_string = SCENE_XML.read_text(encoding="utf-8")
    vfs_assets = build_vfs()
    print(f"[INFO] VFS files: {len(vfs_assets)}")

    try:
        model = mujoco.MjModel.from_xml_string(xml_string, assets=vfs_assets)
        print("[INFO] MJCF loaded with VFS.")
        return model
    except Exception as e:
        print(f"[ERROR] MuJoCo VFS scene load failed: {e}")
        for name in sorted(vfs_assets.keys()):
            print(f"  - {name}")
        raise


class Ransac3DObjectDetector(Node):
    def __init__(self, camera_name="ceiling_camera", img_width=640, img_height=480):
        super().__init__("ransac_3d_object_detector")

        self.camera_name = camera_name
        self.width = img_width
        self.height = img_height

        # TF 리스너
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 토픽 퍼블리셔
        self.pose_pub = self.create_publisher(PoseStamped, "/target_object_pose", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/detected_objects_markers", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/object_pointcloud", 10)

        # MuJoCo 모델 및 렌더러 초기화
        self.model = load_mujoco_scene()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id == -1:
            raise RuntimeError(f"Camera '{self.camera_name}' not found in MJCF scene.")

        self.cam_id = cam_id
        fovy = float(self.model.cam_fovy[cam_id])

        # 카메라 내부 파라미터 계산 (Square pixel 모델: fx == fy)
        self.fy = (self.height / 2.0) / math.tan(math.radians(fovy / 2.0))
        self.fx = self.fy  # 픽셀 종횡비 1.0
        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0

        self.get_logger().info(
            f"[INIT] Camera: '{self.camera_name}', fovy={fovy:.1f}deg, "
            f"fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
        )

        self.stride = 2
        u_coords, v_coords = np.meshgrid(
            np.arange(0, self.width, self.stride),
            np.arange(0, self.height, self.stride)
        )
        self.u_flat = u_coords.flatten().astype(np.float64)
        self.v_flat = v_coords.flatten().astype(np.float64)
        self.u_factor = (self.u_flat - self.cx) / self.fx
        self.v_factor = (self.v_flat - self.cy) / self.fy

        # Panda 로봇(+손목 카메라) geom ID 사전 수집
        self._build_panda_geom_ids()

        # 10Hz 검출 타이머
        self.timer = self.create_timer(0.1, self.process_detection)
        self.get_logger().info("[INIT] 3D RANSAC + DBSCAN Object Detector initialized and running at 10Hz.")

    # ------------------------------------------------------------------
    # Panda Geom ID 수집 및 Segmentation 마스크 생성
    # ------------------------------------------------------------------

    # Panda 로봇 본체와 hand에 부착된 손목 카메라 바디까지 포함
    _PANDA_BODY_NAMES = {
        "link0", "link1", "link2", "link3", "link4",
        "link5", "link6", "link7",
        "hand", "left_finger", "right_finger",
        "wrist_camera_link",  # 손목 카메라 외형 geom (cam_body, cam_lens)
    }

    def _build_panda_geom_ids(self):
        """
        모델 로드 직후 1회 실행.
        Panda 로봇에 속하는 모든 geom의 정수 ID를 self.panda_geom_ids 에 수집.
        body_geomadr / body_geomnum 을 사용하므로 geom에 name 이 없어도 안전.
        """
        ids: set[int] = set()
        for bname in self._PANDA_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid == -1:
                self.get_logger().warn(
                    f"[INIT] Body '{bname}' not found in model – skipping.",
                    throttle_duration_sec=5.0
                )
                continue
            start = int(self.model.body_geomadr[bid])
            count = int(self.model.body_geomnum[bid])
            for gid in range(start, start + count):
                ids.add(gid)
        self.panda_geom_ids = ids
        self.get_logger().info(
            f"[INIT] Panda geom mask: {len(ids)} geoms collected "
            f"from {len(self._PANDA_BODY_NAMES)} bodies."
        )

    def _get_robot_mask(self) -> np.ndarray:
        """
        MuJoCo Segmentation 렌더링으로 Panda geom 픽셀 마스크를 반환.

        Returns
        -------
        mask : np.ndarray, shape (H, W), dtype bool
            True 인 픽셀 = Panda 로봇(또는 손목 카메라)에 해당하는 depth 픽셀.
        """
        if not self.panda_geom_ids:
            return np.zeros((self.height, self.width), dtype=bool)

        # update_scene() 은 이미 호출된 상태 – 동일 프레임에서 재렌더링
        self.renderer.enable_segmentation_rendering()
        seg = self.renderer.render()          # (H, W, 2)  int32
        self.renderer.disable_segmentation_rendering()

        geom_id_map = seg[:, :, 0]           # 채널 0: 픽셀별 geom ID (-1 = 배경)

        # 벡터화: panda_geom_ids 집합에 속하는 픽셀을 한 번에 마스킹
        panda_ids_arr = np.array(list(self.panda_geom_ids), dtype=np.int32)
        mask = np.isin(geom_id_map, panda_ids_arr)
        return mask

    # ------------------------------------------------------------------

    def generate_point_cloud(self, depth_metric):
        """
        카메라 metric depth image -> ROS camera optical frame (X:right, Y:down, Z:forward) point cloud
        """
        depth_sampled = depth_metric[::self.stride, ::self.stride].flatten()

        # 천장 카메라에서 유효한 거리 범위 (0.2m ~ 3.0m)
        valid_mask = (
            np.isfinite(depth_sampled) &
            (depth_sampled > 0.2) &
            (depth_sampled < 2.3)
        )

        z = depth_sampled[valid_mask]
        x = self.u_factor[valid_mask] * z
        y = self.v_factor[valid_mask] * z

        points = np.column_stack((x, y, z))
        return points

    def process_detection(self):
        try:
            # ----------------------------------------------------
            # [단계 1/6] MuJoCo 카메라 씬 업데이트 및 Depth 렌더링
            # ----------------------------------------------------
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=self.camera_name)

            # Depth 렌더링
            self.renderer.enable_depth_rendering()
            depth_image = self.renderer.render()
            self.renderer.disable_depth_rendering()

            # Segmentation으로 Panda 로봇(+손목 카메라) 픽셀 마스킹
            # → 마스킹된 픽셀은 nan 처리되어 generate_point_cloud의
            #   np.isfinite() 필터에서 자동으로 제외됨
            robot_mask = self._get_robot_mask()
            depth_metric = depth_image.astype(np.float64)
            depth_metric[robot_mask] = np.nan

            masked_px = int(robot_mask.sum())
            self.get_logger().debug(
                f"[STEP 1/6] Robot mask applied: {masked_px} px removed."
            )
            d_min, d_max = np.min(depth_metric), np.max(depth_metric)

            # ----------------------------------------------------
            # [단계 2/6] Depth 이미지 -> 3D Point Cloud 변환
            # ----------------------------------------------------
            points = self.generate_point_cloud(depth_metric)

            if len(points) < 100:
                self.get_logger().warn(
                    f"[STEP 2/6] Point cloud too small ({len(points)} points). Check camera FOV and depth range.",
                    throttle_duration_sec=2.0
                )
                return

            # ----------------------------------------------------
            # [단계 3/6] RANSAC 평면 추정 및 테이블 위 물체 점군 분리
            # ----------------------------------------------------
            plane, inliers, object_points = ransac_plane_segmentation(
                points,
                distance_threshold=0.015,
                max_iterations=120
            )

            num_inliers = len(inliers) if inliers is not None else 0
            num_obj_pts = len(object_points) if object_points is not None else 0

            self.get_logger().info(
                f"[STEP 1~3/6] Depth:[{d_min:.2f}m~{d_max:.2f}m] | "
                f"Total Pts: {len(points)} | Table Inliers: {num_inliers} | "
                f"Object Pts: {num_obj_pts}",
                throttle_duration_sec=2.0
            )

            if object_points is None or num_obj_pts < 10:
                self.get_logger().warn(
                    f"[STEP 3/6] No valid object points found on table (num_obj_pts={num_obj_pts}).",
                    throttle_duration_sec=2.0
                )
                return

            # ----------------------------------------------------
            # [단계 4/6] /object_pointcloud 토픽 발행
            # ----------------------------------------------------
            stamp = self.get_clock().now().to_msg()
            camera_optical_frame = f"{self.camera_name}_optical_frame"

            cloud_msg = create_pointcloud2_msg(
                stamp,
                camera_optical_frame,
                object_points
            )
            self.cloud_pub.publish(cloud_msg)

            # ----------------------------------------------------
            # [단계 5/6] DBSCAN 물체 군집화 (Clustering)
            # ----------------------------------------------------
            clustering = DBSCAN(eps=0.06, min_samples=8).fit(object_points)
            labels = clustering.labels_
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)  # 노이즈 제거

            if not unique_labels:
                self.get_logger().warn(
                    "[STEP 5/6] DBSCAN found only noise. No distinct clusters.",
                    throttle_duration_sec=2.0
                )
                return

            # ----------------------------------------------------
            # [단계 6/6] TF 좌표 변환 및 Pose / Marker 토픽 발행
            # ----------------------------------------------------
            marker_array = MarkerArray()
            target_frame = "link0"

            # 기존 마커 클리어 마커 추가
            delete_marker = Marker()
            delete_marker.header.frame_id = target_frame
            delete_marker.header.stamp = stamp
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)

            valid_cluster_idx = 0

            for label in sorted(unique_labels):
                cluster_pts = object_points[labels == label]
                if len(cluster_pts) < 8:
                    continue

                min_bound = np.min(cluster_pts, axis=0)
                max_bound = np.max(cluster_pts, axis=0)
                center = np.mean(cluster_pts, axis=0)
                size = max_bound - min_bound

                # Optical Frame의 포인트를 로봇 베이스(link0) 좌표계로 변환
                pt_cam = PointStamped()
                pt_cam.header.stamp = rclpy.time.Time().to_msg()  # 최신 TF 변환 사용
                pt_cam.header.frame_id = camera_optical_frame
                pt_cam.point.x = float(center[0])
                pt_cam.point.y = float(center[1])
                pt_cam.point.z = float(center[2])

                try:
                    pt_robot = self.tf_buffer.transform(
                        pt_cam,
                        target_frame,
                        timeout=rclpy.duration.Duration(seconds=0.05)
                    )
                except Exception as e:
                    self.get_logger().warn(
                        f"[STEP 6/6 TF ERROR] Transform ({camera_optical_frame} -> {target_frame}) failed: {e}. "
                        f"Ensure mujoco_ros2_bridge.py is running to broadcast TF.",
                        throttle_duration_sec=2.0
                    )
                    continue

                # 첫 번째 검출 물체 -> /target_object_pose 발행
                if valid_cluster_idx == 0:
                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = target_frame
                    pose_msg.header.stamp = stamp
                    pose_msg.pose.position = pt_robot.point
                    pose_msg.pose.orientation.w = 1.0
                    self.pose_pub.publish(pose_msg)

                    self.get_logger().info(
                        f"[PUB -> /target_object_pose] Target pos: "
                        f"X={pt_robot.point.x:.3f}, Y={pt_robot.point.y:.3f}, Z={pt_robot.point.z:.3f} | "
                        f"Size: {size[0]:.2f}x{size[1]:.2f}x{size[2]:.2f}",
                        throttle_duration_sec=2.0
                    )

                # Bounding Box 큐브 마커
                box_marker = Marker()
                box_marker.header.frame_id = target_frame
                box_marker.header.stamp = stamp
                box_marker.ns = "object_bbox"
                box_marker.id = valid_cluster_idx * 2
                box_marker.type = Marker.CUBE
                box_marker.action = Marker.ADD
                box_marker.pose.position = pt_robot.point
                box_marker.pose.orientation.w = 1.0
                box_marker.scale.x = max(float(size[0]), 0.04)
                box_marker.scale.y = max(float(size[1]), 0.04)
                box_marker.scale.z = max(float(size[2]), 0.04)
                box_marker.color.r = 0.1
                box_marker.color.g = 0.8
                box_marker.color.b = 0.2
                box_marker.color.a = 0.6
                marker_array.markers.append(box_marker)

                # 텍스트 라벨 마커
                text_marker = Marker()
                text_marker.header.frame_id = target_frame
                text_marker.header.stamp = stamp
                text_marker.ns = "object_label"
                text_marker.id = valid_cluster_idx * 2 + 1
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = pt_robot.point.x
                text_marker.pose.position.y = pt_robot.point.y
                text_marker.pose.position.z = pt_robot.point.z + 0.1
                text_marker.pose.orientation.w = 1.0
                text_marker.scale.z = 0.04
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                text_marker.text = (
                    f"Obj_{valid_cluster_idx} "
                    f"({pt_robot.point.x:.2f}, {pt_robot.point.y:.2f}, {pt_robot.point.z:.2f})"
                )
                marker_array.markers.append(text_marker)

                valid_cluster_idx += 1

            if len(marker_array.markers) > 1:
                self.marker_pub.publish(marker_array)
                self.get_logger().info(
                    f"[PUB -> /detected_objects_markers] Published {valid_cluster_idx} detected objects markers.",
                    throttle_duration_sec=2.0
                )

        except Exception as e:
            self.get_logger().error(f"Detection processing failed: {e}", throttle_duration_sec=2.0)


def main(args=None):
    parser = argparse.ArgumentParser(description="3D RANSAC + DBSCAN Object Detector")
    parser.add_argument("--camera", type=str, default="ceiling_camera", help="MuJoCo camera name")
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    try:
        node = Ransac3DObjectDetector(camera_name=cli_args.camera)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[FATAL] {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()