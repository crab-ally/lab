#!/usr/bin/env python3
"""
Node 2: 3D Fusion Node (2D BBox + Depth Image + 2D LiDAR /scan + TF -> 3D Position)

Subscribes:
    - /detections_2d (std_msgs/msg/String - JSON Format from Node 1)
    - /camera/depth/image_raw
    - /camera/depth/camera_info
    - /scan

TF Transformations:
    - camera_color_optical_frame -> base_link

Publishes:
    - /tracks_3d (std_msgs/msg/String - JSON Format)
      [Fields: track_id, class_name, position [x, y, z], velocity [vx, vy], ppe_ok, confidence, stamp]
    - /perception/debug_markers (visualization_msgs/msg/MarkerArray)
"""

import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from cv_bridge import CvBridge

# TF2
import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from visualization_msgs.msg import Marker, MarkerArray


class EKF3D:
    """DeepSORT track_id별 3D 위치 및 속도 추정 칼만 필터"""
    def __init__(self, initial_x: float, initial_y: float, stamp: float) -> None:
        self.state = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float64) # [x, y, vx, vy]
        self.P = np.diag([0.5, 0.5, 2.0, 2.0])
        self.q_var_pos = 0.1
        self.q_var_vel = 0.5
        self.R = np.diag([0.15, 0.15]) # 측정 노이즈
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.last_stamp = stamp
        self.miss_count = 0

    def predict(self, current_stamp: float) -> None:
        dt = current_stamp - self.last_stamp
        if dt <= 0:
            return
        F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ], dtype=np.float64)
        Q = np.diag([self.q_var_pos * dt, self.q_var_pos * dt, self.q_var_vel * dt, self.q_var_vel * dt])
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        self.last_stamp = current_stamp

    def update(self, z_x: float, z_y: float) -> None:
        z = np.array([z_x, z_y], dtype=np.float64)
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.miss_count = 0


class FusionNode3D(Node):
    def __init__(self) -> None:
        super().__init__('fusion_node_3d')

        self.bridge = CvBridge()

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('target_frame', 'base_link')
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        # ── TF2 Listener Setup ─────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Camera Intrinsics (CameraInfo 수신 전 기본값) ───────────────
        self.fx = 600.0
        self.fy = 600.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_info_received = False

        # ── Latest Data Storage ────────────────────────────────────────
        self.latest_depth_img: Optional[np.ndarray] = None
        self.latest_depth_encoding: str = "16UC1"
        self.latest_scan: Optional[LaserScan] = None
        # CameraInfo header.frame_id 기준으로 설정 (depth frame_id와 다를 수 있으므로 여기서 고정)
        self.camera_frame_id: str = "camera_color_optical_frame"

        # EKF Trackers
        self.track_ekf_map: Dict[int, EKF3D] = {}

        # ── QoS Profile ───────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # ── Subscriptions & Publishers ─────────────────────────────────
        self.sub_detections = self.create_subscription(
            String, '/detections_2d', self._detections_callback, 10
        )
        self.sub_depth = self.create_subscription(
            Image, '/camera/depth/image_raw', self._depth_callback, sensor_qos
        )
        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self._camera_info_callback, 10
        )
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, sensor_qos
        )

        self.pub_tracks_3d = self.create_publisher(String, '/tracks_3d', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/perception/debug_markers', 10)

        self.get_logger().info('Node 2: 3D Fusion Node (Depth + /scan + TF) is ready.')

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            # TF 트리에 등록된 frame과 일치시키기 위해 CameraInfo의 frame_id로 고정
            if msg.header.frame_id:
                self.camera_frame_id = msg.header.frame_id
            self.camera_info_received = True
            self.get_logger().info(
                f'Camera Intrinsics Loaded: fx={self.fx:.1f}, fy={self.fy:.1f}, '
                f'cx={self.cx:.1f}, cy={self.cy:.1f}, frame_id={self.camera_frame_id}'
            )

    def _depth_callback(self, msg: Image) -> None:
        try:
            # camera_frame_id는 _camera_info_callback에서 고정 설정하므로 여기서 덮어쓰지 않음
            if msg.encoding in ['16UC1', 'mono16']:
                self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self.latest_depth_encoding = '16UC1'
            elif msg.encoding in ['32FC1']:
                self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                self.latest_depth_encoding = '32FC1'
        except Exception as e:
            self.get_logger().error(f'Depth Image Exception: {e}')

    def _scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def _detections_callback(self, msg: String) -> None:
        if self.latest_depth_img is None:
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON Decode Error: {e}')
            return

        stamp = payload['header']['stamp']
        detections = payload.get('detections', [])

        # TF 변환 검색 (Camera Frame -> Base Link)
        try:
            tf_transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame_id,
                rclpy.time.Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF Lookup Failed ({self.camera_frame_id} -> {self.target_frame}): {e}')
            return

        tracks_3d_payload = []
        active_track_ids = set()

        for det in detections:
            track_id = int(det['track_id'])
            class_name = det['class_name']
            bbox = det['bbox'] # [xmin, ymin, xmax, ymax]
            ppe_ok = det['ppe_ok']
            confidence = det['confidence']

            active_track_ids.add(track_id)

            # 1. Depth Image + 2D BBox로 카메라 좌표계 3D 연산
            cam_x, cam_y, cam_z = self._calculate_3d_from_depth(bbox)
            if cam_z is None:
                continue

            # 2. 2D LiDAR /scan 데이터로 Distance 교차 검증 (보정)
            cam_z = self._refine_with_scan(cam_x, cam_z)

            # 3. TF2로 Camera Frame -> base_link 좌표 변환
            p_cam = PointStamped()
            p_cam.header.frame_id = self.camera_frame_id
            p_cam.point.x = cam_x
            p_cam.point.y = cam_y
            p_cam.point.z = cam_z

            p_base = do_transform_point(p_cam, tf_transform)
            base_x = p_base.point.x
            base_y = p_base.point.y
            base_z = p_base.point.z

            # 4. DeepSORT track_id 기준 EKF 추적
            if track_id not in self.track_ekf_map:
                self.track_ekf_map[track_id] = EKF3D(base_x, base_y, stamp)

            ekf = self.track_ekf_map[track_id]
            ekf.predict(stamp)
            ekf.update(base_x, base_y)

            est_x, est_y, est_vx, est_vy = ekf.state

            item_3d = {
                'track_id': track_id,
                'class_name': class_name,
                'position': [round(float(est_x), 2), round(float(est_y), 2), round(float(base_z), 2)],
                'velocity': [round(float(est_vx), 2), round(float(est_vy), 2)],
                'ppe_ok': ppe_ok,
                'confidence': confidence,
                'stamp': stamp
            }
            tracks_3d_payload.append(item_3d)

        # 오랫동안 미감지된 Track 정리
        for trk_id in list(self.track_ekf_map.keys()):
            if trk_id not in active_track_ids:
                self.track_ekf_map[trk_id].miss_count += 1
                if self.track_ekf_map[trk_id].miss_count > 10:
                    del self.track_ekf_map[trk_id]

        # ── 1. /tracks_3d 토픽 발행 ────────────────────────────────────
        json_msg = String()
        json_msg.data = json.dumps({
            'header': {'stamp': stamp, 'frame_id': self.target_frame},
            'tracks': tracks_3d_payload
        }, ensure_ascii=False)
        self.pub_tracks_3d.publish(json_msg)

        # ── 2. Debug Marker 발행 ──────────────────────────────────────
        self._publish_markers(tracks_3d_payload)

    def _calculate_3d_from_depth(self, bbox: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """2D BBox 중앙 ROI 영역의 Depth 데이터를 3D 좌표로 변환"""
        h_img, w_img = self.latest_depth_img.shape[:2]
        xmin, ymin, xmax, ymax = map(int, bbox)

        # ROI 좁히기 (BBox 중앙 50% 영역만 사용해 경계선 노이즈 제거)
        cx_box = (xmin + xmax) / 2.0
        cy_box = (ymin + ymax) / 2.0
        w_box = (xmax - xmin) * 0.5
        h_box = (ymax - ymin) * 0.5

        rx1 = max(0, int(cx_box - w_box / 2))
        rx2 = min(w_img, int(cx_box + w_box / 2))
        ry1 = max(0, int(cy_box - h_box / 2))
        ry2 = min(h_img, int(cy_box + h_box / 2))

        if rx2 <= rx1 or ry2 <= ry1:
            return None, None, None

        depth_roi = self.latest_depth_img[ry1:ry2, rx1:rx2]

        # 단위 정규화 (미터 단위)
        if self.latest_depth_encoding == '16UC1':
            valid_depths = depth_roi[depth_roi > 0] / 1000.0
        else:
            valid_depths = depth_roi[~np.isnan(depth_roi) & (depth_roi > 0.1)]

        if len(valid_depths) == 0:
            return None, None, None

        # Depth 대표값 (중앙값 사용)
        z_cam = float(np.median(valid_depths))
        if z_cam < 0.2 or z_cam > 15.0: # 유효 거리 초과 시 무시
            return None, None, None

        # Pin-hole Camera Model 역투영
        x_cam = (cx_box - self.cx) * z_cam / self.fx
        y_cam = (cy_box - self.cy) * z_cam / self.fy

        return x_cam, y_cam, z_cam

    def _refine_with_scan(self, cam_x: float, cam_z: float) -> float:
        """2D LiDAR /scan 데이터로 Depth 센서 측정거리 보정"""
        if self.latest_scan is None:
            return cam_z

        # 카메라의 Horizontal Angle (azimuth) 계산
        angle_rad = math.atan2(cam_x, cam_z)

        # /scan 해상도 내 해당 각도 index 계산
        scan = self.latest_scan
        if angle_rad < scan.angle_min or angle_rad > scan.angle_max:
            return cam_z

        idx = int((angle_rad - scan.angle_min) / scan.angle_increment)
        if 0 <= idx < len(scan.ranges):
            scan_dist = scan.ranges[idx]
            if scan.range_min <= scan_dist <= scan.range_max:
                # Depth와 Scan 오차가 0.5m 이내일 경우 보정 융합 (가중 평균)
                if abs(scan_dist - cam_z) < 0.5:
                    return 0.7 * cam_z + 0.3 * scan_dist

        return cam_z

    def _publish_markers(self, tracks: List[dict]) -> None:
        """RViz2 시각화 마커 발행"""
        marker_array = MarkerArray()

        for trk in tracks:
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "tracks_3d"
            marker.id = trk['track_id']
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            pos = trk['position']
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0

            if trk['class_name'] == 'person':
                marker.scale.x, marker.scale.y, marker.scale.z = 0.6, 0.6, 1.7
            else:
                marker.scale.x, marker.scale.y, marker.scale.z = 2.0, 1.2, 1.8

            if trk['class_name'] == 'person' and not trk['ppe_ok']:
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 0.8
            else:
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.8, 0.2, 0.8

            marker_array.markers.append(marker)

            # Text Marker
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "tracks_3d_text"
            text_marker.id = trk['track_id'] + 10000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(pos[0])
            text_marker.pose.position.y = float(pos[1])
            text_marker.pose.position.z = float(pos[2]) + 1.2
            
            vx, vy = trk['velocity']
            text_marker.text = f"ID:{trk['track_id']} ({trk['class_name']})\nV:{math.hypot(vx, vy):.1f}m/s"
            text_marker.scale.z = 0.4
            text_marker.color.r, text_marker.color.g, text_marker.color.b, text_marker.color.a = 1.0, 1.0, 1.0, 1.0

            marker_array.markers.append(text_marker)

        self.pub_markers.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FusionNode3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('3D Fusion Node Stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()