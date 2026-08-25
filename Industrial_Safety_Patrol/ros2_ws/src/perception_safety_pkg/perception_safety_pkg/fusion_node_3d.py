#!/usr/bin/env python3
"""
Node 2: 3D Fusion Node (2D BBox + Depth Image + 2D LiDAR /scan + TF -> 3D Position)

Subscribes:
    - /detections_2d (std_msgs/msg/String - JSON Format from Node 1)
    - /camera/depth/image_raw
    - /camera/depth/camera_info
    - /scan

TF Transformations:
    - camera_frame_id (camera_optical_frame) -> target_frame (base_link)
    - camera_frame_id (camera_optical_frame) -> scan_frame_id (lidar_link)

Publishes:
    - /tracks_3d (std_msgs/msg/String - JSON Format)
    - /perception/debug_markers (visualization_msgs/msg/MarkerArray)
"""

from collections import deque
import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
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
        self.state = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float64)  # [x, y, vx, vy]
        self.P = np.diag([0.5, 0.5, 2.0, 2.0])
        
        # 가속도 노이즈 표준편차 (m/s^2)
        self.sigma_a = 1.2
        
        # 측정 노이즈
        self.R = np.diag([0.15, 0.15])
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

        # Continuous White Noise Acceleration Model (Q 행렬 보정)
        dt2 = (dt ** 2) / 2.0
        dt3 = (dt ** 3) / 3.0
        q_var = self.sigma_a ** 2

        Q = np.array([
            [dt3 * q_var, 0.0,         dt2 * q_var, 0.0],
            [0.0,         dt3 * q_var, 0.0,         dt2 * q_var],
            [dt2 * q_var, 0.0,         dt * q_var,  0.0],
            [0.0,         dt2 * q_var, 0.0,         dt * q_var]
        ], dtype=np.float64)

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
        self.declare_parameter('depth_buffer_size', 60)
        self.declare_parameter('max_depth_time_diff', 0.3)

        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.depth_buffer_size = self.get_parameter('depth_buffer_size').get_parameter_value().integer_value
        self.max_depth_time_diff = self.get_parameter('max_depth_time_diff').get_parameter_value().double_value

        # ── TF2 Listener Setup ─────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Camera Intrinsics ─────────────────────────────────────────
        self.fx = 600.0
        self.fy = 600.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_info_received = False

        # ── Depth Image Buffer (Timestamp 동기화용) ───────────────────
        # 원소: (stamp: float, depth_img: np.ndarray, encoding: str)
        self.depth_buffer: deque = deque(maxlen=self.depth_buffer_size)

        self.latest_scan: Optional[LaserScan] = None
        self.latest_scan_stamp: Optional[float] = None
        
        self.camera_frame_id: str = "camera_optical_frame"
        self.scan_frame_id: str = "lidar_link"

        # RViz Marker Cleanup용 이전 ID 저장소
        self.prev_active_marker_ids: set = set()

        # EKF Trackers
        self.track_ekf_map: Dict[int, EKF3D] = {}

        # ── QoS Profile ───────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        depth_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # ── Subscriptions & Publishers ─────────────────────────────────
        self.sub_detections = self.create_subscription(
            String,
            '/detections_2d',
            self._detections_callback,
            10
        )

        self.sub_depth = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self._depth_callback,
            depth_qos
        )

        self.sub_info = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self._camera_info_callback,
            10
        )

        self.sub_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            sensor_qos
        )

        self.pub_tracks_3d = self.create_publisher(
            String,
            '/tracks_3d',
            10
        )

        self.pub_markers = self.create_publisher(
            MarkerArray,
            '/perception/debug_markers',
            10
        )

        self.get_logger().info('Node 2: 3D Fusion Node is ready.')

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            if msg.header.frame_id:
                self.camera_frame_id = msg.header.frame_id
            self.camera_info_received = True
            self.get_logger().info(
                f'Camera Intrinsics Loaded: fx={self.fx:.1f}, fy={self.fy:.1f}, '
                f'cx={self.cx:.1f}, cy={self.cy:.1f}, frame_id={self.camera_frame_id}'
            )

    def _depth_callback(self, msg: Image) -> None:
        try:
            if msg.encoding in ['16UC1', 'mono16']:
                depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                encoding = '16UC1'
            elif msg.encoding in ['32FC1']:
                depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                encoding = '32FC1'
            else:
                self.get_logger().warning(f'Unsupported depth encoding: {msg.encoding}')
                return

            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.depth_buffer.append((stamp, depth_img, encoding))

        except Exception as e:
            self.get_logger().error(f'Depth Image Exception: {e}')

    def _find_matching_depth(self, target_stamp: float) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
        """RGB 이미지 타임스탬프와 가장 일치하는 Depth 이미지를 버퍼에서 검색"""
        if not self.depth_buffer:
            return None, None, None

        best_entry = None
        min_diff = float('inf')

        # 버퍼 내에서 타임스탬프 오차가 가장 작은 프레임 탐색
        for stamp, depth_img, encoding in self.depth_buffer:
            diff = abs(stamp - target_stamp)
            if diff < min_diff:
                min_diff = diff
                best_entry = (depth_img, encoding, stamp)

        # 허용 시간 오차 이내인 경우 반환
        if min_diff <= self.max_depth_time_diff and best_entry is not None:
            # target_stamp보다 1.0초 이상 지난 너무 오래된 버퍼 데이터 정리
            while self.depth_buffer and (target_stamp - self.depth_buffer[0][0]) > 1.0:
                self.depth_buffer.popleft()
            return best_entry[0], best_entry[1], best_entry[2]

        return None, None, None

    def _scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self.latest_scan_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if hasattr(msg, 'header') and msg.header.frame_id:
            self.scan_frame_id = msg.header.frame_id

    def _detections_callback(self, msg: String) -> None:
        if not self.camera_info_received:
            self.get_logger().warning(
                'Camera info not received yet. Skipping 3D fusion.'
            )
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON Decode Error: {e}')
            return

        stamp = payload['header']['stamp']
        detections = payload.get('detections', [])

        # Depth 버퍼에서 RGB 타임스탬프(stamp)에 매칭되는 Depth 이미지 검색
        depth_img, depth_encoding, matched_depth_stamp = self._find_matching_depth(stamp)

        # 타임스탬프 동기화 기반 TF 조회
        sec = int(stamp)
        nanosec = int((stamp - sec) * 1e9)
        lookup_time = Time(seconds=sec, nanoseconds=nanosec)

        try:
            tf_transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame_id,
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            try:
                tf_transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.camera_frame_id,
                    rclpy.time.Time()
                )
            except Exception as e:
                self.get_logger().warning(
                    f'TF Lookup Failed ({self.camera_frame_id} -> {self.target_frame}): {e}'
                )
                return

        tracks_3d_payload = []
        active_track_ids = set()

        has_person = False
        has_forklift = False

        for det in detections:
            track_id = int(det['track_id'])
            class_name = det['class_name']
            bbox = det['bbox']
            ppe_ok = det['ppe_ok']
            confidence = det['confidence']

            # 1. 버퍼에서 매칭된 Depth Image + 2D BBox → 카메라 좌표계 3D 연산
            cam_x, cam_y, cam_z = self._calculate_3d_from_depth(bbox, depth_img, depth_encoding)
            if cam_z is None:
                # Depth 연산 실패 시 active_track_ids에 추가하지 않음 -> EKF miss_count 정상 실시간 증가
                continue

            # 2. 2D LiDAR /scan 데이터로 Distance 교차 검증 (높이 차 검증 및 stamp 전달 포함)
            cls_h = 1.7 if class_name == 'person' else 1.8
            cam_z = self._refine_with_scan(cam_x, cam_y, cam_z, stamp, bbox_height_m=cls_h)

            # 3D 위치 산출 성공 시에만 활성 트랙 ID 목록에 등록
            active_track_ids.add(track_id)

            p_cam = PointStamped()
            p_cam.header.frame_id = self.camera_frame_id  # camera_link
            p_cam.point.x = cam_x
            p_cam.point.y = cam_y
            p_cam.point.z = cam_z

            # 3. TF2로 camera_link → base_link 좌표 변환
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

            if class_name == 'person':
                has_person = True
            elif class_name == 'forklift':
                has_forklift = True

        # 클래스 존재 상태 판별
        if has_person and has_forklift:
            presence_state = 'BOTH'
        elif has_person:
            presence_state = 'PERSON_ONLY'
        elif has_forklift:
            presence_state = 'FORKLIFT_ONLY'
        else:
            presence_state = 'NONE'

        # 오랫동안 미감지된 Track 정리 (미감지 프레임에서도 EKF predict 수행하여 위치 추정 유지)
        for trk_id in list(self.track_ekf_map.keys()):
            if trk_id not in active_track_ids:
                self.track_ekf_map[trk_id].predict(stamp)
                self.track_ekf_map[trk_id].miss_count += 1
                if self.track_ekf_map[trk_id].miss_count > 10:
                    del self.track_ekf_map[trk_id]

        # ── 1. /tracks_3d 토픽 발행 ────────────────────────────────────
        json_msg = String()
        json_msg.data = json.dumps({
            'header': {
                'stamp': stamp,
                'frame_id': self.target_frame
            },
            'class_presence': {
                'person': has_person,
                'forklift': has_forklift,
                'state': presence_state
            },
            'tracks': tracks_3d_payload
        }, ensure_ascii=False)

        self.pub_tracks_3d.publish(json_msg)

        # ── 2. Debug Marker 발행 ──────────────────────────────────────
        self._publish_markers(tracks_3d_payload)

    def _calculate_3d_from_depth(
        self,
        bbox: List[float],
        depth_img: np.ndarray,
        depth_encoding: str
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        h_img, w_img = depth_img.shape[:2]
        xmin, ymin, xmax, ymax = map(int, bbox)

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

        depth_roi = depth_img[ry1:ry2, rx1:rx2]

        if depth_encoding == '16UC1':
            valid_depths = (depth_roi[depth_roi > 0] / 1000.0)
        else:
            valid_depths = depth_roi[~np.isnan(depth_roi) & (depth_roi > 0.1)]

        if len(valid_depths) == 0:
            return None, None, None

        z_cam = float(np.median(valid_depths))

        if z_cam < 0.2 or z_cam > 8.0:
            return None, None, None

        x_cam = (cx_box - self.cx) * z_cam / self.fx
        y_cam = (cy_box - self.cy) * z_cam / self.fy

        return x_cam, y_cam, z_cam

    def _refine_with_scan(self, cam_x: float, cam_y: float, cam_z: float, detection_stamp: float, bbox_height_m: float = 1.7) -> float:
        """2D LiDAR /scan 데이터로 Depth 센서 측정거리 보정 (높이 차이 검증 포함)"""

        # 라이다 데이터가 없다면 그대로 리턴
        if self.latest_scan is None or self.latest_scan_stamp is None:
            return cam_z

        # 라이다 데이터가 너무 오래되었다면 그대로 리턴
        if abs(detection_stamp - self.latest_scan_stamp) > 0.2:
            return cam_z

        scan = self.latest_scan
        scan_frame = getattr(scan.header, 'frame_id', self.scan_frame_id)

        # Camera Frame -> LiDAR Frame 좌표 변환
        p_cam = PointStamped()
        p_cam.header.frame_id = self.camera_frame_id
        p_cam.point.x = float(cam_x)
        p_cam.point.y = float(cam_y)
        p_cam.point.z = float(cam_z)

        try:
            tf_cam_to_scan = self.tf_buffer.lookup_transform(
                scan_frame,
                self.camera_frame_id,
                rclpy.time.Time()
            )
            p_scan_frame = do_transform_point(p_cam, tf_cam_to_scan)
            
            lx = p_scan_frame.point.x
            ly = p_scan_frame.point.y
            lz = p_scan_frame.point.z

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return cam_z

        # LiDAR 수평 스캔 평면(z≈0)이 물체 높이 범위(±half_h) 바깥이면 보정 패스
        half_h = bbox_height_m / 2.0
        if abs(lz) > half_h:
            return cam_z

        angle_rad = math.atan2(ly, lx)
        dist_from_lidar = math.hypot(lx, ly)

        if angle_rad < scan.angle_min or angle_rad > scan.angle_max:
            return cam_z

        idx = int((angle_rad - scan.angle_min) / scan.angle_increment)

        # 주변 빔(Window Search: idx ± 2) 탐색으로 노이즈 및 결측치 방지
        valid_ranges = []
        window_size = 2
        min_idx = max(0, idx - window_size)
        max_idx = min(len(scan.ranges) - 1, idx + window_size)

        for i in range(min_idx, max_idx + 1):
            r = scan.ranges[i]
            if scan.range_min <= r <= scan.range_max and not math.isnan(r) and not math.isinf(r):
                valid_ranges.append(r)

        if valid_ranges:
            scan_dist = float(np.median(valid_ranges))
            if abs(scan_dist - dist_from_lidar) < 0.5:
                scale = (0.7 * dist_from_lidar + 0.3 * scan_dist) / dist_from_lidar
                return cam_z * scale

        return cam_z

    def _publish_markers(self, tracks: List[dict]) -> None:
        """RViz2 시각화 마커 발행 (DELETE Cleanup 포함)"""
        marker_array = MarkerArray()
        current_active_marker_ids = set()

        for trk in tracks:
            track_id = trk['track_id']
            cube_id = track_id
            text_id = track_id + 10000

            current_active_marker_ids.add(cube_id)
            current_active_marker_ids.add(text_id)

            # Cube Marker
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "tracks_3d"
            marker.id = cube_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            pos = trk['position']
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0

            if trk['class_name'] == 'person':
                marker.scale.x = 0.6
                marker.scale.y = 0.6
                marker.scale.z = 1.7
            else:
                marker.scale.x = 2.0
                marker.scale.y = 1.2
                marker.scale.z = 1.8

            if trk['class_name'] == 'person' and not trk['ppe_ok']:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.8
            else:
                marker.color.r = 0.0
                marker.color.g = 0.8
                marker.color.b = 0.2
                marker.color.a = 0.8

            marker_array.markers.append(marker)

            # Text Marker
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "tracks_3d_text"
            text_marker.id = text_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(pos[0])
            text_marker.pose.position.y = float(pos[1])
            text_marker.pose.position.z = float(pos[2]) + 1.2

            vx, vy = trk['velocity']
            text_marker.text = (
                f"ID:{trk['track_id']} ({trk['class_name']})\n"
                f"V:{math.hypot(vx, vy):.1f}m/s"
            )

            text_marker.scale.z = 0.4
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            marker_array.markers.append(text_marker)

        # 삭제된 마커 Cleanup
        removed_ids = self.prev_active_marker_ids - current_active_marker_ids
        for m_id in removed_ids:
            del_marker = Marker()
            del_marker.header.frame_id = self.target_frame
            del_marker.header.stamp = self.get_clock().now().to_msg()
            del_marker.ns = "tracks_3d_text" if m_id >= 10000 else "tracks_3d"
            del_marker.id = m_id
            del_marker.action = Marker.DELETE
            marker_array.markers.append(del_marker)

        self.prev_active_marker_ids = current_active_marker_ids
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