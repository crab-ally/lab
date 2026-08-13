#!/usr/bin/env python3
"""
## 13. Camera + LiDAR Sensor Fusion Node

카메라 Depth Projection + LiDAR 거리 검증으로
감지된 객체의 3D 월드 좌표(odom 프레임)를 추정합니다.

### 융합 전략
  1차: Depth Projection  — BBox 중심 픽셀의 depth 값으로 Camera 3D 좌표 계산
  2차: LiDAR Fallback    — Depth가 무효(inf, 0, NaN)일 때 LiDAR 빔으로 대체

### 구독 토픽
  /camera/image_raw           — RGB 이미지 (YOLO 입력)
  /camera/depth/image_raw     — Depth 이미지 (32FC1, meters)
  /scan                       — LaserScan (LiDAR 거리 검증)

### 발행 토픽
  /detected_objects            — visualization_msgs/MarkerArray (RViz 3D 마커)
  /detected_objects/poses      — geometry_msgs/PoseArray (3D 위치 배열)
  /camera/fusion/image         — sensor_msgs/Image (디버그 오버레이)

### 카메라 파라미터 (turtlebot_patrol.xml 기준)
  렌더 해상도: 480(H) x 640(W)
  fovy = 58°
  fx = fy = (H/2) / tan(fovy_rad/2)
  cx = W/2, cy = H/2
"""

import math

import cv2
import numpy as np
import rclpy
import rclpy.duration
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion
from nav_msgs.msg import Odometry  # noqa: F401 (가능한 확장을 위해 유지)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RclpyTime
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import ColorRGBA
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray
import message_filters

# ──────────────────────────────────────────────
# 감지 클래스 설정 (ppe_forklift_yolov8n 기준)
# ──────────────────────────────────────────────
CLASS_NAMES = {0: 'Person', 1: 'Helmet', 2: 'Vest', 3: 'Forklift'}
CLASS_COLORS_BGR = {
    0: (0, 200, 255),    # Person   — 주황
    1: (0, 255, 80),     # Helmet   — 초록
    2: (255, 200, 0),    # Vest     — 파랑
    3: (0, 80, 255),     # Forklift — 빨강
}
CLASS_MARKER_RGBA = {
    0: (1.0, 0.78, 0.0, 0.85),   # Person
    1: (0.0, 1.0, 0.31, 0.85),   # Helmet
    2: (1.0, 0.78, 0.0, 0.85),   # Vest
    3: (1.0, 0.31, 0.0, 0.85),   # Forklift
}

# 3D 위치 추정 대상 클래스 (Person, Forklift)
TARGET_CLASSES = {0, 3}


class SensorFusionNode(Node):
    """Camera + LiDAR Sensor Fusion ROS2 Node."""

    def __init__(self):
        super().__init__('sensor_fusion_node')

        # ──────────────────────────────────────────────
        # 카메라 Intrinsics
        # turtlebot_patrol.xml: fovy=58°, render 480x640
        # ──────────────────────────────────────────────
        self._img_h = 480
        self._img_w = 640
        fovy_rad = math.radians(58.0)
        self._fy = (self._img_h / 2.0) / math.tan(fovy_rad / 2.0)
        self._fx = self._fy          # 정사각형 픽셀
        self._cx = self._img_w / 2.0
        self._cy = self._img_h / 2.0

        self.get_logger().info(
            f'Camera intrinsics — fx={self._fx:.2f}, fy={self._fy:.2f}, '
            f'cx={self._cx:.1f}, cy={self._cy:.1f}'
        )

        # ──────────────────────────────────────────────
        # YOLO 모델 (독립 인스턴스 — ppe_detection_node와 분리)
        # ──────────────────────────────────────────────
        model_path = '/workspace/models/ppe_forklift_yolov8n/best.pt'
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self._yolo = YOLO(model_path)
        self._yolo_conf = 0.45
        self.get_logger().info('YOLO model loaded.')

        # ──────────────────────────────────────────────
        # TF2
        # ──────────────────────────────────────────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ──────────────────────────────────────────────
        # 최신 LaserScan 캐시 (LiDAR Fallback용)
        # ──────────────────────────────────────────────
        self._latest_scan: LaserScan | None = None
        self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)

        # ──────────────────────────────────────────────
        # RGB + Depth 시간 동기화
        # ──────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._rgb_sub = message_filters.Subscriber(
            self, Image, '/camera/image_raw', qos_profile=sensor_qos
        )
        self._depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw', qos_profile=sensor_qos
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub], queue_size=10, slop=0.12
        )
        self._sync.registerCallback(self._image_callback)

        # ──────────────────────────────────────────────
        # Publishers
        # ──────────────────────────────────────────────
        self._marker_pub = self.create_publisher(MarkerArray, '/detected_objects', 10)
        self._pose_pub = self.create_publisher(PoseArray, '/detected_objects/poses', 10)
        self._debug_pub = self.create_publisher(Image, '/camera/fusion/image', 10)

        self._bridge = CvBridge()
        self._marker_counter = 0

        self.get_logger().info('Sensor Fusion Node ready.')

    # ──────────────────────────────────────────────────────────
    # 콜백: LiDAR 캐시
    # ──────────────────────────────────────────────────────────
    def _scan_callback(self, msg: LaserScan):
        self._latest_scan = msg

    # ──────────────────────────────────────────────────────────
    # 콜백: RGB + Depth 메인 처리
    # ──────────────────────────────────────────────────────────
    def _image_callback(self, rgb_msg: Image, depth_msg: Image):
        # 1. 이미지 변환
        try:
            bgr = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0   # mm → m
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}', once=True)
            return

        # 2. YOLO 탐지
        results = self._yolo(bgr, conf=self._yolo_conf, verbose=False)
        boxes = results[0].boxes if results else None

        debug_img = bgr.copy()
        markers = MarkerArray()
        poses = PoseArray()
        poses.header.stamp = rgb_msg.header.stamp
        poses.header.frame_id = 'odom'

        # 이전 프레임 마커 클리어
        clr = Marker()
        clr.action = Marker.DELETEALL
        clr.header.frame_id = 'odom'
        clr.header.stamp = rgb_msg.header.stamp
        markers.markers.append(clr)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # BBox 중심 픽셀
                u = int(np.clip((x1 + x2) // 2, 0, self._img_w - 1))
                v = int(np.clip((y1 + y2) // 2, 0, self._img_h - 1))

                world_pos = None
                depth_source = ''

                if cls_id in TARGET_CLASSES:
                    # ── 1차: Depth Projection
                    r = 7
                    roi = depth[
                        max(0, v - r):min(self._img_h, v + r),
                        max(0, u - r):min(self._img_w, u + r),
                    ]
                    valid = roi[np.isfinite(roi) & (roi > 0.1) & (roi < 10.0)]
                    d = float(np.median(valid)) if len(valid) >= 5 else 0.0

                    if d > 0.1:
                        depth_source = 'depth'
                    else:
                        # ── 2차: LiDAR Fallback
                        d = self._lidar_distance_at_pixel(u)
                        if d > 0.1:
                            depth_source = 'lidar'

                    if d > 0.1:
                        world_pos = self._project_to_world(u, v, d, rgb_msg.header.stamp)

                # ── 디버그 오버레이
                color = CLASS_COLORS_BGR.get(cls_id, (180, 180, 180))
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                label = f'{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}'

                if world_pos is not None:
                    wx, wy, wz = world_pos
                    label += f' [{depth_source}] ({wx:.2f},{wy:.2f})'
                    markers.markers.append(
                        self._sphere_marker(rgb_msg.header.stamp, cls_id, wx, wy, wz)
                    )
                    p = Pose()
                    p.position = Point(x=wx, y=wy, z=wz)
                    p.orientation = Quaternion(w=1.0)
                    poses.poses.append(p)

                cv2.putText(
                    debug_img, label,
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )
                cv2.circle(debug_img, (u, v), 4, (255, 255, 255), -1)

        # 3. 발행
        self._marker_pub.publish(markers)
        self._pose_pub.publish(poses)
        try:
            dbg_msg = self._bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            dbg_msg.header = rgb_msg.header
            self._debug_pub.publish(dbg_msg)
        except Exception as e:
            self.get_logger().error(f'Debug publish error: {e}', once=True)

    # ──────────────────────────────────────────────────────────
    # Depth Projection: 픽셀 → Camera 3D → odom
    # ──────────────────────────────────────────────────────────
    def _project_to_world(self, u: int, v: int, d: float, stamp) -> tuple | None:
        """
        픽셀 (u, v) + depth d [m] → odom 프레임 3D 좌표 (x, y, z)

        카메라 광학 좌표:
            X_cam = (u - cx) * d / fx
            Y_cam = (v - cy) * d / fy
            Z_cam = d   (전방)
        """
        x_cam = (u - self._cx) * d / self._fx
        y_cam = (v - self._cy) * d / self._fy
        z_cam = d

        try:
            tf_s = self._tf_buffer.lookup_transform(
                'odom', 'camera_link',
                RclpyTime(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF camera_link→odom failed: {e}',
                throttle_duration_sec=2.0,
            )
            return None

        t = tf_s.transform
        tx, ty, tz = t.translation.x, t.translation.y, t.translation.z
        qx, qy, qz, qw = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w

        R = self._quat_to_rot(qx, qy, qz, qw)
        world = R @ np.array([x_cam, y_cam, z_cam]) + np.array([tx, ty, tz])
        return float(world[0]), float(world[1]), float(world[2])

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """단위 쿼터니언 → 3×3 회전 행렬"""
        return np.array([
            [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)

    # ──────────────────────────────────────────────────────────
    # LiDAR Fallback
    # ──────────────────────────────────────────────────────────
    def _lidar_distance_at_pixel(self, u: int) -> float:
        """
        BBox 수평 중심 픽셀 u → 카메라 수평 각도 → LiDAR 빔 거리 반환

        카메라 수평 각도: θ = atan((u - cx) / fx)   (오른쪽 = +)
        LiDAR 매핑:       전방=0°, 반시계 양수
        """
        if self._latest_scan is None:
            return 0.0

        scan = self._latest_scan
        angle_cam = math.atan2(u - self._cx, self._fx)
        lidar_angle = -angle_cam   # 카메라 오른쪽 → LiDAR 시계 방향(음수)

        # 각도 범위 정규화
        while lidar_angle < scan.angle_min:
            lidar_angle += 2 * math.pi
        while lidar_angle > scan.angle_max:
            lidar_angle -= 2 * math.pi

        idx = int(round((lidar_angle - scan.angle_min) / scan.angle_increment))
        idx = max(0, min(idx, len(scan.ranges) - 1))

        # ±5 빔 중 유효 최솟값
        lo = max(0, idx - 5)
        hi = min(len(scan.ranges), idx + 6)
        near = [
            r for r in scan.ranges[lo:hi]
            if math.isfinite(r) and scan.range_min < r < scan.range_max
        ]
        return min(near) if near else 0.0

    # ──────────────────────────────────────────────────────────
    # RViz Sphere Marker
    # ──────────────────────────────────────────────────────────
    def _sphere_marker(
        self, stamp, cls_id: int,
        wx: float, wy: float, wz: float
    ) -> Marker:
        self._marker_counter += 1
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = stamp
        m.ns = 'sensor_fusion'
        m.id = self._marker_counter
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = wx
        m.pose.position.y = wy
        m.pose.position.z = wz + 0.5   # 객체 위에 표시
        m.pose.orientation.w = 1.0
        radius = 0.30 if cls_id == 3 else 0.20
        m.scale.x = radius
        m.scale.y = radius
        m.scale.z = radius
        r, g, b, a = CLASS_MARKER_RGBA.get(cls_id, (0.8, 0.8, 0.8, 0.8))
        m.color = ColorRGBA(r=r, g=g, b=b, a=a)
        m.lifetime.sec = 2
        return m


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Sensor Fusion Node stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
