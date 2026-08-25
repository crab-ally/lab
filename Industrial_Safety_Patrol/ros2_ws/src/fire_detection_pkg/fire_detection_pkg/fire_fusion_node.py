#!/usr/bin/env python3
"""
Node 2: Fire Fusion Node (2D Fire Candidate + Depth Image + 2D LiDAR /scan + TF -> 3D Fire Position & Alarm)

Subscribes:
    - /fire_candidates (std_msgs/msg/String - JSON Format)
    - /camera/image_raw (sensor_msgs/msg/Image)
    - /camera/depth/image_raw (sensor_msgs/msg/Image)
    - /camera/depth/camera_info (sensor_msgs/msg/CameraInfo)
    - /scan (sensor_msgs/msg/LaserScan)

Publishes:
    - /fire_tracks_3d (std_msgs/msg/String - JSON Format)
    - /fire_alarm (std_msgs/msg/Bool)
    - /fire_fusion/debug_markers (visualization_msgs/msg/MarkerArray)
    - /camera/fire_fusion/debug_image (sensor_msgs/msg/Image)
"""

import json
import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import rclpy
import tf2_ros
import message_filters
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time


class FireFusionNode(Node):
    """2D 화재 후보군 + Depth + LiDAR + TF2 기반 3D 화재 위치 추정 및 알람 노드"""

    def __init__(self) -> None:
        super().__init__('fire_fusion_node')

        self.bridge = CvBridge()

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('target_frame', 'base_link')
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        self.declare_parameter('map_frame', 'map')
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value

        self.declare_parameter('lidar_depth_tolerance', 0.5)
        self.lidar_depth_tolerance = self.get_parameter('lidar_depth_tolerance').get_parameter_value().double_value

        self.declare_parameter('min_depth', 0.2)
        self.min_depth = self.get_parameter('min_depth').get_parameter_value().double_value

        self.declare_parameter('max_depth', 8.0)
        self.max_depth = self.get_parameter('max_depth').get_parameter_value().double_value

        self.declare_parameter('sync_slop', 0.1)
        self.sync_slop = self.get_parameter('sync_slop').get_parameter_value().double_value

        self.declare_parameter('candidate_sync_tolerance', 0.1)
        self.candidate_sync_tolerance = self.get_parameter('candidate_sync_tolerance').get_parameter_value().double_value

        self.declare_parameter('lidar_sync_tolerance', 0.15)
        self.lidar_sync_tolerance = self.get_parameter('lidar_sync_tolerance').get_parameter_value().double_value

        # ── TF2 Listener Setup ─────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Camera Intrinsics ─────────────────────────────────────────
        self.fx: float = 600.0
        self.fy: float = 600.0
        self.cx: float = 320.0
        self.cy: float = 240.0
        self.camera_info_received: bool = False
        self.camera_frame_id: str = 'camera_optical_frame'

        # ── Frame Buffers & Storage ───────────────────────────────────
        self.sync_frames: deque = deque(maxlen=20)
        self.scan_cache: deque = deque(maxlen=30)
        self.latest_rgb_img: Optional[np.ndarray] = None
        self.latest_depth_img: Optional[np.ndarray] = None
        self.latest_depth_encoding: str = '16UC1'
        self.latest_scan: Optional[LaserScan] = None

        # RViz Marker Cleanup용 이전 ID 저장소
        self.prev_active_marker_ids: Set[int] = set()

        # ── QoS Profile ───────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # ── Subscriptions & Publishers ─────────────────────────────────
        self.sub_candidates = self.create_subscription(
            String,
            '/fire_candidates',
            self._candidates_callback,
            10
        )

        self.rgb_sub = message_filters.Subscriber(
            self, Image, '/camera/image_raw', qos_profile=sensor_qos
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw', qos_profile=sensor_qos
        )
        self.rgb_depth_sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=self.sync_slop
        )
        self.rgb_depth_sync.registerCallback(self._rgb_depth_callback)

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

        self.pub_fire_tracks = self.create_publisher(
            String,
            '/fire_tracks_3d',
            10
        )

        self.pub_alarm = self.create_publisher(
            Bool,
            '/fire_alarm',
            10
        )

        self.pub_markers = self.create_publisher(
            MarkerArray,
            '/fire_fusion/debug_markers',
            10
        )

        self.pub_debug_image = self.create_publisher(
            Image,
            '/camera/fire_fusion/debug_image',
            10
        )

        self.get_logger().info('Fire Fusion Node (3D Fire Detection + TF + Alarm) is ready.')

    @staticmethod
    def _stamp_to_float(stamp: Time) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _float_to_ros_time(self, value: float) -> Time:
        try:
            val = float(value)
            sec = int(val)
            nanosec = int((val - sec) * 1e9)
            if nanosec >= 1000000000:
                sec += 1
                nanosec -= 1000000000
            return Time(sec=sec, nanosec=nanosec)
        except Exception:
            return Time()

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if msg.k[0] > 0:
            self.fx = msg.k[0]
        if msg.k[4] > 0:
            self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        if msg.header.frame_id:
            self.camera_frame_id = msg.header.frame_id

        if not self.camera_info_received:
            self.camera_info_received = True
            self.get_logger().info(
                f'Camera Intrinsics Loaded: fx={self.fx:.1f}, fy={self.fy:.1f}, '
                f'cx={self.cx:.1f}, cy={self.cy:.1f}, frame_id={self.camera_frame_id}'
            )

    def _rgb_depth_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

            if depth_msg.encoding in ['16UC1', 'mono16']:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                depth_encoding = '16UC1'
            elif depth_msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
                depth_encoding = '32FC1'
            else:
                self.get_logger().warn(f'Unsupported depth encoding: {depth_msg.encoding}')
                return

            rgb_stamp = self._stamp_to_float(rgb_msg.header.stamp)
            depth_stamp = self._stamp_to_float(depth_msg.header.stamp)

            if abs(rgb_stamp - depth_stamp) > self.sync_slop:
                return

            self.sync_frames.append({
                'stamp': rgb_stamp,
                'rgb': rgb_image,
                'depth': depth_image,
                'depth_encoding': depth_encoding,
                'rgb_header': rgb_msg.header,
                'depth_header': depth_msg.header
            })
        except Exception as e:
            self.get_logger().error(f'RGB/Depth synchronization error: {e}')

    def _scan_callback(self, msg: LaserScan) -> None:
        self.scan_cache.append({
            'stamp': self._stamp_to_float(msg.header.stamp),
            'scan': msg
        })

    def _get_synced_frame(self, target_stamp: float) -> Optional[dict]:
        if not self.sync_frames:
            return None

        best = min(self.sync_frames, key=lambda x: abs(x['stamp'] - target_stamp))
        delta = abs(best['stamp'] - target_stamp)

        if delta > self.candidate_sync_tolerance:
            self.get_logger().debug(f'No synchronized RGB/Depth frame: delta={delta:.3f}s')
            return None

        return best

    def _get_synced_scan(self, target_stamp: float) -> Optional[LaserScan]:
        if not self.scan_cache:
            return None

        best = min(self.scan_cache, key=lambda x: abs(x['stamp'] - target_stamp))
        delta = abs(best['stamp'] - target_stamp)

        if delta > self.lidar_sync_tolerance:
            return None

        return best['scan']

    def _get_tf(self, target_frame: str, source_frame: str, stamp: Time) -> Optional[tf2_ros.TransformStamped]:
        try:
            tf_time = rclpy.time.Time.from_msg(stamp)
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                tf_time,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            try:
                return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except Exception as e:
                self.get_logger().warn_throttle(
                    2.0,
                    f'TF unavailable ({source_frame} -> {target_frame}): {e}'
                )
                return None

    def _candidates_callback(self, msg: String) -> None:
        if not self.camera_info_received:
            self.get_logger().warn_throttle(
                2.0,
                'Camera info not received yet. Skipping fire 3D fusion.'
            )
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Fire candidate JSON error: {e}')
            return

        header = payload.get('header', {})
        stamp_value = float(header.get('stamp', 0.0))
        candidates = payload.get('candidates', [])

        if not candidates:
            self._publish_alarm(False)
            self._publish_tracks(stamp_value, [])
            self._publish_markers([])
            return

        synced_frame = self._get_synced_frame(stamp_value)
        if synced_frame is None:
            return

        rgb_image = synced_frame['rgb']
        self.latest_rgb_img = rgb_image
        self.latest_depth_img = synced_frame['depth']
        self.latest_depth_encoding = synced_frame['depth_encoding']

        synced_scan = self._get_synced_scan(stamp_value)
        self.latest_scan = synced_scan

        stamp = self._float_to_ros_time(stamp_value)

        tf_camera_to_base = self._get_tf(self.target_frame, self.camera_frame_id, stamp)
        if tf_camera_to_base is None:
            self.get_logger().warn_throttle(
                2.0,
                f'Camera TF unavailable: {self.camera_frame_id} -> {self.target_frame}'
            )
            return

        fire_tracks: List[dict] = []

        for candidate in candidates:
            try:
                candidate_id = int(candidate['candidate_id'])
                bbox = candidate['bbox']
            except (KeyError, TypeError, ValueError):
                continue

            result = self._calculate_3d_from_depth(bbox)
            if result is None:
                continue

            cam_x, cam_y, cam_z = result
            lidar_distance = self._get_lidar_distance(cam_x, cam_z, synced_scan)

            fusion_distance = cam_z
            lidar_valid = False

            if lidar_distance is not None and abs(lidar_distance - cam_z) <= self.lidar_depth_tolerance:
                fusion_distance = 0.7 * cam_z + 0.3 * lidar_distance
                lidar_valid = True

            if cam_z <= 0:
                continue

            scale = fusion_distance / cam_z
            cam_x *= scale
            cam_y *= scale
            cam_z = fusion_distance

            plane_valid = self._check_plane_geometry(bbox, cam_z)

            if synced_scan is not None and lidar_distance is not None and not lidar_valid:
                continue
            if not plane_valid:
                continue

            p_cam = PointStamped()
            p_cam.header.frame_id = self.camera_frame_id
            p_cam.header.stamp = stamp
            p_cam.point.x = cam_x
            p_cam.point.y = cam_y
            p_cam.point.z = cam_z

            try:
                p_base = do_transform_point(p_cam, tf_camera_to_base)
            except Exception as e:
                self.get_logger().warn_throttle(2.0, f'Camera -> base transform failed: {e}')
                continue

            base_x = p_base.point.x
            base_y = p_base.point.y
            base_z = p_base.point.z

            map_position = None
            tf_base_to_map = self._get_tf(self.map_frame, self.target_frame, stamp)

            if tf_base_to_map is not None:
                try:
                    p_map = do_transform_point(p_base, tf_base_to_map)
                    map_position = [
                        round(float(p_map.point.x), 2),
                        round(float(p_map.point.y), 2),
                        round(float(p_map.point.z), 2)
                    ]
                except Exception as e:
                    self.get_logger().debug(f'Base -> map transform failed: {e}')

            fire_track = {
                'fire_id': candidate_id,
                'bbox': bbox,
                'position': [round(float(base_x), 2), round(float(base_y), 2), round(float(base_z), 2)],
                'position_map': map_position,
                'distance': round(float(math.hypot(base_x, base_y)), 2),
                'depth': round(float(cam_z), 2),
                'lidar_distance': round(float(lidar_distance), 2) if lidar_distance is not None else None,
                'lidar_valid': lidar_valid,
                'plane_valid': plane_valid,
                'temporal_hits': candidate.get('temporal_hits', 0),
                'stamp': stamp_value
            }

            fire_tracks.append(fire_track)

        self._publish_tracks(stamp_value, fire_tracks)
        self._publish_alarm(len(fire_tracks) > 0)
        self._publish_markers(fire_tracks)

        if fire_tracks:
            self._publish_debug_image(
                rgb_image,
                fire_tracks,
                synced_frame['rgb_header']
            )

            for fire in fire_tracks:
                self.get_logger().warn_throttle(
                    1.0,
                    f'FIRE DETECTED id={fire["fire_id"]} '
                    f'base={fire["position"]} '
                    f'map={fire["position_map"]} '
                    f'depth={fire["depth"]:.2f}m '
                    f'lidar={fire["lidar_distance"]}'
                )

    def _calculate_3d_from_depth(self, bbox: List[float]) -> Optional[Tuple[float, float, float]]:
        if self.latest_depth_img is None:
            return None

        h_img, w_img = self.latest_depth_img.shape[:2]
        xmin, ymin, xmax, ymax = map(int, bbox)

        xmin = max(0, min(w_img - 1, xmin))
        xmax = max(0, min(w_img, xmax))
        ymin = max(0, min(h_img - 1, ymin))
        ymax = max(0, min(h_img, ymax))

        if xmax <= xmin or ymax <= ymin:
            return None

        cx_box = (xmin + xmax) / 2.0
        cy_box = (ymin + ymax) / 2.0
        box_w = xmax - xmin
        box_h = ymax - ymin

        rx1 = max(0, int(cx_box - box_w * 0.25))
        rx2 = min(w_img, int(cx_box + box_w * 0.25))
        ry1 = max(0, int(cy_box - box_h * 0.25))
        ry2 = min(h_img, int(cy_box + box_h * 0.25))

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        depth_roi = self.latest_depth_img[ry1:ry2, rx1:rx2]

        if self.latest_depth_encoding == '16UC1':
            valid_depths = depth_roi[depth_roi > 0].astype(np.float32) / 1000.0
        else:
            valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0.1)]

        if len(valid_depths) < 10:
            return None

        z_cam = float(np.median(valid_depths))

        if z_cam < self.min_depth or z_cam > self.max_depth:
            return None

        x_cam = (cx_box - self.cx) * z_cam / self.fx
        y_cam = (cy_box - self.cy) * z_cam / self.fy

        return x_cam, y_cam, z_cam

    def _get_lidar_distance(self, cam_x: float, cam_z: float, scan: Optional[LaserScan]) -> Optional[float]:
        """2D LiDAR /scan 데이터 기반 거리 교차 검증 (Window Search 적용)"""
        if scan is None:
            return None

        angle_rad = -math.atan2(cam_x, cam_z)

        if angle_rad < scan.angle_min or angle_rad > scan.angle_max:
            return None

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
            return float(np.median(valid_ranges))

        return None

    def _check_plane_geometry(self, bbox: List[float], representative_depth: float) -> bool:
        if self.latest_depth_img is None:
            return False

        h_img, w_img = self.latest_depth_img.shape[:2]
        xmin, ymin, xmax, ymax = map(int, bbox)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w_img, xmax)
        ymax = min(h_img, ymax)

        if xmax <= xmin or ymax <= ymin:
            return False

        roi = self.latest_depth_img[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            return False

        step_y = max(1, roi.shape[0] // 15)
        step_x = max(1, roi.shape[1] // 15)

        ys, xs = np.mgrid[ymin:ymax:step_y, xmin:xmax:step_x]
        xs = xs.flatten()
        ys = ys.flatten()

        if len(xs) < 10:
            return True

        depth_values = []

        for px, py in zip(xs, ys):
            depth = self.latest_depth_img[py, px]
            if self.latest_depth_encoding == '16UC1':
                depth_val = float(depth) / 1000.0
            else:
                depth_val = float(depth)

            if not np.isfinite(depth_val) or depth_val < self.min_depth or depth_val > self.max_depth:
                continue

            depth_values.append((float(px), float(py), depth_val))

        if len(depth_values) < 10:
            return True

        points = []

        for px, py, z in depth_values:
            x = (px - self.cx) * z / self.fx
            y = (py - self.cy) * z / self.fy
            points.append([x, y, z])

        points_arr = np.asarray(points, dtype=np.float64)
        center = np.mean(points_arr, axis=0)
        centered = points_arr - center

        try:
            _, _, vh = np.linalg.svd(centered)
        except np.linalg.LinAlgError:
            return True

        normal = vh[-1]
        residuals = np.abs(centered @ normal)

        median_residual = float(np.median(residuals))
        p95_residual = float(np.percentile(residuals, 95))

        plane_like = median_residual < 0.015 and p95_residual < 0.05

        return not plane_like

    def _publish_debug_image(self, image: np.ndarray, fires: List[dict], header) -> None:
        if image is None or not fires:
            return

        display = image.copy()

        for fire in fires:
            x1, y1, x2, y2 = map(int, fire['bbox'])

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)

            label = f'FINAL FIRE ID:{fire["fire_id"]} D:{fire["depth"]:.2f}m'

            cv2.putText(
                display,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
            debug_msg.header = header
            self.pub_debug_image.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')

    def _publish_tracks(self, stamp: float, fires: List[dict]) -> None:
        output = {
            'header': {
                'stamp': stamp,
                'frame_id': self.target_frame,
                'map_frame': self.map_frame
            },
            'fire_detected': len(fires) > 0,
            'fire_count': len(fires),
            'fires': fires
        }

        msg = String()
        msg.data = json.dumps(output, ensure_ascii=False)
        self.pub_fire_tracks.publish(msg)

    def _publish_alarm(self, state: bool) -> None:
        msg = Bool()
        msg.data = bool(state)
        self.pub_alarm.publish(msg)

    def _publish_markers(self, fires: List[dict]) -> None:
        """RViz2 시각화 마커 발행 (DELETE Cleanup 포함)"""
        marker_array = MarkerArray()
        current_active_marker_ids: Set[int] = set()

        for fire in fires:
            fire_id = int(fire['fire_id'])
            sphere_id = fire_id
            text_id = fire_id + 10000

            current_active_marker_ids.add(sphere_id)
            current_active_marker_ids.add(text_id)

            # Sphere Marker (Red for fire)
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fire_tracks"
            marker.id = sphere_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            x, y, z = fire['position']
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.9

            marker_array.markers.append(marker)

            # Text Marker
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "fire_tracks_text"
            text_marker.id = text_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(x)
            text_marker.pose.position.y = float(y)
            text_marker.pose.position.z = float(z) + 0.6

            text_marker.text = f"FIRE ID:{fire_id}\nD:{fire['depth']:.2f}m"
            text_marker.scale.z = 0.35
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            marker_array.markers.append(text_marker)

        # 소화/미감지 마커 삭제 Cleanup (Marker.DELETE)
        removed_ids = self.prev_active_marker_ids - current_active_marker_ids
        for m_id in removed_ids:
            del_marker = Marker()
            del_marker.header.frame_id = self.target_frame
            del_marker.header.stamp = self.get_clock().now().to_msg()
            del_marker.ns = "fire_tracks_text" if m_id >= 10000 else "fire_tracks"
            del_marker.id = m_id
            del_marker.action = Marker.DELETE
            marker_array.markers.append(del_marker)

        self.prev_active_marker_ids = current_active_marker_ids
        self.pub_markers.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FireFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Fire Fusion Node stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()