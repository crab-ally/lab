#!/usr/bin/env python3
import math

import cv2
import numpy as np
import rclpy
import rclpy.duration
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RclpyTime
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
import message_filters

CLASS_NAMES = {0: 'Person', 1: 'Helmet', 2: 'Vest', 3: 'Forklift'}
CLASS_COLORS_BGR = {
    0: (0, 255, 255),    # Person   — 노랑
    1: (255, 0, 0),      # Helmet   — 파랑
    2: (255, 255, 0),    # Vest     — 청록
    3: (0, 165, 255),    # Forklift — 주황
}
CLASS_MARKER_RGBA = {
    0: (1.0, 1.0, 0.0, 0.85),    # Person
    1: (0.0, 0.0, 1.0, 0.85),    # Helmet
    2: (0.0, 1.0, 1.0, 0.85),    # Vest
    3: (1.0, 0.647, 0.0, 0.85),  # Forklift
}

TARGET_CLASSES = {0, 3}


class SensorFusionNode(Node):
    """
    Subscribe:
        /camera/image_raw
        /camera/depth/image_raw
        /yolo/detections
        /scan
    Publish:
        /detected_objects          (Rviz Marker)
        /detected_objects/poses    (odom 3D 좌표 — Person+Forklift 혼합)
        /detected_objects/person_poses   (odom 3D 좌표 — Person만)
        /detected_objects/forklift_poses (odom 3D 좌표 — Forklift만)
        /camera/fusion/image       (debug Image)
    """
    def __init__(self):
        super().__init__('sensor_fusion_node')

        self._img_h = 480
        self._img_w = 640
        fovy_rad = math.radians(58.0)
        self._fy = (self._img_h / 2.0) / math.tan(fovy_rad / 2.0)
        self._fx = self._fy
        self._cx = self._img_w / 2.0
        self._cy = self._img_h / 2.0

        self.get_logger().info(
            f'Camera intrinsics — fx={self._fx:.2f}, fy={self._fy:.2f}, '
            f'cx={self._cx:.1f}, cy={self._cy:.1f}'
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._latest_scan: LaserScan | None = None
        self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._rgb_sub = message_filters.Subscriber(self, Image, '/camera/image_raw', qos_profile=sensor_qos)
        self._depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw', qos_profile=sensor_qos)
        self._det_sub = message_filters.Subscriber(self, Detection2DArray, '/yolo/detections', qos_profile=sensor_qos)
        
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._det_sub], queue_size=10, slop=0.12
        )
        self._sync.registerCallback(self._fusion_callback)

        self._marker_pub = self.create_publisher(MarkerArray, '/detected_objects', 10)
        self._pose_pub = self.create_publisher(PoseArray, '/detected_objects/poses', 10)
        self._person_pose_pub = self.create_publisher(PoseArray, '/detected_objects/person_poses', 10)
        self._forklift_pose_pub = self.create_publisher(PoseArray, '/detected_objects/forklift_poses', 10)
        self._debug_pub = self.create_publisher(Image, '/camera/fusion/image', 10)

        self._bridge = CvBridge()
        self._marker_counter = 0

        self.get_logger().info('Sensor Fusion Node ready.')

    def _scan_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def _fusion_callback(self, rgb_msg: Image, depth_msg: Image, det_msg: Detection2DArray):
        try:
            bgr = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}', once=True)
            return

        debug_img = bgr.copy()
        markers = MarkerArray()
        poses = PoseArray()
        poses.header.stamp = rgb_msg.header.stamp
        poses.header.frame_id = 'odom'

        # 클래스별 PoseArray (TTC 노드 입력용)
        person_poses = PoseArray()
        person_poses.header = poses.header
        forklift_poses = PoseArray()
        forklift_poses.header = poses.header

        clr = Marker()
        clr.action = Marker.DELETEALL
        clr.header.frame_id = 'odom'
        clr.header.stamp = rgb_msg.header.stamp
        markers.markers.append(clr)

        if det_msg.detections and len(det_msg.detections) > 0:
            for det in det_msg.detections:
                if not det.results:
                    continue
                cls_id = int(det.results[0].hypothesis.class_id)
                conf = float(det.results[0].hypothesis.score)

                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                w = det.bbox.size_x
                h = det.bbox.size_y

                x1 = int(cx - w / 2.0)
                y1 = int(cy - h / 2.0)
                x2 = int(cx + w / 2.0)
                y2 = int(cy + h / 2.0)

                u = int(np.clip(cx, 0, self._img_w - 1))
                v = int(np.clip(cy, 0, self._img_h - 1))

                odom_pos = None
                depth_source = ''

                if cls_id in TARGET_CLASSES:
                    r = 7
                    roi = depth[
                        max(0, v - r):min(self._img_h, v + r),
                        max(0, u - r):min(self._img_w, u + r),
                    ]
                    valid = roi[np.isfinite(roi) & (roi > 0.1) & (roi < 6.0)]
                    d = float(np.median(valid)) if len(valid) >= 5 else 0.0

                    if d > 0.1:
                        depth_source = 'depth'
                    else:
                        d = self._lidar_distance_at_pixel(u)
                        if d > 0.1:
                            depth_source = 'lidar'

                    if d > 0.1:
                        odom_pos = self._project_to_odom(u, v, d, rgb_msg.header.stamp)

                color = CLASS_COLORS_BGR.get(cls_id, (180, 180, 180))
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                label = f'{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}'

                if odom_pos is not None:
                    wx, wy, wz = odom_pos
                    label += f' [{depth_source}] ({wx:.2f},{wy:.2f})'
                    markers.markers.append(
                        self._sphere_marker(rgb_msg.header.stamp, cls_id, wx, wy, wz)
                    )
                    p = Pose()
                    p.position = Point(x=wx, y=wy, z=wz)
                    p.orientation = Quaternion(w=1.0)
                    poses.poses.append(p)

                    # 클래스별 PoseArray에도 동일 odom 좌표 분리 수집
                    wp = Pose()
                    wp.position = Point(x=wx, y=wy, z=wz)
                    wp.orientation = Quaternion(w=1.0)
                    if cls_id == 0:    # Person
                        person_poses.poses.append(wp)
                    elif cls_id == 3:  # Forklift
                        forklift_poses.poses.append(wp)

                cv2.putText(
                    debug_img, label,
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )
                cv2.circle(debug_img, (u, v), 4, (255, 255, 255), -1)

        self._marker_pub.publish(markers)
        self._pose_pub.publish(poses)
        self._person_pose_pub.publish(person_poses)
        self._forklift_pose_pub.publish(forklift_poses)

        try:
            dbg_msg = self._bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            dbg_msg.header = rgb_msg.header
            self._debug_pub.publish(dbg_msg)
        except Exception as e:
            self.get_logger().error(f'Debug publish error: {e}', once=True)

    def _project_to_odom(self, u: int, v: int, d: float, stamp) -> tuple | None:
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
        odom_pos = R @ np.array([x_cam, y_cam, z_cam]) + np.array([tx, ty, tz])
        return float(odom_pos[0]), float(odom_pos[1]), float(odom_pos[2])

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        return np.array([
            [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
            [   2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [   2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)

    def _lidar_distance_at_pixel(self, u: int) -> float:
        if self._latest_scan is None:
            return 0.0

        scan = self._latest_scan
        angle_cam = math.atan2(u - self._cx, self._fx)
        lidar_angle = -angle_cam

        while lidar_angle < scan.angle_min:
            lidar_angle += 2 * math.pi
        while lidar_angle > scan.angle_max:
            lidar_angle -= 2 * math.pi

        idx = int(round((lidar_angle - scan.angle_min) / scan.angle_increment))
        idx = max(0, min(idx, len(scan.ranges) - 1))

        beam_count = len(scan.ranges)
        indices = [
            (idx + offset) % beam_count
            for offset in range(-5, 6)
        ]

        near = [
            scan.ranges[i]
            for i in indices
            if math.isfinite(scan.ranges[i])
            and scan.range_min < scan.ranges[i] < scan.range_max
        ]

        return min(near) if near else 0.0

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
        m.pose.position.z = wz + 0.5
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