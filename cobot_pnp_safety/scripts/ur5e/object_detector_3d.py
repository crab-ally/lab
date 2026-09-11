#!/usr/bin/env python3
"""
sub:
  - /camera/depth/image_raw
  - /camera/depth/camera_info
  - /camera/segmentation/image_raw
pub:
  - /target_object_pose
  - /object_pointcloud
  - /detected_objects_markers
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

class Object3DDetector(Node):
    def __init__(self):
        super().__init__("object_3d_detector")

        self.camera_frame = "gripper_camera_optical_frame"
        self.target_frame = "base"
        self.depth = self.seg = None
        self.depth_stamp = None
        self.fx = self.fy = self.cx = self.cy = None
        self.stride = 2
        self.last_pose_log_time = 0.0

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        self.pose_pub = self.create_publisher(PoseStamped, "/target_object_pose", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/detected_objects_markers", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/object_pointcloud", 10)

        self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)
        self.create_subscription(CameraInfo, "/camera/depth/camera_info", self.camera_info_callback, 10)
        self.create_subscription(Image, "/camera/segmentation/image_raw", self.seg_callback, 10)
        self.create_timer(0.1, self.process)

        self.get_logger().info("3D Object Detector started.")

    def camera_info_callback(self, msg):
        if len(msg.k) >= 9:
            self.fx, self.fy = float(msg.k[0]), float(msg.k[4])
            self.cx, self.cy = float(msg.k[2]), float(msg.k[5])

    def depth_callback(self, msg):
        if msg.encoding == "32FC1":
            self.depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
        elif msg.encoding == "16UC1":
            self.depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        else:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
            return
        self.depth_stamp = msg.header.stamp

    def seg_callback(self, msg):
        if msg.encoding == "32SC1":
            self.seg = np.frombuffer(msg.data, np.int32).reshape(msg.height, msg.width).copy()
        else:
            self.get_logger().warn(f"Unsupported segmentation encoding: {msg.encoding}")

    def depth_to_points(self):
        d = self.depth[::self.stride, ::self.stride]
        s = self.seg[::self.stride, ::self.stride]
        mask = cv2.erode((s != 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

        h, w = d.shape
        u, v = np.meshgrid(
            np.arange(0, w * self.stride, self.stride),
            np.arange(0, h * self.stride, self.stride)
        )

        d, mask, u, v = d.ravel(), mask.ravel(), u.ravel(), v.ravel()

        valid = np.isfinite(d) & (d > 0.01) & (d < 50.0) & (mask > 0)
        if not np.any(valid):
            return np.empty((0, 3), np.float32)

        z = d[valid]
        x = (u[valid] - self.cx) * z / self.fx
        y = (v[valid] - self.cy) * z / self.fy

        return np.column_stack((x, y, z)).astype(np.float32)

    def transform_pose(self, pose):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
            return tf2_geometry_msgs.do_transform_pose_stamped(pose, tf)
        except Exception as e:
            self.get_logger().warn(f"TF Transform failed: {e}")
            return None

    def publish_pointcloud(self, points):
        msg = PointCloud2()
        msg.header.stamp = self.depth_stamp
        msg.header.frame_id = self.camera_frame
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
        self.cloud_pub.publish(msg)

    def publish_marker(self, pose):
        stamp = self.depth_stamp
        markers = MarkerArray()

        delete = Marker()
        delete.header.frame_id = self.target_frame
        delete.header.stamp = stamp
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)

        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns = "objects"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = marker.scale.y = marker.scale.z = 0.03
        marker.color.r = marker.color.a = 1.0
        markers.markers.append(marker)

        self.marker_pub.publish(markers)

    def process(self):
        if self.depth is None or self.seg is None:
            return
        if None in (self.fx, self.fy, self.cx, self.cy):
            return
        if self.depth.shape != self.seg.shape:
            self.get_logger().warn("Depth and segmentation resolution mismatch.")
            return

        points = self.depth_to_points()
        if len(points) < 20:
            return

        self.publish_pointcloud(points)

        center = points.mean(axis=0)

        pose = PoseStamped()
        pose.header.frame_id = self.camera_frame
        pose.header.stamp = self.depth_stamp
        pose.pose.position.x = float(center[0])
        pose.pose.position.y = float(center[1])
        pose.pose.position.z = float(center[2])
        pose.pose.orientation.w = 1.0

        target = self.transform_pose(pose)
        if target is None:
            return

        self.pose_pub.publish(target)
        self.publish_marker(target)

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_pose_log_time >= 1.0:
            self.get_logger().info(
                f"Target(base): "
                f"({target.pose.position.x:.3f}, {target.pose.position.y:.3f}, {target.pose.position.z:.3f})"
            )
            self.last_pose_log_time = now

def main():
    rclpy.init()
    node = Object3DDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()