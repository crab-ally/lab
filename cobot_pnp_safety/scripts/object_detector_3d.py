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
  - /detected_objects_markers
  - /object_pointcloud
"""

import math
import numpy as np
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from tf2_ros import Buffer, TransformListener


class Ransac3DObjectDetector(Node):
    def __init__(self):
        super().__init__("ransac_3d_object_detector")

        self.camera_frame = "ceiling_camera_optical_frame"
        self.target_frame = "link0"
        self.depth = None
        self.seg = None
        self.fx = self.fy = 432.97
        self.cx, self.cy = 319.5, 239.5
        self.stride = 2

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_pub = self.create_publisher(PoseStamped, "/target_object_pose", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/detected_objects_markers", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/object_pointcloud", 10)

        self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)
        self.create_subscription(Image, "/camera/segmentation/image_raw", self.seg_callback, 10)
        self.create_subscription(CameraInfo, "/camera/depth/camera_info", self.info_callback, 10)

        self.create_timer(0.1, self.process)

        self.get_logger().info("3D RANSAC + DBSCAN Object Detector started.")

    def depth_callback(self, msg):
        if msg.encoding == "32FC1":
            self.depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
        elif msg.encoding == "16UC1":
            self.depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0

    def seg_callback(self, msg):
        if msg.encoding == "32SC1":
            self.seg = np.frombuffer(msg.data, np.int32).reshape(msg.height, msg.width).copy()

    def info_callback(self, msg):
        if msg.k[0] > 0:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]

    def depth_to_points(self):
        d = self.depth[::self.stride, ::self.stride]
        s = self.seg[::self.stride, ::self.stride]

        h, w = d.shape
        u, v = np.meshgrid(
            np.arange(0, w * self.stride, self.stride),
            np.arange(0, h * self.stride, self.stride)
        )

        d = d.reshape(-1)
        s = s.reshape(-1)
        u = u.reshape(-1)
        v = v.reshape(-1)

        valid = np.isfinite(d) & (d > 0.01) & (d < 50.0) & (s != 0)

        z = d[valid]
        x = (u[valid] - self.cx) * z / self.fx
        y = (v[valid] - self.cy) * z / self.fy

        return np.column_stack((x, y, z))

    def ransac_plane(self, points, threshold=0.015, iterations=100):
        if len(points) < 50:
            return None

        best_mask = None
        best_count = 0

        for _ in range(iterations):
            p1, p2, p3 = points[np.random.choice(len(points), 3, replace=False)]
            n = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(n)

            if norm < 1e-6:
                continue

            n /= norm
            d = -np.dot(n, p1)
            mask = np.abs(points @ n + d) < threshold
            count = np.sum(mask)

            if count > best_count:
                best_count = count
                best_mask = mask

        return points[~best_mask] if best_mask is not None else None

    def transform_point(self, point):
        p = PointStamped()
        p.header.frame_id = self.camera_frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.point.x, p.point.y, p.point.z = map(float, point)

        try:
            return self.tf_buffer.transform(
                p, self.target_frame,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except Exception:
            return None

    def publish_pointcloud(self, points):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
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

    def process(self):
        if self.depth is None or self.seg is None:
            return

        if self.depth.shape != self.seg.shape:
            return

        points = self.depth_to_points()

        if len(points) < 100:
            return

        object_points = self.ransac_plane(points)

        if object_points is None or len(object_points) < 20:
            return

        self.publish_pointcloud(object_points)

        labels = DBSCAN(eps=0.06, min_samples=8).fit_predict(object_points)
        clusters = [object_points[labels == i] for i in set(labels) if i >= 0 and np.sum(labels == i) >= 20]

        if not clusters:
            return

        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        delete = Marker()
        delete.header.frame_id = self.target_frame
        delete.header.stamp = stamp
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)

        for i, cluster in enumerate(clusters):
            center = (cluster.min(axis=0) + cluster.max(axis=0)) / 2.0
            size = cluster.max(axis=0) - cluster.min(axis=0)

            p = self.transform_point(center)

            if p is None:
                continue

            if i == 0:
                pose = PoseStamped()
                pose.header.frame_id = self.target_frame
                pose.header.stamp = stamp
                pose.pose.position = p.point
                pose.pose.orientation.w = 1.0
                self.pose_pub.publish(pose)

            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = stamp
            marker.ns = "objects"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = p.point
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(float(size[0]), 0.04)
            marker.scale.y = max(float(size[1]), 0.04)
            marker.scale.z = max(float(size[2]), 0.04)
            marker.color.a = 0.6
            markers.markers.append(marker)

        self.marker_pub.publish(markers)


def main():
    rclpy.init()
    node = Ransac3DObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()