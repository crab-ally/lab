#!/usr/bin/env python3
"""
sub:
  - /camera/depth/image_raw
  - /camera/depth/camera_info
  - /camera/segmentation/image_raw
pub:
  - /target_object_pose
  - /detected_objects
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
from sklearn.cluster import DBSCAN
from cobot_pnp_msgs.msg import ObjectInfo, ObjectInfoArray


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
        self.objects_pub = self.create_publisher(ObjectInfoArray, "/detected_objects", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/detected_objects_markers", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/object_pointcloud", 10)

        self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)
        self.create_subscription(CameraInfo, "/camera/depth/camera_info", self.camera_info_callback, 10)
        self.create_subscription(Image, "/camera/segmentation/image_raw", self.seg_callback, 10)
        self.create_timer(0.1, self.process)

        self.get_logger().info("3D Object Detector started.")

    def camera_info_callback(self, msg):
        if len(msg.k) >= 9:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])

    def depth_callback(self, msg):
        if msg.encoding == "32FC1":
            self.depth = np.frombuffer(
                msg.data, np.float32
            ).reshape(msg.height, msg.width).copy()
        elif msg.encoding == "16UC1":
            self.depth = np.frombuffer(
                msg.data, np.uint16
            ).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        else:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
            return

        self.depth_stamp = msg.header.stamp

    def seg_callback(self, msg):
        if msg.encoding == "32SC1":
            self.seg = np.frombuffer(
                msg.data, np.int32
            ).reshape(msg.height, msg.width).copy()
        else:
            self.get_logger().warn(f"Unsupported segmentation encoding: {msg.encoding}")

    def depth_to_points(self):
        d = self.depth[::self.stride, ::self.stride]
        s = self.seg[::self.stride, ::self.stride]

        mask = cv2.erode(
            (s != 0).astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1
        )

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

    def cluster_points(self, points):
        labels = DBSCAN(eps=0.025, min_samples=10).fit_predict(points)
        return [points[labels == i] for i in set(labels) if i >= 0]

    def get_object_info(self, cluster):
        center = cluster.mean(axis=0)
        cmin = cluster.min(axis=0)
        cmax = cluster.max(axis=0)
        size = cmax - cmin

        xy = cluster[:, :2] - center[:2]

        if len(cluster) > 2:
            _, _, vh = np.linalg.svd(
                xy,
                full_matrices=False
            )
            yaw = float(
                np.arctan2(vh[0, 1], vh[0, 0])
            )
        else:
            yaw = 0.0

        return center, size, yaw

    def yaw_to_quaternion(self, yaw):
        return (
            0.0,
            0.0,
            float(np.sin(yaw / 2.0)),
            float(np.cos(yaw / 2.0))
        )

    def publish_pointcloud(self, points):
        msg = PointCloud2()
        msg.header.stamp = self.depth_stamp
        msg.header.frame_id = self.camera_frame
        msg.height = 1
        msg.width = len(points)

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(points)
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()

        self.cloud_pub.publish(msg)

    def publish_objects(self, objects):
        msg = ObjectInfoArray()

        for obj in objects:
            info = ObjectInfo()
            info.pose = obj["pose"].pose
            info.size_x = float(obj["size"][0])
            info.size_y = float(obj["size"][1])
            info.size_z = float(obj["size"][2])
            info.yaw = float(obj["yaw"])
            msg.objects.append(info)

        self.objects_pub.publish(msg)

    def publish_markers(self, objects, best_idx):
        markers = MarkerArray()

        delete = Marker()
        delete.header.frame_id = self.target_frame
        delete.header.stamp = self.depth_stamp
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)

        for i, obj in enumerate(objects):
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = self.depth_stamp
            marker.ns = "objects"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = obj["pose"].pose
            marker.scale.x = max(float(obj["size"][0]), 0.005)
            marker.scale.y = max(float(obj["size"][1]), 0.005)
            marker.scale.z = max(float(obj["size"][2]), 0.005)
            marker.color.r = 1.0 if i == best_idx else 0.0
            marker.color.g = 0.0 if i == best_idx else 1.0
            marker.color.b = 0.0
            marker.color.a = 0.35
            markers.markers.append(marker)

            text = Marker()
            text.header.frame_id = self.target_frame
            text.header.stamp = self.depth_stamp
            text.ns = "object_labels"
            text.id = 1000 + i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = obj["pose"].pose
            text.pose.position.z += marker.scale.z * 0.5 + 0.02
            text.scale.z = 0.04
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"Object {i}"
            markers.markers.append(text)

        self.marker_pub.publish(markers)

    def process(self):
        if self.depth is None or self.seg is None or None in (self.fx, self.fy, self.cx, self.cy):
            return

        if self.depth.shape != self.seg.shape:
            return

        points = self.depth_to_points()

        if len(points) < 20:
            return

        clusters = self.cluster_points(points)

        if not clusters:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        objects = []
        best_idx = -1
        best_dist = float("inf")

        for i, cluster in enumerate(clusters):
            center, size, yaw = self.get_object_info(cluster)

            pose = PoseStamped()
            pose.header.frame_id = self.camera_frame
            pose.header.stamp = self.depth_stamp
            pose.pose.position.x = float(center[0])
            pose.pose.position.y = float(center[1])
            pose.pose.position.z = float(center[2])
            pose.pose.orientation.w = 1.0

            target = tf2_geometry_msgs.do_transform_pose_stamped(pose, tf)

            qx, qy, qz, qw = self.yaw_to_quaternion(yaw)

            target.pose.orientation.x = qx
            target.pose.orientation.y = qy
            target.pose.orientation.z = qz
            target.pose.orientation.w = qw

            p = target.pose.position
            dist = np.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)

            objects.append({
                "pose": target,
                "size": size,
                "yaw": yaw,
                "points": cluster
            })

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx < 0:
            return

        self.publish_pointcloud(points)
        self.publish_objects(objects)
        self.pose_pub.publish(objects[best_idx]["pose"])
        self.publish_markers(objects, best_idx)

        now = self.get_clock().now().nanoseconds * 1e-9

        if now - self.last_pose_log_time >= 1.0:
            for i, obj in enumerate(objects):
                p = obj["pose"].pose.position
                s = obj["size"]

                self.get_logger().info(
                    f"Object {i}: "
                    f"pos=({p.x:.3f},{p.y:.3f},{p.z:.3f}) "
                    f"size=({s[0]:.3f},{s[1]:.3f},{s[2]:.3f}) "
                    f"yaw={np.degrees(obj['yaw']):.1f} deg"
                )
            self.get_logger().info(f"Target object index={best_idx}")
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