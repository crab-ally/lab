#!/usr/bin/env python3
"""
3D RANSAC + DBSCAN Object Detector for Franka Emika Panda

Input:
  - /camera/depth/image_raw      sensor_msgs/Image (32FC1, meters)
  - /camera/depth/camera_info    sensor_msgs/CameraInfo

TF:
  - ceiling_camera_optical_frame -> link0

Output:
  - /object_pointcloud
  - /target_object_pose
      geometry_msgs/PoseStamped
      pose : 물체 중심 PoseStamped (link0 기준)
  - /detected_objects_markers
"""

import numpy as np
from sklearn.cluster import DBSCAN
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs


def create_pointcloud2_msg(stamp, frame_id, points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(len(points))
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * int(len(points))
    msg.is_dense = True
    msg.data = points.tobytes()
    return msg


def ransac_plane_segmentation(points, distance_threshold=0.015, max_iterations=120):
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
        d = -float(np.dot(normal, p1))
        distances = np.abs(np.dot(points, normal) + d)
        inliers = np.where(distances < distance_threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal.copy(), d)
    if best_plane is None or len(best_inliers) < 50:
        return None, None, None
    normal, d = best_plane
    if normal[2] > 0:
        normal = -normal
        d = -d
    signed_distances = np.dot(points, normal) + d
    object_indices = np.where((signed_distances > distance_threshold) & (signed_distances < 0.40))[0]
    object_points = points[object_indices]
    return (normal, d), best_inliers, object_points


class Ransac3DObjectDetector(Node):
    def __init__(self):
        super().__init__("ransac_3d_object_detector")

        self.depth_topic = "/camera/depth/image_raw"
        self.camera_info_topic = "/camera/depth/camera_info"
        self.camera_optical_frame = "ceiling_camera_optical_frame"
        self.target_frame = "link0"

        self.width = 640
        self.height = 480
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_received = False
        self.depth_received = False
        self.latest_depth = None
        self.latest_depth_stamp = None
        self.stride = 2

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_pub = self.create_publisher(PoseStamped, "/target_object_pose", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/detected_objects_markers", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/object_pointcloud", 10)

        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.timer = self.create_timer(0.1, self.process_detection)

        self.get_logger().info("[INIT] 3D RANSAC + DBSCAN Object Detector started.")
        self.get_logger().info(f"[INIT] Depth topic: {self.depth_topic}")

    def camera_info_callback(self, msg):
        if self.camera_info_received:
            return

        self.width = int(msg.width)
        self.height = int(msg.height)
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

        if self.fx <= 0.0 or self.fy <= 0.0:
            self.get_logger().error("[CAMERA INFO] Invalid camera intrinsics.")
            return

        self.camera_info_received = True
        self.get_logger().info(f"[CAMERA INFO] width={self.width}, height={self.height}, fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

    def depth_callback(self, msg):
        try:
            if msg.encoding != "32FC1":
                self.get_logger().warn(f"[DEPTH] Expected 32FC1 but received {msg.encoding}", throttle_duration_sec=3.0)
                return

            if msg.height <= 0 or msg.width <= 0:
                return

            if msg.step < msg.width * 4:
                self.get_logger().warn("[DEPTH] Invalid step size.", throttle_duration_sec=3.0)
                return

            if msg.step == msg.width * 4:
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            else:
                depth = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
                depth = depth[:, :msg.width * 4].view(np.float32).reshape(msg.height, msg.width)

            self.latest_depth = depth.copy()
            self.latest_depth_stamp = msg.header.stamp
            self.depth_received = True

        except Exception as e:
            self.get_logger().error(f"[DEPTH] Failed to parse depth image: {e}", throttle_duration_sec=3.0)

    def generate_point_cloud(self, depth):
        if self.fx is None or self.fy is None:
            return np.empty((0, 3), dtype=np.float32)

        sampled = depth[::self.stride, ::self.stride]
        h, w = sampled.shape
        v_coords, u_coords = np.indices((h, w), dtype=np.float64)
        u = u_coords * self.stride
        v = v_coords * self.stride
        z = sampled.astype(np.float64)

        valid_mask = np.isfinite(z) & (z > 0.20) & (z < 2.30)

        if not np.any(valid_mask):
            return np.empty((0, 3), dtype=np.float32)

        z = z[valid_mask]
        u = u[valid_mask]
        v = v[valid_mask]

        x = (u - self.cx) / self.fx * z
        y = (v - self.cy) / self.fy * z

        points = np.column_stack((x, y, z))
        points = points[np.all(np.isfinite(points), axis=1)]
        return points.astype(np.float32)

    def process_detection(self):
        try:
            if not self.camera_info_received or not self.depth_received or self.latest_depth is None:
                return

            depth = self.latest_depth
            points = self.generate_point_cloud(depth)

            if len(points) < 100:
                self.clear_markers()
                return

            plane_eq, inliers, object_points = ransac_plane_segmentation(points, distance_threshold=0.015, max_iterations=120)

            if plane_eq is None:
                self.clear_markers()
                return

            if object_points is None or len(object_points) < 10:
                self.clear_markers()
                return

            plane_normal, plane_d = plane_eq
            stamp = self.latest_depth_stamp

            if stamp is None:
                stamp = self.get_clock().now().to_msg()

            cloud_msg = create_pointcloud2_msg(stamp, self.camera_optical_frame, object_points)
            self.cloud_pub.publish(cloud_msg)

            clustering = DBSCAN(eps=0.06, min_samples=8).fit(object_points)
            labels = clustering.labels_
            unique_labels = set(labels)

            if -1 in unique_labels:
                unique_labels.remove(-1)

            if not unique_labels:
                self.clear_markers()
                return

            marker_array = MarkerArray()

            delete_marker = Marker()
            delete_marker.header.frame_id = self.target_frame
            delete_marker.header.stamp = stamp
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)

            valid_cluster_idx = 0

            for label in sorted(unique_labels):
                cluster_pts = object_points[labels == label]

                if len(cluster_pts) < 8:
                    continue

                distances_from_plane = np.dot(cluster_pts, plane_normal) + plane_d
                estimated_height = float(np.max(distances_from_plane))

                if estimated_height < 0.01 or estimated_height > 0.40:
                    continue

                min_bound = np.min(cluster_pts, axis=0)
                max_bound = np.max(cluster_pts, axis=0)

                size_x = float(max_bound[0] - min_bound[0])
                size_y = float(max_bound[1] - min_bound[1])

                center_x_cam = float((min_bound[0] + max_bound[0]) / 2.0)
                center_y_cam = float((min_bound[1] + max_bound[1]) / 2.0)

                nx, ny, nz = plane_normal

                if abs(float(nz)) < 1e-6:
                    continue

                z_table_cam = float(-(float(nx) * center_x_cam + float(ny) * center_y_cam + float(plane_d)) / float(nz))
                z_object_top_cam = float(z_table_cam + estimated_height / float(nz))
                z_object_center_cam = float(z_table_cam + estimated_height / (2.0 * float(nz)))

                pt_cam = PointStamped()
                pt_cam.header.stamp = stamp
                pt_cam.header.frame_id = self.camera_optical_frame
                pt_cam.point.x = float(center_x_cam)
                pt_cam.point.y = float(center_y_cam)
                pt_cam.point.z = float(z_object_center_cam)

                try:
                    pt_robot = self.tf_buffer.transform(pt_cam, self.target_frame, timeout=rclpy.duration.Duration(seconds=0.05))
                except Exception as e:
                    self.get_logger().warn(f"[TF ERROR] {e}", throttle_duration_sec=2.0)
                    continue

                target_x = float(pt_robot.point.x)
                target_y = float(pt_robot.point.y)
                target_z = float(pt_robot.point.z)

                if not np.isfinite([target_x, target_y, target_z]).all():
                    continue

                if valid_cluster_idx == 0:
                    target_msg = PoseStamped()
                    target_msg.header.stamp = stamp
                    target_msg.header.frame_id = self.target_frame
                    target_msg.pose.position.x = target_x
                    target_msg.pose.position.y = target_y
                    target_msg.pose.position.z = target_z
                    target_msg.pose.orientation.x = 0.0
                    target_msg.pose.orientation.y = 0.0
                    target_msg.pose.orientation.z = 0.0
                    target_msg.pose.orientation.w = 1.0
                    self.pose_pub.publish(target_msg)

                    self.get_logger().info(f"[TARGET DETECTED] Center Pos(link0): ({target_x:.3f}, {target_y:.3f}, {target_z:.3f}) | Object Size: {size_x * 100:.1f} x {size_y * 100:.1f} x {estimated_height * 100:.1f}cm", throttle_duration_sec=2.0)

                center_pt_cam = PointStamped()
                center_pt_cam.header.stamp = stamp
                center_pt_cam.header.frame_id = self.camera_optical_frame
                center_pt_cam.point.x = float(center_x_cam)
                center_pt_cam.point.y = float(center_y_cam)
                center_pt_cam.point.z = float(z_object_center_cam)

                try:
                    center_pt_robot = self.tf_buffer.transform(center_pt_cam, self.target_frame, timeout=rclpy.duration.Duration(seconds=0.05))
                except Exception as e:
                    self.get_logger().warn(f"[TF ERROR] Marker center transform failed: {e}", throttle_duration_sec=2.0)
                    continue

                marker_center_x = float(center_pt_robot.point.x)
                marker_center_y = float(center_pt_robot.point.y)
                marker_center_z = float(center_pt_robot.point.z)

                box_marker = Marker()
                box_marker.header.frame_id = self.target_frame
                box_marker.header.stamp = stamp
                box_marker.ns = "object_bbox"
                box_marker.id = valid_cluster_idx * 2
                box_marker.type = Marker.CUBE
                box_marker.action = Marker.ADD
                box_marker.pose.position.x = marker_center_x
                box_marker.pose.position.y = marker_center_y
                box_marker.pose.position.z = marker_center_z
                box_marker.pose.orientation.w = 1.0
                box_marker.scale.x = float(max(size_x, 0.03))
                box_marker.scale.y = float(max(size_y, 0.03))
                box_marker.scale.z = float(max(estimated_height, 0.01))
                box_marker.color.r = 0.1
                box_marker.color.g = 0.8
                box_marker.color.b = 0.2
                box_marker.color.a = 0.6
                marker_array.markers.append(box_marker)

                text_marker = Marker()
                text_marker.header.frame_id = self.target_frame
                text_marker.header.stamp = stamp
                text_marker.ns = "object_label"
                text_marker.id = valid_cluster_idx * 2 + 1
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = marker_center_x
                text_marker.pose.position.y = marker_center_y
                text_marker.pose.position.z = target_z + 0.05
                text_marker.pose.orientation.w = 1.0
                text_marker.scale.z = 0.035
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                text_marker.text = f"Obj_{valid_cluster_idx} (H: {estimated_height * 100:.1f}cm)"
                marker_array.markers.append(text_marker)

                valid_cluster_idx += 1

            if valid_cluster_idx > 0:
                self.marker_pub.publish(marker_array)
            else:
                self.clear_markers()

        except Exception as e:
            self.get_logger().error(f"[DETECTION ERROR] {e}", throttle_duration_sec=2.0)

    def clear_markers(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Ransac3DObjectDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[FATAL] {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()