#!/usr/bin/env python3
"""
SLAM 매핑을 위한 자동 탐색 주행 스크립트.
Bridge의 /odom 오프셋 및 초기 회전(Quaternion)을 동적으로 반영하도록 수정되었습니다.
"""

import math
import time
import signal
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class AutoExploreNode(Node):
    def __init__(self):
        super().__init__("auto_explore_for_mapping")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_received = False
        self.waypoints_converted = False

        # 공장 월드 좌표계 기준 Waypoints [World_X, World_Y, hold_sec]
        # (MuJoCo 월드 기준: -4~4m 범위)
        self.raw_world_waypoints = [
            # 1. 초기 위치 (-4,-4) 주변
            (-4.0, -4.0, 3.0),
            (-4.0, -2.0, 3.0),
            (-4.0, 0.0, 3.0),
            (-4.0, 2.0, 3.0),
            (-4.0, 4.0, 10.0),
            # 2. 북쪽 벽 전체 스캔
            (-2.0, 4.2, 3.0),
            (0.0, 4.2, 3.0),
            (2.0, 4.2, 3.0),
            (4.0, 4.2, 10.0),
            # 3. 동쪽 벽 + 화재구역 주변
            (4.4, 3.5, 3.0),
            (4.4, 1.5, 3.0),
            (4.4, -1.0, 3.0),
            (4.4, -2.5, 10.0),
            # 4. 남쪽 벽 스캔
            (3.0, -2.5, 3.0),
            (3.0, -4.2, 3.0),
            (1.5, -4.2, 3.0),
            (0.0, -4.2, 3.0),
            (-2.0, -4.2, 10.0),
            # 5. 내부 지그재그 탐색
            (-3.0, -2.5, 3.0),
            (0.0, -2.5, 3.0),
            (3.0, -2.5, 3.0),
            (3.0, -1.0, 3.0),
            (0.5, -1.0, 3.0),
            (-4.5, -1.0, 3.0),
            (-4.0, 1.5, 3.0),
            (0.5, 1.5, 3.0),
            (3.0, 1.5, 3.0),
            (2.0, 3.0, 3.0),
            (-1.5, 3.0, 3.0),
            (-1.5, 2.5, 3.0),
            (-1.5, 2.0, 3.0),
            (0.0, 1.5, 10.0),
            # 6. 중앙 회전
            (1.0, 0.0, 10.0),
            # 7. 시작점 복귀
            (1.0, -1.0, 3.0),
            (-3.0, -1.0, 3.0),
            (-3.0, -4.0, 3.0),
            (-4.0, -4.0, 10.0)
        ]

        self.waypoints = []
        self.waypoint_idx = 0
        self.arrival_threshold = 0.35
        self.linear_speed = 0.25
        self.angular_speed = 0.6
        self.hold_until = None
        self.exploration_done = False

        # MuJoCo 월드에서 로봇의 초기 물리 위치 및 회전각 (XML 기준)
        self.world_start_x = -4.0
        self.world_start_y = -4.0
        self.world_start_yaw = math.pi / 2.0  # quat="0.707 0 0 0.707" -> +90도(1.57rad)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Auto explore node initialized. Waiting for first /odom...")

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

        # 최초 /odom 수신 시 Odom 좌표계에 맞게 Waypoint 변환
        if not self.waypoints_converted:
            self._convert_waypoints()
            self.waypoints_converted = True

        self.odom_received = True

    def _convert_waypoints(self):
        """
        MuJoCo 월드 좌표계 기준의 Waypoints를 
        Bridge가 변환해 준 /odom 상대 좌표계에 맞춰 변환 (평행이동 + 90도 역회전 적용)
        """
        self.waypoints = []
        cos_yaw = math.cos(-self.world_start_yaw)
        sin_yaw = math.sin(-self.world_start_yaw)

        for wx, wy, hold in self.raw_world_waypoints:
            # 1. 시작점 오프셋 차감 (Translation)
            dx = wx - self.world_start_x
            dy = wy - self.world_start_y

            # 2. 로봇 초기 Orientation(+90도) 역회전 적용 (Rotation)
            odom_x = dx * cos_yaw - dy * sin_yaw
            odom_y = dx * sin_yaw + dy * cos_yaw

            self.waypoints.append((odom_x, odom_y, hold))

        self.get_logger().info(f"Successfully converted {len(self.waypoints)} waypoints to /odom frame!")

    def control_loop(self):
        if not self.odom_received or not self.waypoints_converted:
            return

        if self.exploration_done:
            self.publish_cmd(0.0, 0.0)
            return

        if self.hold_until is not None:
            if time.time() < self.hold_until:
                self.publish_cmd(0.0, 0.0)
                return
            self.hold_until = None
            self.waypoint_idx += 1

        if self.waypoint_idx >= len(self.waypoints):
            self.exploration_done = True
            self.get_logger().info("Exploration complete. Stop and save the map.")
            self.publish_cmd(0.0, 0.0)
            return

        target_x, target_y, hold_sec = self.waypoints[self.waypoint_idx]
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)

        if distance < self.arrival_threshold:
            self.publish_cmd(0.0, 0.0)
            self.hold_until = time.time() + hold_sec
            return

        target_yaw = math.atan2(dy, dx)
        yaw_error = self._normalize_angle(target_yaw - self.yaw)

        if abs(yaw_error) > 0.25:
            self.publish_cmd(0.0, self.angular_speed * math.copysign(1.0, yaw_error))
        else:
            self.publish_cmd(self.linear_speed, 0.4 * yaw_error)

    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.get_logger().info("Stopping robot...")
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0

        if self.timer:
            self.timer.cancel()

        for _ in range(5):
            self.cmd_pub.publish(msg)
            time.sleep(0.05)

def main():
    rclpy.init()
    node = AutoExploreNode()

    def shutdown_handler(sig, frame):
        node.get_logger().info("Shutdown signal received")
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(str(e))
    finally:
        if rclpy.ok():
            node.stop_robot()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()