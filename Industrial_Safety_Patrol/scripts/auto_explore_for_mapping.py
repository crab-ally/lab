#!/usr/bin/env python3
"""SLAM 매핑을 위한 자동 탐색 주행 스크립트.

공장 환경을 순회하며 /cmd_vel을 발행해 SLAM Toolbox가 지도를 생성하도록 한다.
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

import signal
import sys


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

        # 공장 바닥(10m x 10m) 순회 경로 [x, y, hold_sec]
        self.waypoints = [
            (-3.8, -3.8, 2.0),
            (-4.0, -0.5, 2.0),
            (-4.0, 3.5, 2.0),
            (-1.0, 4.0, 2.0),
            (2.5,4.3,2.0),
            (4.3,4.3,2.0),
            (4.3,2.5,2.0),
            (4.0, 0.0, 2.0),
            (4.0, -2.5, 2.0),
            (1.5, -4.0, 2.0),
            (-4.0, -3.8, 2.0)
        ]
        self.waypoint_idx = 0
        self.arrival_threshold = 0.35
        self.linear_speed = 0.25
        self.angular_speed = 0.6
        self.hold_until = None
        self.exploration_done = False

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            "Auto explore started. Drive the robot to build the SLAM map."
        )

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
        self.odom_received = True

    def control_loop(self):
        if not self.odom_received:
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
            self.publish_cmd(self.linear_speed, 0.4 * yaw_error) # p제어

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

        # 여러 번 보내서 확실히 전달
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

    signal.signal(
        signal.SIGINT,
        shutdown_handler
    )

    signal.signal(
        signal.SIGTERM,
        shutdown_handler
    )

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
