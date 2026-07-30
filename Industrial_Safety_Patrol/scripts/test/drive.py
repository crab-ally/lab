#!/usr/bin/env python3
"""
1m 전진 -> 90도 반시계 회전 -> 1m 전진
"""

import math
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class ScanDirectionTest(Node):

    def __init__(self):
        super().__init__("scan_direction_test")

        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.get_logger().info("Waiting for /scan...")

        # ROS 통신 안정화
        time.sleep(1.0)

        # 1. 1m 전진
        self.drive_forward(1.0)

        time.sleep(1.0)

        # 2. 90도 반시계 회전
        self.rotate_ccw(90)

        time.sleep(1.0)

        # 3. 1m 전진
        self.drive_forward(1.0)

        self.get_logger().info("Mission Complete!")

    def drive_forward(self, distance):
        """distance(m) 만큼 전진"""

        speed = 0.1  # m/s
        duration = distance / speed

        twist = Twist()

        start = time.time()

        while rclpy.ok() and (time.time() - start) < duration:
            twist.linear.x = speed
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            time.sleep(0.05)

        # 정지
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

        self.get_logger().info(f"Drive {distance:.2f} m finished.")

    def rotate_ccw(self, angle_deg):
        """angle_deg 만큼 회전 (+: CCW, -: CW)"""

        angular_speed = 0.5  # rad/s

        angle_rad = math.radians(abs(angle_deg))
        duration = angle_rad / angular_speed

        twist = Twist()
        twist.linear.x = 0.0

        if angle_deg >= 0:
            twist.angular.z = angular_speed
        else:
            twist.angular.z = -angular_speed

        start = time.time()

        while rclpy.ok() and (time.time() - start) < duration:
            self.cmd_pub.publish(twist)
            time.sleep(0.05)

        # 정지
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

        self.get_logger().info(f"Rotate {angle_deg:.1f}° finished.")

    def scan_callback(self, scan):
        pass


def main():
    rclpy.init()

    node = ScanDirectionTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()