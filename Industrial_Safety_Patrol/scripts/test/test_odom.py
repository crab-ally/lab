#!/usr/bin/env python3
"""
1m 이동
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

        # 노드가 시작되면 1m 직진
        self.drive_forward_1m()

    def drive_forward_1m(self):

        twist = Twist()

        speed = 0.2          # m/s
        distance = 1.0       # m
        duration = distance / speed

        start = time.time()

        while rclpy.ok() and (time.time() - start) < duration:
            twist.linear.x = speed
            self.cmd_pub.publish(twist)
            time.sleep(0.05)

        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)

        self.get_logger().info("Finished driving 1 meter.")

    def get_range(self, scan, angle_deg):
        angle = math.radians(angle_deg)

        index = int(
            round((angle - scan.angle_min) /
                  scan.angle_increment)
        )

        index = max(0, min(index, len(scan.ranges) - 1))

        return scan.ranges[index]

    def scan_callback(self, scan):

        # print("=" * 50)

        for i in range(0, 360, 30):
            angle = math.radians(i)

            index = int(
                round((angle - scan.angle_min) /
                      scan.angle_increment)
            )

            index = max(0, min(index, len(scan.ranges) - 1))

            # print(f"{i:3d}° : {scan.ranges[index]:.3f}")


def main():
    rclpy.init()

    node = ScanDirectionTest() # 1m 이동

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()