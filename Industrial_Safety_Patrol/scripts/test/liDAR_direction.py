#!/usr/bin/env python3
"""
라이다 방향 출력
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanDirectionTest(Node):

    def __init__(self):
        super().__init__("scan_direction_test")

        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.get_logger().info("Waiting for /scan...")

    def get_range(self, scan, angle_deg):
        angle = math.radians(angle_deg)

        index = int(
            round((angle - scan.angle_min) /
                  scan.angle_increment)
        )

        index = max(0, min(index, len(scan.ranges) - 1))

        return scan.ranges[index]

    def scan_callback(self, scan):

        r0 = self.get_range(scan, 0)
        r90 = self.get_range(scan, 90)
        r180 = self.get_range(scan, 180)
        r270 = self.get_range(scan, 270)

        print(f"angle_min = {math.degrees(scan.angle_min):.1f}°")
        print(f"angle_max = {math.degrees(scan.angle_max):.1f}°")
        print(f"increment = {math.degrees(scan.angle_increment):.1f}°")
        print(f"num_ranges = {len(scan.ranges)}")

        print(
            f"0°   : {r0:.3f} m\n"
            f"90°  : {r90:.3f} m\n"
            f"180° : {r180:.3f} m\n"
            f"270° : {r270:.3f} m\n"
            "----------------------"
        )


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