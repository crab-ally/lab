#!/usr/bin/env python3

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

        for i in range(0, 360, 30):
            print(i, msg.ranges[i])

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