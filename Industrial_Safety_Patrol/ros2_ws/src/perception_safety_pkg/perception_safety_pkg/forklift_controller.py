#!/usr/bin/env python3
"""
publish
  /forklift_1/cmd_vel
  /forklift_2/cmd_vel
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ForkliftController(Node):
    def __init__(self):
        super().__init__('forklift_controller')

        self.declare_parameter('forklift_1_linear_x',0.3)
        self.declare_parameter('forklift_1_linear_y',0.0)
        self.declare_parameter('forklift_1_angular_z',0.0)

        self.declare_parameter('forklift_2_linear_x',0.2)
        self.declare_parameter('forklift_2_linear_y',0.0)
        self.declare_parameter('forklift_2_angular_z',0.0)

        self.declare_parameter('publish_rate',10.0)

        self.fl1_linear_x=self.get_parameter('forklift_1_linear_x').value
        self.fl1_linear_y=self.get_parameter('forklift_1_linear_y').value
        self.fl1_angular_z=self.get_parameter('forklift_1_angular_z').value

        self.fl2_linear_x=self.get_parameter('forklift_2_linear_x').value
        self.fl2_linear_y=self.get_parameter('forklift_2_linear_y').value
        self.fl2_angular_z=self.get_parameter('forklift_2_angular_z').value

        publish_rate=self.get_parameter('publish_rate').value

        self.fl1_pub=self.create_publisher(
            Twist,
            '/forklift_1/cmd_vel',
            10
        )

        self.fl2_pub=self.create_publisher(
            Twist,
            '/forklift_2/cmd_vel',
            10
        )

        self.timer=self.create_timer(
            1.0/publish_rate,
            self.publish_commands
        )

        self.get_logger().info(
            'Forklift Controller is ready.'
        )

        self.get_logger().info(
            f'Forklift 1: x={self.fl1_linear_x:.2f}, '
            f'y={self.fl1_linear_y:.2f}, '
            f'z={self.fl1_angular_z:.2f}'
        )

        self.get_logger().info(
            f'Forklift 2: x={self.fl2_linear_x:.2f}, '
            f'y={self.fl2_linear_y:.2f}, '
            f'z={self.fl2_angular_z:.2f}'
        )

    def publish_commands(self):
        fl1_cmd=Twist()

        fl1_cmd.linear.x=self.fl1_linear_x
        fl1_cmd.linear.y=self.fl1_linear_y
        fl1_cmd.angular.z=self.fl1_angular_z

        self.fl1_pub.publish(fl1_cmd)

        fl2_cmd=Twist()

        fl2_cmd.linear.x=self.fl2_linear_x
        fl2_cmd.linear.y=self.fl2_linear_y
        fl2_cmd.angular.z=self.fl2_angular_z

        self.fl2_pub.publish(fl2_cmd)


def main(args=None):
    rclpy.init(args=args)

    node=ForkliftController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd=Twist()

        node.fl1_pub.publish(stop_cmd)
        node.fl2_pub.publish(stop_cmd)

        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()