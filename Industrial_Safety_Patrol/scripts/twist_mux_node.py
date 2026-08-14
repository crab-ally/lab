#!/usr/bin/env python3
"""
Node 4: Twist Mux Node

Priority:
    1. EMERGENCY  -> 정지
    2. WARNING    -> 현재 명령 감속
    3. TELEOP     -> 수동 조작
    4. NAV2       -> 자율주행
    5. timeout    -> 정지

Subscribes:
    - /cmd_vel_nav
    - /cmd_vel_teleop
    - /ttc_alerts

Publishes:
    - /cmd_vel
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class TwistMuxNode(Node):

    def __init__(self) -> None:
        super().__init__('twist_mux_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('slowdown_ratio', 0.5)

        self.cmd_timeout = (
            self.get_parameter('cmd_timeout')
            .get_parameter_value()
            .double_value
        )

        self.slowdown_ratio = (
            self.get_parameter('slowdown_ratio')
            .get_parameter_value()
            .double_value
        )

        # ── Latest Commands ───────────────────────────────────────────
        self.latest_nav_cmd = Twist()
        self.latest_teleop_cmd = Twist()

        self.last_nav_time = 0.0
        self.last_teleop_time = 0.0
        self.last_alert_time = 0.0

        # 현재 TTC 위험 상태
        self.current_risk_level = "NORMAL"

        # ── Subscribers ───────────────────────────────────────────────

        # Nav2
        self.sub_nav_cmd = self.create_subscription(
            Twist,
            '/cmd_vel_nav',
            self._nav_cmd_callback,
            10
        )

        # Teleop
        self.sub_teleop_cmd = self.create_subscription(
            Twist,
            '/cmd_vel_teleop',
            self._teleop_cmd_callback,
            10
        )

        # TTC Alert
        self.sub_ttc_alerts = self.create_subscription(
            String,
            '/ttc_alerts',
            self._ttc_alerts_callback,
            10
        )

        # ── Publisher ────────────────────────────────────────────────
        self.pub_final_cmd_vel = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # 20 Hz
        self.timer = self.create_timer(
            0.05,
            self._control_loop
        )

        self.get_logger().info(
            'Node 4: Twist Mux Node is ready.'
        )

    # ==================================================================
    # Callbacks
    # ==================================================================

    def _nav_cmd_callback(self, msg: Twist) -> None:
        """Nav2 명령 수신"""
        self.latest_nav_cmd = msg
        self.last_nav_time = (
            self.get_clock().now().nanoseconds * 1e-9
        )

    def _teleop_cmd_callback(self, msg: Twist) -> None:
        """Teleop 명령 수신"""
        self.latest_teleop_cmd = msg
        self.last_teleop_time = (
            self.get_clock().now().nanoseconds * 1e-9
        )

    def _ttc_alerts_callback(self, msg: String) -> None:
        """TTC 위험 상태 수신"""
        try:
            payload = json.loads(msg.data)

            self.current_risk_level = payload.get(
                'risk_level',
                'NORMAL'
            )

            self.last_alert_time = (
                self.get_clock().now().nanoseconds * 1e-9
            )

        except json.JSONDecodeError as e:
            self.get_logger().error(
                f'TTC Alert JSON Decode Error: {e}'
            )

    # ==================================================================
    # Control
    # ==================================================================

    def _control_loop(self) -> None:
        """최종 /cmd_vel 결정"""

        now = self.get_clock().now().nanoseconds * 1e-9

        final_cmd = Twist()

        # --------------------------------------------------------------
        # 1. TTC Alert timeout
        # --------------------------------------------------------------
        if (
            self.last_alert_time > 0
            and (now - self.last_alert_time) > self.cmd_timeout
        ):
            self.get_logger().warn(
                'TTC Alert stream timeout! Safety stop applied.'
            )

            self.pub_final_cmd_vel.publish(final_cmd)
            return

        # --------------------------------------------------------------
        # 2. EMERGENCY
        # --------------------------------------------------------------
        if self.current_risk_level == "EMERGENCY":

            # 무조건 정지
            final_cmd.linear.x = 0.0
            final_cmd.linear.y = 0.0
            final_cmd.angular.z = 0.0

        # --------------------------------------------------------------
        # 3. WARNING
        # --------------------------------------------------------------
        elif self.current_risk_level == "WARNING":

            # Teleop이 최근에 들어왔다면 Teleop 우선
            if (
                now - self.last_teleop_time
                <= self.cmd_timeout
            ):
                final_cmd.linear.x = (
                    self.latest_teleop_cmd.linear.x
                    * self.slowdown_ratio
                )

                final_cmd.linear.y = (
                    self.latest_teleop_cmd.linear.y
                    * self.slowdown_ratio
                )

                final_cmd.angular.z = (
                    self.latest_teleop_cmd.angular.z
                    * self.slowdown_ratio
                )

            # Teleop이 없으면 Nav2
            elif (
                now - self.last_nav_time
                <= self.cmd_timeout
            ):
                final_cmd.linear.x = (
                    self.latest_nav_cmd.linear.x
                    * self.slowdown_ratio
                )

                final_cmd.linear.y = (
                    self.latest_nav_cmd.linear.y
                    * self.slowdown_ratio
                )

                final_cmd.angular.z = (
                    self.latest_nav_cmd.angular.z
                    * self.slowdown_ratio
                )

            # 둘 다 없으면 정지
            else:
                final_cmd.linear.x = 0.0
                final_cmd.linear.y = 0.0
                final_cmd.angular.z = 0.0

        # --------------------------------------------------------------
        # 4. NORMAL
        # --------------------------------------------------------------
        else:

            # ----------------------------------------------------------
            # Teleop 우선
            # ----------------------------------------------------------
            if (
                now - self.last_teleop_time
                <= self.cmd_timeout
            ):
                final_cmd = self.latest_teleop_cmd

            # ----------------------------------------------------------
            # Teleop이 없으면 Nav2
            # ----------------------------------------------------------
            elif (
                now - self.last_nav_time
                <= self.cmd_timeout
            ):
                final_cmd = self.latest_nav_cmd

            # ----------------------------------------------------------
            # 둘 다 없으면 정지
            # ----------------------------------------------------------
            else:
                final_cmd.linear.x = 0.0
                final_cmd.linear.y = 0.0
                final_cmd.angular.z = 0.0

        # --------------------------------------------------------------
        # 최종 명령
        # --------------------------------------------------------------
        self.pub_final_cmd_vel.publish(final_cmd)


def main(args=None) -> None:

    rclpy.init(args=args)

    node = TwistMuxNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info(
            'Twist Mux Node Stopped.'
        )

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()