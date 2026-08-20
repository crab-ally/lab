#!/usr/bin/env python3
"""
SLAM 매핑을 위한 자동 탐색 주행 스크립트
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
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_nav", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_received = False
        self.waypoints_converted = False

        # - LiDAR 최대거리 3.5m를 고려하여 약 2~3m 간격으로 스캔
        # - World_X, World_Y, hold_sec
        self.raw_world_waypoints = [
            # =========================================================================
            # 1. START / 3사분면 아래
            # =========================================================================
            (-9.5, -9.5, 2.0),      # [START] 충전소
            (-9.4, -6.0, 1.5),      # 서쪽 외벽 스캔
            (-6.0, -6.0, 1.0),
            (-6.0, -9.5, 2.0),

            # =========================================================================
            # 2. 4사분면 / 3사분면 위
            # =========================================================================
            (5.0, -9.5, 2.0),
            (5.0, -5.0, 1.0),
            (9.0, -5.0, 1.0),
            (9.0, -3.0, 1.0),
            (5.0, -3.0, 1.0),
            (2.0, -3.0, 1.0),
            (-2.0, -3.0, 1.0),
            (-5.0, -3.0, 1.0),
            (-9.0, -3.0, 1.0),

            # =========================================================================
            # 3. 2사분면
            # =========================================================================
            (-9.0, -1.0, 1.0),
            (-9.0, 1.0, 1.0),
            (-9.0, 3.0, 1.0),
            (-9.0, 5.0, 1.0),
            (-9.0, 7.0, 2.0),
            (-6.0, 7.0, 2.0),
            (-2.5, 7.0, 1.0),
            (0.0, 7.0, 1.0),
            (0.0, 3.0, 1.0),

            # =========================================================================
            # 4. 1사분면
            # =========================================================================
            (3.0, 3.0, 1.0),
            (5.5, 3.0, 2.0),
            (8.0, 3.0, 1.0),
            (8.0, 7.5, 1.0),
            (4.0, 7.5, 3.0),
            (4.0, 9.5, 1.0),
            (0.0, 9.5, 3.0),

            # =========================================================================
            # 5. 북쪽 벽 / 서쪽 벽
            # =========================================================================
            (-3.0, 9.5, 1.0),
            (-5.0, 9.5, 1.0),
            (-7.0, 9.5, 1.0),
            (-9.5, 9.5, 1.0),        # [FIRE 2] 관측 (Z축 상승)
            (-9.5, 7.0, 1.0),        # Z축 복귀
            (-9.5, 4.0, 1.0),
            (-9.5, 1.0, 1.0),
            (-9.5, -1.0, 1.0),

            # =========================================================================
            # 6. 십자 복도
            # =========================================================================
            (-5.0, -1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (3.0, -1.0, 1.0),
            (7.0, -1.0, 1.0),
            (9.0, -1.0, 1.0),

            # =========================================================================
            # 7. 4사분면 중간 / 복귀
            # =========================================================================
            (9.0, -5.5, 1.0),
            (6.0, -5.5, 1.0),
            (3.0, -5.5, 2.0),
            (0.0, -5.5, 1.0),
            (-3.0, -5.5, 2.0),
            (-6.0, -5.5, 2.0),
            (-9.0, -5.5, 1.0),

            (-9.5, -9.5, 5.0)       # [HOME] 최종 귀환 복귀
        ]

        self.waypoints = []
        self.waypoint_idx = 0
        self.arrival_threshold = 0.35
        self.linear_speed = 0.18
        self.angular_speed = 0.40
        self.hold_until = None
        self.exploration_done = False

        # 로봇의 초기 위치 및 회전각
        self.world_start_x = -9.5
        self.world_start_y = -9.5
        self.world_start_yaw = math.pi / 2.0  # quat="0.707107 0 0 0.707107" -> +90도(1.570796rad)

        self.last_reported_wp = -1
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Auto explore node initialized for 20x20 Factory. Waiting for first /odom...")

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

            self.waypoints.append((odom_x, odom_y, hold, wx, wy))

        self.get_logger().info(
            f"Successfully converted {len(self.waypoints)} waypoints to /odom frame! "
            f"(World Start: [{self.world_start_x}, {self.world_start_y}], Yaw: {self.world_start_yaw:.2f} rad)"
        )

    def control_loop(self):
        if not self.odom_received or not self.waypoints_converted:
            return

        if self.exploration_done:
            self.publish_cmd(0.0, 0.0)
            return

        current_time = self.get_clock().now()

        if self.hold_until is not None:
            if current_time < self.hold_until:
                self.publish_cmd(0.0, 0.0)
                return
            self.hold_until = None
            self.waypoint_idx += 1

        if self.waypoint_idx >= len(self.waypoints):
            self.exploration_done = True
            self.get_logger().info("All waypoints explored successfully! Stopping robot.")
            self.publish_cmd(0.0, 0.0)
            self.shutdown_timer = self.create_timer(1.0, self._shutdown_after_exploration)
            return

        odom_x, odom_y, hold_sec, world_x, world_y = self.waypoints[self.waypoint_idx]

        if self.last_reported_wp != self.waypoint_idx:
            self.last_reported_wp = self.waypoint_idx
            self.get_logger().info(
                f"[{self.waypoint_idx + 1}/{len(self.waypoints)}] Navigating -> World: ({world_x:.1f}, {world_y:.1f}) | Odom: ({odom_x:.2f}, {odom_y:.2f}) (Hold: {hold_sec}s)"
            )

        dx = odom_x - self.x
        dy = odom_y - self.y
        distance = math.hypot(dx, dy)

        if distance < self.arrival_threshold:
            self.publish_cmd(0.0, 0.0)
            if hold_sec > 0.0:
                self.get_logger().info(f"Arrived at WP {self.waypoint_idx + 1}. Scanning for {hold_sec}s...")
                from rclpy.duration import Duration
                self.hold_until = current_time + Duration(seconds=hold_sec)
            else:
                self.waypoint_idx += 1
            return

        target_yaw = math.atan2(dy, dx)
        yaw_error = self._normalize_angle(target_yaw - self.yaw)

        # 큰 각도 오차 시 제자리 회전, 작은 각도 오차 시 직진 + 조향
        if abs(yaw_error) > 0.30:
            ang_cmd = self.angular_speed * math.copysign(1.0, yaw_error)
            self.publish_cmd(0.0, ang_cmd)
        else:
            lin_cmd = self.linear_speed * max(0.2, math.cos(yaw_error))
            ang_cmd = 0.5 * yaw_error
            self.publish_cmd(lin_cmd, ang_cmd)

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
    
    def _shutdown_after_exploration(self):
        self.publish_cmd(0.0, 0.0)
        self.get_logger().info("Exploration finished. Shutting down node...")
        self.destroy_node()
        rclpy.shutdown()

def main():
    rclpy.init()
    node = AutoExploreNode()

    def shutdown_handler(sig, frame):
        node.get_logger().info("Shutdown signal received")
        node.stop_robot()
        if rclpy.ok():
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