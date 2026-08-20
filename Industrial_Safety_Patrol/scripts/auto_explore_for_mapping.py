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

        # - LiDAR 최대거리 3.5m를 고려하여 약 2~3m 간격으로 스캔
        # - World_X, World_Y, hold_sec
        self.raw_world_waypoints = [
            # =========================================================================
            # 1. START / 남서 외곽 / 작업대 및 앉은 작업자
            # =========================================================================
            (-9.5, -9.5, 2.0),      # [START] 충전소
            (-9.4, -9.4, 1.0),
            (-9.4, -7.0, 1.5),      # 서쪽 외벽 스캔
            (-9.4, -5.0, 1.0),
            (-9.4, -3.0, 1.0),

            # =========================================================================
            # 2. 남서 내부 / 쓰러진 작업자 / 컨베이어
            # =========================================================================
            (-7.0, -9.4, 1.0),      # 남쪽 외곽
            (-5.0, -9.4, 1.0),
            (-3.0, -9.4, 1.0),
            (-1.0, -9.4, 1.0),
            (1.0, -9.4, 1.0),
            (3.5, -9.4, 1.0),
            (5.0, -9.4, 1.0),       # 전기실 남쪽 진입 전

            # =========================================================================
            # 3. 남동 외곽 / 전기실 우회
            # =========================================================================
            (5.0, -7.0, 2.0),       # 화재 cargo 남쪽 접근
            (4.0, -6.0, 3.0),       # [FIRE] 화재 1 정밀 관측
            (5.0, -5.0, 1.0),

            # 전기실 서측 안전 통로
            (5.0, -3.0, 1.0),
            (5.0, -1.0, 1.0),
            (5.0, 1.0, 1.0),
            (5.0, 3.0, 1.0),
            (5.0, 5.0, 1.0),

            # =========================================================================
            # 4. 동쪽 내부 / 팔레트 랙 우회
            # =========================================================================
            (7.0, 5.0, 1.5),        # 팔레트 랙 동측
            (7.0, 7.0, 1.0),
            (7.0, 9.0, 1.0),

            # =========================================================================
            # 5. 동쪽 외벽 / 화재 2
            # =========================================================================
            (9.4, 9.0, 1.0),
            (9.4, 6.5, 1.0),
            (9.4, 4.5, 1.0),
            (9.0, 3.5, 3.0),        # [FIRE] 동쪽 벽 화재 정밀 관측
            (9.4, 2.0, 1.0),
            (9.4, 0.0, 1.0),
            (9.4, -2.0, 1.0),
            (9.4, -4.0, 1.0),
            (9.4, -5.8, 1.0),

            # =========================================================================
            # 6. 전기실 남/동측 확인
            # =========================================================================
            (8.0, -5.8, 2.0),       # 전기실 북동쪽 접근
            (7.0, -5.8, 1.0),
            (5.0, -5.8, 1.0),

            # =========================================================================
            # 7. 중앙 동측 / 정상 작업자 / 불량 PPE
            # =========================================================================
            (3.5, -5.8, 1.0),
            (2.5, -4.5, 3.0),       # [PPE] 정상 작업자
            (2.5, -2.5, 1.0),
            (3.5, -1.0, 1.0),
            (4.5, 0.5, 1.0),
            (4.5, 3.0, 1.0),
            (4.0, 4.0, 3.0),        # [PPE] 불량 PPE 작업자

            # =========================================================================
            # 8. 북동 / 팔레트 랙 및 지게차 2 주변
            # =========================================================================
            (4.5, 6.5, 1.0),        # 랙 남동측
            (4.5, 7.5, 1.0),
            (4.5, 9.0, 1.0),

            (1.0, 9.0, 1.0),
            (-1.0, 9.0, 1.0),
            (-3.0, 9.0, 1.0),
            (-5.0, 9.0, 1.0),

            # =========================================================================
            # 9. 북서 / 지게차 1 우회
            # =========================================================================
            (-6.5, 9.0, 1.0),
            (-6.5, 7.0, 1.0),
            (-5.0, 7.0, 1.0),
            (-4.0, 6.5, 1.0),       # 90도 bending worker 접근
            (-4.0, 5.5, 2.5),       # [FALL/PPE] 90도 작업자

            # =========================================================================
            # 10. 북쪽 중앙 / 지게차 및 중앙 작업자
            # =========================================================================
            (-2.5, 5.5, 1.0),
            (-2.5, 3.5, 1.0),
            (-2.5, 2.0, 1.0),
            (-2.5, 1.2, 2.5),       # 45도 bending worker 접근
            (-2.5, -0.5, 1.0),

            # =========================================================================
            # 11. 중앙 / 45도 작업자 / 컨베이어
            # =========================================================================
            (-1.0, -0.5, 1.0),
            (1.0, -0.5, 1.0),
            (2.5, -0.5, 1.0),
            (3.5, -0.5, 1.0),

            (3.5, -2.5, 1.0),
            (2.5, -3.5, 1.0),

            # =========================================================================
            # 12. 컨베이어 동측 → 북측 우회
            # =========================================================================
            (1.5, -3.5, 1.0),
            (0.0, -3.5, 1.0),
            (-1.5, -3.5, 1.0),
            (-3.0, -3.5, 1.0),

            # 컨베이어 위치(-5.5,-4.0) 접근
            (-3.5, -3.0, 1.0),
            (-3.5, -2.0, 1.0),

            # =========================================================================
            # 13. 쓰러진 작업자 접근
            # =========================================================================
            (-4.0, -6.0, 1.0),
            (-4.0, -5.5, 2.5),       # [FALL] 쓰러진 작업자 정밀 관측
            (-5.5, -5.5, 1.0),

            # =========================================================================
            # 14. 서쪽 내부 스캔
            # =========================================================================
            (-6.5, -5.5, 1.0),
            (-7.5, -5.5, 1.0),
            (-7.5, -3.5, 1.0),
            (-7.5, -1.5, 1.0),
            (-7.5, 0.5, 1.0),
            (-7.5, 2.5, 1.0),
            (-7.5, 4.5, 1.0),
            (-7.5, 6.0, 1.0),

            # =========================================================================
            # 15. 지게차 1 남측/동측 우회
            # =========================================================================
            (-6.5, 6.0, 1.0),
            (-6.5, 7.5, 1.0),
            (-7.0, 7.5, 1.0),

            # =========================================================================
            # 16. 서쪽 외곽 하강 / LOOP CLOSURE
            # =========================================================================
            (-9.4, 7.5, 1.0),
            (-9.4, 5.0, 1.0),
            (-9.4, 2.5, 1.0),
            (-9.4, 0.0, 1.0),
            (-9.4, -2.5, 1.0),
            (-9.4, -5.0, 1.0),
            (-9.4, -7.0, 1.0),
            (-9.4, -9.0, 1.0),

            # =========================================================================
            # 17. HOME
            # =========================================================================
            (-9.5, -9.5, 5.0)
        ]

        self.waypoints = []
        self.waypoint_idx = 0
        self.arrival_threshold = 0.35
        self.linear_speed = 0.15
        self.angular_speed = 0.35
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
            self.shutdown_timer = self.create_timer(0.5,self._shutdown_after_exploration)
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
    
    def _shutdown_after_exploration(self):
        self.publish_cmd(0.0,0.0)
        if rclpy.ok():
            rclpy.shutdown()

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