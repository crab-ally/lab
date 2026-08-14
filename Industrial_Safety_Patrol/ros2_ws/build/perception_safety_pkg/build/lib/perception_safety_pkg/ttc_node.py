#!/usr/bin/env python3
"""
Node 3: TTC (Time-To-Collision) Node

Subscribes:
    - /tracks_3d (std_msgs/msg/String - JSON Format from Node 2)
    - /odom (nav_msgs/msg/Odometry) - 로봇의 현재 속도 측정용

Publishes:
    - /ttc_alerts (std_msgs/msg/String - JSON Format)
      [Fields: min_ttc, risk_level, target_track_id, ppe_violation_count, timestamp]
    - /cmd_vel_safety (geometry_msgs/msg/Twist) - 감속 및 비상 정지 명령
"""

import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class TTCNode(Node):
    def __init__(self) -> None:
        super().__init__('ttc_node')

        # ── Parameter Settings ─────────────────────────────────────────
        self.declare_parameter('warning_ttc', 3.0)     # 주의(Warning) 임계 시간 (초)
        self.declare_parameter('emergency_ttc', 1.5)   # 비상(Emergency) 임계 시간 (초)
        self.declare_parameter('robot_radius', 0.5)    # 로봇 안전 반경 (미터)
        self.declare_parameter('slowdown_factor', 0.5) # Warning 시 감속 비율

        self.warning_ttc = self.get_parameter('warning_ttc').get_parameter_value().double_value
        self.emergency_ttc = self.get_parameter('emergency_ttc').get_parameter_value().double_value
        self.robot_radius = self.get_parameter('robot_radius').get_parameter_value().double_value
        self.slowdown_factor = self.get_parameter('slowdown_factor').get_parameter_value().double_value

        # 로봇 속도 상태 변수 (base_link 기준)
        self.robot_vx = 0.0
        self.robot_vy = 0.0

        # ── QoS Profile ───────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # ── Subscriptions & Publishers ─────────────────────────────────
        self.sub_tracks_3d = self.create_subscription(
            String,
            '/tracks_3d',
            self._tracks_callback,
            10
        )

        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            sensor_qos
        )

        self.pub_ttc_alerts = self.create_publisher(
            String,
            '/ttc_alerts',
            10
        )

        self.pub_cmd_vel_safety = self.create_publisher(
            Twist,
            '/cmd_vel_safety',
            10
        )

        self.get_logger().info('Node 3: TTC Node (Risk Assessment & Alert) is ready.')

    def _odom_callback(self, msg: Odometry) -> None:
        """로봇의 현재 선속도 저장"""
        self.robot_vx = msg.twist.twist.linear.x
        self.robot_vy = msg.twist.twist.linear.y

    def _tracks_callback(self, msg: String) -> None:
        """3D Track 수신 시 각 객체별 TTC 계산 및 위험 수준 판단"""
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON Decode Error: {e}')
            return

        stamp = payload['header']['stamp']
        tracks = payload.get('tracks', [])

        min_ttc = float('inf')
        most_dangerous_track_id = -1
        ppe_violation_count = 0
        overall_risk_level = "NORMAL"  # NORMAL, WARNING, EMERGENCY

        for trk in tracks:
            track_id = trk['track_id']
            px, py, _ = trk['position']
            vx, vy = trk['velocity']
            ppe_ok = trk['ppe_ok']

            if trk['class_name'] == 'person' and not ppe_ok:
                ppe_violation_count += 1

            # 상대 위치 및 상대 속도 연산
            # relative_pos: 로봇에서 장애물 방향 Vector [px, py]
            # relative_vel: 장애물 속도 - 로봇 속도 [vx - robot_vx, vy - robot_vy]
            rel_px = px
            rel_py = py
            rel_vx = vx - self.robot_vx
            rel_vy = vy - self.robot_vy

            distance = math.hypot(rel_px, rel_py) - self.robot_radius
            if distance <= 0.0:
                distance = 0.01  # 최소 거리 보정

            # 상대 접근 속도 (Vector Projection)
            # 음수(-)인 경우 서로 접근하고 있음을 의미
            closing_speed = -(rel_px * rel_vx + rel_py * rel_vy) / math.hypot(rel_px, rel_py)

            # 충돌 임계 시간 (TTC) 산출
            if closing_speed > 0.01:  # 접근 중인 경우에만 계산
                ttc = distance / closing_speed
            else:
                ttc = float('inf')  # 멀어지거나 정지 상태

            if ttc < min_ttc:
                min_ttc = ttc
                most_dangerous_track_id = track_id

        # ── Risk Level 판단 ───────────────────────────────────────────
        if min_ttc <= self.emergency_ttc:
            overall_risk_level = "EMERGENCY"
        elif min_ttc <= self.warning_ttc or ppe_violation_count > 0:
            overall_risk_level = "WARNING"
        else:
            overall_risk_level = "NORMAL"

        # ── 1. Safety Command (/cmd_vel_safety) 생성 및 발행 ──────────
        self._publish_safety_cmd(overall_risk_level)

        # ── 2. Alert Payload (/ttc_alerts) 발행 ────────────────────────
        alert_payload = {
            'header': {'stamp': stamp},
            'risk_level': overall_risk_level,
            'min_ttc': round(min_ttc, 2) if min_ttc != float('inf') else -1.0,
            'target_track_id': most_dangerous_track_id,
            'ppe_violation_count': ppe_violation_count
        }

        json_msg = String()
        json_msg.data = json.dumps(alert_payload, ensure_ascii=False)
        self.pub_ttc_alerts.publish(json_msg)

        # 로깅
        if overall_risk_level != "NORMAL":
            self.get_logger().warn(
                f'[{overall_risk_level}] Min TTC: {min_ttc:.2f}s (Track ID: {most_dangerous_track_id}), PPE Violations: {ppe_violation_count}'
            )

    def _publish_safety_cmd(self, risk_level: str) -> None:
        """위험 수준에 따른 감속/정지 제어 명령 발행"""
        safety_cmd = Twist()

        if risk_level == "EMERGENCY":
            # 비상 정지
            safety_cmd.linear.x = 0.0
            safety_cmd.linear.y = 0.0
            safety_cmd.angular.z = 0.0
        elif risk_level == "WARNING":
            # 속도 감속 (slowdown_factor 적용)
            safety_cmd.linear.x = self.robot_vx * self.slowdown_factor
            safety_cmd.linear.y = self.robot_vy * self.slowdown_factor
            safety_cmd.angular.z = 0.0
        else:
            # NORMAL 상태일 때는 안전 노드에서 별도 개입하지 않음 (Twist Mux priority 제어용)
            return

        self.pub_cmd_vel_safety.publish(safety_cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TTCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('TTC Node Stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()