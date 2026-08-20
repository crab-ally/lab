#!/usr/bin/env python3
"""
Node 3: TTC (Time-To-Collision) Node

Subscribes:
    - /tracks_3d (std_msgs/msg/String - JSON Format from Node 2)
    - /odom (nav_msgs/msg/Odometry) - 로봇의 현재 속도 측정용

Publishes:
    - /ttc_alerts (std_msgs/msg/String - JSON Format)
      [Fields: min_ttc, risk_level, target_track_id, target_subject, timestamp]
"""

import json
import math
from typing import Tuple

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
        self.declare_parameter('warning_ttc', 3.0)
        self.declare_parameter('emergency_ttc', 1.5)

        # 객체 반경
        self.declare_parameter('robot_radius', 0.5)
        self.declare_parameter('person_radius', 0.2)
        self.declare_parameter('forklift_radius', 0.9)

        self.warning_ttc = self.get_parameter('warning_ttc').get_parameter_value().double_value
        self.emergency_ttc = self.get_parameter('emergency_ttc').get_parameter_value().double_value

        self.robot_radius = self.get_parameter('robot_radius').get_parameter_value().double_value
        self.person_radius = self.get_parameter('person_radius').get_parameter_value().double_value
        self.forklift_radius = self.get_parameter('forklift_radius').get_parameter_value().double_value

        # ── Robot velocity state ───────────────────────────────────────
        self.robot_vx = 0.0
        self.robot_vy = 0.0

        # ── QoS Profile ────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # ── Subscriptions ──────────────────────────────────────────────
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

        # ── Publishers ─────────────────────────────────────────────────
        self.pub_ttc_alerts = self.create_publisher(
            String,
            '/ttc_alerts',
            10
        )

        # ── Heartbeat Timer (10 Hz) ────────────────────────────────────
        # 인지 파이프라인의 프레임레이트 지연과 무관하게 스트림 유지
        self.latest_alert_payload = {
            'risk_level': 'NORMAL',
            'min_ttc': -1.0,
            'target_track_id': -1,
            'target_subject': 'NONE'
        }
        self.last_track_update_time = 0.0

        self.timer = self.create_timer(
            0.1,  # 10Hz
            self._timer_callback
        )

        self.get_logger().info('Node 3: TTC Node (Risk Assessment & Alert) is ready.')

    def _timer_callback(self) -> None:
        """10Hz 주기로 최신 TTC Alert 스트림을 안정적으로 발행 (Heartbeat)"""
        now = self.get_clock().now().nanoseconds * 1e-9

        # 마지막 트랙 업데이트 후 1.0초 이상 미수신 시 안전하게 NORMAL로 복구
        if self.last_track_update_time == 0.0 or (now - self.last_track_update_time) > 1.0:
            payload = {
                'header': {'stamp': now},
                'risk_level': 'NORMAL',
                'min_ttc': -1.0,
                'target_track_id': -1,
                'target_subject': 'NONE'
            }
        else:
            payload = dict(self.latest_alert_payload)
            payload['header'] = {'stamp': now}

        json_msg = String()
        json_msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_ttc_alerts.publish(json_msg)

    def _odom_callback(self, msg: Odometry) -> None:
        """로봇의 현재 선속도 저장"""
        self.robot_vx = msg.twist.twist.linear.x
        self.robot_vy = msg.twist.twist.linear.y

    def _calculate_ttc(
        self,
        pos_a: Tuple[float, float],
        vel_a: Tuple[float, float],
        pos_b: Tuple[float, float],
        vel_b: Tuple[float, float],
        radius_sum: float = 0.0
    ) -> float:
        """
        두 객체의 상대 위치/속도를 이용하여 TTC 계산.

        radius_sum:
            두 객체의 충돌 반경 합.
            예:
                로봇-사람     = robot_radius + person_radius
                로봇-지게차   = robot_radius + forklift_radius
                지게차-사람   = forklift_radius + person_radius
        """

        # 상대 위치
        rel_px = pos_b[0] - pos_a[0]
        rel_py = pos_b[1] - pos_a[1]

        # 상대 속도
        rel_vx = vel_b[0] - vel_a[0]
        rel_vy = vel_b[1] - vel_a[1]

        # 두 객체 중심 사이 거리
        dist_val = math.hypot(rel_px, rel_py)

        # 중심이 거의 같은 위치
        if dist_val < 1e-6:
            return 0.01

        # 객체의 실제 외곽까지 남은 거리
        distance = dist_val - radius_sum

        # 이미 충돌 영역에 들어온 경우
        if distance <= 0.0:
            distance = 0.01

        # 상대 접근 속도 (내적)
        closing_speed = -(
            rel_px * rel_vx +
            rel_py * rel_vy
        ) / dist_val

        # 서로 가까워지고 있는 경우에만 TTC 계산
        if closing_speed > 0.01:
            return distance / closing_speed

        # 서로 멀어지거나 정지 상태
        return float('inf')

    def _tracks_callback(self, msg: String) -> None:
        """3D Track 수신 → TTC 계산 → 위험 수준 판단"""

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON Decode Error: {e}')
            return

        stamp = payload.get('header', {}).get('stamp')
        tracks = payload.get('tracks', [])

        class_presence = payload.get('class_presence', {})
        state = class_presence.get('state', 'NONE')

        min_ttc = float('inf')
        most_dangerous_track_id = -1
        most_dangerous_subject = "NONE"

        # 로봇은 현재 위치를 (0, 0)으로 사용 (base_link 기준)
        robot_pos = (0.0, 0.0)
        robot_vel = (self.robot_vx, self.robot_vy)

        persons = [
            t for t in tracks
            if t.get('class_name') == 'person'
        ]

        forklifts = [
            t for t in tracks
            if t.get('class_name') == 'forklift'
        ]

        # ================================================================
        # BOTH (사람 + 지게차가 모두 존재)
        # ================================================================
        if state == 'BOTH':

            # 1. 지게차 - 로봇
            for f in forklifts:

                f_pos = (f['position'][0], f['position'][1])
                f_vel = (f['velocity'][0], f['velocity'][1])

                radius_sum = self.robot_radius + self.forklift_radius

                ttc = self._calculate_ttc(
                    robot_pos,
                    robot_vel,
                    f_pos,
                    f_vel,
                    radius_sum
                )

                if ttc < min_ttc:
                    min_ttc = ttc
                    most_dangerous_track_id = f.get(
                        'track_id',
                        -1
                    )
                    most_dangerous_subject = "로봇-지게차"

            # 2. 사람 - 로봇
            for p in persons:

                p_pos = (p['position'][0], p['position'][1])
                p_vel = (p['velocity'][0], p['velocity'][1])

                radius_sum = self.robot_radius + self.person_radius

                ttc = self._calculate_ttc(
                    robot_pos,
                    robot_vel,
                    p_pos,
                    p_vel,
                    radius_sum
                )

                if ttc < min_ttc:
                    min_ttc = ttc
                    most_dangerous_track_id = p.get(
                        'track_id',
                        -1
                    )
                    most_dangerous_subject = "로봇-사람"

            # 3. 지게차 - 사람
            for f in forklifts:

                f_pos = (f['position'][0], f['position'][1])
                f_vel = (f['velocity'][0], f['velocity'][1])

                for p in persons:

                    p_pos = (p['position'][0], p['position'][1])
                    p_vel = (p['velocity'][0], p['velocity'][1])

                    radius_sum = self.forklift_radius + self.person_radius

                    ttc = self._calculate_ttc(
                        f_pos,
                        f_vel,
                        p_pos,
                        p_vel,
                        radius_sum
                    )

                    if ttc < min_ttc:
                        min_ttc = ttc
                        most_dangerous_track_id = f.get(
                            'track_id',
                            p.get('track_id', -1)
                        )
                        most_dangerous_subject = "지게차-사람"

        # ================================================================
        # *_ONLY
        # ================================================================
        elif state.endswith('_ONLY'):

            target_class = state.replace('_ONLY', '').lower()

            target_tracks = [
                t for t in tracks
                if t.get('class_name') == target_class
            ]

            for trk in target_tracks:

                trk_pos = (trk['position'][0], trk['position'][1])
                trk_vel = (trk['velocity'][0], trk['velocity'][1])

                # 객체 종류에 따라 반경 및 주체 선택
                if target_class == 'person':
                    target_radius = self.person_radius
                    subject_name = "로봇-사람"
                elif target_class == 'forklift':
                    target_radius = self.forklift_radius
                    subject_name = "로봇-지게차"
                else:
                    target_radius = 0.0
                    subject_name = "NONE"

                radius_sum = self.robot_radius + target_radius

                ttc = self._calculate_ttc(
                    robot_pos,
                    robot_vel,
                    trk_pos,
                    trk_vel,
                    radius_sum
                )

                if ttc < min_ttc:
                    min_ttc = ttc
                    most_dangerous_track_id = trk.get(
                        'track_id',
                        -1
                    )
                    most_dangerous_subject = subject_name

        # ================================================================
        # Risk Level
        # ================================================================
        if min_ttc <= self.emergency_ttc:
            overall_risk_level = "EMERGENCY"

        elif min_ttc <= self.warning_ttc:
            overall_risk_level = "WARNING"

        else:
            overall_risk_level = "NORMAL"

        # Alert
        alert_payload = {
            'header': {
                'stamp': stamp
            },
            'risk_level': overall_risk_level,
            'min_ttc': (
                round(min_ttc, 2)
                if min_ttc != float('inf')
                else -1.0
            ),
            'target_track_id': most_dangerous_track_id,
            'target_subject': most_dangerous_subject if min_ttc != float('inf') else 'NONE'
        }

        self.latest_alert_payload = alert_payload
        self.last_track_update_time = self.get_clock().now().nanoseconds * 1e-9

        json_msg = String()
        json_msg.data = json.dumps(
            alert_payload,
            ensure_ascii=False
        )

        self.pub_ttc_alerts.publish(json_msg)

        # Logging
        if overall_risk_level != "NORMAL":
            self.get_logger().warn(
                f'[{overall_risk_level}] [{most_dangerous_subject}] '
                f'Min TTC: {min_ttc:.2f}s '
                f'(Track ID: {most_dangerous_track_id})'
            )


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