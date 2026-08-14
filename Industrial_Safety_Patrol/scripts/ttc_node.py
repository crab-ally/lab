#!/usr/bin/env python3
"""
TTC (Time-To-Collision) 충돌 위험 예측 노드

Subscribe:
    /detected_objects/person_poses
    /detected_objects/forklift_poses
    /odom

Publish:
    /ttc/alert      (std_msgs/String  — JSON: [{pair, distance_m, rel_vel_mps, ttc_sec, level}])
    /ttc/markers    (MarkerArray      — Rviz 거리선 + TTC 텍스트)
    /ttc/emergency  (std_msgs/Bool    — Robot↔Forklift DANGER 시 True)
    /cmd_vel        (geometry_msgs/Twist — Robot↔Forklift DANGER 시 zero velocity)

위험 시나리오별 동작:
    Forklift ↔ Person   DANGER →  경고 로그 + /ttc/alert 발행  (작업자 회피 유도)
    Robot    ↔ Forklift DANGER →  로봇 긴급 정지(/cmd_vel zero) + /ttc/emergency True
"""

import json
import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

# ── 위험 등급 임계값 ──────────────────────────────────────────────────
TTC_SAFE_SEC: float = 3.0
TTC_CAUTION_SEC: float = 1.5

# 최소 상대 속도 (m/s) — 이 이하이면 정지 물체로 판단, TTC=∞(SAFE)
MIN_REL_VEL: float = 0.05

# 속도 추정 이동 평균 윈도우 (프레임 수)
SMOOTH_WINDOW: int = 5


# ─────────────────────────────────────────────────────────────────────
class ObjectTracker:
    """
    프레임 간 위치 차분과 이동 평균(window=SMOOTH_WINDOW)으로 속도를 추정하는 추적기.
    PoseArray 는 객체 ID 정보가 없으므로 최근접 매칭(nearest-neighbor)으로 연결한다.
    """

    def __init__(self, smooth_window: int = SMOOTH_WINDOW) -> None:
        self._prev_positions: list[np.ndarray] = []
        self._vel_histories: list[deque] = []
        self._smooth_window = smooth_window

    def update(
        self, positions: list[np.ndarray], dt: float
    ) -> list[tuple[np.ndarray, float]]:
        """
        매 프레임 호출. (position, smoothed_speed) 리스트를 반환한다.
        dt: 직전 호출과의 경과 시간(초). 0 이하이면 속도 0 으로 처리.
        """
        results: list[tuple[np.ndarray, float]] = []
        new_prev: list[np.ndarray] = []
        new_hists: list[deque] = []

        for curr in positions:
            if self._prev_positions and dt > 0.0:
                dists = [
                    np.linalg.norm(curr[:2] - prev[:2])
                    for prev in self._prev_positions
                ]
                bi = int(np.argmin(dists))
                inst_v = (
                    np.linalg.norm(curr[:2] - self._prev_positions[bi][:2]) / dt
                )
                hist: deque = (
                    self._vel_histories[bi]
                    if bi < len(self._vel_histories)
                    else deque(maxlen=self._smooth_window)
                )
                hist.append(inst_v)
                speed = float(np.mean(hist))
            else:
                hist = deque(maxlen=self._smooth_window)
                speed = 0.0

            new_prev.append(curr.copy())
            new_hists.append(hist)
            results.append((curr, speed))

        self._prev_positions = new_prev
        self._vel_histories = new_hists
        return results

    def reset(self) -> None:
        self._prev_positions = []
        self._vel_histories = []


# ─────────────────────────────────────────────────────────────────────
class TTCNode(Node):
    """TTC 충돌 위험 예측 ROS2 노드."""

    def __init__(self) -> None:
        super().__init__('ttc_node')

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Subscriptions ─────────────────────────────────────────────
        self.create_subscription(
            PoseArray,
            '/detected_objects/person_poses',
            self._person_poses_cb,
            sensor_qos,
        )
        self.create_subscription(
            PoseArray,
            '/detected_objects/forklift_poses',
            self._forklift_poses_cb,
            sensor_qos,
        )
        self.create_subscription(
            Odometry,
            '/odom',
            self._odom_cb,
            sensor_qos,
        )

        # ── Publishers ────────────────────────────────────────────────
        self._alert_pub     = self.create_publisher(String,      '/ttc/alert',     10)
        self._marker_pub    = self.create_publisher(MarkerArray, '/ttc/markers',   10)
        self._emergency_pub = self.create_publisher(Bool,        '/ttc/emergency', 10)
        self._cmd_vel_pub   = self.create_publisher(Twist,       '/cmd_vel',       10)

        # ── State ─────────────────────────────────────────────────────
        self._person_results:   list[tuple[np.ndarray, float]] = []
        self._forklift_results: list[tuple[np.ndarray, float]] = []
        self._robot_pos:   np.ndarray = np.zeros(3)
        self._robot_speed: float = 0.0

        self._person_tracker   = ObjectTracker()
        self._forklift_tracker = ObjectTracker()

        self._last_person_t:   float | None = None
        self._last_forklift_t: float | None = None

        self._marker_id = 0

        self.get_logger().info(
            'TTC Node ready. '
            f'SAFE>{TTC_SAFE_SEC}s | CAUTION>{TTC_CAUTION_SEC}s | DANGER<{TTC_CAUTION_SEC}s'
        )

    # ── Callbacks ─────────────────────────────────────────────────────

    def _person_poses_cb(self, msg: PoseArray) -> None:
        now = self._now_sec()
        dt = (now - self._last_person_t) if self._last_person_t is not None else 0.0
        self._last_person_t = now

        positions = self._extract_positions(msg)
        self._person_results = self._person_tracker.update(positions, dt)
        self._evaluate()

    def _forklift_poses_cb(self, msg: PoseArray) -> None:
        now = self._now_sec()
        dt = (now - self._last_forklift_t) if self._last_forklift_t is not None else 0.0
        self._last_forklift_t = now

        positions = self._extract_positions(msg)
        self._forklift_results = self._forklift_tracker.update(positions, dt)
        self._evaluate()

    def _odom_cb(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        self._robot_pos = np.array([pos.x, pos.y, pos.z])
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._robot_speed = math.sqrt(vx * vx + vy * vy)

    # ── Core ──────────────────────────────────────────────────────────

    def _evaluate(self) -> None:
        """매 프레임: 모든 쌍에 대해 TTC 계산 후 알림/마커/긴급정지 발행."""
        stamp = self.get_clock().now().to_msg()
        alerts: list[dict] = []
        markers = MarkerArray()
        emergency = False

        # 이전 마커 전부 삭제
        clr = Marker()
        clr.action = Marker.DELETEALL
        clr.header.frame_id = 'odom'
        clr.header.stamp = stamp
        markers.markers.append(clr)
        self._marker_id = 0

        # ① Forklift ↔ Person
        for fl_pos, fl_speed in self._forklift_results:
            for p_pos, p_speed in self._person_results:
                dist = float(np.linalg.norm(fl_pos[:2] - p_pos[:2]))
                rel_vel = fl_speed + p_speed  # 최악 케이스(서로 접근)
                ttc, level = self._compute_ttc(dist, rel_vel)

                alerts.append({
                    'pair': 'Forklift↔Person',
                    'distance_m': round(dist, 2),
                    'rel_vel_mps': round(rel_vel, 3),
                    'ttc_sec': round(ttc, 2) if ttc < 999.0 else None,
                    'level': level,
                })

                if level == 'DANGER':
                    self.get_logger().warn(
                        f'[DANGER] Forklift↔Person  TTC={ttc:.2f}s  dist={dist:.2f}m'
                    )
                elif level == 'CAUTION':
                    self.get_logger().info(
                        f'[CAUTION] Forklift↔Person  TTC={ttc:.2f}s  dist={dist:.2f}m',
                        throttle_duration_sec=1.0,
                    )

                self._add_pair_markers(
                    markers, stamp, fl_pos, p_pos, dist, ttc, level, 'FL↔P'
                )

        # ② Robot ↔ Forklift
        for fl_pos, fl_speed in self._forklift_results:
            dist = float(np.linalg.norm(self._robot_pos[:2] - fl_pos[:2]))
            rel_vel = fl_speed + self._robot_speed  # 로봇 속도는 odom twist에서 취득
            ttc, level = self._compute_ttc(dist, rel_vel)

            alerts.append({
                'pair': 'Robot↔Forklift',
                'distance_m': round(dist, 2),
                'rel_vel_mps': round(rel_vel, 3),
                'ttc_sec': round(ttc, 2) if ttc < 999.0 else None,
                'level': level,
            })

            if level == 'DANGER':
                self.get_logger().warn(
                    f'[DANGER] Robot↔Forklift  TTC={ttc:.2f}s  dist={dist:.2f}m'
                    '  → Emergency Stop!'
                )
                emergency = True
            elif level == 'CAUTION':
                self.get_logger().info(
                    f'[CAUTION] Robot↔Forklift  TTC={ttc:.2f}s  dist={dist:.2f}m',
                    throttle_duration_sec=1.0,
                )

            self._add_pair_markers(
                markers, stamp, self._robot_pos, fl_pos, dist, ttc, level, 'Rb↔FL'
            )

        # ── Publish ───────────────────────────────────────────────────
        if alerts:
            alert_msg = String()
            alert_msg.data = json.dumps(alerts, ensure_ascii=False)
            self._alert_pub.publish(alert_msg)

        self._marker_pub.publish(markers)

        em_msg = Bool()
        em_msg.data = emergency
        self._emergency_pub.publish(em_msg)

        if emergency:
            # Robot↔Forklift DANGER: 순찰 로봇만 즉시 정지
            self._cmd_vel_pub.publish(Twist())

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_positions(msg: PoseArray) -> list[np.ndarray]:
        return [
            np.array([p.position.x, p.position.y, p.position.z])
            for p in msg.poses
        ]

    @staticmethod
    def _compute_ttc(distance: float, rel_vel: float) -> tuple[float, str]:
        """
        TTC = distance / rel_vel.
        rel_vel < MIN_REL_VEL 이면 정지 물체로 보고 TTC=∞(SAFE) 반환.
        """
        if rel_vel < MIN_REL_VEL:
            return 999.0, 'SAFE'
        ttc = distance / rel_vel
        if ttc > TTC_SAFE_SEC:
            level = 'SAFE'
        elif ttc >= TTC_CAUTION_SEC:
            level = 'CAUTION'
        else:
            level = 'DANGER'
        return ttc, level

    def _add_pair_markers(
        self,
        markers: MarkerArray,
        stamp,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        dist: float,
        ttc: float,
        level: str,
        label: str,
    ) -> None:
        """Rviz용 거리 라인(LINE_STRIP) + TTC 텍스트(TEXT_VIEW_FACING) 마커 추가."""
        color_map: dict[str, ColorRGBA] = {
            'SAFE':    ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.7),
            'CAUTION': ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.9),
            'DANGER':  ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
        }
        color = color_map.get(level, color_map['SAFE'])
        line_w = 0.08 if level == 'DANGER' else 0.04

        pa = Point(x=float(pos_a[0]), y=float(pos_a[1]), z=float(pos_a[2]) + 0.3)
        pb = Point(x=float(pos_b[0]), y=float(pos_b[1]), z=float(pos_b[2]) + 0.3)
        mid = (pos_a + pos_b) / 2.0

        # ① 거리 라인
        self._marker_id += 1
        line = Marker()
        line.header.frame_id = 'odom'
        line.header.stamp = stamp
        line.ns = 'ttc_lines'
        line.id = self._marker_id
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = line_w
        line.color = color
        line.lifetime.sec = 1
        line.points = [pa, pb]
        markers.markers.append(line)

        # ② TTC 텍스트
        self._marker_id += 1
        ttc_str = f'{ttc:.1f}s' if ttc < 999.0 else '∞'
        text = Marker()
        text.header.frame_id = 'odom'
        text.header.stamp = stamp
        text.ns = 'ttc_text'
        text.id = self._marker_id
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(mid[0])
        text.pose.position.y = float(mid[1])
        text.pose.position.z = float(mid[2]) + 1.4
        text.pose.orientation.w = 1.0
        text.scale.z = 0.35
        text.color = color
        text.text = f'{label} [{level}]\nTTC:{ttc_str}  d:{dist:.1f}m'
        text.lifetime.sec = 1
        markers.markers.append(text)

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


# ─────────────────────────────────────────────────────────────────────
def main(args=None) -> None:
    rclpy.init(args=args)
    node = TTCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('TTC Node stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
