#!/usr/bin/env python3
"""
Point-Foot Biped Robot Command Publisher Node
"""

import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class PointFootControllerNode(Node):
    def __init__(self):
        super().__init__('point_foot_controller')

        # ------------------------------------------------------------------
        # 1. Publisher / Subscriber 설정
        # ------------------------------------------------------------------
        # 브릿지 노드로 관절 목표 각도 전송
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/point_foot/commands',
            10
        )

        # 브릿지 노드로부터 현재 관절 상태 수신
        self.state_sub = self.create_subscription(
            JointState,
            '/point_foot/joint_states',
            self.joint_state_callback,
            10
        )

        # ------------------------------------------------------------------
        # 2. 관절 목표값 및 제어 파라미터
        # ------------------------------------------------------------------
        # 관절 순서: [abad_L, hip_L, knee_L, abad_R, hip_R, knee_R]
        # 기본 직립 목표 각도 (rad)
        self.default_qpos = [0.0, 0.2, -0.4, 0.0, -0.2, 0.4]
        self.current_cmd = list(self.default_qpos)

        self.latest_joint_state = None
        self.start_time = time.time()

        # 100Hz 주기 타이머 (0.01초)
        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info("Point-Foot Controller 노드가 시작되었습니다.")

    def joint_state_callback(self, msg: JointState):
        """브릿지에서 피드백받는 현재 관절 상태 정보"""
        self.latest_joint_state = msg

    def control_loop(self):
        """100Hz 루프: 명령 생성 및 토픽 발행"""
        t = time.time() - self.start_time

        # ------------------------------------------------------------------
        # 제어 모드 선택 (필요에 따라 주석 해제하여 테스트)
        # ------------------------------------------------------------------

        # [모드 1] 기본 직립 유지 (Standing Hold)
        target_qpos = list(self.default_qpos)

        # [모드 2] Sine 파형 관절 궤적 테스트 (Hip & Knee 모션)
        # freq = 1.0  # 1 Hz
        # amp = 0.15  # 0.15 rad
        # target_qpos = list(self.default_qpos)
        # target_qpos[1] += math.sin(2 * math.pi * freq * t) * amp       # hip_L
        # target_qpos[2] -= math.sin(2 * math.pi * freq * t) * amp       # knee_L
        # target_qpos[4] -= math.sin(2 * math.pi * freq * t) * amp       # hip_R
        # target_qpos[5] += math.sin(2 * math.pi * freq * t) * amp       # knee_R

        # [모드 3] 간단한 제자리 걸음 패턴 (Alternate Leg Lift)
        # freq = 1.5
        # phase = math.sin(2 * math.pi * freq * t)
        # target_qpos = list(self.default_qpos)
        # if phase > 0:
        #     # 왼발 들기
        #     target_qpos[1] += phase * 0.2
        #     target_qpos[2] -= phase * 0.3
        # else:
        #     # 오른발 들기
        #     target_qpos[4] += phase * 0.2
        #     target_qpos[5] -= phase * 0.3

        # ------------------------------------------------------------------
        # 토픽 발행
        # ------------------------------------------------------------------
        msg = Float64MultiArray()
        msg.data = target_qpos
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PointFootControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("사용자에 의해 노드가 종료되었습니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()