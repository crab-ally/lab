#!/usr/bin/env python3
"""MoveIt 2 기반 UR5e + Robotiq 2F-85 3D Vision Pick & Place Controller."""

import time
import math
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Bool
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, PositionConstraint,
    OrientationConstraint, BoundingVolume, PlanningOptions,
    JointConstraint
)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from moveit_msgs.msg import RobotState
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import GripperCommand


class Ur5e2f85MoveItPickAndPlace(Node):
    def __init__(self):
        super().__init__("ur5e_2f85_moveit_pnp_node")
        self.cb_group = ReentrantCallbackGroup()
        self.state = "IDLE"
        self.target_pose = None
        self.target_yaw = 0.0
        self.is_busy = False
        self.shutdown_requested = False
        self.grasped = False

        # UR5e home/ready pose & joint names
        self.home_qpos = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708]
        self.arm_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # Place target location
        self.place_x, self.place_y = 0.8, 0.0
        self.table_top_z = 0.74

        # Motion offsets
        self.pre_grasp_z_offset = 0.10
        self.lift_z_offset = 0.15
        self.pre_place_z_offset = 0.10
        self.post_place_z_offset = 0.10
        self.pre_place_xy_step = 0.05

        # Side grasp 파라미터: 물체 옆에서 수평 접근할 때 사용할 접근 거리
        self.side_grasp_approach_offset = 0.15
        self.side_grasp_insertion_offset = 0.02

        # Lift / fallback parameters
        self.lift_tilt_tolerance = math.radians(25.0)
        self.fallback_planning_attempts = 15
        self.fallback_planning_time = 6.0
        self.fallback_velocity_scale = 0.10
        self.fallback_acceleration_scale = 0.10
        self.fallback_orientation_tolerance = 0.08

        # Step 6 diagnostics
        self.step6_state_wait = 0.3
        self.joint_state_timeout = 2.0
        self.step6_joint_error_warning = 0.05
        self.step6_joint_error_critical = 0.15
        self.latest_joint_state = None
        self.latest_joint_state_time = None

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10,
            callback_group=self.cb_group
        )

        # Robotiq 2F-85 Gripper parameters (0.0 = Open, 0.8 rad = Closed)
        self.gripper_open_position = 0.0
        self.gripper_close_position = 0.8
        self.gripper_open_effort = 20.0
        self.gripper_close_effort = 30.0

        # Gripper geometry
        self.gripper_clearance = 0.100
        self.grasp_margin = 0.010
        self.max_grasp_depth = self.gripper_clearance - self.grasp_margin
        self.tcp_to_fingertip = 0.0

        # Object validation
        self.cartesian_fraction_threshold = 0.95

        # MoveIt action clients & services
        self.move_group_client = ActionClient(
            self, MoveGroup, "/move_action", callback_group=self.cb_group
        )
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory",
            callback_group=self.cb_group
        )
        self.gripper_client = ActionClient(
            self, GripperCommand, "/robotiq_gripper_controller/gripper_action",
            callback_group=self.cb_group
        )
        self.cartesian_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path",
            callback_group=self.cb_group
        )
        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik",
            callback_group=self.cb_group
        )

        # gripper grasped topic subscription
        self.grasped_sub = self.create_subscription(
            Bool,
            "/robotiq_gripper/grasped",
            self.grasped_callback,
            10
        )

        # Vision topic subscriptions
        self.target_sub = self.create_subscription(
            PoseStamped, "/target_object_pose",
            self.target_pose_callback, 10, callback_group=self.cb_group
        )

        self.get_logger().info(
            "[PnP INIT] UR5e + Robotiq 2F-85 MoveIt 9-Step Pick & Place Controller Ready."
        )
        self.get_logger().info(
            f"[PnP INIT] Table Z={self.table_top_z:.3f}, "
            f"Place XY=({self.place_x:.3f},{self.place_y:.3f})"
        )

        self.worker = threading.Thread(
            target=self.pnp_worker_loop, daemon=True
        )
        self.worker.start()

    # Joint state callback
    def joint_state_callback(self, msg):
        self.latest_joint_state = msg
        self.latest_joint_state_time = time.monotonic()

    def get_current_arm_joint_state(self):
        if self.latest_joint_state is None:
            return None
        states = dict(zip(
            self.latest_joint_state.name,
            self.latest_joint_state.position
        ))
        if any(j not in states for j in self.arm_joints):
            return None
        return np.array(
            [float(states[j]) for j in self.arm_joints],
            dtype=np.float64
        )

    def grasped_callback(self, msg):
        self.grasped = bool(msg.data)

        self.get_logger().info(
            f"[GRASP CONTACT] grasped={self.grasped}"
        )

    def unwrap_trajectory_joint_positions(self, trajectory, label="[Trajectory]"):
        jt = trajectory.joint_trajectory
        names = list(jt.joint_names)
        points = jt.points

        if not points:
            return trajectory

        current = self.get_current_arm_joint_state()

        if current is None:
            self.get_logger().warn(
                f"{label} /joint_states unavailable. "
                f"Using trajectory first point as unwrap reference."
            )
            reference = np.array(
                points[0].positions,
                dtype=np.float64
            )
        else:
            current_map = dict(zip(self.arm_joints, current))
            reference = np.array(
                [
                    float(
                        current_map.get(
                            name,
                            points[0].positions[i]
                        )
                    )
                    for i, name in enumerate(names)
                ],
                dtype=np.float64
            )

        two_pi = 2.0 * math.pi
        changed = False

        # 첫 번째 point를 현재 실제 관절각에 가장 가까운 branch로 맞춤
        first = np.array(
            points[0].positions,
            dtype=np.float64
        )

        for j, name in enumerate(names):
            if name not in self.arm_joints:
                continue

            delta = first[j] - reference[j]
            k = round(delta / two_pi)
            corrected = first[j] - k * two_pi

            if abs(corrected - first[j]) > 1e-6:
                self.get_logger().info(
                    f"{label} unwrap first: "
                    f"{name}: {first[j]:.6f} -> {corrected:.6f}"
                )
                first[j] = corrected
                changed = True

        previous = first.copy()

        # 이후 모든 point를 바로 이전 point와 가장 가까운 branch로 맞춤
        for point_index in range(len(points)):
            point = points[point_index]

            if point_index == 0:
                corrected_positions = first
            else:
                positions = np.array(
                    point.positions,
                    dtype=np.float64
                )
                corrected_positions = positions.copy()

                for j, name in enumerate(names):
                    if name not in self.arm_joints:
                        continue

                    raw = positions[j]

                    k = round(
                        (raw - previous[j]) / two_pi
                    )

                    candidate = raw - k * two_pi

                    if abs(candidate - raw) > 1e-6:
                        changed = True

                    corrected_positions[j] = candidate

            point.positions = [
                float(v) for v in corrected_positions
            ]

            previous = corrected_positions.copy()

        if changed:
            self.get_logger().info(f"{label} Joint-angle 2π unwrap applied.")
            self.get_logger().info(
                f"{label} corrected first_position="
                f"{[round(float(v), 6) for v in points[0].positions]}"
            )
            self.get_logger().info(
                f"{label} corrected last_position="
                f"{[round(float(v), 6) for v in points[-1].positions]}"
            )
        else:
            self.get_logger().info(f"{label} No joint-angle 2π wrapping detected.")

        return trajectory

    # Trajectory diagnostics
    def log_trajectory_execution_diagnostics(self, trajectory, label="[Trajectory]"):
        jt = trajectory.joint_trajectory
        names, points = list(jt.joint_names), list(jt.points)
        self.get_logger().info(f"{label} ===== Execution Diagnostics =====")
        self.get_logger().info(f"{label} joint_names={names}")
        self.get_logger().info(f"{label} point_count={len(points)}")

        if not points:
            self.get_logger().error(f"{label} ERROR: trajectory has 0 points.")
            return

        def tsec(t):
            return t.sec + t.nanosec * 1e-9

        start_sec, end_sec = tsec(points[0].time_from_start), tsec(points[-1].time_from_start)
        self.get_logger().info(
            f"{label} first_time={start_sec:.6f}s, "
            f"last_time={end_sec:.6f}s, duration={end_sec - start_sec:.6f}s"
        )
        self.get_logger().info(
            f"{label} first_position="
            f"{[round(float(v), 6) for v in points[0].positions]}"
        )
        self.get_logger().info(
            f"{label} last_position="
            f"{[round(float(v), 6) for v in points[-1].positions]}"
        )

        current = self.get_current_arm_joint_state()
        if current is None:
            self.get_logger().warn(
                f"{label} Current /joint_states unavailable or incomplete."
            )
            self.get_logger().info(f"{label} ==================================")
            return

        start = np.array(points[0].positions, dtype=np.float64)
        trajectory_arm = []
        for joint in self.arm_joints:
            if joint not in names:
                self.get_logger().warn(f"{label} Missing trajectory joint: {joint}")
                return
            idx = names.index(joint)
            if idx >= len(start):
                self.get_logger().warn(f"{label} Invalid trajectory index: {joint}")
                return
            trajectory_arm.append(start[idx])

        error = np.array(trajectory_arm) - current
        abs_error = np.abs(error)
        max_error = float(np.max(abs_error))
        max_index = int(np.argmax(abs_error))

        self.get_logger().info(
            f"{label} current_joint_position="
            f"{[round(float(v), 6) for v in current]}"
        )
        self.get_logger().info(
            f"{label} start_state_error="
            f"{[round(float(v), 6) for v in error]}"
        )
        self.get_logger().info(
            f"{label} max_start_state_error={max_error:.6f} rad "
            f"({math.degrees(max_error):.3f} deg)"
        )
        self.get_logger().info(
            f"{label} worst_joint={self.arm_joints[max_index]}"
        )

        if max_error >= self.step6_joint_error_critical:
            self.get_logger().error(
                f"{label} CRITICAL: trajectory start state differs "
                f"from current by {max_error:.4f} rad."
            )
        elif max_error >= self.step6_joint_error_warning:
            self.get_logger().warn(
                f"{label} WARNING: trajectory start state differs "
                f"from current by {max_error:.4f} rad."
            )
        else:
            self.get_logger().info(
                f"{label} Start-state synchronization looks OK."
            )

        self.get_logger().info(f"{label} ==================================")

    def execute_trajectory_with_diagnostics(
        self, trajectory, label="[ExecuteTrajectory]", timeout_sec=30.0
    ):
        if not self.execute_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(
                f"{label} ExecuteTrajectory server unavailable."
            )
            return False

        self.log_trajectory_execution_diagnostics(trajectory, label)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        self.get_logger().info(
            f"{label} Sending trajectory to ExecuteTrajectory..."
        )

        send_time = time.monotonic()
        handle = self.wait_future(
            self.execute_client.send_goal_async(goal),
            10.0, f"{label} goal"
        )

        if handle is None:
            self.get_logger().error(f"{label} Goal response timeout.")
            return False
        if not handle.accepted:
            self.get_logger().error(f"{label} Goal REJECTED.")
            return False

        self.get_logger().info(f"{label} Goal ACCEPTED.")
        result = self.wait_future(
            handle.get_result_async(), timeout_sec, f"{label} result"
        )
        elapsed = time.monotonic() - send_time

        if result is None:
            self.get_logger().error(f"{label} Result timeout.")
            return False

        code = result.result.error_code.val
        self.get_logger().info(
            f"{label} Result received after {elapsed:.3f}s"
        )
        self.get_logger().info(
            f"{label} MoveIt error_code={code}"
        )

        if code == 1:
            self.get_logger().info(f"{label} SUCCESS")
            return True

        errors = {
            -4: "CONTROL_FAILED",
            -1: "PLANNING_FAILED",
            -2: "INVALID_MOTION_PLAN",
            -3: "MOTION_PLAN_INVALIDATED",
            -5: "INVALID_ROBOT_STATE",
            -6: "INVALID_LINK_NAME",
            -7: "INVALID_GROUP_NAME",
            -10: "TIMED_OUT",
        }
        self.get_logger().error(
            f"{label} FAILURE: {errors.get(code, 'UNKNOWN')} ({code})"
        )

        current = self.get_current_arm_joint_state()
        if current is not None:
            self.get_logger().error(
                f"{label} Current joint state AFTER failure="
                f"{[round(float(v), 6) for v in current]}"
            )
        self.get_logger().error(f"{label} ==================================")
        return False

    # Topic subscribers
    def target_pose_callback(self, msg):
        if self.is_busy or self.state != "IDLE":
            return
        if msg.header.frame_id not in ("base", "world"):
            self.get_logger().warn(
                f"[PnP] Invalid target frame: {msg.header.frame_id}"
            )
            return

        p, q = msg.pose.position, msg.pose.orientation
        x, y, z = float(p.x), float(p.y), float(p.z)
        if not all(np.isfinite(v) for v in (x, y, z)):
            self.get_logger().warn("[PnP] Invalid target position.")
            return

        r11 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        r21 = 2.0 * (q.x * q.y + q.w * q.z)
        self.target_pose = np.array([x, y, z], dtype=np.float64)
        self.target_yaw = math.atan2(r21, r11)
        self.is_busy = True
        self.state = "TRIGGER_PICK"

        self.get_logger().info(
            f"[PnP] Target received: xyz=({x:.3f},{y:.3f},{z:.3f}), "
            f"yaw={math.degrees(self.target_yaw):.1f} deg"
        )

    # Future helper
    def wait_future(self, future, timeout, description):
        event, result = threading.Event(), [None]

        def done_callback(f):
            result[0] = f
            event.set()

        future.add_done_callback(done_callback)
        if not event.wait(timeout):
            self.get_logger().error(
                f"[PnP] Timeout waiting for {description}"
            )
            return None
        try:
            return result[0].result()
        except Exception as e:
            self.get_logger().error(
                f"[PnP] {description} failed: {e}"
            )
            return None

    # Grasp orientation (Top-Down: 위에서 집기, 현재 미사용)
    def yaw_to_grasp_quaternion(self, yaw):
        """yaw 각도에 맞춰 위에서 아래로 집는 쿼터니언 반환 (qx 축 기준)."""
        while yaw > math.pi / 2:
            yaw -= math.pi
        while yaw < -math.pi / 2:
            yaw += math.pi
        return math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0

    # Side Grasp orientation: 옆에서 수평으로 집기
    def yaw_to_side_grasp_quaternion(self, approach_yaw):
        """그리퍼가 approach_yaw 방향으로 수평 접근하도록 쿼터니언을 계산한다.

        q = Rz(approach_yaw) ⊗ Ry(90°) 조합.
        결과: pinch-site의 TCP Z축이 XY 평면상 approach_yaw 방향을 향함.
        즉, 그리퍼가 수평으로 물체 측면에 접근하는 자세가 됨.

        Args:
            approach_yaw: 로봇 베이스 → 물체 방향의 수평 각도 (rad)

        Returns:
            (qx, qy, qz, qw): MoveIt OrientationConstraint에 사용할 쿼터니언
        """
        # Ry(90°) 성분: Z축을 +X 방향으로 회전 (수평 기본 자세)
        s = math.sin(math.pi / 4)   # sin(45°) ≈ 0.7071
        c = math.cos(math.pi / 4)   # cos(45°) ≈ 0.7071
        # Rz(approach_yaw) 성분: XY 평면 내 접근 방향 회전
        a = math.sin(approach_yaw / 2)
        b = math.cos(approach_yaw / 2)
        # 쿼터니언 곱: q = Rz ⊗ Ry(90°) → (qx, qy, qz, qw)
        qx = -a * s
        qy =  b * s
        qz =  a * c
        qw =  b * c
        return qx, qy, qz, qw

    def normalize_angle_near(self,angle,reference):
        two_pi=2.0*math.pi
        return angle-round((angle-reference)/two_pi)*two_pi

    def joint_distance(self,q1,q2):
        q1=np.asarray(q1,dtype=np.float64)
        q2=np.asarray(q2,dtype=np.float64)
        d=q1-q2
        d=np.arctan2(np.sin(d),np.cos(d))
        weights=np.array([2.0,1.0,1.0,1.0,1.0,0.5],dtype=np.float64)
        return float(np.sum(weights*np.abs(d)))

    def make_ik_seed(self,current,offsets=None):
        seed=np.array(current,dtype=np.float64)
        if offsets is not None:
            for i,v in offsets.items():
                seed[i]+=v
        return seed

    def compute_ik_candidate(
        self,x,y,z,qx,qy,qz,qw,seed,timeout_sec=1.0
    ):
        if not self.ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("[IK] /compute_ik service unavailable.")
            return None

        req=GetPositionIK.Request()
        req.ik_request.group_name="ur5e_arm"
        req.ik_request.ik_link_name="pinch-site"
        req.ik_request.pose_stamped.header.frame_id="base"

        req.ik_request.pose_stamped.pose.position.x=float(x)
        req.ik_request.pose_stamped.pose.position.y=float(y)
        req.ik_request.pose_stamped.pose.position.z=float(z)
        req.ik_request.pose_stamped.pose.orientation.x=float(qx)
        req.ik_request.pose_stamped.pose.orientation.y=float(qy)
        req.ik_request.pose_stamped.pose.orientation.z=float(qz)
        req.ik_request.pose_stamped.pose.orientation.w=float(qw)

        req.ik_request.robot_state.joint_state.name=list(self.arm_joints)
        req.ik_request.robot_state.joint_state.position=[
            float(v) for v in seed
        ]

        req.ik_request.avoid_collisions=True

        timeout_ns=int(timeout_sec*1e9)
        req.ik_request.timeout.sec=timeout_ns//1000000000
        req.ik_request.timeout.nanosec=timeout_ns%1000000000

        response=self.wait_future(
            self.ik_client.call_async(req),
            timeout_sec+2.0,
            "IK computation"
        )

        if response is None:
            return None

        if response.error_code.val != 1:
            return None

        result=response.solution.joint_state
        result_map=dict(zip(result.name,result.position))

        if any(j not in result_map for j in self.arm_joints):
            return None

        candidate=np.array(
            [float(result_map[j]) for j in self.arm_joints],
            dtype=np.float64
        )

        return candidate

    def find_nearest_ik_solution(
        self,x,y,z,qx,qy,qz,qw
    ):
        current=self.get_current_arm_joint_state()

        if current is None:
            self.get_logger().error("[IK SELECT] Current joint state unavailable.")
            return None

        self.get_logger().info("[IK SELECT] current="f"{[round(float(v),6) for v in current]}")

        # 여러 IK branch를 탐색하기 위한 seed
        seeds=[
            {},
            {0:math.pi},
            {0:-math.pi},
            {1:math.pi},
            {1:-math.pi},
            {2:math.pi},
            {2:-math.pi},
            {3:math.pi},
            {3:-math.pi},
            {4:math.pi},
            {4:-math.pi},
            {5:math.pi},
            {5:-math.pi},
            {0:math.pi,2:math.pi},
            {0:-math.pi,2:-math.pi},
            {0:math.pi,2:-math.pi},
            {0:-math.pi,2:math.pi},
        ]

        candidates=[]

        for index,offsets in enumerate(seeds):
            seed=self.make_ik_seed(current,offsets)

            candidate=self.compute_ik_candidate(
                x,y,z,qx,qy,qz,qw,seed
            )

            if candidate is None:
                continue

            # 현재 joint state 기준으로 각도 branch 정규화
            for i in range(len(candidate)):
                candidate[i]=self.normalize_angle_near(
                    candidate[i],current[i]
                )

            distance=self.joint_distance(current,candidate)

            duplicate=False
            for old_candidate,_ in candidates:
                if self.joint_distance(old_candidate,candidate)<0.01:
                    duplicate=True
                    break

            if duplicate:
                continue

            candidates.append((candidate,distance))

            self.get_logger().info(
                f"[IK SELECT] candidate {len(candidates)} "
                f"seed={index} "
                f"distance={distance:.4f} "
                f"q={[round(float(v),6) for v in candidate]}"
            )

        if not candidates:
            self.get_logger().error(
                "[IK SELECT] No valid IK solution found."
            )
            return None

        candidates.sort(key=lambda item:item[1])

        best_q,best_distance=candidates[0]

        self.get_logger().info(
            "[IK SELECT] BEST solution: "
            f"distance={best_distance:.4f}, "
            f"q={[round(float(v),6) for v in best_q]}"
        )

        return best_q

    def find_nearest_ik_solution_relaxed(
        self,
        x,y,z,
        qx,qy,qz,qw,
        tilt_deg=30.0
    ):
        current=self.get_current_arm_joint_state()
        if current is None:
            self.get_logger().error("[IK RELAXED] Current joint state unavailable.")
            return None

        self.get_logger().info(
            f"[IK RELAXED] target=({x:.3f},{y:.3f},{z:.3f}), "
            f"orientation tilt ±{tilt_deg:.1f}°"
        )

        base_q=np.array([qx,qy,qz,qw],dtype=np.float64)

        def quat_mul(q1,q2):
            x1,y1,z1,w1=q1
            x2,y2,z2,w2=q2
            return np.array([
                w1*x2+x1*w2+y1*z2-z1*y2,
                w1*y2-x1*z2+y1*w2+z1*x2,
                w1*z2+x1*y2-y1*x2+z1*w2,
                w1*w2-x1*x2-y1*y2-z1*z2
            ],dtype=np.float64)

        def quat_normalize(q):
            n=np.linalg.norm(q)
            if n<1e-12:
                return q
            return q/n

        def axis_angle_quat(axis,angle):
            axis=np.asarray(axis,dtype=np.float64)
            axis=axis/np.linalg.norm(axis)
            s=math.sin(angle/2.0)
            return np.array([
                axis[0]*s,
                axis[1]*s,
                axis[2]*s,
                math.cos(angle/2.0)
            ],dtype=np.float64)

        tilt=math.radians(float(tilt_deg))

        # 현재 side-grasp orientation을 기준으로
        # X/Y 방향으로 ±tilt 후보를 생성한다.
        tilt_angles=[
            0.0,
            -tilt,
            tilt,
            -tilt*0.5,
            tilt*0.5,
        ]

        # quaternion의 local X/Y 축 기준으로 tilt 후보 생성
        candidates=[]

        for angle in tilt_angles:
            for axis in ([1.0,0.0,0.0],[0.0,1.0,0.0]):
                dq=axis_angle_quat(axis,angle)
                q=quat_normalize(quat_mul(base_q,dq))
                candidates.append(q)

        # X+Y 조합도 추가
        for ax in (-tilt,tilt):
            for ay in (-tilt,tilt):
                qx_rot=axis_angle_quat([1.0,0.0,0.0],ax)
                qy_rot=axis_angle_quat([0.0,1.0,0.0],ay)
                q=quat_mul(base_q,qx_rot)
                q=quat_mul(q,qy_rot)
                candidates.append(quat_normalize(q))

        best_q=None
        best_distance=float("inf")
        valid_count=0

        seeds=[
            {},
            {0:math.pi},{0:-math.pi},
            {1:math.pi},{1:-math.pi},
            {2:math.pi},{2:-math.pi},
            {3:math.pi},{3:-math.pi},
            {4:math.pi},{4:-math.pi},
            {5:math.pi},{5:-math.pi},
            {0:math.pi,2:math.pi},
            {0:-math.pi,2:-math.pi},
            {0:math.pi,2:-math.pi},
            {0:-math.pi,2:math.pi},
        ]

        for orientation_index,target_q in enumerate(candidates):
            oqx,oqy,oqz,oqw=[float(v) for v in target_q]

            for seed in seeds:
                req=GetPositionIK.Request()
                req.ik_request.group_name="ur5e_arm"
                req.ik_request.ik_link_name="pinch-site"

                req.ik_request.pose_stamped.header.frame_id="base"
                req.ik_request.pose_stamped.pose.position.x=float(x)
                req.ik_request.pose_stamped.pose.position.y=float(y)
                req.ik_request.pose_stamped.pose.position.z=float(z)
                req.ik_request.pose_stamped.pose.orientation.x=oqx
                req.ik_request.pose_stamped.pose.orientation.y=oqy
                req.ik_request.pose_stamped.pose.orientation.z=oqz
                req.ik_request.pose_stamped.pose.orientation.w=oqw

                req.ik_request.robot_state.joint_state.name=list(self.arm_joints)

                seed_q=list(current)
                for index,value in seed.items():
                    seed_q[index]=float(value)

                req.ik_request.robot_state.joint_state.position=[
                    float(v) for v in seed_q
                ]

                # IK 완화 단계에서는 collision을 제외한다.
                # 최종 경로 collision 검사는 MoveGroup이 수행한다.
                req.ik_request.avoid_collisions=False

                result=self.wait_future(
                    self.ik_client.call_async(req),
                    1.0,
                    "relaxed IK"
                )

                if result is None:
                    continue

                if result.error_code.val!=1:
                    continue

                candidate_q=[]
                for joint in self.arm_joints:
                    try:
                        index=result.solution.joint_state.name.index(joint)
                        candidate_q.append(
                            float(result.solution.joint_state.position[index])
                        )
                    except ValueError:
                        candidate_q=None
                        break

                if candidate_q is None or len(candidate_q)!=6:
                    continue

                candidate_q=[
                    self.normalize_angle_near(v,ref)
                    for v,ref in zip(candidate_q,current)
                ]

                distance=self.joint_distance(current,candidate_q)

                valid_count+=1

                self.get_logger().info(
                    f"[IK RELAXED] valid orientation={orientation_index}, "
                    f"distance={distance:.4f}"
                )

                if distance<best_distance:
                    best_distance=distance
                    best_q=candidate_q

        if best_q is None:
            self.get_logger().error(
                "[IK RELAXED] No valid IK solution found "
                f"within ±{tilt_deg:.1f}°."
            )
            return None

        self.get_logger().info(
            f"[IK RELAXED] selected distance={best_distance:.4f}, "
            f"valid_candidates={valid_count}"
        )

        return best_q

    # Motion Planning
    def plan_and_execute_pose(
        self,x,y,z,qx=1.0,qy=0.0,qz=0.0,qw=0.0,
        num_attempts=10,planning_time=5.0,
        vel_scale=0.1,acc_scale=0.1,
        pos_tol=0.015,ori_tol=0.25
    ):
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[PnP] MoveGroup server unavailable.")
            return False

        # 현재 joint state와 가장 가까운 IK branch 선택
        target_q=self.find_nearest_ik_solution(
            x,y,z,qx,qy,qz,qw
        )

        if target_q is None:
            self.get_logger().error("[PnP] Failed to find nearest IK solution.")
            return False

        current=self.get_current_arm_joint_state()

        if current is not None:
            self.get_logger().info(
                "[IK SELECT] current -> target:"
                f"\n  current={[round(float(v),6) for v in current]}"
                f"\n  target={[round(float(v),6) for v in target_q]}"
                f"\n  distance={self.joint_distance(current,target_q):.4f}"
            )

        req=MotionPlanRequest()
        req.group_name="ur5e_arm"
        req.num_planning_attempts=num_attempts
        req.allowed_planning_time=planning_time
        req.max_velocity_scaling_factor=vel_scale
        req.max_acceleration_scaling_factor=acc_scale
        req.start_state.is_diff=True

        constraints=Constraints()

        for joint,value in zip(self.arm_joints,target_q):
            jc=JointConstraint()
            jc.joint_name=joint
            jc.position=float(value)
            jc.tolerance_above=0.02
            jc.tolerance_below=0.02
            jc.weight=1.0
            constraints.joint_constraints.append(jc)

        req.goal_constraints.append(constraints)

        options=PlanningOptions()
        options.plan_only=False
        options.look_around=False
        options.replan=True
        options.replan_attempts=5

        goal=MoveGroup.Goal()
        goal.request=req
        goal.planning_options=options

        self.get_logger().info(
            "[PnP] MoveGroup joint target="
            f"{[round(float(v),6) for v in target_q]}"
        )

        handle=self.wait_future(
            self.move_group_client.send_goal_async(goal),
            10.0,
            "MoveGroup goal"
        )

        if handle is None or not handle.accepted:
            self.get_logger().error("[PnP] MoveGroup goal rejected.")
            return False

        result=self.wait_future(
            handle.get_result_async(),
            30.0,
            "MoveGroup result"
        )

        if result is None:
            return False

        code=result.result.error_code.val

        if code!=1:
            self.get_logger().error(f"[PnP] MoveGroup failed: error_code={code}")
            return False

        self.get_logger().info("[PnP] MoveGroup joint-target planning SUCCESS.")
        return True

    def plan_and_execute_step6(
        self,
        x,y,z,
        qx,qy,qz,qw,
        tilt_deg=30.0,
        num_attempts=15,
        planning_time=6.0,
        vel_scale=0.10,
        acc_scale=0.10
    ):
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[Step 6] MoveGroup server unavailable.")
            return False

        target_q=self.find_nearest_ik_solution_relaxed(
            x,y,z,
            qx,qy,qz,qw,
            tilt_deg=tilt_deg
        )

        if target_q is None:
            self.get_logger().error("[Step 6] No valid relaxed IK solution found.")
            return False

        current=self.get_current_arm_joint_state()

        if current is not None:
            self.get_logger().info(
                "[Step 6] IK current -> target:"
                f"\n  current={[round(float(v),6) for v in current]}"
                f"\n  target={[round(float(v),6) for v in target_q]}"
                f"\n  distance={self.joint_distance(current,target_q):.4f}"
            )

        req=MotionPlanRequest()
        req.group_name="ur5e_arm"
        req.num_planning_attempts=num_attempts
        req.allowed_planning_time=planning_time
        req.max_velocity_scaling_factor=vel_scale
        req.max_acceleration_scaling_factor=acc_scale
        req.start_state.is_diff=True

        goal_constraints=Constraints()

        for joint,value in zip(self.arm_joints,target_q):
            jc=JointConstraint()
            jc.joint_name=joint
            jc.position=float(value)
            jc.tolerance_above=0.05
            jc.tolerance_below=0.05
            jc.weight=1.0
            goal_constraints.joint_constraints.append(jc)

        req.goal_constraints.append(goal_constraints)

        # Step 6 경로 전체에서 side-grasp 기준 ±30° tilt 허용
        oc=OrientationConstraint()
        oc.header.frame_id="base"
        oc.link_name="pinch-site"
        oc.orientation.x=float(qx)
        oc.orientation.y=float(qy)
        oc.orientation.z=float(qz)
        oc.orientation.w=float(qw)

        tolerance=math.radians(float(tilt_deg))

        oc.absolute_x_axis_tolerance=tolerance
        oc.absolute_y_axis_tolerance=tolerance
        oc.absolute_z_axis_tolerance=math.pi
        oc.weight=1.0

        req.path_constraints=Constraints()
        req.path_constraints.orientation_constraints.append(oc)

        options=PlanningOptions()
        options.plan_only=False
        options.look_around=False
        options.replan=True
        options.replan_attempts=5

        goal=MoveGroup.Goal()
        goal.request=req
        goal.planning_options=options

        self.get_logger().info(
            f"[Step 6] MoveGroup planning: "
            f"relaxed IK + side-grasp orientation path tolerance ±{tilt_deg:.1f}°"
        )

        handle=self.wait_future(
            self.move_group_client.send_goal_async(goal),
            10.0,
            "Step 6 MoveGroup goal"
        )

        if handle is None or not handle.accepted:
            self.get_logger().error("[Step 6] MoveGroup goal rejected.")
            return False

        result=self.wait_future(
            handle.get_result_async(),
            30.0,
            "Step 6 MoveGroup result"
        )

        if result is None:
            return False

        code=result.result.error_code.val

        if code!=1:
            self.get_logger().error(f"[Step 6] MoveGroup failed: error_code={code}")
            return False

        self.get_logger().info("[Step 6] Pre-place planning SUCCESS.")
        return True

    def scale_trajectory_time(self, trajectory, scale=2.5):
        jt = trajectory.joint_trajectory

        for point in jt.points:
            total_ns = (
                point.time_from_start.sec * 1_000_000_000
                + point.time_from_start.nanosec
            )

            total_ns = int(total_ns * scale)

            point.time_from_start.sec = total_ns // 1_000_000_000
            point.time_from_start.nanosec = total_ns % 1_000_000_000

            if point.velocities:
                point.velocities = [
                    float(v / scale)
                    for v in point.velocities
                ]

            if point.accelerations:
                point.accelerations = [
                    float(a / (scale * scale))
                    for a in point.accelerations
                ]

        return trajectory

    # Cartesian Z
    def cartesian_z_move(
        self, x, y, start_z, end_z,
        qx, qy, qz, qw, label="[Cartesian Z]"
    ):
        if abs(end_z - start_z) < 0.001:
            return True

        if not self.cartesian_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                f"{label} /compute_cartesian_path unavailable."
            )
            return False

        req = GetCartesianPath.Request()
        req.header.frame_id = "base"
        req.group_name = "ur5e_arm"
        req.link_name = "pinch-site"
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state.is_diff = True

        waypoint = Pose()
        waypoint.position.x, waypoint.position.y, waypoint.position.z = x, y, end_z
        waypoint.orientation.x, waypoint.orientation.y = qx, qy
        waypoint.orientation.z, waypoint.orientation.w = qz, qw
        req.waypoints = [waypoint]

        response = self.wait_future(
            self.cartesian_client.call_async(req),
            20.0, "Cartesian path computation"
        )

        if response is None:
            return False

        self.get_logger().info(
            f"{label} z {start_z:.3f} -> {end_z:.3f}, "
            f"fraction={response.fraction:.3f}"
        )

        if response.fraction < self.cartesian_fraction_threshold:
            self.get_logger().warn(
                f"{label} Fraction too low: "
                f"{response.fraction:.3f} < "
                f"{self.cartesian_fraction_threshold:.2f}"
            )
            return False

        trajectory = response.solution

        trajectory = self.unwrap_trajectory_joint_positions(
            trajectory,
            label
        )

        trajectory = self.scale_trajectory_time(
            trajectory,
            scale=2.5
        )

        return self.execute_trajectory_with_diagnostics(
            trajectory,
            label,
            30.0
        )

    # Cartesian XYZ
    def cartesian_xyz_move(
        self, start_x, start_y, start_z,
        end_x, end_y, end_z,
        qx, qy, qz, qw
    ):
        dx, dy, dz = end_x - start_x, end_y - start_y, end_z - start_z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance < 0.001:
            return True

        if not self.cartesian_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                "[Cartesian XYZ] /compute_cartesian_path unavailable."
            )
            return False

        step = max(0.005, float(self.pre_place_xy_step))
        segments = int(math.ceil(distance / step))
        waypoints = []

        for i in range(1, segments + 1):
            r = i / segments
            p = Pose()
            p.position.x = start_x + dx * r
            p.position.y = start_y + dy * r
            p.position.z = start_z + dz * r
            p.orientation.x, p.orientation.y = qx, qy
            p.orientation.z, p.orientation.w = qz, qw
            waypoints.append(p)

        self.get_logger().info(
            f"[Cartesian XYZ] distance={distance:.4f} m, "
            f"step={step:.4f} m, waypoints={len(waypoints)}"
        )

        req = GetCartesianPath.Request()
        req.header.frame_id = "base"
        req.group_name = "ur5e_arm"
        req.link_name = "pinch-site"
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state.is_diff = True
        req.waypoints = waypoints

        response = self.wait_future(
            self.cartesian_client.call_async(req),
            30.0, "Cartesian XYZ path computation"
        )

        if response is None:
            return False

        self.get_logger().info(
            f"[Cartesian XYZ] fraction={response.fraction:.3f}"
        )

        if response.fraction < self.cartesian_fraction_threshold:
            self.get_logger().warn(
                f"[Cartesian XYZ] Path fraction too low: "
                f"{response.fraction:.3f}"
            )
            return False

        self.get_logger().info(
            "[Cartesian XYZ] Path computation SUCCESS. Executing trajectory..."
        )

        trajectory=response.solution
        trajectory=self.unwrap_trajectory_joint_positions(
            trajectory,"[Step 6 Cartesian XYZ]"
        )
        trajectory=self.scale_trajectory_time(
            trajectory,
            scale=2.5
        )

        return self.execute_trajectory_with_diagnostics(
            trajectory,"[Step 6 Cartesian XYZ]",30.0
        )

    # Step 6
    def move_to_pre_place_position(
        self,start_x,start_y,start_z,
        target_x,target_y,target_z,
        qx,qy,qz,qw
    ):
        self.get_logger().info(
            f"[Step 6/9] Pre-place 이동 "
            f"(orientation ±30° 허용): "
            f"({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
        )

        time.sleep(self.step6_state_wait)

        current=self.get_current_arm_joint_state()

        if current is None:
            self.get_logger().warn("[Step 6/9] 현재 /joint_states를 가져오지 못했습니다.")
        else:
            self.get_logger().info(
                f"[Step 6/9] current arm joint state={[round(float(v),6) for v in current]}"
            )

        return self.plan_and_execute_step6(
            target_x,target_y,target_z,
            qx,qy,qz,qw,
            tilt_deg=30.0,
            num_attempts=self.fallback_planning_attempts,
            planning_time=self.fallback_planning_time,
            vel_scale=self.fallback_velocity_scale,
            acc_scale=self.fallback_acceleration_scale
        )

    # Z fallback
    def lift_joint_space_fallback(
        self, x, y, target_z, qx, qy, qz, qw
    ):
        self.get_logger().warn(f"[Z FALLBACK] Pose fallback: z={target_z:.3f}")
        return self.plan_and_execute_pose(
            x, y, target_z, qx, qy, qz, qw,
            num_attempts=self.fallback_planning_attempts,
            planning_time=self.fallback_planning_time,
            vel_scale=self.fallback_velocity_scale,
            acc_scale=self.fallback_acceleration_scale,
            pos_tol=0.015,
            ori_tol=self.fallback_orientation_tolerance
        )

    # Step 5 fallback
    def lift_position_downward_fallback(
        self, x, y, target_z, qx, qy, qz, qw
    ):
        self.get_logger().warn(
            f"[LIFT FALLBACK] Position + downward orientation: "
            f"({x:.3f},{y:.3f},{target_z:.3f})"
        )

        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(
                "[LIFT FALLBACK] MoveGroup server unavailable."
            )
            return False

        req = MotionPlanRequest()
        req.group_name = "ur5e_arm"
        req.num_planning_attempts = self.fallback_planning_attempts
        req.allowed_planning_time = self.fallback_planning_time
        req.max_velocity_scaling_factor = self.fallback_velocity_scale
        req.max_acceleration_scaling_factor = self.fallback_acceleration_scale
        req.start_state.is_diff = True

        pc = PositionConstraint()
        pc.header.frame_id = "base"
        pc.link_name = "pinch-site"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.015]

        pc.constraint_region = BoundingVolume()
        pc.constraint_region.primitives.append(primitive)

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, target_z
        pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = "base"
        oc.link_name = "pinch-site"
        oc.orientation.x, oc.orientation.y = qx, qy
        oc.orientation.z, oc.orientation.w = qz, qw
        oc.absolute_x_axis_tolerance = self.lift_tilt_tolerance
        oc.absolute_y_axis_tolerance = self.lift_tilt_tolerance
        oc.absolute_z_axis_tolerance = math.pi
        oc.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)
        req.goal_constraints.append(constraints)

        options = PlanningOptions()
        options.plan_only = False
        options.look_around = False
        options.replan = True
        options.replan_attempts = 5

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = options

        handle = self.wait_future(
            self.move_group_client.send_goal_async(goal),
            10.0, "Lift fallback MoveGroup goal"
        )

        if handle is None or not handle.accepted:
            self.get_logger().error(
                "[LIFT FALLBACK] MoveGroup goal rejected."
            )
            return False

        result = self.wait_future(
            handle.get_result_async(),
            30.0, "Lift fallback MoveGroup result"
        )

        if result is None:
            return False

        ok = result.result.error_code.val == 1
        self.get_logger().info(
            f"[LIFT FALLBACK] {'SUCCESS' if ok else 'FAILED'}"
        )
        return ok

    # Named state (e.g. ready / home)
    def plan_and_execute_named_state(
        self, named_state="ready", num_attempts=5, planning_time=3.0
    ):
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[PnP] MoveGroup server unavailable.")
            return False

        self.get_logger().info(
            f"[PnP] Returning to state: {named_state}"
        )

        req = MotionPlanRequest()
        req.group_name = "ur5e_arm"
        req.num_planning_attempts = num_attempts
        req.allowed_planning_time = planning_time
        req.max_velocity_scaling_factor = 0.1
        req.max_acceleration_scaling_factor = 0.1
        req.start_state.is_diff = True

        constraints = Constraints()
        for joint, value in zip(self.arm_joints, self.home_qpos):
            jc = JointConstraint()
            jc.joint_name = joint
            jc.position = value
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        req.goal_constraints.append(constraints)

        options = PlanningOptions()
        options.plan_only = False
        options.replan = True
        options.replan_attempts = 3

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = options

        handle = self.wait_future(
            self.move_group_client.send_goal_async(goal),
            10.0, "Named-state MoveGroup goal"
        )

        if handle is None or not handle.accepted:
            self.get_logger().error("[PnP] Named-state goal rejected.")
            return False

        result = self.wait_future(
            handle.get_result_async(),
            20.0, "Named-state MoveGroup result"
        )

        if result is None:
            return False

        ok = result.result.error_code.val == 1
        self.get_logger().info(
            f"[PnP] {'Successfully returned to' if ok else 'Failed to return to'} "
            f"{named_state}{'' if ok else f': error_code={result.result.error_code.val}'}"
        )
        return ok

    # Unified pose move
    def move_to_pose(
        self, target_x, target_y, target_z,
        qx, qy, qz, qw,
        start_x=None, start_y=None, start_z=None,
        description="Target"
    ):
        self.get_logger().info(
            f"[MOVE] {description}: "
            f"({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
        )

        if start_x is not None and start_y is not None and start_z is not None:
            ok = self.cartesian_xyz_move(
                start_x, start_y, start_z,
                target_x, target_y, target_z,
                qx, qy, qz, qw
            )
            if ok:
                return True
            self.get_logger().warn(f"[MOVE] {description} Cartesian 실패. Fallback 시도.")
        else:
            ok = self.plan_and_execute_pose(
                target_x, target_y, target_z,
                qx, qy, qz, qw
            )
            if ok:
                return True
            self.get_logger().warn(f"[MOVE] {description} Pose planning 실패. Fallback 시도.")

        return self.plan_and_execute_pose(
            target_x, target_y, target_z,
            qx, qy, qz, qw,
            num_attempts=self.fallback_planning_attempts,
            planning_time=self.fallback_planning_time,
            vel_scale=self.fallback_velocity_scale,
            acc_scale=self.fallback_acceleration_scale,
            pos_tol=0.02,
            ori_tol=self.fallback_orientation_tolerance
        )

    # Gripper action control
    def control_gripper(self, action):
        action = action.upper()
        if action not in ("OPEN", "CLOSE"):
            self.get_logger().error(f"[GRIPPER] Invalid action: {action}")
            return False, None

        if not self.gripper_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(
                "[GRIPPER] Action server unavailable."
            )
            return False, None

        goal = GripperCommand.Goal()
        if action == "OPEN":
            goal.command.position = self.gripper_open_position
            goal.command.max_effort = self.gripper_open_effort
        else:
            goal.command.position = self.gripper_close_position
            goal.command.max_effort = self.gripper_close_effort

        self.get_logger().info(
            f"[GRIPPER] Sending {action}: "
            f"position={goal.command.position:.3f}"
        )

        try:
            handle = self.wait_future(
                self.gripper_client.send_goal_async(goal),
                5.0, f"Gripper {action} goal"
            )

            if handle is None or not handle.accepted:
                self.get_logger().error(
                    f"[GRIPPER] {action} goal rejected."
                )
                return False, None

            res = self.wait_future(
                handle.get_result_async(),
                10.0, f"Gripper {action} result"
            )

            if res is None:
                return False, None

            obj = res.result
            self.get_logger().info(
                f"[GRIPPER] {action}: position={obj.position:.4f}, "
                f"reached={obj.reached_goal}, stalled={obj.stalled}"
            )

            time.sleep(0.2)
            return obj.reached_goal, obj

        except Exception as e:
            self.get_logger().error(
                f"[GRIPPER] {action} exception: {e}"
            )
            return False, None

    def check_grasp_success(self, result_obj):
        if result_obj is None:
            return False

        pos = float(result_obj.position)

        if self.grasped:
            self.get_logger().info(
                "[GRASP CHECK] 물체 파지 성공: "
                f"finger={pos:.4f}rad, contact=True"
            )
            return True

        self.get_logger().warn(
            "[GRASP CHECK] 물체 파지 실패: "
            f"finger={pos:.4f}rad, contact=False"
        )
        return False

    def calculate_place_pose(self):
        center_z = self.table_top_z + pre_place_z_offset
        return (
            self.place_x,
            self.place_y,
            center_z - self.tcp_to_fingertip
        )

    # Failure recovery
    def reset_after_failure(self, reason, need_open_gripper=False):
        self.get_logger().error(f"[PnP FAIL] {reason}")

        if need_open_gripper:
            self.get_logger().info(
                "[PnP RECOVERY] Opening gripper..."
            )
            self.control_gripper("OPEN")
            time.sleep(0.3)

        self.get_logger().info(
            "[PnP RECOVERY] Returning to READY..."
        )

        try:
            if not self.plan_and_execute_named_state("ready"):
                self.get_logger().warn(
                    "[PnP RECOVERY] Ready fallback attempt..."
                )
                self.plan_and_execute_named_state(
                    "ready", num_attempts=10, planning_time=5.0
                )
        except Exception as e:
            self.get_logger().error(
                f"[PnP RECOVERY] Exception: {e}"
            )

        self.target_pose = None
        self.target_yaw = 0.0
        self.is_busy = False
        self.state = "IDLE"

    # 9-Step PnP Worker Loop
    def pnp_worker_loop(self):
        while rclpy.ok() and not self.shutdown_requested:
            try:
                if self.state == "TRIGGER_PICK" and self.target_pose is not None:
                    self.state = "PICKING"
                    tx, ty, tz = self.target_pose

                    # 로봇 베이스 → 물체 방향의 수평 접근각 계산 (atan2)
                    approach_yaw = math.atan2(ty, tx)
                    # 옆에서 집기용 쿼터니언: TCP Z축이 approach_yaw 방향으로 수평을 향함
                    qx, qy, qz, qw = self.yaw_to_side_grasp_quaternion(approach_yaw)

                    # 접근 방향 벡터 = approach_yaw 방향 × offset 거리
                    approach_dx = math.cos(approach_yaw) * self.side_grasp_approach_offset
                    approach_dy = math.sin(approach_yaw) * self.side_grasp_approach_offset
                    # 파지 목표: 물체 중심 위치에서 수평 파지
                    grasp_x = tx + math.cos(approach_yaw) * self.side_grasp_insertion_offset
                    grasp_y = ty + math.sin(approach_yaw) * self.side_grasp_insertion_offset
                    grasp_z = tz

                    self.get_logger().info("=" * 60)
                    self.get_logger().info("[PnP] Starting UR5e + 2F-85 9-Step Side Grasp Pick & Place")
                    self.get_logger().info(
                        f"[PnP] Target=({tx:.3f},{ty:.3f},{tz:.3f}), "
                        f"approach_yaw={math.degrees(approach_yaw):.1f}°"
                    )
                    self.get_logger().info("=" * 60)

                    # 1. Open
                    self.get_logger().info("[Step 1/9] Gripper OPEN")
                    ok, _ = self.control_gripper("OPEN")
                    if not ok:
                        self.get_logger().warn("[Step 1/9] OPEN retry")
                        time.sleep(0.3)
                        ok, _ = self.control_gripper("OPEN")
                        if not ok:
                            self.reset_after_failure("Step 1 Gripper OPEN failed.")
                            continue

                    # 2. Pre-grasp: 물체와 같은 높이의 옆 위치로 이동
                    pre_grasp_x = grasp_x - approach_dx
                    pre_grasp_y = grasp_y - approach_dy
                    pre_grasp_z = grasp_z  # 수평 접근이므로 Z 고정
                    if not self.move_to_pose(
                        pre_grasp_x, pre_grasp_y, pre_grasp_z,
                        qx, qy, qz, qw,
                        description="[Step 2/9] Pre-grasp (side)"
                    ):
                        self.reset_after_failure("Step 2 Pre-grasp failed.")
                        continue

                    # 3. 수평 접근: pre-grasp → 물체 중심으로 Cartesian XY 이동
                    self.get_logger().info(
                        f"[Step 3/9] 수평 접근: "
                        f"({pre_grasp_x:.3f},{pre_grasp_y:.3f}) -> "
                        f"({grasp_x:.3f},{grasp_y:.3f}), z={grasp_z:.3f}"
                    )

                    ok = self.cartesian_xyz_move(
                        pre_grasp_x, pre_grasp_y, pre_grasp_z,
                        grasp_x, grasp_y, grasp_z,
                        qx, qy, qz, qw
                    )

                    if not ok:
                        self.get_logger().warn("[Step 3/9] Cartesian 수평접근 실패. Pose fallback")
                        # fallback: joint-space로 파지 위치 직접 이동
                        ok = self.lift_joint_space_fallback(
                            grasp_x, grasp_y, grasp_z, qx, qy, qz, qw
                        )

                    if not ok:
                        self.reset_after_failure("Step 3 Horizontal approach failed.")
                        continue

                    # 4. Close / grasp check
                    self.get_logger().info("[Step 4/9] Gripper CLOSE & grasp check")
                    cmd_ok, result_obj = self.control_gripper("CLOSE")
                    grasped = self.check_grasp_success(result_obj) if cmd_ok else False

                    if not grasped:
                        self.get_logger().warn("[Step 4/9] Grasp retry")
                        self.control_gripper("OPEN")
                        time.sleep(0.3)
                        cmd_ok, result_obj = self.control_gripper("CLOSE")
                        grasped = self.check_grasp_success(result_obj) if cmd_ok else False

                        if not grasped:
                            self.reset_after_failure(
                                "Step 4 Gripper CLOSE/grasp check failed.",
                                need_open_gripper=True
                            )
                            continue

                    # 5. Lift: 파지 후 수직으로 들어올림 (side grasp 자세 유지)
                    after_grasp_z = grasp_z + self.lift_z_offset
                    self.get_logger().info(
                        f"[Step 5/9] Lift (side grasp): {grasp_z:.3f} -> {after_grasp_z:.3f}"
                    )

                    ok = self.cartesian_z_move(
                        grasp_x, grasp_y, grasp_z, after_grasp_z,
                        qx, qy, qz, qw,
                        "[Step 5 Cartesian Lift]"
                    )

                    if not ok:
                        self.get_logger().warn("[Step 5/9] Cartesian Lift 실패. Fallback")
                        ok = self.lift_position_downward_fallback(
                            grasp_x, grasp_y, after_grasp_z,
                            qx, qy, qz, qw
                        )

                    if not ok:
                        self.reset_after_failure(
                            "Step 5 After-grasp lift failed.",
                            need_open_gripper=True
                        )
                        continue

                    self.get_logger().info("[Step 5/9] After-grasp Lift SUCCESS")

                    # 6. Pre-place: 파지된 물체를 place 위치 상단으로 이동
                    px, py, pz = self.calculate_place_pose()
                    pre_place_z = pz + self.post_place_z_offset

                    ok = self.move_to_pre_place_position(
                        grasp_x, grasp_y, after_grasp_z,   # grasp_x/y = tx/ty
                        px, py, pre_place_z,
                        qx, qy, qz, qw
                    )

                    if not ok:
                        self.reset_after_failure(
                            "Step 6 Pre-place failed.",
                            need_open_gripper=True
                        )
                        continue

                    self.get_logger().info("[Step 6/9] Pre-place 완료")

                    # 7. Place descent / open
                    self.get_logger().info(
                        f"[Step 7/9] Place 하강: {pre_place_z:.3f} -> {pz:.3f}"
                    )

                    ok = self.cartesian_z_move(
                        px, py, pre_place_z, pz,
                        qx, qy, qz, qw,
                        "[Step 7 Cartesian Z]"
                    )

                    if not ok:
                        self.get_logger().warn("[Step 7/9] Cartesian 실패. Pose fallback")
                        ok = self.lift_joint_space_fallback(
                            px, py, pz, qx, qy, qz, qw
                        )

                    if not ok:
                        self.reset_after_failure(
                            "Step 7 Place descent failed.",
                            need_open_gripper=True
                        )
                        continue

                    self.get_logger().info("[Step 7/9] Place 도착 -> Gripper OPEN")

                    ok, _ = self.control_gripper("OPEN")
                    if not ok:
                        self.get_logger().warn("[Step 7/9] OPEN retry")
                        time.sleep(0.3)
                        ok, _ = self.control_gripper("OPEN")
                        if not ok:
                            self.reset_after_failure("Step 7 Gripper OPEN failed.")
                            continue

                    time.sleep(0.5)

                    # 8. Retract: Place 위치에서 수평으로 후퇴
                    retract_x = px - approach_dx
                    retract_y = py - approach_dy
                    retract_z = pz

                    self.get_logger().info(
                        f"[Step 8/9] Retract (side): "
                        f"({px:.3f},{py:.3f},{pz:.3f}) -> "
                        f"({retract_x:.3f},{retract_y:.3f},{retract_z:.3f})"
                    )

                    ok = self.cartesian_xyz_move(
                        px, py, pz,
                        retract_x, retract_y, retract_z,
                        qx, qy, qz, qw
                    )

                    if not ok:
                        self.get_logger().warn(
                            "[Step 8/9] Side retract 실패. Pose fallback"
                        )
                        ok = self.lift_joint_space_fallback(
                            retract_x, retract_y, retract_z,
                            qx, qy, qz, qw
                        )

                    if not ok:
                        self.reset_after_failure("Step 8 Retract failed.")
                        continue

                    # 9. Ready
                    self.get_logger().info("[Step 9/9] Ready 복귀")
                    ok = self.plan_and_execute_named_state("ready")

                    if not ok:
                        self.get_logger().warn("[Step 9/9] Ready fallback")
                        ok = self.plan_and_execute_named_state(
                            "ready", num_attempts=10, planning_time=5.0
                        )

                    if not ok:
                        self.reset_after_failure("Step 9 Return to ready failed.")
                        continue

                    self.get_logger().info("=" * 60)
                    self.get_logger().info("[PnP SUCCESS] 9-Step Pick & Place Completed!")
                    self.get_logger().info("=" * 60)

                    self.target_pose = None
                    self.target_yaw = 0.0
                    self.is_busy = False
                    self.state = "IDLE"

                time.sleep(0.05)

            except Exception as e:
                self.get_logger().error(f"[PnP WORKER] Exception: {e}")
                self.reset_after_failure(f"Worker exception: {e}")

    def shutdown(self):
        self.shutdown_requested = True
        if self.worker.is_alive() and threading.current_thread() is not self.worker:
            self.worker.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = Ur5e2f85MoveItPickAndPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()