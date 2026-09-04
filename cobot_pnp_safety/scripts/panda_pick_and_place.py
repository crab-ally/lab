#!/usr/bin/env python3
"""
MoveIt 2 기반 Franka Emika Panda 3D Vision Pick & Place Controller Node

Pick & Place Sequence (9 Steps):
  1. 그리퍼 open
  2. pre grasp 위치 계산 및 이동
  3. grasp 위치로 이동 (z축 하강)
  4. 그리퍼 닫기 & 물체 잡았는지 여부 체크
  5. after grasp 위치로 이동 (z축 상승)
     - 한 번의 Cartesian Z 이동
     - segment 분할하지 않는다.
  6. pre place 위치로 이동
     - 이동 중 grasp orientation 유지
     - 별도의 orientation 재정렬을 하지 않는다.
  7. place 위치로 이동(z축 하강) & 그리퍼 오픈
  8. after place 위치로 이동(z축 상승)
  9. ready 자세로 복귀

Step 5 Lift 조건:
  - TCP 위치는 목표 x/y/z를 맞춘다.
  - 그리퍼는 아래쪽을 향하면 된다.
  - 정확한 yaw는 강제하지 않는다.
  - Cartesian Lift는 한 번에 계산/실행한다.
  - Lift Cartesian 실패 시 Position + Downward Orientation fallback을 사용한다.

Step 6 Pre-place 조건:
  - Pre-place 위치까지 이동한다.
  - 이동 전체에서 grasp orientation을 유지한다.
  - orientation을 별도로 회전시키지 않는다.
  - Cartesian trajectory 실행 실패 시 실제 joint state와
    trajectory 시작 상태를 비교하여 원인을 진단한다.

Step 6 실행 진단:
  - /joint_states 구독
  - trajectory joint names 출력
  - trajectory point 수 출력
  - trajectory 시작/종료 시간 출력
  - trajectory 첫 joint position 출력
  - 현재 실제 joint position 출력
  - trajectory 시작점과 현재 joint position의 최대 오차 출력
  - ExecuteTrajectory error code 해석 출력
"""

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
from std_msgs.msg import Float32

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    PlanningOptions,
    JointConstraint,
)

from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import GripperCommand


class PandaMoveItPickAndPlace(Node):

    def __init__(self):
        super().__init__("panda_moveit_pnp_node")

        self.cb_group = ReentrantCallbackGroup()

        self.state = "IDLE"
        self.target_pose = None
        self.object_height = None
        self.target_yaw = 0.0
        self.is_busy = False
        self.shutdown_requested = False

        # ============================================================
        # Panda initial / ready pose
        # ============================================================
        self.home_qpos = [
            0.0,
            -0.785398,
            0.0,
            -2.35619,
            0.0,
            1.57079,
            0.785398,
        ]

        self.arm_joints = [
            f"joint{i}"
            for i in range(1, 8)
        ]

        # ============================================================
        # Place target position
        # ============================================================
        self.place_x = 0.0
        self.place_y = -0.55
        self.table_top_z = 0.44

        # ============================================================
        # Motion offsets (m)
        # ============================================================
        self.pre_grasp_z_offset = 0.10
        self.lift_z_offset = 0.10
        self.post_place_z_offset = 0.10

        # ============================================================
        # Step sizes
        #
        # Step 5에서는 사용하지 않는다.
        # ============================================================
        self.pre_place_xy_step = 0.05

        # ============================================================
        # Step 5 Lift orientation
        # ============================================================
        self.lift_tilt_tolerance = math.radians(25.0)

        # ============================================================
        # Fallback planning parameters
        # ============================================================
        self.fallback_planning_attempts = 15
        self.fallback_planning_time = 6.0
        self.fallback_velocity_scale = 0.20
        self.fallback_acceleration_scale = 0.20

        self.fallback_orientation_tolerance = 0.08

        # ============================================================
        # Step 6 execution diagnostics
        # ============================================================
        self.step6_state_wait = 0.3
        self.joint_state_timeout = 2.0
        self.step6_joint_error_warning = 0.05
        self.step6_joint_error_critical = 0.15

        # ============================================================
        # Joint state diagnostic
        # ============================================================
        self.latest_joint_state = None
        self.latest_joint_state_time = None

        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
            callback_group=self.cb_group,
        )

        # ============================================================
        # Gripper configuration
        # ============================================================
        self.gripper_open_position = 0.04
        self.gripper_close_position = 0.0
        self.gripper_open_effort = 20.0
        self.gripper_close_effort = 30.0

        self.min_grasp_position_threshold = 0.002

        # ============================================================
        # Gripper geometry
        # ============================================================
        self.gripper_clearance = 0.100
        self.grasp_margin = 0.010

        self.max_grasp_depth = (
            self.gripper_clearance -
            self.grasp_margin
        )

        self.tcp_to_fingertip = (
            0.1100 -
            (0.0584 + 0.0445)
        )

        # ============================================================
        # Object validation
        # ============================================================
        self.min_object_height = 0.005
        self.max_object_height = 0.40

        # ============================================================
        # Cartesian threshold
        # ============================================================
        self.cartesian_fraction_threshold = 0.95

        # ============================================================
        # MoveIt action/service clients
        # ============================================================
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
            callback_group=self.cb_group,
        )

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
            callback_group=self.cb_group,
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            "/panda_hand_controller/gripper_action",
            callback_group=self.cb_group,
        )

        self.cartesian_client = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
            callback_group=self.cb_group,
        )

        # ============================================================
        # Target subscribers
        # ============================================================
        self.target_sub = self.create_subscription(
            PoseStamped,
            "/target_object_pose",
            self.target_pose_callback,
            10,
            callback_group=self.cb_group,
        )

        self.height_sub = self.create_subscription(
            Float32,
            "/object_height",
            self.object_height_callback,
            10,
            callback_group=self.cb_group,
        )

        # ============================================================
        # Logs
        # ============================================================
        self.get_logger().info(
            "[PnP INIT] Franka Panda MoveIt2 "
            "9-Step Pick & Place Controller Ready."
        )

        self.get_logger().info(
            f"[PnP INIT] Table Z={self.table_top_z:.3f}, "
            f"Place XY=({self.place_x:.3f},"
            f"{self.place_y:.3f})"
        )

        self.get_logger().info(
            "[PnP INIT] Step 5 = "
            "Single Cartesian Z Lift"
        )

        self.get_logger().info(
            "[PnP INIT] Step 6 = "
            "Pre-place Position + "
            "grasp orientation 유지"
        )

        self.get_logger().info(
            "[PnP INIT] Step 6 diagnostics = "
            "/joint_states + trajectory start-state comparison"
        )

        # ============================================================
        # Worker thread
        # ============================================================
        self.worker = threading.Thread(
            target=self.pnp_worker_loop,
            daemon=True,
        )

        self.worker.start()

    # ============================================================
    # Joint State Callback
    # ============================================================
    def joint_state_callback(self, msg):
        self.latest_joint_state = msg
        self.latest_joint_state_time = time.monotonic()

    # ============================================================
    # Get current arm joint state
    # ============================================================
    def get_current_arm_joint_state(self):
        msg = self.latest_joint_state

        if msg is None:
            return None

        name_to_position = {}

        for name, position in zip(
            msg.name,
            msg.position,
        ):
            name_to_position[name] = float(position)

        values = []

        for joint_name in self.arm_joints:
            if joint_name not in name_to_position:
                return None

            values.append(
                name_to_position[joint_name]
            )

        return np.array(
            values,
            dtype=np.float64,
        )

    # ============================================================
    # Joint State Diagnostic Log
    # ============================================================
    def log_trajectory_execution_diagnostics(
        self,
        trajectory,
        label="[Trajectory]",
    ):
        joint_names = list(
            trajectory.joint_trajectory.joint_names
        )

        points = list(
            trajectory.joint_trajectory.points
        )

        self.get_logger().info(
            f"{label} ===== Execution Diagnostics ====="
        )

        self.get_logger().info(
            f"{label} joint_names={joint_names}"
        )

        self.get_logger().info(
            f"{label} point_count={len(points)}"
        )

        if len(points) == 0:
            self.get_logger().error(
                f"{label} ERROR: trajectory has 0 points."
            )
            return

        first_point = points[0]
        last_point = points[-1]

        start_sec = (
            first_point.time_from_start.sec +
            first_point.time_from_start.nanosec * 1e-9
        )

        end_sec = (
            last_point.time_from_start.sec +
            last_point.time_from_start.nanosec * 1e-9
        )

        duration = end_sec - start_sec

        self.get_logger().info(
            f"{label} first_time={start_sec:.6f}s, "
            f"last_time={end_sec:.6f}s, "
            f"duration={duration:.6f}s"
        )

        self.get_logger().info(
            f"{label} first_position="
            f"{[round(float(v), 6) for v in first_point.positions]}"
        )

        self.get_logger().info(
            f"{label} last_position="
            f"{[round(float(v), 6) for v in last_point.positions]}"
        )

        current = self.get_current_arm_joint_state()

        if current is None:
            self.get_logger().warn(
                f"{label} Current /joint_states unavailable "
                f"or arm joint names incomplete."
            )
            self.get_logger().info(
                f"{label} =================================="
            )
            return

        trajectory_start = np.array(
            first_point.positions,
            dtype=np.float64,
        )

        if len(joint_names) != len(
            self.arm_joints
        ):
            self.get_logger().warn(
                f"{label} Trajectory joint count "
                f"does not equal Panda arm joint count."
            )

        trajectory_arm = []

        for joint_name in self.arm_joints:
            if joint_name not in joint_names:
                self.get_logger().warn(
                    f"{label} Missing trajectory joint: "
                    f"{joint_name}"
                )
                self.get_logger().info(
                    f"{label} =================================="
                )
                return

            index = joint_names.index(joint_name)

            if index >= len(trajectory_start):
                self.get_logger().warn(
                    f"{label} Invalid trajectory index "
                    f"for {joint_name}"
                )
                return

            trajectory_arm.append(
                trajectory_start[index]
            )

        trajectory_arm = np.array(
            trajectory_arm,
            dtype=np.float64,
        )

        error = trajectory_arm - current
        abs_error = np.abs(error)

        max_error = float(
            np.max(abs_error)
        )

        max_index = int(
            np.argmax(abs_error)
        )

        self.get_logger().info(
            f"{label} current_joint_position="
            f"{[round(float(v), 6) for v in current]}"
        )

        self.get_logger().info(
            f"{label} start_state_error="
            f"{[round(float(v), 6) for v in error]}"
        )

        self.get_logger().info(
            f"{label} max_start_state_error="
            f"{max_error:.6f} rad "
            f"({math.degrees(max_error):.3f} deg)"
        )

        self.get_logger().info(
            f"{label} worst_joint="
            f"{self.arm_joints[max_index]}"
        )

        if max_error >= self.step6_joint_error_critical:
            self.get_logger().error(
                f"{label} CRITICAL: trajectory start state "
                f"differs from current joint state by "
                f"{max_error:.4f} rad."
            )
            self.get_logger().error(
                f"{label} This strongly suggests "
                f"MoveIt/controller state synchronization issue."
            )

        elif max_error >= self.step6_joint_error_warning:
            self.get_logger().warn(
                f"{label} WARNING: trajectory start state "
                f"differs from current joint state by "
                f"{max_error:.4f} rad."
            )

        else:
            self.get_logger().info(
                f"{label} Start-state synchronization "
                f"looks OK."
            )

        self.get_logger().info(
            f"{label} =================================="
        )

    # ============================================================
    # Execute trajectory with diagnostics
    # ============================================================
    def execute_trajectory_with_diagnostics(
        self,
        trajectory,
        label="[ExecuteTrajectory]",
        timeout_sec=30.0,
    ):
        if not self.execute_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                f"{label} ExecuteTrajectory "
                f"server unavailable."
            )
            return False

        self.log_trajectory_execution_diagnostics(
            trajectory,
            label,
        )

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        self.get_logger().info(
            f"{label} Sending trajectory "
            f"to ExecuteTrajectory..."
        )

        send_time = time.monotonic()

        future = self.execute_client.send_goal_async(
            goal
        )

        handle = self.wait_future(
            future,
            10.0,
            f"{label} goal",
        )

        if handle is None:
            self.get_logger().error(
                f"{label} Goal response timeout."
            )
            return False

        if not handle.accepted:
            self.get_logger().error(
                f"{label} Goal REJECTED by action server."
            )
            self.get_logger().error(
                f"{label} Failure stage = "
                f"goal acceptance"
            )
            return False

        self.get_logger().info(
            f"{label} Goal ACCEPTED."
        )

        result = self.wait_future(
            handle.get_result_async(),
            timeout_sec,
            f"{label} result",
        )

        elapsed = time.monotonic() - send_time

        if result is None:
            self.get_logger().error(
                f"{label} Result timeout."
            )
            self.get_logger().error(
                f"{label} Failure stage = "
                f"controller/action result timeout"
            )
            return False

        error_code = (
            result.result.error_code.val
        )

        self.get_logger().info(
            f"{label} Result received after "
            f"{elapsed:.3f}s"
        )

        self.get_logger().info(
            f"{label} MoveIt error_code="
            f"{error_code}"
        )

        if error_code == 1:
            self.get_logger().info(
                f"{label} SUCCESS"
            )
            return True

        # ------------------------------------------------------------
        # MoveIt Error Code Diagnostics
        # ------------------------------------------------------------
        if error_code == -4:
            self.get_logger().error(
                f"{label} CONTROL_FAILED (-4)"
            )
            self.get_logger().error(
                f"{label} Planning 자체가 아니라 "
                f"trajectory 실행/controller 단계에서 "
                f"실패한 것이다."
            )

            self.get_logger().error(
                f"{label} 가능한 원인:"
            )

            self.get_logger().error(
                f"{label} 1) controller trajectory rejection"
            )

            self.get_logger().error(
                f"{label} 2) trajectory 시작 joint state 불일치"
            )

            self.get_logger().error(
                f"{label} 3) controller execution timeout"
            )

            self.get_logger().error(
                f"{label} 4) trajectory timing 문제"
            )

            self.get_logger().error(
                f"{label} 5) MuJoCo <-> ROS joint-state "
                f"synchronization 문제"
            )

        elif error_code == -1:
            self.get_logger().error(
                f"{label} FAILURE (-1): "
                f"PLANNING_FAILED"
            )

        elif error_code == -2:
            self.get_logger().error(
                f"{label} FAILURE (-2): "
                f"INVALID_MOTION_PLAN"
            )

        elif error_code == -3:
            self.get_logger().error(
                f"{label} FAILURE (-3): "
                f"MOTION_PLAN_INVALIDATED"
            )

        elif error_code == -5:
            self.get_logger().error(
                f"{label} FAILURE (-5): "
                f"INVALID_ROBOT_STATE"
            )

        elif error_code == -6:
            self.get_logger().error(
                f"{label} FAILURE (-6): "
                f"INVALID_LINK_NAME"
            )

        elif error_code == -7:
            self.get_logger().error(
                f"{label} FAILURE (-7): "
                f"INVALID_GROUP_NAME"
            )

        elif error_code == -10:
            self.get_logger().error(
                f"{label} FAILURE (-10): "
                f"TIMED_OUT"
            )

        else:
            self.get_logger().error(
                f"{label} FAILURE: "
                f"unrecognized MoveIt error code "
                f"{error_code}"
            )

        # ------------------------------------------------------------
        # Failure 후 실제 joint state 다시 출력
        # ------------------------------------------------------------
        current_after = (
            self.get_current_arm_joint_state()
        )

        if current_after is not None:
            self.get_logger().error(
                f"{label} Current joint state AFTER "
                f"failure="
                f"{[round(float(v), 6) for v in current_after]}"
            )

        self.get_logger().error(
            f"{label} =================================="
        )

        return False

    # ============================================================
    # Subscriber Callbacks
    # ============================================================
    def object_height_callback(self, msg):
        if self.is_busy:
            return

        height = float(msg.data)

        if not np.isfinite(height):
            return

        self.object_height = height

    def target_pose_callback(self, msg):
        if self.is_busy or self.state != "IDLE":
            return

        if msg.header.frame_id != "link0":
            self.get_logger().warn(
                f"[PnP] Invalid target frame: "
                f"{msg.header.frame_id}"
            )
            return

        p = msg.pose.position
        q = msg.pose.orientation

        x = float(p.x)
        y = float(p.y)
        z = float(p.z)

        if not all(
            np.isfinite(v)
            for v in (x, y, z)
        ):
            self.get_logger().warn(
                "[PnP] Invalid target position."
            )
            return

        if (
            self.object_height is None
            or not np.isfinite(self.object_height)
        ):
            self.get_logger().warn(
                "[PnP] Object height is not available."
            )
            return

        h = float(self.object_height)

        if not (
            self.min_object_height
            <= h
            <= self.max_object_height
        ):
            self.get_logger().warn(
                f"[PnP] Invalid object height: "
                f"{h:.4f} m"
            )
            return

        r11 = (
            1.0 -
            2.0 * (
                q.y * q.y +
                q.z * q.z
            )
        )

        r21 = (
            2.0 * (
                q.x * q.y +
                q.w * q.z
            )
        )

        yaw = math.atan2(
            r21,
            r11,
        )

        self.target_pose = np.array(
            [x, y, z],
            dtype=np.float64,
        )

        self.target_yaw = yaw
        self.is_busy = True
        self.state = "TRIGGER_PICK"

        self.get_logger().info(
            f"[PnP] Target received: "
            f"xyz=({x:.3f},{y:.3f},{z:.3f}), "
            f"h={h:.3f}, "
            f"yaw={math.degrees(yaw):.1f} deg"
        )

    # ============================================================
    # Future helper
    # ============================================================
    def wait_future(
        self,
        future,
        timeout,
        description,
    ):
        event = threading.Event()
        result = [None]

        def done_callback(f):
            result[0] = f
            event.set()

        future.add_done_callback(done_callback)

        if not event.wait(timeout):
            self.get_logger().error(
                f"[PnP] Timeout waiting for "
                f"{description}"
            )
            return None

        try:
            return result[0].result()

        except Exception as e:
            self.get_logger().error(
                f"[PnP] {description} failed: {e}"
            )
            return None

    # ============================================================
    # Grasp orientation
    # ============================================================
    def yaw_to_grasp_quaternion(self, yaw):
        while yaw > math.pi / 2:
            yaw -= math.pi

        while yaw < -math.pi / 2:
            yaw += math.pi

        cy = math.cos(
            yaw / 2.0
        )

        sy = math.sin(
            yaw / 2.0
        )

        return (
            cy,
            sy,
            0.0,
            0.0,
        )

    # ============================================================
    # MoveIt Pose Planning & Execution
    # ============================================================
    def plan_and_execute_pose(
        self,
        x,
        y,
        z,
        qx=1.0,
        qy=0.0,
        qz=0.0,
        qw=0.0,
        num_attempts=10,
        planning_time=5.0,
        vel_scale=0.25,
        acc_scale=0.25,
        pos_tol=0.015,
        ori_tol=0.25,
    ):
        if not self.move_group_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[PnP] MoveGroup server unavailable."
            )
            return False

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()

        req.group_name = "panda_arm"
        req.num_planning_attempts = num_attempts
        req.allowed_planning_time = planning_time
        req.max_velocity_scaling_factor = vel_scale
        req.max_acceleration_scaling_factor = acc_scale
        req.start_state.is_diff = True

        # ------------------------------------------------------------
        # Position constraint
        # ------------------------------------------------------------
        pc = PositionConstraint()

        pc.header.frame_id = "link0"
        pc.link_name = "hand_tcp"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [pos_tol]

        pc.constraint_region = BoundingVolume()
        pc.constraint_region.primitives.append(
            primitive
        )

        pose = Pose()

        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0

        pc.constraint_region.primitive_poses.append(
            pose
        )

        pc.weight = 1.0

        # ------------------------------------------------------------
        # Orientation constraint
        # ------------------------------------------------------------
        oc = OrientationConstraint()

        oc.header.frame_id = "link0"
        oc.link_name = "hand_tcp"

        oc.orientation.x = qx
        oc.orientation.y = qy
        oc.orientation.z = qz
        oc.orientation.w = qw

        oc.absolute_x_axis_tolerance = ori_tol
        oc.absolute_y_axis_tolerance = ori_tol
        oc.absolute_z_axis_tolerance = (
            ori_tol + 0.10
        )

        oc.weight = 1.0

        constraints = Constraints()

        constraints.position_constraints.append(
            pc
        )

        constraints.orientation_constraints.append(
            oc
        )

        req.goal_constraints.append(
            constraints
        )

        options = PlanningOptions()

        options.plan_only = False
        options.look_around = False
        options.replan = True
        options.replan_attempts = 5

        goal.request = req
        goal.planning_options = options

        future = (
            self.move_group_client.send_goal_async(
                goal
            )
        )

        handle = self.wait_future(
            future,
            10.0,
            "MoveGroup goal",
        )

        if (
            handle is None
            or not handle.accepted
        ):
            self.get_logger().error(
                "[PnP] MoveGroup goal rejected."
            )
            return False

        result = self.wait_future(
            handle.get_result_async(),
            30.0,
            "MoveGroup result",
        )

        if result is None:
            return False

        ok = (
            result.result.error_code.val
            == 1
        )

        if not ok:
            self.get_logger().error(
                f"[PnP] MoveGroup failed: "
                f"error_code="
                f"{result.result.error_code.val}"
            )

        return ok

    # ============================================================
    # Cartesian Z Move
    # ============================================================
    def cartesian_z_move(
        self,
        x,
        y,
        start_z,
        end_z,
        qx,
        qy,
        qz,
        qw,
        label="[Cartesian Z]",
    ):
        if abs(end_z - start_z) < 0.001:
            return True

        if not self.cartesian_client.wait_for_service(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                f"{label} "
                "/compute_cartesian_path "
                "unavailable."
            )
            return False

        req = GetCartesianPath.Request()

        req.header.frame_id = "link0"
        req.group_name = "panda_arm"
        req.link_name = "hand_tcp"
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state.is_diff = True

        waypoint = Pose()

        waypoint.position.x = x
        waypoint.position.y = y
        waypoint.position.z = end_z

        waypoint.orientation.x = qx
        waypoint.orientation.y = qy
        waypoint.orientation.z = qz
        waypoint.orientation.w = qw

        req.waypoints = [waypoint]

        future = self.cartesian_client.call_async(
            req
        )

        response = self.wait_future(
            future,
            20.0,
            "Cartesian path computation",
        )

        if response is None:
            return False

        self.get_logger().info(
            f"{label} "
            f"z {start_z:.3f} -> {end_z:.3f}, "
            f"fraction={response.fraction:.3f}"
        )

        if (
            response.fraction
            < self.cartesian_fraction_threshold
        ):
            self.get_logger().warn(
                f"{label} Fraction too low: "
                f"{response.fraction:.3f} < "
                f"{self.cartesian_fraction_threshold:.2f}"
            )
            return False

        return self.execute_trajectory_with_diagnostics(
            response.solution,
            label=label,
            timeout_sec=30.0,
        )

    # ============================================================
    # Cartesian XYZ Move
    #
    # orientation은 모든 waypoint에서 동일하게 유지한다.
    # ============================================================
    def cartesian_xyz_move(
        self,
        start_x,
        start_y,
        start_z,
        end_x,
        end_y,
        end_z,
        qx,
        qy,
        qz,
        qw,
    ):
        dx = end_x - start_x
        dy = end_y - start_y
        dz = end_z - start_z

        distance = math.sqrt(
            dx * dx +
            dy * dy +
            dz * dz
        )

        if distance < 0.001:
            return True

        if not self.cartesian_client.wait_for_service(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[Cartesian XYZ] "
                "/compute_cartesian_path "
                "unavailable."
            )
            return False

        step = max(
            0.005,
            float(self.pre_place_xy_step),
        )

        segments = int(
            math.ceil(
                distance / step
            )
        )

        waypoints = []

        for i in range(
            1,
            segments + 1,
        ):
            ratio = (
                float(i) /
                float(segments)
            )

            waypoint = Pose()

            waypoint.position.x = (
                start_x +
                dx * ratio
            )

            waypoint.position.y = (
                start_y +
                dy * ratio
            )

            waypoint.position.z = (
                start_z +
                dz * ratio
            )

            # --------------------------------------------------------
            # 모든 waypoint에서 동일한 grasp orientation
            # --------------------------------------------------------
            waypoint.orientation.x = qx
            waypoint.orientation.y = qy
            waypoint.orientation.z = qz
            waypoint.orientation.w = qw

            waypoints.append(
                waypoint
            )

        self.get_logger().info(
            "[Cartesian XYZ] "
            f"distance={distance:.4f} m, "
            f"step={step:.4f} m, "
            f"waypoints={len(waypoints)}"
        )

        req = GetCartesianPath.Request()

        req.header.frame_id = "link0"
        req.group_name = "panda_arm"
        req.link_name = "hand_tcp"
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state.is_diff = True
        req.waypoints = waypoints

        future = self.cartesian_client.call_async(
            req
        )

        response = self.wait_future(
            future,
            30.0,
            "Cartesian XYZ path computation",
        )

        if response is None:
            return False

        self.get_logger().info(
            f"[Cartesian XYZ] "
            f"fraction={response.fraction:.3f}"
        )

        if (
            response.fraction
            < self.cartesian_fraction_threshold
        ):
            self.get_logger().warn(
                f"[Cartesian XYZ] "
                f"Path fraction too low: "
                f"{response.fraction:.3f}"
            )
            return False

        self.get_logger().info(
            "[Cartesian XYZ] "
            "Path computation SUCCESS. "
            "Executing trajectory..."
        )

        return self.execute_trajectory_with_diagnostics(
            response.solution,
            label="[Step 6 Cartesian XYZ]",
            timeout_sec=30.0,
        )

    # ============================================================
    # Step 6:
    # Pre-place Position Move
    #
    # 핵심:
    #   - grasp orientation을 그대로 유지한다.
    #   - Cartesian 모든 waypoint에서 동일한 q를 사용한다.
    #   - fallback에서도 orientation tolerance를 작게 한다.
    #   - 별도의 orientation 정렬을 하지 않는다.
    # ============================================================
    def move_to_pre_place_position(
        self,
        start_x,
        start_y,
        start_z,
        target_x,
        target_y,
        target_z,
        qx,
        qy,
        qz,
        qw,
    ):
        self.get_logger().info(
            f"[Step 6/9] "
            f"Pre-place 위치 이동 "
            f"(grasp orientation 유지): "
            f"({target_x:.3f}, "
            f"{target_y:.3f}, "
            f"{target_z:.3f})"
        )

        # ------------------------------------------------------------
        # Step 5 직후 실제 joint state가 업데이트될 시간을 준다.
        # ------------------------------------------------------------
        self.get_logger().info(
            "[Step 6/9] "
            f"Step 5 완료 후 "
            f"{self.step6_state_wait:.1f}s "
            f"상태 안정화 대기..."
        )

        time.sleep(
            self.step6_state_wait
        )

        # ------------------------------------------------------------
        # 현재 joint state 출력
        # ------------------------------------------------------------
        current = (
            self.get_current_arm_joint_state()
        )

        if current is None:
            self.get_logger().warn(
                "[Step 6/9] "
                "현재 /joint_states를 가져오지 못했습니다."
            )
        else:
            self.get_logger().info(
                "[Step 6/9] "
                "현재 arm joint state="
                f"{[round(float(v), 6) for v in current]}"
            )

        # ------------------------------------------------------------
        # Cartesian XYZ
        # ------------------------------------------------------------
        ok = self.cartesian_xyz_move(
            start_x,
            start_y,
            start_z,
            target_x,
            target_y,
            target_z,
            qx,
            qy,
            qz,
            qw,
        )

        if ok:
            self.get_logger().info(
                "[Step 6/9] "
                "Pre-place 위치 이동 SUCCESS "
                "(grasp orientation 유지)."
            )
            return True

        # ------------------------------------------------------------
        # Cartesian execution 실패
        # ------------------------------------------------------------
        self.get_logger().error(
            "[Step 6/9] "
            "Cartesian trajectory EXECUTION FAILED."
        )

        self.get_logger().error(
            "[Step 6/9] "
            "위의 [Step 6 Cartesian XYZ] "
            "diagnostic log를 확인하십시오."
        )

        self.get_logger().warn(
            "[Step 6/9] "
            "Orientation 유지 Fallback Pose Planning 시도..."
        )

        # ------------------------------------------------------------
        # Fallback
        #
        # orientation tolerance = 0.05 rad
        # ------------------------------------------------------------
        fallback_ok = self.plan_and_execute_pose(
            target_x,
            target_y,
            target_z,
            qx,
            qy,
            qz,
            qw,
            num_attempts=self.fallback_planning_attempts,
            planning_time=self.fallback_planning_time,
            vel_scale=0.10,
            acc_scale=0.10,
            pos_tol=0.015,
            ori_tol=0.05,
        )

        if fallback_ok:
            self.get_logger().info(
                "[Step 6/9] "
                "Pre-place Fallback SUCCESS "
                "(grasp orientation 유지)."
            )
        else:
            self.get_logger().error(
                "[Step 6/9] "
                "Pre-place Fallback FAILED."
            )

        return fallback_ok

    # ============================================================
    # Generic Joint-space Pose Fallback
    # ============================================================
    def lift_joint_space_fallback(
        self,
        x,
        y,
        target_z,
        qx,
        qy,
        qz,
        qw,
    ):
        self.get_logger().warn(
            f"[Z FALLBACK] "
            f"Executing pose fallback to "
            f"z={target_z:.3f}"
        )

        return self.plan_and_execute_pose(
            x,
            y,
            target_z,
            qx,
            qy,
            qz,
            qw,
            num_attempts=self.fallback_planning_attempts,
            planning_time=self.fallback_planning_time,
            vel_scale=self.fallback_velocity_scale,
            acc_scale=self.fallback_acceleration_scale,
            pos_tol=0.015,
            ori_tol=self.fallback_orientation_tolerance,
        )

    # ============================================================
    # Step 5 Lift-specific relaxed fallback
    #
    # 조건:
    #   1. TCP position -> target x/y/z
    #   2. Gripper downward orientation 유지
    #   3. yaw는 자유
    # ============================================================
    def lift_position_downward_fallback(
        self,
        x,
        y,
        target_z,
        qx,
        qy,
        qz,
        qw,
    ):
        self.get_logger().warn(
            f"[LIFT FALLBACK] "
            f"Position + downward orientation "
            f"fallback: "
            f"target=({x:.3f},"
            f"{y:.3f},"
            f"{target_z:.3f})"
        )

        if not self.move_group_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[LIFT FALLBACK] "
                "MoveGroup server unavailable."
            )
            return False

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()

        req.group_name = "panda_arm"
        req.num_planning_attempts = (
            self.fallback_planning_attempts
        )

        req.allowed_planning_time = (
            self.fallback_planning_time
        )

        req.max_velocity_scaling_factor = (
            self.fallback_velocity_scale
        )

        req.max_acceleration_scaling_factor = (
            self.fallback_acceleration_scale
        )

        req.start_state.is_diff = True

        pc = PositionConstraint()

        pc.header.frame_id = "link0"
        pc.link_name = "hand_tcp"

        primitive = SolidPrimitive()

        primitive.type = (
            SolidPrimitive.SPHERE
        )

        primitive.dimensions = [0.015]

        pc.constraint_region = (
            BoundingVolume()
        )

        pc.constraint_region.primitives.append(
            primitive
        )

        pose = Pose()

        pose.position.x = x
        pose.position.y = y
        pose.position.z = target_z
        pose.orientation.w = 1.0

        pc.constraint_region.primitive_poses.append(
            pose
        )

        pc.weight = 1.0

        # ------------------------------------------------------------
        # Downward Orientation Constraint
        # ------------------------------------------------------------
        oc = OrientationConstraint()

        oc.header.frame_id = "link0"
        oc.link_name = "hand_tcp"

        oc.orientation.x = qx
        oc.orientation.y = qy
        oc.orientation.z = qz
        oc.orientation.w = qw

        oc.absolute_x_axis_tolerance = (
            self.lift_tilt_tolerance
        )

        oc.absolute_y_axis_tolerance = (
            self.lift_tilt_tolerance
        )

        oc.absolute_z_axis_tolerance = math.pi

        oc.weight = 1.0

        constraints = Constraints()

        constraints.position_constraints.append(
            pc
        )

        constraints.orientation_constraints.append(
            oc
        )

        req.goal_constraints.append(
            constraints
        )

        options = PlanningOptions()

        options.plan_only = False
        options.look_around = False
        options.replan = True
        options.replan_attempts = 5

        goal.request = req
        goal.planning_options = options

        future = (
            self.move_group_client.send_goal_async(
                goal
            )
        )

        handle = self.wait_future(
            future,
            10.0,
            "Lift fallback MoveGroup goal",
        )

        if (
            handle is None
            or not handle.accepted
        ):
            self.get_logger().error(
                "[LIFT FALLBACK] "
                "MoveGroup goal rejected."
            )
            return False

        result = self.wait_future(
            handle.get_result_async(),
            30.0,
            "Lift fallback MoveGroup result",
        )

        if result is None:
            return False

        ok = (
            result.result.error_code.val
            == 1
        )

        if ok:
            self.get_logger().info(
                "[LIFT FALLBACK] SUCCESS "
                "(position + downward orientation)"
            )
        else:
            self.get_logger().error(
                f"[LIFT FALLBACK] FAILED: "
                f"error_code="
                f"{result.result.error_code.val}"
            )

        return ok

    # ============================================================
    # Named State (Ready / Home)
    # ============================================================
    def plan_and_execute_named_state(
        self,
        named_state="ready",
        num_attempts=5,
        planning_time=3.0,
    ):
        if not self.move_group_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[PnP] MoveGroup server unavailable."
            )
            return False

        self.get_logger().info(
            f"[PnP] Returning to state: "
            f"{named_state}"
        )

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()

        req.group_name = "panda_arm"

        req.num_planning_attempts = (
            num_attempts
        )

        req.allowed_planning_time = (
            planning_time
        )

        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5
        req.start_state.is_diff = True

        constraints = Constraints()

        for joint, value in zip(
            self.arm_joints,
            self.home_qpos,
        ):
            jc = JointConstraint()

            jc.joint_name = joint
            jc.position = value
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0

            constraints.joint_constraints.append(
                jc
            )

        req.goal_constraints.append(
            constraints
        )

        options = PlanningOptions()

        options.plan_only = False
        options.replan = True
        options.replan_attempts = 3

        goal.request = req
        goal.planning_options = options

        future = (
            self.move_group_client.send_goal_async(
                goal
            )
        )

        handle = self.wait_future(
            future,
            10.0,
            "Named-state MoveGroup goal",
        )

        if (
            handle is None
            or not handle.accepted
        ):
            self.get_logger().error(
                "[PnP] Named-state goal rejected."
            )
            return False

        result = self.wait_future(
            handle.get_result_async(),
            20.0,
            "Named-state MoveGroup result",
        )

        if result is None:
            return False

        ok = (
            result.result.error_code.val
            == 1
        )

        if ok:
            self.get_logger().info(
                f"[PnP] Successfully returned "
                f"to {named_state} pose."
            )
        else:
            self.get_logger().error(
                f"[PnP] Failed to return "
                f"{named_state}: "
                f"error_code="
                f"{result.result.error_code.val}"
            )

        return ok

    # ============================================================
    # Unified 3D Pose Move
    # ============================================================
    def move_to_pose(
        self,
        target_x,
        target_y,
        target_z,
        qx,
        qy,
        qz,
        qw,
        start_x=None,
        start_y=None,
        start_z=None,
        description="Target",
    ):
        self.get_logger().info(
            f"[MOVE] {description} 목표 위치로 이동: "
            f"({target_x:.3f}, "
            f"{target_y:.3f}, "
            f"{target_z:.3f})"
        )

        if (
            start_x is not None
            and start_y is not None
            and start_z is not None
        ):
            ok = self.cartesian_xyz_move(
                start_x,
                start_y,
                start_z,
                target_x,
                target_y,
                target_z,
                qx,
                qy,
                qz,
                qw,
            )

            if ok:
                return True

            self.get_logger().warn(
                f"[MOVE] {description} "
                f"Cartesian 이동 실패. "
                f"Fallback Pose Planning 시도..."
            )

        else:
            ok = self.plan_and_execute_pose(
                target_x,
                target_y,
                target_z,
                qx,
                qy,
                qz,
                qw,
                num_attempts=10,
                planning_time=5.0,
            )

            if ok:
                return True

            self.get_logger().warn(
                f"[MOVE] {description} "
                f"기본 Pose Planning 실패. "
                f"Fallback Planning 시도..."
            )

        fallback_ok = self.plan_and_execute_pose(
            target_x,
            target_y,
            target_z,
            qx,
            qy,
            qz,
            qw,
            num_attempts=self.fallback_planning_attempts,
            planning_time=self.fallback_planning_time,
            vel_scale=self.fallback_velocity_scale,
            acc_scale=self.fallback_acceleration_scale,
            pos_tol=0.02,
            ori_tol=self.fallback_orientation_tolerance,
        )

        return fallback_ok

    # ============================================================
    # Gripper Control
    # ============================================================
    def control_gripper(self, action):
        action = action.upper()

        if action not in (
            "OPEN",
            "CLOSE",
        ):
            self.get_logger().error(
                f"[GRIPPER] Invalid action: "
                f"{action}"
            )
            return False, None

        if not self.gripper_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[GRIPPER] Action server unavailable."
            )
            return False, None

        goal = GripperCommand.Goal()

        if action == "OPEN":
            goal.command.position = (
                self.gripper_open_position
            )

            goal.command.max_effort = (
                self.gripper_open_effort
            )

        else:
            goal.command.position = (
                self.gripper_close_position
            )

            goal.command.max_effort = (
                self.gripper_close_effort
            )

        self.get_logger().info(
            f"[GRIPPER] Sending {action}: "
            f"position="
            f"{goal.command.position:.3f}"
        )

        try:
            future = (
                self.gripper_client.send_goal_async(
                    goal
                )
            )

            handle = self.wait_future(
                future,
                5.0,
                f"Gripper {action} goal",
            )

            if (
                handle is None
                or not handle.accepted
            ):
                self.get_logger().error(
                    f"[GRIPPER] {action} "
                    f"goal rejected."
                )
                return False, None

            res = self.wait_future(
                handle.get_result_async(),
                10.0,
                f"Gripper {action} result",
            )

            if res is None:
                return False, None

            result_obj = res.result

            self.get_logger().info(
                f"[GRIPPER] {action} result: "
                f"position="
                f"{result_obj.position:.4f}, "
                f"reached="
                f"{result_obj.reached_goal}, "
                f"stalled="
                f"{result_obj.stalled}"
            )

            success = result_obj.reached_goal

            time.sleep(0.2)

            return success, result_obj

        except Exception as e:
            self.get_logger().error(
                f"[GRIPPER] {action} exception: "
                f"{e}"
            )
            return False, None

    # ============================================================
    # Grasp Check
    # ============================================================
    def check_grasp_success(
        self,
        result_obj,
    ):
        if result_obj is None:
            return False

        pos = result_obj.position

        if (
            self.min_grasp_position_threshold
            <= pos
            <= (
                self.gripper_open_position -
                0.002
            )
        ):
            self.get_logger().info(
                f"[GRASP CHECK] "
                f"물체 파지 성공: "
                f"finger position="
                f"{pos:.4f}m"
            )
            return True

        self.get_logger().warn(
            f"[GRASP CHECK] "
            f"물체 파지 실패: "
            f"finger position="
            f"{pos:.4f}m"
        )

        return False

    # ============================================================
    # Place Pose Calculation
    # ============================================================
    def calculate_place_pose(self):
        object_center_z = (
            self.table_top_z +
            self.object_height / 2.0
        )

        place_tcp_z = (
            object_center_z -
            self.tcp_to_fingertip
        )

        return (
            self.place_x,
            self.place_y,
            place_tcp_z,
        )

    # ============================================================
    # Failure Reset & Return to Ready
    # ============================================================
    def reset_after_failure(
        self,
        reason,
        need_open_gripper=False,
    ):
        self.get_logger().error(
            f"[PnP FAIL] {reason}"
        )

        if need_open_gripper:
            self.get_logger().info(
                "[PnP RECOVERY] "
                "Opening gripper for safety..."
            )

            self.control_gripper(
                "OPEN"
            )

            time.sleep(0.3)

        self.get_logger().info(
            "[PnP RECOVERY] "
            "Returning to READY pose..."
        )

        try:
            ok = self.plan_and_execute_named_state(
                "ready"
            )

            if not ok:
                self.get_logger().warn(
                    "[PnP RECOVERY] "
                    "Ready move fallback attempt..."
                )

                self.plan_and_execute_named_state(
                    "ready",
                    num_attempts=10,
                    planning_time=5.0,
                )

        except Exception as e:
            self.get_logger().error(
                f"[PnP RECOVERY] "
                f"Recovery exception: {e}"
            )

        self.target_pose = None
        self.object_height = None
        self.target_yaw = 0.0
        self.is_busy = False
        self.state = "IDLE"

    # ============================================================
    # Pick & Place Worker Loop
    # ============================================================
    def pnp_worker_loop(self):
        while (
            rclpy.ok()
            and not self.shutdown_requested
        ):
            try:
                if (
                    self.state == "TRIGGER_PICK"
                    and self.target_pose is not None
                ):
                    self.state = "PICKING"

                    tx, ty, tz = (
                        self.target_pose
                    )

                    half_height = (
                        self.object_height / 2.0
                    )

                    qx, qy, qz, qw = (
                        self.yaw_to_grasp_quaternion(
                            self.target_yaw
                        )
                    )

                    self.get_logger().info(
                        "=" * 60
                    )

                    self.get_logger().info(
                        "[PnP] Starting 9-Step "
                        "Pick and Place Sequence"
                    )

                    self.get_logger().info(
                        f"[PnP] Target: "
                        f"({tx:.3f}, "
                        f"{ty:.3f}, "
                        f"{tz:.3f}), "
                        f"h={self.object_height:.3f}m, "
                        f"yaw="
                        f"{math.degrees(self.target_yaw):.1f}°"
                    )

                    self.get_logger().info(
                        "=" * 60
                    )

                    # ==================================================
                    # Step 1. 그리퍼 OPEN
                    # ==================================================
                    self.get_logger().info(
                        "[Step 1/9] 그리퍼 OPEN"
                    )

                    step1_ok, _ = (
                        self.control_gripper(
                            "OPEN"
                        )
                    )

                    if not step1_ok:
                        self.get_logger().warn(
                            "[Step 1/9] "
                            "Gripper OPEN failed. "
                            "Trying Fallback (retry)..."
                        )

                        time.sleep(0.3)

                        step1_ok, _ = (
                            self.control_gripper(
                                "OPEN"
                            )
                        )

                        if not step1_ok:
                            self.reset_after_failure(
                                "Step 1 "
                                "(Gripper OPEN) "
                                "failed after fallback."
                            )
                            continue

                    # ==================================================
                    # Step 2. Pre-grasp
                    # ==================================================
                    pre_grasp_z = (
                        tz +
                        half_height +
                        self.pre_grasp_z_offset
                    )

                    step2_ok = self.move_to_pose(
                        tx,
                        ty,
                        pre_grasp_z,
                        qx,
                        qy,
                        qz,
                        qw,
                        description=(
                            "[Step 2/9] Pre-grasp"
                        ),
                    )

                    if not step2_ok:
                        self.reset_after_failure(
                            "Step 2 "
                            "(Pre-grasp move) "
                            "failed after fallback."
                        )
                        continue

                    # ==================================================
                    # Step 3. Grasp descent
                    # ==================================================
                    if (
                        half_height
                        <= self.max_grasp_depth
                    ):
                        grasp_z = tz
                    else:
                        grasp_z = (
                            tz +
                            half_height -
                            self.max_grasp_depth
                        )

                    self.get_logger().info(
                        f"[Step 3/9] "
                        f"Grasp 위치 하강: "
                        f"z={pre_grasp_z:.3f} "
                        f"-> {grasp_z:.3f}"
                    )

                    step3_ok = (
                        self.cartesian_z_move(
                            tx,
                            ty,
                            pre_grasp_z,
                            grasp_z,
                            qx,
                            qy,
                            qz,
                            qw,
                            label="[Step 3 Cartesian Z]",
                        )
                    )

                    if not step3_ok:
                        self.get_logger().warn(
                            "[Step 3/9] "
                            "Cartesian descent failed. "
                            "Trying Pose fallback..."
                        )

                        step3_ok = (
                            self.lift_joint_space_fallback(
                                tx,
                                ty,
                                grasp_z,
                                qx,
                                qy,
                                qz,
                                qw,
                            )
                        )

                        if not step3_ok:
                            self.reset_after_failure(
                                "Step 3 "
                                "(Grasp descent) "
                                "failed after fallback."
                            )
                            continue

                    # ==================================================
                    # Step 4. Gripper CLOSE + grasp check
                    # ==================================================
                    self.get_logger().info(
                        "[Step 4/9] "
                        "그리퍼 CLOSE & "
                        "물체 파지 여부 체크"
                    )

                    step4_cmd_ok, step4_res = (
                        self.control_gripper(
                            "CLOSE"
                        )
                    )

                    step4_grasped = (
                        self.check_grasp_success(
                            step4_res
                        )
                        if step4_cmd_ok
                        else False
                    )

                    if not step4_grasped:
                        self.get_logger().warn(
                            "[Step 4/9] "
                            "Grasp check failed. "
                            "Trying Fallback "
                            "(reopen & reclose)..."
                        )

                        self.control_gripper(
                            "OPEN"
                        )

                        time.sleep(0.3)

                        step4_cmd_ok, step4_res = (
                            self.control_gripper(
                                "CLOSE"
                            )
                        )

                        step4_grasped = (
                            self.check_grasp_success(
                                step4_res
                            )
                            if step4_cmd_ok
                            else False
                        )

                        if not step4_grasped:
                            self.reset_after_failure(
                                "Step 4 "
                                "(Gripper CLOSE & "
                                "grasp check) "
                                "failed after fallback.",
                                need_open_gripper=True,
                            )
                            continue

                    # ==================================================
                    # Step 5. After-grasp Lift
                    #
                    # 중요:
                    # segment 분할하지 않는다.
                    # 한 번의 Cartesian Z 이동으로 상승한다.
                    # ==================================================
                    after_grasp_z = (
                        tz +
                        half_height +
                        self.lift_z_offset
                    )

                    self.get_logger().info(
                        f"[Step 5/9] "
                        f"After-grasp 위치 상승 "
                        f"(Single Cartesian Lift): "
                        f"z={grasp_z:.3f} "
                        f"-> {after_grasp_z:.3f}"
                    )

                    step5_ok = (
                        self.cartesian_z_move(
                            tx,
                            ty,
                            grasp_z,
                            after_grasp_z,
                            qx,
                            qy,
                            qz,
                            qw,
                            label="[Step 5 Cartesian Lift]",
                        )
                    )

                    if not step5_ok:
                        self.get_logger().warn(
                            "[Step 5/9] "
                            "Single Cartesian Lift failed. "
                            "Trying downward orientation fallback..."
                        )

                        step5_ok = (
                            self.lift_position_downward_fallback(
                                tx,
                                ty,
                                after_grasp_z,
                                qx,
                                qy,
                                qz,
                                qw,
                            )
                        )

                        if not step5_ok:
                            self.reset_after_failure(
                                "Step 5 "
                                "(After-grasp lift) "
                                "failed after single "
                                "Cartesian move and fallback.",
                                need_open_gripper=True,
                            )
                            continue

                    self.get_logger().info(
                        "[Step 5/9] "
                        "After-grasp Lift SUCCESS "
                        "(single Cartesian move)."
                    )

                    # ==================================================
                    # Step 6. Pre-place
                    #
                    # Step 5 완료 직후 실제 joint state 안정화 후
                    # Cartesian XYZ 한 번에 이동한다.
                    # ==================================================
                    px, py, pz = (
                        self.calculate_place_pose()
                    )

                    pre_place_z = (
                        pz +
                        self.post_place_z_offset
                    )

                    step6_ok = (
                        self.move_to_pre_place_position(
                            tx,
                            ty,
                            after_grasp_z,
                            px,
                            py,
                            pre_place_z,
                            qx,
                            qy,
                            qz,
                            qw,
                        )
                    )

                    if not step6_ok:
                        self.reset_after_failure(
                            "Step 6 "
                            "(Pre-place position move "
                            "with grasp orientation "
                            "maintenance) "
                            "failed after fallback.",
                            need_open_gripper=True,
                        )
                        continue

                    self.get_logger().info(
                        "[Step 6/9] "
                        "Pre-place 완료 "
                        "(grasp orientation 유지)."
                    )

                    # ==================================================
                    # Step 7. Place descent + OPEN
                    # ==================================================
                    self.get_logger().info(
                        f"[Step 7/9] "
                        f"Place 위치 하강: "
                        f"z={pre_place_z:.3f} "
                        f"-> {pz:.3f}"
                    )

                    step7_z_ok = (
                        self.cartesian_z_move(
                            px,
                            py,
                            pre_place_z,
                            pz,
                            qx,
                            qy,
                            qz,
                            qw,
                            label="[Step 7 Cartesian Z]",
                        )
                    )

                    if not step7_z_ok:
                        self.get_logger().warn(
                            "[Step 7/9] "
                            "Cartesian place descent "
                            "failed. "
                            "Trying Pose fallback..."
                        )

                        step7_z_ok = (
                            self.lift_joint_space_fallback(
                                px,
                                py,
                                pz,
                                qx,
                                qy,
                                qz,
                                qw,
                            )
                        )

                        if not step7_z_ok:
                            self.reset_after_failure(
                                "Step 7 "
                                "(Place descent) "
                                "failed after fallback.",
                                need_open_gripper=True,
                            )
                            continue

                    self.get_logger().info(
                        "[Step 7/9] "
                        "Place 위치 도착 -> "
                        "그리퍼 OPEN"
                    )

                    step7_open_ok, _ = (
                        self.control_gripper(
                            "OPEN"
                        )
                    )

                    if not step7_open_ok:
                        self.get_logger().warn(
                            "[Step 7/9] "
                            "Gripper OPEN at place "
                            "failed. "
                            "Trying Fallback (retry)..."
                        )

                        time.sleep(0.3)

                        step7_open_ok, _ = (
                            self.control_gripper(
                                "OPEN"
                            )
                        )

                        if not step7_open_ok:
                            self.reset_after_failure(
                                "Step 7 "
                                "(Gripper OPEN at place) "
                                "failed after fallback."
                            )
                            continue

                    time.sleep(0.5)

                    # ==================================================
                    # Step 8. After-place Retract
                    # ==================================================
                    after_place_z = pre_place_z

                    self.get_logger().info(
                        f"[Step 8/9] "
                        f"After-place 위치 상승 "
                        f"(Retract): "
                        f"z={pz:.3f} "
                        f"-> {after_place_z:.3f}"
                    )

                    step8_ok = (
                        self.cartesian_z_move(
                            px,
                            py,
                            pz,
                            after_place_z,
                            qx,
                            qy,
                            qz,
                            qw,
                            label="[Step 8 Cartesian Z]",
                        )
                    )

                    if not step8_ok:
                        self.get_logger().warn(
                            "[Step 8/9] "
                            "Retract failed. "
                            "Trying Pose fallback..."
                        )

                        step8_ok = (
                            self.lift_joint_space_fallback(
                                px,
                                py,
                                after_place_z,
                                qx,
                                qy,
                                qz,
                                qw,
                            )
                        )

                        if not step8_ok:
                            self.reset_after_failure(
                                "Step 8 "
                                "(After place retract) "
                                "failed after fallback."
                            )
                            continue

                    # ==================================================
                    # Step 9. Ready
                    # ==================================================
                    self.get_logger().info(
                        "[Step 9/9] "
                        "Ready 자세로 복귀"
                    )

                    step9_ok = (
                        self.plan_and_execute_named_state(
                            "ready"
                        )
                    )

                    if not step9_ok:
                        self.get_logger().warn(
                            "[Step 9/9] "
                            "Ready move failed. "
                            "Trying Fallback..."
                        )

                        step9_ok = (
                            self.plan_and_execute_named_state(
                                "ready",
                                num_attempts=10,
                                planning_time=5.0,
                            )
                        )

                        if not step9_ok:
                            self.reset_after_failure(
                                "Step 9 "
                                "(Return to ready) "
                                "failed after fallback."
                            )
                            continue

                    # ==================================================
                    # SUCCESS
                    # ==================================================
                    self.get_logger().info(
                        "=" * 60
                    )

                    self.get_logger().info(
                        "[PnP SUCCESS] "
                        "9-Step Pick & Place "
                        "Completed Successfully!"
                    )

                    self.get_logger().info(
                        "=" * 60
                    )

                    self.target_pose = None
                    self.object_height = None
                    self.target_yaw = 0.0
                    self.is_busy = False
                    self.state = "IDLE"

                time.sleep(0.05)

            except Exception as e:
                self.get_logger().error(
                    f"[PnP WORKER] Exception: {e}"
                )

                self.reset_after_failure(
                    f"Worker exception: {e}"
                )

    # ============================================================
    # Shutdown
    # ============================================================
    def shutdown(self):
        self.shutdown_requested = True

        if (
            self.worker.is_alive()
            and threading.current_thread()
            is not self.worker
        ):
            self.worker.join(
                timeout=2.0
            )


def main(args=None):
    rclpy.init(
        args=args
    )

    node = PandaMoveItPickAndPlace()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()