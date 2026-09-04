#!/usr/bin/env python3
"""
MoveIt 2 기반 Franka Emika Panda 3D Vision Pick & Place Controller Node

Pick & Place Sequence (9 Steps):
  1. 그리퍼 open (실패 시 1회 재시도 fallback)
  2. pre grasp 위치 계산 및 이동 (실패 시 1회 planning 완화 fallback)
  3. grasp 위치로 이동 (z축 하강) (실패 시 1회 joint-space / segmented fallback)
  4. 그리퍼 닫기 & 물체 잡았는지 여부 체크 (실패 시 1회 reopen & reclose fallback)
  5. after grasp 위치로 이동 (z축 상승) (실패 시 1회 orientation-constrained joint-space fallback)
  6. pre place 위치 계산 및 이동 (실패 시 1회 orientation-constrained joint-space fallback)
  7. place 위치로 이동 (z축 하강) & 그리퍼 오픈 (실패 시 1회 z축 / open fallback)
  8. after place 위치로 이동 (z축 상승) (실패 시 1회 z축 상승 fallback)
  9. ready 자세로 복귀 (실패 시 1회 planning 완화 fallback)

* 모든 step에서 동작 또는 경로 계산 실패 시 1번의 fallback 적용,
  그럼에도 실패 시 fail 처리 후 ready 자세로 안전 복귀.
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

        self.arm_joints = [f"joint{i}" for i in range(1, 8)]

        # ============================================================
        # Place target position
        # ============================================================
        self.place_x = 0.0
        self.place_y = -0.55
        self.table_top_z = 0.44

        # ============================================================
        # Motion offsets (m)
        # ============================================================
        self.pre_grasp_z_offset = 0.20
        self.lift_z_offset = 0.20
        self.post_place_z_offset = 0.20

        # ============================================================
        # Step sizes
        # ============================================================
        self.lift_segment_step = 0.05
        self.pre_place_xy_step = 0.05

        # ============================================================
        # Fallback planning parameters
        # ============================================================
        self.fallback_planning_attempts = 15
        self.fallback_planning_time = 6.0
        self.fallback_velocity_scale = 0.20
        self.fallback_acceleration_scale = 0.20
        self.fallback_orientation_tolerance = 0.08

        # ============================================================
        # Gripper configuration
        # ============================================================
        self.gripper_open_position = 0.04
        self.gripper_close_position = 0.0
        self.gripper_open_effort = 20.0
        self.gripper_close_effort = 30.0

        # Grasp check thresholds (물체가 잡혔을 때의 최소 열림 간격)
        self.min_grasp_position_threshold = 0.002

        # ============================================================
        # Gripper geometry
        # ============================================================
        self.gripper_clearance = 0.100
        self.grasp_margin = 0.010
        self.max_grasp_depth = self.gripper_clearance - self.grasp_margin

        self.tcp_to_fingertip = 0.1100 - (0.0584 + 0.0445)

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
            "[PnP INIT] Franka Panda MoveIt2 9-Step Pick & Place Controller Ready."
        )
        self.get_logger().info(
            f"[PnP INIT] Table Z={self.table_top_z:.3f}, Place XY=({self.place_x:.3f},{self.place_y:.3f})"
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
                f"[PnP] Invalid target frame: {msg.header.frame_id}"
            )
            return

        p = msg.pose.position
        q = msg.pose.orientation

        x = float(p.x)
        y = float(p.y)
        z = float(p.z)

        if not all(np.isfinite(v) for v in (x, y, z)):
            self.get_logger().warn("[PnP] Invalid target position.")
            return

        if self.object_height is None or not np.isfinite(self.object_height):
            self.get_logger().warn("[PnP] Object height is not available.")
            return

        h = float(self.object_height)
        if not (self.min_object_height <= h <= self.max_object_height):
            self.get_logger().warn(f"[PnP] Invalid object height: {h:.4f} m")
            return

        r11 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        r21 = 2.0 * (q.x * q.y + q.w * q.z)
        yaw = math.atan2(r21, r11)

        self.target_pose = np.array([x, y, z], dtype=np.float64)
        self.target_yaw = yaw
        self.is_busy = True
        self.state = "TRIGGER_PICK"

        self.get_logger().info(
            f"[PnP] Target received: xyz=({x:.3f},{y:.3f},{z:.3f}), h={h:.3f}, yaw={math.degrees(yaw):.1f} deg"
        )

    # ============================================================
    # Future helper
    # ============================================================
    def wait_future(self, future, timeout, description):
        event = threading.Event()
        result = [None]

        def done_callback(f):
            result[0] = f
            event.set()

        future.add_done_callback(done_callback)

        if not event.wait(timeout):
            self.get_logger().error(f"[PnP] Timeout waiting for {description}")
            return None

        try:
            return result[0].result()
        except Exception as e:
            self.get_logger().error(f"[PnP] {description} failed: {e}")
            return None

    # ============================================================
    # Grasp orientation
    # ============================================================
    def yaw_to_grasp_quaternion(self, yaw):
        while yaw > math.pi / 2:
            yaw -= math.pi
        while yaw < -math.pi / 2:
            yaw += math.pi

        cy = math.cos(yaw / 2.0)
        sy = math.sin(yaw / 2.0)
        return cy, sy, 0.0, 0.0

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
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[PnP] MoveGroup server unavailable.")
            return False

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = "panda_arm"
        req.num_planning_attempts = num_attempts
        req.allowed_planning_time = planning_time
        req.max_velocity_scaling_factor = vel_scale
        req.max_acceleration_scaling_factor = acc_scale
        req.start_state.is_diff = True

        pc = PositionConstraint()
        pc.header.frame_id = "link0"
        pc.link_name = "hand_tcp"
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [pos_tol]
        pc.constraint_region = BoundingVolume()
        pc.constraint_region.primitives.append(primitive)

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = "link0"
        oc.link_name = "hand_tcp"
        oc.orientation.x = qx
        oc.orientation.y = qy
        oc.orientation.z = qz
        oc.orientation.w = qw
        oc.absolute_x_axis_tolerance = ori_tol
        oc.absolute_y_axis_tolerance = ori_tol
        oc.absolute_z_axis_tolerance = ori_tol + 0.10
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

        goal.request = req
        goal.planning_options = options

        future = self.move_group_client.send_goal_async(goal)
        handle = self.wait_future(future, 10.0, "MoveGroup goal")

        if handle is None or not handle.accepted:
            self.get_logger().error("[PnP] MoveGroup goal rejected.")
            return False

        result = self.wait_future(handle.get_result_async(), 30.0, "MoveGroup result")
        if result is None:
            return False

        ok = result.result.error_code.val == 1
        if not ok:
            self.get_logger().error(
                f"[PnP] MoveGroup failed: error_code={result.result.error_code.val}"
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
    ):
        if abs(end_z - start_z) < 0.001:
            return True

        if not self.cartesian_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("[PnP] /compute_cartesian_path unavailable.")
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

        future = self.cartesian_client.call_async(req)
        response = self.wait_future(future, 20.0, "Cartesian path computation")
        if response is None:
            return False

        self.get_logger().info(
            f"[Cartesian Z] z {start_z:.3f} -> {end_z:.3f}, fraction={response.fraction:.3f}"
        )

        if response.fraction < self.cartesian_fraction_threshold:
            self.get_logger().warn(
                f"[Cartesian Z] Fraction too low: {response.fraction:.3f} < {self.cartesian_fraction_threshold:.2f}"
            )
            return False

        # Execute trajectory
        if not self.execute_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[PnP] ExecuteTrajectory server unavailable.")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        future = self.execute_client.send_goal_async(goal)
        handle = self.wait_future(future, 10.0, "ExecuteTrajectory goal")

        if handle is None or not handle.accepted:
            self.get_logger().error("[PnP] ExecuteTrajectory goal rejected.")
            return False

        result = self.wait_future(handle.get_result_async(), 30.0, "ExecuteTrajectory result")
        if result is None:
            return False

        ok = result.result.error_code.val == 1
        if not ok:
            self.get_logger().error(
                f"[PnP] ExecuteTrajectory failed: error_code={result.result.error_code.val}"
            )
        return ok

    # ============================================================
    # Segmented Cartesian Z Move
    # ============================================================
    def cartesian_z_move_segmented(
        self,
        x,
        y,
        start_z,
        end_z,
        qx,
        qy,
        qz,
        qw,
        step=0.05,
    ):
        delta = end_z - start_z
        if abs(delta) < 0.001:
            return True

        direction = 1.0 if delta > 0.0 else -1.0
        total_segments = int(math.ceil(abs(delta) / max(step, 0.005)))
        current_z = float(start_z)

        self.get_logger().info(
            f"[Segmented Z] z={start_z:.3f}->{end_z:.3f}, step={step:.3f}, segments={total_segments}"
        )

        for segment_index in range(1, total_segments + 1):
            remaining = abs(end_z - current_z)
            move = min(step, remaining)
            next_z = current_z + direction * move

            # 1. Cartesian 이동 시도
            if self.cartesian_z_move(x, y, current_z, next_z, qx, qy, qz, qw):
                current_z = next_z
                continue

            # 2. Cartesian 실패 시 세그먼트 fallback (Joint-space Z)
            self.get_logger().warn(
                f"[Segmented Z] Segment {segment_index} Cartesian failed. Trying joint-space Z fallback."
            )
            if self.lift_joint_space_fallback(x, y, next_z, qx, qy, qz, qw):
                current_z = next_z
                continue

            self.get_logger().error(f"[Segmented Z] Segment {segment_index} fallback failed.")
            return False

        return True

    # ============================================================
    # Cartesian XYZ Move (3D Linear Motion)
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
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance < 0.001:
            return True

        if not self.cartesian_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("[Cartesian XYZ] /compute_cartesian_path unavailable.")
            return False

        step = max(0.005, float(self.pre_place_xy_step))
        segments = int(math.ceil(distance / step))
        waypoints = []

        for i in range(1, segments + 1):
            ratio = float(i) / float(segments)
            waypoint = Pose()
            waypoint.position.x = start_x + dx * ratio
            waypoint.position.y = start_y + dy * ratio
            waypoint.position.z = start_z + dz * ratio
            waypoint.orientation.x = qx
            waypoint.orientation.y = qy
            waypoint.orientation.z = qz
            waypoint.orientation.w = qw
            waypoints.append(waypoint)

        req = GetCartesianPath.Request()
        req.header.frame_id = "link0"
        req.group_name = "panda_arm"
        req.link_name = "hand_tcp"
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state.is_diff = True
        req.waypoints = waypoints

        future = self.cartesian_client.call_async(req)
        response = self.wait_future(future, 30.0, "Cartesian XYZ path computation")
        if response is None:
            return False

        self.get_logger().info(f"[Cartesian XYZ] fraction={response.fraction:.3f}")

        if response.fraction < self.cartesian_fraction_threshold:
            self.get_logger().warn(
                f"[Cartesian XYZ] Path fraction too low: {response.fraction:.3f}"
            )
            return False

        if not self.execute_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[Cartesian XYZ] ExecuteTrajectory server unavailable.")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        future = self.execute_client.send_goal_async(goal)
        handle = self.wait_future(future, 10.0, "Cartesian XYZ ExecuteTrajectory goal")

        if handle is None or not handle.accepted:
            self.get_logger().error("[Cartesian XYZ] ExecuteTrajectory goal rejected.")
            return False

        result = self.wait_future(handle.get_result_async(), 30.0, "Cartesian XYZ result")
        if result is None:
            return False

        ok = result.result.error_code.val == 1
        return ok

    # ============================================================
    # Fallbacks with Orientation Constraints
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
            f"[Z FALLBACK] Executing joint-space fallback to z={target_z:.3f}"
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
    # Named State (Ready / Home)
    # ============================================================
    def plan_and_execute_named_state(
        self,
        named_state="ready",
        num_attempts=5,
        planning_time=3.0,
    ):
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[PnP] MoveGroup server unavailable.")
            return False

        self.get_logger().info(f"[PnP] Returning to state: {named_state}")

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = "panda_arm"
        req.num_planning_attempts = num_attempts
        req.allowed_planning_time = planning_time
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5
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

        goal.request = req
        goal.planning_options = options

        future = self.move_group_client.send_goal_async(goal)
        handle = self.wait_future(future, 10.0, "Named-state MoveGroup goal")

        if handle is None or not handle.accepted:
            self.get_logger().error("[PnP] Named-state goal rejected.")
            return False

        result = self.wait_future(handle.get_result_async(), 20.0, "Named-state MoveGroup result")
        if result is None:
            return False

        ok = result.result.error_code.val == 1
        if ok:
            self.get_logger().info(f"[PnP] Successfully returned to {named_state} pose.")
        else:
            self.get_logger().error(
                f"[PnP] Failed to return {named_state}: error_code={result.result.error_code.val}"
            )
        return ok

    # ============================================================
    # Unified 3D Pose Move (Pre-grasp, Pre-place 등 공통 이동)
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
        """
        목표 3D Pose로의 통합 이동 함수 (Pre-grasp, Pre-place 등 공통 사용)
        1차 시도:
          - start 좌표가 전달되면 Cartesian 3D 선형 이동(cartesian_xyz_move) 시도
          - start 좌표가 없으면(자유공간/Ready 등) MoveGroup Pose Planning 시도
        실패 시 1회 Fallback:
          - 완화된 파라미터로 MoveGroup Pose Planning 재시도
        """
        self.get_logger().info(
            f"[MOVE] {description} 목표 위치로 이동: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
        )

        # 1차 시도
        if start_x is not None and start_y is not None and start_z is not None:
            ok = self.cartesian_xyz_move(
                start_x, start_y, start_z, target_x, target_y, target_z, qx, qy, qz, qw
            )
            if ok:
                return True
            self.get_logger().warn(
                f"[MOVE] {description} Cartesian 이동 실패. Fallback (Joint-space Pose Planning) 시도..."
            )
        else:
            ok = self.plan_and_execute_pose(
                target_x, target_y, target_z, qx, qy, qz, qw, num_attempts=10, planning_time=5.0
            )
            if ok:
                return True
            self.get_logger().warn(
                f"[MOVE] {description} 기본 Pose Planning 실패. Fallback (완화된 Planning) 시도..."
            )

        # 1회 Fallback
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
    # Gripper Control & Grasp Check (PnP 노드 전담)
    # ============================================================
    def control_gripper(self, action):
        action = action.upper()
        if action not in ("OPEN", "CLOSE"):
            self.get_logger().error(f"[GRIPPER] Invalid action: {action}")
            return False, None

        if not self.gripper_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[GRIPPER] Action server unavailable.")
            return False, None

        goal = GripperCommand.Goal()
        if action == "OPEN":
            goal.command.position = self.gripper_open_position
            goal.command.max_effort = self.gripper_open_effort
        else:
            goal.command.position = self.gripper_close_position
            goal.command.max_effort = self.gripper_close_effort

        self.get_logger().info(
            f"[GRIPPER] Sending {action}: position={goal.command.position:.3f}"
        )

        try:
            future = self.gripper_client.send_goal_async(goal)
            handle = self.wait_future(future, 5.0, f"Gripper {action} goal")

            if handle is None or not handle.accepted:
                self.get_logger().error(f"[GRIPPER] {action} goal rejected.")
                return False, None

            res = self.wait_future(handle.get_result_async(), 10.0, f"Gripper {action} result")
            if res is None:
                return False, None

            result_obj = res.result
            self.get_logger().info(
                f"[GRIPPER] {action} result: position={result_obj.position:.4f}, "
                f"reached={result_obj.reached_goal}, stalled={result_obj.stalled}"
            )

            success = result_obj.reached_goal
            time.sleep(0.2)
            return success, result_obj

        except Exception as e:
            self.get_logger().error(f"[GRIPPER] {action} exception: {e}")
            return False, None

    def check_grasp_success(self, result_obj):
        """
        물체 파지 감지 (PnP 노드에서 직접 수행)
        - Gripper CLOSE 후 반환된 핑거 위치(position)가 완전히 0(빈손)으로 닫히지 않고
          물체의 크기에 걸려있는지(min_grasp_position <= pos <= max_grasp_position) 검증
        """
        if result_obj is None:
            return False

        pos = result_obj.position
        # 물체가 있을 때 핑거가 물체 두께로 인해 0.002m ~ 0.038m 사이에서 멈춤
        if self.min_grasp_position_threshold <= pos <= (self.gripper_open_position - 0.002):
            self.get_logger().info(
                f"[GRASP CHECK] 물체 파지 성공 (Grasp SUCCESS): finger position={pos:.4f}m"
            )
            return True

        self.get_logger().warn(
            f"[GRASP CHECK] 물체 파지 실패 (Grasp FAILED): finger position={pos:.4f}m (물체 없음 또는 닫힘 실패)"
        )
        return False

    # ============================================================
    # Place Pose Calculation
    # ============================================================
    def calculate_place_pose(self):
        object_center_z = self.table_top_z + self.object_height / 2.0
        place_tcp_z = object_center_z - self.tcp_to_fingertip
        return self.place_x, self.place_y, place_tcp_z

    # ============================================================
    # Failure Reset & Return to Ready
    # ============================================================
    def reset_after_failure(self, reason, need_open_gripper=False):
        self.get_logger().error(f"[PnP FAIL] {reason}")

        if need_open_gripper:
            self.get_logger().info("[PnP RECOVERY] Opening gripper for safety...")
            self.control_gripper("OPEN")
            time.sleep(0.3)

        self.get_logger().info("[PnP RECOVERY] Returning to READY pose...")
        try:
            ok = self.plan_and_execute_named_state("ready")
            if not ok:
                # Fallback for recovery
                self.get_logger().warn("[PnP RECOVERY] Ready move fallback attempt...")
                self.plan_and_execute_named_state("ready", num_attempts=10, planning_time=5.0)
        except Exception as e:
            self.get_logger().error(f"[PnP RECOVERY] Recovery exception: {e}")

        self.target_pose = None
        self.object_height = None
        self.target_yaw = 0.0
        self.is_busy = False
        self.state = "IDLE"

    # ============================================================
    # Pick & Place Worker Loop (9-Step Sequence)
    # ============================================================
    def pnp_worker_loop(self):
        while rclpy.ok() and not self.shutdown_requested:
            try:
                if self.state == "TRIGGER_PICK" and self.target_pose is not None:
                    self.state = "PICKING"
                    tx, ty, tz = self.target_pose
                    half_height = self.object_height / 2.0
                    qx, qy, qz, qw = self.yaw_to_grasp_quaternion(self.target_yaw)

                    self.get_logger().info("=" * 60)
                    self.get_logger().info("[PnP] Starting 9-Step Pick and Place Sequence")
                    self.get_logger().info(
                        f"[PnP] Target: ({tx:.3f}, {ty:.3f}, {tz:.3f}), h={self.object_height:.3f}m, yaw={math.degrees(self.target_yaw):.1f}°"
                    )
                    self.get_logger().info("=" * 60)

                    # ----------------------------------------------------
                    # Step 1. 그리퍼 open
                    # ----------------------------------------------------
                    self.get_logger().info("[Step 1/9] 그리퍼 OPEN")
                    step1_ok, _ = self.control_gripper("OPEN")
                    if not step1_ok:
                        self.get_logger().warn("[Step 1/9] Gripper OPEN failed. Trying Fallback (retry)...")
                        time.sleep(0.3)
                        step1_ok, _ = self.control_gripper("OPEN")
                        if not step1_ok:
                            self.reset_after_failure("Step 1 (Gripper OPEN) failed after fallback.")
                            continue

                    # ----------------------------------------------------
                    # Step 2. pre grasp 위치 계산 및 이동
                    # ----------------------------------------------------
                    pre_grasp_z = tz + half_height + self.pre_grasp_z_offset
                    step2_ok = self.move_to_pose(
                        tx, ty, pre_grasp_z, qx, qy, qz, qw, description="[Step 2/9] Pre-grasp"
                    )
                    if not step2_ok:
                        self.reset_after_failure("Step 2 (Pre-grasp move) failed after fallback.")
                        continue

                    # ----------------------------------------------------
                    # Step 3. grasp 위치로 이동(z축 하강)
                    # ----------------------------------------------------
                    if half_height <= self.max_grasp_depth:
                        grasp_z = tz
                    else:
                        grasp_z = (tz + half_height) - self.max_grasp_depth

                    self.get_logger().info(
                        f"[Step 3/9] Grasp 위치 하강: z={pre_grasp_z:.3f} -> {grasp_z:.3f}"
                    )

                    step3_ok = self.cartesian_z_move(
                        tx, ty, pre_grasp_z, grasp_z, qx, qy, qz, qw
                    )
                    if not step3_ok:
                        self.get_logger().warn("[Step 3/9] Cartesian descent failed. Trying Fallback (segmented/joint Z)...")
                        step3_ok = self.lift_joint_space_fallback(tx, ty, grasp_z, qx, qy, qz, qw)
                        if not step3_ok:
                            self.reset_after_failure("Step 3 (Grasp descent) failed after fallback.")
                            continue

                    # ----------------------------------------------------
                    # Step 4. 그리퍼 닫기 & 물체 잡았는지 여부 체크
                    # ----------------------------------------------------
                    self.get_logger().info("[Step 4/9] 그리퍼 CLOSE & 물체 파지 여부 체크")
                    step4_cmd_ok, step4_res = self.control_gripper("CLOSE")
                    step4_grasped = self.check_grasp_success(step4_res) if step4_cmd_ok else False

                    if not step4_grasped:
                        self.get_logger().warn("[Step 4/9] Grasp check failed. Trying Fallback (reopen & reclose)...")
                        # Fallback: 살짝 열고 다시 CLOSE 시도
                        self.control_gripper("OPEN")
                        time.sleep(0.3)
                        step4_cmd_ok, step4_res = self.control_gripper("CLOSE")
                        step4_grasped = self.check_grasp_success(step4_res) if step4_cmd_ok else False

                        if not step4_grasped:
                            self.reset_after_failure(
                                "Step 4 (Gripper CLOSE & grasp check) failed after fallback.",
                                need_open_gripper=True,
                            )
                            continue

                    # ----------------------------------------------------
                    # Step 5. after grasp 위치로 이동(z축 상승)
                    # ----------------------------------------------------
                    after_grasp_z = tz + half_height + self.lift_z_offset
                    self.get_logger().info(
                        f"[Step 5/9] After-grasp 위치 상승 (Lift): z={grasp_z:.3f} -> {after_grasp_z:.3f}"
                    )

                    step5_ok = self.cartesian_z_move_segmented(
                        tx, ty, grasp_z, after_grasp_z, qx, qy, qz, qw, self.lift_segment_step
                    )
                    if not step5_ok:
                        self.get_logger().warn("[Step 5/9] Cartesian lift failed. Trying Fallback (joint-space Z lift)...")
                        step5_ok = self.lift_joint_space_fallback(tx, ty, after_grasp_z, qx, qy, qz, qw)
                        if not step5_ok:
                            self.reset_after_failure(
                                "Step 5 (After-grasp lift) failed after fallback.",
                                need_open_gripper=True,
                            )
                            continue

                    # ----------------------------------------------------
                    # Step 6. pre place 위치 계산 및 이동 (X, Y, Z 한 번에 이동)
                    # ----------------------------------------------------
                    px, py, pz = self.calculate_place_pose()
                    pre_place_z = pz + self.post_place_z_offset

                    step6_ok = self.move_to_pose(
                        px,
                        py,
                        pre_place_z,
                        qx,
                        qy,
                        qz,
                        qw,
                        start_x=tx,
                        start_y=ty,
                        start_z=after_grasp_z,
                        description="[Step 6/9] Pre-place",
                    )
                    if not step6_ok:
                        self.reset_after_failure(
                            "Step 6 (Pre-place XYZ move) failed after fallback.",
                            need_open_gripper=True,
                        )
                        continue

                    # ----------------------------------------------------
                    # Step 7. place 위치로 이동(z축 하강) & 그리퍼 오픈
                    # ----------------------------------------------------
                    self.get_logger().info(
                        f"[Step 7/9] Place 위치 하강: z={pre_place_z:.3f} -> {pz:.3f}"
                    )
                    step7_z_ok = self.cartesian_z_move(
                        px, py, pre_place_z, pz, qx, qy, qz, qw
                    )
                    if not step7_z_ok:
                        self.get_logger().warn("[Step 7/9] Cartesian place descent failed. Trying Fallback...")
                        step7_z_ok = self.lift_joint_space_fallback(px, py, pz, qx, qy, qz, qw)
                        if not step7_z_ok:
                            self.reset_after_failure(
                                "Step 7 (Place descent) failed after fallback.",
                                need_open_gripper=True,
                            )
                            continue

                    self.get_logger().info("[Step 7/9] Place 위치 도착 -> 그리퍼 OPEN")
                    step7_open_ok, _ = self.control_gripper("OPEN")
                    if not step7_open_ok:
                        self.get_logger().warn("[Step 7/9] Gripper OPEN at place failed. Trying Fallback (retry)...")
                        time.sleep(0.3)
                        step7_open_ok, _ = self.control_gripper("OPEN")
                        if not step7_open_ok:
                            self.reset_after_failure("Step 7 (Gripper OPEN at place) failed after fallback.")
                            continue

                    time.sleep(0.5)

                    # ----------------------------------------------------
                    # Step 8. after place 위치로 이동(z축 상승)
                    # ----------------------------------------------------
                    after_place_z = pre_place_z
                    self.get_logger().info(
                        f"[Step 8/9] After-place 위치 상승 (Retract): z={pz:.3f} -> {after_place_z:.3f}"
                    )
                    step8_ok = self.cartesian_z_move(
                        px, py, pz, after_place_z, qx, qy, qz, qw
                    )
                    if not step8_ok:
                        self.get_logger().warn("[Step 8/9] Retract failed. Trying Fallback...")
                        step8_ok = self.lift_joint_space_fallback(
                            px, py, after_place_z, qx, qy, qz, qw
                        )
                        if not step8_ok:
                            self.reset_after_failure("Step 8 (After place retract) failed after fallback.")
                            continue

                    # ----------------------------------------------------
                    # Step 9. ready 자세로 복귀
                    # ----------------------------------------------------
                    self.get_logger().info("[Step 9/9] Ready 자세로 복귀")
                    step9_ok = self.plan_and_execute_named_state("ready")
                    if not step9_ok:
                        self.get_logger().warn("[Step 9/9] Ready move failed. Trying Fallback...")
                        step9_ok = self.plan_and_execute_named_state(
                            "ready", num_attempts=10, planning_time=5.0
                        )
                        if not step9_ok:
                            self.reset_after_failure("Step 9 (Return to ready) failed after fallback.")
                            continue

                    self.get_logger().info("=" * 60)
                    self.get_logger().info("[PnP SUCCESS] 9-Step Pick & Place Completed Successfully!")
                    self.get_logger().info("=" * 60)

                    self.target_pose = None
                    self.object_height = None
                    self.target_yaw = 0.0
                    self.is_busy = False
                    self.state = "IDLE"

                time.sleep(0.05)

            except Exception as e:
                self.get_logger().error(f"[PnP WORKER] Exception: {e}")
                self.reset_after_failure(f"Worker exception: {e}")

    # ============================================================
    # Shutdown
    # ============================================================
    def shutdown(self):
        self.shutdown_requested = True
        if self.worker.is_alive() and threading.current_thread() is not self.worker:
            self.worker.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = PandaMoveItPickAndPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__=="__main__":
    main()