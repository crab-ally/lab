#!/usr/bin/env python3
"""MoveIt 2 기반 Franka Panda 3D Vision Pick & Place Controller."""

import time, math, threading
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
    MotionPlanRequest, Constraints, PositionConstraint,
    OrientationConstraint, BoundingVolume, PlanningOptions,
    JointConstraint
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

        # Panda ready pose / joints
        self.home_qpos = [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.57079, 0.785398]
        self.arm_joints = [f"joint{i}" for i in range(1, 8)]

        # Place
        self.place_x, self.place_y = 0.0, -0.55
        self.table_top_z = 0.44

        # Motion offsets
        self.pre_grasp_z_offset = 0.10
        self.lift_z_offset = 0.10
        self.post_place_z_offset = 0.10
        self.pre_place_xy_step = 0.05

        # Lift / fallback
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
            callback_group=self.cb_group)

        # Gripper
        self.gripper_open_position = 0.04
        self.gripper_close_position = 0.0
        self.gripper_open_effort = 20.0
        self.gripper_close_effort = 30.0
        self.min_grasp_position_threshold = 0.002

        # Gripper geometry
        self.gripper_clearance = 0.100
        self.grasp_margin = 0.010
        self.max_grasp_depth = self.gripper_clearance - self.grasp_margin
        self.tcp_to_fingertip = 0.1100 - (0.0584 + 0.0445)

        # Object validation
        self.min_object_height = 0.005
        self.max_object_height = 0.40
        self.cartesian_fraction_threshold = 0.95

        # MoveIt clients
        self.move_group_client = ActionClient(
            self, MoveGroup, "/move_action", callback_group=self.cb_group)
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory",
            callback_group=self.cb_group)
        self.gripper_client = ActionClient(
            self, GripperCommand, "/panda_hand_controller/gripper_action",
            callback_group=self.cb_group)
        self.cartesian_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path",
            callback_group=self.cb_group)

        # Target
        self.target_sub = self.create_subscription(
            PoseStamped, "/target_object_pose",
            self.target_pose_callback, 10, callback_group=self.cb_group)
        self.height_sub = self.create_subscription(
            Float32, "/object_height",
            self.object_height_callback, 10, callback_group=self.cb_group)

        self.get_logger().info(
            "[PnP INIT] Franka Panda 9-Step Pick & Place Controller Ready.")
        self.get_logger().info(
            f"[PnP INIT] Table Z={self.table_top_z:.3f}, "
            f"Place XY=({self.place_x:.3f},{self.place_y:.3f})")
        self.get_logger().info(
            "[PnP INIT] Step 5=Single Cartesian Z Lift, "
            "Step 6=Cartesian XYZ + grasp orientation 유지")

        self.worker = threading.Thread(
            target=self.pnp_worker_loop, daemon=True)
        self.worker.start()

    # Joint state
    def joint_state_callback(self, msg):
        self.latest_joint_state = msg
        self.latest_joint_state_time = time.monotonic()

    def get_current_arm_joint_state(self):
        if self.latest_joint_state is None:
            return None
        states = dict(zip(
            self.latest_joint_state.name,
            self.latest_joint_state.position))
        if any(j not in states for j in self.arm_joints):
            return None
        return np.array(
            [float(states[j]) for j in self.arm_joints],
            dtype=np.float64)

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
            f"last_time={end_sec:.6f}s, duration={end_sec - start_sec:.6f}s")
        self.get_logger().info(
            f"{label} first_position="
            f"{[round(float(v), 6) for v in points[0].positions]}")
        self.get_logger().info(
            f"{label} last_position="
            f"{[round(float(v), 6) for v in points[-1].positions]}")

        current = self.get_current_arm_joint_state()
        if current is None:
            self.get_logger().warn(
                f"{label} Current /joint_states unavailable or incomplete.")
            self.get_logger().info(f"{label} ==================================")
            return

        start = np.array(points[0].positions, dtype=np.float64)
        if len(names) != len(self.arm_joints):
            self.get_logger().warn(
                f"{label} Trajectory joint count != Panda arm joint count.")

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
            f"{[round(float(v), 6) for v in current]}")
        self.get_logger().info(
            f"{label} start_state_error="
            f"{[round(float(v), 6) for v in error]}")
        self.get_logger().info(
            f"{label} max_start_state_error={max_error:.6f} rad "
            f"({math.degrees(max_error):.3f} deg)")
        self.get_logger().info(
            f"{label} worst_joint={self.arm_joints[max_index]}")

        if max_error >= self.step6_joint_error_critical:
            self.get_logger().error(
                f"{label} CRITICAL: trajectory start state differs "
                f"from current by {max_error:.4f} rad.")
        elif max_error >= self.step6_joint_error_warning:
            self.get_logger().warn(
                f"{label} WARNING: trajectory start state differs "
                f"from current by {max_error:.4f} rad.")
        else:
            self.get_logger().info(
                f"{label} Start-state synchronization looks OK.")

        self.get_logger().info(f"{label} ==================================")

    def execute_trajectory_with_diagnostics(
        self, trajectory, label="[ExecuteTrajectory]", timeout_sec=30.0
    ):
        if not self.execute_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(
                f"{label} ExecuteTrajectory server unavailable.")
            return False

        self.log_trajectory_execution_diagnostics(trajectory, label)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        self.get_logger().info(
            f"{label} Sending trajectory to ExecuteTrajectory...")

        send_time = time.monotonic()
        handle = self.wait_future(
            self.execute_client.send_goal_async(goal),
            10.0, f"{label} goal")

        if handle is None:
            self.get_logger().error(f"{label} Goal response timeout.")
            return False
        if not handle.accepted:
            self.get_logger().error(f"{label} Goal REJECTED.")
            return False

        self.get_logger().info(f"{label} Goal ACCEPTED.")
        result = self.wait_future(
            handle.get_result_async(), timeout_sec, f"{label} result")
        elapsed = time.monotonic() - send_time

        if result is None:
            self.get_logger().error(f"{label} Result timeout.")
            return False

        code = result.result.error_code.val
        self.get_logger().info(
            f"{label} Result received after {elapsed:.3f}s")
        self.get_logger().info(
            f"{label} MoveIt error_code={code}")

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
            f"{label} FAILURE: {errors.get(code, 'UNKNOWN')} ({code})")

        current = self.get_current_arm_joint_state()
        if current is not None:
            self.get_logger().error(
                f"{label} Current joint state AFTER failure="
                f"{[round(float(v), 6) for v in current]}")
        self.get_logger().error(f"{label} ==================================")
        return False

    # Subscribers
    def object_height_callback(self, msg):
        if self.is_busy:
            return
        height = float(msg.data)
        if np.isfinite(height):
            self.object_height = height

    def target_pose_callback(self, msg):
        if self.is_busy or self.state != "IDLE":
            return
        if msg.header.frame_id != "link0":
            self.get_logger().warn(
                f"[PnP] Invalid target frame: {msg.header.frame_id}")
            return

        p, q = msg.pose.position, msg.pose.orientation
        x, y, z = float(p.x), float(p.y), float(p.z)
        if not all(np.isfinite(v) for v in (x, y, z)):
            self.get_logger().warn("[PnP] Invalid target position.")
            return

        if self.object_height is None or not np.isfinite(self.object_height):
            self.get_logger().warn("[PnP] Object height is not available.")
            return

        h = float(self.object_height)
        if not self.min_object_height <= h <= self.max_object_height:
            self.get_logger().warn(f"[PnP] Invalid object height: {h:.4f} m")
            return

        r11 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        r21 = 2.0 * (q.x * q.y + q.w * q.z)
        self.target_pose = np.array([x, y, z], dtype=np.float64)
        self.target_yaw = math.atan2(r21, r11)
        self.is_busy = True
        self.state = "TRIGGER_PICK"

        self.get_logger().info(
            f"[PnP] Target received: xyz=({x:.3f},{y:.3f},{z:.3f}), "
            f"h={h:.3f}, yaw={math.degrees(self.target_yaw):.1f} deg")

    # Future helper
    def wait_future(self, future, timeout, description):
        event, result = threading.Event(), [None]

        def done_callback(f):
            result[0] = f
            event.set()

        future.add_done_callback(done_callback)
        if not event.wait(timeout):
            self.get_logger().error(
                f"[PnP] Timeout waiting for {description}")
            return None
        try:
            return result[0].result()
        except Exception as e:
            self.get_logger().error(
                f"[PnP] {description} failed: {e}")
            return None

    # Grasp orientation
    def yaw_to_grasp_quaternion(self, yaw):
        while yaw > math.pi / 2:
            yaw -= math.pi
        while yaw < -math.pi / 2:
            yaw += math.pi
        return math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0

    # Pose planning
    def plan_and_execute_pose(
        self, x, y, z, qx=1.0, qy=0.0, qz=0.0, qw=0.0,
        num_attempts=10, planning_time=5.0,
        vel_scale=0.1, acc_scale=0.1,
        pos_tol=0.015, ori_tol=0.25
    ):
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[PnP] MoveGroup server unavailable.")
            return False

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
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = "link0"
        oc.link_name = "hand_tcp"
        oc.orientation.x, oc.orientation.y = qx, qy
        oc.orientation.z, oc.orientation.w = qz, qw
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

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = options

        handle = self.wait_future(
            self.move_group_client.send_goal_async(goal),
            10.0, "MoveGroup goal")

        if handle is None or not handle.accepted:
            self.get_logger().error("[PnP] MoveGroup goal rejected.")
            return False

        result = self.wait_future(
            handle.get_result_async(), 30.0, "MoveGroup result")
        if result is None:
            return False

        code = result.result.error_code.val
        if code != 1:
            self.get_logger().error(
                f"[PnP] MoveGroup failed: error_code={code}")
        return code == 1

    # Cartesian Z
    def cartesian_z_move(
        self, x, y, start_z, end_z,
        qx, qy, qz, qw, label="[Cartesian Z]"
    ):
        if abs(end_z - start_z) < 0.001:
            return True

        if not self.cartesian_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                f"{label} /compute_cartesian_path unavailable.")
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
        waypoint.position.x, waypoint.position.y, waypoint.position.z = x, y, end_z
        waypoint.orientation.x, waypoint.orientation.y = qx, qy
        waypoint.orientation.z, waypoint.orientation.w = qz, qw
        req.waypoints = [waypoint]

        response = self.wait_future(
            self.cartesian_client.call_async(req),
            20.0, "Cartesian path computation")

        if response is None:
            return False

        self.get_logger().info(
            f"{label} z {start_z:.3f} -> {end_z:.3f}, "
            f"fraction={response.fraction:.3f}")

        if response.fraction < self.cartesian_fraction_threshold:
            self.get_logger().warn(
                f"{label} Fraction too low: "
                f"{response.fraction:.3f} < "
                f"{self.cartesian_fraction_threshold:.2f}")
            return False

        return self.execute_trajectory_with_diagnostics(
            response.solution, label, 30.0)

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
                "[Cartesian XYZ] /compute_cartesian_path unavailable.")
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
            f"step={step:.4f} m, waypoints={len(waypoints)}")

        req = GetCartesianPath.Request()
        req.header.frame_id = "link0"
        req.group_name = "panda_arm"
        req.link_name = "hand_tcp"
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state.is_diff = True
        req.waypoints = waypoints

        response = self.wait_future(
            self.cartesian_client.call_async(req),
            30.0, "Cartesian XYZ path computation")

        if response is None:
            return False

        self.get_logger().info(
            f"[Cartesian XYZ] fraction={response.fraction:.3f}")

        if response.fraction < self.cartesian_fraction_threshold:
            self.get_logger().warn(
                f"[Cartesian XYZ] Path fraction too low: "
                f"{response.fraction:.3f}")
            return False

        self.get_logger().info(
            "[Cartesian XYZ] Path computation SUCCESS. Executing trajectory...")

        return self.execute_trajectory_with_diagnostics(
            response.solution, "[Step 6 Cartesian XYZ]", 30.0)

    # Step 6
    def move_to_pre_place_position(
        self, start_x, start_y, start_z,
        target_x, target_y, target_z,
        qx, qy, qz, qw
    ):
        self.get_logger().info(
            f"[Step 6/9] Pre-place 이동 "
            f"(grasp orientation 유지): "
            f"({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")

        time.sleep(self.step6_state_wait)

        current = self.get_current_arm_joint_state()
        if current is None:
            self.get_logger().warn(
                "[Step 6/9] 현재 /joint_states를 가져오지 못했습니다.")
        else:
            self.get_logger().info(
                f"[Step 6/9] current arm joint state="
                f"{[round(float(v), 6) for v in current]}")

        if self.cartesian_xyz_move(
            start_x, start_y, start_z,
            target_x, target_y, target_z,
            qx, qy, qz, qw
        ):
            self.get_logger().info(
                "[Step 6/9] Pre-place 이동 SUCCESS.")
            return True

        self.get_logger().error(
            "[Step 6/9] Cartesian execution FAILED. "
            "Pose fallback 시도.")

        return self.plan_and_execute_pose(
            target_x, target_y, target_z,
            qx, qy, qz, qw,
            num_attempts=self.fallback_planning_attempts,
            planning_time=self.fallback_planning_time,
            vel_scale=0.10,
            acc_scale=0.10,
            pos_tol=0.015,
            ori_tol=0.05)

    # Z fallback
    def lift_joint_space_fallback(
        self, x, y, target_z, qx, qy, qz, qw
    ):
        self.get_logger().warn(
            f"[Z FALLBACK] Pose fallback: z={target_z:.3f}")
        return self.plan_and_execute_pose(
            x, y, target_z, qx, qy, qz, qw,
            num_attempts=self.fallback_planning_attempts,
            planning_time=self.fallback_planning_time,
            vel_scale=self.fallback_velocity_scale,
            acc_scale=self.fallback_acceleration_scale,
            pos_tol=0.015,
            ori_tol=self.fallback_orientation_tolerance)

    # Step 5 fallback
    def lift_position_downward_fallback(
        self, x, y, target_z, qx, qy, qz, qw
    ):
        self.get_logger().warn(
            f"[LIFT FALLBACK] Position + downward orientation: "
            f"({x:.3f},{y:.3f},{target_z:.3f})")

        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(
                "[LIFT FALLBACK] MoveGroup server unavailable.")
            return False

        req = MotionPlanRequest()
        req.group_name = "panda_arm"
        req.num_planning_attempts = self.fallback_planning_attempts
        req.allowed_planning_time = self.fallback_planning_time
        req.max_velocity_scaling_factor = self.fallback_velocity_scale
        req.max_acceleration_scaling_factor = self.fallback_acceleration_scale
        req.start_state.is_diff = True

        pc = PositionConstraint()
        pc.header.frame_id = "link0"
        pc.link_name = "hand_tcp"

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
        oc.header.frame_id = "link0"
        oc.link_name = "hand_tcp"
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
            10.0, "Lift fallback MoveGroup goal")

        if handle is None or not handle.accepted:
            self.get_logger().error(
                "[LIFT FALLBACK] MoveGroup goal rejected.")
            return False

        result = self.wait_future(
            handle.get_result_async(),
            30.0, "Lift fallback MoveGroup result")

        if result is None:
            return False

        ok = result.result.error_code.val == 1
        self.get_logger().info(
            f"[LIFT FALLBACK] {'SUCCESS' if ok else 'FAILED'}")
        return ok

    # Ready
    def plan_and_execute_named_state(
        self, named_state="ready", num_attempts=5, planning_time=3.0
    ):
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[PnP] MoveGroup server unavailable.")
            return False

        self.get_logger().info(
            f"[PnP] Returning to state: {named_state}")

        req = MotionPlanRequest()
        req.group_name = "panda_arm"
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
            10.0, "Named-state MoveGroup goal")

        if handle is None or not handle.accepted:
            self.get_logger().error("[PnP] Named-state goal rejected.")
            return False

        result = self.wait_future(
            handle.get_result_async(),
            20.0, "Named-state MoveGroup result")

        if result is None:
            return False

        ok = result.result.error_code.val == 1
        self.get_logger().info(
            f"[PnP] {'Successfully returned to' if ok else 'Failed to return to'} "
            f"{named_state}{'' if ok else f': error_code={result.result.error_code.val}'}")
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
            f"({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")

        if start_x is not None and start_y is not None and start_z is not None:
            ok = self.cartesian_xyz_move(
                start_x, start_y, start_z,
                target_x, target_y, target_z,
                qx, qy, qz, qw)
            if ok:
                return True
            self.get_logger().warn(
                f"[MOVE] {description} Cartesian 실패. Fallback 시도.")
        else:
            ok = self.plan_and_execute_pose(
                target_x, target_y, target_z,
                qx, qy, qz, qw)
            if ok:
                return True
            self.get_logger().warn(
                f"[MOVE] {description} Pose planning 실패. Fallback 시도.")

        return self.plan_and_execute_pose(
            target_x, target_y, target_z,
            qx, qy, qz, qw,
            num_attempts=self.fallback_planning_attempts,
            planning_time=self.fallback_planning_time,
            vel_scale=self.fallback_velocity_scale,
            acc_scale=self.fallback_acceleration_scale,
            pos_tol=0.02,
            ori_tol=self.fallback_orientation_tolerance)

    # Gripper
    def control_gripper(self, action):
        action = action.upper()
        if action not in ("OPEN", "CLOSE"):
            self.get_logger().error(f"[GRIPPER] Invalid action: {action}")
            return False, None

        if not self.gripper_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(
                "[GRIPPER] Action server unavailable.")
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
            f"position={goal.command.position:.3f}")

        try:
            handle = self.wait_future(
                self.gripper_client.send_goal_async(goal),
                5.0, f"Gripper {action} goal")

            if handle is None or not handle.accepted:
                self.get_logger().error(
                    f"[GRIPPER] {action} goal rejected.")
                return False, None

            res = self.wait_future(
                handle.get_result_async(),
                10.0, f"Gripper {action} result")

            if res is None:
                return False, None

            obj = res.result
            self.get_logger().info(
                f"[GRIPPER] {action}: position={obj.position:.4f}, "
                f"reached={obj.reached_goal}, stalled={obj.stalled}")

            time.sleep(0.2)
            return obj.reached_goal, obj

        except Exception as e:
            self.get_logger().error(
                f"[GRIPPER] {action} exception: {e}")
            return False, None

    def check_grasp_success(self, result_obj):
        if result_obj is None:
            return False
        pos = result_obj.position
        if self.min_grasp_position_threshold <= pos <= self.gripper_open_position - 0.002:
            self.get_logger().info(
                f"[GRASP CHECK] 물체 파지 성공: finger={pos:.4f}m")
            return True
        self.get_logger().warn(
            f"[GRASP CHECK] 물체 파지 실패: finger={pos:.4f}m")
        return False

    def calculate_place_pose(self):
        center_z = self.table_top_z + self.object_height / 2.0
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
                "[PnP RECOVERY] Opening gripper...")
            self.control_gripper("OPEN")
            time.sleep(0.3)

        self.get_logger().info(
            "[PnP RECOVERY] Returning to READY...")

        try:
            if not self.plan_and_execute_named_state("ready"):
                self.get_logger().warn(
                    "[PnP RECOVERY] Ready fallback attempt...")
                self.plan_and_execute_named_state(
                    "ready", num_attempts=10, planning_time=5.0)
        except Exception as e:
            self.get_logger().error(
                f"[PnP RECOVERY] Exception: {e}")

        self.target_pose = None
        self.object_height = None
        self.target_yaw = 0.0
        self.is_busy = False
        self.state = "IDLE"

    # 9-Step PnP
    def pnp_worker_loop(self):
        while rclpy.ok() and not self.shutdown_requested:
            try:
                if self.state == "TRIGGER_PICK" and self.target_pose is not None:
                    self.state = "PICKING"
                    tx, ty, tz = self.target_pose
                    half_height = self.object_height / 2.0
                    qx, qy, qz, qw = self.yaw_to_grasp_quaternion(self.target_yaw)

                    self.get_logger().info("=" * 60)
                    self.get_logger().info(
                        "[PnP] Starting 9-Step Pick and Place Sequence")
                    self.get_logger().info(
                        f"[PnP] Target=({tx:.3f},{ty:.3f},{tz:.3f}), "
                        f"h={self.object_height:.3f}m, "
                        f"yaw={math.degrees(self.target_yaw):.1f}°")
                    self.get_logger().info("=" * 60)

                    # 1. Open
                    self.get_logger().info("[Step 1/9] Gripper OPEN")
                    ok, _ = self.control_gripper("OPEN")
                    if not ok:
                        self.get_logger().warn("[Step 1/9] OPEN retry")
                        time.sleep(0.3)
                        ok, _ = self.control_gripper("OPEN")
                        if not ok:
                            self.reset_after_failure(
                                "Step 1 Gripper OPEN failed.")
                            continue

                    # 2. Pre-grasp
                    pre_grasp_z = tz + half_height + self.pre_grasp_z_offset
                    if not self.move_to_pose(
                        tx, ty, pre_grasp_z, qx, qy, qz, qw,
                        description="[Step 2/9] Pre-grasp"):
                        self.reset_after_failure(
                            "Step 2 Pre-grasp failed.")
                        continue

                    # 3. Descent
                    grasp_z = (
                        tz if half_height <= self.max_grasp_depth
                        else tz + half_height - self.max_grasp_depth)

                    self.get_logger().info(
                        f"[Step 3/9] Grasp 하강: "
                        f"{pre_grasp_z:.3f} -> {grasp_z:.3f}")

                    ok = self.cartesian_z_move(
                        tx, ty, pre_grasp_z, grasp_z,
                        qx, qy, qz, qw,
                        "[Step 3 Cartesian Z]")

                    if not ok:
                        self.get_logger().warn(
                            "[Step 3/9] Cartesian 실패. Pose fallback")
                        ok = self.lift_joint_space_fallback(
                            tx, ty, grasp_z, qx, qy, qz, qw)

                    if not ok:
                        self.reset_after_failure(
                            "Step 3 Grasp descent failed.")
                        continue

                    # 4. Close / grasp check
                    self.get_logger().info(
                        "[Step 4/9] Gripper CLOSE & grasp check")
                    cmd_ok, result_obj = self.control_gripper("CLOSE")
                    grasped = self.check_grasp_success(result_obj) if cmd_ok else False

                    if not grasped:
                        self.get_logger().warn(
                            "[Step 4/9] Grasp retry")
                        self.control_gripper("OPEN")
                        time.sleep(0.3)
                        cmd_ok, result_obj = self.control_gripper("CLOSE")
                        grasped = self.check_grasp_success(result_obj) if cmd_ok else False

                        if not grasped:
                            self.reset_after_failure(
                                "Step 4 Gripper CLOSE/grasp check failed.",
                                need_open_gripper=True)
                            continue

                    # 5. Lift
                    after_grasp_z = tz + half_height + self.lift_z_offset
                    self.get_logger().info(
                        f"[Step 5/9] Single Cartesian Lift: "
                        f"{grasp_z:.3f} -> {after_grasp_z:.3f}")

                    ok = self.cartesian_z_move(
                        tx, ty, grasp_z, after_grasp_z,
                        qx, qy, qz, qw,
                        "[Step 5 Cartesian Lift]")

                    if not ok:
                        self.get_logger().warn(
                            "[Step 5/9] Cartesian Lift 실패. Fallback")
                        ok = self.lift_position_downward_fallback(
                            tx, ty, after_grasp_z,
                            qx, qy, qz, qw)

                    if not ok:
                        self.reset_after_failure(
                            "Step 5 After-grasp lift failed.",
                            need_open_gripper=True)
                        continue

                    self.get_logger().info(
                        "[Step 5/9] After-grasp Lift SUCCESS")

                    # 6. Pre-place
                    px, py, pz = self.calculate_place_pose()
                    pre_place_z = pz + self.post_place_z_offset

                    ok = self.move_to_pre_place_position(
                        tx, ty, after_grasp_z,
                        px, py, pre_place_z,
                        qx, qy, qz, qw)

                    if not ok:
                        self.reset_after_failure(
                            "Step 6 Pre-place failed.",
                            need_open_gripper=True)
                        continue

                    self.get_logger().info(
                        "[Step 6/9] Pre-place 완료")

                    # 7. Place descent / open
                    self.get_logger().info(
                        f"[Step 7/9] Place 하강: "
                        f"{pre_place_z:.3f} -> {pz:.3f}")

                    ok = self.cartesian_z_move(
                        px, py, pre_place_z, pz,
                        qx, qy, qz, qw,
                        "[Step 7 Cartesian Z]")

                    if not ok:
                        self.get_logger().warn(
                            "[Step 7/9] Cartesian 실패. Pose fallback")
                        ok = self.lift_joint_space_fallback(
                            px, py, pz, qx, qy, qz, qw)

                    if not ok:
                        self.reset_after_failure(
                            "Step 7 Place descent failed.",
                            need_open_gripper=True)
                        continue

                    self.get_logger().info(
                        "[Step 7/9] Place 도착 -> Gripper OPEN")

                    ok, _ = self.control_gripper("OPEN")
                    if not ok:
                        self.get_logger().warn(
                            "[Step 7/9] OPEN retry")
                        time.sleep(0.3)
                        ok, _ = self.control_gripper("OPEN")
                        if not ok:
                            self.reset_after_failure(
                                "Step 7 Gripper OPEN failed.")
                            continue

                    time.sleep(0.5)

                    # 8. Retract
                    after_place_z = pre_place_z
                    self.get_logger().info(
                        f"[Step 8/9] Retract: "
                        f"{pz:.3f} -> {after_place_z:.3f}")

                    ok = self.cartesian_z_move(
                        px, py, pz, after_place_z,
                        qx, qy, qz, qw,
                        "[Step 8 Cartesian Z]")

                    if not ok:
                        self.get_logger().warn(
                            "[Step 8/9] Retract 실패. Pose fallback")
                        ok = self.lift_joint_space_fallback(
                            px, py, after_place_z,
                            qx, qy, qz, qw)

                    if not ok:
                        self.reset_after_failure(
                            "Step 8 Retract failed.")
                        continue

                    # 9. Ready
                    self.get_logger().info("[Step 9/9] Ready 복귀")
                    ok = self.plan_and_execute_named_state("ready")

                    if not ok:
                        self.get_logger().warn(
                            "[Step 9/9] Ready fallback")
                        ok = self.plan_and_execute_named_state(
                            "ready", num_attempts=10, planning_time=5.0)

                    if not ok:
                        self.reset_after_failure(
                            "Step 9 Return to ready failed.")
                        continue

                    self.get_logger().info("=" * 60)
                    self.get_logger().info(
                        "[PnP SUCCESS] 9-Step Pick & Place Completed!")
                    self.get_logger().info("=" * 60)

                    self.target_pose = None
                    self.object_height = None
                    self.target_yaw = 0.0
                    self.is_busy = False
                    self.state = "IDLE"

                time.sleep(0.05)

            except Exception as e:
                self.get_logger().error(
                    f"[PnP WORKER] Exception: {e}")
                self.reset_after_failure(f"Worker exception: {e}")

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


if __name__ == "__main__":
    main()