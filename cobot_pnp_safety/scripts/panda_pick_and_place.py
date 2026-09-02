#!/usr/bin/env python3
"""
MoveIt 2 기반 Franka Emika Panda 3D Vision Pick & Place Controller Node

Subscribes:
  - /target_object_pose (geometry_msgs/PoseStamped)
      pose.position : 물체 중심 좌표 (link0 기준)

Actions:
  - /move_action
  - /panda_hand_controller/gripper_action
"""

import time
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped, Pose

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, PlanningOptions, JointConstraint

from shape_msgs.msg import SolidPrimitive
from control_msgs.action import GripperCommand


class PandaMoveItPickAndPlace(Node):
    def __init__(self):
        super().__init__("panda_moveit_pnp_node")
        self.cb_group = ReentrantCallbackGroup()

        self.state = "IDLE"
        self.target_pose = None
        self.target_height = None
        self.target_size_x = None
        self.target_size_y = None
        self.is_busy = False
        self.shutdown_requested = False

        self.home_qpos = [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.57079, 0.785398]

        self.place_x = 0.0
        self.place_y = -0.55
        self.table_top_z = 0.44

        self.pre_grasp_z_offset = 0.12
        self.post_place_z_offset = 0.12
        self.lift_z_offset = 0.17

        self.tcp_to_fingertip = 0.1100 - (0.0584 + 0.0445)

        self.move_group_client = ActionClient(self, MoveGroup, "/move_action", callback_group=self.cb_group)
        self.gripper_client = ActionClient(self, GripperCommand, "/panda_hand_controller/gripper_action", callback_group=self.cb_group)

        self.target_sub = self.create_subscription(PoseStamped, "/target_object_pose", self.target_pose_callback, 10, callback_group=self.cb_group)

        self.get_logger().info("[PnP INIT] Franka Panda MoveIt2 Pick and Place Controller Ready.")
        self.get_logger().info(f"[PnP INIT] TCP-to-fingertip offset: {self.tcp_to_fingertip * 1000:.1f} mm")
        self.get_logger().info(f"[PnP INIT] Place table top Z: {self.table_top_z:.3f} m")
        self.get_logger().info(f"[PnP INIT] Place XY: ({self.place_x:.3f}, {self.place_y:.3f})")

        self.worker_thread = threading.Thread(target=self.pnp_worker_loop, daemon=True)
        self.worker_thread.start()

    def target_pose_callback(self, msg: PoseStamped):
        if self.is_busy or self.state != "IDLE":
            return

        if msg.header.frame_id != "link0":
            self.get_logger().warn(f"[VISION] Expected frame 'link0' but received '{msg.header.frame_id}'")
            return

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)

        if not np.isfinite([x, y, z]).all():
            self.get_logger().warn("[VISION] Invalid target position.")
            return

        estimated_height = 2.0 * (z - self.table_top_z)

        if estimated_height <= 0.0:
            self.get_logger().warn(f"[VISION] Invalid estimated object height: {estimated_height:.3f} m")
            return

        if estimated_height > 0.40:
            self.get_logger().warn(f"[VISION] Object height too large: {estimated_height:.3f} m")
            return

        self.is_busy = True
        self.target_pose = np.array([x, y, z], dtype=np.float64)
        self.target_height = estimated_height
        self.target_size_x = None
        self.target_size_y = None
        self.state = "TRIGGER_PICK"

        self.get_logger().info(f"[VISION TARGET DETECTED] Object Center(link0): X={x:.3f}, Y={y:.3f}, Z={z:.3f} | Estimated Height={estimated_height:.3f} m")

    def wait_future(self, future, timeout_sec):
        event = threading.Event()
        result_holder = {"result": None, "exception": None}

        def done_callback(done_future):
            try:
                result_holder["result"] = done_future.result()
            except Exception as e:
                result_holder["exception"] = e
            finally:
                event.set()

        future.add_done_callback(done_callback)

        if not event.wait(timeout=timeout_sec):
            self.get_logger().warn("[ACTION] Future timeout.")
            return None

        if result_holder["exception"] is not None:
            self.get_logger().error(f"[ACTION] Future exception: {result_holder['exception']}")
            return None

        return result_holder["result"]

    def plan_and_execute_pose(self, x: float, y: float, z: float, qx=1.0, qy=0.0, qz=0.0, qw=0.0) -> bool:
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("MoveGroup Action Server not available!")
            return False

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()

        req.group_name = "panda_arm"
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5
        req.start_state.is_diff = True

        constraints = Constraints()

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "link0"
        pos_constraint.link_name = "hand_tcp"
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.03]

        bv = BoundingVolume()
        bv.primitives.append(sphere)

        target_p = Pose()
        target_p.position.x = float(x)
        target_p.position.y = float(y)
        target_p.position.z = float(z)
        target_p.orientation.w = 1.0

        bv.primitive_poses.append(target_p)
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0
        constraints.position_constraints.append(pos_constraint)

        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = "link0"
        ori_constraint.link_name = "hand_tcp"
        ori_constraint.orientation.x = float(qx)
        ori_constraint.orientation.y = float(qy)
        ori_constraint.orientation.z = float(qz)
        ori_constraint.orientation.w = float(qw)
        ori_constraint.absolute_x_axis_tolerance = 0.5
        ori_constraint.absolute_y_axis_tolerance = 0.5
        ori_constraint.absolute_z_axis_tolerance = 3.14
        ori_constraint.weight = 0.8
        constraints.orientation_constraints.append(ori_constraint)

        req.goal_constraints.append(constraints)
        goal.request = req

        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 5

        self.get_logger().info(f"[PLANNING] Moving TCP to ({x:.3f}, {y:.3f}, {z:.3f})...")

        try:
            future = self.move_group_client.send_goal_async(goal)
            goal_handle = self.wait_future(future, 10.0)
        except Exception as e:
            self.get_logger().error(f"[ACTION] Failed to send MoveGroup goal: {e}")
            return False

        if goal_handle is None:
            self.get_logger().warn("[PLANNING] MoveGroup goal response timeout.")
            return False

        if not goal_handle.accepted:
            self.get_logger().warn("[PLANNING] Goal rejected by MoveGroup.")
            return False

        try:
            result_future = goal_handle.get_result_async()
            result_wrapper = self.wait_future(result_future, 20.0)
        except Exception as e:
            self.get_logger().error(f"[ACTION] Failed to get MoveGroup result: {e}")
            return False

        if result_wrapper is None:
            self.get_logger().warn("[EXECUTION] MoveGroup result timeout.")
            return False

        res = result_wrapper.result

        if res.error_code.val == 1:
            self.get_logger().info("[EXECUTION] Motion completed successfully.")
            return True

        self.get_logger().warn(f"[EXECUTION] Motion failed with error code: {res.error_code.val}")
        return False

    def plan_and_execute_named_state(self, named_state: str) -> bool:
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("MoveGroup Action Server not available!")
            return False

        if named_state not in ["ready", "home"]:
            self.get_logger().error(f"Unknown named state: {named_state}")
            return False

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()

        req.group_name = "panda_arm"
        req.num_planning_attempts = 5
        req.allowed_planning_time = 3.0
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5

        constraints = Constraints()

        joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        for jname, value in zip(joints, self.home_qpos):
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = float(value)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        req.goal_constraints.append(constraints)
        goal.request = req

        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        self.get_logger().info(f"[PLANNING] Moving to MuJoCo initial pose: '{named_state}'...")

        try:
            future = self.move_group_client.send_goal_async(goal)
            goal_handle = self.wait_future(future, 10.0)
        except Exception as e:
            self.get_logger().error(f"[ACTION] Failed to send Home goal: {e}")
            return False

        if goal_handle is None:
            self.get_logger().warn("[PLANNING] Home goal response timeout.")
            return False

        if not goal_handle.accepted:
            self.get_logger().warn("[PLANNING] Home goal rejected.")
            return False

        try:
            result_future = goal_handle.get_result_async()
            result_wrapper = self.wait_future(result_future, 15.0)
        except Exception as e:
            self.get_logger().error(f"[ACTION] Failed to get Home result: {e}")
            return False

        if result_wrapper is None:
            self.get_logger().warn("[PLANNING] Home result timeout.")
            return False

        res = result_wrapper.result

        if res.error_code.val == 1:
            self.get_logger().info("[EXECUTION] Returned to MuJoCo initial pose.")
            return True

        self.get_logger().warn(f"[EXECUTION] Home motion failed: {res.error_code.val}")
        return False

    def control_gripper(self, action: str):
        action = action.upper()

        if not self.gripper_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn("[GRIPPER] Action server not available.")
            time.sleep(0.5)
            return False

        goal = GripperCommand.Goal()
        goal.command.position = 0.04 if action == "OPEN" else 0.00
        goal.command.max_effort = 20.0

        try:
            future = self.gripper_client.send_goal_async(goal)
            goal_handle = self.wait_future(future, 3.0)

            if goal_handle is None:
                self.get_logger().warn("[GRIPPER] Goal response timeout.")
                return False

            if not goal_handle.accepted:
                self.get_logger().warn("[GRIPPER] Goal rejected.")
                return False

            result_future = goal_handle.get_result_async()
            result_wrapper = self.wait_future(result_future, 5.0)

            if result_wrapper is None:
                self.get_logger().warn("[GRIPPER] Result timeout.")
                return False

            result = result_wrapper.result

            self.get_logger().info(f"[GRIPPER] {action} completed: position={result.position:.6f}, reached_goal={result.reached_goal}")

            if not result.reached_goal:
                self.get_logger().error(f"[GRIPPER] {action} failed: gripper did not reach target.")
                return False

            time.sleep(0.2)
            return True

        except Exception as e:
            self.get_logger().error(f"[GRIPPER] Action error: {e}")
            return False

    def calculate_place_pose(self):
        if self.target_height is None:
            self.get_logger().error("[PLACE] Target object height is not available.")
            return None

        object_center_z = self.table_top_z + self.target_height / 2.0
        place_tcp_z = object_center_z - self.tcp_to_fingertip

        self.get_logger().info(f"[PLACE CALC] Table Z={self.table_top_z:.4f}, Object Height={self.target_height:.4f}, Object Center Z={object_center_z:.4f}, TCP Place Z={place_tcp_z:.4f}")

        return self.place_x, self.place_y, place_tcp_z

    def reset_after_failure(self, reason: str):
        self.get_logger().error(reason)
        self.plan_and_execute_named_state("ready")
        self.target_pose = None
        self.target_height = None
        self.target_size_x = None
        self.target_size_y = None
        self.is_busy = False
        self.state = "IDLE"

    def pnp_worker_loop(self):
        grasp_orientation = (1.0, 0.0, 0.0, 0.0)

        while rclpy.ok() and not self.shutdown_requested:
            if self.state == "TRIGGER_PICK" and self.target_pose is not None:
                self.is_busy = True
                self.state = "PICKING"

                tx, ty, tz = self.target_pose

                self.get_logger().info("========== [STARTING PICK & PLACE SEQUENCE] ==========")

                self.get_logger().info("[1/8] Opening Gripper...")
                if not self.control_gripper("OPEN"):
                    self.reset_after_failure("Gripper OPEN failed.")
                    continue

                time.sleep(0.2)

                self.get_logger().info("[2/8] Approaching Pre-Grasp Pose...")
                pre_grasp_z = tz + self.pre_grasp_z_offset

                if not self.plan_and_execute_pose(tx, ty, pre_grasp_z, *grasp_orientation):
                    self.reset_after_failure("Pre-grasp approach failed! Returning to initial pose...")
                    continue

                self.get_logger().info("[3/8] Lowering to Grasp Pose...")

                grasp_z = tz - self.tcp_to_fingertip

                self.get_logger().info(f"[GRASP] Object Center Z={tz:.4f}, TCP Grasp Z={grasp_z:.4f}, Offset={self.tcp_to_fingertip * 1000:.1f} mm")

                if not self.plan_and_execute_pose(tx, ty, grasp_z, *grasp_orientation):
                    self.reset_after_failure("Grasp approach failed.")
                    continue

                self.get_logger().info("[4/8] Closing Gripper to Grasp Object...")

                if not self.control_gripper("CLOSE"):
                    self.reset_after_failure("Gripper CLOSE failed.")
                    continue

                time.sleep(0.2)

                self.get_logger().info("[5/8] Lifting Object...")
                lift_z = tz + self.lift_z_offset

                if not self.plan_and_execute_pose(tx, ty, lift_z, *grasp_orientation):
                    self.reset_after_failure("Lift failed.")
                    continue

                self.get_logger().info("[6/8] Moving to Place Location...")

                place_pose = self.calculate_place_pose()

                if place_pose is None:
                    self.reset_after_failure("Failed to calculate Place pose.")
                    continue

                px, py, pz = place_pose
                pre_place_z = pz + self.post_place_z_offset

                self.get_logger().info(f"[PRE-PLACE] TCP target: ({px:.3f}, {py:.3f}, {pre_place_z:.3f})")

                if not self.plan_and_execute_pose(px, py, pre_place_z, *grasp_orientation):
                    self.reset_after_failure("Pre-place motion failed.")
                    continue

                self.get_logger().info("[7/8] Lowering to Place Height & Releasing Object...")
                self.get_logger().info(f"[PLACE] TCP target: ({px:.3f}, {py:.3f}, {pz:.3f})")

                if not self.plan_and_execute_pose(px, py, pz, *grasp_orientation):
                    self.reset_after_failure("Place motion failed.")
                    continue

                if not self.control_gripper("OPEN"):
                    self.reset_after_failure("Gripper OPEN at place failed.")
                    continue

                time.sleep(0.6)

                self.get_logger().info("[8/8] Retracting to Ready Pose...")

                if not self.plan_and_execute_pose(px, py, pre_place_z, *grasp_orientation):
                    self.get_logger().warn("Retract motion failed. Continuing to Ready Pose.")

                self.plan_and_execute_named_state("ready")

                self.get_logger().info("========== [PICK & PLACE FINISHED SUCCESSFULLY] ==========")

                self.target_pose = None
                self.target_height = None
                self.target_size_x = None
                self.target_size_y = None
                self.is_busy = False
                self.state = "IDLE"

            time.sleep(0.1)

    def shutdown(self):
        self.shutdown_requested = True

        if self.worker_thread.is_alive() and threading.current_thread() is not self.worker_thread:
            self.worker_thread.join(timeout=2.0)


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

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()