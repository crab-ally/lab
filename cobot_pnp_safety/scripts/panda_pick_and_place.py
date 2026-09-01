#!/usr/bin/env python3
"""
MoveIt 2 기반 Franka Emika Panda 3D Vision Pick & Place Controller Node

Franka Emika Panda 사양
  - 최대 도달 반경: 855 mm (Base 관절 중심부터 End-effector 연결부까지)
  - 자유도 (DOF): 7 DOF
  - 가Payload (가용 하중): 3 kg
  - 파지 가능 범위 (외부 파지): 지름 0mm ~ 80mm 사이의 물체

Subscribes:
  - /target_object_pose (geometry_msgs/PoseStamped): 3D 비전 노드로부터 수신한 물체 중심 좌표 (link0 기준)

Actions / Commands:
  - MoveGroup Action (/move_action)
  - Gripper Action (/panda_hand_controller/gripper_action)
  - Gripper Topic (/panda_gripper_cmd)
"""

import time
from pathlib import Path
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import String

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    PlanningOptions,
    JointConstraint,
)

from shape_msgs.msg import SolidPrimitive
from control_msgs.action import GripperCommand


class PandaMoveItPickAndPlace(Node):

    def __init__(self):
        super().__init__("panda_moveit_pnp_node")

        self.cb_group = ReentrantCallbackGroup()

        # ==============================================================
        # 1. 상태
        # ==============================================================

        self.state = "IDLE"
        self.target_pose = None

        # PnP 작업 진행 여부
        # True이면 새로운 /target_object_pose는 무시
        self.is_busy = False

        self.shutdown_requested = False

        # ==============================================================
        # 2. MuJoCo Panda 초기 자세
        # ==============================================================

        self.home_qpos = [
            0.0,
            -0.785398,
            0.0,
            -2.35619,
            0.0,
            1.57079,
            0.785398
        ]

        # ==============================================================
        # 3. Pick & Place 설정
        # ==============================================================

        self.place_location = np.array([
            0.0,
            -0.55,
            0.50
        ])

        self.pre_grasp_z_offset = 0.12
        self.post_place_z_offset = 0.12

        # ==============================================================
        # 4. MoveIt Action Client
        # ==============================================================

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
            callback_group=self.cb_group
        )

        # ==============================================================
        # 5. Gripper Action Client
        # ==============================================================

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            "/panda_hand_controller/gripper_action",
            callback_group=self.cb_group
        )

        self.gripper_topic_pub = self.create_publisher(
            String,
            "/panda_gripper_cmd",
            10
        )

        # ==============================================================
        # 6. Vision Target
        # ==============================================================

        self.target_sub = self.create_subscription(
            PoseStamped,
            "/target_object_pose",
            self.target_pose_callback,
            10,
            callback_group=self.cb_group
        )

        self.get_logger().info(
            "[PnP INIT] Franka Panda MoveIt2 Pick and Place Controller Ready."
        )

        # ==============================================================
        # 7. PnP Worker
        # ==============================================================

        self.worker_thread = threading.Thread(
            target=self.pnp_worker_loop,
            daemon=True
        )

        self.worker_thread.start()

    # ==================================================================
    # Vision Target Callback
    # ==================================================================

    def target_pose_callback(self,msg: PoseStamped):
        if self.is_busy:
            self.get_logger().info("[VISION] PnP 작업 진행 중 - 새 Target 무시.")
            return

        if self.state != "IDLE":
            return

        self.is_busy=True
        self.target_pose=np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        self.state="TRIGGER_PICK"

        self.get_logger().info(f"[VISION TARGET DETECTED] X={self.target_pose[0]:.3f}, Y={self.target_pose[1]:.3f}, Z={self.target_pose[2]:.3f}")

    # ==================================================================
    # Future 대기
    # ==================================================================

    def wait_future(self, future, timeout_sec):

        event = threading.Event()

        result_holder = {
            "result": None,
            "exception": None
        }

        def done_callback(done_future):

            try:
                result_holder["result"] = done_future.result()

            except Exception as e:
                result_holder["exception"] = e

            finally:
                event.set()

        future.add_done_callback(done_callback)

        if not event.wait(timeout=timeout_sec):

            self.get_logger().warn(
                "[ACTION] Future timeout."
            )

            return None

        if result_holder["exception"] is not None:

            self.get_logger().error(
                f"[ACTION] Future exception: "
                f"{result_holder['exception']}"
            )

            return None

        return result_holder["result"]

    # ==================================================================
    # MoveIt Pose
    # ==================================================================

    def plan_and_execute_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx=1.0,
        qy=0.0,
        qz=0.0,
        qw=0.0
    ) -> bool:

        if not self.move_group_client.wait_for_server(
            timeout_sec=3.0
        ):

            self.get_logger().error(
                "MoveGroup Action Server not available!"
            )

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

        # --------------------------------------------------------------
        # Position Constraint
        # --------------------------------------------------------------

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

        target_p.orientation.x = 0.0
        target_p.orientation.y = 0.0
        target_p.orientation.z = 0.0
        target_p.orientation.w = 1.0

        bv.primitive_poses.append(target_p)

        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0

        constraints.position_constraints.append(
            pos_constraint
        )

        # --------------------------------------------------------------
        # Orientation Constraint
        # --------------------------------------------------------------

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

        constraints.orientation_constraints.append(
            ori_constraint
        )

        req.goal_constraints.append(constraints)

        goal.request = req

        goal.planning_options = PlanningOptions()

        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 5

        self.get_logger().info(
            f"[PLANNING] Moving TCP to "
            f"({x:.3f}, {y:.3f}, {z:.3f})..."
        )

        # --------------------------------------------------------------
        # Action Goal 전송
        # --------------------------------------------------------------

        try:

            future = self.move_group_client.send_goal_async(
                goal
            )

            goal_handle = self.wait_future(
                future,
                10.0
            )

        except Exception as e:

            self.get_logger().error(
                f"[ACTION] Failed to send MoveGroup goal: {e}"
            )

            return False

        if goal_handle is None:

            self.get_logger().warn(
                "[PLANNING] MoveGroup goal response timeout."
            )

            return False

        if not goal_handle.accepted:

            self.get_logger().warn(
                "[PLANNING] Goal rejected by MoveGroup."
            )

            return False

        # --------------------------------------------------------------
        # 실행 결과 대기
        # --------------------------------------------------------------

        try:

            result_future = goal_handle.get_result_async()

            result_wrapper = self.wait_future(
                result_future,
                20.0
            )

        except Exception as e:

            self.get_logger().error(
                f"[ACTION] Failed to get MoveGroup result: {e}"
            )

            return False

        if result_wrapper is None:

            self.get_logger().warn(
                "[EXECUTION] MoveGroup result timeout."
            )

            return False

        res = result_wrapper.result

        if res.error_code.val == 1:

            self.get_logger().info(
                "[EXECUTION] Motion completed successfully."
            )

            return True

        self.get_logger().warn(
            f"[EXECUTION] Motion failed with error code: "
            f"{res.error_code.val}"
        )

        return False

    # ==================================================================
    # MoveIt Home / Ready
    # ==================================================================

    def plan_and_execute_named_state(
        self,
        named_state: str
    ) -> bool:

        if not self.move_group_client.wait_for_server(
            timeout_sec=3.0
        ):

            self.get_logger().error(
                "MoveGroup Action Server not available!"
            )

            return False

        if named_state not in [
            "ready",
            "home"
        ]:

            self.get_logger().error(
                f"Unknown named state: {named_state}"
            )

            return False

        goal = MoveGroup.Goal()

        req = MotionPlanRequest()

        req.group_name = "panda_arm"
        req.num_planning_attempts = 5
        req.allowed_planning_time = 3.0
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5

        constraints = Constraints()

        joints = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7"
        ]

        for jname, value in zip(
            joints,
            self.home_qpos
        ):

            jc = JointConstraint()

            jc.joint_name = jname
            jc.position = float(value)

            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0

            constraints.joint_constraints.append(
                jc
            )

        req.goal_constraints.append(
            constraints
        )

        goal.request = req

        goal.planning_options = PlanningOptions()

        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        self.get_logger().info(
            f"[PLANNING] Moving to MuJoCo initial pose: "
            f"'{named_state}'..."
        )

        # --------------------------------------------------------------
        # Goal 전송
        # --------------------------------------------------------------

        try:

            future = self.move_group_client.send_goal_async(
                goal
            )

            goal_handle = self.wait_future(
                future,
                10.0
            )

        except Exception as e:

            self.get_logger().error(
                f"[ACTION] Failed to send Home goal: {e}"
            )

            return False

        if goal_handle is None:

            self.get_logger().warn(
                "[PLANNING] Home goal response timeout."
            )

            return False

        if not goal_handle.accepted:

            self.get_logger().warn(
                "[PLANNING] Home goal rejected."
            )

            return False

        # --------------------------------------------------------------
        # 결과 대기
        # --------------------------------------------------------------

        try:

            result_future = goal_handle.get_result_async()

            result_wrapper = self.wait_future(
                result_future,
                15.0
            )

        except Exception as e:

            self.get_logger().error(
                f"[ACTION] Failed to get Home result: {e}"
            )

            return False

        if result_wrapper is None:

            self.get_logger().warn(
                "[PLANNING] Home result timeout."
            )

            return False

        res = result_wrapper.result

        if res.error_code.val == 1:

            self.get_logger().info(
                "[EXECUTION] Returned to MuJoCo initial pose."
            )

            return True

        self.get_logger().warn(
            f"[EXECUTION] Home motion failed: "
            f"{res.error_code.val}"
        )

        return False

    # ==================================================================
    # Gripper
    # ==================================================================

    def control_gripper(
        self,
        action: str
    ):

        action = action.upper()

        msg = String()

        msg.data = (
            "open"
            if action == "OPEN"
            else "close"
        )

        self.gripper_topic_pub.publish(
            msg
        )

        if self.gripper_client.wait_for_server(
            timeout_sec=0.5
        ):

            goal = GripperCommand.Goal()

            goal.command.position = (
                0.04
                if action == "OPEN"
                else 0.00
            )

            goal.command.max_effort = 20.0

            try:

                future = self.gripper_client.send_goal_async(
                    goal
                )

                goal_handle = self.wait_future(
                    future,
                    3.0
                )

                if goal_handle is None:

                    self.get_logger().warn(
                        "[GRIPPER] Goal response timeout."
                    )

                elif not goal_handle.accepted:

                    self.get_logger().warn(
                        "[GRIPPER] Goal rejected."
                    )

            except Exception as e:

                self.get_logger().warn(
                    f"[GRIPPER] Action error: {e}"
                )

        time.sleep(0.5)

        self.get_logger().info(
            f"[GRIPPER] Executed {action} command."
        )

    # ==================================================================
    # 실패 처리
    # ==================================================================

    def reset_after_failure(
        self,
        reason: str
    ):

        self.get_logger().error(
            reason
        )

        self.plan_and_execute_named_state(
            "ready"
        )

        self.target_pose = None
        self.is_busy = False
        self.state = "IDLE"

    # ==================================================================
    # Pick & Place FSM
    # ==================================================================

    def pnp_worker_loop(self):

        grasp_orientation = (
            1.0,
            0.0,
            0.0,
            0.0
        )

        while rclpy.ok() and not self.shutdown_requested:

            if (
                self.state == "TRIGGER_PICK"
                and self.target_pose is not None
            ):

                # ======================================================
                # 중요:
                # 작업 시작 즉시 busy=True
                # 이후 들어오는 모든 Target은 callback에서 무시됨
                # ======================================================

                self.is_busy = True
                self.state = "PICKING"

                tx, ty, tz = self.target_pose

                self.get_logger().info(
                    "========== [STARTING PICK & PLACE SEQUENCE] =========="
                )

                # ======================================================
                # 1. Gripper OPEN
                # ======================================================

                self.control_gripper(
                    "OPEN"
                )

                time.sleep(0.2)

                # ======================================================
                # 2. Pre-Grasp
                # ======================================================

                self.get_logger().info(
                    "[1/7] Approaching Pre-Grasp Pose..."
                )

                pre_grasp_z = (
                    tz +
                    self.pre_grasp_z_offset
                )

                if not self.plan_and_execute_pose(
                    tx,
                    ty,
                    pre_grasp_z,
                    *grasp_orientation
                ):

                    self.reset_after_failure(
                        "Pre-grasp approach failed! "
                        "Returning to initial pose..."
                    )

                    continue

                # ======================================================
                # 3. Grasp
                # ======================================================

                self.get_logger().info(
                    "[2/7] Lowering to Grasp Pose..."
                )

                grasp_z = max(
                    0.04,
                    tz + 0.02
                )

                if not self.plan_and_execute_pose(
                    tx,
                    ty,
                    grasp_z,
                    *grasp_orientation
                ):

                    self.reset_after_failure(
                        "Grasp approach failed."
                    )

                    continue

                # ======================================================
                # 4. Close Gripper
                # ======================================================

                self.get_logger().info(
                    "[3/7] Closing Gripper "
                    "to Grasp Object..."
                )

                self.control_gripper(
                    "CLOSE"
                )

                time.sleep(0.8)

                # ======================================================
                # 5. Lift
                # ======================================================

                self.get_logger().info(
                    "[4/7] Lifting Object..."
                )

                lift_z = (
                    tz +
                    self.pre_grasp_z_offset +
                    0.05
                )

                if not self.plan_and_execute_pose(
                    tx,
                    ty,
                    lift_z,
                    *grasp_orientation
                ):

                    self.reset_after_failure(
                        "Lift failed."
                    )

                    continue

                # ======================================================
                # 6. Pre-Place
                # ======================================================

                self.get_logger().info(
                    "[5/7] Moving to Place Location..."
                )

                px, py, pz = self.place_location

                pre_place_z = (
                    pz +
                    self.post_place_z_offset
                )

                if not self.plan_and_execute_pose(
                    px,
                    py,
                    pre_place_z,
                    *grasp_orientation
                ):

                    self.reset_after_failure(
                        "Pre-place motion failed."
                    )

                    continue

                # ======================================================
                # 7. Place
                # ======================================================

                self.get_logger().info(
                    "[6/7] Lowering to Place Height "
                    "& Releasing Object..."
                )

                if not self.plan_and_execute_pose(
                    px,
                    py,
                    pz,
                    *grasp_orientation
                ):

                    self.reset_after_failure(
                        "Place motion failed."
                    )

                    continue

                self.control_gripper(
                    "OPEN"
                )

                time.sleep(0.6)

                # ======================================================
                # 8. Retract
                # ======================================================

                self.get_logger().info(
                    "[7/7] Retracting to Ready Pose..."
                )

                self.plan_and_execute_pose(
                    px,
                    py,
                    pre_place_z,
                    *grasp_orientation
                )

                self.plan_and_execute_named_state(
                    "ready"
                )

                self.get_logger().info(
                    "========== "
                    "[PICK & PLACE FINISHED SUCCESSFULLY]"
                    " =========="
                )

                # ======================================================
                # 작업 완료
                # ======================================================

                self.target_pose = None
                self.is_busy = False
                self.state = "IDLE"

            time.sleep(0.1)

    # ==================================================================
    # Shutdown
    # ==================================================================

    def shutdown(self):

        self.shutdown_requested = True

        if (
            self.worker_thread.is_alive()
            and threading.current_thread()
            is not self.worker_thread
        ):

            self.worker_thread.join(
                timeout=2.0
            )


# ======================================================================
# Main
# ======================================================================

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