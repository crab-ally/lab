#!/usr/bin/env python3
"""
MoveIt 2 기반 Franka Emika Panda 3D Vision Pick & Place Controller Node

Subscribes:
  - /target_object_pose (geometry_msgs/PoseStamped)

Actions:
  - /move_action
  - /panda_hand_controller/gripper_action
"""

import time,math,threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped,Pose
from moveit_msgs.action import MoveGroup,ExecuteTrajectory
from moveit_msgs.msg import MotionPlanRequest,Constraints,PositionConstraint,OrientationConstraint,BoundingVolume,PlanningOptions,JointConstraint
from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import GripperCommand


class PandaMoveItPickAndPlace(Node):
    def __init__(self):
        super().__init__("panda_moveit_pnp_node")
        self.cb_group=ReentrantCallbackGroup()

        self.state="IDLE"
        self.target_pose=None
        self.target_height=None
        self.target_yaw=0.0
        self.is_busy=False
        self.shutdown_requested=False

        # ============================================================
        # Panda initial pose
        # ============================================================

        self.home_qpos=[
            0.0,
            -0.785398,
            0.0,
            -2.35619,
            0.0,
            1.57079,
            0.785398
        ]

        # ============================================================
        # Place position
        # ============================================================

        self.place_x=0.0
        self.place_y=-0.55
        self.table_top_z=0.44

        # ============================================================
        # Motion offsets
        # ============================================================

        self.pre_grasp_z_offset=0.12
        self.post_place_z_offset=0.12
        self.lift_z_offset=0.17
        self.lift_clearance=0.10

        # ============================================================
        # Gripper geometry
        # ============================================================

        self.gripper_clearance=0.100
        self.grasp_margin=0.010
        self.max_grasp_depth=self.gripper_clearance-self.grasp_margin

        # TCP -> fingertip
        self.tcp_to_fingertip=0.1100-(0.0584+0.0445)

        # ============================================================
        # Object validation
        # ============================================================

        self.min_object_height=0.005
        self.max_object_height=0.40

        # ============================================================
        # MoveIt clients
        # ============================================================

        self.move_group_client=ActionClient(
            self,
            MoveGroup,
            "/move_action",
            callback_group=self.cb_group
        )

        self.execute_client=ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
            callback_group=self.cb_group
        )

        self.gripper_client=ActionClient(
            self,
            GripperCommand,
            "/panda_hand_controller/gripper_action",
            callback_group=self.cb_group
        )

        self.cartesian_client=self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
            callback_group=self.cb_group
        )

        # ============================================================
        # Target subscriber
        # ============================================================

        self.target_sub=self.create_subscription(
            PoseStamped,
            "/target_object_pose",
            self.target_pose_callback,
            10,
            callback_group=self.cb_group
        )

        self.get_logger().info(
            "[PnP INIT] Franka Panda MoveIt2 Pick and Place Controller Ready."
        )

        self.get_logger().info(
            f"[PnP INIT] Table Z: {self.table_top_z:.3f} m"
        )

        self.get_logger().info(
            f"[PnP INIT] Place XY: ({self.place_x:.3f}, {self.place_y:.3f})"
        )

        self.get_logger().info(
            f"[PnP INIT] Max grasp depth: {self.max_grasp_depth*1000:.1f} mm"
        )

        # ============================================================
        # Worker
        # ============================================================

        self.worker_thread=threading.Thread(
            target=self.pnp_worker_loop,
            daemon=True
        )

        self.worker_thread.start()

    # ================================================================
    # Target
    # ================================================================

    def target_pose_callback(self,msg):
        if self.is_busy or self.state!="IDLE":
            return

        if msg.header.frame_id!="link0":
            self.get_logger().warn(
                f"[VISION] Expected frame 'link0', got '{msg.header.frame_id}'"
            )
            return

        x=float(msg.pose.position.x)
        y=float(msg.pose.position.y)
        z=float(msg.pose.position.z)

        if not np.isfinite([x,y,z]).all():
            self.get_logger().warn(
                "[VISION] Invalid target position."
            )
            return

        # Detector는 물체 중심을 publish하므로 table 기준으로 높이 계산
        estimated_height=2.0*(z-self.table_top_z)

        if not np.isfinite(estimated_height):
            return

        if estimated_height<self.min_object_height:
            self.get_logger().warn(
                f"[VISION] Object too low: height={estimated_height:.3f} m"
            )
            return

        if estimated_height>self.max_object_height:
            self.get_logger().warn(
                f"[VISION] Object too high: height={estimated_height:.3f} m"
            )
            return

        # ============================================================
        # Quaternion -> yaw
        # ============================================================

        q=msg.pose.orientation

        yaw=math.atan2(
            2.0*(q.w*q.z+q.x*q.y),
            1.0-2.0*(q.y*q.y+q.z*q.z)
        )

        self.target_pose=np.array(
            [x,y,z],
            dtype=np.float64
        )

        self.target_height=float(estimated_height)
        self.target_yaw=float(yaw)

        self.is_busy=True
        self.state="TRIGGER_PICK"

        self.get_logger().info(
            f"[VISION TARGET DETECTED] "
            f"Center(link0)=({x:.3f},{y:.3f},{z:.3f}) "
            f"Height={estimated_height:.3f} m "
            f"Yaw={math.degrees(yaw):.1f} deg"
        )

    # ================================================================
    # Future helper
    # ================================================================

    def wait_future(self,future,timeout_sec):
        event=threading.Event()
        result_holder={
            "result":None,
            "exception":None
        }

        def done_callback(done_future):
            try:
                result_holder["result"]=done_future.result()
            except Exception as e:
                result_holder["exception"]=e
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
                f"[ACTION] Future exception: {result_holder['exception']}"
            )
            return None

        return result_holder["result"]

    # ================================================================
    # Quaternion
    # ================================================================

    def yaw_to_grasp_quaternion(self,yaw):
        # q = q_z(yaw) * q_x(pi)
        #
        # TCP Z axis -> downward
        # TCP yaw -> object yaw
        #
        # ROS quaternion order:
        # (x,y,z,w)

        cy=math.cos(yaw*0.5)
        sy=math.sin(yaw*0.5)

        return (
            cy,
            sy,
            0.0,
            0.0
        )

    # ================================================================
    # MoveGroup pose
    # ================================================================

    def plan_and_execute_pose(
        self,
        x,
        y,
        z,
        qx=1.0,
        qy=0.0,
        qz=0.0,
        qw=0.0
    ):
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("[PLANNING] MoveGroup Action Server not available.")
            return False

        goal=MoveGroup.Goal()
        req=MotionPlanRequest()

        req.group_name="panda_arm"
        req.num_planning_attempts=10              # planning 시도 횟수
        req.allowed_planning_time=5.0             # 최대 planning 시간
        req.max_velocity_scaling_factor=0.5       # 속도 제한
        req.max_acceleration_scaling_factor=0.5   # 가속도 제한
        req.start_state.is_diff=True              # 현재 관절 위치를 시작 상태로 사용

        constraints=Constraints()

        # ============================================================
        # Position constraint
        # ============================================================

        pos=PositionConstraint()

        pos.header.frame_id="link0"
        pos.link_name="hand_tcp"    # 좌표를 맞출 link

        pos.target_point_offset.x=0.0
        pos.target_point_offset.y=0.0
        pos.target_point_offset.z=0.0

        sphere=SolidPrimitive()
        sphere.type=SolidPrimitive.SPHERE
        sphere.dimensions=[0.01]

        bv=BoundingVolume()
        bv.primitives.append(sphere)

        target=Pose()

        target.position.x=float(x)
        target.position.y=float(y)
        target.position.z=float(z)
        target.orientation.w=1.0

        bv.primitive_poses.append(target)

        pos.constraint_region=bv
        pos.weight=1.0

        constraints.position_constraints.append(pos)

        # ============================================================
        # Orientation constraint
        # ============================================================

        ori=OrientationConstraint()

        ori.header.frame_id="link0"
        ori.link_name="hand_tcp"        # 방향을 맞출 link

        ori.orientation.x=float(qx)
        ori.orientation.y=float(qy)
        ori.orientation.z=float(qz)
        ori.orientation.w=float(qw)

        ori.absolute_x_axis_tolerance=0.05
        ori.absolute_y_axis_tolerance=0.05
        ori.absolute_z_axis_tolerance=0.05
        ori.weight=1.0

        constraints.orientation_constraints.append(ori)

        req.goal_constraints.append(constraints)

        goal.request=req

        # ============================================================
        # Planning options
        # ============================================================

        goal.planning_options=PlanningOptions()

        goal.planning_options.plan_only=False
        goal.planning_options.look_around=False
        goal.planning_options.replan=True
        goal.planning_options.replan_attempts=5 # 재계획 회수

        self.get_logger().info(f"[PLANNING] TCP -> ({x:.3f},{y:.3f},{z:.3f})")

        # ============================================================
        # Send
        # ============================================================

        goal_handle=self.wait_future(
            self.move_group_client.send_goal_async(goal),
            10.0
        )

        if goal_handle is None:
            return False

        if not goal_handle.accepted:
            self.get_logger().warn("[PLANNING] Goal rejected.")
            return False

        # ============================================================
        # Result
        # ============================================================

        result_wrapper=self.wait_future(
            goal_handle.get_result_async(),
            30.0
        )

        if result_wrapper is None:
            self.get_logger().warn(
                "[EXECUTION] MoveGroup result timeout."
            )
            return False

        code=result_wrapper.result.error_code.val

        if code==1:
            self.get_logger().info(
                "[EXECUTION] Motion completed successfully."
            )
            return True

        self.get_logger().warn(
            f"[EXECUTION] Motion failed: error_code={code}"
        )

        return False

    # ================================================================
    # Cartesian Z
    # ================================================================

    def cartesian_z_move(
        self,
        x,
        y,
        start_z,
        end_z,
        qx,
        qy,
        qz,
        qw
    ):
        if abs(end_z-start_z)<0.001:
            return True

        if not self.cartesian_client.wait_for_service(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[CARTESIAN] /compute_cartesian_path unavailable."
            )
            return False

        start=Pose()

        start.position.x=float(x)
        start.position.y=float(y)
        start.position.z=float(start_z)

        start.orientation.x=float(qx)
        start.orientation.y=float(qy)
        start.orientation.z=float(qz)
        start.orientation.w=float(qw)

        end=Pose()

        end.position.x=float(x)
        end.position.y=float(y)
        end.position.z=float(end_z)

        end.orientation.x=float(qx)
        end.orientation.y=float(qy)
        end.orientation.z=float(qz)
        end.orientation.w=float(qw)

        request=GetCartesianPath.Request()

        request.header.frame_id="link0"
        request.group_name="panda_arm"
        request.link_name="hand_tcp"

        request.waypoints=[
            start,
            end
        ]

        request.max_step=0.005
        request.jump_threshold=0.0
        request.avoid_collisions=True

        self.get_logger().info(
            f"[CARTESIAN] Z: {start_z:.3f} -> {end_z:.3f}"
        )

        response=self.wait_future(
            self.cartesian_client.call_async(request),
            20.0
        )

        if response is None:
            return False

        self.get_logger().info(
            f"[CARTESIAN] Path fraction={response.fraction:.3f}"
        )

        if response.fraction<0.99:
            self.get_logger().error(
                "[CARTESIAN] Path planning incomplete."
            )
            return False

        # ============================================================
        # Execute trajectory
        # ============================================================

        if not self.execute_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[CARTESIAN] ExecuteTrajectory unavailable."
            )
            return False

        goal=ExecuteTrajectory.Goal()
        goal.trajectory=response.solution

        goal_handle=self.wait_future(
            self.execute_client.send_goal_async(goal),
            10.0
        )

        if goal_handle is None:
            return False

        if not goal_handle.accepted:
            self.get_logger().error(
                "[CARTESIAN] Trajectory rejected."
            )
            return False

        result=self.wait_future(
            goal_handle.get_result_async(),
            30.0
        )

        if result is None:
            self.get_logger().error(
                "[CARTESIAN] Trajectory result timeout."
            )
            return False

        code=result.result.error_code.val

        if code==1:
            self.get_logger().info(
                "[CARTESIAN] Vertical motion completed."
            )
            return True

        self.get_logger().error(
            f"[CARTESIAN] Execution failed: {code}"
        )

        return False

    # ================================================================
    # Ready / Home
    # ================================================================

    def plan_and_execute_named_state(self,named_state):
        if named_state not in ("ready","home"):
            self.get_logger().error(
                f"[PLANNING] Unknown named state: {named_state}"
            )
            return False

        if not self.move_group_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[PLANNING] MoveGroup Action Server unavailable."
            )
            return False

        goal=MoveGroup.Goal()
        req=MotionPlanRequest()

        req.group_name="panda_arm"
        req.num_planning_attempts=5
        req.allowed_planning_time=3.0
        req.max_velocity_scaling_factor=0.5
        req.max_acceleration_scaling_factor=0.5

        constraints=Constraints()

        joints=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7"
        ]

        for name,value in zip(
            joints,
            self.home_qpos
        ):
            jc=JointConstraint()

            jc.joint_name=name
            jc.position=float(value)
            jc.tolerance_above=0.02
            jc.tolerance_below=0.02
            jc.weight=1.0

            constraints.joint_constraints.append(jc)

        req.goal_constraints.append(constraints)

        goal.request=req

        goal.planning_options=PlanningOptions()
        goal.planning_options.plan_only=False
        goal.planning_options.replan=True
        goal.planning_options.replan_attempts=3

        self.get_logger().info(
            f"[PLANNING] Returning to '{named_state}'..."
        )

        goal_handle=self.wait_future(
            self.move_group_client.send_goal_async(goal),
            10.0
        )

        if goal_handle is None:
            return False

        if not goal_handle.accepted:
            self.get_logger().warn(
                "[PLANNING] Ready goal rejected."
            )
            return False

        result_wrapper=self.wait_future(
            goal_handle.get_result_async(),
            20.0
        )

        if result_wrapper is None:
            return False

        code=result_wrapper.result.error_code.val

        if code==1:
            self.get_logger().info(
                "[EXECUTION] Returned to MuJoCo initial pose."
            )
            return True

        self.get_logger().warn(
            f"[EXECUTION] Ready motion failed: {code}"
        )

        return False

    # ================================================================
    # Gripper
    # ================================================================

    def control_gripper(self,action):
        action=action.upper()

        if action not in ("OPEN","CLOSE"):
            self.get_logger().error(
                f"[GRIPPER] Unknown action: {action}"
            )
            return False

        if not self.gripper_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[GRIPPER] Action server unavailable."
            )
            return False

        goal=GripperCommand.Goal()

        if action=="OPEN":
            goal.command.position=0.04
        else:
            goal.command.position=0.0

        goal.command.max_effort=20.0

        self.get_logger().info(
            f"[GRIPPER] Sending {action} command..."
        )

        try:
            # ========================================================
            # Send goal
            # ========================================================

            goal_handle=self.wait_future(
                self.gripper_client.send_goal_async(goal),
                5.0
            )

            if goal_handle is None:
                self.get_logger().error(
                    "[GRIPPER] Goal future timeout."
                )
                return False

            if not goal_handle.accepted:
                self.get_logger().error(
                    f"[GRIPPER] {action} goal rejected."
                )
                return False

            self.get_logger().info(
                f"[GRIPPER] {action} goal accepted."
            )

            # ========================================================
            # Wait result
            # ========================================================

            result_wrapper=self.wait_future(
                goal_handle.get_result_async(),
                10.0
            )

            if result_wrapper is None:
                self.get_logger().error(
                    f"[GRIPPER] {action} result timeout."
                )
                return False

            result=result_wrapper.result

            self.get_logger().info(
                f"[GRIPPER] {action}: "
                f"position={result.position:.4f}, "
                f"reached={result.reached_goal}, "
                f"stalled={result.stalled}"
            )

            if not result.reached_goal:
                self.get_logger().error(
                    f"[GRIPPER] {action} did not reach target."
                )
                return False

            time.sleep(0.2)

            return True

        except Exception as e:
            self.get_logger().error(
                f"[GRIPPER] Action exception: {e}"
            )
            return False

    # ================================================================
    # Place calculation
    # ================================================================

    def calculate_place_pose(self):
        if self.target_height is None:
            return None

        object_center_z=(
            self.table_top_z+
            self.target_height/2.0
        )

        place_tcp_z=(
            object_center_z-
            self.tcp_to_fingertip
        )

        self.get_logger().info(
            f"[PLACE CALC] "
            f"Object center Z={object_center_z:.4f}, "
            f"TCP Z={place_tcp_z:.4f}"
        )

        return (
            self.place_x,
            self.place_y,
            place_tcp_z
        )

    # ================================================================
    # Failure
    # ================================================================

    def reset_after_failure(self,reason):
        self.get_logger().error(
            f"[PnP FAILED] {reason}"
        )

        # 현재 로봇을 ready로 복귀
        self.plan_and_execute_named_state("ready")

        self.target_pose=None
        self.target_height=None
        self.target_yaw=0.0
        self.is_busy=False
        self.state="IDLE"

        self.get_logger().info(
            "[PnP] Reset to IDLE."
        )

    # ================================================================
    # PnP
    # ================================================================

    def pnp_worker_loop(self):
        while rclpy.ok() and not self.shutdown_requested:

            if self.state=="TRIGGER_PICK" and self.target_pose is not None:

                self.state="PICKING"

                tx,ty,tz=self.target_pose

                qx,qy,qz,qw=self.yaw_to_grasp_quaternion(
                    self.target_yaw
                )

                self.get_logger().info(
                    "========== [START PICK & PLACE] =========="
                )

                self.get_logger().info(
                    f"[TARGET] "
                    f"XYZ=({tx:.3f},{ty:.3f},{tz:.3f}) "
                    f"Height={self.target_height:.3f} "
                    f"Yaw={math.degrees(self.target_yaw):.1f} deg"
                )

                # ====================================================
                # 1. Open
                # ====================================================

                self.get_logger().info("[1/8] Opening gripper.")

                if not self.control_gripper("OPEN"):
                    self.reset_after_failure("Gripper OPEN failed.")
                    continue

                # ====================================================
                # 2. Pre-grasp
                # ====================================================

                pre_grasp_z = tz + self.pre_grasp_z_offset

                self.get_logger().info(f"[2/8] Pre-grasp Z={pre_grasp_z:.3f}")

                if not self.plan_and_execute_pose(
                    tx,
                    ty,
                    pre_grasp_z,
                    qx,
                    qy,
                    qz,
                    qw
                ):
                    self.reset_after_failure("Pre-grasp motion failed.")
                    continue

                # ====================================================
                # 3. Calculate grasp
                # ====================================================

                half_height=self.target_height/2.0

                if half_height<=self.max_grasp_depth:
                    grasp_z=tz
                    grasp_mode="CENTER"
                else:
                    object_top_z=tz+half_height
                    grasp_z=object_top_z-self.max_grasp_depth
                    grasp_mode="TOP-LIMITED"

                if grasp_z>=pre_grasp_z:
                    self.reset_after_failure(
                        "Calculated grasp Z is invalid."
                    )
                    continue

                self.get_logger().info(
                    f"[3/8] "
                    f"Mode={grasp_mode}, "
                    f"Grasp Z={grasp_z:.4f}"
                )

                # ====================================================
                # 3-1. XY / orientation alignment
                # ====================================================

                if not self.plan_and_execute_pose(
                    tx,
                    ty,
                    pre_grasp_z,
                    qx,
                    qy,
                    qz,
                    qw
                ):
                    self.reset_after_failure(
                        "Object alignment failed."
                    )
                    continue

                # ====================================================
                # 3-2. Descend
                # ====================================================

                self.get_logger().info(
                    "[3-2/8] Cartesian grasp descent."
                )

                if not self.cartesian_z_move(
                    tx,
                    ty,
                    pre_grasp_z,
                    grasp_z,
                    qx,
                    qy,
                    qz,
                    qw
                ):
                    self.reset_after_failure(
                        "Grasp descent failed."
                    )
                    continue

                # ====================================================
                # 4. Close
                # ====================================================

                self.get_logger().info(
                    "[4/8] Closing gripper."
                )

                if not self.control_gripper("CLOSE"):
                    self.reset_after_failure(
                        "Gripper CLOSE failed."
                    )
                    continue

                # ====================================================
                # 5. Lift
                # ====================================================

                lift_z=max(
                    tz+self.lift_z_offset,
                    grasp_z+self.lift_clearance
                )

                self.get_logger().info(
                    f"[5/8] Lift Z={grasp_z:.3f}->{lift_z:.3f}"
                )

                if not self.cartesian_z_move(
                    tx,
                    ty,
                    grasp_z,
                    lift_z,
                    qx,
                    qy,
                    qz,
                    qw
                ):
                    self.reset_after_failure(
                        "Lift failed."
                    )
                    continue

                # ====================================================
                # 6. Move to place
                # ====================================================

                place_pose=self.calculate_place_pose()

                if place_pose is None:
                    self.reset_after_failure(
                        "Place pose calculation failed."
                    )
                    continue

                px,py,pz=place_pose

                pre_place_z=pz+self.post_place_z_offset

                self.get_logger().info(
                    f"[6/8] "
                    f"Pre-place=({px:.3f},{py:.3f},{pre_place_z:.3f})"
                )

                if not self.plan_and_execute_pose(
                    px,
                    py,
                    pre_place_z,
                    qx,
                    qy,
                    qz,
                    qw
                ):
                    self.reset_after_failure(
                        "Pre-place motion failed."
                    )
                    continue

                # ====================================================
                # 7. Place
                # ====================================================

                self.get_logger().info(
                    f"[7/8] Place Z={pz:.3f}"
                )

                if not self.cartesian_z_move(
                    px,
                    py,
                    pre_place_z,
                    pz,
                    qx,
                    qy,
                    qz,
                    qw
                ):
                    self.reset_after_failure(
                        "Place descent failed."
                    )
                    continue

                # Open gripper
                if not self.control_gripper("OPEN"):
                    self.reset_after_failure(
                        "Place gripper OPEN failed."
                    )
                    continue

                time.sleep(0.6)

                # ====================================================
                # 8. Retract
                # ====================================================

                self.get_logger().info(
                    "[8/8] Retracting."
                )

                if not self.cartesian_z_move(
                    px,
                    py,
                    pz,
                    pre_place_z,
                    qx,
                    qy,
                    qz,
                    qw
                ):
                    self.get_logger().warn(
                        "[PnP] Retract failed. Continuing to ready."
                    )

                # Ready
                self.plan_and_execute_named_state(
                    "ready"
                )

                self.get_logger().info(
                    "========== [PICK & PLACE SUCCESS] =========="
                )

                # ====================================================
                # Reset
                # ====================================================

                self.target_pose=None
                self.target_height=None
                self.target_yaw=0.0
                self.is_busy=False
                self.state="IDLE"

            time.sleep(0.05)

    # ================================================================
    # Shutdown
    # ================================================================

    def shutdown(self):
        self.shutdown_requested=True

        if (
            self.worker_thread.is_alive()
            and threading.current_thread() is not self.worker_thread
        ):
            self.worker_thread.join(timeout=2.0)


# ====================================================================
# Main
# ====================================================================

def main(args=None):
    rclpy.init(args=args)

    node=PandaMoveItPickAndPlace()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__=="__main__":
    main()