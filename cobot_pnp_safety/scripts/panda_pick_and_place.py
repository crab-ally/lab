#!/usr/bin/env python3
"""
MoveIt 2 기반 Franka Emika Panda 3D Vision Pick & Place Controller Node

Subscribes:
  - /target_object_pose (geometry_msgs/PoseStamped)
  - /object_height (std_msgs/Float32)

Actions:
  - /move_action
  - /panda_hand_controller/gripper_action

Gripper:
  - OPEN  : position=0.04
  - CLOSE : position=0.0
  - CLOSE 성공 후 MuJoCo Bridge가 gripper force를 유지

Motion:
  - Pre-grasp
  - Cartesian vertical grasp descent
  - Gripper CLOSE
  - Cartesian segmented vertical lift
  - Cartesian XY pre-place movement with fixed orientation
  - XY Cartesian 실패 시 orientation-constrained joint-space fallback
  - Cartesian vertical place descent
  - Gripper OPEN
  - Cartesian retract
  - Ready/Home
"""

import time,math,threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped,Pose
from std_msgs.msg import Float32
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
        self.object_height=None
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

        self.arm_joints=[
            f"joint{i}" for i in range(1,8)
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

        self.pre_grasp_z_offset=0.2
        self.post_place_z_offset=0.2
        self.lift_z_offset=0.2

        # ============================================================
        # Segmented Cartesian Lift
        # ============================================================

        self.lift_segment_step=0.05

        # ============================================================
        # Cartesian XY pre-place
        # ============================================================

        self.pre_place_xy_step=0.05

        # ============================================================
        # XY joint-space fallback
        # ============================================================

        self.xy_fallback_planning_attempts=10
        self.xy_fallback_planning_time=5.0
        self.xy_fallback_velocity_scale=0.20
        self.xy_fallback_acceleration_scale=0.20

        # 강한 orientation 유지
        self.xy_fallback_orientation_tolerance=0.05

        # ============================================================
        # Gripper
        # ============================================================

        self.gripper_open_position=0.04
        self.gripper_close_position=0.0

        # Bridge에서는 max_effort를 실제 힘 제어에 사용하지 않음
        self.gripper_open_effort=20.0
        self.gripper_close_effort=30.0

        # ============================================================
        # Gripper geometry
        # ============================================================

        self.gripper_clearance=0.100
        self.grasp_margin=0.010
        self.max_grasp_depth=(
            self.gripper_clearance-
            self.grasp_margin
        )

        self.tcp_to_fingertip=(
            0.1100-
            (0.0584+0.0445)
        )

        # ============================================================
        # Object validation
        # ============================================================

        self.min_object_height=0.005
        self.max_object_height=0.40

        # ============================================================
        # Cartesian
        # ============================================================

        self.cartesian_fraction_threshold=0.99

        # ============================================================
        # Lift fallback
        # ============================================================

        self.lift_fallback_planning_attempts=10
        self.lift_fallback_planning_time=5.0
        self.lift_fallback_velocity_scale=0.35
        self.lift_fallback_acceleration_scale=0.35

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
        # Target subscribers
        # ============================================================

        self.target_sub=self.create_subscription(
            PoseStamped,
            "/target_object_pose",
            self.target_pose_callback,
            10,
            callback_group=self.cb_group
        )

        self.height_sub=self.create_subscription(
            Float32,
            "/object_height",
            self.object_height_callback,
            10,
            callback_group=self.cb_group
        )

        # ============================================================
        # Init logs
        # ============================================================

        self.get_logger().info(
            "[PnP INIT] Franka Panda MoveIt2 Pick and Place Controller Ready."
        )

        self.get_logger().info(
            f"[PnP INIT] Table Z={self.table_top_z:.3f}, "
            f"Place XY=({self.place_x:.3f},{self.place_y:.3f}), "
            f"Max grasp depth={self.max_grasp_depth:.3f}, "
            f"Lift step={self.lift_segment_step:.3f}"
        )

        self.get_logger().info(
            f"[PnP INIT] Lift Cartesian fraction threshold="
            f"{self.cartesian_fraction_threshold:.2f}"
        )

        self.get_logger().info(
            f"[PnP INIT] Pre-place XY Cartesian step="
            f"{self.pre_place_xy_step:.3f} m"
        )

        self.get_logger().info(
            "[PnP INIT] Lift Cartesian failure -> "
            "joint-space pose fallback enabled."
        )

        self.get_logger().info(
            "[PnP INIT] Pre-place XY Cartesian failure -> "
            "orientation-constrained joint-space fallback enabled."
        )

        self.get_logger().info(
            f"[PnP INIT] XY fallback orientation tolerance="
            f"{self.xy_fallback_orientation_tolerance:.3f} rad"
        )

        self.get_logger().info(
            "[PnP INIT] Gripper CLOSE uses Bridge contact/stall detection."
        )

        # ============================================================
        # Worker
        # ============================================================

        self.worker=threading.Thread(
            target=self.pnp_worker_loop,
            daemon=True
        )

        self.worker.start()

    # ============================================================
    # Subscribers
    # ============================================================

    def object_height_callback(self,msg):

        if self.is_busy:
            return

        height=float(msg.data)

        if not np.isfinite(height):
            self.get_logger().warn(
                f"[PnP] Invalid object height: {height}"
            )
            return

        self.object_height=height

    def target_pose_callback(self,msg):

        if self.is_busy or self.state!="IDLE":
            return

        if msg.header.frame_id!="link0":
            self.get_logger().warn(
                f"[PnP] Invalid target frame: {msg.header.frame_id}"
            )
            return

        p=msg.pose.position
        q=msg.pose.orientation

        x=float(p.x)
        y=float(p.y)
        z=float(p.z)

        if not all(np.isfinite(v) for v in (x,y,z)):
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

        h=float(self.object_height)

        if not self.min_object_height<=h<=self.max_object_height:
            self.get_logger().warn(
                f"[PnP] Invalid object height: {h:.4f} m"
            )
            return

        r11=1.0-2.0*(q.y*q.y+q.z*q.z)
        r21=2.0*(q.x*q.y+q.w*q.z)

        yaw=math.atan2(r21,r11)

        self.target_pose=np.array(
            [x,y,z],
            dtype=np.float64
        )

        self.target_yaw=yaw
        self.is_busy=True
        self.state="TRIGGER_PICK"

        self.get_logger().info(
            f"[PnP] Target received: "
            f"xyz=({x:.3f},{y:.3f},{z:.3f}), "
            f"h={h:.3f}, "
            f"yaw={math.degrees(yaw):.1f} deg"
        )

    # ============================================================
    # Future helper
    # ============================================================

    def wait_future(self,future,timeout,description):

        event=threading.Event()
        result=[None]

        def done_callback(f):
            result[0]=f
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

    # ============================================================
    # Grasp orientation
    # ============================================================

    def yaw_to_grasp_quaternion(self,yaw):

        while yaw>math.pi/2:
            yaw-=math.pi

        while yaw<-math.pi/2:
            yaw+=math.pi

        cy=math.cos(yaw/2.0)
        sy=math.sin(yaw/2.0)

        # Preserve existing Panda grasp orientation convention
        return cy,sy,0.0,0.0

    # ============================================================
    # MoveIt Pose Planning
    # ============================================================

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

        if not self.move_group_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[PnP] MoveGroup server unavailable."
            )
            return False

        goal=MoveGroup.Goal()

        req=MotionPlanRequest()

        req.group_name="panda_arm"
        req.num_planning_attempts=10
        req.allowed_planning_time=5.0

        req.max_velocity_scaling_factor=0.25
        req.max_acceleration_scaling_factor=0.25

        req.start_state.is_diff=True

        # --------------------------------------------------------
        # Position constraint
        # --------------------------------------------------------

        pc=PositionConstraint()

        pc.header.frame_id="link0"
        pc.link_name="hand_tcp"

        pc.target_point_offset.x=0.0
        pc.target_point_offset.y=0.0
        pc.target_point_offset.z=0.0

        primitive=SolidPrimitive()

        primitive.type=SolidPrimitive.SPHERE
        primitive.dimensions=[0.015]

        pc.constraint_region=BoundingVolume()
        pc.constraint_region.primitives.append(
            primitive
        )

        pose=Pose()

        pose.position.x=x
        pose.position.y=y
        pose.position.z=z

        pose.orientation.w=1.0

        pc.constraint_region.primitive_poses.append(
            pose
        )

        pc.weight=1.0

        # --------------------------------------------------------
        # Orientation constraint
        # --------------------------------------------------------

        oc=OrientationConstraint()

        oc.header.frame_id="link0"
        oc.link_name="hand_tcp"

        oc.orientation.x=qx
        oc.orientation.y=qy
        oc.orientation.z=qz
        oc.orientation.w=qw

        oc.absolute_x_axis_tolerance=0.25
        oc.absolute_y_axis_tolerance=0.25
        oc.absolute_z_axis_tolerance=0.35
        oc.weight=1.0

        constraints=Constraints()

        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)

        req.goal_constraints.append(
            constraints
        )

        # --------------------------------------------------------
        # Planning options
        # --------------------------------------------------------

        options=PlanningOptions()

        options.plan_only=False
        options.look_around=False
        options.replan=True
        options.replan_attempts=5

        goal.request=req
        goal.planning_options=options

        # --------------------------------------------------------
        # Send goal
        # --------------------------------------------------------

        future=self.move_group_client.send_goal_async(goal)

        handle=self.wait_future(
            future,
            10.0,
            "MoveGroup goal"
        )

        if handle is None or not handle.accepted:
            self.get_logger().error(
                "[PnP] MoveGroup goal rejected."
            )
            return False

        result=self.wait_future(
            handle.get_result_async(),
            30.0,
            "MoveGroup result"
        )

        if result is None:
            return False

        ok=result.result.error_code.val==1

        if not ok:
            self.get_logger().error(
                f"[PnP] MoveGroup failed: "
                f"error_code={result.result.error_code.val}"
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
        qw
    ):

        if abs(end_z-start_z)<0.001:
            return True

        if not self.cartesian_client.wait_for_service(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[PnP] /compute_cartesian_path unavailable."
            )
            return False

        req=GetCartesianPath.Request()

        req.header.frame_id="link0"
        req.group_name="panda_arm"
        req.link_name="hand_tcp"

        req.max_step=0.005
        req.jump_threshold=0.0
        req.avoid_collisions=True

        req.start_state.is_diff=True

        waypoint=Pose()

        waypoint.position.x=x
        waypoint.position.y=y
        waypoint.position.z=end_z

        waypoint.orientation.x=qx
        waypoint.orientation.y=qy
        waypoint.orientation.z=qz
        waypoint.orientation.w=qw

        req.waypoints=[waypoint]

        future=self.cartesian_client.call_async(req)

        response=self.wait_future(
            future,
            20.0,
            "Cartesian path computation"
        )

        if response is None:
            return False

        self.get_logger().info(
            f"[Cartesian] z {start_z:.3f} -> {end_z:.3f}, "
            f"fraction={response.fraction:.3f}"
        )

        if (
            response.fraction
            <self.cartesian_fraction_threshold
        ):

            self.get_logger().error(
                f"[Cartesian] Path fraction too low: "
                f"{response.fraction:.3f}"
            )

            if response.solution.joint_trajectory.points:

                last=response.solution.joint_trajectory.points[-1]

                self.get_logger().error(
                    f"[Cartesian] points="
                    f"{len(response.solution.joint_trajectory.points)}, "
                    f"joints={len(last.positions)}, "
                    f"last_positions={list(last.positions)}"
                )

            return False

        # --------------------------------------------------------
        # Execute trajectory
        # --------------------------------------------------------

        if not self.execute_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[PnP] ExecuteTrajectory server unavailable."
            )
            return False

        goal=ExecuteTrajectory.Goal()
        goal.trajectory=response.solution

        future=self.execute_client.send_goal_async(goal)

        handle=self.wait_future(
            future,
            10.0,
            "ExecuteTrajectory goal"
        )

        if handle is None or not handle.accepted:
            self.get_logger().error(
                "[PnP] ExecuteTrajectory goal rejected."
            )
            return False

        result=self.wait_future(
            handle.get_result_async(),
            30.0,
            "ExecuteTrajectory result"
        )

        if result is None:
            return False

        ok=result.result.error_code.val==1

        if not ok:
            self.get_logger().error(
                f"[PnP] ExecuteTrajectory failed: "
                f"error_code={result.result.error_code.val}"
            )

        return ok

    # ============================================================
    # Cartesian XY Move
    #
    # Z 고정
    # Orientation 고정
    # XY만 이동
    #
    # 먼저 작은 waypoint 구간으로 Cartesian path를 계산한다.
    # Cartesian fraction이 충분하지 않으면 호출자가
    # orientation-constrained joint-space fallback을 수행한다.
    # ============================================================

    def cartesian_xy_move(
        self,
        start_x,
        start_y,
        end_x,
        end_y,
        z,
        qx,
        qy,
        qz,
        qw
    ):

        dx=end_x-start_x
        dy=end_y-start_y

        distance=math.sqrt(
            dx*dx+
            dy*dy
        )

        if distance<0.001:
            self.get_logger().info(
                "[Cartesian XY] Start and target XY are already equal."
            )
            return True

        if not self.cartesian_client.wait_for_service(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[Cartesian XY] /compute_cartesian_path unavailable."
            )
            return False

        # --------------------------------------------------------
        # Waypoint 생성
        #
        # 약 5 cm 간격
        # Z 고정
        # Orientation 고정
        # --------------------------------------------------------

        step=max(
            0.005,
            float(self.pre_place_xy_step)
        )

        segments=int(
            math.ceil(distance/step)
        )

        waypoints=[]

        for i in range(1,segments+1):

            ratio=float(i)/float(segments)

            waypoint=Pose()

            waypoint.position.x=(
                start_x+
                dx*ratio
            )

            waypoint.position.y=(
                start_y+
                dy*ratio
            )

            waypoint.position.z=z

            waypoint.orientation.x=qx
            waypoint.orientation.y=qy
            waypoint.orientation.z=qz
            waypoint.orientation.w=qw

            waypoints.append(
                waypoint
            )

        self.get_logger().info(
            f"[Cartesian XY] "
            f"({start_x:.3f},{start_y:.3f},{z:.3f}) -> "
            f"({end_x:.3f},{end_y:.3f},{z:.3f}), "
            f"distance={distance:.3f} m, "
            f"segments={segments}, "
            f"orientation fixed"
        )

        # --------------------------------------------------------
        # GetCartesianPath
        # --------------------------------------------------------

        req=GetCartesianPath.Request()

        req.header.frame_id="link0"
        req.group_name="panda_arm"
        req.link_name="hand_tcp"

        req.max_step=0.005
        req.jump_threshold=0.0
        req.avoid_collisions=True

        req.start_state.is_diff=True

        req.waypoints=waypoints

        future=self.cartesian_client.call_async(req)

        response=self.wait_future(
            future,
            30.0,
            "Cartesian XY path computation"
        )

        if response is None:
            return False

        self.get_logger().info(
            f"[Cartesian XY] "
            f"fraction={response.fraction:.3f}"
        )

        # --------------------------------------------------------
        # Cartesian fraction 검사
        # --------------------------------------------------------

        if (
            response.fraction
            <self.cartesian_fraction_threshold
        ):

            self.get_logger().warn(
                f"[Cartesian XY] Path fraction too low: "
                f"{response.fraction:.3f}"
            )

            if response.solution.joint_trajectory.points:

                last=(
                    response
                    .solution
                    .joint_trajectory
                    .points[-1]
                )

                self.get_logger().warn(
                    f"[Cartesian XY] partial points="
                    f"{len(response.solution.joint_trajectory.points)}, "
                    f"joints={len(last.positions)}, "
                    f"last_positions={list(last.positions)}"
                )

            self.get_logger().warn(
                "[Cartesian XY] Cartesian failed. "
                "Caller will try orientation-constrained "
                "joint-space fallback."
            )

            return False

        # --------------------------------------------------------
        # Execute
        # --------------------------------------------------------

        if not self.execute_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[Cartesian XY] ExecuteTrajectory server unavailable."
            )
            return False

        goal=ExecuteTrajectory.Goal()
        goal.trajectory=response.solution

        future=self.execute_client.send_goal_async(goal)

        handle=self.wait_future(
            future,
            10.0,
            "Cartesian XY ExecuteTrajectory goal"
        )

        if handle is None or not handle.accepted:
            self.get_logger().error(
                "[Cartesian XY] ExecuteTrajectory goal rejected."
            )
            return False

        result=self.wait_future(
            handle.get_result_async(),
            30.0,
            "Cartesian XY ExecuteTrajectory result"
        )

        if result is None:
            return False

        ok=result.result.error_code.val==1

        if not ok:
            self.get_logger().error(
                f"[Cartesian XY] ExecuteTrajectory failed: "
                f"error_code={result.result.error_code.val}"
            )
            return False

        self.get_logger().info(
            "[Cartesian XY] Movement completed successfully."
        )

        return True

    # ============================================================
    # XY Joint-space Fallback with Strong Orientation Constraint
    #
    # Cartesian XY가 fraction 부족으로 실패했을 때 사용.
    #
    # 핵심:
    #   - Z는 lift_z 그대로 유지
    #   - 목표 XY만 변경
    #   - grasp quaternion 그대로 유지
    #   - OrientationConstraint를 강하게 적용
    #   - 일반 plan_and_execute_pose()를 사용하지 않음
    #   - Gripper CLOSE 상태는 변경하지 않음
    #
    # 이 fallback은 물체를 잡은 상태에서 손목이 불필요하게
    # 회전하는 것을 방지하기 위한 전용 함수이다.
    # ============================================================

    def xy_joint_space_fallback(
        self,
        start_x,
        start_y,
        target_x,
        target_y,
        z,
        qx,
        qy,
        qz,
        qw
    ):

        distance=math.sqrt(
            (target_x-start_x)*(target_x-start_x)+
            (target_y-start_y)*(target_y-start_y)
        )

        self.get_logger().warn(
            "[XY FALLBACK] Cartesian XY failed."
        )

        self.get_logger().warn(
            f"[XY FALLBACK] "
            f"Joint-space planning: "
            f"({start_x:.3f},{start_y:.3f},{z:.3f}) -> "
            f"({target_x:.3f},{target_y:.3f},{z:.3f}), "
            f"distance={distance:.3f} m"
        )

        self.get_logger().warn(
            "[XY FALLBACK] "
            "Grasp orientation will be strongly constrained."
        )

        self.get_logger().warn(
            f"[XY FALLBACK] Orientation tolerance="
            f"{self.xy_fallback_orientation_tolerance:.3f} rad"
        )

        if not self.move_group_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[XY FALLBACK] MoveGroup server unavailable."
            )
            return False

        goal=MoveGroup.Goal()

        req=MotionPlanRequest()

        req.group_name="panda_arm"

        req.num_planning_attempts=(
            self.xy_fallback_planning_attempts
        )

        req.allowed_planning_time=(
            self.xy_fallback_planning_time
        )

        req.max_velocity_scaling_factor=(
            self.xy_fallback_velocity_scale
        )

        req.max_acceleration_scaling_factor=(
            self.xy_fallback_acceleration_scale
        )

        req.start_state.is_diff=True

        # --------------------------------------------------------
        # Position constraint
        # --------------------------------------------------------

        pc=PositionConstraint()

        pc.header.frame_id="link0"
        pc.link_name="hand_tcp"

        pc.target_point_offset.x=0.0
        pc.target_point_offset.y=0.0
        pc.target_point_offset.z=0.0

        primitive=SolidPrimitive()

        primitive.type=SolidPrimitive.SPHERE
        primitive.dimensions=[0.015]

        pc.constraint_region=BoundingVolume()

        pc.constraint_region.primitives.append(
            primitive
        )

        target_pose=Pose()

        target_pose.position.x=target_x
        target_pose.position.y=target_y
        target_pose.position.z=z

        # Position constraint의 pose orientation은 실제 orientation
        # constraint와 별개이므로 identity로 둔다.
        target_pose.orientation.w=1.0

        pc.constraint_region.primitive_poses.append(
            target_pose
        )

        pc.weight=1.0

        # --------------------------------------------------------
        # Strong Orientation Constraint
        # --------------------------------------------------------

        oc=OrientationConstraint()

        oc.header.frame_id="link0"
        oc.link_name="hand_tcp"

        oc.orientation.x=qx
        oc.orientation.y=qy
        oc.orientation.z=qz
        oc.orientation.w=qw

        tolerance=(
            self.xy_fallback_orientation_tolerance
        )

        oc.absolute_x_axis_tolerance=tolerance
        oc.absolute_y_axis_tolerance=tolerance
        oc.absolute_z_axis_tolerance=tolerance
        oc.weight=1.0

        constraints=Constraints()

        constraints.position_constraints.append(
            pc
        )

        constraints.orientation_constraints.append(
            oc
        )

        req.goal_constraints.append(
            constraints
        )

        # --------------------------------------------------------
        # Planning options
        # --------------------------------------------------------

        options=PlanningOptions()

        options.plan_only=False
        options.look_around=False
        options.replan=True
        options.replan_attempts=5

        goal.request=req
        goal.planning_options=options

        # --------------------------------------------------------
        # Send MoveGroup goal
        # --------------------------------------------------------

        future=self.move_group_client.send_goal_async(
            goal
        )

        handle=self.wait_future(
            future,
            10.0,
            "XY fallback MoveGroup goal"
        )

        if handle is None or not handle.accepted:
            self.get_logger().error(
                "[XY FALLBACK] MoveGroup goal rejected."
            )
            return False

        result=self.wait_future(
            handle.get_result_async(),
            30.0,
            "XY fallback MoveGroup result"
        )

        if result is None:
            return False

        error_code=result.result.error_code.val

        if error_code!=1:

            self.get_logger().error(
                f"[XY FALLBACK] "
                f"Planning/execution failed: "
                f"error_code={error_code}"
            )

            return False

        # --------------------------------------------------------
        # Success
        # --------------------------------------------------------

        self.get_logger().info(
            "[XY FALLBACK] "
            "Joint-space XY movement completed successfully."
        )

        self.get_logger().info(
            "[XY FALLBACK] "
            "Grasp orientation constraint was applied."
        )

        self.get_logger().info(
            "[XY FALLBACK] "
            "Gripper CLOSE remains held."
        )

        return True

    # ============================================================
    # LIFT Joint-space Fallback
    # ============================================================

    def lift_joint_space_fallback(
        self,
        x,
        y,
        target_z,
        qx,
        qy,
        qz,
        qw
    ):

        self.get_logger().warn(
            f"[LIFT FALLBACK] Cartesian failed. "
            f"Joint-space planning to z={target_z:.3f}"
        )

        if not self.move_group_client.wait_for_server(
            timeout_sec=3.0
        ):
            self.get_logger().error(
                "[LIFT FALLBACK] MoveGroup server unavailable."
            )
            return False

        goal=MoveGroup.Goal()

        req=MotionPlanRequest()

        req.group_name="panda_arm"

        req.num_planning_attempts=(
            self.lift_fallback_planning_attempts
        )

        req.allowed_planning_time=(
            self.lift_fallback_planning_time
        )

        req.max_velocity_scaling_factor=(
            self.lift_fallback_velocity_scale
        )

        req.max_acceleration_scaling_factor=(
            self.lift_fallback_acceleration_scale
        )

        req.start_state.is_diff=True

        # --------------------------------------------------------
        # Position constraint
        # --------------------------------------------------------

        pc=PositionConstraint()

        pc.header.frame_id="link0"
        pc.link_name="hand_tcp"

        pc.target_point_offset.x=0.0
        pc.target_point_offset.y=0.0
        pc.target_point_offset.z=0.0

        primitive=SolidPrimitive()

        primitive.type=SolidPrimitive.SPHERE
        primitive.dimensions=[0.015]

        pc.constraint_region=BoundingVolume()

        pc.constraint_region.primitives.append(
            primitive
        )

        target_pose=Pose()

        target_pose.position.x=x
        target_pose.position.y=y
        target_pose.position.z=target_z

        target_pose.orientation.x=qx
        target_pose.orientation.y=qy
        target_pose.orientation.z=qz
        target_pose.orientation.w=qw

        pc.constraint_region.primitive_poses.append(
            target_pose
        )

        pc.weight=1.0

        # --------------------------------------------------------
        # Orientation constraint
        # --------------------------------------------------------

        oc=OrientationConstraint()

        oc.header.frame_id="link0"
        oc.link_name="hand_tcp"

        oc.orientation.x=qx
        oc.orientation.y=qy
        oc.orientation.z=qz
        oc.orientation.w=qw

        oc.absolute_x_axis_tolerance=0.25
        oc.absolute_y_axis_tolerance=0.25
        oc.absolute_z_axis_tolerance=0.35
        oc.weight=1.0

        constraints=Constraints()

        constraints.position_constraints.append(
            pc
        )

        constraints.orientation_constraints.append(
            oc
        )

        req.goal_constraints.append(
            constraints
        )

        # --------------------------------------------------------
        # Planning options
        # --------------------------------------------------------

        options=PlanningOptions()

        options.plan_only=False
        options.look_around=False
        options.replan=True
        options.replan_attempts=5

        goal.request=req
        goal.planning_options=options

        # --------------------------------------------------------
        # Send MoveGroup goal
        # --------------------------------------------------------

        future=self.move_group_client.send_goal_async(
            goal
        )

        handle=self.wait_future(
            future,
            10.0,
            "LIFT fallback MoveGroup goal"
        )

        if handle is None or not handle.accepted:
            self.get_logger().error(
                "[LIFT FALLBACK] MoveGroup goal rejected."
            )
            return False

        result=self.wait_future(
            handle.get_result_async(),
            30.0,
            "LIFT fallback MoveGroup result"
        )

        if result is None:
            return False

        error_code=result.result.error_code.val

        if error_code!=1:
            self.get_logger().error(
                f"[LIFT FALLBACK] Planning/execution failed: "
                f"error_code={error_code}"
            )
            return False

        # --------------------------------------------------------
        # Actual execution success
        # --------------------------------------------------------

        self.get_logger().info(
            "[LIFT FALLBACK] MoveGroup execution successful."
        )

        time.sleep(0.1)

        self.get_logger().info(
            f"[LIFT FALLBACK] Target pose: "
            f"x={x:.3f}, "
            f"y={y:.3f}, "
            f"z={target_z:.3f}"
        )

        self.get_logger().info(
            "[LIFT FALLBACK] Gripper CLOSE remains held. "
            "No OPEN/CLOSE command sent during fallback."
        )

        return True

    # ============================================================
    # Segmented Cartesian Lift + Joint-space Fallback
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
        step=0.05
    ):

        delta=end_z-start_z

        if abs(delta)<0.001:
            return True

        if step<=0.0:
            self.get_logger().error(
                "[LIFT] Invalid lift segment step."
            )
            return False

        direction=1.0 if delta>0.0 else -1.0

        total_segments=int(
            math.ceil(abs(delta)/step)
        )

        current_z=float(start_z)

        self.get_logger().info(
            f"[LIFT] Start segmented lift: "
            f"z={start_z:.3f}->{end_z:.3f}, "
            f"step={step:.3f}, "
            f"segments={total_segments}"
        )

        # ========================================================
        # Gripper 상태 유지
        # ========================================================

        for segment_index in range(
            1,
            total_segments+1
        ):

            remaining=abs(
                end_z-current_z
            )

            move=min(
                step,
                remaining
            )

            next_z=(
                current_z+
                direction*move
            )

            self.get_logger().info(
                f"[LIFT] Segment "
                f"[{segment_index}/{total_segments}] "
                f"z={current_z:.3f}->{next_z:.3f}"
            )

            # ====================================================
            # 1. Cartesian 시도
            # ====================================================

            cartesian_success=self.cartesian_z_move(
                x,
                y,
                current_z,
                next_z,
                qx,
                qy,
                qz,
                qw
            )

            if cartesian_success:

                current_z=next_z

                self.get_logger().info(
                    f"[LIFT] Segment {segment_index} "
                    f"Cartesian success."
                )

                continue

            # ====================================================
            # 2. Cartesian 실패
            # ====================================================

            self.get_logger().warn(
                f"[LIFT] Segment {segment_index} "
                f"Cartesian failed. "
                f"Trying joint-space fallback."
            )

            # ====================================================
            # 3. Joint-space fallback
            # ====================================================

            fallback_success=self.lift_joint_space_fallback(
                x,
                y,
                next_z,
                qx,
                qy,
                qz,
                qw
            )

            if not fallback_success:

                self.get_logger().error(
                    f"[LIFT] Segment {segment_index} failed. "
                    f"Both Cartesian and joint-space "
                    f"fallback failed."
                )

                return False

            # ====================================================
            # 4. Fallback 성공
            # ====================================================

            current_z=next_z

            self.get_logger().info(
                f"[LIFT] Segment {segment_index} "
                f"joint-space fallback success."
            )

        self.get_logger().info(
            f"[LIFT] Completed: "
            f"z={start_z:.3f}->{end_z:.3f}"
        )

        self.get_logger().info(
            "[LIFT] Gripper CLOSE remains held by MuJoCo Bridge."
        )

        return True

    # ============================================================
    # Named State / Home
    # ============================================================

    def plan_and_execute_named_state(
        self,
        named_state
    ):

        if named_state not in (
            "ready",
            "home"
        ):

            self.get_logger().error(
                f"[PnP] Unsupported named state: {named_state}"
            )

            return False

        if not self.move_group_client.wait_for_server(
            timeout_sec=3.0
        ):

            self.get_logger().error(
                "[PnP] MoveGroup server unavailable."
            )

            return False

        self.get_logger().info(
            f"[PnP] Returning to named state: {named_state}"
        )

        goal=MoveGroup.Goal()

        req=MotionPlanRequest()

        req.group_name="panda_arm"
        req.num_planning_attempts=5
        req.allowed_planning_time=3.0

        req.max_velocity_scaling_factor=0.5
        req.max_acceleration_scaling_factor=0.5

        req.start_state.is_diff=True

        constraints=Constraints()

        for joint,value in zip(
            self.arm_joints,
            self.home_qpos
        ):

            jc=JointConstraint()

            jc.joint_name=joint
            jc.position=value

            jc.tolerance_above=0.02
            jc.tolerance_below=0.02
            jc.weight=1.0

            constraints.joint_constraints.append(
                jc
            )

        req.goal_constraints.append(
            constraints
        )

        options=PlanningOptions()

        options.plan_only=False
        options.replan=True
        options.replan_attempts=3

        goal.request=req
        goal.planning_options=options

        future=self.move_group_client.send_goal_async(
            goal
        )

        handle=self.wait_future(
            future,
            10.0,
            "Named-state MoveGroup goal"
        )

        if handle is None or not handle.accepted:

            self.get_logger().error(
                "[PnP] Named-state goal rejected."
            )

            return False

        result=self.wait_future(
            handle.get_result_async(),
            20.0,
            "Named-state MoveGroup result"
        )

        if result is None:
            return False

        ok=result.result.error_code.val==1

        if ok:

            self.get_logger().info(
                "[PnP] Returned to MuJoCo initial pose."
            )

        else:

            self.get_logger().error(
                f"[PnP] Failed to return home: "
                f"error_code={result.result.error_code.val}"
            )

        return ok

    # ============================================================
    # Gripper
    # ============================================================

    def control_gripper(
        self,
        action
    ):

        action=action.upper()

        if action not in (
            "OPEN",
            "CLOSE"
        ):

            self.get_logger().error(
                f"[GRIPPER] Invalid action: {action}"
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

            goal.command.position=(
                self.gripper_open_position
            )

            goal.command.max_effort=(
                self.gripper_open_effort
            )

        else:

            goal.command.position=(
                self.gripper_close_position
            )

            goal.command.max_effort=(
                self.gripper_close_effort
            )

        self.get_logger().info(
            f"[GRIPPER] Sending {action}: "
            f"position={goal.command.position:.3f}"
        )

        try:

            future=self.gripper_client.send_goal_async(
                goal
            )

            handle=self.wait_future(
                future,
                5.0,
                f"Gripper {action} goal"
            )

            if handle is None or not handle.accepted:

                self.get_logger().error(
                    f"[GRIPPER] {action} goal rejected."
                )

                return False

            result=self.wait_future(
                handle.get_result_async(),
                10.0,
                f"Gripper {action} result"
            )

            if result is None:
                return False

            result=result.result

            self.get_logger().info(
                f"[GRIPPER] {action}: "
                f"position={result.position:.4f}, "
                f"reached={result.reached_goal}, "
                f"stalled={result.stalled}"
            )

            if not result.reached_goal:

                self.get_logger().error(
                    f"[GRIPPER] {action} failed."
                )

                return False

            time.sleep(0.2)

            return True

        except Exception as e:

            self.get_logger().error(
                f"[GRIPPER] {action} exception: {e}"
            )

            return False

    # ============================================================
    # Place Pose
    # ============================================================

    def calculate_place_pose(self):

        object_center_z=(
            self.table_top_z+
            self.object_height/2.0
        )

        place_tcp_z=(
            object_center_z-
            self.tcp_to_fingertip
        )

        return (
            self.place_x,
            self.place_y,
            place_tcp_z
        )

    # ============================================================
    # Failure Reset
    # ============================================================

    def reset_after_failure(
        self,
        reason
    ):

        self.get_logger().error(
            f"[PnP] Pick & Place failed: {reason}"
        )

        try:

            self.plan_and_execute_named_state(
                "ready"
            )

        except Exception as e:

            self.get_logger().error(
                f"[PnP] Recovery failed: {e}"
            )

        self.target_pose=None
        self.object_height=None
        self.target_yaw=0.0
        self.is_busy=False
        self.state="IDLE"

    # ============================================================
    # Pick & Place Worker
    # ============================================================

    def pnp_worker_loop(self):

        while (
            rclpy.ok()
            and not self.shutdown_requested
        ):

            try:

                if (
                    self.state=="TRIGGER_PICK"
                    and self.target_pose is not None
                ):

                    self.state="PICKING"

                    tx,ty,tz=self.target_pose

                    half_height=(
                        self.object_height/2.0
                    )

                    qx,qy,qz,qw=(
                        self.yaw_to_grasp_quaternion(
                            self.target_yaw
                        )
                    )

                    self.get_logger().info(
                        "================================================"
                    )

                    self.get_logger().info(
                        "[PnP] Pick & Place sequence started."
                    )

                    self.get_logger().info(
                        f"[PnP] Target: "
                        f"({tx:.3f},{ty:.3f},{tz:.3f}), "
                        f"h={self.object_height:.3f}"
                    )

                    self.get_logger().info(
                        "================================================"
                    )

                    # ====================================================
                    # [1/8] Open Gripper
                    # ====================================================

                    self.get_logger().info(
                        "[1/8] Opening gripper."
                    )

                    if not self.control_gripper(
                        "OPEN"
                    ):

                        self.reset_after_failure(
                            "Gripper OPEN failed."
                        )

                        continue

                    # ====================================================
                    # [2/8] Move to Pre-Grasp
                    # ====================================================

                    pre_grasp_z=(
                        tz+
                        half_height+
                        self.pre_grasp_z_offset
                    )

                    self.get_logger().info(
                        f"[2/8] Moving to pre-grasp: "
                        f"z={pre_grasp_z:.3f}"
                    )

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
                            "Pre-grasp planning/execution failed."
                        )

                        continue

                    # ====================================================
                    # [3/8] Cartesian Descend to Grasp
                    # ====================================================

                    if (
                        half_height
                        <=self.max_grasp_depth
                    ):

                        grasp_z=tz
                        grasp_mode="CENTER"

                    else:

                        object_top_z=(
                            tz+
                            half_height
                        )

                        grasp_z=(
                            object_top_z-
                            self.max_grasp_depth
                        )

                        grasp_mode="TOP-LIMITED"

                    self.get_logger().info(
                        f"[3/8] Grasp position: "
                        f"z={grasp_z:.3f}, "
                        f"mode={grasp_mode}"
                    )

                    if grasp_z>=pre_grasp_z:

                        self.reset_after_failure(
                            "Invalid grasp Z."
                        )

                        continue

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
                            "Cartesian grasp descent failed."
                        )

                        continue

                    # ====================================================
                    # [4/8] Close Gripper
                    # ====================================================

                    self.get_logger().info(
                        "[4/8] Closing gripper and "
                        "detecting object contact."
                    )

                    if not self.control_gripper(
                        "CLOSE"
                    ):

                        self.reset_after_failure(
                            "Gripper CLOSE/contact detection failed."
                        )

                        continue

                    self.get_logger().info(
                        "[4/8] Gripper CLOSE successful. "
                        "Bridge grasp force remains active during LIFT "
                        "and PRE-PLACE."
                    )

                    # ====================================================
                    # [5/8] Vertical Lift
                    # ====================================================

                    lift_z=(
                        tz+
                        half_height+
                        self.lift_z_offset
                    )

                    self.get_logger().info(
                        f"[5/8] Lifting vertically: "
                        f"z={grasp_z:.3f}->{lift_z:.3f}"
                    )

                    if not self.cartesian_z_move_segmented(
                        tx,
                        ty,
                        grasp_z,
                        lift_z,
                        qx,
                        qy,
                        qz,
                        qw,
                        self.lift_segment_step
                    ):

                        self.reset_after_failure(
                            "Segmented Cartesian lift failed."
                        )

                        continue

                    self.get_logger().info(
                        f"[LIFT] Completed: "
                        f"z={grasp_z:.3f}->{lift_z:.3f}"
                    )

                    self.get_logger().info(
                        "[LIFT] Gripper CLOSE remains held."
                    )

                    # ====================================================
                    # [6/8] Cartesian XY PRE-PLACE
                    #
                    # 1차:
                    #   5 cm waypoint Cartesian XY
                    #
                    # 실패:
                    #   orientation-constrained joint-space fallback
                    #
                    # Z:
                    #   lift_z 고정
                    #
                    # Orientation:
                    #   grasp quaternion 고정
                    #
                    # Gripper:
                    #   CLOSE 유지
                    # ====================================================

                    px,py,pz=(
                        self.calculate_place_pose()
                    )

                    pre_place_z=(
                        pz+
                        self.post_place_z_offset
                    )

                    # XY 이동은 실제 lift 높이에서 수행
                    xy_move_z=lift_z

                    self.get_logger().info(
                        "[6/8] Cartesian XY pre-place:"
                    )

                    self.get_logger().info(
                        f"[6/8] "
                        f"({tx:.3f},{ty:.3f},{xy_move_z:.3f}) -> "
                        f"({px:.3f},{py:.3f},{xy_move_z:.3f})"
                    )

                    self.get_logger().info(
                        "[6/8] Z fixed. "
                        "Grasp orientation fixed. "
                        "Gripper CLOSE remains held."
                    )

                    # ----------------------------------------------------
                    # 1차: Cartesian XY
                    # ----------------------------------------------------

                    xy_cartesian_success=self.cartesian_xy_move(
                        tx,
                        ty,
                        px,
                        py,
                        xy_move_z,
                        qx,
                        qy,
                        qz,
                        qw
                    )

                    # ----------------------------------------------------
                    # 2차: Orientation-constrained joint-space fallback
                    # ----------------------------------------------------

                    if not xy_cartesian_success:

                        self.get_logger().warn(
                            "[6/8] Cartesian XY failed."
                        )

                        self.get_logger().warn(
                            "[6/8] Starting orientation-constrained "
                            "joint-space fallback."
                        )

                        xy_fallback_success=(
                            self.xy_joint_space_fallback(
                                tx,
                                ty,
                                px,
                                py,
                                xy_move_z,
                                qx,
                                qy,
                                qz,
                                qw
                            )
                        )

                        if not xy_fallback_success:

                            self.reset_after_failure(
                                "Cartesian XY failed and "
                                "orientation-constrained "
                                "joint-space fallback failed."
                            )

                            continue

                        self.get_logger().info(
                            "[6/8] XY joint-space fallback completed."
                        )

                    else:

                        self.get_logger().info(
                            "[6/8] Cartesian XY pre-place completed."
                        )

                    # ====================================================
                    # [6.5/8] Vertical adjustment to pre-place Z
                    #
                    # Lift Z와 pre-place Z가 다를 수 있으므로
                    # 여기서 수직 이동.
                    #
                    # Orientation 계속 고정.
                    # ====================================================

                    if abs(
                        xy_move_z-
                        pre_place_z
                    )>=0.001:

                        self.get_logger().info(
                            f"[6.5/8] Vertical move to pre-place height: "
                            f"z={xy_move_z:.3f}->{pre_place_z:.3f}"
                        )

                        if not self.cartesian_z_move(
                            px,
                            py,
                            xy_move_z,
                            pre_place_z,
                            qx,
                            qy,
                            qz,
                            qw
                        ):

                            self.reset_after_failure(
                                "Vertical move to pre-place height failed."
                            )

                            continue

                    # ====================================================
                    # [7/8] Vertical Place + Open
                    # ====================================================

                    self.get_logger().info(
                        f"[7/8] Cartesian descend to place: "
                        f"z={pre_place_z:.3f}->{pz:.3f}"
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
                            "Cartesian place descent failed."
                        )

                        continue

                    self.get_logger().info(
                        "[7/8] Place position reached."
                    )

                    # ----------------------------------------------------
                    # 여기서 처음으로 OPEN
                    # ----------------------------------------------------

                    self.get_logger().info(
                        "[7/8] Opening gripper."
                    )

                    if not self.control_gripper(
                        "OPEN"
                    ):

                        self.reset_after_failure(
                            "Gripper OPEN at place failed."
                        )

                        continue

                    time.sleep(0.6)

                    # ====================================================
                    # [8/8] Retract + Home
                    # ====================================================

                    self.get_logger().info(
                        f"[8/8] Retracting: "
                        f"z={pz:.3f}->{pre_place_z:.3f}"
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
                            "[PnP] Retract failed."
                        )

                    if not self.plan_and_execute_named_state(
                        "ready"
                    ):

                        self.get_logger().warn(
                            "[PnP] Failed to return home."
                        )

                    self.get_logger().info(
                        "================================================"
                    )

                    self.get_logger().info(
                        "[PnP] Pick & Place completed successfully."
                    )

                    self.get_logger().info(
                        "================================================"
                    )

                    self.target_pose=None
                    self.object_height=None
                    self.target_yaw=0.0
                    self.is_busy=False
                    self.state="IDLE"

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

        self.shutdown_requested=True

        if (
            self.worker.is_alive()
            and threading.current_thread() is not self.worker
        ):

            self.worker.join(
                timeout=2.0
            )


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
        rclpy.shutdown()


if __name__=="__main__":
    main()