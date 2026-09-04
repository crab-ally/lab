#!/usr/bin/env python3
"""
MuJoCo ROS 2 Bridge with FollowJointTrajectory & GripperCommand Action Servers
Supports MoveIt 2 Execution for Franka Emika Panda.

HW Spec
  - 최대 작업 반경: 0.855m
  - 그리퍼 동작 범위: 0.08m
  - 최대 하중: 2.27kg (순수 물체 무게)

Publishes
  /joint_states
  TF
  /camera/image_raw
  /camera/depth/image_raw
  /camera/depth/camera_info
  /camera/segmentation/image_raw

Arm controller 구조

  MoveIt 2
      │
      │ FollowJointTrajectory
      ▼
  MuJoCo FollowJointTrajectory Controller
      │
      ├─ trajectory validation
      ├─ monotonic trajectory clock
      ├─ q(t) interpolation
      ├─ qpos tracking
      ├─ qvel tracking
      ├─ path tolerance
      ├─ goal tolerance
      ├─ feedback
      ├─ cancel / abort
      └─ current-position hold
      │
      ▼
  MuJoCo actuator
      │
      ▼
  actual qpos / qvel
      │
      ├───────────────> /joint_states
      │
      └───────────────> controller monitoring
"""


import argparse
from pathlib import Path
import time
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String, Float64

from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from control_msgs.action import FollowJointTrajectory, GripperCommand

import mujoco
import mujoco.viewer


class MjcfBridgeNode(Node):

    def __init__(self, model, data):
        super().__init__("mujoco_ros_bridge")

        self.model = model
        self.data = data

        # ============================================================
        # Thread synchronization
        # ============================================================
        self.lock = threading.Lock()
        self.running = True
        self.cb_group = ReentrantCallbackGroup()

        # ============================================================
        # Panda joints
        # ============================================================
        self.arm_joints = [
            f"joint{i}"
            for i in range(1, 8)
        ]

        self.finger_joints = [
            "finger_joint1",
            "finger_joint2"
        ]

        self.all_joints = (
            self.arm_joints +
            self.finger_joints
        )

        self.joint_ids = {
            name: mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name
            )
            for name in self.all_joints
        }

        # ============================================================
        # Arm actuator IDs
        # ============================================================
        self.actuator_ids = {
            f"joint{i}": mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"actuator{i}"
            )
            for i in range(1, 8)
        }

        # MuJoCo actuator8
        self.GRIPPER_ACTUATOR_ID = 7

        # ============================================================
        # Finger joint / DOF IDs
        #
        # qfrc_applied는 joint ID가 아니라 DOF index를 사용한다.
        # 따라서 jnt_dofadr를 통해 정확한 DOF를 얻는다.
        # ============================================================
        self.finger_joint_ids = {
            name: self.joint_ids[name]
            for name in self.finger_joints
        }

        self.finger_dof_ids = {
            name: self.model.jnt_dofadr[
                self.finger_joint_ids[name]
            ]
            for name in self.finger_joints
        }

        self.finger1_dof_id = (
            self.finger_dof_ids["finger_joint1"]
        )

        self.finger2_dof_id = (
            self.finger_dof_ids["finger_joint2"]
        )

        # ============================================================
        # Extra gripping force
        #
        # finger당 적용되는 추가 generalized force [N]
        #
        # CLOSE 방향이 negative q 방향이므로 -force 사용.
        # ============================================================
        self.GRIPPER_EXTRA_FORCE = 30.0

        # CLOSE 이후 LIFT까지 압착력을 유지
        self.gripper_force_enabled = False

        # ============================================================
        # Camera
        # ============================================================
        self.camera_name = "ceiling_camera"

        self.camera_width = 640
        self.camera_height = 480

        self.camera_rate = 10.0

        self.camera_optical_frame = (
            "ceiling_camera_optical_frame"
        )

        self.camera_link_frame = (
            "ceiling_camera_link"
        )

        # ============================================================
        # Panda geom IDs
        # ============================================================
        self.finger_geom_ids = (
            self._collect_finger_geom_ids()
        )

        self.panda_geom_ids = (
            self._collect_panda_geom_ids()
        )

        # ============================================================
        # ROS publishers
        # ============================================================
        self.joint_pub = self.create_publisher(
            JointState,
            "/joint_states",
            10
        )

        self.rgb_pub = self.create_publisher(
            Image,
            "/camera/image_raw",
            10
        )

        self.depth_pub = self.create_publisher(
            Image,
            "/camera/depth/image_raw",
            10
        )

        self.seg_pub = self.create_publisher(
            Image,
            "/camera/segmentation/image_raw",
            10
        )

        self.info_pub = self.create_publisher(
            CameraInfo,
            "/camera/depth/camera_info",
            10
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.static_tf_broadcaster = (
            StaticTransformBroadcaster(self)
        )

        # ============================================================
        # Gripper topics
        # ============================================================
        self.create_subscription(
            String,
            "/panda_gripper/cmd",
            self.gripper_string_callback,
            10,
            callback_group=self.cb_group
        )

        self.create_subscription(
            Float64,
            "/panda_gripper/command",
            self.gripper_float_callback,
            10,
            callback_group=self.cb_group
        )

        # ============================================================
        # Arm Action
        # ============================================================
        self.arm_action = ActionServer(
            self,
            FollowJointTrajectory,
            "/panda_arm_controller/follow_joint_trajectory",
            execute_callback=self.execute_arm,
            goal_callback=self.arm_goal,
            cancel_callback=self.cancel_goal,
            callback_group=self.cb_group
        )

        # ============================================================
        # Gripper Action
        # ============================================================
        self.gripper_action = ActionServer(
            self,
            GripperCommand,
            "/panda_hand_controller/gripper_action",
            execute_callback=self.execute_gripper,
            goal_callback=self.gripper_goal,
            cancel_callback=self.cancel_goal,
            callback_group=self.cb_group
        )

        # ============================================================
        # Motion parameters
        # ============================================================
        self.home_qpos = np.array(
            [
                0.0,
                -0.785398,
                0.0,
                -2.35619,
                0.0,
                1.57079,
                0.785398
            ],
            dtype=float
        )

        # ------------------------------------------------------------
        # Goal tolerance
        #
        # trajectory 종료 시 실제 qpos가 최종 목표와 이 값 이내인지
        # 검사한다.
        # ------------------------------------------------------------
        self.POSITION_TOLERANCE = 0.01
        self.VELOCITY_TOLERANCE = 0.1

        # ------------------------------------------------------------
        # Path tolerance
        #
        # trajectory 실행 도중 실제 qpos와 desired qpos 차이가
        # 이 값을 초과하면 CONTROL_FAILED가 아니라
        # PATH_TOLERANCE_VIOLATED를 반환한다.
        #
        # 너무 작게 잡으면 MuJoCo position actuator의 tracking
        # 특성 때문에 정상적인 trajectory도 실패할 수 있으므로
        # 초기값은 0.15 rad로 둔다.
        # ------------------------------------------------------------
        self.PATH_POSITION_TOLERANCE = 0.20
        self.PATH_VELOCITY_TOLERANCE = 2.0

        # ------------------------------------------------------------
        # Controller update rate
        #
        # trajectory command를 wall-clock 기준으로 약 200 Hz로
        # 갱신한다.
        # ------------------------------------------------------------
        self.CONTROLLER_RATE = 200.0
        self.CONTROLLER_PERIOD = (
            1.0 / self.CONTROLLER_RATE
        )

        # ------------------------------------------------------------
        # Trajectory 시작 전 현재 상태와 첫 waypoint의 차이를
        # 확인하는 tolerance.
        # ------------------------------------------------------------
        self.START_POSITION_TOLERANCE = 0.10

        # ------------------------------------------------------------
        # 최종 settle timeout
        # ------------------------------------------------------------
        self.SETTLE_TIMEOUT = 5.0

        # ============================================================
        # Gripper parameters
        # ============================================================
        self.GRIPPER_STABLE_TIME = 0.15
        self.GRIPPER_TIMEOUT = 2.0
        self.GRIPPER_CONTACT_STABLE_TIME = 0.15

        self.GRIPPER_CTRL_MIN = 0.0
        self.GRIPPER_CTRL_MAX = 255.0

        # ============================================================
        # Active arm goal
        #
        # 하나의 Panda arm trajectory만 실행하도록 관리한다.
        # ============================================================
        self.active_arm_goal = None
        self.active_arm_goal_lock = threading.Lock()

        # ============================================================
        # Initial pose
        # ============================================================
        self._set_initial_pose()

        self.publish_static_camera_tf()

        # ============================================================
        # Camera rendering thread
        # ============================================================
        self.camera_thread = threading.Thread(
            target=self._camera_render_loop,
            daemon=True
        )

        self.camera_thread.start()

        self.get_logger().info(
            "MuJoCo ROS 2 Bridge started."
        )

        self.get_logger().info(
            f"[GRIPPER] Extra gripping force="
            f"{self.GRIPPER_EXTRA_FORCE:.1f} N per finger"
        )

        self.get_logger().info(
            f"[GRIPPER] finger_joint1 DOF="
            f"{self.finger1_dof_id}, "
            f"finger_joint2 DOF="
            f"{self.finger2_dof_id}"
        )

        self.get_logger().info(
            "[ARM CONTROLLER] "
            f"rate={self.CONTROLLER_RATE:.1f} Hz, "
            f"path_tol={self.PATH_POSITION_TOLERANCE:.3f} rad, "
            f"goal_tol={self.POSITION_TOLERANCE:.3f} rad"
        )

    # ================================================================
    # Extra gripping force
    # ================================================================
    def _clear_gripper_extra_force(self):
        """
        qfrc_applied에서 finger에 넣었던 추가 힘을 제거한다.

        실제로는 physics step 직전에 전체 qfrc_applied를 초기화하므로
        여기서는 명시적으로 finger DOF만 0으로 만든다.
        """
        with self.lock:
            self.data.qfrc_applied[
                self.finger1_dof_id
            ] = 0.0

            self.data.qfrc_applied[
                self.finger2_dof_id
            ] = 0.0

    def _apply_gripper_extra_force(self):
        """
        양쪽 finger에 추가 압착력을 적용한다.

        finger joint의 positive 방향이 OPEN이므로
        negative generalized force를 사용하여 CLOSE 방향으로 힘을 준다.
        """
        if not self.gripper_force_enabled:
            return

        force = -abs(
            float(self.GRIPPER_EXTRA_FORCE)
        )

        self.data.qfrc_applied[
            self.finger1_dof_id
        ] += force

        self.data.qfrc_applied[
            self.finger2_dof_id
        ] += force

    def _prepare_physics_step(self):
        """
        모든 MuJoCo physics step 직전에 호출한다.

        qfrc_applied는 persistent actuator가 아니므로
        매 step마다 원하는 외력을 다시 넣어준다.
        """
        self.data.qfrc_applied[:] = 0.0

        if self.gripper_force_enabled:
            self._apply_gripper_extra_force()

    def _gripper_extra_force_values(self):
        with self.lock:
            f1 = float(
                self.data.qfrc_applied[
                    self.finger1_dof_id
                ]
            )

            f2 = float(
                self.data.qfrc_applied[
                    self.finger2_dof_id
                ]
            )

        return f1, f2

    # ================================================================
    # Static camera TF
    # ================================================================
    def publish_static_camera_tf(self):
        stamp = self.get_clock().now().to_msg()
        transforms = []

        cam_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.camera_name
        )

        cam_body = self.model.cam_bodyid[cam_id]

        # ------------------------------------------------------------
        # world -> ceiling_camera_link
        # ------------------------------------------------------------
        cam_name = mujoco.mj_id2name(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            cam_body
        )

        if cam_name:
            t = TransformStamped()

            t.header.stamp = stamp
            t.header.frame_id = "world"
            t.child_frame_id = self.camera_link_frame

            p = self.data.xpos[cam_body]

            q = self._mat_to_quat(
                self.data.xmat[cam_body].reshape(3, 3)
            )

            t.transform.translation.x = float(p[0])
            t.transform.translation.y = float(p[1])
            t.transform.translation.z = float(p[2])

            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            transforms.append(t)

        # ------------------------------------------------------------
        # camera_link -> optical
        # ------------------------------------------------------------
        t = TransformStamped()

        t.header.stamp = stamp
        t.header.frame_id = self.camera_link_frame
        t.child_frame_id = self.camera_optical_frame

        t.transform.rotation.x = 1.0
        t.transform.rotation.w = 0.0

        transforms.append(t)

        self.static_tf_broadcaster.sendTransform(
            transforms
        )

    # ================================================================
    # Panda geom collection
    # ================================================================
    def _collect_panda_geom_ids(self):
        root = self.model.body("link0").id

        ids = set()

        for gid in range(self.model.ngeom):
            body = int(
                self.model.geom_bodyid[gid]
            )

            while body > 0:
                if body == root:
                    ids.add(gid)
                    break

                body = int(
                    self.model.body_parentid[body]
                )

        return ids

    def _collect_finger_geom_ids(self):
        ids = set()

        for gid in range(self.model.ngeom):
            body = int(
                self.model.geom_bodyid[gid]
            )

            while body > 0:
                name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body
                )

                if name and (
                    "finger" in name
                    or
                    "hand" in name
                ):
                    ids.add(gid)
                    break

                body = int(
                    self.model.body_parentid[body]
                )

        return ids

    # ================================================================
    # Initial pose
    # ================================================================
    def _set_initial_pose(self):
        for i, name in enumerate(
            self.arm_joints
        ):
            jid = self.joint_ids[name]

            self.data.qpos[
                self.model.jnt_qposadr[jid]
            ] = self.home_qpos[i]

        # 초기에는 압착력 OFF
        self.gripper_force_enabled = False

        self.data.qfrc_applied[:] = 0.0

        self.data.ctrl[
            self.GRIPPER_ACTUATOR_ID
        ] = self.GRIPPER_CTRL_MAX

        # arm actuator도 초기 자세로 설정
        for i, name in enumerate(
            self.arm_joints
        ):
            self.data.ctrl[
                self.actuator_ids[name]
            ] = self.home_qpos[i]

        mujoco.mj_forward(
            self.model,
            self.data
        )

    # ================================================================
    # Camera render loop
    # ================================================================
    def _camera_render_loop(self):
        renderer = mujoco.Renderer(
            self.model,
            height=self.camera_height,
            width=self.camera_width
        )

        depth_renderer = mujoco.Renderer(
            self.model,
            height=self.camera_height,
            width=self.camera_width
        )

        seg_renderer = mujoco.Renderer(
            self.model,
            height=self.camera_height,
            width=self.camera_width
        )

        depth_renderer.enable_depth_rendering()
        seg_renderer.enable_segmentation_rendering()

        render_data = mujoco.MjData(
            self.model
        )

        period = 1.0 / self.camera_rate
        next_time = time.monotonic()

        while self.running:

            next_time += period

            # --------------------------------------------------------
            # Copy simulation state
            # --------------------------------------------------------
            with self.lock:

                render_data.qpos[:] = (
                    self.data.qpos
                )

                render_data.qvel[:] = (
                    self.data.qvel
                )

                render_data.act[:] = (
                    self.data.act
                )

                if self.model.nmocap:
                    render_data.mocap_pos[:] = (
                        self.data.mocap_pos
                    )

                    render_data.mocap_quat[:] = (
                        self.data.mocap_quat
                    )

                mujoco.mj_forward(
                    self.model,
                    render_data
                )

            # --------------------------------------------------------
            # RGB
            # --------------------------------------------------------
            renderer.update_scene(
                render_data,
                camera=self.camera_name
            )

            rgb = np.asarray(
                renderer.render()
            ).copy()

            # --------------------------------------------------------
            # Depth
            # --------------------------------------------------------
            depth_renderer.update_scene(
                render_data,
                camera=self.camera_name
            )

            depth = np.asarray(
                depth_renderer.render()
            ).copy()

            # --------------------------------------------------------
            # Segmentation
            # --------------------------------------------------------
            seg_renderer.update_scene(
                render_data,
                camera=self.camera_name
            )

            seg_raw = np.asarray(
                seg_renderer.render()
            ).copy()

            seg_id = (
                seg_raw[:, :, 0]
                .astype(np.int32)
            )

            seg_type = (
                seg_raw[:, :, 1]
                .astype(np.int32)
            )

            geom_mask = (
                seg_type ==
                int(mujoco.mjtObj.mjOBJ_GEOM)
            )

            panda_mask = (
                geom_mask
                &
                np.isin(
                    seg_id,
                    np.asarray(
                        list(self.panda_geom_ids),
                        dtype=np.int32
                    )
                )
            )

            seg = seg_id.copy()

            seg[~geom_mask] = 0
            seg[panda_mask] = 0

            # --------------------------------------------------------
            # Same timestamp
            # --------------------------------------------------------
            stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )

            # --------------------------------------------------------
            # RGB
            # --------------------------------------------------------
            self.rgb_pub.publish(
                self._image_msg(
                    rgb,
                    "rgb8",
                    3,
                    stamp
                )
            )

            # --------------------------------------------------------
            # Depth
            # --------------------------------------------------------
            self.depth_pub.publish(
                self._image_msg(
                    depth,
                    "32FC1",
                    4,
                    stamp
                )
            )

            # --------------------------------------------------------
            # Segmentation
            # --------------------------------------------------------
            self.seg_pub.publish(
                self._image_msg(
                    seg,
                    "32SC1",
                    4,
                    stamp
                )
            )

            # --------------------------------------------------------
            # CameraInfo
            # --------------------------------------------------------
            self.info_pub.publish(
                self._camera_info(stamp)
            )

            # --------------------------------------------------------
            # Maintain camera rate
            # --------------------------------------------------------
            sleep_time = (
                next_time -
                time.monotonic()
            )

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.monotonic()

        renderer.close()
        depth_renderer.close()
        seg_renderer.close()

    def _image_msg(
        self,
        data,
        encoding,
        bpp,
        stamp
    ):
        msg = Image()

        msg.header.stamp = stamp
        msg.header.frame_id = (
            self.camera_optical_frame
        )

        msg.height = self.camera_height
        msg.width = self.camera_width

        msg.encoding = encoding
        msg.is_bigendian = False

        msg.step = (
            self.camera_width *
            bpp
        )

        msg.data = data.astype(
            np.uint8
            if encoding == "rgb8"
            else
            np.float32
            if encoding == "32FC1"
            else
            np.int32
        ).tobytes()

        return msg

    # ================================================================
    # Camera info
    # ================================================================
    def _camera_info(self, stamp):
        cid = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.camera_name
        )

        fovy = self.model.cam_fovy[cid]

        fy = (
            self.camera_height / 2.0
        ) / np.tan(
            np.deg2rad(fovy) / 2.0
        )

        fx = fy

        cx = (
            self.camera_width - 1
        ) / 2.0

        cy = (
            self.camera_height - 1
        ) / 2.0

        msg = CameraInfo()

        msg.header.stamp = stamp
        msg.header.frame_id = (
            self.camera_optical_frame
        )

        msg.width = self.camera_width
        msg.height = self.camera_height

        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        return msg

    # ================================================================
    # Joint states
    # ================================================================
    def publish_joint_states(self):
        msg = JointState()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        for name in self.all_joints:

            jid = self.joint_ids[name]

            msg.name.append(name)

            msg.position.append(
                float(
                    self.data.qpos[
                        self.model.jnt_qposadr[jid]
                    ]
                )
            )

            msg.velocity.append(
                float(
                    self.data.qvel[
                        self.model.jnt_dofadr[jid]
                    ]
                )
            )

        self.joint_pub.publish(msg)

    # ================================================================
    # Rotation / quaternion
    # ================================================================
    def _mat_to_quat(self, R):
        q = np.zeros(4)

        mujoco.mju_mat2Quat(
            q,
            R.reshape(-1)
        )

        return (
            float(q[1]),
            float(q[2]),
            float(q[3]),
            float(q[0])
        )

    # ================================================================
    # Relative transform
    # ================================================================
    def _relative_transform(
        self,
        body_id,
        parent_id
    ):
        p = self.data.xpos[body_id]
        pp = self.data.xpos[parent_id]

        R = self.data.xmat[
            body_id
        ].reshape(3, 3)

        Rp = self.data.xmat[
            parent_id
        ].reshape(3, 3)

        return (
            Rp.T @ (p - pp),
            Rp.T @ R
        )

    # ================================================================
    # TF
    # ================================================================
    def publish_tf(self):
        stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        root = self.model.body(
            "link0"
        ).id

        # ------------------------------------------------------------
        # world -> link0
        # ------------------------------------------------------------
        t = TransformStamped()

        t.header.stamp = stamp
        t.header.frame_id = "world"
        t.child_frame_id = "link0"

        p = self.data.xpos[root]

        q = self._mat_to_quat(
            self.data.xmat[
                root
            ].reshape(3, 3)
        )

        t.transform.translation.x = (
            float(p[0])
        )

        t.transform.translation.y = (
            float(p[1])
        )

        t.transform.translation.z = (
            float(p[2])
        )

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

        # ------------------------------------------------------------
        # Panda body TF
        # ------------------------------------------------------------
        for bid in range(
            1,
            self.model.nbody
        ):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                bid
            )

            if not name or name in (
                "link0",
                self.camera_link_frame
            ):
                continue

            parent = self.model.body_parentid[
                bid
            ]

            parent_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    parent
                )
            )

            if not parent_name or parent == 0:
                continue

            pos, rot = (
                self._relative_transform(
                    bid,
                    parent
                )
            )

            t = TransformStamped()

            t.header.stamp = stamp
            t.header.frame_id = parent_name
            t.child_frame_id = name

            t.transform.translation.x = (
                float(pos[0])
            )

            t.transform.translation.y = (
                float(pos[1])
            )

            t.transform.translation.z = (
                float(pos[2])
            )

            q = self._mat_to_quat(rot)

            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.tf_broadcaster.sendTransform(t)

    # ================================================================
    # Arm action callbacks
    # ================================================================
    def arm_goal(self, goal):
        """
        FollowJointTrajectory goal validation.

        실제 trajectory 내용의 상세 validation은 execute_arm()
        내부에서 수행한다.
        """
        if not goal.trajectory.joint_names:
            return GoalResponse.REJECT

        with self.active_arm_goal_lock:
            if self.active_arm_goal is not None:
                self.get_logger().warn(
                    "[ARM] Existing arm goal is active. "
                    "Rejecting new goal."
                )
                return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def gripper_goal(self, goal):
        return GoalResponse.ACCEPT

    def cancel_goal(self, goal):
        return CancelResponse.ACCEPT

    # ================================================================
    # Simulation Time
    # ================================================================
    def _get_sim_time(self):
        """
        MuJoCo 물리 시뮬레이션 시간(data.time)을 반환한다.
        """
        with self.lock:
            return float(self.data.time)

    # ================================================================
    # Read actual arm state
    # ================================================================
    def _get_arm_state(self):
        """
        현재 MuJoCo 실제 arm qpos / qvel을 읽는다.
        """
        with self.lock:

            q = np.array([
                self.data.qpos[
                    self.model.jnt_qposadr[
                        self.joint_ids[name]
                    ]
                ]
                for name in self.arm_joints
            ], dtype=float)

            v = np.array([
                self.data.qvel[
                    self.model.jnt_dofadr[
                        self.joint_ids[name]
                    ]
                ]
                for name in self.arm_joints
            ], dtype=float)

        return q, v

    # ================================================================
    # Hold current arm position
    # ================================================================
    def _hold_current_arm_position(self):
        """
        현재 실제 qpos를 읽어서 MuJoCo position actuator의
        target으로 유지한다.

        cancel / abort 시 arm을 마지막 trajectory command로
        계속 보내지 않고 현재 위치를 유지하도록 한다.
        """
        q, _ = self._get_arm_state()

        with self.lock:
            for i, name in enumerate(
                self.arm_joints
            ):
                self.data.ctrl[
                    self.actuator_ids[name]
                ] = q[i]

        return q

    # ================================================================
    # Parse trajectory points
    # ================================================================
    def _parse_trajectory(self, traj):
        """
        ROS FollowJointTrajectory 메시지를 controller 내부에서
        사용할 numpy 배열 형태로 변환한다.

        반환:
          times
          positions
          velocities

        velocity가 제공되지 않은 waypoint는 NaN으로 둔다.
        """

        name_to_idx = {
            name: i
            for i, name in enumerate(
                traj.joint_names
            )
        }

        # ------------------------------------------------------------
        # 모든 Panda arm joint가 trajectory에 존재해야 한다.
        # ------------------------------------------------------------
        for name in self.arm_joints:
            if name not in name_to_idx:
                raise ValueError(
                    f"Missing joint in trajectory: {name}"
                )

        times = []
        positions = []
        velocities = []

        prev_t = -1.0

        for point_index, point in enumerate(
            traj.points
        ):

            t = (
                float(point.time_from_start.sec)
                +
                float(
                    point.time_from_start.nanosec
                ) * 1e-9
            )

            # --------------------------------------------------------
            # trajectory time은 반드시 증가해야 한다.
            # --------------------------------------------------------
            if t < 0.0:
                raise ValueError(
                    f"Negative trajectory time at point "
                    f"{point_index}"
                )

            if t <= prev_t:
                raise ValueError(
                    f"Non-increasing trajectory time at point "
                    f"{point_index}: {t} <= {prev_t}"
                )

            if len(point.positions) < len(
                traj.joint_names
            ):
                raise ValueError(
                    f"Invalid position array at point "
                    f"{point_index}"
                )

            q = np.array([
                point.positions[
                    name_to_idx[name]
                ]
                for name in self.arm_joints
            ], dtype=float)

            # --------------------------------------------------------
            # velocity가 trajectory에 들어있으면 사용한다.
            # 없으면 NaN.
            # --------------------------------------------------------
            if len(point.velocities) >= len(
                traj.joint_names
            ):
                v = np.array([
                    point.velocities[
                        name_to_idx[name]
                    ]
                    for name in self.arm_joints
                ], dtype=float)

                if not np.all(
                    np.isfinite(v)
                ):
                    v[:] = np.nan

            else:
                v = np.full(
                    len(self.arm_joints),
                    np.nan,
                    dtype=float
                )

            times.append(t)
            positions.append(q)
            velocities.append(v)

            prev_t = t

        return (
            np.asarray(times, dtype=float),
            np.asarray(positions, dtype=float),
            np.asarray(velocities, dtype=float)
        )

    # ================================================================
    # Desired trajectory interpolation
    # ================================================================
    def _interpolate_segment(
        self,
        q0,
        q1,
        v0,
        v1,
        duration,
        tau
    ):
        """
        하나의 trajectory segment에서 desired q / qdot을 계산한다.

        velocity 정보가 있으면 cubic Hermite interpolation을 사용한다.

        없으면 선형 interpolation을 사용한다.

        cubic Hermite:

          q(s) =
            h00 q0 +
            h10 T v0 +
            h01 q1 +
            h11 T v1

        여기서 s = tau / T

        position actuator에 전달할 것은 q_desired이며,
        qdot_desired는 monitoring / feedback에 사용한다.
        """

        if duration <= 1e-9:
            return q1.copy(), np.zeros_like(q1)

        s = np.clip(
            tau / duration,
            0.0,
            1.0
        )

        # ------------------------------------------------------------
        # velocity가 모두 존재하면 cubic Hermite
        # ------------------------------------------------------------
        if (
            np.all(np.isfinite(v0))
            and
            np.all(np.isfinite(v1))
        ):
            h00 = (
                2.0 * s**3
                - 3.0 * s**2
                + 1.0
            )

            h10 = (
                s**3
                - 2.0 * s**2
                + s
            )

            h01 = (
                -2.0 * s**3
                + 3.0 * s**2
            )

            h11 = (
                s**3
                - s**2
            )

            q = (
                h00 * q0
                +
                h10 * duration * v0
                +
                h01 * q1
                +
                h11 * duration * v1
            )

            dh00 = (
                6.0 * s**2
                - 6.0 * s
            )

            dh10 = (
                3.0 * s**2
                - 4.0 * s
                + 1.0
            )

            dh01 = (
                -6.0 * s**2
                + 6.0 * s
            )

            dh11 = (
                3.0 * s**2
                - 2.0 * s
            )

            qdot = (
                dh00 * q0 / duration
                +
                dh10 * v0
                +
                dh01 * q1 / duration
                +
                dh11 * v1
            )

            return q, qdot

        # ------------------------------------------------------------
        # velocity 정보가 없으면 linear interpolation
        # ------------------------------------------------------------
        q = (
            q0
            +
            (q1 - q0) * s
        )

        qdot = (
            (q1 - q0) / duration
        )

        return q, qdot

    # ================================================================
    # Desired trajectory state
    # ================================================================
    def _trajectory_state_at(
        self,
        elapsed,
        times,
        positions,
        velocities
    ):
        """
        trajectory의 elapsed time 위치에서

          desired q
          desired qdot

        을 계산한다.
        """

        if elapsed <= times[0]:
            return (
                positions[0].copy(),
                (
                    velocities[0].copy()
                    if np.all(
                        np.isfinite(
                            velocities[0]
                        )
                    )
                    else
                    np.zeros(
                        len(self.arm_joints)
                    )
                )
            )

        if elapsed >= times[-1]:
            return (
                positions[-1].copy(),
                (
                    velocities[-1].copy()
                    if np.all(
                        np.isfinite(
                            velocities[-1]
                        )
                    )
                    else
                    np.zeros(
                        len(self.arm_joints)
                    )
                )
            )

        index = (
            np.searchsorted(
                times,
                elapsed,
                side="right"
            ) - 1
        )

        index = max(
            0,
            min(
                index,
                len(times) - 2
            )
        )

        t0 = times[index]
        t1 = times[index + 1]

        duration = t1 - t0
        tau = elapsed - t0

        return self._interpolate_segment(
            positions[index],
            positions[index + 1],
            velocities[index],
            velocities[index + 1],
            duration,
            tau
        )

    # ================================================================
    # Extract tolerance
    # ================================================================
    def _extract_joint_tolerance(
        self,
        tolerance_list,
        default
    ):
        """
        FollowJointTrajectory goal의 JointTolerance에서
        Panda joint별 tolerance를 읽는다.

        tolerance가 지정되지 않은 joint는 default 사용.
        """

        result = {
            name: float(default)
            for name in self.arm_joints
        }

        for tol in tolerance_list:

            if tol.name not in result:
                continue

            # position tolerance
            if (
                hasattr(tol, "position")
                and
                tol.position > 0.0
            ):
                result[
                    tol.name
                ] = float(
                    tol.position
                )

        return np.array([
            result[name]
            for name in self.arm_joints
        ], dtype=float)

    # ================================================================
    # Extract velocity tolerance
    # ================================================================
    def _extract_velocity_tolerance(
        self,
        tolerance_list,
        default
    ):
        result = {
            name: float(default)
            for name in self.arm_joints
        }

        for tol in tolerance_list:

            if tol.name not in result:
                continue

            if (
                hasattr(tol, "velocity")
                and
                tol.velocity > 0.0
            ):
                result[
                    tol.name
                ] = float(
                    tol.velocity
                )

        return np.array([
            result[name]
            for name in self.arm_joints
        ], dtype=float)

    # ================================================================
    # Publish FollowJointTrajectory feedback
    # ================================================================
    def _publish_arm_feedback(
        self,
        goal_handle,
        desired_q,
        desired_v,
        actual_q,
        actual_v
    ):
        """
        FollowJointTrajectory feedback를 publish한다.

        desired / actual / error를 함께 전달한다.
        """

        feedback = (
            FollowJointTrajectory.Feedback()
        )

        feedback.joint_names = list(
            self.arm_joints
        )

        feedback.desired.positions = (
            desired_q.tolist()
        )

        feedback.desired.velocities = (
            desired_v.tolist()
        )

        feedback.actual.positions = (
            actual_q.tolist()
        )

        feedback.actual.velocities = (
            actual_v.tolist()
        )

        error_q = (
            desired_q - actual_q
        )

        error_v = (
            desired_v - actual_v
        )

        feedback.error.positions = (
            error_q.tolist()
        )

        feedback.error.velocities = (
            error_v.tolist()
        )

        goal_handle.publish_feedback(
            feedback
        )

    # ================================================================
    # Arm execution
    # ================================================================
    def execute_arm(self, goal_handle):
        """
        MuJoCo FollowJointTrajectory controller.

        기존 구현과 가장 큰 차이:

        기존:
          waypoint
             ↓
          선형 interpolation
             ↓
          time.sleep()
             ↓
          ctrl

        현재:
          trajectory
             ↓
          validation
             ↓
          monotonic trajectory clock
             ↓
          q(t) interpolation
             ↓
          actual qpos/qvel read
             ↓
          tracking error
             ↓
          path tolerance
             ↓
          feedback
             ↓
          final goal tolerance

        중요:
        이 함수는 MuJoCo physics를 직접 실행하지 않는다.

        physics loop는 별도의 main thread에서 계속 실행되고,
        이 controller thread는 desired actuator target을
        주기적으로 갱신한다.
        """

        goal=goal_handle.request
        traj=goal.trajectory

        # ============================================================
        # Active goal 등록
        # ============================================================
        with self.active_arm_goal_lock:
            self.active_arm_goal = goal_handle

        try:

            # ========================================================
            # 1. Basic validation
            # ========================================================
            if not traj.points:

                self.get_logger().error(
                    "[ARM] Empty trajectory."
                )

                goal_handle.abort()

                result = (
                    FollowJointTrajectory.Result()
                )

                result.error_code = (
                    FollowJointTrajectory.Result.INVALID_GOAL
                )

                return result

            # --------------------------------------------------------
            # joint 이름 validation
            # --------------------------------------------------------
            try:
                (
                    times,
                    positions,
                    velocities
                ) = self._parse_trajectory(
                    traj
                )

            except Exception as e:

                self.get_logger().error(
                    f"[ARM] Trajectory validation failed: {e}"
                )

                goal_handle.abort()

                result = (
                    FollowJointTrajectory.Result()
                )

                result.error_code = (
                    FollowJointTrajectory.Result.INVALID_JOINTS
                )

                return result

            # ========================================================
            # 2. Read actual current state
            # ========================================================
            current_q, current_v = (
                self._get_arm_state()
            )

            first_q = positions[0]

            start_error = (
                first_q - current_q
            )

            max_start_error = float(
                np.max(
                    np.abs(start_error)
                )
            )

            self.get_logger().info(
                "[ARM] "
                f"trajectory points={len(times)}, "
                f"duration={times[-1]:.6f}s"
            )

            self.get_logger().info(
                "[ARM] "
                f"current_q={np.round(current_q,6).tolist()}"
            )

            self.get_logger().info(
                "[ARM] "
                f"first_q={np.round(first_q,6).tolist()}"
            )

            self.get_logger().info(
                "[ARM] "
                f"max_start_state_error="
                f"{max_start_error:.6f} rad"
            )

            # --------------------------------------------------------
            # 첫 waypoint와 현재 실제 위치가 지나치게 다르면
            # trajectory 시작 자체가 안전하지 않다고 판단.
            # --------------------------------------------------------
            if (
                max_start_error
                >
                self.START_POSITION_TOLERANCE
            ):

                self.get_logger().error(
                    "[ARM] Start state mismatch: "
                    f"{max_start_error:.6f} rad > "
                    f"{self.START_POSITION_TOLERANCE:.6f} rad"
                )

                self._hold_current_arm_position()

                goal_handle.abort()

                result = (
                    FollowJointTrajectory.Result()
                )

                result.error_code = (
                    FollowJointTrajectory.Result.INVALID_GOAL
                )

                return result

            # ========================================================
            # 3. Parse path / goal tolerances
            # ========================================================
            path_pos_tol = (
                self._extract_joint_tolerance(
                    goal.path_tolerance,
                    self.PATH_POSITION_TOLERANCE
                )
            )

            path_vel_tol = (
                self._extract_velocity_tolerance(
                    goal.path_tolerance,
                    self.PATH_VELOCITY_TOLERANCE
                )
            )

            goal_pos_tol = (
                self._extract_joint_tolerance(
                    goal.goal_tolerance,
                    self.POSITION_TOLERANCE
                )
            )

            goal_vel_tol = (
                self._extract_velocity_tolerance(
                    goal.goal_tolerance,
                    self.VELOCITY_TOLERANCE
                )
            )

            self.get_logger().info(
                "[ARM] "
                f"path_pos_tol="
                f"{np.round(path_pos_tol,4).tolist()}"
            )

            self.get_logger().info(
                "[ARM] "
                f"goal_pos_tol="
                f"{np.round(goal_pos_tol,4).tolist()}"
            )

            # ========================================================
            # 4. Start controller clock (MuJoCo simulation time)
            # ========================================================
            start_time = self._get_sim_time()

            last_feedback_time = start_time

            feedback_period = 0.05

            last_log_time = start_time

            # ========================================================
            # 5. Main trajectory execution loop
            # ========================================================
            while True:

                # ----------------------------------------------------
                # Cancel check
                # ----------------------------------------------------
                if goal_handle.is_cancel_requested:

                    self.get_logger().warn(
                        "[ARM] Cancel requested. "
                        "Holding current position."
                    )

                    self._hold_current_arm_position()

                    goal_handle.canceled()

                    return (
                        FollowJointTrajectory.Result()
                    )

                # ----------------------------------------------------
                # Controller clock
                #
                # MuJoCo 물리 시뮬레이션 시간(data.time)을 기준으로 하여
                # 실시간 계수(RTF)의 변화에 관계없이 물리 엔진과
                # trajectory 타이밍이 완벽하게 1:1 동기화된다.
                # ----------------------------------------------------
                now = self._get_sim_time()

                elapsed = (
                    now - start_time
                )

                # ----------------------------------------------------
                # Desired trajectory state
                # ----------------------------------------------------
                (
                    desired_q,
                    desired_v
                ) = self._trajectory_state_at(
                    elapsed,
                    times,
                    positions,
                    velocities
                )

                # ----------------------------------------------------
                # Actual state
                # ----------------------------------------------------
                actual_q, actual_v = (
                    self._get_arm_state()
                )

                # ----------------------------------------------------
                # Tracking error
                # ----------------------------------------------------
                position_error = (
                    desired_q - actual_q
                )

                velocity_error = (
                    desired_v - actual_v
                )

                max_position_error = float(
                    np.max(
                        np.abs(position_error)
                    )
                )

                max_velocity_error = float(
                    np.max(
                        np.abs(velocity_error)
                    )
                )

                # ====================================================
                # Path tolerance
                #
                # trajectory 실행 중에 actual qpos가 desired qpos를
                # 지속적으로 크게 벗어나면 abort.
                #
                # 단순 max 값 하나가 아니라 joint별 tolerance를
                # 사용한다.
                # ====================================================
                position_violation = (
                    np.abs(position_error)
                    >
                    path_pos_tol
                )

                velocity_violation = (
                    np.abs(velocity_error)
                    >
                    path_vel_tol
                )

                if np.any(
                    position_violation
                ):

                    bad_indices = np.where(
                        position_violation
                    )[0]

                    self.get_logger().error(
                        "[ARM] PATH POSITION TOLERANCE "
                        "VIOLATED: "
                        f"elapsed={elapsed:.3f}s, "
                        f"max_error="
                        f"{max_position_error:.6f} rad, "
                        f"joints="
                        f"{[self.arm_joints[i] for i in bad_indices]}"
                    )

                    self._hold_current_arm_position()

                    goal_handle.abort()

                    result = (
                        FollowJointTrajectory.Result()
                    )

                    result.error_code = (
                        FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    )

                    return result

                # ----------------------------------------------------
                # velocity tolerance는 trajectory velocity 정보가
                # 실제로 있을 때 의미가 있으므로 desired_v가
                # non-zero / finite인지 확인한다.
                # ----------------------------------------------------
                if (
                    np.any(
                        np.abs(desired_v)
                        >
                        1e-9
                    )
                    and
                    np.any(
                        velocity_violation
                    )
                ):

                    bad_indices = np.where(
                        velocity_violation
                    )[0]

                    self.get_logger().error(
                        "[ARM] PATH VELOCITY TOLERANCE "
                        "VIOLATED: "
                        f"elapsed={elapsed:.3f}s, "
                        f"max_velocity_error="
                        f"{max_velocity_error:.6f} rad/s, "
                        f"joints="
                        f"{[self.arm_joints[i] for i in bad_indices]}"
                    )

                    self._hold_current_arm_position()

                    goal_handle.abort()

                    result = (
                        FollowJointTrajectory.Result()
                    )

                    result.error_code = (
                        FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    )

                    return result

                # ====================================================
                # Command actuator
                #
                # MuJoCo position actuator에 desired q를 넣는다.
                #
                # 중요한 점:
                # 실제 qpos를 ctrl에 넣는 것이 아니라
                # trajectory에서 계산한 desired q를 넣는다.
                # ====================================================
                with self.lock:

                    for i, name in enumerate(
                        self.arm_joints
                    ):

                        self.data.ctrl[
                            self.actuator_ids[name]
                        ] = desired_q[i]

                # ====================================================
                # Feedback
                # ====================================================
                if (
                    now -
                    last_feedback_time
                    >=
                    feedback_period
                ):

                    self._publish_arm_feedback(
                        goal_handle,
                        desired_q,
                        desired_v,
                        actual_q,
                        actual_v
                    )

                    last_feedback_time = now

                # ====================================================
                # Diagnostics
                # ====================================================
                if (
                    now -
                    last_log_time
                    >=
                    1.0
                ):

                    self.get_logger().info(
                        "[ARM] tracking: "
                        f"t={elapsed:.3f}/"
                        f"{times[-1]:.3f}s, "
                        f"max_q_error="
                        f"{max_position_error:.6f} rad, "
                        f"max_qvel_error="
                        f"{max_velocity_error:.6f} rad/s"
                    )

                    last_log_time = now

                # ====================================================
                # Trajectory finished
                # ====================================================
                if elapsed >= times[-1]:

                    break

                # ----------------------------------------------------
                # Controller loop rate
                #
                # trajectory timing은 simulation time으로 결정되므로
                # sleep은 CPU 점유율을 낮추고 physics thread를
                # 원활하게 실행시키는 역할만 수행한다.
                # ----------------------------------------------------
                remaining = (
                    self.CONTROLLER_PERIOD
                )

                time.sleep(remaining)

            # ========================================================
            # 6. Final settle
            # ========================================================
            #
            # 마지막 desired q를 계속 command하면서
            # 실제 qpos / qvel이 최종 goal tolerance에 들어오는지
            # 확인한다.
            # ========================================================
            final_q = positions[-1]

            if np.all(
                np.isfinite(
                    velocities[-1]
                )
            ):
                final_v = velocities[-1]
            else:
                final_v = np.zeros(
                    len(self.arm_joints)
                )

            settle_start = (
                self._get_sim_time()
            )

            while (
                self._get_sim_time()
                -
                settle_start
                <
                self.SETTLE_TIMEOUT
            ):

                if goal_handle.is_cancel_requested:

                    self.get_logger().warn(
                        "[ARM] Cancel requested "
                        "during final settle."
                    )

                    self._hold_current_arm_position()

                    goal_handle.canceled()

                    return (
                        FollowJointTrajectory.Result()
                    )

                # ----------------------------------------------------
                # 마지막 target 유지
                # ----------------------------------------------------
                with self.lock:

                    for i, name in enumerate(
                        self.arm_joints
                    ):

                        self.data.ctrl[
                            self.actuator_ids[name]
                        ] = final_q[i]

                # ----------------------------------------------------
                # Actual state
                # ----------------------------------------------------
                actual_q, actual_v = (
                    self._get_arm_state()
                )

                goal_position_error = (
                    final_q - actual_q
                )

                goal_velocity_error = (
                    final_v - actual_v
                )

                # ----------------------------------------------------
                # Goal tolerance
                # ----------------------------------------------------
                position_ok = np.all(
                    np.abs(
                        goal_position_error
                    )
                    <=
                    goal_pos_tol
                )

                velocity_ok = np.all(
                    np.abs(
                        actual_v
                    )
                    <=
                    goal_vel_tol
                )

                # ----------------------------------------------------
                # Feedback
                # ----------------------------------------------------
                self._publish_arm_feedback(
                    goal_handle,
                    final_q,
                    final_v,
                    actual_q,
                    actual_v
                )

                # ----------------------------------------------------
                # Success
                # ----------------------------------------------------
                if (
                    position_ok
                    and
                    velocity_ok
                ):

                    max_goal_error = float(
                        np.max(
                            np.abs(
                                goal_position_error
                            )
                        )
                    )

                    max_goal_velocity = float(
                        np.max(
                            np.abs(actual_v)
                        )
                    )

                    self.get_logger().info(
                        "[ARM] Trajectory SUCCESS: "
                        f"max_goal_error="
                        f"{max_goal_error:.6f} rad, "
                        f"max_velocity="
                        f"{max_goal_velocity:.6f} rad/s"
                    )

                    result = (
                        FollowJointTrajectory.Result()
                    )

                    result.error_code = (
                        FollowJointTrajectory.Result.SUCCESSFUL
                    )

                    goal_handle.succeed()

                    return result

                time.sleep(0.01)

            # ========================================================
            # 7. Goal tolerance failed
            # ========================================================
            actual_q, actual_v = (
                self._get_arm_state()
            )

            final_error = (
                final_q - actual_q
            )

            max_final_error = float(
                np.max(
                    np.abs(final_error)
                )
            )

            max_final_velocity = float(
                np.max(
                    np.abs(actual_v)
                )
            )

            self.get_logger().error(
                "[ARM] GOAL TOLERANCE VIOLATED: "
                f"max_position_error="
                f"{max_final_error:.6f} rad, "
                f"max_velocity="
                f"{max_final_velocity:.6f} rad/s"
            )

            self._hold_current_arm_position()

            goal_handle.abort()

            result = (
                FollowJointTrajectory.Result()
            )

            result.error_code = (
                FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            )

            return result

        except Exception as e:

            # ========================================================
            # Unexpected controller exception
            # ========================================================
            self.get_logger().error(
                f"[ARM] Controller exception: {e}"
            )

            self._hold_current_arm_position()

            goal_handle.abort()

            result = (
                FollowJointTrajectory.Result()
            )

            result.error_code = (
                FollowJointTrajectory.Result.INVALID_GOAL
            )

            return result

        finally:

            # ========================================================
            # Active goal 해제
            # ========================================================
            with self.active_arm_goal_lock:

                if (
                    self.active_arm_goal
                    is goal_handle
                ):
                    self.active_arm_goal = None

    # ================================================================
    # Gripper
    # ================================================================
    def _gripper_position(self):

        if not self.finger_joints:
            return 0.0

        with self.lock:

            values = [
                self.data.qpos[
                    self.model.jnt_qposadr[
                        self.joint_ids[n]
                    ]
                ]
                for n in self.finger_joints
            ]

        return float(
            np.mean(values)
        )

    def _gripper_velocity_ok(self):

        with self.lock:

            v = [
                abs(
                    self.data.qvel[
                        self.model.jnt_dofadr[
                            self.joint_ids[n]
                        ]
                    ]
                )
                for n in self.finger_joints
            ]

        return max(
            v,
            default=0.0
        ) <= self.VELOCITY_TOLERANCE

    # ================================================================
    # Gripper action
    # ================================================================
    def execute_gripper(self, goal_handle):

        target = float(
            np.clip(
                goal_handle.request.command.position,
                0.0,
                0.04
            )
        )

        # ============================================================
        # OPEN
        # ============================================================
        if target > 0.02:

            self.gripper_force_enabled = False

            with self.lock:

                self.data.qfrc_applied[
                    self.finger1_dof_id
                ] = 0.0

                self.data.qfrc_applied[
                    self.finger2_dof_id
                ] = 0.0

            ctrl = float(
                np.clip(
                    target / 0.04 * 255.0,
                    self.GRIPPER_CTRL_MIN,
                    self.GRIPPER_CTRL_MAX
                )
            )

            with self.lock:

                self.data.ctrl[
                    self.GRIPPER_ACTUATOR_ID
                ] = ctrl

            self.get_logger().info(
                f"[GRIPPER] OPEN "
                f"target={target:.4f} "
                f"ctrl={ctrl:.2f} "
                f"extra_force=OFF"
            )

            start = self._get_sim_time()
            stable_start = None

            while (
                self._get_sim_time() - start
                <
                self.GRIPPER_TIMEOUT
            ):

                if goal_handle.is_cancel_requested:

                    goal_handle.canceled()

                    return (
                        GripperCommand.Result()
                    )

                pos = self._gripper_position()

                reached = (
                    pos >= 0.035
                    and
                    self._gripper_velocity_ok()
                )

                if reached:

                    if stable_start is None:
                        stable_start = (
                            self._get_sim_time()
                        )

                    elif (
                        self._get_sim_time()
                        -
                        stable_start
                        >=
                        self.GRIPPER_STABLE_TIME
                    ):

                        result = (
                            GripperCommand.Result()
                        )

                        result.position = pos
                        result.reached_goal = True
                        result.stalled = False

                        goal_handle.succeed()

                        self.get_logger().info(
                            f"[GRIPPER] OPEN complete "
                            f"position={pos:.4f}"
                        )

                        return result

                else:
                    stable_start = None

                time.sleep(0.01)

            result = (
                GripperCommand.Result()
            )

            result.position = (
                self._gripper_position()
            )

            result.reached_goal = False
            result.stalled = True

            goal_handle.abort()

            return result

        # ============================================================
        # CLOSE
        # ============================================================
        self.gripper_force_enabled = True

        with self.lock:

            self.data.ctrl[
                self.GRIPPER_ACTUATOR_ID
            ] = self.GRIPPER_CTRL_MIN

        self.get_logger().info(
            f"[GRIPPER] CLOSE start "
            f"ctrl={self.GRIPPER_CTRL_MIN:.3f} "
            f"extra_force="
            f"{self.GRIPPER_EXTRA_FORCE:.1f}N/finger"
        )

        start = self._get_sim_time()
        stable_start = None

        while (
            self._get_sim_time() - start
            <
            self.GRIPPER_TIMEOUT
        ):

            if goal_handle.is_cancel_requested:

                goal_handle.canceled()

                return (
                    GripperCommand.Result()
                )

            pos = self._gripper_position()
            v_ok = self._gripper_velocity_ok()

            if v_ok:

                if stable_start is None:
                    stable_start = (
                        self._get_sim_time()
                    )

                elif (
                    self._get_sim_time()
                    -
                    stable_start
                    >=
                    self.GRIPPER_CONTACT_STABLE_TIME
                ):

                    result = (
                        GripperCommand.Result()
                    )

                    result.position = pos
                    result.reached_goal = True
                    result.stalled = (
                        pos > 0.001
                    )

                    goal_handle.succeed()

                    self.get_logger().info(
                        f"[GRIPPER] CLOSE complete "
                        f"position={pos:.4f}"
                    )

                    return result

            else:
                stable_start = None

            time.sleep(0.01)

        result = (
            GripperCommand.Result()
        )

        result.position = (
            self._gripper_position()
        )

        result.reached_goal = True
        result.stalled = (
            result.position > 0.001
        )

        goal_handle.succeed()

        self.get_logger().info(
            f"[GRIPPER] CLOSE complete "
            f"position={result.position:.4f}"
        )

        return result

    # ================================================================
    # Gripper topic callbacks
    # ================================================================
    def gripper_string_callback(self, msg):

        cmd = msg.data.lower().strip()

        # ------------------------------------------------------------
        # OPEN
        # ------------------------------------------------------------
        if cmd in (
            "open",
            "release"
        ):

            self.gripper_force_enabled = False

            with self.lock:

                self.data.qfrc_applied[
                    self.finger1_dof_id
                ] = 0.0

                self.data.qfrc_applied[
                    self.finger2_dof_id
                ] = 0.0

                self.data.ctrl[
                    self.GRIPPER_ACTUATOR_ID
                ] = self.GRIPPER_CTRL_MAX

            self.get_logger().info(
                "[GRIPPER] Topic OPEN "
                "extra_force=OFF"
            )

        # ------------------------------------------------------------
        # CLOSE
        # ------------------------------------------------------------
        elif cmd in (
            "close",
            "grasp"
        ):

            self.gripper_force_enabled = True

            with self.lock:

                self.data.ctrl[
                    self.GRIPPER_ACTUATOR_ID
                ] = self.GRIPPER_CTRL_MIN

            self.get_logger().info(
                f"[GRIPPER] Topic CLOSE "
                f"extra_force="
                f"{self.GRIPPER_EXTRA_FORCE:.1f}N/finger"
            )

        else:
            return

    def gripper_float_callback(self, msg):

        value = float(
            np.clip(
                msg.data,
                0.0,
                255.0
            )
        )

        # ------------------------------------------------------------
        # Float command는 기존 actuator command 방식 유지.
        #
        # OPEN 값이면 extra force OFF
        # CLOSE 값이면 extra force ON
        # ------------------------------------------------------------
        if value > 127.5:

            self.gripper_force_enabled = False

            with self.lock:

                self.data.qfrc_applied[
                    self.finger1_dof_id
                ] = 0.0

                self.data.qfrc_applied[
                    self.finger2_dof_id
                ] = 0.0

        else:

            self.gripper_force_enabled = True

        with self.lock:

            self.data.ctrl[
                self.GRIPPER_ACTUATOR_ID
            ] = value

    # ================================================================
    # Destroy
    # ================================================================
    def destroy_node(self):

        self.running = False

        # ------------------------------------------------------------
        # 종료 시 압착력 제거
        # ------------------------------------------------------------
        self.gripper_force_enabled = False

        with self.lock:

            self.data.qfrc_applied[
                self.finger1_dof_id
            ] = 0.0

            self.data.qfrc_applied[
                self.finger2_dof_id
            ] = 0.0

        if (
            hasattr(
                self,
                "camera_thread"
            )
            and
            self.camera_thread.is_alive()
        ):

            self.camera_thread.join(
                timeout=2.0
            )

        super().destroy_node()


# ====================================================================
# MuJoCo model loading
# ====================================================================
def load_model(xml_path):

    xml_path = Path(xml_path)

    try:

        return mujoco.MjModel.from_xml_path(
            str(xml_path)
        )

    except Exception as e:

        print(
            f"[WARN] from_xml_path failed: {e}"
        )

        print(
            "[INFO] Trying VFS fallback..."
        )

    root = Path("/workspace")
    vfs = {}

    for path in root.rglob("*"):

        if (
            not path.is_file()
            or
            path.suffix.lower()
            not in {
                ".xml",
                ".stl",
                ".obj",
                ".png",
                ".jpg",
                ".jpeg"
            }
        ):
            continue

        try:

            vfs[
                path.relative_to(
                    root
                ).as_posix()
            ] = path.read_bytes()

        except Exception:
            pass

    print(
        f"[INFO] VFS files: {len(vfs)}"
    )

    return mujoco.MjModel.from_xml_string(
        xml_path.read_text(),
        assets=vfs
    )


# ====================================================================
# Main
# ====================================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="/workspace/scene/panda_test.xml"
    )

    parser.add_argument(
        "--headless",
        action="store_true"
    )

    args = parser.parse_args()

    rclpy.init()

    model = load_model(
        args.model
    )

    data = mujoco.MjData(
        model
    )

    node = MjcfBridgeNode(
        model,
        data
    )

    executor = (
        rclpy.executors.MultiThreadedExecutor(
            num_threads=4
        )
    )

    executor.add_node(node)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True
    )

    spin_thread.start()

    try:

        # ============================================================
        # HEADLESS
        # ============================================================
        if args.headless:

            print(
                "[INFO] Running in headless mode (no viewer)...",
                flush=True
            )

            wall_start = time.perf_counter()
            sim_start = float(data.time)

            while rclpy.ok():

                with node.lock:

                    # ------------------------------------------------
                    # 매 physics step마다 qfrc_applied 재설정
                    # ------------------------------------------------
                    node._prepare_physics_step()

                    mujoco.mj_step(
                        model,
                        data
                    )

                    node.publish_joint_states()
                    node.publish_tf()

                # ----------------------------------------------------
                # Real-Time Factor (1.0x) 동기화
                # ----------------------------------------------------
                sim_elapsed = (
                    float(data.time) - sim_start
                )
                wall_elapsed = (
                    time.perf_counter() - wall_start
                )
                time_until_next_step = (
                    sim_elapsed - wall_elapsed
                )

                if time_until_next_step > 0.0:
                    time.sleep(time_until_next_step)

        # ============================================================
        # VIEWER
        # ============================================================
        else:

            print(
                "[INFO] Launching MuJoCo passive viewer...",
                flush=True
            )

            with mujoco.viewer.launch_passive(
                model,
                data
            ) as viewer:

                viewer.cam.distance = 1.5
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -20

                viewer.cam.lookat[:] = [
                    0,
                    0.5,
                    0.5
                ]

                print(
                    "[INFO] MuJoCo passive viewer "
                    "launched successfully.",
                    flush=True
                )

                wall_start = time.perf_counter()
                sim_start = float(data.time)

                while (
                    viewer.is_running()
                    and
                    rclpy.ok()
                ):

                    with node.lock:

                        # ------------------------------------------------
                        # CLOSE/LIFT 상태이면 매 step마다
                        # finger에 추가 압착력 적용
                        # ------------------------------------------------
                        node._prepare_physics_step()

                        mujoco.mj_step(
                            model,
                            data
                        )

                        node.publish_joint_states()
                        node.publish_tf()

                    viewer.sync()

                    # ------------------------------------------------
                    # Real-Time Factor (1.0x) 동기화
                    # ------------------------------------------------
                    sim_elapsed = (
                        float(data.time) - sim_start
                    )
                    wall_elapsed = (
                        time.perf_counter() - wall_start
                    )
                    time_until_next_step = (
                        sim_elapsed - wall_elapsed
                    )

                    if time_until_next_step > 0.0:
                        time.sleep(time_until_next_step)

                print(
                    "[INFO] Viewer closed or loop finished.",
                    flush=True
                )

    except KeyboardInterrupt:

        print(
            "[INFO] KeyboardInterrupt received.",
            flush=True
        )

    except Exception as e:

        import traceback

        print(
            f"[ERROR] Exception in simulation loop: {e}",
            flush=True
        )

        traceback.print_exc()

    finally:

        print(
            "[INFO] Shutting down bridge node...",
            flush=True
        )

        node.running = False

        executor.shutdown()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        spin_thread.join(
            timeout=2.0
        )


if __name__ == "__main__":
    main()