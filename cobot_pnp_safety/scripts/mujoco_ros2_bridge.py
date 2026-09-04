#!/usr/bin/env python3
"""MuJoCo ROS 2 Bridge with Panda FollowJointTrajectory/GripperCommand."""

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
        self.model, self.data = model, data
        self.lock = threading.Lock()
        self.running = True
        self.cb_group = ReentrantCallbackGroup()

        self.arm_joints = [f"joint{i}" for i in range(1, 8)]
        self.finger_joints = ["finger_joint1", "finger_joint2"]
        self.all_joints = self.arm_joints + self.finger_joints
        self.joint_ids = {
            n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.all_joints
        }
        self.actuator_ids = {
            f"joint{i}": mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}"
            )
            for i in range(1, 8)
        }

        self.GRIPPER_ACTUATOR_ID = 7
        self.finger_joint_ids = {n: self.joint_ids[n] for n in self.finger_joints}
        self.finger_dof_ids = {
            n: model.jnt_dofadr[self.finger_joint_ids[n]]
            for n in self.finger_joints
        }
        self.finger1_dof_id = self.finger_dof_ids["finger_joint1"]
        self.finger2_dof_id = self.finger_dof_ids["finger_joint2"]
        self.GRIPPER_EXTRA_FORCE = 30.0
        self.gripper_force_enabled = False

        self.camera_name = "ceiling_camera"
        self.camera_width = 640
        self.camera_height = 480
        self.camera_rate = 10.0
        self.camera_optical_frame = "ceiling_camera_optical_frame"
        self.camera_link_frame = "ceiling_camera_link"

        self.finger_geom_ids = self._collect_finger_geom_ids()
        self.panda_geom_ids = self._collect_panda_geom_ids()

        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.rgb_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.seg_pub = self.create_publisher(Image, "/camera/segmentation/image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/depth/camera_info", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.create_subscription(
            String, "/panda_gripper/cmd", self.gripper_string_callback, 10,
            callback_group=self.cb_group
        )
        self.create_subscription(
            Float64, "/panda_gripper/command", self.gripper_float_callback, 10,
            callback_group=self.cb_group
        )

        self.arm_action = ActionServer(
            self, FollowJointTrajectory,
            "/panda_arm_controller/follow_joint_trajectory",
            execute_callback=self.execute_arm,
            goal_callback=self.arm_goal,
            cancel_callback=self.cancel_goal,
            callback_group=self.cb_group
        )
        self.gripper_action = ActionServer(
            self, GripperCommand,
            "/panda_hand_controller/gripper_action",
            execute_callback=self.execute_gripper,
            goal_callback=self.gripper_goal,
            cancel_callback=self.cancel_goal,
            callback_group=self.cb_group
        )

        self.home_qpos = np.array(
            [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.57079, 0.785398],
            dtype=float
        )
        self.POSITION_TOLERANCE = 0.01
        self.VELOCITY_TOLERANCE = 0.1
        self.PATH_POSITION_TOLERANCE = 0.20
        self.PATH_VELOCITY_TOLERANCE = 2.0
        self.CONTROLLER_RATE = 200.0
        self.CONTROLLER_PERIOD = 1.0 / self.CONTROLLER_RATE
        self.START_POSITION_TOLERANCE = 0.10
        self.SETTLE_TIMEOUT = 5.0

        self.GRIPPER_STABLE_TIME = 0.15
        self.GRIPPER_TIMEOUT = 2.0
        self.GRIPPER_CONTACT_STABLE_TIME = 0.15
        self.GRIPPER_CTRL_MIN = 0.0
        self.GRIPPER_CTRL_MAX = 255.0

        self.active_arm_goal = None
        self.active_arm_goal_lock = threading.Lock()

        self._set_initial_pose()
        self.publish_static_camera_tf()

        self.camera_thread = threading.Thread(
            target=self._camera_render_loop, daemon=True
        )
        self.camera_thread.start()

        self.get_logger().info("MuJoCo ROS 2 Bridge started.")
        self.get_logger().info(
            f"[GRIPPER] Extra gripping force={self.GRIPPER_EXTRA_FORCE:.1f} N per finger"
        )
        self.get_logger().info(
            f"[GRIPPER] finger_joint1 DOF={self.finger1_dof_id}, "
            f"finger_joint2 DOF={self.finger2_dof_id}"
        )
        self.get_logger().info(
            f"[ARM CONTROLLER] rate={self.CONTROLLER_RATE:.1f} Hz, "
            f"path_tol={self.PATH_POSITION_TOLERANCE:.3f} rad, "
            f"goal_tol={self.POSITION_TOLERANCE:.3f} rad"
        )

    def _clear_gripper_extra_force(self):
        with self.lock:
            self.data.qfrc_applied[self.finger1_dof_id] = 0.0
            self.data.qfrc_applied[self.finger2_dof_id] = 0.0

    def _apply_gripper_extra_force(self):
        if not self.gripper_force_enabled:
            return
        force = -abs(float(self.GRIPPER_EXTRA_FORCE))
        self.data.qfrc_applied[self.finger1_dof_id] += force
        self.data.qfrc_applied[self.finger2_dof_id] += force

    def _prepare_physics_step(self):
        self.data.qfrc_applied[:] = 0.0
        if self.gripper_force_enabled:
            self._apply_gripper_extra_force()

    def _gripper_extra_force_values(self):
        with self.lock:
            return (
                float(self.data.qfrc_applied[self.finger1_dof_id]),
                float(self.data.qfrc_applied[self.finger2_dof_id])
            )

    def publish_static_camera_tf(self):
        stamp = self.get_clock().now().to_msg()
        transforms = []
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        cam_body = self.model.cam_bodyid[cam_id]
        cam_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, cam_body
        )

        if cam_name:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "world"
            t.child_frame_id = self.camera_link_frame
            p = self.data.xpos[cam_body]
            q = self._mat_to_quat(self.data.xmat[cam_body].reshape(3, 3))
            t.transform.translation.x = float(p[0])
            t.transform.translation.y = float(p[1])
            t.transform.translation.z = float(p[2])
            t.transform.rotation.x, t.transform.rotation.y = q[0], q[1]
            t.transform.rotation.z, t.transform.rotation.w = q[2], q[3]
            transforms.append(t)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.camera_link_frame
        t.child_frame_id = self.camera_optical_frame
        t.transform.rotation.x = 1.0
        t.transform.rotation.w = 0.0
        transforms.append(t)
        self.static_tf_broadcaster.sendTransform(transforms)

    def _collect_panda_geom_ids(self):
        root = self.model.body("link0").id
        ids = set()
        for gid in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[gid])
            while body > 0:
                if body == root:
                    ids.add(gid)
                    break
                body = int(self.model.body_parentid[body])
        return ids

    def _collect_finger_geom_ids(self):
        ids = set()
        for gid in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[gid])
            while body > 0:
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body
                )
                if name and ("finger" in name or "hand" in name):
                    ids.add(gid)
                    break
                body = int(self.model.body_parentid[body])
        return ids

    def _set_initial_pose(self):
        for i, name in enumerate(self.arm_joints):
            jid = self.joint_ids[name]
            self.data.qpos[self.model.jnt_qposadr[jid]] = self.home_qpos[i]

        self.gripper_force_enabled = False
        self.data.qfrc_applied[:] = 0.0
        self.data.ctrl[self.GRIPPER_ACTUATOR_ID] = self.GRIPPER_CTRL_MAX

        for i, name in enumerate(self.arm_joints):
            self.data.ctrl[self.actuator_ids[name]] = self.home_qpos[i]

        mujoco.mj_forward(self.model, self.data)

    def _camera_render_loop(self):
        renderer = mujoco.Renderer(
            self.model, height=self.camera_height, width=self.camera_width
        )
        depth_renderer = mujoco.Renderer(
            self.model, height=self.camera_height, width=self.camera_width
        )
        seg_renderer = mujoco.Renderer(
            self.model, height=self.camera_height, width=self.camera_width
        )
        depth_renderer.enable_depth_rendering()
        seg_renderer.enable_segmentation_rendering()

        render_data = mujoco.MjData(self.model)
        period = 1.0 / self.camera_rate
        next_time = time.monotonic()

        while self.running:
            next_time += period

            with self.lock:
                render_data.qpos[:] = self.data.qpos
                render_data.qvel[:] = self.data.qvel
                render_data.act[:] = self.data.act
                if self.model.nmocap:
                    render_data.mocap_pos[:] = self.data.mocap_pos
                    render_data.mocap_quat[:] = self.data.mocap_quat
                mujoco.mj_forward(self.model, render_data)

            renderer.update_scene(render_data, camera=self.camera_name)
            rgb = np.asarray(renderer.render()).copy()

            depth_renderer.update_scene(render_data, camera=self.camera_name)
            depth = np.asarray(depth_renderer.render()).copy()

            seg_renderer.update_scene(render_data, camera=self.camera_name)
            seg_raw = np.asarray(seg_renderer.render()).copy()

            seg_id = seg_raw[:, :, 0].astype(np.int32)
            seg_type = seg_raw[:, :, 1].astype(np.int32)
            geom_mask = seg_type == int(mujoco.mjtObj.mjOBJ_GEOM)
            panda_mask = geom_mask & np.isin(
                seg_id, np.asarray(list(self.panda_geom_ids), dtype=np.int32)
            )
            seg = seg_id.copy()
            seg[~geom_mask] = 0
            seg[panda_mask] = 0

            stamp = self.get_clock().now().to_msg()
            self.rgb_pub.publish(self._image_msg(rgb, "rgb8", 3, stamp))
            self.depth_pub.publish(self._image_msg(depth, "32FC1", 4, stamp))
            self.seg_pub.publish(self._image_msg(seg, "32SC1", 4, stamp))
            self.info_pub.publish(self._camera_info(stamp))

            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.monotonic()

        renderer.close()
        depth_renderer.close()
        seg_renderer.close()

    def _image_msg(self, data, encoding, bpp, stamp):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_optical_frame
        msg.height, msg.width = self.camera_height, self.camera_width
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = self.camera_width * bpp
        dtype = np.uint8 if encoding == "rgb8" else (
            np.float32 if encoding == "32FC1" else np.int32
        )
        msg.data = data.astype(dtype).tobytes()
        return msg

    def _camera_info(self, stamp):
        cid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        fovy = self.model.cam_fovy[cid]
        fy = (self.camera_height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        fx = fy
        cx = (self.camera_width - 1) / 2.0
        cy = (self.camera_height - 1) / 2.0

        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_optical_frame
        msg.width, msg.height = self.camera_width, self.camera_height
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        return msg

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        for name in self.all_joints:
            jid = self.joint_ids[name]
            msg.name.append(name)
            msg.position.append(float(
                self.data.qpos[self.model.jnt_qposadr[jid]]
            ))
            msg.velocity.append(float(
                self.data.qvel[self.model.jnt_dofadr[jid]]
            ))

        self.joint_pub.publish(msg)

    def _mat_to_quat(self, R):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        return float(q[1]), float(q[2]), float(q[3]), float(q[0])

    def _relative_transform(self, body_id, parent_id):
        p = self.data.xpos[body_id]
        pp = self.data.xpos[parent_id]
        R = self.data.xmat[body_id].reshape(3, 3)
        Rp = self.data.xmat[parent_id].reshape(3, 3)
        return Rp.T @ (p - pp), Rp.T @ R

    def publish_tf(self):
        stamp = self.get_clock().now().to_msg()
        root = self.model.body("link0").id

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "world"
        t.child_frame_id = "link0"
        p = self.data.xpos[root]
        q = self._mat_to_quat(self.data.xmat[root].reshape(3, 3))
        t.transform.translation.x = float(p[0])
        t.transform.translation.y = float(p[1])
        t.transform.translation.z = float(p[2])
        t.transform.rotation.x, t.transform.rotation.y = q[0], q[1]
        t.transform.rotation.z, t.transform.rotation.w = q[2], q[3]
        self.tf_broadcaster.sendTransform(t)

        for bid in range(1, self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bid
            )
            if not name or name in ("link0", self.camera_link_frame):
                continue

            parent = self.model.body_parentid[bid]
            parent_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, parent
            )
            if not parent_name or parent == 0:
                continue

            pos, rot = self._relative_transform(bid, parent)
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = parent_name
            t.child_frame_id = name
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            q = self._mat_to_quat(rot)
            t.transform.rotation.x, t.transform.rotation.y = q[0], q[1]
            t.transform.rotation.z, t.transform.rotation.w = q[2], q[3]
            self.tf_broadcaster.sendTransform(t)

    def arm_goal(self, goal):
        if not goal.trajectory.joint_names:
            return GoalResponse.REJECT

        with self.active_arm_goal_lock:
            if self.active_arm_goal is not None:
                self.get_logger().warn(
                    "[ARM] Existing arm goal is active. Rejecting new goal."
                )
                return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def gripper_goal(self, goal):
        return GoalResponse.ACCEPT

    def cancel_goal(self, goal):
        return CancelResponse.ACCEPT

    def _get_sim_time(self):
        with self.lock:
            return float(self.data.time)

    def _get_arm_state(self):
        with self.lock:
            q = np.array([
                self.data.qpos[
                    self.model.jnt_qposadr[self.joint_ids[name]]
                ]
                for name in self.arm_joints
            ], dtype=float)
            v = np.array([
                self.data.qvel[
                    self.model.jnt_dofadr[self.joint_ids[name]]
                ]
                for name in self.arm_joints
            ], dtype=float)
        return q, v

    def _hold_current_arm_position(self):
        q, _ = self._get_arm_state()
        with self.lock:
            for i, name in enumerate(self.arm_joints):
                self.data.ctrl[self.actuator_ids[name]] = q[i]
        return q

    def _parse_trajectory(self, traj):
        name_to_idx = {name: i for i, name in enumerate(traj.joint_names)}

        for name in self.arm_joints:
            if name not in name_to_idx:
                raise ValueError(f"Missing joint in trajectory: {name}")

        times, positions, velocities = [], [], []
        prev_t = -1.0

        for point_index, point in enumerate(traj.points):
            t = float(point.time_from_start.sec) + float(
                point.time_from_start.nanosec
            ) * 1e-9

            if t < 0.0:
                raise ValueError(f"Negative trajectory time at point {point_index}")
            if t <= prev_t:
                raise ValueError(
                    f"Non-increasing trajectory time at point {point_index}: "
                    f"{t} <= {prev_t}"
                )
            if len(point.positions) < len(traj.joint_names):
                raise ValueError(f"Invalid position array at point {point_index}")

            q = np.array([
                point.positions[name_to_idx[name]]
                for name in self.arm_joints
            ], dtype=float)

            if len(point.velocities) >= len(traj.joint_names):
                v = np.array([
                    point.velocities[name_to_idx[name]]
                    for name in self.arm_joints
                ], dtype=float)
                if not np.all(np.isfinite(v)):
                    v[:] = np.nan
            else:
                v = np.full(len(self.arm_joints), np.nan, dtype=float)

            times.append(t)
            positions.append(q)
            velocities.append(v)
            prev_t = t

        return (
            np.asarray(times, dtype=float),
            np.asarray(positions, dtype=float),
            np.asarray(velocities, dtype=float)
        )

    def _interpolate_segment(self, q0, q1, v0, v1, duration, tau):
        if duration <= 1e-9:
            return q1.copy(), np.zeros_like(q1)

        s = np.clip(tau / duration, 0.0, 1.0)

        if np.all(np.isfinite(v0)) and np.all(np.isfinite(v1)):
            h00 = 2*s**3 - 3*s**2 + 1
            h10 = s**3 - 2*s**2 + s
            h01 = -2*s**3 + 3*s**2
            h11 = s**3 - s**2

            q = (
                h00*q0 +
                h10*duration*v0 +
                h01*q1 +
                h11*duration*v1
            )

            dh00 = 6*s**2 - 6*s
            dh10 = 3*s**2 - 4*s + 1
            dh01 = -6*s**2 + 6*s
            dh11 = 3*s**2 - 2*s

            qdot = (
                dh00*q0/duration +
                dh10*v0 +
                dh01*q1/duration +
                dh11*v1
            )
            return q, qdot

        return q0 + (q1 - q0)*s, (q1 - q0)/duration

    def _trajectory_state_at(self, elapsed, times, positions, velocities):
        if elapsed <= times[0]:
            return (
                positions[0].copy(),
                velocities[0].copy()
                if np.all(np.isfinite(velocities[0]))
                else np.zeros(len(self.arm_joints))
            )

        if elapsed >= times[-1]:
            return (
                positions[-1].copy(),
                velocities[-1].copy()
                if np.all(np.isfinite(velocities[-1]))
                else np.zeros(len(self.arm_joints))
            )

        index = np.searchsorted(times, elapsed, side="right") - 1
        index = max(0, min(index, len(times) - 2))
        duration = times[index + 1] - times[index]
        tau = elapsed - times[index]

        return self._interpolate_segment(
            positions[index],
            positions[index + 1],
            velocities[index],
            velocities[index + 1],
            duration,
            tau
        )

    def _extract_joint_tolerance(self, tolerance_list, default):
        result = {name: float(default) for name in self.arm_joints}
        for tol in tolerance_list:
            if tol.name not in result:
                continue
            if hasattr(tol, "position") and tol.position > 0.0:
                result[tol.name] = float(tol.position)
        return np.array([result[name] for name in self.arm_joints], dtype=float)

    def _extract_velocity_tolerance(self, tolerance_list, default):
        result = {name: float(default) for name in self.arm_joints}
        for tol in tolerance_list:
            if tol.name not in result:
                continue
            if hasattr(tol, "velocity") and tol.velocity > 0.0:
                result[tol.name] = float(tol.velocity)
        return np.array([result[name] for name in self.arm_joints], dtype=float)

    def _publish_arm_feedback(
        self, goal_handle, desired_q, desired_v, actual_q, actual_v
    ):
        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = list(self.arm_joints)
        feedback.desired.positions = desired_q.tolist()
        feedback.desired.velocities = desired_v.tolist()
        feedback.actual.positions = actual_q.tolist()
        feedback.actual.velocities = actual_v.tolist()
        feedback.error.positions = (desired_q - actual_q).tolist()
        feedback.error.velocities = (desired_v - actual_v).tolist()
        goal_handle.publish_feedback(feedback)

    def execute_arm(self, goal_handle):
        goal = goal_handle.request
        traj = goal.trajectory

        with self.active_arm_goal_lock:
            self.active_arm_goal = goal_handle

        try:
            if not traj.points:
                self.get_logger().error("[ARM] Empty trajectory.")
                goal_handle.abort()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                return result

            try:
                times, positions, velocities = self._parse_trajectory(traj)
            except Exception as e:
                self.get_logger().error(
                    f"[ARM] Trajectory validation failed: {e}"
                )
                goal_handle.abort()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
                return result

            current_q, current_v = self._get_arm_state()
            first_q = positions[0]
            start_error = first_q - current_q
            max_start_error = float(np.max(np.abs(start_error)))

            self.get_logger().info(
                f"[ARM] trajectory points={len(times)}, duration={times[-1]:.6f}s"
            )
            self.get_logger().info(
                f"[ARM] current_q={np.round(current_q,6).tolist()}"
            )
            self.get_logger().info(
                f"[ARM] first_q={np.round(first_q,6).tolist()}"
            )
            self.get_logger().info(
                f"[ARM] max_start_state_error={max_start_error:.6f} rad"
            )

            if max_start_error > self.START_POSITION_TOLERANCE:
                self.get_logger().error(
                    f"[ARM] Start state mismatch: {max_start_error:.6f} rad > "
                    f"{self.START_POSITION_TOLERANCE:.6f} rad"
                )
                self._hold_current_arm_position()
                goal_handle.abort()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                return result

            path_pos_tol = self._extract_joint_tolerance(
                goal.path_tolerance, self.PATH_POSITION_TOLERANCE
            )
            path_vel_tol = self._extract_velocity_tolerance(
                goal.path_tolerance, self.PATH_VELOCITY_TOLERANCE
            )
            goal_pos_tol = self._extract_joint_tolerance(
                goal.goal_tolerance, self.POSITION_TOLERANCE
            )
            goal_vel_tol = self._extract_velocity_tolerance(
                goal.goal_tolerance, self.VELOCITY_TOLERANCE
            )

            self.get_logger().info(
                f"[ARM] path_pos_tol={np.round(path_pos_tol,4).tolist()}"
            )
            self.get_logger().info(
                f"[ARM] goal_pos_tol={np.round(goal_pos_tol,4).tolist()}"
            )

            start_time = self._get_sim_time()
            last_feedback_time = start_time
            last_log_time = start_time
            feedback_period = 0.05

            while True:
                if goal_handle.is_cancel_requested:
                    self.get_logger().warn(
                        "[ARM] Cancel requested. Holding current position."
                    )
                    self._hold_current_arm_position()
                    goal_handle.canceled()
                    return FollowJointTrajectory.Result()

                now = self._get_sim_time()
                elapsed = now - start_time

                desired_q, desired_v = self._trajectory_state_at(
                    elapsed, times, positions, velocities
                )
                actual_q, actual_v = self._get_arm_state()

                position_error = desired_q - actual_q
                velocity_error = desired_v - actual_v
                max_position_error = float(np.max(np.abs(position_error)))
                max_velocity_error = float(np.max(np.abs(velocity_error)))

                position_violation = np.abs(position_error) > path_pos_tol
                velocity_violation = np.abs(velocity_error) > path_vel_tol

                if np.any(position_violation):
                    bad_indices = np.where(position_violation)[0]
                    self.get_logger().error(
                        f"[ARM] PATH POSITION TOLERANCE VIOLATED: "
                        f"elapsed={elapsed:.3f}s, max_error={max_position_error:.6f} rad, "
                        f"joints={[self.arm_joints[i] for i in bad_indices]}"
                    )
                    self._hold_current_arm_position()
                    goal_handle.abort()
                    result = FollowJointTrajectory.Result()
                    result.error_code = (
                        FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    )
                    return result

                if np.any(np.abs(desired_v) > 1e-9) and np.any(velocity_violation):
                    bad_indices = np.where(velocity_violation)[0]
                    self.get_logger().error(
                        f"[ARM] PATH VELOCITY TOLERANCE VIOLATED: "
                        f"elapsed={elapsed:.3f}s, "
                        f"max_velocity_error={max_velocity_error:.6f} rad/s, "
                        f"joints={[self.arm_joints[i] for i in bad_indices]}"
                    )
                    self._hold_current_arm_position()
                    goal_handle.abort()
                    result = FollowJointTrajectory.Result()
                    result.error_code = (
                        FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    )
                    return result

                with self.lock:
                    for i, name in enumerate(self.arm_joints):
                        self.data.ctrl[self.actuator_ids[name]] = desired_q[i]

                if now - last_feedback_time >= feedback_period:
                    self._publish_arm_feedback(
                        goal_handle, desired_q, desired_v, actual_q, actual_v
                    )
                    last_feedback_time = now

                if now - last_log_time >= 1.0:
                    self.get_logger().info(
                        f"[ARM] tracking: t={elapsed:.3f}/{times[-1]:.3f}s, "
                        f"max_q_error={max_position_error:.6f} rad, "
                        f"max_qvel_error={max_velocity_error:.6f} rad/s"
                    )
                    last_log_time = now

                if elapsed >= times[-1]:
                    break

                time.sleep(self.CONTROLLER_PERIOD)

            final_q = positions[-1]
            final_v = (
                velocities[-1]
                if np.all(np.isfinite(velocities[-1]))
                else np.zeros(len(self.arm_joints))
            )
            settle_start = self._get_sim_time()

            while self._get_sim_time() - settle_start < self.SETTLE_TIMEOUT:
                if goal_handle.is_cancel_requested:
                    self.get_logger().warn(
                        "[ARM] Cancel requested during final settle."
                    )
                    self._hold_current_arm_position()
                    goal_handle.canceled()
                    return FollowJointTrajectory.Result()

                with self.lock:
                    for i, name in enumerate(self.arm_joints):
                        self.data.ctrl[self.actuator_ids[name]] = final_q[i]

                actual_q, actual_v = self._get_arm_state()
                goal_position_error = final_q - actual_q

                position_ok = np.all(
                    np.abs(goal_position_error) <= goal_pos_tol
                )
                velocity_ok = np.all(np.abs(actual_v) <= goal_vel_tol)

                self._publish_arm_feedback(
                    goal_handle, final_q, final_v, actual_q, actual_v
                )

                if position_ok and velocity_ok:
                    max_goal_error = float(
                        np.max(np.abs(goal_position_error))
                    )
                    max_goal_velocity = float(np.max(np.abs(actual_v)))
                    self.get_logger().info(
                        f"[ARM] Trajectory SUCCESS: "
                        f"max_goal_error={max_goal_error:.6f} rad, "
                        f"max_velocity={max_goal_velocity:.6f} rad/s"
                    )
                    result = FollowJointTrajectory.Result()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    goal_handle.succeed()
                    return result

                time.sleep(0.01)

            actual_q, actual_v = self._get_arm_state()
            final_error = final_q - actual_q
            max_final_error = float(np.max(np.abs(final_error)))
            max_final_velocity = float(np.max(np.abs(actual_v)))

            self.get_logger().error(
                f"[ARM] GOAL TOLERANCE VIOLATED: "
                f"max_position_error={max_final_error:.6f} rad, "
                f"max_velocity={max_final_velocity:.6f} rad/s"
            )

            self._hold_current_arm_position()
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = (
                FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            )
            return result

        except Exception as e:
            self.get_logger().error(f"[ARM] Controller exception: {e}")
            self._hold_current_arm_position()
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return result

        finally:
            with self.active_arm_goal_lock:
                if self.active_arm_goal is goal_handle:
                    self.active_arm_goal = None

    def _gripper_position(self):
        if not self.finger_joints:
            return 0.0
        with self.lock:
            values = [
                self.data.qpos[
                    self.model.jnt_qposadr[self.joint_ids[n]]
                ]
                for n in self.finger_joints
            ]
        return float(np.mean(values))

    def _gripper_velocity_ok(self):
        with self.lock:
            v = [
                abs(self.data.qvel[
                    self.model.jnt_dofadr[self.joint_ids[n]]
                ])
                for n in self.finger_joints
            ]
        return max(v, default=0.0) <= self.VELOCITY_TOLERANCE

    def execute_gripper(self, goal_handle):
        target = float(np.clip(
            goal_handle.request.command.position, 0.0, 0.04
        ))

        if target > 0.02:
            self.gripper_force_enabled = False

            with self.lock:
                self.data.qfrc_applied[self.finger1_dof_id] = 0.0
                self.data.qfrc_applied[self.finger2_dof_id] = 0.0

            ctrl = float(np.clip(
                target / 0.04 * 255.0,
                self.GRIPPER_CTRL_MIN,
                self.GRIPPER_CTRL_MAX
            ))

            with self.lock:
                self.data.ctrl[self.GRIPPER_ACTUATOR_ID] = ctrl

            self.get_logger().info(
                f"[GRIPPER] OPEN target={target:.4f} ctrl={ctrl:.2f} "
                f"extra_force=OFF"
            )

            start = self._get_sim_time()
            stable_start = None

            while self._get_sim_time() - start < self.GRIPPER_TIMEOUT:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return GripperCommand.Result()

                pos = self._gripper_position()
                reached = pos >= 0.035 and self._gripper_velocity_ok()

                if reached:
                    if stable_start is None:
                        stable_start = self._get_sim_time()
                    elif (
                        self._get_sim_time() - stable_start
                        >= self.GRIPPER_STABLE_TIME
                    ):
                        result = GripperCommand.Result()
                        result.position = pos
                        result.reached_goal = True
                        result.stalled = False
                        goal_handle.succeed()
                        self.get_logger().info(
                            f"[GRIPPER] OPEN complete position={pos:.4f}"
                        )
                        return result
                else:
                    stable_start = None

                time.sleep(0.01)

            result = GripperCommand.Result()
            result.position = self._gripper_position()
            result.reached_goal = False
            result.stalled = True
            goal_handle.abort()
            return result

        self.gripper_force_enabled = True

        with self.lock:
            self.data.ctrl[self.GRIPPER_ACTUATOR_ID] = self.GRIPPER_CTRL_MIN

        self.get_logger().info(
            f"[GRIPPER] CLOSE start ctrl={self.GRIPPER_CTRL_MIN:.3f} "
            f"extra_force={self.GRIPPER_EXTRA_FORCE:.1f}N/finger"
        )

        start = self._get_sim_time()
        stable_start = None

        while self._get_sim_time() - start < self.GRIPPER_TIMEOUT:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return GripperCommand.Result()

            pos = self._gripper_position()
            v_ok = self._gripper_velocity_ok()

            if v_ok:
                if stable_start is None:
                    stable_start = self._get_sim_time()
                elif (
                    self._get_sim_time() - stable_start
                    >= self.GRIPPER_CONTACT_STABLE_TIME
                ):
                    result = GripperCommand.Result()
                    result.position = pos
                    result.reached_goal = True
                    result.stalled = pos > 0.001
                    goal_handle.succeed()
                    self.get_logger().info(
                        f"[GRIPPER] CLOSE complete position={pos:.4f}"
                    )
                    return result
            else:
                stable_start = None

            time.sleep(0.01)

        result = GripperCommand.Result()
        result.position = self._gripper_position()
        result.reached_goal = True
        result.stalled = result.position > 0.001
        goal_handle.succeed()
        self.get_logger().info(
            f"[GRIPPER] CLOSE complete position={result.position:.4f}"
        )
        return result

    def gripper_string_callback(self, msg):
        cmd = msg.data.lower().strip()

        if cmd in ("open", "release"):
            self.gripper_force_enabled = False
            with self.lock:
                self.data.qfrc_applied[self.finger1_dof_id] = 0.0
                self.data.qfrc_applied[self.finger2_dof_id] = 0.0
                self.data.ctrl[self.GRIPPER_ACTUATOR_ID] = self.GRIPPER_CTRL_MAX
            self.get_logger().info("[GRIPPER] Topic OPEN extra_force=OFF")

        elif cmd in ("close", "grasp"):
            self.gripper_force_enabled = True
            with self.lock:
                self.data.ctrl[self.GRIPPER_ACTUATOR_ID] = self.GRIPPER_CTRL_MIN
            self.get_logger().info(
                f"[GRIPPER] Topic CLOSE extra_force="
                f"{self.GRIPPER_EXTRA_FORCE:.1f}N/finger"
            )

    def gripper_float_callback(self, msg):
        value = float(np.clip(msg.data, 0.0, 255.0))

        if value > 127.5:
            self.gripper_force_enabled = False
            with self.lock:
                self.data.qfrc_applied[self.finger1_dof_id] = 0.0
                self.data.qfrc_applied[self.finger2_dof_id] = 0.0
        else:
            self.gripper_force_enabled = True

        with self.lock:
            self.data.ctrl[self.GRIPPER_ACTUATOR_ID] = value

    def destroy_node(self):
        self.running = False
        self.gripper_force_enabled = False

        with self.lock:
            self.data.qfrc_applied[self.finger1_dof_id] = 0.0
            self.data.qfrc_applied[self.finger2_dof_id] = 0.0

        if hasattr(self, "camera_thread") and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)

        super().destroy_node()


def load_model(xml_path):
    xml_path = Path(xml_path)

    try:
        return mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as e:
        print(f"[WARN] from_xml_path failed: {e}")
        print("[INFO] Trying VFS fallback...")

    root = Path("/workspace")
    vfs = {}

    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() not in {
                ".xml", ".stl", ".obj", ".png", ".jpg", ".jpeg"
            }
        ):
            continue
        try:
            vfs[path.relative_to(root).as_posix()] = path.read_bytes()
        except Exception:
            pass

    print(f"[INFO] VFS files: {len(vfs)}")

    return mujoco.MjModel.from_xml_string(
        xml_path.read_text(), assets=vfs
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/workspace/scene/panda_test.xml"
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    model = load_model(args.model)
    data = mujoco.MjData(model)
    node = MjcfBridgeNode(model, data)

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(
        target=executor.spin, daemon=True
    )
    spin_thread.start()

    try:
        if args.headless:
            print(
                "[INFO] Running in headless mode (no viewer)...",
                flush=True
            )

            wall_start = time.perf_counter()
            sim_start = float(data.time)

            while rclpy.ok():
                with node.lock:
                    node._prepare_physics_step()
                    mujoco.mj_step(model, data)
                    node.publish_joint_states()
                    node.publish_tf()

                sim_elapsed = float(data.time) - sim_start
                wall_elapsed = time.perf_counter() - wall_start
                delay = sim_elapsed - wall_elapsed

                if delay > 0.0:
                    time.sleep(delay)

        else:
            print(
                "[INFO] Launching MuJoCo passive viewer...",
                flush=True
            )

            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.distance = 1.5
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -20
                viewer.cam.lookat[:] = [0, 0.5, 0.5]

                print(
                    "[INFO] MuJoCo passive viewer launched successfully.",
                    flush=True
                )

                wall_start = time.perf_counter()
                sim_start = float(data.time)

                while viewer.is_running() and rclpy.ok():
                    with node.lock:
                        node._prepare_physics_step()
                        mujoco.mj_step(model, data)
                        node.publish_joint_states()
                        node.publish_tf()

                    viewer.sync()

                    sim_elapsed = float(data.time) - sim_start
                    wall_elapsed = time.perf_counter() - wall_start
                    delay = sim_elapsed - wall_elapsed

                    if delay > 0.0:
                        time.sleep(delay)

                print(
                    "[INFO] Viewer closed or loop finished.",
                    flush=True
                )

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received.", flush=True)

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

        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()