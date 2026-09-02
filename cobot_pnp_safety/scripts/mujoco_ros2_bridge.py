#!/usr/bin/env python3
"""
MuJoCo ROS 2 Bridge with FollowJointTrajectory & GripperCommand Action Servers
Supports MoveIt 2 Execution for Franka Emika Panda.

Publishes:
  /joint_states
  TF
  /camera/image_raw
  /camera/depth/image_raw
  /camera/depth/camera_info
"""
import argparse
from pathlib import Path
import time
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import String, Float64
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from control_msgs.action import FollowJointTrajectory, GripperCommand
import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = PROJECT_ROOT / "scene" / "panda_test.xml"
MODEL_DIR = PROJECT_ROOT / "model" / "franka_emika_panda"

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
FINGER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]

CAMERA_NAME = "ceiling_camera"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_PUBLISH_HZ = 10.0


def quat_conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2-w1*x2*0 + 0, 0, 0, 0], dtype=float) if False else np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dtype=float)


def quat_rotate(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[1:]


class MjcfBridgeNode(Node):
    def __init__(self, model, data):
        super().__init__("mujoco_ros2_bridge")
        self.model = model
        self.data = data
        self.cb_group = ReentrantCallbackGroup()

        self.lock = threading.Lock()
        self.current_arm_goal_handle = None
        self.active_trajectory = None
        self.traj_start_time = 0.0

        self.home_qpos = [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.57079, 0.785398]
        self._init_robot_pose()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera/depth/camera_info", 10)

        self.camera_name = CAMERA_NAME
        self.camera_width = CAMERA_WIDTH
        self.camera_height = CAMERA_HEIGHT
        self.camera_optical_frame = "ceiling_camera_optical_frame"

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id < 0:
            raise RuntimeError(f"Camera '{self.camera_name}' not found in MJCF.")

        self.camera_id = cam_id

        fovy = float(self.model.cam_fovy[self.camera_id])
        self.fy = (self.camera_height / 2.0) / np.tan(np.deg2rad(fovy / 2.0))
        self.fx = self.fy
        self.cx = (self.camera_width - 1) / 2.0
        self.cy = (self.camera_height - 1) / 2.0

        self.get_logger().info(f"[CAMERA] {self.camera_name}: {self.camera_width}x{self.camera_height}, fovy={fovy:.2f}, fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

        self.camera_period = 1.0 / CAMERA_PUBLISH_HZ
        self.camera_running = True

        self.env_bodies = self._collect_env_bodies()
        self.get_logger().info(f"Auto-tracked environment bodies for TF: {list(self.env_bodies.values())}")

        self._arm_action_server = ActionServer(self, FollowJointTrajectory, "/panda_arm_controller/follow_joint_trajectory", execute_callback=self.execute_arm_trajectory, goal_callback=self.goal_arm_callback, cancel_callback=self.cancel_arm_callback, callback_group=self.cb_group)

        self._gripper_action_server = ActionServer(self, GripperCommand, "/panda_hand_controller/gripper_action", execute_callback=self.execute_gripper_command, goal_callback=self.goal_gripper_callback, cancel_callback=self.cancel_gripper_callback, callback_group=self.cb_group)

        self.gripper_cmd_sub = self.create_subscription(String, "/panda_gripper_cmd", self.gripper_str_callback, 10)
        self.gripper_val_sub = self.create_subscription(Float64, "/panda_gripper_pos", self.gripper_float_callback, 10)

        # Start camera worker thread
        self.camera_thread = threading.Thread(target=self._camera_render_loop, daemon=True)
        self.camera_thread.start()

        self.get_logger().info("[READY] MuJoCo ROS2 Bridge with MoveIt 2 Controllers initialized (Camera decoupled in separate thread).")

    def _init_robot_pose(self):
        for i, name in enumerate(ARM_JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                qpos_adr = self.model.jnt_qposadr[jid]
                self.data.qpos[qpos_adr] = self.home_qpos[i]
                if i < len(self.data.ctrl):
                    self.data.ctrl[i] = self.home_qpos[i]
        if len(self.data.ctrl) >= 8:
            self.data.ctrl[7] = 255.0

    def _collect_env_bodies(self):
        env_bodies = {}
        robot_root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        for i in range(1, self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not body_name:
                continue
            parent_id = int(self.model.body_parentid[i])
            is_robot = False
            current_id = i
            while current_id != 0:
                if current_id == robot_root_id:
                    is_robot = True
                    break
                current_id = int(self.model.body_parentid[current_id])
            if is_robot:
                continue
            if parent_id == 0:
                parent_name = "world"
            else:
                parent_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
                if not parent_name:
                    parent_name = "world"
            env_bodies[i] = (parent_id, parent_name, body_name)
        return env_bodies

    def _get_relative_transform(self, body_id, parent_id):
        child_pos_world = np.asarray(self.data.xpos[body_id], dtype=float)
        child_quat_world = np.asarray(self.data.xquat[body_id], dtype=float)
        if parent_id == 0:
            return child_pos_world, child_quat_world
        parent_pos_world = np.asarray(self.data.xpos[parent_id], dtype=float)
        parent_quat_world = np.asarray(self.data.xquat[parent_id], dtype=float)
        parent_quat_inv = quat_conjugate(parent_quat_world)
        position_delta_world = child_pos_world - parent_pos_world
        position_parent = quat_rotate(parent_quat_inv, position_delta_world)
        relative_quat = quat_multiply(parent_quat_inv, child_quat_world)
        return position_parent, relative_quat

    def publish_env_tf(self, stamp):
        transforms = []
        for body_id, (parent_id, parent_name, body_name) in self.env_bodies.items():
            pos, quat = self._get_relative_transform(body_id, parent_id)
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = parent_name
            t.child_frame_id = body_name
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            t.transform.rotation.w = float(quat[0])
            t.transform.rotation.x = float(quat[1])
            t.transform.rotation.y = float(quat[2])
            t.transform.rotation.z = float(quat[3])
            transforms.append(t)
            if "camera_link" in body_name:
                optical_tf = TransformStamped()
                optical_tf.header.stamp = stamp
                optical_tf.header.frame_id = body_name
                optical_tf.child_frame_id = body_name.replace("_link", "_optical_frame")
                optical_tf.transform.translation.x = 0.0
                optical_tf.transform.translation.y = 0.0
                optical_tf.transform.translation.z = 0.0
                optical_tf.transform.rotation.w = 0.0
                optical_tf.transform.rotation.x = 1.0
                optical_tf.transform.rotation.y = 0.0
                optical_tf.transform.rotation.z = 0.0
                transforms.append(optical_tf)
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def publish_joint_states(self, stamp):
        msg = JointState()
        msg.header.stamp = stamp
        all_joints = ARM_JOINT_NAMES + FINGER_JOINT_NAMES
        for jname in all_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qpos_adr = self.model.jnt_qposadr[jid]
            dof_adr = self.model.jnt_dofadr[jid]
            msg.name.append(jname)
            msg.position.append(float(self.data.qpos[qpos_adr]))
            msg.velocity.append(float(self.data.qvel[dof_adr]))
        self.joint_pub.publish(msg)

    def _camera_render_loop(self):
        """Worker thread loop dedicated to rendering and publishing camera topics."""
        renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)
        render_data = mujoco.MjData(self.model)
        rate_interval = self.camera_period

        while self.camera_running and rclpy.ok():
            start_time = time.monotonic()

            # Snapshot the state with minimal lock holding time (<0.1ms)
            with self.lock:
                render_data.qpos[:] = self.data.qpos[:]
                render_data.qvel[:] = self.data.qvel[:]
                if self.model.nmocap > 0:
                    render_data.mocap_pos[:] = self.data.mocap_pos[:]
                    render_data.mocap_quat[:] = self.data.mocap_quat[:]

            mujoco.mj_forward(self.model, render_data)
            stamp = self.get_clock().now().to_msg()

            # 1. RGB Image
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
            renderer.update_scene(render_data, camera=self.camera_name)

            rgb = np.asarray(renderer.render(), dtype=np.uint8)
            image_msg = Image()
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = self.camera_optical_frame
            image_msg.height = self.camera_height
            image_msg.width = self.camera_width
            image_msg.encoding = "rgb8"
            image_msg.is_bigendian = False
            image_msg.step = self.camera_width * 3
            image_msg.data = rgb.tobytes()
            self.image_pub.publish(image_msg)

            # 2. Depth Image
            renderer.enable_depth_rendering()
            depth = np.asarray(renderer.render(), dtype=np.float32)
            depth_msg = Image()
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = self.camera_optical_frame
            depth_msg.height = self.camera_height
            depth_msg.width = self.camera_width
            depth_msg.encoding = "32FC1"
            depth_msg.is_bigendian = False
            depth_msg.step = self.camera_width * 4
            depth_msg.data = depth.tobytes()
            self.depth_pub.publish(depth_msg)
            renderer.disable_depth_rendering()

            # 3. Camera Info
            info_msg = CameraInfo()
            info_msg.header.stamp = stamp
            info_msg.header.frame_id = self.camera_optical_frame
            info_msg.width = self.camera_width
            info_msg.height = self.camera_height
            info_msg.distortion_model = "plumb_bob"
            info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            info_msg.k = [float(self.fx), 0.0, float(self.cx), 0.0, float(self.fy), float(self.cy), 0.0, 0.0, 1.0]
            info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info_msg.p = [float(self.fx), 0.0, float(self.cx), 0.0, 0.0, float(self.fy), float(self.cy), 0.0, 0.0, 0.0, 1.0, 0.0]
            self.camera_info_pub.publish(info_msg)

            elapsed = time.monotonic() - start_time
            sleep_time = rate_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def destroy_node(self):
        self.camera_running = False
        if hasattr(self, "camera_thread") and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        super().destroy_node()

    def goal_arm_callback(self, goal_request):
        self.get_logger().info(f"[ACTION] Arm trajectory goal received with {len(goal_request.trajectory.points)} points.")
        return GoalResponse.ACCEPT

    def cancel_arm_callback(self, goal_handle):
        self.get_logger().info("[ACTION] Arm trajectory goal cancelled.")
        return CancelResponse.ACCEPT

    def execute_arm_trajectory(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        joint_names = trajectory.joint_names
        points = trajectory.points
        if not points:
            goal_handle.succeed()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return result

        joint_indices = []
        for name in joint_names:
            joint_indices.append(ARM_JOINT_NAMES.index(name) if name in ARM_JOINT_NAMES else -1)

        start_time = time.time()
        point_idx = 0
        total_points = len(points)
        self.get_logger().info(f"[ACTION] Executing trajectory with {total_points} waypoints...")

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return result

            now_sec = time.time() - start_time

            while point_idx < total_points - 1:
                pt_time = points[point_idx].time_from_start.sec + points[point_idx].time_from_start.nanosec * 1e-9
                next_pt_time = points[point_idx + 1].time_from_start.sec + points[point_idx + 1].time_from_start.nanosec * 1e-9
                if now_sec < next_pt_time:
                    break
                point_idx += 1

            if point_idx >= total_points - 1:
                final_pt = points[-1]
                with self.lock:
                    for i, j_idx in enumerate(joint_indices):
                        if 0 <= j_idx < 7 and i < len(final_pt.positions):
                            self.data.ctrl[j_idx] = float(final_pt.positions[i])
                final_time = final_pt.time_from_start.sec + final_pt.time_from_start.nanosec * 1e-9
                if now_sec >= final_time + 0.1:
                    break
            else:
                p0 = points[point_idx]
                p1 = points[point_idx + 1]
                t0 = p0.time_from_start.sec + p0.time_from_start.nanosec * 1e-9
                t1 = p1.time_from_start.sec + p1.time_from_start.nanosec * 1e-9
                alpha = max(0.0, min(1.0, (now_sec - t0) / max(1e-4, t1 - t0)))
                with self.lock:
                    for i, j_idx in enumerate(joint_indices):
                        if 0 <= j_idx < 7 and i < len(p0.positions) and i < len(p1.positions):
                            self.data.ctrl[j_idx] = float(p0.positions[i] + alpha * (p1.positions[i] - p0.positions[i]))
            time.sleep(0.005)

        goal_handle.succeed()
        self.get_logger().info("[ACTION] Arm trajectory completed successfully.")
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def goal_gripper_callback(self, goal_request):
        self.get_logger().info(f"[ACTION] Gripper command received: pos={goal_request.command.position}")
        return GoalResponse.ACCEPT

    def cancel_gripper_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_gripper_command(self, goal_handle):
        pos = goal_handle.request.command.position
        ctrl_val = float(np.clip(pos / 0.04 * 255.0, 0.0, 255.0))
        with self.lock:
            if len(self.data.ctrl) >= 8:
                self.data.ctrl[7] = ctrl_val
        time.sleep(0.4)
        goal_handle.succeed()
        result = GripperCommand.Result()
        result.position = pos
        result.reached_goal = True
        return result

    def set_gripper_state(self, open_state: bool):
        with self.lock:
            if len(self.data.ctrl) >= 8:
                self.data.ctrl[7] = 255.0 if open_state else 0.0

    def gripper_str_callback(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd in ["open", "release"]:
            self.set_gripper_state(True)
            self.get_logger().info("[GRIPPER] Opened (ctrl=255)")
        elif cmd in ["close", "grasp"]:
            self.set_gripper_state(False)
            self.get_logger().info("[GRIPPER] Closed (ctrl=0)")

    def gripper_float_callback(self, msg: Float64):
        val = max(0.0, min(255.0, msg.data))
        with self.lock:
            if len(self.data.ctrl) >= 8:
                self.data.ctrl[7] = val

    def step_and_publish(self):
        stamp = self.get_clock().now().to_msg()
        self.publish_joint_states(stamp)
        self.publish_env_tf(stamp)


def build_vfs():
    vfs_assets = {}
    world_xml = PROJECT_ROOT / "world" / "test.xml"
    if world_xml.exists():
        vfs_assets["../world/test.xml"] = world_xml.read_bytes()
    panda_xml = MODEL_DIR / "panda.xml"
    if panda_xml.exists():
        vfs_assets["../model/franka_emika_panda/panda.xml"] = panda_xml.read_bytes()
    assets_dir = MODEL_DIR / "assets"
    if assets_dir.exists():
        for file_path in assets_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(assets_dir)
                vfs_path = f"assets/{str(rel_path).replace(chr(92), '/')}"
                vfs_assets[vfs_path] = file_path.read_bytes()
    return vfs_assets


def main(args=None):
    parser = argparse.ArgumentParser(description="MuJoCo ROS2 MoveIt2 Auto Bridge")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    xml_string = SCENE_XML.read_text(encoding="utf-8")
    vfs_assets = build_vfs()

    try:
        model = mujoco.MjModel.from_xml_string(xml_string, assets=vfs_assets)
    except Exception as e:
        print(f"[ERROR] Failed to load MJCF: {e}")
        rclpy.shutdown()
        return

    data = mujoco.MjData(model)
    node = MjcfBridgeNode(model, data)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if not cli_args.headless:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.lookat[:] = [0, 0, 0]
                viewer.cam.distance = 4
                viewer.cam.azimuth = 0
                viewer.cam.elevation = -45
                while viewer.is_running() and rclpy.ok():
                    with node.lock:
                        mujoco.mj_step(model, data)
                        node.step_and_publish()
                    viewer.sync()
                    time.sleep(0.002)
        else:
            while rclpy.ok():
                with node.lock:
                    mujoco.mj_step(model, data)
                    node.step_and_publish()
                time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()