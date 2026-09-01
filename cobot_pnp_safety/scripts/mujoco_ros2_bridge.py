#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = PROJECT_ROOT / "scene" / "panda_test.xml"
MODEL_DIR = PROJECT_ROOT / "model" / "franka_emika_panda"


def quat_conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_rotate(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[1:]


class MjcfBridgeNode(Node):
    def __init__(self, model, data):
        super().__init__('mujoco_ros2_bridge')
        self.model = model
        self.data = data

        self.tf_broadcaster = TransformBroadcaster(self)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        self.env_bodies = self._collect_env_bodies()
        self.get_logger().info(f"Auto-tracked environment bodies for TF: {list(self.env_bodies.values())}")

    def _collect_env_bodies(self):
        env_bodies={}
        robot_root_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"link0")

        for i in range(1,self.model.nbody):
            body_name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,i)
            if not body_name:
                continue

            parent_id=int(self.model.body_parentid[i])

            is_robot=False
            current_id=i

            while current_id!=0:
                if current_id==robot_root_id:
                    is_robot=True
                    break
                current_id=int(self.model.body_parentid[current_id])

            if is_robot:
                continue

            if parent_id==0:
                parent_name="world"
            else:
                parent_name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,parent_id)
                if not parent_name:
                    parent_name="world"

            env_bodies[i]=(parent_id,parent_name,body_name)

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

        relative_quat = quat_multiply(
            parent_quat_inv,
            child_quat_world
        )

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
                optical_tf.child_frame_id = body_name.replace(
                    "_link",
                    "_optical_frame"
                )

                optical_tf.transform.translation.x = 0.0
                optical_tf.transform.translation.y = 0.0
                optical_tf.transform.translation.z = 0.0

                # 현재 MuJoCo ceiling camera에 맞는 Optical Frame 회전
                # ROS quaternion: [x,y,z,w] = [1,0,0,0]
                # 즉 X축 기준 180도 회전
                optical_tf.transform.rotation.w = 0.0
                optical_tf.transform.rotation.x = 1.0
                optical_tf.transform.rotation.y = 0.0
                optical_tf.transform.rotation.z = 0.0

                transforms.append(optical_tf)

        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def publish_joint_states(self,stamp):
        msg=JointState()
        msg.header.stamp=stamp
        for jname in ["joint1","joint2","joint3","joint4","joint5","joint6","joint7","finger_joint1"]:
            jid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,jname)
            if jid<0: continue
            qpos_adr=self.model.jnt_qposadr[jid]
            msg.name.append(jname)
            msg.position.append(float(self.data.qpos[qpos_adr]))
        self.joint_pub.publish(msg)

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
    parser = argparse.ArgumentParser(
        description="MuJoCo ROS2 Auto Bridge"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer"
    )

    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    xml_string = SCENE_XML.read_text(encoding="utf-8")
    vfs_assets = build_vfs()

    try:
        model = mujoco.MjModel.from_xml_string(
            xml_string,
            assets=vfs_assets
        )
    except Exception as e:
        print(f"[ERROR] Failed to load MJCF: {e}")
        rclpy.shutdown()
        return

    data = mujoco.MjData(model)
    node = MjcfBridgeNode(model, data)

    try:
        if not cli_args.headless:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and rclpy.ok():
                    mujoco.mj_step(model, data)
                    node.step_and_publish()
                    rclpy.spin_once(node, timeout_sec=0.0)
                    viewer.sync()
        else:
            while rclpy.ok():
                mujoco.mj_step(model, data)
                node.step_and_publish()
                rclpy.spin_once(node, timeout_sec=0.002)
                time.sleep(0.002)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()