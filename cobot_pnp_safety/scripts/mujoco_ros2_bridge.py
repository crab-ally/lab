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

PROJECT_ROOT=Path(__file__).resolve().parent.parent
SCENE_XML=PROJECT_ROOT/"scene/panda_test.xml"
MODEL_DIR=PROJECT_ROOT/"model/franka_emika_panda"


def quat_conjugate(q):
    return np.array([q[0],-q[1],-q[2],-q[3]],dtype=np.float64)


def quat_multiply(q1,q2):
    w1,x1,y1,z1=q1
    w2,x2,y2,z2=q2
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2
    ],dtype=np.float64)


def quat_rot_vec(q,v):
    qv=np.array([0.0,v[0],v[1],v[2]],dtype=np.float64)
    return quat_multiply(quat_multiply(q,qv),quat_conjugate(q))[1:]


class MjcfTfPublisher(Node):
    def __init__(self,model,data):
        super().__init__('mjcf_tf_publisher')
        self.model=model
        self.data=data

        self.tf_broadcaster=TransformBroadcaster(self)
        self.joint_pub=self.create_publisher(JointState,'/joint_states',10)

        self.robot_root_id=mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "link0"
        )

        if self.robot_root_id==-1:
            self.get_logger().error("Panda root body 'link0' not found.")
            self.body_tree=[]
        else:
            self.body_tree=self._collect_all_bodies()
            self.get_logger().info(
                f"MJCF TF Publisher initialized with {len(self.body_tree)} bodies."
            )

        self.camera_frames=[]
        for cam_link in ["wrist_camera_link","ceiling_camera_link"]:
            cid=mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                cam_link
            )
            if cid!=-1:
                optical_name=cam_link.replace("_link","_optical_frame")
                self.camera_frames.append((cam_link,optical_name))
                self.get_logger().info(
                    f"Camera '{cam_link}' -> '{optical_name}' enabled."
                )

    def _collect_all_bodies(self):
        bodies=[]
        for i in range(1,self.model.nbody):
            child_name=mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                i
            )
            if not child_name:
                child_name=f"body_{i}"

            parent_id=int(self.model.body_parentid[i])

            if parent_id==0:
                parent_name="world"
            else:
                parent_name=mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    parent_id
                )
                if not parent_name:
                    parent_name=f"body_{parent_id}"

            bodies.append(
                (i,parent_id,parent_name,child_name)
            )

        return bodies

    def publish_mjcf_tf(self,stamp):
        transforms=[]

        for child_id,parent_id,parent_name,child_name in self.body_tree:
            child_pos=self.data.xpos[child_id]
            child_quat=self.data.xquat[child_id]

            if parent_id==0:
                rel_pos=child_pos
                rel_quat=child_quat
            else:
                parent_pos=self.data.xpos[parent_id]
                parent_quat=self.data.xquat[parent_id]

                parent_quat_inv=quat_conjugate(parent_quat)

                rel_pos=quat_rot_vec(
                    parent_quat_inv,
                    child_pos-parent_pos
                )

                rel_quat=quat_multiply(
                    parent_quat_inv,
                    child_quat
                )

            t=TransformStamped()
            t.header.stamp=stamp
            t.header.frame_id=parent_name
            t.child_frame_id=child_name

            t.transform.translation.x=float(rel_pos[0])
            t.transform.translation.y=float(rel_pos[1])
            t.transform.translation.z=float(rel_pos[2])

            t.transform.rotation.x=float(rel_quat[1])
            t.transform.rotation.y=float(rel_quat[2])
            t.transform.rotation.z=float(rel_quat[3])
            t.transform.rotation.w=float(rel_quat[0])

            transforms.append(t)

        for cam_link,optical_name in self.camera_frames:
            t=TransformStamped()
            t.header.stamp=stamp
            t.header.frame_id=cam_link
            t.child_frame_id=optical_name

            t.transform.translation.x=0.0
            t.transform.translation.y=0.0
            t.transform.translation.z=0.0

            q_optical=np.array([0.0,1.0,0.0,0.0],dtype=np.float64)

            t.transform.rotation.x=float(q_optical[1])
            t.transform.rotation.y=float(q_optical[2])
            t.transform.rotation.z=float(q_optical[3])
            t.transform.rotation.w=float(q_optical[0])

            transforms.append(t)

        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def publish_joint_states(self,stamp):
        if self.model.njnt==0:
            return

        msg=JointState()
        msg.header.stamp=stamp

        for i in range(self.model.njnt):
            jname=mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                i
            )

            if not jname:
                continue

            qpos_adr=self.model.jnt_qposadr[i]

            msg.name.append(jname)
            msg.position.append(
                float(self.data.qpos[qpos_adr])
            )

        self.joint_pub.publish(msg)

    def step_and_publish(self):
        stamp=self.get_clock().now().to_msg()
        self.publish_mjcf_tf(stamp)
        self.publish_joint_states(stamp)


def build_vfs():
    vfs_assets={}

    # world/test.xml
    world_xml=PROJECT_ROOT/"world"/"test.xml"
    if world_xml.exists():
        vfs_assets["../world/test.xml"]=world_xml.read_bytes()

    # panda.xml
    panda_xml=MODEL_DIR/"panda.xml"
    if panda_xml.exists():
        vfs_assets["../model/franka_emika_panda/panda.xml"]=panda_xml.read_bytes()

    # Panda assets
    assets_dir=MODEL_DIR/"assets"
    if assets_dir.exists():
        for file_path in assets_dir.rglob("*"):
            if not file_path.is_file():
                continue

            rel_path=file_path.relative_to(assets_dir)
            vfs_path=f"assets/{str(rel_path).replace(chr(92),'/')}"
            vfs_assets[vfs_path]=file_path.read_bytes()

    return vfs_assets


def main(args=None):
    parser=argparse.ArgumentParser(
        description="MJCF Panda TF Publisher"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer"
    )

    cli_args,ros_args=parser.parse_known_args()

    rclpy.init(args=ros_args)

    print(f"[INFO] Loading MJCF Scene: {SCENE_XML}")

    xml_string=SCENE_XML.read_text(encoding="utf-8")
    vfs_assets=build_vfs()

    try:
        model=mujoco.MjModel.from_xml_string(
            xml_string,
            assets=vfs_assets
        )
    except Exception as e:
        print(f"[ERROR] Failed to load MJCF: {e}")
        print("[ERROR] Registered VFS assets:")

        for name in sorted(vfs_assets.keys()):
            print(f"  - {name}")

        rclpy.shutdown()
        return

    data=mujoco.MjData(model)

    node=MjcfTfPublisher(model,data)

    try:
        if not cli_args.headless:
            with mujoco.viewer.launch_passive(
                model,
                data
            ) as viewer:

                while viewer.is_running() and rclpy.ok():
                    mujoco.mj_step(model,data)

                    node.step_and_publish()

                    rclpy.spin_once(
                        node,
                        timeout_sec=0.0
                    )

                    viewer.sync()

        else:
            print("[INFO] Running in headless mode...")

            while rclpy.ok():
                mujoco.mj_step(model,data)

                node.step_and_publish()

                rclpy.spin_once(
                    node,
                    timeout_sec=0.002
                )

                time.sleep(0.002)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=="__main__":
    main()