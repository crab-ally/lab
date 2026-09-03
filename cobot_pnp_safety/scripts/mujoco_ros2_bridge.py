#!/usr/bin/env python3
"""
MuJoCo ROS 2 Bridge with FollowJointTrajectory & GripperCommand Action Servers
Supports MoveIt 2 Execution for Franka Emika Panda.

HW Spec
  - 최대 작업 반경: 0.855m
  - 그리퍼 동작 범위: 0.08m
  - 최대 하중: 2.27kg (순수 물체 무게)

Publishes:
  /joint_states
  TF
  /camera/image_raw
  /camera/depth/image_raw
  /camera/depth/camera_info
  /camera/segmentation/image_raw
"""

import argparse
from pathlib import Path
import time,threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer,GoalResponse,CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState,Image,CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String,Float64
from tf2_ros import TransformBroadcaster,StaticTransformBroadcaster
from control_msgs.action import FollowJointTrajectory,GripperCommand

import mujoco
import mujoco.viewer


class MjcfBridgeNode(Node):
    def __init__(self,model,data):
        super().__init__("mujoco_ros_bridge")
        self.model,self.data=model,data
        self.lock=threading.Lock()
        self.running=True
        self.cb_group=ReentrantCallbackGroup()

        # ============================================================
        # Panda joints
        # ============================================================
        self.arm_joints=[f"joint{i}" for i in range(1,8)]
        self.finger_joints=["finger_joint1","finger_joint2"]
        self.all_joints=self.arm_joints+self.finger_joints
        self.joint_ids={
            n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
            for n in self.all_joints
        }
        self.actuator_ids={
            f"joint{i}":mujoco.mj_name2id(
                model,mujoco.mjtObj.mjOBJ_ACTUATOR,f"actuator{i}"
            ) for i in range(1,8)
        }
        self.GRIPPER_ACTUATOR_ID=7

        # ============================================================
        # Camera
        # ============================================================
        self.camera_name="ceiling_camera"
        self.camera_width,self.camera_height=640,480
        self.camera_rate=10.0
        self.camera_optical_frame="ceiling_camera_optical_frame"
        self.camera_link_frame="ceiling_camera_link"

        # ============================================================
        # PNP object
        # ============================================================
        self.pnp_object_geom="pnp_object_geom"
        self.pnp_object_geom_id=mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_GEOM,self.pnp_object_geom
        )

        # ============================================================
        # Panda geom IDs
        # ============================================================
        self.finger_geom_ids=self._collect_finger_geom_ids()
        self.panda_geom_ids=self._collect_panda_geom_ids()

        # ============================================================
        # ROS publishers
        # ============================================================
        self.joint_pub=self.create_publisher(JointState,"/joint_states",10)
        self.rgb_pub=self.create_publisher(Image,"/camera/image_raw",10)
        self.depth_pub=self.create_publisher(Image,"/camera/depth/image_raw",10)
        self.seg_pub=self.create_publisher(Image,"/camera/segmentation/image_raw",10)
        self.info_pub=self.create_publisher(CameraInfo,"/camera/depth/camera_info",10)
        self.tf_broadcaster=TransformBroadcaster(self)
        self.static_tf_broadcaster=StaticTransformBroadcaster(self)

        # ============================================================
        # Gripper topics
        # ============================================================
        self.gripper_cmd_pub=self.create_publisher(
            Float64,"/panda_gripper/command",10
        )
        self.gripper_string_pub=self.create_publisher(
            String,"/panda_gripper/cmd",10
        )
        self.create_subscription(
            String,"/panda_gripper/cmd",self.gripper_string_callback,10,
            callback_group=self.cb_group
        )
        self.create_subscription(
            Float64,"/panda_gripper/command",self.gripper_float_callback,10,
            callback_group=self.cb_group
        )

        # ============================================================
        # Arm Action
        # ============================================================
        self.arm_action=ActionServer(
            self,FollowJointTrajectory,
            "/panda_arm_controller/follow_joint_trajectory",
            execute_callback=self.execute_arm,
            goal_callback=self.arm_goal,
            cancel_callback=self.cancel_goal,
            callback_group=self.cb_group
        )

        # ============================================================
        # Gripper Action
        # ============================================================
        self.gripper_action=ActionServer(
            self,GripperCommand,
            "/panda_hand_controller/gripper_action",
            execute_callback=self.execute_gripper,
            goal_callback=self.gripper_goal,
            cancel_callback=self.cancel_goal,
            callback_group=self.cb_group
        )

        # ============================================================
        # Motion parameters
        # ============================================================
        self.home_qpos=np.array(
            [0.0,-0.785398,0.0,-2.35619,0.0,1.57079,0.785398]
        )
        self.POSITION_TOLERANCE=0.01
        self.VELOCITY_TOLERANCE=0.1
        self.SETTLE_TIMEOUT=5.0
        self.GRIPPER_STABLE_TIME=0.15
        self.GRIPPER_TIMEOUT=2.0
        self.GRIPPER_CONTACT_STABLE_TIME=0.15

        # ============================================================
        # Initial pose
        # ============================================================
        self._set_initial_pose()
        self.publish_static_camera_tf()

        # ============================================================
        # Camera rendering thread
        # ============================================================
        self.camera_thread=threading.Thread(
            target=self._camera_render_loop,daemon=True
        )
        self.camera_thread.start()
        self.get_logger().info("MuJoCo ROS 2 Bridge started.")

    # ================================================================
    # Static camera TF
    # ================================================================
    def publish_static_camera_tf(self):
        stamp=self.get_clock().now().to_msg()
        transforms=[]
        cam_id=mujoco.mj_name2id(
            self.model,mujoco.mjtObj.mjOBJ_CAMERA,self.camera_name
        )
        cam_body=self.model.cam_bodyid[cam_id]

        # ------------------------------------------------------------
        # world -> ceiling_camera_link
        # ------------------------------------------------------------
        cam_name=mujoco.mj_id2name(
            self.model,mujoco.mjtObj.mjOBJ_BODY,cam_body
        )
        if cam_name:
            t=TransformStamped()
            t.header.stamp=stamp
            t.header.frame_id="world"
            t.child_frame_id=self.camera_link_frame
            p=self.data.xpos[cam_body]
            q=self._mat_to_quat(self.data.xmat[cam_body].reshape(3,3))
            t.transform.translation.x=float(p[0])
            t.transform.translation.y=float(p[1])
            t.transform.translation.z=float(p[2])
            t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w=q
            transforms.append(t)

        # ------------------------------------------------------------
        # camera_link -> optical
        # ------------------------------------------------------------
        t=TransformStamped()
        t.header.stamp=stamp
        t.header.frame_id=self.camera_link_frame
        t.child_frame_id=self.camera_optical_frame
        t.transform.rotation.x=1.0
        t.transform.rotation.w=0.0
        transforms.append(t)
        self.static_tf_broadcaster.sendTransform(transforms)

    # ================================================================
    # Panda geom collection
    # ================================================================
    def _collect_panda_geom_ids(self):
        root=self.model.body("link0").id
        ids=set()
        for gid in range(self.model.ngeom):
            body=int(self.model.geom_bodyid[gid])
            while body>0:
                if body==root:
                    ids.add(gid)
                    break
                body=int(self.model.body_parentid[body])
        return ids

    def _collect_finger_geom_ids(self):
        ids=set()
        for gid in range(self.model.ngeom):
            body=int(self.model.geom_bodyid[gid])
            while body>0:
                name=mujoco.mj_id2name(
                    self.model,mujoco.mjtObj.mjOBJ_BODY,body
                )
                if name and ("finger" in name or "hand" in name):
                    ids.add(gid)
                    break
                body=int(self.model.body_parentid[body])
        return ids

    # ================================================================
    # Initial pose
    # ================================================================
    def _set_initial_pose(self):
        for i,name in enumerate(self.arm_joints):
            jid=self.joint_ids[name]
            self.data.qpos[self.model.jnt_qposadr[jid]]=self.home_qpos[i]
        self.data.ctrl[self.GRIPPER_ACTUATOR_ID]=255.0
        mujoco.mj_forward(self.model,self.data)

    # ================================================================
    # Camera render loop
    # ================================================================
    def _camera_render_loop(self):
        renderer=mujoco.Renderer(
            self.model,height=self.camera_height,width=self.camera_width
        )
        depth_renderer=mujoco.Renderer(
            self.model,height=self.camera_height,width=self.camera_width
        )
        seg_renderer=mujoco.Renderer(
            self.model,height=self.camera_height,width=self.camera_width
        )
        depth_renderer.enable_depth_rendering()
        seg_renderer.enable_segmentation_rendering()

        render_data=mujoco.MjData(self.model)
        period=1.0/self.camera_rate
        next_time=time.monotonic()

        while self.running:
            next_time+=period

            # --------------------------------------------------------
            # Copy simulation state
            # --------------------------------------------------------
            with self.lock:
                render_data.qpos[:]=self.data.qpos
                render_data.qvel[:]=self.data.qvel
                render_data.act[:]=self.data.act
                if self.model.nmocap:
                    render_data.mocap_pos[:]=self.data.mocap_pos
                    render_data.mocap_quat[:]=self.data.mocap_quat
                mujoco.mj_forward(self.model,render_data)

            # --------------------------------------------------------
            # RGB
            # --------------------------------------------------------
            renderer.update_scene(render_data,camera=self.camera_name)
            rgb=np.asarray(renderer.render()).copy()

            # --------------------------------------------------------
            # Depth
            # --------------------------------------------------------
            depth_renderer.update_scene(render_data,camera=self.camera_name)
            depth=np.asarray(depth_renderer.render()).copy()

            # --------------------------------------------------------
            # Segmentation
            # --------------------------------------------------------
            seg_renderer.update_scene(render_data,camera=self.camera_name)
            seg_raw=np.asarray(seg_renderer.render()).copy()
            seg_id=seg_raw[:,:,0].astype(np.int32)
            seg_type=seg_raw[:,:,1].astype(np.int32)

            geom_mask=seg_type==int(mujoco.mjtObj.mjOBJ_GEOM)
            panda_mask=geom_mask & np.isin(
                seg_id,np.asarray(list(self.panda_geom_ids),dtype=np.int32)
            )
            seg=seg_id.copy()
            seg[~geom_mask]=0
            seg[panda_mask]=0

            # --------------------------------------------------------
            # Same timestamp
            # --------------------------------------------------------
            stamp=self.get_clock().now().to_msg()

            # --------------------------------------------------------
            # RGB
            # --------------------------------------------------------
            self.rgb_pub.publish(self._image_msg(
                rgb,"rgb8",3,stamp
            ))

            # --------------------------------------------------------
            # Depth
            # --------------------------------------------------------
            self.depth_pub.publish(self._image_msg(
                depth,"32FC1",4,stamp
            ))

            # --------------------------------------------------------
            # Segmentation
            # --------------------------------------------------------
            self.seg_pub.publish(self._image_msg(
                seg,"32SC1",4,stamp
            ))

            # --------------------------------------------------------
            # CameraInfo
            # ------------------------------------------------------------
            self.info_pub.publish(self._camera_info(stamp))

            # --------------------------------------------------------
            # Maintain camera rate
            # --------------------------------------------------------
            sleep_time=next_time-time.monotonic()
            if sleep_time>0:
                time.sleep(sleep_time)
            else:
                next_time=time.monotonic()

        renderer.close()
        depth_renderer.close()
        seg_renderer.close()

    def _image_msg(self,data,encoding,bpp,stamp):
        msg=Image()
        msg.header.stamp=stamp
        msg.header.frame_id=self.camera_optical_frame
        msg.height,msg.width=self.camera_height,self.camera_width
        msg.encoding=encoding
        msg.is_bigendian=False
        msg.step=self.camera_width*bpp
        msg.data=data.astype(
            np.uint8 if encoding=="rgb8" else (
                np.float32 if encoding=="32FC1" else np.int32
            )
        ).tobytes()
        return msg

    # ================================================================
    # Camera info
    # ================================================================
    def _camera_info(self,stamp):
        cid=mujoco.mj_name2id(
            self.model,mujoco.mjtObj.mjOBJ_CAMERA,self.camera_name
        )
        fovy=self.model.cam_fovy[cid]
        fy=(self.camera_height/2.0)/np.tan(np.deg2rad(fovy)/2.0)
        fx=fy
        cx=(self.camera_width-1)/2.0
        cy=(self.camera_height-1)/2.0

        msg=CameraInfo()
        msg.header.stamp=stamp
        msg.header.frame_id=self.camera_optical_frame
        msg.width,msg.height=self.camera_width,self.camera_height
        msg.k=[fx,0.0,cx,0.0,fy,cy,0.0,0.0,1.0]
        msg.p=[fx,0.0,cx,0.0,0.0,fy,cy,0.0,0.0,0.0,1.0,0.0]
        msg.r=[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
        return msg

    # ================================================================
    # Joint states
    # ================================================================
    def publish_joint_states(self):
        msg=JointState()
        msg.header.stamp=self.get_clock().now().to_msg()
        for name in self.all_joints:
            jid=self.joint_ids[name]
            msg.name.append(name)
            msg.position.append(
                float(self.data.qpos[self.model.jnt_qposadr[jid]])
            )
            msg.velocity.append(
                float(self.data.qvel[self.model.jnt_dofadr[jid]])
            )
        self.joint_pub.publish(msg)

    # ================================================================
    # Rotation / quaternion
    # ================================================================
    def _mat_to_quat(self,R):
        q=np.zeros(4)
        mujoco.mju_mat2Quat(q,R.reshape(-1))
        return float(q[1]),float(q[2]),float(q[3]),float(q[0])

    # ================================================================
    # Relative transform
    # ================================================================
    def _relative_transform(self,body_id,parent_id):
        p=self.data.xpos[body_id]
        pp=self.data.xpos[parent_id]
        R=self.data.xmat[body_id].reshape(3,3)
        Rp=self.data.xmat[parent_id].reshape(3,3)
        return Rp.T@(p-pp),Rp.T@R

    # ================================================================
    # TF
    # ================================================================
    def publish_tf(self):
        stamp=self.get_clock().now().to_msg()

        # ------------------------------------------------------------
        # world -> link0
        # ------------------------------------------------------------
        root=self.model.body("link0").id
        t=TransformStamped()
        t.header.stamp=stamp
        t.header.frame_id="world"
        t.child_frame_id="link0"
        p=self.data.xpos[root]
        q=self._mat_to_quat(self.data.xmat[root].reshape(3,3))
        t.transform.translation.x=float(p[0])
        t.transform.translation.y=float(p[1])
        t.transform.translation.z=float(p[2])
        t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w=q
        self.tf_broadcaster.sendTransform(t)

        # ------------------------------------------------------------
        # Panda body TF
        # ------------------------------------------------------------
        for bid in range(1,self.model.nbody):
            name=mujoco.mj_id2name(
                self.model,mujoco.mjtObj.mjOBJ_BODY,bid
            )
            if not name or name in ("link0",self.camera_link_frame):
                continue

            parent=self.model.body_parentid[bid]
            parent_name=mujoco.mj_id2name(
                self.model,mujoco.mjtObj.mjOBJ_BODY,parent
            )
            if not parent_name or parent==0:
                continue

            pos,rot=self._relative_transform(bid,parent)
            t=TransformStamped()
            t.header.stamp=stamp
            t.header.frame_id=parent_name
            t.child_frame_id=name
            t.transform.translation.x=float(pos[0])
            t.transform.translation.y=float(pos[1])
            t.transform.translation.z=float(pos[2])

            q=self._mat_to_quat(rot)
            t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w=q
            self.tf_broadcaster.sendTransform(t)

    # ================================================================
    # Arm action callbacks
    # ================================================================
    def arm_goal(self,goal):
        # FollowJointTrajectory.Goal에는 request가 없다.
        # trajectory가 직접 존재한다.
        return GoalResponse.ACCEPT if goal.trajectory.joint_names else GoalResponse.REJECT

    def gripper_goal(self,goal):
        return GoalResponse.ACCEPT

    def cancel_goal(self,goal):
        return CancelResponse.ACCEPT

    # ================================================================
    # Arm execution
    # ================================================================
    def execute_arm(self,goal_handle):
        traj=goal_handle.request.trajectory

        if not traj.points:
            goal_handle.abort()
            result=FollowJointTrajectory.Result()
            result.error_code=FollowJointTrajectory.Result.INVALID_GOAL
            return result

        name_to_idx={n:i for i,n in enumerate(traj.joint_names)}

        if any(n not in name_to_idx for n in self.arm_joints):
            goal_handle.abort()
            result=FollowJointTrajectory.Result()
            result.error_code=FollowJointTrajectory.Result.INVALID_JOINTS
            return result

        with self.lock:
            current=np.array([
                self.data.qpos[self.model.jnt_qposadr[self.joint_ids[n]]]
                for n in self.arm_joints
            ])

        prev_q=current.copy()
        prev_t=0.0

        for point in traj.points:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            t=float(point.time_from_start.sec)+point.time_from_start.nanosec*1e-9
            target=np.array([
                point.positions[name_to_idx[n]]
                for n in self.arm_joints
            ],dtype=float)

            dt=max(t-prev_t,0.001)
            steps=max(1,int(np.ceil(dt/0.005)))

            for k in range(1,steps+1):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return FollowJointTrajectory.Result()

                q=prev_q+(target-prev_q)*(k/steps)

                with self.lock:
                    for i,name in enumerate(self.arm_joints):
                        self.data.ctrl[self.actuator_ids[name]]=q[i]

                # asyncio 사용하지 않는다.
                time.sleep(dt/steps)

            prev_q=target.copy()
            prev_t=t

        # ------------------------------------------------------------
        # Settle
        # ------------------------------------------------------------
        deadline=time.monotonic()+self.SETTLE_TIMEOUT

        while time.monotonic()<deadline:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            with self.lock:
                q=np.array([
                    self.data.qpos[self.model.jnt_qposadr[self.joint_ids[n]]]
                    for n in self.arm_joints
                ])
                v=np.array([
                    self.data.qvel[self.model.jnt_dofadr[self.joint_ids[n]]]
                    for n in self.arm_joints
                ])

            if np.max(np.abs(q-prev_q))<=self.POSITION_TOLERANCE and \
               np.max(np.abs(v))<=self.VELOCITY_TOLERANCE:
                result=FollowJointTrajectory.Result()
                result.error_code=FollowJointTrajectory.Result.SUCCESSFUL
                goal_handle.succeed()
                return result

            time.sleep(0.01)

        goal_handle.abort()
        result=FollowJointTrajectory.Result()
        result.error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
        return result

    # ================================================================
    # Gripper
    # ================================================================
    def _gripper_position(self):
        if not self.finger_joints:
            return 0.0

        with self.lock:
            values=[
                self.data.qpos[self.model.jnt_qposadr[self.joint_ids[n]]]
                for n in self.finger_joints
            ]
        return float(np.mean(values))

    def _gripper_contact(self):
        if self.pnp_object_geom_id<0:
            return False

        with self.lock:
            for i in range(self.data.ncon):
                a,b=int(self.data.contact[i].geom1),int(self.data.contact[i].geom2)
                if ((a==self.pnp_object_geom_id and b in self.finger_geom_ids) or
                    (b==self.pnp_object_geom_id and a in self.finger_geom_ids)):
                    return True
        return False

    def _gripper_velocity_ok(self):
        with self.lock:
            v=[
                abs(self.data.qvel[self.model.jnt_dofadr[self.joint_ids[n]]])
                for n in self.finger_joints
            ]
        return max(v,default=0.0)<=self.VELOCITY_TOLERANCE

    # ================================================================
    # Gripper action
    # ================================================================
    def execute_gripper(self,goal_handle):
        target=float(np.clip(goal_handle.request.command.position,0.0,0.04))
        ctrl=float(np.clip(target/0.04*255.0,0.0,255.0))

        with self.lock:
            self.data.ctrl[self.GRIPPER_ACTUATOR_ID]=ctrl

        start=time.monotonic()
        stable_start=None

        while time.monotonic()-start<self.GRIPPER_TIMEOUT:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return GripperCommand.Result()

            pos=self._gripper_position()

            if target>0.02:
                reached=pos>=0.035 and self._gripper_velocity_ok()
                stable_time=self.GRIPPER_STABLE_TIME
            else:
                contact=self._gripper_contact()
                velocity_ok=self._gripper_velocity_ok()
                reached=contact and velocity_ok
                stable_time=self.GRIPPER_CONTACT_STABLE_TIME

            if reached:
                if stable_start is None:
                    stable_start=time.monotonic()
                elif time.monotonic()-stable_start>=stable_time:
                    result=GripperCommand.Result()
                    result.position=pos
                    result.reached_goal=True
                    result.stalled=False
                    goal_handle.succeed()
                    return result
            else:
                stable_start=None

            time.sleep(0.01)

        result=GripperCommand.Result()
        result.position=self._gripper_position()
        result.reached_goal=False
        result.stalled=True
        goal_handle.abort()
        return result

    # ================================================================
    # Gripper topic callbacks
    # ================================================================
    def gripper_string_callback(self,msg):
        cmd=msg.data.lower().strip()

        if cmd in ("open","release"):
            value=255.0
        elif cmd in ("close","grasp"):
            value=0.0
        else:
            return

        with self.lock:
            self.data.ctrl[self.GRIPPER_ACTUATOR_ID]=value

    def gripper_float_callback(self,msg):
        with self.lock:
            self.data.ctrl[self.GRIPPER_ACTUATOR_ID]=float(
                np.clip(msg.data,0.0,255.0)
            )

    # ================================================================
    # Destroy
    # ================================================================
    def destroy_node(self):
        self.running=False

        if hasattr(self,"camera_thread") and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)

        super().destroy_node()


# ====================================================================
# MuJoCo model loading
# ====================================================================
def load_model(xml_path):
    xml_path=Path(xml_path)

    try:
        return mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as e:
        print(f"[WARN] from_xml_path failed: {e}")
        print("[INFO] Trying VFS fallback...")

    root=Path("/workspace")
    vfs={}

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".xml",".stl",".obj",".png",".jpg",".jpeg"
        }:
            continue
        try:
            vfs[path.relative_to(root).as_posix()]=path.read_bytes()
        except Exception:
            pass

    print(f"[INFO] VFS files: {len(vfs)}")
    return mujoco.MjModel.from_xml_string(
        xml_path.read_text(),assets=vfs
    )


# ====================================================================
# Main
# ====================================================================
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="/workspace/scene/panda_test.xml")
    parser.add_argument("--headless",action="store_true")
    args=parser.parse_args()

    rclpy.init()
    model=load_model(args.model)
    data=mujoco.MjData(model)
    node=MjcfBridgeNode(model,data)

    executor=rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread=threading.Thread(target=executor.spin,daemon=True)
    spin_thread.start()

    try:
        if args.headless:
            print("[INFO] Running in headless mode (no viewer)...",flush=True)

            while rclpy.ok():
                with node.lock:
                    mujoco.mj_step(model,data)
                    node.publish_joint_states()
                    node.publish_tf()
                time.sleep(0.002)

        else:
            print("[INFO] Launching MuJoCo passive viewer...",flush=True)

            with mujoco.viewer.launch_passive(model,data) as viewer:
                viewer.cam.distance=1.5
                viewer.cam.azimuth=180
                viewer.cam.elevation=-20
                viewer.cam.lookat[:]=[0,0.5,0.5]
                print(
                    "[INFO] MuJoCo passive viewer launched successfully.",
                    flush=True
                )

                while viewer.is_running() and rclpy.ok():
                    with node.lock:
                        mujoco.mj_step(model,data)
                        node.publish_joint_states()
                        node.publish_tf()

                    viewer.sync()
                    time.sleep(0.002)

                print("[INFO] Viewer closed or loop finished.",flush=True)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received.",flush=True)

    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in simulation loop: {e}",flush=True)
        traceback.print_exc()

    finally:
        print("[INFO] Shutting down bridge node...",flush=True)
        node.running=False
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        spin_thread.join(timeout=2.0)


if __name__=="__main__":
    main()