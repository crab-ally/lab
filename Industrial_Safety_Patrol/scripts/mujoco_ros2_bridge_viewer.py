#!/usr/bin/env python3
"""
timestep: 0.005
viewer.sync(): 30Hz

Subscribes:
    - /cmd_vel
    - /forklift_1/cmd_vel
    - /forklift_2/cmd_vel

Publishes:
    - /odom (50Hz)
    - /scan (10Hz)
    - /camera/image_raw (10Hz)
    - /camera/depth/image_raw (10Hz)
    - /camera/depth/camera_info (10Hz)
    - /clock (50Hz)
    - /forklift_1/pose
    - /forklift_2/pose
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import cv_bridge
import math
from pathlib import Path
from rosgraph_msgs.msg import Clock
from rclpy.parameter import Parameter
from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class MujocoRosBridge(Node):
    ODOM_INTERVAL   = 1.0 / 50   # 50 Hz
    SCAN_INTERVAL   = 1.0 / 10   # 10 Hz
    CAMERA_INTERVAL = 1.0 / 10   # 10 Hz
    VIEWER_HZ       = 30

    def __init__(self, model, data):
        super().__init__(
            'mujoco_ros_bridge',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        self.model = model
        self.data = data
        self.cv_bridge = cv_bridge.CvBridge()

        # 스레드 동기화용 락
        self.physics_lock = threading.Lock()

        # 제원 및 이미지 설정
        self.max_linear_vel = 0.25
        self.max_angular_vel = 1.0
        self.track_width = 0.160
        self.wheel_radius = 0.033
        self.img_width = 640
        self.img_height = 480

        self.last_odom_time = 0.0
        self.sim_time = 0.0
        self.last_scan_time = 0.0
        self.last_scan_publish_time = None

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.fl1_cmd_vel_sub = self.create_subscription(Twist, '/forklift_1/cmd_vel', self.fl1_cmd_vel_callback, 10)
        self.fl2_cmd_vel_sub = self.create_subscription(Twist, '/forklift_2/cmd_vel', self.fl2_cmd_vel_callback, 10)

        self.fl1_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "fl1_freejoint")
        self.fl2_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "fl2_freejoint")

        # Publishers
        clock_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.clock_pub = self.create_publisher(Clock, '/clock', clock_qos)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.camera_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 10)
        self.fl1_pose_pub = self.create_publisher(PoseStamped, '/forklift_1/pose', 10)
        self.fl2_pose_pub = self.create_publisher(PoseStamped, '/forklift_2/pose', 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.lidar_beam_count = 360
        self._init_lidar_sensors()

        # Odom 원점 캡처
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if joint_id == -1:
            raise RuntimeError("root freejoint를 모델에서 찾을 수 없습니다")

        self._qpos_adr = self.model.jnt_qposadr[joint_id]
        self._qvel_adr = self.model.jnt_dofadr[joint_id]

        self.origin_pos = np.array(self.data.qpos[self._qpos_adr: self._qpos_adr + 3], dtype=np.float64).copy()
        self.origin_quat = np.array(self.data.qpos[self._qpos_adr + 3: self._qpos_adr + 7], dtype=np.float64).copy()
        self.origin_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(self.origin_quat_inv, self.origin_quat)

        # 재사용 버퍼
        self._buf_quat_inv     = np.zeros(4)
        self._buf_vel_base     = np.zeros(3)
        self._buf_ang_vel_base = np.zeros(3)
        self._buf_quat_odom    = np.zeros(4)
        self._buf_pos_odom     = np.zeros(3)

        self._publish_static_transforms()
        self._init_camera_params()

    def _publish_static_transforms(self):
        now = self.get_clock().now().to_msg()
        
        lidar_tf = TransformStamped()
        lidar_tf.header.stamp = now
        lidar_tf.header.frame_id = 'base_footprint'
        lidar_tf.child_frame_id = 'lidar_link'
        lidar_tf.transform.translation.z = 0.1675
        lidar_tf.transform.rotation.w = 1.0

        camera_tf = TransformStamped()
        camera_tf.header.stamp = now
        camera_tf.header.frame_id = 'base_footprint'
        camera_tf.child_frame_id = 'camera_link'
        camera_tf.transform.translation.x = -0.05
        camera_tf.transform.translation.z = 0.35
        camera_tf.transform.rotation.x = -0.5
        camera_tf.transform.rotation.y = 0.5
        camera_tf.transform.rotation.z = -0.5
        camera_tf.transform.rotation.w = 0.5

        self.static_tf_broadcaster.sendTransform([lidar_tf, camera_tf])

    def _init_lidar_sensors(self):
        combined_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "lidar")
        if combined_id != -1 and self.model.sensor_dim[combined_id] >= self.lidar_beam_count:
            self.lidar_mode = "combined"
            self.lidar_sensor_id = combined_id
            return

        replicated = []
        for sensor_id in range(self.model.nsensor):
            if self.model.sensor_type[sensor_id] != mujoco.mjtSensor.mjSENS_RANGEFINDER:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            if name and (name == "lidar" or name.startswith("lidar-")):
                replicated.append((name, sensor_id))
        replicated.sort(key=lambda item: item[0])
        self.lidar_mode = "replicated"
        self.lidar_sensor_ids = replicated[: self.lidar_beam_count]
        self._lidar_adrs = np.array([self.model.sensor_adr[sid] for _, sid in self.lidar_sensor_ids], dtype=np.intp)

    def _read_lidar_ranges(self):
        if self.lidar_mode == "combined":
            sensor_id = self.lidar_sensor_id
            adr = self.model.sensor_adr[sensor_id]
            dim = self.model.sensor_dim[sensor_id]
            return self.data.sensordata[adr : adr + dim].tolist()
        return self.data.sensordata[self._lidar_adrs].tolist()

    def _apply_forklift_vel(self,msg:Twist,joint_id):
        if joint_id==-1:return

        qvel_adr=self.model.jnt_dofadr[joint_id]
        qpos_adr=self.model.jnt_qposadr[joint_id]

        v=np.clip(msg.linear.x,-0.5,0.5)
        w=np.clip(msg.angular.z,-1.5,1.5)

        quat=self.data.qpos[qpos_adr+3:qpos_adr+7]
        qw,qx,qy,qz=quat

        siny_cosp=2.0*(qw*qz+qx*qy)
        cosy_cosp=1.0-2.0*(qy*qy+qz*qz)
        yaw=math.atan2(siny_cosp,cosy_cosp)

        vx_world=v*math.cos(yaw)
        vy_world=v*math.sin(yaw)

        self.data.qvel[qvel_adr]=vx_world
        self.data.qvel[qvel_adr+1]=vy_world
        self.data.qvel[qvel_adr+2]=0.0

        self.data.qvel[qvel_adr+3]=0.0
        self.data.qvel[qvel_adr+4]=0.0
        self.data.qvel[qvel_adr+5]=w

    def fl1_cmd_vel_callback(self, msg: Twist):
        with self.physics_lock:
            self._apply_forklift_vel(msg, self.fl1_joint_id)

    def fl2_cmd_vel_callback(self, msg: Twist):
        with self.physics_lock:
            self._apply_forklift_vel(msg, self.fl2_joint_id)

    def cmd_vel_callback(self, msg: Twist):
        with self.physics_lock:
            v = np.clip(msg.linear.x, -self.max_linear_vel, self.max_linear_vel)
            w = np.clip(msg.angular.z, -self.max_angular_vel, self.max_angular_vel)
            v_left = v - (w * self.track_width / 2.0)
            v_right = v + (w * self.track_width / 2.0)
            self.data.ctrl[0] = v_left / self.wheel_radius
            self.data.ctrl[1] = v_right / self.wheel_radius

    def publish_forklift_pose(self, stamp, joint_id, publisher, frame_id='odom'):
        if joint_id == -1: return
        qpos_adr = self.model.jnt_qposadr[joint_id]
        pos = self.data.qpos[qpos_adr:qpos_adr+3]
        quat = self.data.qpos[qpos_adr+3:qpos_adr+7]
        
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        publisher.publish(msg)

    def publish_odom(self, stamp):
        try:
            qpos_adr = self._qpos_adr
            qvel_adr = self._qvel_adr

            pos_world = self.data.qpos[qpos_adr: qpos_adr + 3]
            quat_world = self.data.qpos[qpos_adr + 3: qpos_adr + 7]
            vel = self.data.qvel[qvel_adr : qvel_adr + 3]
            ang_vel = self.data.qvel[qvel_adr + 3 : qvel_adr + 6]

            mujoco.mju_negQuat(self._buf_quat_inv, quat_world)
            mujoco.mju_rotVecQuat(self._buf_vel_base, vel, self._buf_quat_inv)
            mujoco.mju_rotVecQuat(self._buf_ang_vel_base, ang_vel, self._buf_quat_inv)

            diff = pos_world - self.origin_pos
            mujoco.mju_rotVecQuat(self._buf_pos_odom, diff, self.origin_quat_inv)
            mujoco.mju_mulQuat(self._buf_quat_odom, self.origin_quat_inv, quat_world)

            msg = Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = 'odom'
            msg.child_frame_id = 'base_footprint'

            msg.pose.pose.position.x = float(self._buf_pos_odom[0])
            msg.pose.pose.position.y = float(self._buf_pos_odom[1])
            msg.pose.pose.position.z = float(self._buf_pos_odom[2])
            msg.pose.pose.orientation.w = float(self._buf_quat_odom[0])
            msg.pose.pose.orientation.x = float(self._buf_quat_odom[1])
            msg.pose.pose.orientation.y = float(self._buf_quat_odom[2])
            msg.pose.pose.orientation.z = float(self._buf_quat_odom[3])

            msg.twist.twist.linear.x = float(self._buf_vel_base[0])
            msg.twist.twist.linear.y = float(self._buf_vel_base[1])
            msg.twist.twist.linear.z = float(self._buf_vel_base[2])
            msg.twist.twist.angular.x = float(self._buf_ang_vel_base[0])
            msg.twist.twist.angular.y = float(self._buf_ang_vel_base[1])
            msg.twist.twist.angular.z = float(self._buf_ang_vel_base[2])

            self.odom_pub.publish(msg)
            self._publish_tf(stamp, self._buf_pos_odom, self._buf_quat_odom)
        except Exception as e:
            self.get_logger().error(f"Odom publish error: {e}", once=True)

    def _publish_tf(self, stamp, pos, quat):
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = stamp
        odom_to_base.header.frame_id = 'odom'
        odom_to_base.child_frame_id = 'base_footprint'
        odom_to_base.transform.translation.x = float(pos[0])
        odom_to_base.transform.translation.y = float(pos[1])
        odom_to_base.transform.translation.z = float(pos[2])
        odom_to_base.transform.rotation.w = float(quat[0])
        odom_to_base.transform.rotation.x = float(quat[1])
        odom_to_base.transform.rotation.y = float(quat[2])
        odom_to_base.transform.rotation.z = float(quat[3])
        self.tf_broadcaster.sendTransform(odom_to_base)

    def publish_scan(self, stamp):
        try:
            sensor_data = self._read_lidar_ranges()
            if not sensor_data: return
            beam_count = len(sensor_data)

            msg = LaserScan()
            msg.header.stamp = stamp
            msg.header.frame_id = "lidar_link"
            msg.angle_min = 0.0
            msg.angle_increment = math.radians(360.0 / beam_count)
            msg.angle_max = msg.angle_min + msg.angle_increment * (beam_count-1)

            if self.last_scan_publish_time is None:
                msg.scan_time = 0.1
            else:
                msg.scan_time = self.sim_time - self.last_scan_publish_time
            self.last_scan_publish_time = self.sim_time

            msg.time_increment = msg.scan_time / beam_count
            msg.range_min = 0.12
            msg.range_max = 3.5

            arr = np.array(sensor_data, dtype=np.float32)
            invalid = (arr >= msg.range_max) | (arr < msg.range_min)
            arr[invalid] = np.inf
            msg.ranges = arr.tolist()
            self.scan_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Scan publish error: {e}", once=True)

    def _init_camera_params(self):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "patrol_camera")
        fovy = self.model.cam_fovy[cam_id] if cam_id != -1 else 45.0
        fovy_rad = math.radians(fovy)
        fy = (self.img_height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0

        self._cam_K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self._cam_R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self._cam_P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    def _get_camera_info(self, stamp) -> CameraInfo:
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "camera_link"
        info.height = self.img_height
        info.width = self.img_width
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = self._cam_K
        info.r = self._cam_R
        info.p = self._cam_P
        return info

    def get_sim_stamp(self):
        stamp = Time()
        stamp.sec = int(self.sim_time)
        stamp.nanosec = int((self.sim_time - int(self.sim_time)) * 1e9)
        return stamp


def camera_render_worker(node: MujocoRosBridge, is_running_flag: list):
    """별도 스레드에서 10Hz 주기로 Offscreen Rendering을 수행하는 워커"""
    renderer = mujoco.Renderer(node.model, node.img_height, node.img_width)
    render_option = mujoco.MjvOption()
    render_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    render_option.sitegroup[5] = 0

    while is_running_flag[0] and rclpy.ok():
        start_t = time.time()

        # 1. 시각적 씬 업데이트 (물리 데이터 복사이므로 짧게 Lock)
        with node.physics_lock:
            renderer.update_scene(node.data, camera="patrol_camera", scene_option=render_option)
            stamp = node.get_sim_stamp()

        # 2. GPU 기반 Offscreen Rendering (Lock 없이 수행 - 물리 루프 차단 안 함)
        rgb_pixels = renderer.render()
        renderer.enable_depth_rendering()
        depth_pixels = renderer.render()
        renderer.disable_depth_rendering()

        # 3. ROS2 토픽 발행
        rgb_msg = node.cv_bridge.cv2_to_imgmsg(rgb_pixels, encoding="rgb8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = "camera_link"
        node.camera_pub.publish(rgb_msg)

        depth_msg = node.cv_bridge.cv2_to_imgmsg(depth_pixels.astype(np.float32), encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "camera_link"
        node.depth_pub.publish(depth_msg)

        camera_info_msg = node._get_camera_info(stamp)
        node.camera_info_pub.publish(camera_info_msg)

        # 10Hz 주기를 맞추기 위한 대기 (0.1초)
        elapsed = time.time() - start_t
        remaining = MujocoRosBridge.CAMERA_INTERVAL - elapsed
        if remaining > 0:
            time.sleep(remaining)


def main():
    rclpy.init()

    BASE_DIR = Path(__file__).resolve().parent
    xml_path = BASE_DIR.parent / "scenes" / "patrol_20x20_factory.xml"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    node = MujocoRosBridge(model, data)

    # ROS 2 Spin 스레드
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 카메라 렌더링 전용 스레드
    is_running_flag = [True]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.distance = 28
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -90

        render_thread = threading.Thread(target=camera_render_worker, args=(node, is_running_flag), daemon=True)
        render_thread.start()

        sync_interval = 1.0 / MujocoRosBridge.VIEWER_HZ
        last_sync_time = time.time()

        while viewer.is_running() and rclpy.ok():
            step_start = time.time()

            # 물리 연산 실행 (Lock 적용)
            with node.physics_lock:
                mujoco.mj_step(model, data)
                node.sim_time = data.time
                stamp = node.get_sim_stamp()

            current_real_time = time.time()
            if (current_real_time - last_sync_time) >= sync_interval:
                viewer.sync()
                last_sync_time = current_real_time

            # 1. Odom (50Hz)
            if node.sim_time >= node.last_odom_time + MujocoRosBridge.ODOM_INTERVAL:
                with node.physics_lock:
                    node.publish_odom(stamp)
                    node.publish_forklift_pose(stamp, node.fl1_joint_id, node.fl1_pose_pub, 'odom')
                    node.publish_forklift_pose(stamp, node.fl2_joint_id, node.fl2_pose_pub, 'odom')
                    node.last_odom_time = node.sim_time

                clock_msg = Clock()
                clock_msg.clock = stamp
                node.clock_pub.publish(clock_msg)

            # 2. Scan (10Hz)
            if node.sim_time >= node.last_scan_time + MujocoRosBridge.SCAN_INTERVAL:
                with node.physics_lock:
                    node.publish_scan(stamp)
                    node.last_scan_time = node.sim_time

            # 시뮬레이션 타임스텝 연산 지연 보장
            deadline = step_start + model.opt.timestep
            remaining = deadline - time.time()
            if remaining > 0:
                time.sleep(remaining)

    # 종수 처리
    is_running_flag[0] = False
    render_thread.join(timeout=1.0)
    node.destroy_node()
    rclpy.shutdown()
    spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()