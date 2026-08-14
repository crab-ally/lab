#!/usr/bin/env python3
"""
timestep: 0.005
odom → base_footprint: 50Hz(0.02s)
/scan: 10Hz(0.1s)
/camera/image_raw: 10Hz(0.1s)
/camera/depth/image_raw: 10Hz(0.1s)
/camera/depth/camera_info: 10Hz(0.1s)
/clock: 50Hz
viewer.sync(): 60Hz
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
import cv2
import cv_bridge
import math
from pathlib import Path
from rosgraph_msgs.msg import Clock
from rclpy.parameter import Parameter
from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import DurabilityPolicy

class MujocoRosBridge(Node):
    def __init__(self, model, data):
        super().__init__(
            'mujoco_ros_bridge',
            parameter_overrides=[
                Parameter(
                    'use_sim_time',
                    Parameter.Type.BOOL,
                    True
                )
            ]
        )

        self.max_linear_vel = 0.25
        self.max_angular_vel = 1.0
        self.last_odom_time = 0.0
        self.sim_time = 0.0
        self.last_scan_time = 0.0
        self.last_scan_publish_time = None
        self.model = model
        self.data = data
        self.cv_bridge = cv_bridge.CvBridge()

        # Renderer는 Viewer 생성 이후 Main Thread에서 초기화
        self.renderer = None

        # 로봇 제원 [m]
        self.track_width = 0.160
        self.wheel_radius = 0.033

        # 이미지 해상도
        self.img_width = 640
        self.img_height = 480

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.fl1_cmd_vel_sub = self.create_subscription(Twist, '/forklift_1/cmd_vel', self.fl1_cmd_vel_callback, 10)
        self.fl2_cmd_vel_sub = self.create_subscription(Twist, '/forklift_2/cmd_vel', self.fl2_cmd_vel_callback, 10)
        
        # Get freejoint IDs for forklifts
        self.fl1_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "fl1_freejoint")
        self.fl2_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "fl2_freejoint")

        clock_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.clock_pub = self.create_publisher(
            Clock,
            '/clock',
            clock_qos
        )

        self.camera_image = None
        self.depth_image = None
        self.camera_lock = threading.Lock()

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.camera_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # Forklift Pose Publishers
        self.fl1_pose_pub = self.create_publisher(PoseStamped, '/forklift_1/pose', 10)
        self.fl2_pose_pub = self.create_publisher(PoseStamped, '/forklift_2/pose', 10)
        
        # Depth 이미지 및 CameraInfo Publisher
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 10)
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.lidar_beam_count = 360
        self._init_lidar_sensors()

        # =====================================================================
        # odom 원점 캡처 (root freejoint)
        # =====================================================================
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if joint_id == -1:
            raise RuntimeError("root freejoint를 모델에서 찾을 수 없습니다")

        self._qpos_adr = self.model.jnt_qposadr[joint_id]
        self._qvel_adr = self.model.jnt_dofadr[joint_id]

        self.origin_pos = np.array(
            self.data.qpos[self._qpos_adr: self._qpos_adr + 3], dtype=np.float64
        ).copy()
        self.origin_quat = np.array(
            self.data.qpos[self._qpos_adr + 3: self._qpos_adr + 7], dtype=np.float64
        ).copy()

        self.origin_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(self.origin_quat_inv, self.origin_quat)

        self.get_logger().info(
            f"Odom origin captured — world pos={self.origin_pos.tolist()}, "
            f"world quat={self.origin_quat.tolist()}"
        )

        self._publish_static_transforms()

        self.get_logger().info(
            f"MuJoCo-ROS2 Bridge Node initialized "
            f"(LiDAR beams: {self.lidar_beam_count}, mode: {self.lidar_mode})"
        )

    def _publish_static_transforms(self):
        now = self.get_clock().now().to_msg()
        transforms = []

        # base_footprint → lidar_link
        lidar_tf = TransformStamped()
        lidar_tf.header.stamp = now
        lidar_tf.header.frame_id = 'base_footprint'
        lidar_tf.child_frame_id = 'lidar_link'
        lidar_tf.transform.translation.x = 0.0
        lidar_tf.transform.translation.y = 0.0
        lidar_tf.transform.translation.z = 0.1675
        lidar_tf.transform.rotation.w = 1.0
        lidar_tf.transform.rotation.x = 0.0
        lidar_tf.transform.rotation.y = 0.0
        lidar_tf.transform.rotation.z = 0.0
        transforms.append(lidar_tf)

        # base_footprint → camera_link
        camera_tf = TransformStamped()
        camera_tf.header.stamp = now
        camera_tf.header.frame_id = 'base_footprint'
        camera_tf.child_frame_id = 'camera_link'
        camera_tf.transform.translation.x = -0.05
        camera_tf.transform.translation.y = 0.0
        camera_tf.transform.translation.z = 0.35
        camera_tf.transform.rotation.x = 0.5
        camera_tf.transform.rotation.y = -0.5
        camera_tf.transform.rotation.z = -0.5
        camera_tf.transform.rotation.w = 0.5
        transforms.append(camera_tf)

        self.static_tf_broadcaster.sendTransform(transforms)
        self.get_logger().info('Static TFs published: base_footprint → camera_link, lidar_link')

    def _init_lidar_sensors(self):
        combined_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "lidar"
        )
        if combined_id != -1 and self.model.sensor_dim[combined_id] >= self.lidar_beam_count:
            self.lidar_mode = "combined"
            self.lidar_sensor_id = combined_id
            return

        replicated = []
        for sensor_id in range(self.model.nsensor):
            if self.model.sensor_type[sensor_id] != mujoco.mjtSensor.mjSENS_RANGEFINDER:
                continue
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id
            )
            if name and (name == "lidar" or name.startswith("lidar-")):
                replicated.append((name, sensor_id))
        replicated.sort(key=lambda item: item[0])
        if len(replicated) < self.lidar_beam_count:
            self.get_logger().warn(
                f"Expected {self.lidar_beam_count} LiDAR beams, found {len(replicated)}"
            )
        self.lidar_mode = "replicated"
        self.lidar_sensor_ids = replicated[: self.lidar_beam_count]

    def _read_lidar_ranges(self):
        if self.lidar_mode == "combined":
            sensor_id = self.lidar_sensor_id
            adr = self.model.sensor_adr[sensor_id]
            dim = self.model.sensor_dim[sensor_id]
            return list(self.data.sensordata[adr : adr + dim])

        ranges = []
        for _, sensor_id in self.lidar_sensor_ids:
            adr = self.model.sensor_adr[sensor_id]
            ranges.append(float(self.data.sensordata[adr]))
        return ranges

    def _apply_forklift_vel(self, msg: Twist, joint_id):
        if joint_id == -1: return
        
        qvel_adr = self.model.jnt_dofadr[joint_id]
        qpos_adr = self.model.jnt_qposadr[joint_id]
        
        self.data.qvel[qvel_adr] = msg.linear.x
        self.data.qvel[qvel_adr+1] = msg.linear.y
        
        if abs(msg.linear.x) > 0.01 or abs(msg.linear.y) > 0.01:
            yaw = math.atan2(msg.linear.y, msg.linear.x)
            self.data.qpos[qpos_adr+3] = math.cos(yaw/2.0)
            self.data.qpos[qpos_adr+4] = 0.0
            self.data.qpos[qpos_adr+5] = 0.0
            self.data.qpos[qpos_adr+6] = math.sin(yaw/2.0)

    def fl1_cmd_vel_callback(self, msg: Twist):
        self._apply_forklift_vel(msg, self.fl1_joint_id)

    def fl2_cmd_vel_callback(self, msg: Twist):
        self._apply_forklift_vel(msg, self.fl2_joint_id)

    def publish_forklift_pose(self, stamp, joint_id, publisher, frame_id):
        if joint_id == -1: return
        qpos_adr = self.model.jnt_qposadr[joint_id]
        pos = self.data.qpos[qpos_adr:qpos_adr+3]
        quat = self.data.qpos[qpos_adr+3:qpos_adr+7]
        
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        publisher.publish(msg)

    # /cmd_vel_nav 콜백 함수
    def cmd_vel_callback(self, msg: Twist):
        v = np.clip(msg.linear.x, -self.max_linear_vel, self.max_linear_vel)
        w = np.clip(msg.angular.z, -self.max_angular_vel, self.max_angular_vel)

        v_left = v - (w * self.track_width / 2.0)
        v_right = v + (w * self.track_width / 2.0)

        self.data.ctrl[0] = v_left / self.wheel_radius
        self.data.ctrl[1] = v_right / self.wheel_radius

    def publish_odom(self, stamp):
        try:
            qpos_adr = self._qpos_adr
            qvel_adr = self._qvel_adr

            pos_world = np.array(
                self.data.qpos[qpos_adr: qpos_adr + 3], dtype=np.float64
            )
            quat_world = np.array(
                self.data.qpos[qpos_adr + 3: qpos_adr + 7], dtype=np.float64
            )
            vel = self.data.qvel[qvel_adr : qvel_adr + 3]
            ang_vel = self.data.qvel[qvel_adr + 3 : qvel_adr + 6]

            quat_world_inv = np.zeros(4)
            mujoco.mju_negQuat(quat_world_inv, quat_world)

            vel_base = np.zeros(3)
            mujoco.mju_rotVecQuat(vel_base, vel, quat_world_inv)

            ang_vel_base = np.zeros(3)
            mujoco.mju_rotVecQuat(ang_vel_base, ang_vel, quat_world_inv)

            diff = pos_world - self.origin_pos
            pos_odom = np.zeros(3)
            mujoco.mju_rotVecQuat(pos_odom, diff, self.origin_quat_inv)

            quat_odom = np.zeros(4)
            mujoco.mju_mulQuat(quat_odom, self.origin_quat_inv, quat_world)

            msg = Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = 'odom'
            msg.child_frame_id = 'base_footprint'

            msg.pose.pose.position.x = float(pos_odom[0])
            msg.pose.pose.position.y = float(pos_odom[1])
            msg.pose.pose.position.z = float(pos_odom[2])

            msg.pose.pose.orientation.w = float(quat_odom[0])
            msg.pose.pose.orientation.x = float(quat_odom[1])
            msg.pose.pose.orientation.y = float(quat_odom[2])
            msg.pose.pose.orientation.z = float(quat_odom[3])

            msg.twist.twist.linear.x = float(vel_base[0])
            msg.twist.twist.linear.y = float(vel_base[1])
            msg.twist.twist.linear.z = float(vel_base[2])

            msg.twist.twist.angular.x = float(ang_vel_base[0])
            msg.twist.twist.angular.y = float(ang_vel_base[1])
            msg.twist.twist.angular.z = float(ang_vel_base[2])

            self.odom_pub.publish(msg)
            self._publish_tf(stamp, pos_odom, quat_odom)
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
            if not sensor_data:
                return

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

            clean_ranges = []
            for r in sensor_data:
                val = float(r)
                if val >= msg.range_max - 0.05 or val < msg.range_min:
                    clean_ranges.append(float("inf"))
                else:
                    clean_ranges.append(val)

            msg.ranges = clean_ranges
            self.scan_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Scan publish error: {e}", once=True)

    def _get_camera_info(self, stamp) -> CameraInfo:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "patrol_camera")
        fovy = self.model.cam_fovy[cam_id] if cam_id != -1 else 45.0

        fovy_rad = math.radians(fovy)
        fy = (self.img_height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "camera_link"
        info.height = self.img_height
        info.width = self.img_width
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        info.k = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
                  
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
                  
        info.p = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]

        return info

    def publish_camera(self, stamp):
        with self.camera_lock:
            if self.camera_image is None or self.depth_image is None:
                return

            rgb_pixels = self.camera_image.copy()
            depth_pixels = self.depth_image.copy()

        rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb_pixels, encoding="rgb8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = "camera_link"
        self.camera_pub.publish(rgb_msg)

        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_pixels.astype(np.float32), encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "camera_link"
        self.depth_pub.publish(depth_msg)

        camera_info_msg = self._get_camera_info(stamp)
        self.camera_info_pub.publish(camera_info_msg)

    def get_sim_stamp(self):
        stamp = Time()
        stamp.sec = int(self.sim_time)
        stamp.nanosec = int(
            (self.sim_time - int(self.sim_time)) * 1e9
        )
        return stamp


def ros_spin_thread(node):
    rclpy.spin(node)

def main():
    rclpy.init()

    BASE_DIR = Path(__file__).resolve().parent
    xml_path = BASE_DIR.parent / "scenes" / "patrol_20x20_factory.xml"

    print(f"Loading model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    node = None
    last_camera_time = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

        print("터틀봇 실내 순찰 시뮬레이션 및 ROS2 브릿지 시작...")

        node = MujocoRosBridge(model, data)
        node.renderer = mujoco.Renderer(model, node.img_height, node.img_width)

        node.render_option = mujoco.MjvOption()
        node.render_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        node.render_option.sitegroup[5] = 0

        spin_thread = threading.Thread(
            target=ros_spin_thread,
            args=(node,),
            daemon=True
        )
        spin_thread.start()

        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.distance = 28
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -90

        sync_interval = 1.0 / 60.0
        last_sync_time = time.time()

        while viewer.is_running() and rclpy.ok():

            step_start = time.time()

            mujoco.mj_step(model, data)

            current_real_time = time.time()
            if (current_real_time - last_sync_time) >= sync_interval:
                viewer.sync()
                last_sync_time = current_real_time

            node.sim_time = data.time
            stamp = node.get_sim_stamp()

            # 1. Odom (50Hz / 0.02초 간격)
            if data.time >= node.last_odom_time + 0.02:
                node.publish_odom(stamp)
                node.publish_forklift_pose(stamp, node.fl1_joint_id, node.fl1_pose_pub, 'forklift_1')
                node.publish_forklift_pose(stamp, node.fl2_joint_id, node.fl2_pose_pub, 'forklift_2')
                node.last_odom_time = data.time

                clock_msg = Clock()
                clock_msg.clock = stamp
                node.clock_pub.publish(clock_msg)

            # 2. Scan (10Hz / 0.1초 간격)
            if data.time >= node.last_scan_time + 0.1:
                node.publish_scan(stamp)
                node.last_scan_time = node.sim_time

            # 3. Camera RGB + Depth + CameraInfo (10Hz / 0.1초 간격)
            if data.time >= last_camera_time + 0.1:
                if node.renderer is not None:
                    node.renderer.update_scene(data, camera="patrol_camera", scene_option=node.render_option)
                    
                    rgb_pixels = node.renderer.render()
                    
                    node.renderer.enable_depth_rendering()
                    depth_pixels = node.renderer.render()
                    node.renderer.disable_depth_rendering()

                    with node.camera_lock:
                        node.camera_image = rgb_pixels
                        node.depth_image = depth_pixels

                    node.publish_camera(stamp)

                last_camera_time = data.time

            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        print("시뮬레이션 종료 중...")
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()