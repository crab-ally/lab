# MuJoCo를 ROS2 로봇으로 만들어주는 ROS2 Bridge

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from tf2_ros import TransformBroadcaster
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

class MujocoRosBridge(Node):
    def __init__(self, model, data):
        super().__init__('mujoco_ros_bridge')

        self.set_parameters([
            Parameter(
                'use_sim_time',
                Parameter.Type.BOOL,
                True
            )
        ])
        self.model = model
        self.data = data
        self.cv_bridge = cv_bridge.CvBridge()

        # Renderer는 Viewer 생성 이후 Main Thread에서 초기화
        self.renderer = None

        # 로봇 제원 [m]
        self.track_width = 0.160
        self.wheel_radius = 0.033

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.clock_pub = self.create_publisher(
            Clock,
            '/clock',
            10
        )

        self.camera_image = None
        self.camera_lock = threading.Lock()

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.camera_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # base_link → lidar_link (MuJoCo 모델 좌표 기준)
        self.lidar_offset = (0.07, 0.0, 0.162)
        self.lidar_quat = (0.707, 0.0, 0.707, 0.0)

        self.lidar_beam_count = 360
        self._init_lidar_sensors()

        # =====================================================================
        # [수정] odom 원점 캡처 (root freejoint)
        # -----------------------------------------------------------------
        # 문제: 기존 코드는 MuJoCo 월드 절대좌표(qpos)를 그대로 /odom으로
        #       publish했음. 로봇이 월드 (-4,-4)에서 시작하면 odom도 거기서
        #       시작 → SLAM Toolbox가 map→odom을 초기화할 때 base_link가
        #       odom 원점(0,0,0)에 있다고 가정하므로, map 좌표계가 MuJoCo
        #       월드 좌표계와 크게 어긋나는 결과가 생김.
        #
        # 해결: 노드 생성 시점(=시뮬레이션 초기 상태)의 위치/자세를 저장해두고,
        #       매 publish_odom() 호출마다 "시작점 대비 상대 이동량"을
        #       시작 자세의 역회전으로 변환해 odom으로 내보냄.
        #       → odom 프레임 원점 = 로봇이 시작한 지점 (REP-105 관례와 일치)
        # =====================================================================
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if joint_id == -1:
            raise RuntimeError("root freejoint를 모델에서 찾을 수 없습니다")

        self._qpos_adr = self.model.jnt_qposadr[joint_id]
        self._qvel_adr = self.model.jnt_dofadr[joint_id]

        # MjData는 생성 시점에 qpos0(모델 초기값)로 채워져 있으므로,
        # 여기서 읽는 값이 곧 "시뮬레이션 시작 위치/자세"가 됨.
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
            f"world quat={self.origin_quat.tolist()} "
            f"(odom(0,0,0)이 이 지점에 대응합니다)"
        )
        # =====================================================================

        # Timers (Publishing Loop)
        self.odom_timer = self.create_timer(0.05, self.publish_odom)  # 20Hz
        self.scan_timer = self.create_timer(0.1, self.publish_scan)   # 10Hz
        #self.camera_timer = self.create_timer(0.1, self.publish_camera) # 10Hz

        self.get_logger().info(
            f"MuJoCo-ROS2 Bridge Node initialized "
            f"(LiDAR beams: {self.lidar_beam_count}, mode: {self.lidar_mode})"
        )

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

    # ROS2 /cmd_vel → MuJoCo 좌/우 바퀴 회전 속도
    def cmd_vel_callback(self, msg: Twist):
        v = msg.linear.x  # 앞/뒤 이동
        w = msg.angular.z # z축 기준 회전

        # 역운동학 계산 (Differential Drive)
        v_left = v - (w * self.track_width / 2.0)
        v_right = v + (w * self.track_width / 2.0)

        # 모터 제어 인가 (rad/s)
        self.data.ctrl[0] = v_left / self.wheel_radius
        self.data.ctrl[1] = v_right / self.wheel_radius

    # MuJoCo 로봇 상태(qpos, qvel) → ROS2 Odometry
    def publish_odom(self):
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

            # ==========================================================
            # [추가]
            # MuJoCo freejoint의 qvel은 World Frame 기준 속도이다.
            #
            # 하지만 ROS Odometry(Twist)는 child_frame(base_footprint)
            # 기준 속도를 publish해야 한다.
            #
            # 따라서 현재 자세(quaternion)의 역회전을 이용하여
            # World Velocity → Base Velocity 로 변환한다.
            # ==========================================================

            quat_world_inv = np.zeros(4)
            mujoco.mju_negQuat(quat_world_inv, quat_world)

            vel_base = np.zeros(3)
            mujoco.mju_rotVecQuat(
                vel_base,
                vel,
                quat_world_inv
            )

            ang_vel_base = np.zeros(3)
            mujoco.mju_rotVecQuat(
                ang_vel_base,
                ang_vel,
                quat_world_inv
            )

            # -----------------------------------------------------------
            # [수정] 월드 절대좌표 → odom 상대좌표 변환
            #
            # 위치: (현재 위치 - 시작 위치)를 "시작 자세" 기준으로 회전.
            #       시작 시점에 로봇이 90도 돌아가 있었다면(quat="0.707 0 0 0.707"
            #       같은 경우), 그 회전까지 같이 걷어내야 odom의 +X가
            #       "로봇이 원래 보던 방향"이 됨.
            # 자세: origin_quat의 역 * 현재 quat = 시작 자세 대비 상대 회전.
            # -----------------------------------------------------------
            diff = pos_world - self.origin_pos
            pos_odom = np.zeros(3)
            mujoco.mju_rotVecQuat(pos_odom, diff, self.origin_quat_inv)

            quat_odom = np.zeros(4)
            mujoco.mju_mulQuat(quat_odom, self.origin_quat_inv, quat_world)

            msg = Odometry()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'odom'
            msg.child_frame_id = 'base_footprint'

            msg.pose.pose.position.x = float(pos_odom[0])
            msg.pose.pose.position.y = float(pos_odom[1])
            msg.pose.pose.position.z = float(pos_odom[2])

            msg.pose.pose.orientation.w = float(quat_odom[0])
            msg.pose.pose.orientation.x = float(quat_odom[1])
            msg.pose.pose.orientation.y = float(quat_odom[2])
            msg.pose.pose.orientation.z = float(quat_odom[3])

            # ==========================================================
            # ROS REP-105
            #
            # Twist는 child_frame(base_footprint) 기준 속도이다.
            #
            # World 기준 속도를 Base 기준으로 변환한 값을 publish.
            # ==========================================================

            msg.twist.twist.linear.x = float(vel_base[0])
            msg.twist.twist.linear.y = float(vel_base[1])
            msg.twist.twist.linear.z = float(vel_base[2])

            msg.twist.twist.angular.x = float(ang_vel_base[0])
            msg.twist.twist.angular.y = float(ang_vel_base[1])
            msg.twist.twist.angular.z = float(ang_vel_base[2])

            self.odom_pub.publish(msg)
            self._publish_tf(msg.header.stamp, pos_odom, quat_odom)
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

        # lidar TF는 robot_state_publisher가 담당

        # odom -> base_footprint 만 publish
        # base_footprint -> base_link -> lidar_link 는 robot_state_publisher 담당
        self.tf_broadcaster.sendTransform(odom_to_base)

    # MuJoCo LiDAR (36빔) → ROS2 LaserScan
    def publish_scan(self):
        try:
            sensor_data = self._read_lidar_ranges()
            if not sensor_data:
                return

            beam_count = len(sensor_data)

            msg = LaserScan()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "lidar_link"
            msg.angle_min = -math.pi
            msg.angle_increment = math.radians(360.0 / beam_count)
            msg.angle_max = msg.angle_min + msg.angle_increment * (beam_count-1)
            msg.scan_time = 0.1
            msg.time_increment = msg.scan_time / beam_count
            msg.range_min = 0.12
            msg.range_max = 3.5
            msg.ranges = [float(r) if r > 0 else float('inf') for r in sensor_data]

            self.scan_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Scan publish error: {e}",once=True)

    def publish_camera(self):

        with self.camera_lock:

            if self.camera_image is None:
                return

            pixels = self.camera_image.copy()


        img_msg = self.cv_bridge.cv2_to_imgmsg(
            pixels,
            encoding="rgb8"
        )

        img_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        img_msg.header.frame_id = "camera_link"

        self.camera_pub.publish(img_msg)


def ros_spin_thread(node):
    rclpy.spin(node)


def main():
    rclpy.init()

    BASE_DIR = Path(__file__).resolve().parent
    xml_path = BASE_DIR.parent / "scenes" / "patrol_test_LiDAR.xml"

    print(f"Loading model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    node = None

    with mujoco.viewer.launch_passive(model, data) as viewer:

        print("터틀봇 실내 순찰 시뮬레이션 및 ROS2 브릿지 시작...")

        node = MujocoRosBridge(model, data)

        # ROS2 spin 시작
        spin_thread = threading.Thread(
            target=ros_spin_thread,
            args=(node,),
            daemon=True
        )

        spin_thread.start()

        viewer.cam.lookat[:] = [0, 0, 0.9]
        viewer.cam.distance = 15.8
        viewer.cam.azimuth = 85
        viewer.cam.elevation = -85

        while viewer.is_running() and rclpy.ok():
            step_start = time.time()

            # MuJoCo simulation time → ROS2 clock
            clock_msg = Clock()

            sim_time = data.time

            clock_msg.clock.sec = int(sim_time)
            clock_msg.clock.nanosec = int(
                (sim_time - int(sim_time)) * 1e9
            )

            node.clock_pub.publish(clock_msg)

            mujoco.mj_step(model, data)
            viewer.sync()

            if node.renderer is not None:

                node.renderer.update_scene(
                    data,
                    camera="patrol_camera"
                )

                pixels = node.renderer.render()

                with node.camera_lock:
                    node.camera_image = pixels

            time_until_next_step = (
                model.opt.timestep -
                (time.time() - step_start)
            )

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        print("시뮬레이션 종료 중...")
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()