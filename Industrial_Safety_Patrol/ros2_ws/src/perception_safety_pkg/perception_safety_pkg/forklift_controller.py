#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped


class ForkliftController(Node):
    def __init__(self):
        super().__init__('forklift_controller')

        # =========================
        # Parameters
        # =========================
        self.declare_parameter('publish_rate',10.0)
        self.declare_parameter('waypoint_tolerance',0.3)
        self.declare_parameter('linear_speed',0.3)
        self.declare_parameter('angular_speed',1.0)
        self.declare_parameter('yaw_tolerance',0.15)

        self.publish_rate=self.get_parameter('publish_rate').value
        self.waypoint_tolerance=self.get_parameter('waypoint_tolerance').value
        self.linear_speed=self.get_parameter('linear_speed').value
        self.angular_speed=self.get_parameter('angular_speed').value
        self.yaw_tolerance=self.get_parameter('yaw_tolerance').value

        # =========================
        # Waypoints
        # =========================
        self.forklift_1_waypoints=[
            (-8.0,8.0),
            (3.0,8.0),
            (3.0,3.0),
            (-8.0,3.0),
        ]

        self.forklift_2_waypoints=[
            (3.0,8.0),
            (3.0,3.0),
            (8.0,3.0),
            (8.0,8.0),
        ]

        self.fl1_waypoint_index=0
        self.fl2_waypoint_index=0

        # =========================
        # Current Pose
        # =========================
        self.fl1_pose=None
        self.fl2_pose=None

        self.fl1_sub=self.create_subscription(
            PoseStamped,
            '/forklift_1/pose',
            self.forklift_1_pose_callback,
            10
        )

        self.fl2_sub=self.create_subscription(
            PoseStamped,
            '/forklift_2/pose',
            self.forklift_2_pose_callback,
            10
        )

        # =========================
        # Command Publishers
        # =========================
        self.fl1_pub=self.create_publisher(
            Twist,
            '/forklift_1/cmd_vel',
            10
        )

        self.fl2_pub=self.create_publisher(
            Twist,
            '/forklift_2/cmd_vel',
            10
        )

        # =========================
        # Timer
        # =========================
        self.timer=self.create_timer(
            1.0/self.publish_rate,
            self.control_loop
        )

        self.get_logger().info('Forklift Controller is ready.')

    # ==========================================================
    # Pose Callback
    # ==========================================================

    def forklift_1_pose_callback(self,msg):
        self.fl1_pose=msg.pose

    def forklift_2_pose_callback(self,msg):
        self.fl2_pose=msg.pose

    # ==========================================================
    # Quaternion -> Yaw
    # ==========================================================

    def get_yaw(self,orientation):
        siny_cosp=2.0*(orientation.w*orientation.z+orientation.x*orientation.y)
        cosy_cosp=1.0-2.0*(orientation.y**2+orientation.z**2)
        return math.atan2(siny_cosp,cosy_cosp)

    # ==========================================================
    # Normalize Angle
    # ==========================================================

    def normalize_angle(self,angle):
        return math.atan2(math.sin(angle),math.cos(angle))

    # ==========================================================
    # Control Loop
    # ==========================================================

    def control_loop(self):
        if self.fl1_pose is not None:
            self.control_forklift_1()

        if self.fl2_pose is not None:
            self.control_forklift_2()

    # ==========================================================
    # Forklift 1
    # ==========================================================

    def control_forklift_1(self):
        if self.fl1_waypoint_index>=len(self.forklift_1_waypoints):
            self.stop_forklift_1()
            return

        target_x,target_y=self.forklift_1_waypoints[self.fl1_waypoint_index]

        current_x=self.fl1_pose.position.x
        current_y=self.fl1_pose.position.y

        dx=target_x-current_x
        dy=target_y-current_y

        distance=math.hypot(dx,dy)

        # Waypoint 도착
        if distance<self.waypoint_tolerance:
            self.get_logger().info(
                f'Forklift 1 reached waypoint {self.fl1_waypoint_index}: '
                f'({target_x:.1f}, {target_y:.1f})'
            )

            self.fl1_waypoint_index+=1
            self.stop_forklift_1()
            return

        current_yaw=self.get_yaw(self.fl1_pose.orientation)
        target_yaw=math.atan2(dy,dx)
        yaw_error=self.normalize_angle(target_yaw-current_yaw)

        cmd=Twist()

        # 방향이 맞지 않으면 회전
        if abs(yaw_error)>self.yaw_tolerance:
            cmd.linear.x=0.0
            cmd.angular.z=self.angular_speed if yaw_error>0.0 else -self.angular_speed

        # 방향이 맞으면 전진
        else:
            cmd.linear.x=self.linear_speed
            cmd.angular.z=0.0

        self.fl1_pub.publish(cmd)

    # ==========================================================
    # Forklift 2
    # ==========================================================

    def control_forklift_2(self):
        if self.fl2_waypoint_index>=len(self.forklift_2_waypoints):
            self.stop_forklift_2()
            return

        target_x,target_y=self.forklift_2_waypoints[self.fl2_waypoint_index]

        current_x=self.fl2_pose.position.x
        current_y=self.fl2_pose.position.y

        dx=target_x-current_x
        dy=target_y-current_y

        distance=math.hypot(dx,dy)

        # Waypoint 도착
        if distance<self.waypoint_tolerance:
            self.get_logger().info(
                f'Forklift 2 reached waypoint {self.fl2_waypoint_index}: '
                f'({target_x:.1f}, {target_y:.1f})'
            )

            self.fl2_waypoint_index+=1
            self.stop_forklift_2()
            return

        current_yaw=self.get_yaw(self.fl2_pose.orientation)
        target_yaw=math.atan2(dy,dx)
        yaw_error=self.normalize_angle(target_yaw-current_yaw)

        cmd=Twist()

        # 방향이 맞지 않으면 회전
        if abs(yaw_error)>self.yaw_tolerance:
            cmd.linear.x=0.0
            cmd.angular.z=self.angular_speed if yaw_error>0.0 else -self.angular_speed

        # 방향이 맞으면 전진
        else:
            cmd.linear.x=self.linear_speed
            cmd.angular.z=0.0

        self.fl2_pub.publish(cmd)

    # ==========================================================
    # Stop
    # ==========================================================

    def stop_forklift_1(self):
        self.fl1_pub.publish(Twist())

    def stop_forklift_2(self):
        self.fl2_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)

    node=ForkliftController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_forklift_1()
        node.stop_forklift_2()
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()