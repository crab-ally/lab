#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped

class ForkliftController(Node):
    def __init__(self):
        super().__init__('forklift_controller')

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

        self.forklift_1_waypoints=[
            (-3.0,8.0),
            (-3.0,3.0),
            (-8.0,3.0),
            (-8.0,8.0),
        ]

        self.forklift_2_waypoints=[
            (3.0,3.0),
            (8.0,3.0),
            (8.0,8.0),
            (3.0,8.0),
        ]

        self.fl1_waypoint_index=0
        self.fl2_waypoint_index=0

        self.fl1_pose=None
        self.fl2_pose=None

        self.create_subscription(
            PoseStamped,
            '/forklift_1/pose',
            self.forklift_1_pose_callback,
            10
        )

        self.create_subscription(
            PoseStamped,
            '/forklift_2/pose',
            self.forklift_2_pose_callback,
            10
        )

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

        self.timer=self.create_timer(
            1.0/self.publish_rate,
            self.control_loop
        )

        self.get_logger().info('Forklift Controller is ready.')

    def forklift_1_pose_callback(self,msg):
        self.fl1_pose=msg.pose

    def forklift_2_pose_callback(self,msg):
        self.fl2_pose=msg.pose

    def get_yaw(self,orientation):
        siny_cosp=2.0*(orientation.w*orientation.z+orientation.x*orientation.y)
        cosy_cosp=1.0-2.0*(orientation.y**2+orientation.z**2)
        return math.atan2(siny_cosp,cosy_cosp)

    def normalize_angle(self,angle):
        return math.atan2(math.sin(angle),math.cos(angle))

    def control_loop(self):
        if self.fl1_pose is not None:
            self.control_forklift(
                self.fl1_pose,
                self.forklift_1_waypoints,
                'Forklift 1',
                self.fl1_waypoint_index,
                self.fl1_pub,
                1
            )

        if self.fl2_pose is not None:
            self.control_forklift(
                self.fl2_pose,
                self.forklift_2_waypoints,
                'Forklift 2',
                self.fl2_waypoint_index,
                self.fl2_pub,
                2
            )

    def control_forklift(self,pose,waypoints,name,index,pub,forklift_id):
        if index>=len(waypoints):
            pub.publish(Twist())
            return

        target_x,target_y=waypoints[index]

        current_x=pose.position.x
        current_y=pose.position.y
        current_yaw=self.get_yaw(pose.orientation)

        dx=target_x-current_x
        dy=target_y-current_y
        distance=math.hypot(dx,dy)

        if distance<self.waypoint_tolerance:
            self.get_logger().info(
                f'{name} reached waypoint {index}: '
                f'({target_x:.1f},{target_y:.1f})'
            )

            if forklift_id==1:
                self.fl1_waypoint_index+=1
            else:
                self.fl2_waypoint_index+=1

            pub.publish(Twist())
            return

        target_yaw=math.atan2(dy,dx)
        yaw_error=self.normalize_angle(target_yaw-current_yaw)

        cmd=Twist()

        if abs(yaw_error)>self.yaw_tolerance:
            cmd.linear.x=0.0
            cmd.angular.z=self.angular_speed if yaw_error>0.0 else -self.angular_speed
        else:
            cmd.linear.x=self.linear_speed
            cmd.angular.z=0.0

        pub.publish(cmd)

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