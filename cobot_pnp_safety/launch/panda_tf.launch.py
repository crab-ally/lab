from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    urdf = LaunchConfiguration("urdf")

    return LaunchDescription([
        DeclareLaunchArgument(
            "urdf",
            default_value="/workspace/urdf/panda.urdf"
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[
                {
                    "robot_description": Command(["cat ", urdf]),
                    "use_sim_time": True
                }
            ]
        )
    ])