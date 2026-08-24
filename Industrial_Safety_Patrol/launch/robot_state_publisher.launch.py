from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    with open("/workspace/urdf/turtlebot_patrol.urdf","r") as urdf_file:
        robot_description=urdf_file.read()

    return LaunchDescription([
        LogInfo(msg="[Robot] Starting Robot State Publisher..."),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            arguments=["--ros-args","--log-level","warn"],
            parameters=[{
                "robot_description":robot_description,
                "use_sim_time":True
            }]
        )
    ])