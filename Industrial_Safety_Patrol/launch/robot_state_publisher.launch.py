from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    with open(
        "/workspace/urdf/turtlebot_patrol.urdf",
        "r"
    ) as urdf_file:
        robot_description = urdf_file.read()

    return LaunchDescription([

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[
                {
                    "robot_description": robot_description,
                    "use_sim_time": True
                }
            ]
        )

    ])