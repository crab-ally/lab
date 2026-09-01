import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    workspace_dir = Path(__file__).resolve().parent.parent
    urdf_path = workspace_dir / "urdf" / "panda.urdf"

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    with open(urdf_path, 'r', encoding='utf-8') as f:
        robot_description_content = f.read()

    world_to_link0 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_link0',
        arguments=['0','0','0','0','0','0','world','link0'],
        output='screen'
    )

    # 오직 URDF 기반 로봇 매니퓰레이터 TF 발행 노드만 유지
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true'
        ),
        robot_state_publisher_node,
        world_to_link0
    ])