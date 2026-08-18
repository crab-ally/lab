from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Fire Detection Node (RGB + Depth 색상 및 면적 기반 후보 추출)
    fire_detection_node = Node(
        package='fire_detection_pkg',
        executable='fire_detection_node',
        name='fire_detection_node',
        output='screen'
    )

    # 2. Fire Fusion Node (Depth + LiDAR + TF 3D 위치 추정 및 평면 검증)
    fire_fusion_node = Node(
        package='fire_detection_pkg',
        executable='fire_fusion_node',
        name='fire_fusion_node',
        output='screen'
    )

    return LaunchDescription([
        fire_detection_node,
        fire_fusion_node
    ])
