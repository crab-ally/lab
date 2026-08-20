import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('perception_safety_pkg')
    param_file = os.path.join(pkg_share, 'config', 'safety_params.yaml')

    # 1. Static TF Publisher (camera_color_optical_frame -> base_link)
    # x y z yaw pitch roll parent child (필요시 좌표 수정)
    tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_static_tf',
        arguments=['0.5', '0.0', '1.0', '-1.57', '0.0', '-1.57', 'base_link', 'camera_color_optical_frame']
    )

    # 2. Node 1: Perception Node (YOLOv8 + DeepSORT)
    perception_node = Node(
        package='perception_safety_pkg',
        executable='perception_node',
        name='perception_node',
        parameters=[param_file],
        output='screen'
    )

    # 3. Node 2: 3D Fusion Node (Depth + Scan + TF)
    fusion_node = Node(
        package='perception_safety_pkg',
        executable='fusion_node_3d',
        name='fusion_node_3d',
        parameters=[param_file],
        output='screen'
    )

    forklift_controller_node=Node(
        package='perception_safety_pkg',
        executable='forklift_controller_node',
        name='forklift_controller_node',
        output='screen'
    )

    # 4. Node 3: TTC Node (Risk Assessment)
    ttc_node = Node(
        package='perception_safety_pkg',
        executable='ttc_node',
        name='ttc_node',
        parameters=[param_file],
        output='screen'
    )

    return LaunchDescription([
        tf_node,
        perception_node,
        fusion_node,
        forklift_controller_node,
        ttc_node
    ])