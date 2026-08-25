import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share=get_package_share_directory('perception_safety_pkg')
    param_file=os.path.join(pkg_share,'config','safety_params.yaml')
    use_sim_time={'use_sim_time':True}

    # TF (base_link → camera_link) 는 mujoco_ros2_bridge.py / mujoco_ros2_bridge_viewer.py 에서 발행
    # 중복 발행 방지를 위해 여기서는 static_transform_publisher를 실행하지 않음

    perception_node=Node(
        package='perception_safety_pkg',
        executable='perception_node',
        name='perception_node',
        parameters=[param_file,use_sim_time],
        output='screen'
    )

    fusion_node=Node(
        package='perception_safety_pkg',
        executable='fusion_node_3d',
        name='fusion_node_3d',
        parameters=[param_file,use_sim_time],
        output='screen'
    )

    forklift_controller_node=Node(
        package='perception_safety_pkg',
        executable='forklift_controller_node',
        name='forklift_controller_node',
        parameters=[use_sim_time],
        output='screen'
    )

    ttc_node=Node(
        package='perception_safety_pkg',
        executable='ttc_node',
        name='ttc_node',
        parameters=[param_file,use_sim_time],
        output='screen'
    )

    return LaunchDescription([
        perception_node,
        fusion_node,
        forklift_controller_node,
        ttc_node
    ])