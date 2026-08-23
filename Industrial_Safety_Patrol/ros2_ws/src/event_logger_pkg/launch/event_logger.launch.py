from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    event_logger_node = Node(
        package='event_logger_pkg',
        executable='event_logger_node',
        name='event_logger_node',
        parameters=[
            {'use_sim_time': True},
            {'save_image_on_warning': True},
            {'save_image_on_emergency': True},
            {'min_log_interval_sec': 1.0},
            {'db_path': '/workspace/data/safety_events.db'},
            {'image_dir': '/workspace/data/event_images'},
        ],
        output='screen',
    )

    return LaunchDescription([event_logger_node])
