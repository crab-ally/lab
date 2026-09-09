import os
from pathlib import Path
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node


def load_yaml(package_exact_path, file_path):
    try:
        with open(os.path.join(package_exact_path, file_path), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except EnvironmentError:
        return None


def generate_launch_description():
    workspace_dir = Path(__file__).resolve().parent.parent

    # ==================================================================
    # 1. URDF & SRDF
    # ==================================================================

    urdf_path = workspace_dir / "urdf" / "ur5e_2f85.urdf"
    srdf_path = workspace_dir / "config" / "ur5e_2f85" / "ur5e_2f85.srdf"

    with open(urdf_path, "r", encoding="utf-8") as f:
        robot_description_content = f.read()

    with open(srdf_path, "r", encoding="utf-8") as f:
        robot_description_semantic_content = f.read()

    robot_description = {
        "robot_description": robot_description_content
    }

    robot_description_semantic = {
        "robot_description_semantic": robot_description_semantic_content
    }

    # ==================================================================
    # 2. Kinematics
    # ==================================================================

    kinematics_yaml = load_yaml(
        str(workspace_dir),
        "config/ur5e_2f85/kinematics.yaml"
    )

    robot_description_kinematics = {
        "robot_description_kinematics": kinematics_yaml or {}
    }

    # ==================================================================
    # 3. OMPL Planning
    # ==================================================================

    ompl_planning_yaml = load_yaml(
        str(workspace_dir),
        "config/ur5e_2f85/ompl_planning.yaml"
    )

    ompl_config = ompl_planning_yaml or {}
    ompl_config["planning_plugin"] = "ompl_interface/OMPLPlanner"

    planning_pipelines = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl_config,
    }

    # ==================================================================
    # 4. Joint Limits
    # ==================================================================

    joint_limits_yaml = load_yaml(
        str(workspace_dir),
        "config/ur5e_2f85/joint_limits.yaml"
    )

    robot_description_planning = {
        "robot_description_planning": joint_limits_yaml or {}
    }

    # ==================================================================
    # 5. MoveIt Controllers
    # ==================================================================

    moveit_controllers_yaml = load_yaml(
        str(workspace_dir),
        "config/ur5e_2f85/moveit_controllers.yaml"
    )

    moveit_controller_config = moveit_controllers_yaml or {}

    # ==================================================================
    # 6. Trajectory Execution
    # ==================================================================

    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.5,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    # ==================================================================
    # 7. Planning Scene Monitor
    # ==================================================================

    planning_scene_monitor_parameters = {
        "planning_scene_monitor.publish_planning_scene": True,
        "planning_scene_monitor.publish_geometry_updates": True,
        "planning_scene_monitor.publish_state_updates": True,
        "planning_scene_monitor.publish_transforms_updates": True,
    }

    # ==================================================================
    # 8. TF: world -> base
    # ==================================================================

    world_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base",
        arguments=[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "world",
            "base",
        ],
        output="screen",
    )

    # ==================================================================
    # 10. MoveGroup
    # ==================================================================

    move_group_parameters = [
        robot_description,
        robot_description_semantic,
        robot_description_kinematics,
        robot_description_planning,
        planning_pipelines,
        trajectory_execution,
        moveit_controller_config,
        planning_scene_monitor_parameters,
    ]

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=move_group_parameters,
    )

    # ==================================================================
    # 11. Launch
    # ==================================================================

    return LaunchDescription([
        world_to_base,
        move_group_node,
    ])
