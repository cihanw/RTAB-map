import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_drone_sim = get_package_share_directory('drone_sim')

    # Passed through bringup.launch.py to sim.launch.py's 'gui' argument.
    # Default false (headless) preserves every existing invocation; run
    # with 'ros2 launch drone_sim autonomous.launch.py gui:=true' for the
    # interactive Gazebo client (see sim.launch.py for the VRAM note).
    gui_arg = DeclareLaunchArgument(
        'gui', default_value='false',
        description='Launch Gazebo with its GUI client instead of headless.')

    # Include the main bringup launch file (Gazebo, rtabmap, EKF, etc.)
    bringup_launch_path = os.path.join(pkg_drone_sim, 'launch', 'bringup.launch.py')
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bringup_launch_path),
        launch_arguments={'gui': LaunchConfiguration('gui')}.items(),
    )

    # Local Planner: Autonomous obstacle avoidance and goal navigation
    # Note: starting with a 10-second delay to allow bringup nodes to stabilize
    local_planner = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='drone_sim',
                executable='local_planner.py',
                name='local_planner',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    # Listen to the obstacle-aware waypoint published by
                    # theta_star_planner (NOT the raw NBV goal) - so the drone
                    # follows a path around an obstacle instead of blindly heading
                    # straight into a goal right next to it. ZERO code changes
                    # in local_planner.py, only this parameter.
                    'goal_topic': '/theta_star/next_waypoint',
                }]
            )
        ]
    )

    # NBV Planner: Selects the next best goal (Frontier) and sends it to theta_star_planner
    nbv_planner = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='drone_sim',
                executable='nbv_planner.py',
                name='nbv_planner',
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        ]
    )

    # Theta* Global Planner: Adds an obstacle-map-aware global path planner
    # layer between the frontier goal selected by NBV (potentially close to obstacles)
    # and the local_planner's purely reactive APF - solves the
    # "drone getting stuck in a local minimum right next to the goal" issue
    # observed in live testing (see scripts/theta_star_planner.py top-of-file comment).
    # Started 2s after nbv_planner (t=12s) - order is not safety-critical
    # (local_planner just hovers in the "no goal" state anyway), only
    # structured so nbv's first goal usually arrives while this node is up.
    theta_star_planner = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='drone_sim',
                executable='theta_star_planner.py',
                name='theta_star_planner',
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        ]
    )

    return LaunchDescription([
        gui_arg,
        bringup,
        local_planner,
        nbv_planner,
        theta_star_planner
    ])
