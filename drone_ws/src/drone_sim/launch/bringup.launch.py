import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_config import WORLD_NAME  # noqa: E402

# ---------------------------------------------------------------------------
# SINGLE ENTRY POINT: starts/stops the entire stack (Gazebo + ros_gz bridge + SLAM/RViz)
# from a SINGLE launch file, in the CORRECT ORDER and TOGETHER.
#
# WHY: Starting components separately, and especially restarting Gazebo UNDERNEATH
# a live RViz/SLAM stack resets the simulation clock to zero ("jump back in time").
# When RViz operates with use_sim_time, it detects this jump backward and resets itself,
# and this reset sometimes locks up at 100% CPU; furthermore it clears the rtabmap
# TF buffers, breaking the odom->base_link connection, resulting in a "two unconnected trees" error.
# Proof and full diagnosis: tasks/lessons.md (Lesson 3). This file structurally prevents
# that error by ensuring the "restart" action ALWAYS cycles the entire stack together.
#
# STARTUP ORDER (the TimerAction delays below enforce this):
#   1. Gazebo (sim.launch.py)          -> t=0s : world + sensors + /clock
#   2. ros_gz bridge                   -> t=5s : forwards GZ topics to ROS
#   3. SLAM/RViz (slam.launch.py)      -> t=8s : starts while /clock and camera are streaming
# Delays guarantee that sub-components (use_sim_time) start AFTER a healthy /clock
# and sensor stream are ready.
# ---------------------------------------------------------------------------

# World name comes from world_config.py (Phase 1: husarion_office - see
# tasks/loop_closure_roadmap.md).
# D435 -> D455 transition (2026-07-10, user request): IMU now comes from the embedded IMU
# of the D455, NOT from the x500's OWN sensor - under the SAME sub-model (d455) tree,
# under the same link with the camera (see models/d455/model.sdf).
CAMERA_BASE = f'/world/{WORLD_NAME}/model/drone/model/d455/link/link/sensor/realsense_d455'
IMU_TOPIC = f'/world/{WORLD_NAME}/model/drone/model/d455/link/link/sensor/imu_sensor/imu'


def generate_launch_description():
    pkg_drone_sim = get_package_share_directory('drone_sim')
    sim_launch = os.path.join(pkg_drone_sim, 'launch', 'sim.launch.py')
    slam_launch = os.path.join(pkg_drone_sim, 'launch', 'slam.launch.py')

    # Passed through to sim.launch.py's own 'gui' argument (default false).
    gui_arg = DeclareLaunchArgument(
        'gui', default_value='false',
        description='Launch Gazebo with its GUI client instead of headless.')

    # 1) Gazebo (world, drone, sensors, /clock broadcast)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={'gui': LaunchConfiguration('gui')}.items(),
    )

    # 2) ros_gz bridge (GZ -> ROS). Exact same topic mappings as the command
    #    we manually ran before; unidirectional markers: '[' = GZ->ROS,
    #    ']' = ROS->GZ (only /x500/cmd_vel is in this direction).
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        arguments=[
            f'{CAMERA_BASE}/image@sensor_msgs/msg/Image[gz.msgs.Image',
            f'{CAMERA_BASE}/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            f'{CAMERA_BASE}/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            f'{IMU_TOPIC}@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/x500/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        ],
    )

    # 3) SLAM stack: static TFs + rtabmap (with embedded VO) + RViz
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_launch)
    )

    return LaunchDescription([
        gui_arg,
        gazebo,
        TimerAction(period=5.0, actions=[bridge]),
        TimerAction(period=8.0, actions=[slam]),
    ])
