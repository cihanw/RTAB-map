import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_config import WORLD_PACKAGE, WORLD_FILE  # noqa: E402

def generate_launch_description():
    pkg_drone_sim = get_package_share_directory('drone_sim')

    # World file is in the package/name defined in world_config.py (Phase 1:
    # husarion_office - to solve the feature-scarcity/repetitive-surface
    # problem of depot, see tasks/loop_closure_roadmap.md).
    world_file = os.path.join(get_package_share_directory(WORLD_PACKAGE), 'worlds', WORLD_FILE)

    # Path to the model directory so Gazebo can find the local x500_d455 model
    # We must append this path to the environment variable GZ_SIM_RESOURCE_PATH
    # (husarion_gz_worlds automatically adds its own model path with env-hook,
    # this is still required for our x500_d455/d455 models)
    models_path = os.path.join(pkg_drone_sim, 'models')

    current_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if current_resource_path:
        os.environ['GZ_SIM_RESOURCE_PATH'] = f"{models_path}:{current_resource_path}"
    else:
        os.environ['GZ_SIM_RESOURCE_PATH'] = models_path

    # Ensure /usr/share/gz is in GZ_CONFIG_PATH so 'gz sim' is recognized
    # Sourcing ROS 2 Jazzy overrides GZ_CONFIG_PATH, hiding system-wide Gazebo plugins
    current_config_path = os.environ.get('GZ_CONFIG_PATH', '')
    if current_config_path:
        os.environ['GZ_CONFIG_PATH'] = f"/usr/share/gz:{current_config_path}"
    else:
        os.environ['GZ_CONFIG_PATH'] = "/usr/share/gz"

    # Headless by DEFAULT (2026-07-09, user request): '-s' never starts the
    # GUI client (~1.3GB VRAM savings - this machine only has 4GB VRAM,
    # Gazebo GUI+server combined were consuming almost all of it). Exposed
    # as a launch argument (2026-07-30, user request) rather than hardcoded,
    # so an occasional GUI run ('gui:=true') doesn't require editing this
    # file - default stays headless, so every existing invocation and this
    # file's own past VRAM reasoning are unaffected unless gui is requested.
    # '--headless-rendering' still applies in the headless branch only:
    # camera/depth sensors are offscreen-rendered on the server side
    # regardless, so sensor data/quality never depends on this flag - it
    # only controls whether the interactive 3D view exists.
    gui_arg = DeclareLaunchArgument(
        'gui', default_value='false',
        description='Launch Gazebo with its interactive GUI client instead '
                    'of headless. Uses more VRAM (see comment above) - this '
                    'machine has 4GB total, verify headroom with nvidia-smi '
                    'before enabling alongside the full RViz/rtabmap stack.')

    gz_sim_headless = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', '--headless-rendering', world_file],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('gui')),
    )
    gz_sim_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file],
        output='screen',
        condition=IfCondition(LaunchConfiguration('gui')),
    )

    return LaunchDescription([
        gui_arg,
        gz_sim_headless,
        gz_sim_gui,
    ])
