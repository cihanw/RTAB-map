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

    # GUI by DEFAULT (2026-07-31, user request). The earlier headless default
    # (2026-07-09) existed because the machine of the day had only 4GB VRAM and
    # the Gazebo GUI costs ~1.3GB on top of the server. That constraint no longer
    # applies on the current workstation (3x RTX 4090, 24GB each), so the
    # interactive 3D view is on unless 'gui:=false' is passed.
    # '--headless-rendering' still applies in the headless branch only:
    # camera/depth sensors are offscreen-rendered on the server side
    # regardless, so sensor data/quality never depends on this flag - it
    # only controls whether the interactive 3D view exists.
    gui_arg = DeclareLaunchArgument(
        'gui', default_value='true',
        description='Launch Gazebo with its interactive GUI client. Set '
                    'false for a headless server-only run (saves ~1.3GB VRAM).')

    gz_sim_headless = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', '--headless-rendering', world_file],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('gui')),
    )
    # vglrun -d egl (2026-08-17, user-reported flicker over VNC): DISPLAY=:2 is
    # Xtigervnc, a software X server with no GPU. Without VirtualGL, the GUI
    # client's GLX calls take the software path AND get double-buffered again by
    # xfwm4's compositor on the way to the VNC framebuffer - both stages flicker.
    # vglrun intercepts GLX, renders on the real NVIDIA GPU via EGL, and ships
    # only the final frame to the X server, bypassing both. Object-level flicker
    # was fixed separately by disabling xfwm4 compositing; this fixes the rest.
    # Only the GUI branch needs it - headless never creates a window.
    gz_sim_gui = ExecuteProcess(
        cmd=['vglrun', '-d', 'egl', 'gz', 'sim', '-r', world_file],
        output='screen',
        condition=IfCondition(LaunchConfiguration('gui')),
    )

    return LaunchDescription([
        gui_arg,
        gz_sim_headless,
        gz_sim_gui,
    ])
