from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Start the LLM bridge.

    The console is NOT launched here: it needs an interactive terminal, and a
    node started by ros2 launch does not get one. Run it yourself with
        ros2 run drone_llm llm_console
    """
    model_arg = DeclareLaunchArgument(
        'model', default_value='qwen3:14b',
        description='Ollama model tag used for commands and narration.')
    host_arg = DeclareLaunchArgument(
        'ollama_host', default_value='http://localhost:11434',
        description='Ollama server URL. Reachable from inside distrobox '
                    'because the container shares the host network.')
    period_arg = DeclareLaunchArgument(
        'telemetry_period_sec', default_value='5.0',
        description='How often telemetry is narrated, in seconds.')
    narrate_arg = DeclareLaunchArgument(
        'narrate', default_value='true',
        description='Set false to disable periodic narration and use the '
                    'bridge for commands only.')

    bridge = Node(
        package='drone_llm',
        executable='llm_bridge',
        name='llm_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'model': LaunchConfiguration('model'),
            'ollama_host': LaunchConfiguration('ollama_host'),
            'telemetry_period_sec': LaunchConfiguration('telemetry_period_sec'),
            'narrate': LaunchConfiguration('narrate'),
        }],
    )

    return LaunchDescription([
        model_arg, host_arg, period_arg, narrate_arg, bridge,
    ])
