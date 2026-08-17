"""Tool schemas and system prompts handed to the LLM.

Schemas only - the implementations live on the bridge node because they need
ROS publishers and live odometry. Keeping the contract in one file makes it
obvious what the model is allowed to do to a flying vehicle.
"""

# Altitude band the stack is tuned for: nbv_planner clamps its frontier goals
# to [min_altitude, max_altitude] = [0.3, 1.0] and local_planner refuses to
# leave [0.3, 5.0]. Advertising a tighter, honest range to the model keeps it
# from proposing altitudes the controller would silently clamp anyway.
MIN_ALTITUDE = 0.3
MAX_ALTITUDE = 2.0
DEFAULT_STAR_RADIUS = 2.0

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'navigate_to',
            'description': (
                'Fly to a single point in the map frame. The global planner '
                'routes around obstacles automatically, so give the '
                'destination, not a path.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'x': {
                        'type': 'number',
                        'description': 'Target X in metres, map frame.'},
                    'y': {
                        'type': 'number',
                        'description': 'Target Y in metres, map frame.'},
                    'z': {
                        'type': 'number',
                        'description': (
                            f'Optional altitude in metres '
                            f'({MIN_ALTITUDE}-{MAX_ALTITUDE}). Omit to hold '
                            f'the current altitude.')},
                },
                'required': ['x', 'y'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'draw_star',
            'description': (
                'Fly a five-pointed star (pentagram) pattern. Use this for any '
                'request to draw or trace a star.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'radius': {
                        'type': 'number',
                        'description': (
                            f'Star radius in metres. Defaults to '
                            f'{DEFAULT_STAR_RADIUS}.')},
                    'center_x': {
                        'type': 'number',
                        'description': (
                            'Optional star centre X. Omit to centre the star '
                            'on the current position.')},
                    'center_y': {
                        'type': 'number',
                        'description': (
                            'Optional star centre Y. Omit to centre the star '
                            'on the current position.')},
                    'altitude': {
                        'type': 'number',
                        'description': (
                            'Optional altitude in metres. Omit to hold the '
                            'current altitude.')},
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'start_exploration',
            'description': (
                'Begin autonomous frontier exploration: the drone repeatedly '
                'picks the next-best unexplored viewpoint and maps the area on '
                'its own. Use for "explore", "map the area", "look around".'),
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'stop_and_hover',
            'description': (
                'Cancel the current goal or pattern and hold position. Use for '
                '"stop", "halt", "hold", "cancel".'),
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_status',
            'description': (
                'Read live telemetry: position, altitude, speed, odometry '
                'health, current mode and active goal. Use whenever the user '
                'asks how the drone is doing or where it is.'),
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
]

COMMAND_SYSTEM_PROMPT = f"""\
You are the onboard commander of an autonomous quadcopter flying in a simulated \
warehouse. You control it exclusively through the provided functions.

Conventions:
- Coordinates are metres in the map frame. X and Y are horizontal, Z is altitude.
- The map origin is where the drone started.
- Safe altitude range is {MIN_ALTITUDE}-{MAX_ALTITUDE} m.
- Obstacle avoidance and path planning are handled downstream. Never try to \
decompose a route into small steps; give the destination.

Rules:
- Call a function whenever the user asks for an action or for status. Do not \
claim you have done something without calling the matching function.
- If a request is ambiguous or unsafe, ask a brief clarifying question instead \
of guessing coordinates.
- After a function returns, reply in one or two short sentences describing what \
is happening. You are speaking to a pilot watching a screen: be concrete, no \
filler, no emoji.
"""

# Narration runs on a 5s timer, so it must be cheap and must never call tools -
# it is a formatting pass over a telemetry snapshot, nothing more.
NARRATION_SYSTEM_PROMPT = """\
You narrate live drone telemetry for a pilot.

Given a JSON telemetry snapshot, reply with ONE short sentence (max 25 words) in \
plain English covering what matters right now: what the drone is doing, where it \
is, and any problem.

Lead with the problem if odometry is unhealthy or a goal is unreachable - those \
matter more than position. If everything is nominal, keep it brief and factual. \
No preamble, no bullet points, no emoji, no repetition of raw field names.
"""
