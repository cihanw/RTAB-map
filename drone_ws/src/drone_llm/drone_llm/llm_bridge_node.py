#!/usr/bin/env python3
"""LLM bridge: natural language in, drone goals out, telemetry narrated back.

Sits between nbv_planner and theta_star_planner as a GOAL ARBITER:

    nbv_planner --/nbv/goal--> [ llm_bridge ] --/local_planner/current_target--> theta_star

nbv_planner publishes a fresh frontier goal every 2s (planning_period_sec). If
the bridge merely published alongside it, exploration would overwrite a user's
"fly to X" two seconds later. Routing NBV *through* the bridge makes goal
ownership explicit: in EXPLORE mode NBV goals pass through untouched, in every
other mode they are dropped.

Nothing downstream changes - theta_star and local_planner keep their existing
topics, parameters and tuning.
"""

import json
import math
import queue
import threading

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from drone_llm.geometry import distance_2d, star_waypoints, yaw_from_quaternion
from drone_llm.ollama_client import OllamaClient, OllamaError
from drone_llm.tools import (COMMAND_SYSTEM_PROMPT, DEFAULT_STAR_RADIUS,
                             MAX_ALTITUDE, MIN_ALTITUDE,
                             NARRATION_SYSTEM_PROMPT, TOOLS)

# Must match nbv_planner.TARGET_QOS / theta_star_planner.LATCHED_QOS exactly.
# Those publishers are TRANSIENT_LOCAL; a mismatched subscription here would
# silently receive nothing.
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)

MODE_IDLE = 'IDLE'
MODE_EXPLORE = 'EXPLORE'
MODE_NAVIGATE = 'NAVIGATE'
MODE_PATTERN = 'PATTERN'

# Bounded so a wedged or slow model cannot grow an unbounded backlog.
COMMAND_QUEUE_MAX = 8


class LlmBridge(Node):

    def __init__(self):
        super().__init__('llm_bridge')

        self.declare_parameter('ollama_host', 'http://localhost:11434')
        self.declare_parameter('model', 'qwen3:14b')
        self.declare_parameter('telemetry_period_sec', 5.0)
        self.declare_parameter('map_frame', 'map')
        # Matches local_planner's own goal_tolerance (0.3) plus a margin: the
        # bridge should advance to the next star vertex once the controller is
        # done with the current one, not before.
        self.declare_parameter('waypoint_tolerance', 0.5)
        # Same health criteria local_planner uses to decide whether to trust
        # odometry (see local_planner.py odom_cov_threshold / odom_timeout_sec),
        # so the narration agrees with what the controller actually does.
        self.declare_parameter('odom_cov_threshold', 1.0)
        self.declare_parameter('odom_timeout_sec', 1.5)
        self.declare_parameter('narrate', True)

        def p(name):
            return self.get_parameter(name).value

        self.map_frame = p('map_frame')
        self.waypoint_tolerance = float(p('waypoint_tolerance'))
        self.odom_cov_threshold = float(p('odom_cov_threshold'))
        self.odom_timeout_sec = float(p('odom_timeout_sec'))
        self.narrate_enabled = bool(p('narrate'))

        self.llm = OllamaClient(host=p('ollama_host'), model=p('model'))

        # ---- state -------------------------------------------------------
        self._lock = threading.Lock()
        self.mode = MODE_IDLE
        self.pose = None            # (x, y, z, yaw)
        self.speed = 0.0
        self.last_odom_cov = 0.0
        self.last_odom_time = None
        self.active_goal = None     # (x, y, z)
        self.pattern = []           # remaining waypoints
        self.pattern_name = None
        self.unreachable_goal = None
        self.history = [{'role': 'system', 'content': COMMAND_SYSTEM_PROMPT}]

        cb = ReentrantCallbackGroup()

        # ---- ROS I/O -----------------------------------------------------
        self.goal_pub = self.create_publisher(
            PoseStamped, '/local_planner/current_target', LATCHED_QOS)
        self.response_pub = self.create_publisher(String, '/llm/response', 10)
        self.narration_pub = self.create_publisher(String, '/llm/narration', 10)

        self.create_subscription(
            Odometry, '/rtabmap/odom', self._odom_cb, 10, callback_group=cb)
        self.create_subscription(
            PoseStamped, '/nbv/goal', self._nbv_goal_cb, LATCHED_QOS,
            callback_group=cb)
        self.create_subscription(
            PoseStamped, '/theta_star/unreachable_target',
            self._unreachable_cb, LATCHED_QOS, callback_group=cb)
        self.create_subscription(
            String, '/llm/user_input', self._user_input_cb, 10,
            callback_group=cb)

        # Pattern progress is checked on a timer rather than inside the odom
        # callback so waypoint advancement runs at a predictable rate and does
        # not add work to the 14Hz odometry path.
        self.create_timer(0.5, self._pattern_tick, callback_group=cb)
        if self.narrate_enabled:
            self.create_timer(float(p('telemetry_period_sec')),
                              self._telemetry_tick, callback_group=cb)

        # ---- LLM worker --------------------------------------------------
        # One GPU, so all model access is serialised through a single worker.
        # Overlapping requests would only contend for the same device.
        self._commands = queue.Queue(maxsize=COMMAND_QUEUE_MAX)
        self._pending_snapshot = None   # newest telemetry only; stale dropped
        self._snapshot_lock = threading.Lock()
        self._worker = threading.Thread(target=self._llm_worker, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"LLM bridge started (model={p('model')}, mode={self.mode}).")
        if not self.llm.available():
            self.get_logger().warn(
                f"Ollama not reachable at {p('ollama_host')} or model "
                f"'{p('model')}' not pulled - commands will report an error "
                f"until it is up.")

    # ==================================================================
    # ROS callbacks
    # ==================================================================
    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        with self._lock:
            self.pose = (pos.x, pos.y, pos.z,
                         yaw_from_quaternion(ori.x, ori.y, ori.z, ori.w))
            self.speed = math.sqrt(lin.x ** 2 + lin.y ** 2 + lin.z ** 2)
            self.last_odom_cov = msg.pose.covariance[0]
            self.last_odom_time = self.get_clock().now()

    def _nbv_goal_cb(self, msg):
        """Frontier goal from nbv_planner. Forwarded only in EXPLORE mode."""
        with self._lock:
            if self.mode != MODE_EXPLORE:
                return
            pos = msg.pose.position
            self.active_goal = (pos.x, pos.y, pos.z)
        self.goal_pub.publish(msg)

    def _unreachable_cb(self, msg):
        pos = msg.pose.position
        with self._lock:
            self.unreachable_goal = (pos.x, pos.y, pos.z)

    def _user_input_cb(self, msg):
        text = msg.data.strip()
        if not text:
            return
        try:
            self._commands.put_nowait(text)
        except queue.Full:
            self._publish_response(
                'Still working through earlier commands - try again shortly.')

    # ==================================================================
    # Goal publishing / pattern sequencing
    # ==================================================================
    def _publish_goal(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        with self._lock:
            self.active_goal = (float(x), float(y), float(z))
            self.unreachable_goal = None

    def _pattern_tick(self):
        """Advance to the next pattern vertex once the current one is reached.

        Each vertex is published as a normal goal, so theta_star still plans an
        obstacle-free route between star points - the shape is a sequence of
        destinations, not a blindly-followed polyline.
        """
        with self._lock:
            if self.mode != MODE_PATTERN:
                return
            if self.pose is None or self.active_goal is None:
                return
            reached = distance_2d(self.pose[0], self.pose[1],
                                  self.active_goal[0], self.active_goal[1])
            if reached > self.waypoint_tolerance:
                return

            # The current vertex has been reached. Completion is only declared
            # here - NOT when the last waypoint is *published* - otherwise the
            # star would be announced finished while the drone was still flying
            # its final edge.
            if self.pattern:
                nxt = self.pattern.pop(0)
                name = None
            else:
                nxt = None
                name = self.pattern_name
                self.mode = MODE_IDLE
                self.pattern_name = None

        if nxt is not None:
            self._publish_goal(*nxt)
        else:
            self._publish_response(f'{name} complete.')

    # ==================================================================
    # Tool implementations
    # ==================================================================
    def _tool_navigate_to(self, args):
        try:
            x = float(args['x'])
            y = float(args['y'])
        except (KeyError, TypeError, ValueError):
            return {'ok': False, 'error': 'navigate_to needs numeric x and y.'}

        with self._lock:
            if self.pose is None:
                return {'ok': False,
                        'error': 'No odometry yet - cannot determine altitude.'}
            z = self.pose[2]
        if args.get('z') is not None:
            try:
                z = float(args['z'])
            except (TypeError, ValueError):
                return {'ok': False, 'error': 'z must be a number.'}
        z = max(MIN_ALTITUDE, min(MAX_ALTITUDE, z))

        with self._lock:
            self.mode = MODE_NAVIGATE
            self.pattern = []
            self.pattern_name = None
        self._publish_goal(x, y, z)
        return {'ok': True, 'mode': MODE_NAVIGATE,
                'goal': {'x': round(x, 2), 'y': round(y, 2), 'z': round(z, 2)}}

    def _tool_draw_star(self, args):
        with self._lock:
            if self.pose is None:
                return {'ok': False,
                        'error': 'No odometry yet - cannot centre the star.'}
            cur_x, cur_y, cur_z = self.pose[0], self.pose[1], self.pose[2]

        def num(key, default):
            val = args.get(key)
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        radius = num('radius', DEFAULT_STAR_RADIUS)
        cx = num('center_x', cur_x)
        cy = num('center_y', cur_y)
        alt = max(MIN_ALTITUDE, min(MAX_ALTITUDE, num('altitude', cur_z)))

        if radius <= 0:
            return {'ok': False, 'error': 'radius must be positive.'}

        try:
            pts = star_waypoints(cx, cy, radius, alt)
        except ValueError as exc:
            return {'ok': False, 'error': str(exc)}

        with self._lock:
            self.mode = MODE_PATTERN
            self.pattern_name = 'Star'
            self.pattern = list(pts[1:])   # first vertex published immediately
        self._publish_goal(*pts[0])
        return {'ok': True, 'mode': MODE_PATTERN, 'pattern': 'star',
                'vertices': len(pts),
                'center': {'x': round(cx, 2), 'y': round(cy, 2)},
                'radius_m': round(radius, 2), 'altitude_m': round(alt, 2)}

    def _tool_start_exploration(self, _args):
        with self._lock:
            self.mode = MODE_EXPLORE
            self.pattern = []
            self.pattern_name = None
        # No goal is published here on purpose: nbv_planner emits one within
        # planning_period_sec (2s) and _nbv_goal_cb forwards it.
        return {'ok': True, 'mode': MODE_EXPLORE,
                'note': 'Frontier exploration active.'}

    def _tool_stop_and_hover(self, _args):
        with self._lock:
            self.mode = MODE_IDLE
            self.pattern = []
            self.pattern_name = None
            pose = self.pose
        if pose is None:
            return {'ok': False, 'error': 'No odometry yet.'}
        # Re-target the drone's current position so local_planner settles here
        # instead of continuing toward the last goal.
        self._publish_goal(pose[0], pose[1], pose[2])
        return {'ok': True, 'mode': MODE_IDLE, 'note': 'Holding position.'}

    def _tool_get_status(self, _args):
        return self._snapshot()

    TOOL_IMPLS = {
        'navigate_to': _tool_navigate_to,
        'draw_star': _tool_draw_star,
        'start_exploration': _tool_start_exploration,
        'stop_and_hover': _tool_stop_and_hover,
        'get_status': _tool_get_status,
    }

    # ==================================================================
    # Telemetry
    # ==================================================================
    def _odom_healthy(self):
        """Mirrors local_planner.py: stale OR high-covariance means unhealthy."""
        if self.last_odom_time is None:
            return False, 'no odometry received'
        silence = (self.get_clock().now() - self.last_odom_time).nanoseconds / 1e9
        if silence > self.odom_timeout_sec:
            return False, f'odometry stale for {silence:.1f}s'
        if self.last_odom_cov > self.odom_cov_threshold:
            return False, f'odometry covariance {self.last_odom_cov:.2f} is high'
        return True, 'nominal'

    def _snapshot(self):
        with self._lock:
            healthy, reason = self._odom_healthy()
            snap = {
                'mode': self.mode,
                'odometry_healthy': healthy,
                'odometry_note': reason,
                'speed_mps': round(self.speed, 2),
            }
            if self.pose is not None:
                snap['position'] = {
                    'x': round(self.pose[0], 2),
                    'y': round(self.pose[1], 2),
                    'altitude_m': round(self.pose[2], 2),
                }
                snap['heading_deg'] = round(math.degrees(self.pose[3]), 1)
            if self.active_goal is not None:
                snap['goal'] = {
                    'x': round(self.active_goal[0], 2),
                    'y': round(self.active_goal[1], 2),
                }
                if self.pose is not None:
                    snap['distance_to_goal_m'] = round(distance_2d(
                        self.pose[0], self.pose[1],
                        self.active_goal[0], self.active_goal[1]), 2)
            if self.mode == MODE_PATTERN:
                snap['pattern_waypoints_remaining'] = len(self.pattern)
            if self.unreachable_goal is not None:
                snap['warning'] = 'last goal reported unreachable by planner'
            return snap

    def _telemetry_tick(self):
        snap = self._snapshot()
        # Keep only the newest snapshot: if the model is busy, narrating a
        # 10-second-old position is worse than skipping a beat.
        with self._snapshot_lock:
            self._pending_snapshot = snap

    # ==================================================================
    # LLM worker (single thread -> serialised GPU access)
    # ==================================================================
    def _publish_response(self, text):
        self.response_pub.publish(String(data=text))

    def _llm_worker(self):
        while rclpy.ok():
            # User commands take priority; narration fills the idle time.
            try:
                text = self._commands.get(timeout=0.25)
            except queue.Empty:
                self._run_narration()
                continue
            try:
                self._run_command(text)
            except Exception as exc:                      # noqa: BLE001
                self.get_logger().error(f'Command failed: {exc}')
                self._publish_response(f'Command failed: {exc}')

    def _run_command(self, text):
        with self._lock:
            self.history.append({'role': 'user', 'content': text})
            messages = list(self.history)

        try:
            msg = self.llm.chat(messages, tools=TOOLS)
        except OllamaError as exc:
            self._publish_response(f'LLM unavailable: {exc}')
            return

        tool_calls = msg.get('tool_calls') or []
        if not tool_calls:
            reply = (msg.get('content') or '').strip() or '(no reply)'
            with self._lock:
                self.history.append({'role': 'assistant', 'content': reply})
            self._publish_response(reply)
            return

        # Execute every requested call, then let the model summarise results.
        with self._lock:
            self.history.append(msg)
        for call in tool_calls:
            fn = call.get('function', {})
            name = fn.get('name', '')
            args = fn.get('arguments') or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            impl = self.TOOL_IMPLS.get(name)
            if impl is None:
                result = {'ok': False, 'error': f'unknown function {name}'}
            else:
                try:
                    result = impl(self, args)
                except Exception as exc:                  # noqa: BLE001
                    self.get_logger().error(f'{name} raised: {exc}')
                    result = {'ok': False, 'error': str(exc)}
            self.get_logger().info(f'tool {name}({args}) -> {result}')
            with self._lock:
                self.history.append({'role': 'tool', 'name': name,
                                     'content': json.dumps(result)})

        with self._lock:
            messages = list(self.history)
        try:
            follow = self.llm.chat(messages, tools=TOOLS, max_tokens=200)
            reply = (follow.get('content') or '').strip()
        except OllamaError as exc:
            reply = f'(action executed, but summary failed: {exc})'
        if not reply:
            reply = 'Done.'
        with self._lock:
            self.history.append({'role': 'assistant', 'content': reply})
        self._publish_response(reply)
        self._trim_history()

    def _trim_history(self, max_turns=24):
        """Keep the system prompt plus the most recent turns so a long session
        cannot grow the prompt without bound."""
        with self._lock:
            if len(self.history) <= max_turns + 1:
                return
            self.history = [self.history[0]] + self.history[-max_turns:]

    def _run_narration(self):
        with self._snapshot_lock:
            snap = self._pending_snapshot
            self._pending_snapshot = None
        if snap is None:
            return
        try:
            msg = self.llm.chat(
                [{'role': 'system', 'content': NARRATION_SYSTEM_PROMPT},
                 {'role': 'user', 'content': json.dumps(snap)}],
                temperature=0.3, max_tokens=80, timeout=30.0)
        except OllamaError as exc:
            self.get_logger().warn(f'Narration skipped: {exc}')
            return
        text = (msg.get('content') or '').strip()
        if text:
            self.narration_pub.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    node = LlmBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
