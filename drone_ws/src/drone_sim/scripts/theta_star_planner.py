#!/usr/bin/env python3
"""2D global path planner node (Theta*).

Sits between the NBV exploration planner (which picks potentially
obstacle-adjacent frontier targets) and the purely reactive local planner.
Root cause it exists for (diagnosed live): the local planner's APF sums only
the forces it can see from the current pose; a target right next to an
obstacle balances attraction against repulsion and locks the drone in a
local minimum (verified: 5+ minutes oscillating 0.67m from a target).

ARCHITECTURE SEAM (rewrite 2026-07-11):
  - NBV picks 3D targets (its altitude band is a feature-band defense).
  - THIS node plans in 2D (xy only). Vertical routing has only ever caused
    harm here - a 3D waypoint at z=1.8m once routed the drone above the
    feature-rich band (objects at z=0-1.5, horizontal camera) and visual
    odometry was lost for 8+ minutes. Obstacles within a z band are
    projected onto the plane; the search never sees z.
  - The local planner flies 3D and closes any z error gradually.

Waypoint z rule: EVERY published waypoint carries the final NBV goal's z.
Simplest possible rule - no drifting interpolation anchor across the 3s
replans, and the excursion is bounded by construction (NBV goals satisfy
z in [0.3, 1.0]). Coupled consequence: the waypoint ADVANCE check is
XY-ONLY - with goal-z waypoints, a 3D check could carry a permanent 0.7m
z error greater than the 0.5m tolerance and never advance past a waypoint
the drone is standing on in xy (a new deadlock class). Final 3D arrival
precision remains the local planner's job via its own goal_tolerance.

Zero code changes needed in local_planner.py for this integration - its
goal_topic parameter is simply pointed at /theta_star/next_waypoint in
autonomous.launch.py.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from theta_star_grid import OccupancyGrid2D, theta_star, world_to_cell

# "Latched topic" profile (ROS1 latching equivalent). The target chain
# (NBV -> here -> local planner) publishes ONLY on change; a late subscriber
# with default volatile QoS misses the first message and the whole stack
# deadlocks (verified live: local_planner logged "Waiting for goal" forever).
# TRANSIENT_LOCAL redelivers the last message to late joiners.
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)

# Consecutive planning failures for the same goal before it is reported
# unreachable to NBV (which blacklists it and picks another frontier).
# 5 -> 3 (2026-07-13): at the 3s replan period this is now ~9s of
# failures instead of ~15s - still enough transient-flicker tolerance
# for a frontier's still-maturing local grid, but with nbv_planner's
# region-based blacklist (blacklist_radius_m) now doing most of the work
# of avoiding repeat failures against the SAME obstacle, the extra
# patience of 5 attempts bought little; cutting to 3 shrinks the
# per-target cost of the (now rarer) genuine dead-end case.
# 3 -> 2 (2026-07-17, freeze-frequency incident): the original 3rd-retry
# patience existed for a still-maturing/stale grid; GridGlobal/UpdateError
# 0.1 (slam.launch.py) now keeps the grid fresh at ~1Hz, so a 3rd attempt
# against an unchanged grid was pure wasted time - live log showed 14
# consecutive unreachable targets (5 along one dead frontier ridge) each
# paying the full 3-attempt cost, a single 154s freeze. 2 keeps one
# genuine retry while cutting ~3s off every dead candidate.
UNREACHABLE_FAIL_THRESHOLD = 2

# Goal-change detection epsilon (m): NBV voxel targets differ by >= one
# 0.2m voxel, so anything materially smaller works.
GOAL_CHANGE_EPSILON = 1e-3

# Waypoint republish dedup epsilon (m): the local planner resets its
# reached-latch and re-logs on EVERY goal message, so identical waypoints
# must not be re-sent (observed live as log spam + latch churn).
REPUBLISH_EPSILON = 1e-6


class ThetaStarPlanner(Node):

    def __init__(self):
        # TF/cloud data is sim-time stamped; wall-clock would make every TF
        # lookup an extrapolation error. use_sim_time is an rclpy built-in,
        # so parameter_overrides (not declare_parameter) is required.
        super().__init__(
            'theta_star_planner',
            parameter_overrides=[Parameter('use_sim_time', value=True)])

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        # Same source the local planner uses for its own repulsion - the
        # only input needed for the grid ("unknown = free": the grid is
        # built exclusively from obstacles, so frontier goals - which by
        # definition border unknown space - stay reachable).
        self.declare_parameter('obstacle_topic', '/rtabmap/octomap_obstacles')
        # NBV already publishes here; reused so nbv_planner needs no rewiring.
        self.declare_parameter('nbv_goal_topic', '/local_planner/current_target')
        self.declare_parameter('waypoint_topic', '/theta_star/next_waypoint')
        self.declare_parameter('path_topic', '/theta_star/path')
        # == nbv_planner voxel_size (0.2): frontier targets are snapped to
        # that resolution, matching avoids a systematic offset; coarser than
        # RTAB-Map's raw Grid/CellSize (0.1) to bound search cost.
        self.declare_parameter('grid_resolution', 0.2)
        # Projection band: which obstacle points become 2D obstacles -
        # everything the airframe could plausibly occupy. Flight band is
        # z in [0.3, 1.0] (NBV ceiling); 1.5 adds headroom for APF wander
        # and the airframe's own extent. Overhangs ABOVE 1.5 (unreachable
        # by construction) do not wall off usable floor space. If live
        # tests ever show clipping under low overhangs, RAISE this -
        # never re-add z to the search.
        self.declare_parameter('projection_z_min', 0.0)
        self.declare_parameter('projection_z_max', 1.5)
        # Must stay >= local_planner's hard repulsion floor so the global
        # corridor keeps the reactive layer comfortably clear of walls.
        # History: 0.6 -> 0.45 (2026-07-11: 0.6 left a 0.2m dead zone vs
        # goal_snap_radius_m 0.4 where buried frontiers couldn't be
        # snapped, driving unreachable churn); 0.45 -> 0.55 (2026-07-12:
        # min_safe_distance was raised 0.4 -> 0.5 after the frame-verified
        # close-proximity VO death, and the ordering constraint
        # min_safe <= inflation forces this up with it). ACCEPTED COST:
        # the inflation-vs-snap gap grows back to 0.15m, so some
        # wall-adjacent frontiers will again be unreachable-blacklisted
        # (~15s each, self-healing) - safety margin wins over exploration
        # efficiency here.
        self.declare_parameter('inflation_radius_m', 0.55)
        # Slower than NBV's 2s loop (a full search costs more than a
        # frontier scan) but fast enough to absorb newly mapped obstacles.
        self.declare_parameter('replan_interval_sec', 3.0)
        # Cheap TF-proximity check for waypoint hand-off, decoupled from
        # the expensive replan cadence.
        self.declare_parameter('waypoint_advance_hz', 5.0)
        # XY distance for advancing to the next waypoint. Deliberately
        # LOOSER than local_planner's goal_tolerance (0.3): this asks "close
        # enough to steer at the next corridor point", not "arrived".
        self.declare_parameter('waypoint_tolerance', 0.5)
        # Latency cap, not a correctness knob: the warehouse at 0.2m is
        # ~15.6k cells total, so a heuristic search that hasn't reached the
        # goal in 10k expansions has flooded all reachable space. Cap-hit
        # and open-list-exhausted both return None into the same fail path.
        self.declare_parameter('max_search_iterations', 10000)
        # Bounded escape if the drone's own cell reads blocked (transient
        # inflation edge as the map updates). 3 -> 5 (2026-07-17 incident):
        # 3 cells = 0.6m barely exceeded inflation_radius_m (0.55), so a
        # drone physically pinned AGAINST a wall sat deeper in the inflated
        # zone than the snap could escape - every plan failed for 43
        # straight minutes (the latched-goal deadlock incident). The snap
        # must out-reach inflation with margin: 5 cells = 1.0m covers
        # inflation + the drone pressed to the obstacle surface itself.
        # Keep >= ceil(inflation_radius_m/grid_resolution) + 2 if either
        # of those parameters changes.
        self.declare_parameter('start_snap_radius_cells', 5)
        # HARD CONSTRAINT: must stay strictly below nbv_planner's
        # goal_tolerance (0.5). If the snapped path end sat farther than
        # that from the true NBV target, NBV would never detect arrival and
        # would hold the same target forever - a subtler deadlock.
        self.declare_parameter('goal_snap_radius_m', 0.4)

        p = self.get_parameter
        self.map_frame = p('map_frame').value
        self.base_frame = p('base_frame').value
        self.waypoint_tolerance = p('waypoint_tolerance').value
        self.max_search_iterations = p('max_search_iterations').value
        self.start_snap_radius_cells = p('start_snap_radius_cells').value
        self.goal_snap_radius_m = p('goal_snap_radius_m').value

        self.grid = OccupancyGrid2D(
            resolution=p('grid_resolution').value,
            inflation_radius_m=p('inflation_radius_m').value,
            projection_z_min=p('projection_z_min').value,
            projection_z_max=p('projection_z_max').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nbv_goal = None            # np.array([x, y, z]) - full 3D goal
        self.current_path = None        # list of np.array([x, y]) - 2D
        self._path_goal = None          # goal current_path was planned FOR
        self.path_index = 0
        self.fail_counter = 0
        self._last_published_target = None  # np.array([x, y, z])
        # True right after a fresh replan: forces the next advance_and_publish
        # tick to publish path[path_index] unconditionally, bypassing the
        # arrival-check (see do_replan/advance_and_publish for why).
        self._pending_fresh_publish = False

        self.create_subscription(
            PointCloud2, p('obstacle_topic').value, self.obstacle_callback, 1)
        self.create_subscription(
            PoseStamped, p('nbv_goal_topic').value, self.goal_callback,
            LATCHED_QOS)

        self.waypoint_pub = self.create_publisher(
            PoseStamped, p('waypoint_topic').value, LATCHED_QOS)
        self.path_pub = self.create_publisher(Path, p('path_topic').value, 1)
        self.unreachable_pub = self.create_publisher(
            PoseStamped, '/theta_star/unreachable_target', LATCHED_QOS)

        self.create_timer(p('replan_interval_sec').value, self.do_replan)
        self.create_timer(
            1.0 / p('waypoint_advance_hz').value, self.advance_and_publish)

        self.get_logger().info('Theta* global planner started (2D).')

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def obstacle_callback(self, msg):
        raw = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if raw.shape[0] == 0:
            pts = np.zeros((0, 3))
        else:
            pts = np.stack(
                [raw['x'], raw['y'], raw['z']], axis=-1).astype(np.float64)
        self.grid.update_from_points(pts)
        self._invalidate_path_if_blocked()

    def goal_callback(self, msg):
        new_goal = np.array([msg.pose.position.x,
                             msg.pose.position.y,
                             msg.pose.position.z])
        # Change detection on the FULL 3D norm: a z-only change in the NBV
        # target is a real new target (its z is what we annotate waypoints
        # with).
        if (self.nbv_goal is None or
                float(np.linalg.norm(new_goal - self.nbv_goal)) > GOAL_CHANGE_EPSILON):
            self.nbv_goal = new_goal
            self.fail_counter = 0
            self.get_logger().info(
                f'New NBV goal received: ({new_goal[0]:.2f}, '
                f'{new_goal[1]:.2f}, {new_goal[2]:.2f}) - replan triggered.')
            self.do_replan(force=True)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def get_current_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            pos = t.transform.translation
            return np.array([pos.x, pos.y, pos.z])
        except tf2_ros.TransformException:
            return None

    def _invalidate_path_if_blocked(self):
        """Immediate replan if new obstacle data blocks any remaining
        waypoint. Contract: stale-but-valid paths are preferred over
        nothing, but a known-blocked path is NEVER kept."""
        if self.current_path is None:
            return
        remaining = self.current_path[self.path_index:]
        if len(remaining) == 0:
            return
        cells = np.array(
            [world_to_cell(wp, self.grid.resolution) for wp in remaining],
            dtype=np.int64)
        if np.any(self.grid.is_blocked(cells)):
            self.get_logger().warn(
                'Followed path invalidated by new obstacle data - '
                'path cleared, replan triggered.',
                throttle_duration_sec=2.0)
            self.current_path = None
            self.path_index = 0
            self.do_replan(force=True)

    def do_replan(self, force=False):
        if self.nbv_goal is None:
            return
        # The periodic timer call (force=False) is redundant with the two
        # event-driven replan triggers below UNLESS we already have a valid,
        # in-progress path: goal_callback forces a replan the instant the
        # NBV goal changes, and _invalidate_path_if_blocked forces one the
        # instant new obstacle data actually blocks the path being followed
        # (both far more responsive than a 3s tick anyway). If neither
        # fired, discarding a still-valid path and re-solving from the
        # drone's current pose only serves to pick a possibly-different
        # "next hop" for no reason. Live 2026-07-11: in a geometrically
        # tight area this produced a livelock - each 3s replan chose a
        # slightly different immediate waypoint (any-angle search is
        # sensitive to small pose/snap differences), and since turn-then-go
        # needs a full rotate-then-translate cycle per retarget, retargeting
        # faster than that cycle completes left net displacement ~0 for 60+
        # seconds (theta* path length never shrank; target bounced inside a
        # ~1m cluster; "drone stuck looping at the exact same spot"). Rule:
        # a periodic replanner must not discard in-progress, still-valid
        # progress just because its timer fired - only replan on a real
        # reason to.
        #
        # The held path must also BELONG TO THE CURRENT GOAL: on a failed
        # replan we deliberately keep the previous goal's path (see below),
        # and without this check that stale path suppressed every timer
        # retry for the NEW goal - fail_counter froze at 1/5, the
        # unreachable report never fired, and NBV fell back to its slow
        # 122s stall timeout five times in a row (live 2026-07-11 endurance
        # run: 10 minutes lost in a corner that the 15s unreachable loop
        # was designed to escape).
        if (not force and self.current_path is not None and
                self.path_index < len(self.current_path) and
                self._path_goal is not None and
                float(np.linalg.norm(self._path_goal - self.nbv_goal)) < GOAL_CHANGE_EPSILON):
            return
        pose = self.get_current_pose()
        if pose is None:
            self.get_logger().warn(
                'Waiting for TF, skipping replan...', throttle_duration_sec=5.0)
            return

        path = theta_star(
            self.grid, pose[:2], self.nbv_goal[:2],
            self.max_search_iterations,
            start_snap_radius_cells=self.start_snap_radius_cells,
            goal_snap_radius_m=self.goal_snap_radius_m)

        if path is None:
            # NEVER forward the raw NBV goal (that would resurrect the
            # local-minimum bug this node exists to fix). Keep following
            # the previous valid path if one survives; report the goal as
            # unreachable after enough consecutive failures so NBV
            # blacklists it and moves on (breaks the wall-target freeze).
            self.fail_counter += 1
            self.get_logger().warn(
                f'Theta* planning failed ({self.fail_counter}/'
                f'{UNREACHABLE_FAIL_THRESHOLD}): no valid path to '
                f'({self.nbv_goal[0]:.2f}, {self.nbv_goal[1]:.2f}).',
                throttle_duration_sec=3.0)
            if self.fail_counter == UNREACHABLE_FAIL_THRESHOLD:
                msg = PoseStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.map_frame
                msg.pose.position.x = float(self.nbv_goal[0])
                msg.pose.position.y = float(self.nbv_goal[1])
                msg.pose.position.z = float(self.nbv_goal[2])
                msg.pose.orientation.w = 1.0
                self.unreachable_pub.publish(msg)
                self.get_logger().warn(
                    'Goal reported unreachable - NBV will blacklist it.')
            return

        self.fail_counter = 0
        self.current_path = path
        self._path_goal = self.nbv_goal.copy()
        # path[0] is the drone's own (snapped) start cell - it can sit just
        # outside waypoint_tolerance, so publishing it yanks the local
        # planner's heading backward on every replan (live 2026-07-11:
        # target flip-flopped between start cell and waypoint 1 every
        # replan; with turn-then-go the drone oscillated in place for 2+
        # minutes until the NBV target timed out). Hand off from the first
        # real waypoint instead.
        self.path_index = 1 if len(path) > 1 else 0
        self._pending_fresh_publish = True
        self.get_logger().info(
            f'Theta* path found: {len(path)} waypoints, target '
            f'({self.nbv_goal[0]:.2f}, {self.nbv_goal[1]:.2f}, '
            f'{self.nbv_goal[2]:.2f}).')
        self._publish_path_viz()

    # ------------------------------------------------------------------
    # Waypoint hand-off
    # ------------------------------------------------------------------

    def advance_and_publish(self):
        if self.current_path is None or self.path_index >= len(self.current_path):
            return
        pose = self.get_current_pose()
        if pose is None:
            return

        # XY-ONLY advance check (see module docstring: with goal-z
        # annotated waypoints, a 3D check could never clear a waypoint the
        # drone is standing on in xy while still climbing).
        #
        # Skipped on the first tick of a fresh path (_pending_fresh_publish):
        # otherwise, if the drone's current position already happens to be
        # within waypoint_tolerance of path[path_index] (common - many NBV
        # targets are close, e.g. a 2-waypoint direct path whose only real
        # waypoint sits near the drone's own start position), this check
        # fires immediately, advances path_index straight past the end of
        # the path, and returns having NEVER called _publish_waypoint - a
        # silent deadlock (live 2026-07-11: NBV/theta* looped "path found"
        # every 3s forever, local_planner stuck on "Waiting for goal..."
        # the entire time). A fresh path's handoff target must always be
        # published at least once before arrival is judged against it.
        target_xy = self.current_path[self.path_index]
        if not self._pending_fresh_publish and math.hypot(
                pose[0] - target_xy[0], pose[1] - target_xy[1]) < self.waypoint_tolerance:
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                self.get_logger().info(
                    'End of Theta* path reached.', throttle_duration_sec=5.0)
                return
            target_xy = self.current_path[self.path_index]
        self._pending_fresh_publish = False

        target = np.array([target_xy[0], target_xy[1], self.nbv_goal[2]])
        if (self._last_published_target is None or
                float(np.linalg.norm(target - self._last_published_target)) > REPUBLISH_EPSILON):
            self._publish_waypoint(target)
            self._last_published_target = target.copy()

    def _publish_waypoint(self, target_xyz):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = float(target_xyz[0])
        msg.pose.position.y = float(target_xyz[1])
        msg.pose.position.z = float(target_xyz[2])
        msg.pose.orientation.w = 1.0
        self.waypoint_pub.publish(msg)

    def _publish_path_viz(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame
        goal_z = float(self.nbv_goal[2]) if self.nbv_goal is not None else 0.0
        for wp in self.current_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = goal_z  # render at flight altitude
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ThetaStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
