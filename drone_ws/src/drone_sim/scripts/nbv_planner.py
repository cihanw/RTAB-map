#!/usr/bin/env python3
"""NBV (Next-Best-View) frontier exploration planner.

Picks the next exploration target from the frontier set (known-empty voxels
bordering unknown space) and publishes it as a latched PoseStamped. The
global planner (theta_star_planner, 2D) routes to it; the local planner
(3D) flies the route.

REWRITE 2026-07-11 - scoring moved to the literature-standard NBV utility
(Gonzalez-Banos & Latombe; Bircher et al.):

    U(c) = gain(c) * exp(-lambda * d_xy(c))

replacing the old additive formula (distance + z_penalty*|dz|
- info_gain_weight*gain + revisit_penalty), whose four interacting knobs
had to be re-balanced by hand at least three times (tasks/lessons.md).
What the old knobs became:
  - z_penalty: DROPPED - candidates are hard-limited to z in
    [min_altitude, max_altitude] (a 0.7m band), a soft vertical cost is
    meaningless there, and the gain disk is horizontal anyway.
  - info_gain_weight: SUBSUMED - gain is now the multiplicand, not an
    additive bonus fighting a distance term on an arbitrary scale.
  - revisit_penalty/radius/history: DROPPED as structurally redundant -
    already-visited areas have gain ~ 0, so their utility ~ 0 regardless
    of proximity (this was live-tested at penalty 0.0 before committing).

This node stays fully 3D on purpose (the architecture seam): frontier
detection and gain run on the octomap voxels, and the altitude band
[0.3, 1.0] is a load-bearing feature-band defense (all visually distinctive
objects sit at z=0-1.5; the camera is horizontal). Only the global planner
went 2D. The occlusion raycast deliberately does NOT reuse the global
planner's 2D grid: that grid is inflated by 0.6m (a planning-safety
concept that would wrongly zero the gain of legitimate near-wall
frontiers), and projection destroys the per-z-layer visibility answer.

Vectorization discipline (tasks/lessons.md - a 1.4M-point cloud once pegged
a core via Python set/tuple membership): packed-int64 keys + np.unique +
np.searchsorted for ALL batch membership tests; no Python loops over N.
"""
import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

# "Latched topic" profile. This node publishes ONLY when the target changes
# (planning_loop returns early while a target is live) - a late subscriber
# (theta_star_planner starts 2s after this node) with volatile QoS would
# miss the first target and deadlock the whole stack (verified live).
TARGET_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)

# Unreachable-region blacklist: bounded so ancient entries eventually age
# out; entries are XY-only (see blacklist_radius_m declaration) and the
# matching radius is a declared parameter, not derived from voxel_size.
# 50 -> 200 (2026-07-17, hardware/freeze-frequency incident): a 45-
# unreachable-event run wrapped this deque ~1x, aging out and re-trying
# still-dead regions late in the run - directly adding to freeze
# frequency on top of the stale-map-driven Theta* failures (see
# GridGlobal/UpdateError fix in slam.launch.py). 200 gives headroom for
# a full exploration run without re-trying recently-dead regions.
BLACKLIST_MAXLEN = 200


class NBVPlanner(Node):

    def __init__(self):
        # Sim-time stamped TF/clouds; use_sim_time is an rclpy built-in ->
        # parameter_overrides, not declare_parameter.
        super().__init__(
            'nbv_planner',
            parameter_overrides=[Parameter('use_sim_time', value=True)])

        # Must equal theta_star_planner's grid_resolution (0.2): frontier
        # targets are snapped to this resolution, and the global planner
        # converts them back with the same rounding - a mismatch creates a
        # systematic offset between "where NBV thinks the frontier is" and
        # "where the planner thinks the goal cell is".
        self.declare_parameter('voxel_size', 0.2)
        # Gain-disk radius in voxels: gain counts unknown cells in the
        # (2r+1)^2 HORIZONTAL disk at the candidate's own altitude. Disk,
        # never a cube: the cube version rewarded the giant unexplored
        # ceiling volume and repeatedly lured the drone up to z~3-4m where
        # the repetitive roof structure broke visual odometry (live
        # incident, 2026-07-08). Vertical unknown must NEVER count as gain.
        self.declare_parameter('info_gain_radius', 2)
        # The single exploration knob. U = gain * exp(-lambda * d_xy).
        # Sanity math for this room scale (10-25m), the multiplicative
        # analogue of the additive-scale rule in tasks/lessons.md: a far
        # open room (gain 25 @ 15m) must beat a near scrap (gain 5 @ 2m),
        # which requires lambda < ln(5)/13 ~ 0.124. At 0.10 the far room
        # wins with ~35% margin; equal gains always resolve to the nearer
        # candidate; +10m needs ~2.7x gain, +20m needs ~7.4x. Half-utility
        # distance = ln(2)/lambda ~ 6.9m. Tuning rule: lambda ~
        # ln(g_far/g_near) / delta_d at the desired indifference point;
        # larger environments -> smaller lambda (useful range ~0.05-0.12).
        self.declare_parameter('distance_lambda', 0.10)
        # Arrival radius. Must stay STRICTLY GREATER than
        # theta_star_planner's goal_snap_radius_m (0.4): the planner may
        # legally park the drone up to that far from the true frontier
        # (wall-adjacent goals get snapped out of inflation), and arrival
        # must still be detectable here, else the target is held forever.
        self.declare_parameter('goal_tolerance', 0.5)
        # Blacklist exclusion radius, XY-only (2026-07-13, region-based
        # fix - see the masking block below for the full rationale).
        # Matches theta_star_planner's inflation_radius_m by convention
        # (same physical scale: how far the global planner treats a wall
        # as "occupied") - NOT enforced cross-node, keep them in sync
        # manually if either changes.
        self.declare_parameter('blacklist_radius_m', 0.55)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        # Hard candidate altitude band. Shares its name and floor with
        # local_planner's geofence on purpose; the ceiling is deliberately
        # far more conservative (1.0 vs 5.0) - the feature-band defense.
        # A target outside the local planner's own clamp would cause an
        # infinite tug-of-war, so candidates outside the band are never
        # even scored.
        self.declare_parameter('min_altitude', 0.3)
        self.declare_parameter('max_altitude', 1.0)
        # How often the planning loop runs (was a hardcoded 2.0s timer).
        self.declare_parameter('planning_period_sec', 2.0)
        # A target neither reached nor invalidated for this long is dropped
        # AND blacklisted (was hardcoded 120s). Safety net behind the
        # faster theta_star unreachable-report (~15s); catches any future
        # deadlock variant regardless of cause.
        self.declare_parameter('target_timeout_sec', 120.0)

        p = self.get_parameter
        self.voxel_size = p('voxel_size').value
        self.info_gain_radius = p('info_gain_radius').value
        self.distance_lambda = p('distance_lambda').value
        self.goal_tolerance = p('goal_tolerance').value
        self.blacklist_radius_m = p('blacklist_radius_m').value
        self.map_frame = p('map_frame').value
        self.base_frame = p('base_frame').value
        self.min_altitude = p('min_altitude').value
        self.max_altitude = p('max_altitude').value
        self.target_timeout_sec = p('target_timeout_sec').value

        # --- packed-voxel-key machinery (see module docstring) ---
        self._VOXEL_BITS = 21  # +/- ~1M voxels/axis
        self._VOXEL_OFFSET = 1 << (self._VOXEL_BITS - 1)
        self._VOXEL_MASK = (1 << self._VOXEL_BITS) - 1
        self._neighbor_offsets = np.array(
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
             (0, -1, 0), (0, 0, 1), (0, 0, -1)],
            dtype=np.int64)

        # Horizontal gain disk (z offset ALWAYS 0 - see info_gain_radius
        # comment) + per-cell occlusion rays: for each disk cell, the
        # integer ray steps from the candidate toward it. An unknown cell
        # only counts as gain if no ray step hits an obstacle voxel -
        # unknown space hidden BEHIND a wall is unobservable and must not
        # attract the drone (the wall-frontier bug: unknown cells on the
        # far side of depot walls inflated wall-adjacent frontier scores).
        r = self.info_gain_radius
        offsets_1d = np.arange(-r, r + 1)
        gx, gy = np.meshgrid(offsets_1d, offsets_1d, indexing='ij')
        gz = np.zeros_like(gx)
        self._info_gain_offsets = np.stack(
            [gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.int64)

        num_cells = self._info_gain_offsets.shape[0]
        self._ray_offsets = np.zeros((num_cells, r, 3), dtype=np.int64)
        self._ray_valid = np.zeros((num_cells, r), dtype=bool)
        for i, offset in enumerate(self._info_gain_offsets):
            dist = max(abs(offset[0]), abs(offset[1]))
            if dist == 0:
                continue
            for step in range(1, dist + 1):
                self._ray_offsets[i, step - 1] = np.round(
                    offset * (step / dist)).astype(np.int64)
                self._ray_valid[i, step - 1] = True

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.empty_points = None
        self.obs_points = None
        self.current_target = None
        self.target_published_time = None
        self.blacklisted_regions = deque(maxlen=BLACKLIST_MAXLEN)  # xy only

        self.create_subscription(
            PointCloud2, '/rtabmap/octomap_empty_space',
            self.empty_callback, 1)
        self.create_subscription(
            PointCloud2, '/rtabmap/octomap_obstacles',
            self.obs_callback, 1)
        self.create_subscription(
            PoseStamped, '/theta_star/unreachable_target',
            self.unreachable_target_callback, TARGET_QOS)

        self.target_pub = self.create_publisher(
            PoseStamped, '/local_planner/current_target', TARGET_QOS)

        self.create_timer(p('planning_period_sec').value, self.planning_loop)
        self.get_logger().info('NBV planner started.')

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def empty_callback(self, msg):
        raw = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if raw.shape[0] == 0:
            self.empty_points = np.zeros((0, 3))
        else:
            self.empty_points = np.stack(
                [raw['x'], raw['y'], raw['z']], axis=-1).astype(np.float64)

    def obs_callback(self, msg):
        raw = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if raw.shape[0] == 0:
            self.obs_points = np.zeros((0, 3))
        else:
            self.obs_points = np.stack(
                [raw['x'], raw['y'], raw['z']], axis=-1).astype(np.float64)

    def unreachable_target_callback(self, msg):
        """theta_star_planner reports a goal it failed to reach 3 times in a
        row. Blacklist it; if it is the current target, drop it immediately
        so the next planning cycle picks something else (breaks the
        wall-frontier freeze observed live: 20 minutes stuck)."""
        target = np.array([msg.pose.position.x,
                           msg.pose.position.y,
                           msg.pose.position.z])
        # Blacklist XY ONLY (2026-07-13, region-based fix - see
        # blacklist_radius_m comment): the identity check just below still
        # needs the full xyz target (it is answering "is this the exact
        # target I currently hold", not a spatial-region question).
        self.blacklisted_regions.append(target[:2].copy())
        if (self.current_target is not None and
                float(np.linalg.norm(target - self.current_target)) < 1e-3):
            self.get_logger().warn(
                f'Current target ({target[0]:.2f}, {target[1]:.2f}, '
                f'{target[2]:.2f}) marked unreachable by Theta*. '
                f'Clearing and blacklisting.')
            self.current_target = None

    def get_current_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            pos = t.transform.translation
            return np.array([pos.x, pos.y, pos.z])
        except tf2_ros.TransformException:
            return None

    # ------------------------------------------------------------------
    # Voxel-key machinery (batch membership at C speed)
    # ------------------------------------------------------------------

    def _voxel_keys(self, voxel_coords):
        """Packs (N,3) int voxel coordinates into sortable (N,) int64 keys."""
        x = (voxel_coords[:, 0].astype(np.int64) + self._VOXEL_OFFSET) & self._VOXEL_MASK
        y = (voxel_coords[:, 1].astype(np.int64) + self._VOXEL_OFFSET) & self._VOXEL_MASK
        z = (voxel_coords[:, 2].astype(np.int64) + self._VOXEL_OFFSET) & self._VOXEL_MASK
        return (x << (2 * self._VOXEL_BITS)) | (y << self._VOXEL_BITS) | z

    def _unpack_voxel_keys(self, keys):
        z = (keys & self._VOXEL_MASK) - self._VOXEL_OFFSET
        y = ((keys >> self._VOXEL_BITS) & self._VOXEL_MASK) - self._VOXEL_OFFSET
        x = ((keys >> (2 * self._VOXEL_BITS)) & self._VOXEL_MASK) - self._VOXEL_OFFSET
        return np.stack([x, y, z], axis=-1)

    def _is_known(self, voxel_coords, sorted_keys):
        """(N,3) coords -> (N,) bool membership in sorted_keys, vectorized."""
        keys = self._voxel_keys(voxel_coords)
        idx = np.searchsorted(sorted_keys, keys)
        idx = np.clip(idx, 0, len(sorted_keys) - 1)
        return sorted_keys[idx] == keys

    def _unique_voxels(self, raw_points):
        """(N,3) float points -> unique (M,3) int voxels + (M,) sorted keys.
        Pack-then-1D-unique instead of np.unique(axis=0) row sorting
        (~7x faster measured live on 1.4M points)."""
        if raw_points.shape[0] == 0:
            return (np.zeros((0, 3), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64))
        int_coords = np.round(raw_points / self.voxel_size).astype(np.int64)
        keys = np.unique(self._voxel_keys(int_coords))
        return self._unpack_voxel_keys(keys), keys

    # ------------------------------------------------------------------
    # Gain
    # ------------------------------------------------------------------

    def _compute_gain(self, voxel_coords, known_keys, obs_keys):
        """Occlusion-aware information gain, the SOLE gain definition:
        for each candidate, the number of unknown cells in its horizontal
        disk whose straight-line ray from the candidate crosses no obstacle
        voxel. Fully vectorized over candidates x disk cells x ray steps."""
        n = voxel_coords.shape[0]
        m = self._info_gain_offsets.shape[0]
        r = self.info_gain_radius

        neighborhood = voxel_coords[:, None, :] + self._info_gain_offsets[None, :, :]
        unknown_dest = ~self._is_known(
            neighborhood.reshape(-1, 3), known_keys).reshape(n, m)

        rays = voxel_coords[:, None, None, :] + self._ray_offsets[None, :, :, :]
        is_obs = self._is_known(
            rays.reshape(-1, 3), obs_keys).reshape(n, m, r)
        ray_blocked = (is_obs & self._ray_valid[None, :, :]).any(axis=2)

        return (unknown_dest & ~ray_blocked).sum(axis=1)

    # ------------------------------------------------------------------
    # Planning loop
    # ------------------------------------------------------------------

    def planning_loop(self):
        if self.empty_points is None or self.obs_points is None:
            self.get_logger().info(
                'Waiting for map data...', throttle_duration_sec=5.0)
            return
        pose = self.get_current_pose()
        if pose is None:
            self.get_logger().warn(
                'Waiting for TF...', throttle_duration_sec=5.0)
            return

        # Voxelize known space (empty + obstacle = known).
        ev, empty_keys = self._unique_voxels(self.empty_points)
        if ev.shape[0] == 0:
            return
        ov, obs_keys = self._unique_voxels(self.obs_points)
        known_keys = empty_keys
        if ov.shape[0] > 0:
            known_keys = np.unique(np.concatenate([empty_keys, obs_keys]))

        # --- Current-target lifecycle: three exits, else hold. ---
        # (Publish-on-change contract: while a target is live this loop
        # returns early and publishes nothing - the latched QoS exists
        # precisely so late subscribers still get the current target.)
        if self.current_target is not None:
            if np.linalg.norm(pose - self.current_target) < self.goal_tolerance:
                self.get_logger().info(
                    'Target reached, searching for new frontier.')
                self.current_target = None
            else:
                tv = np.round(self.current_target / self.voxel_size).astype(
                    np.int64).reshape(1, 3)
                is_frontier = False
                if self._is_known(tv, empty_keys)[0]:
                    is_frontier = not self._is_known(
                        tv + self._neighbor_offsets, known_keys).all()
                if not is_frontier:
                    self.get_logger().info(
                        'Current target is no longer a frontier '
                        '(explored/obstacle), searching for new target.')
                    self.current_target = None

        if self.current_target is not None:
            elapsed = (self.get_clock().now()
                       - self.target_published_time).nanoseconds / 1e9
            if elapsed > self.target_timeout_sec:
                self.get_logger().warn(
                    f'Target stalled for {elapsed:.1f}s, dropping and '
                    f'blacklisting.')
                self.blacklisted_regions.append(self.current_target[:2].copy())
                self.current_target = None
            else:
                return  # hold current target

        # --- Frontier detection: known-empty voxel with >=1 unknown
        # 6-neighbor. Six vectorized passes, no Python loop over N. ---
        frontier_mask = np.zeros(ev.shape[0], dtype=bool)
        for offset in self._neighbor_offsets:
            frontier_mask |= ~self._is_known(ev + offset, known_keys)
        frontier_voxels = ev[frontier_mask]
        if frontier_voxels.shape[0] == 0:
            self.get_logger().info(
                'Exploration complete, no frontiers left!',
                throttle_duration_sec=10.0)
            return

        frontier_world = frontier_voxels.astype(np.float64) * self.voxel_size

        # --- Candidate validity: not-already-here, altitude band (hard
        # feature-band ceiling), not blacklisted. ---
        d_xy = np.hypot(frontier_world[:, 0] - pose[0],
                        frontier_world[:, 1] - pose[1])
        d_3d = np.hypot(d_xy, frontier_world[:, 2] - pose[2])
        valid = (
            (d_3d >= self.goal_tolerance) &
            (frontier_world[:, 2] >= self.min_altitude) &
            (frontier_world[:, 2] <= self.max_altitude)
        )
        # REGION-based, XY-ONLY exclusion (2026-07-13): theta_star_planner
        # is a fully 2D planner (search happens entirely on the
        # xy-projected grid; z is only ever tacked on afterward as the
        # flight altitude). Reachability at a given (x,y) is therefore
        # IDENTICAL for every z - if theta* fails at z=0.4 it is
        # GUARANTEED to also fail at z=0.6/0.8/1.0, since the search never
        # even looks at z. The old check compared full 3D distance at a
        # tiny voxel_size*2 (0.4m) radius: it mixed z-spacing into a
        # decision that has nothing to do with z, AND was narrow enough
        # that a same-(x,y) candidate exactly one z-step away (0.4m) sat
        # right on the boundary and slipped through unblacklisted - live-
        # observed 2026-07-13: (-5.80,7.60,0.40) blacklisted, next
        # candidate tried was (-5.80,7.60,0.80), failed identically 3s
        # later, then (-2.80,7.60,x) tried and failed too - an 80s+ freeze
        # cycling near-duplicate points in the same unreachable pocket.
        if len(self.blacklisted_regions) > 0:
            blacklisted_xy = np.array(self.blacklisted_regions)
            bl_dist = np.linalg.norm(
                frontier_world[:, None, :2] - blacklisted_xy[None, :, :],
                axis=-1)
            valid &= ~(bl_dist.min(axis=1) < self.blacklist_radius_m)

        if not np.any(valid):
            # Blacklist amnesty (2026-07-11 endurance run): a wedged/stuck
            # phase can blacklist every frontier in reach, after which this
            # branch repeated silently every cycle for 2.5 HOURS. If the
            # blacklist is what emptied the candidate set, clear it once
            # and retry next cycle - genuinely unreachable targets will
            # simply re-earn their blacklisting (15s each via the
            # unreachable loop), while recoverable ones get a second
            # chance. Only if the set is empty WITHOUT any blacklist help
            # is exploration actually out of frontiers.
            if len(self.blacklisted_regions) > 0:
                self.get_logger().warn(
                    f'No valid frontier candidates - clearing blacklist '
                    f'({len(self.blacklisted_regions)} entries) for a '
                    f'second chance.')
                self.blacklisted_regions.clear()
            else:
                self.get_logger().info(
                    'No frontier candidates anywhere (blacklist already '
                    'empty) - exploration appears COMPLETE for the '
                    'reachable map.', throttle_duration_sec=30.0)
            return

        # --- Utility: U = gain * exp(-lambda * d_xy). ---
        cand_world = frontier_world[valid]
        cand_d_xy = d_xy[valid]
        gain = self._compute_gain(frontier_voxels[valid], known_keys, obs_keys)

        if gain.max() == 0:
            # All remaining unknown is occluded from every candidate -
            # increasingly common late in exploration BECAUSE of the
            # occlusion-aware gain. Do not declare completion (occluded
            # unknown is not unmappable): move to the NEAREST candidate,
            # which changes the viewpoint and typically de-occludes
            # something; the staleness/unreachable loops guarantee escape.
            best_idx = int(np.argmin(cand_d_xy))
            self.get_logger().info(
                'All candidate gains are zero (occluded) - falling back to '
                'nearest frontier.')
        else:
            utility = gain * np.exp(-self.distance_lambda * cand_d_xy)
            best_idx = int(np.argmax(utility))

        best = cand_world[best_idx]
        self.current_target = best
        self.target_published_time = self.get_clock().now()

        msg = PoseStamped()
        msg.header.stamp = self.target_published_time.to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = float(best[0])
        msg.pose.position.y = float(best[1])
        msg.pose.position.z = float(best[2])
        msg.pose.orientation.w = 1.0
        self.target_pub.publish(msg)
        # gain/distance/utility logged separately so lambda is tunable
        # from live logs (one opaque "score" hid this before).
        u = gain[best_idx] * math.exp(-self.distance_lambda * cand_d_xy[best_idx])
        self.get_logger().info(
            f'New NBV target published: {best[0]:.2f}, {best[1]:.2f}, '
            f'{best[2]:.2f} (gain={int(gain[best_idx])}, '
            f'd={cand_d_xy[best_idx]:.1f}m, U={u:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = NBVPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
