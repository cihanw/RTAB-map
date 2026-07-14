#!/usr/bin/env python3
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

# theta_star_planner.py (added 2026-07-09) subscribes to this topic AFTER nbv_planner
# - with default "volatile" QoS, as long as current_target does not change
# it is NOT published again (see planning_loop below: if target is still valid
# return early, NO republishing), a late-subscribing node could completely miss
# the first target and lock the system (verified in live test).
# TRANSIENT_LOCAL delivers the last published message to new subscribers as well
# - the "latched topic" pattern, the ROS2 QoS equivalent of ROS1 latching.
TARGET_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)

class NBVPlanner(Node):
    def __init__(self):
        super().__init__(
            'nbv_planner',
            parameter_overrides=[Parameter('use_sim_time', value=True)])
        
        self.declare_parameter('voxel_size', 0.2)
        # max_target_distance limit REMOVED (user request): the drone can
        # now freely explore any distant region of the room.
        # NOTE/risk (knowingly accepted): local_planner is purely reactive (APF),
        # Theta* (global planner) is not present yet - risk of getting stuck in
        # local minimum theoretically increases for very distant/complex targets.
        # If observed, Theta* will be added (see tasks/todo.md).
        # 1.2 -> 2.5: in live test the drone climbed up to z=2.8 and the objects
        # we placed on the floor (statue/vehicle/photo panel, all in z=0-1.5 range)
        # remained outside the camera's (horizontally mounted, no tilt) vertical FOV
        # - height is now much more expensive, preference for going straight
        # is strengthened.
        # 2.5 -> 4.0 (2026-07-08, live observation): after information gain was added,
        # the drone climbed up to z~3.9m and got STUCK there - at high altitude
        # the repeating ceiling/roof structure provided few/ambiguous features,
        # constantly breaking VO, AND (ironically) appeared as a large "unknown"
        # space getting a high information gain score, beating the z_penalty (2.5)
        # and pulling the drone there - for 6 minutes not even a new target was
        # published (although active recovery rotation recovered in a few seconds
        # each time, it would immediately lose it again in the same place). By
        # increasing to 4.0, it is harder for the info-gain term to overcome the
        # height penalty.
        self.declare_parameter('z_penalty', 4.0)
        # Information gain scoring (2026-07-08, user request):
        # in live test the drone always stayed in the small region where it STARTED,
        # and never visited the rest of the room (e.g. the side with the standing_person statue).
        # Root cause: scoring ONLY looked at distance - the closest frontier is
        # always considered the cheapest, but the closest frontier might just be
        # a THIN strip right at the edge of the already explored "bubble"
        # (e.g. a single voxel stuck in a corner), whereas a much further frontier
        # might be a candidate to open up a whole unexplored room.
        # Solution: for each candidate, calculate the number of UNKNOWN cells in its
        # immediate vicinity - if this number is large, that frontier opens up to a truly wide/new region.
        # This value is proportionally SUBTRACTED from the score
        # (lower score = better candidate).
        #
        # FIX (2026-07-08, live observation - 2nd bug): the first version used
        # a 3D CUBE neighborhood (including up/down). This was successful in
        # SPREADING the drone across the room's width BUT had an unwanted side effect:
        # the warehouse CEILING/upper volume is massively unexplored (camera is
        # horizontal, almost never looks there), meaning candidates ABOVE were also
        # automatically considered "opening to wide unknown region" and got high scores
        # - the drone repeatedly climbed towards the ceiling, lost tracking there
        # (repeating brick/roof structure = few features) and the map drifted/corrupted
        # (ghost objects, warped regions). Solution: instead of a cube, use ONLY a
        # HORIZONTAL disk (with info_gain_radius, at FIXED z=candidate's own altitude)
        # - a candidate is NOW NEVER rewarded for unexplored volume ABOVE/BELOW it,
        # only for horizontal extent in its OWN layer.
        # This preserves the horizontal-spreading benefit while fundamentally
        # eliminating the vertical-climbing urge. Since cell count dropped from
        # cube's (2r+1)^3 to disk's (2r+1)^2, the weight was proportionally increased
        # (about 5x, to maintain the same effect magnitude).
        self.declare_parameter('info_gain_radius', 2)
        self.declare_parameter('info_gain_weight', 1.0)
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        # Anti-stuck (observed in Phase 1 live test: drone was going back and
        # forth between small targets in one corner of the room for a long time - greedy
        # nearest-frontier selection made small new frontiers in the local region
        # always score "cheaper" than distant/unexplored regions, so it never went there).
        # By adding a penalty to candidates close to the last N targets,
        # exploration is FORCED to different regions.
        self.declare_parameter('revisit_radius', 1.5)
        # 5.0 -> 1.0 -> 0.0 (2026-07-10, user request, deliberate trade-off):
        # the loop-closure investigation found that exploration NEVER
        # produces a surviving loop closure - near-miss rejections (6-14/15
        # inliers) look like genuine revisits seen from a very different
        # angle, which is exactly what a nonzero revisit_penalty guarantees,
        # since it actively pushes the drone AWAY from anywhere near its own
        # recent path. Dropped to 0.0 (fully disabled) to test the
        # hypothesis that it's no longer needed at all: this parameter was
        # ORIGINALLY added (see comment above) when info_gain_weight was
        # still 0.25, which mathematically couldn't outweigh distance to
        # genuinely far rooms - THAT flaw is what caused the original
        # "stuck in one corner" bug, not the absence of a revisit penalty
        # per se. info_gain_weight has since been strengthened to 1.0
        # (tasks/lessons.md, "Tuning Exploration Heuristics"), which may
        # already be enough on its own to keep exploration pushing outward.
        # NOT CONFIRMED YET - this exact combination (0.25->1.0 info_gain
        # fix + revisit_penalty=0) has not been live-tested before. If the
        # "cycles back to the same small area for a long time" symptom
        # reappears, revisit_penalty is the first knob to bring back
        # (start around 1.0-2.0, not the original 5.0).
        self.declare_parameter('revisit_penalty', 0.0)
        self.declare_parameter('revisit_history', 8)
        # SAME defaults as local_planner.py's parameters with the same names:
        # if local_planner's own altitude-safety clamp (min/max_altitude) conflicts
        # with the target chosen by NBV, the drone is constantly pushed towards a
        # target it can never reach (infinite tug-of-war). NBV therefore filters out
        # frontiers outside this range from the start when selecting them.
        self.declare_parameter('min_altitude', 0.3)
        # 5.0 -> 3.0 -> 1.5 (2026-07-08, live observation): after information gain
        # was added, the drone repeatedly got STUCK at z~2.8-3.9m (repeating
        # brick/roof structure) - even when lowered to 3.0, it continued to accumulate
        # JUST BELOW this limit (14 of 24 targets in z=2.6-3.0 range),
        # because info-gain (3D cube version) still overcame the soft z_penalty then.
        # ROOT CAUSE: ALL unique objects we added (statue/vehicle/photo panel)
        # are at ground level (z=0-1.5) - camera is horizontal/no tilt -
        # meaning there is absolutely NOTHING feature-rich ABOVE z=1.5,
        # only repeating structure. Vertical exploration is in principle
        # VALUELESS in this setup. Info-gain is now changed to horizontal disk
        # (comment above) BUT this hard ceiling (1.5m) is also preserved as
        # DEFENSE IN DEPTH - two independent mechanisms prevent the same
        # issue at different layers. (local_planner's own max_altitude is
        # still 5.0 - not a contradiction, it's safe for NBV to be MORE CONSERVATIVE,
        # if it were the other way around it would be a problem.) 1.5 -> 1.0
        # (2026-07-08, user request): to restrict exploration even more strictly
        # to the narrow band where objects ACTUALLY are (z=0-1.0).
        self.declare_parameter('max_altitude', 1.0)
        # Forward-motion enforcement added (2026-07-07) then REMOVED
        # (2026-07-08): greedy nearest-frontier selection + narrow camera FOV
        # led to a "rotation only" trap (constantly changing direction targets with no translation)
        # so a threshold `min_travel_distance` was added to restrict candidates to >=1.5m.
        # But this threshold pushed NBV to select risky frontiers right next to
        # objects (e.g. prius_hybrid) (safe/small steps nearby were filtered out,
        # forcing the drone to a further/riskier jump). Upon user request, the threshold
        # was lowered first to 1.0, then 0.5, then 0.1 - since 0.1 is SMALLER than
        # the goal_tolerance (0.5) default below, the dist_3d>=goal_tolerance condition
        # in the "valid" mask was already stricter than this threshold, so the parameter
        # was effectively unused. It was COMPLETELY REMOVED with user consent -
        # accepting the rotation-trap risk.
        # Pose jump detection (2026-07-08, tasks/tracking_loss_roadmap.md
        # Phase 0): early defense line on NBV side of the same mechanism in local_planner.
        # In live test, when EKF estimate drifted wildly during tracking loss
        # (e.g. z=-2.47m, BELOW floor), NBV calculated "nearest frontier" based on
        # this corrupted pose and published targets 14-24 METERS away, far outside
        # the warehouse limits - the drone crashed into an object and FLIPPED while
        # chasing these targets. The same fixed threshold
        # (SHARED default value with local_planner) is here too: if the new pose
        # deviates from the last reliable pose by more than this threshold, NO target
        # is calculated/published in that loop.
        self.declare_parameter('pose_jump_threshold', 1.0)
        # Accept-after-confirmation (2026-07-08, live observation): the single-threshold
        # design above correctly rejected CHAOTIC/random jumps (crash scenario)
        # but FOREVER rejected a CONSISTENT, persistent small drift (e.g. a real
        # SLAM/loop-closure correction) - because the reference (last_known_good_pose)
        # was never updated, the drone reached the target and "waited" there constantly.
        # Solution: compare consecutively rejected poses WITH EACH OTHER - if they
        # remain consistent/close to each other for N loops (not chaotic, TRULY a
        # new stable position), accept and update the reference.
        self.declare_parameter('pose_confirm_count', 3)
        # EXPERIMENTAL (2026-07-08, user request, TEMPORARY): For EKF vs embedded-VO
        # comparison, pose-jump detection + confirmation-mechanism is
        # COMPLETELY DISABLED - code NOT DELETED, only the below flag is False.
        # To revert: set to True (see full backup is also available,
        # backups/drone_sim_before_vo_experiment_2026-07-08/).
        self.declare_parameter('pose_jump_check_enabled', False)

        p = self.get_parameter
        self.voxel_size = p('voxel_size').value
        self.z_penalty = p('z_penalty').value
        self.info_gain_radius = p('info_gain_radius').value
        self.info_gain_weight = p('info_gain_weight').value
        self.goal_tolerance = p('goal_tolerance').value
        self.map_frame = p('map_frame').value
        self.base_frame = p('base_frame').value
        self.min_altitude = p('min_altitude').value
        self.max_altitude = p('max_altitude').value
        self.revisit_radius = p('revisit_radius').value
        self.revisit_penalty = p('revisit_penalty').value
        self.recent_targets = deque(maxlen=p('revisit_history').value)
        self.pose_jump_threshold = p('pose_jump_threshold').value
        self.pose_confirm_count = p('pose_confirm_count').value
        self.pose_jump_check_enabled = p('pose_jump_check_enabled').value
        self.last_known_good_pose = None
        self.pending_pose = None
        self.pending_count = 0

        self.blacklisted_targets = deque(maxlen=50)
        self.target_published_time = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # To pack voxel coordinates (x,y,z int) into a SINGLE sortable int64
        # key (see _voxel_keys/_is_known). This DOES the membership check in the
        # "known voxel set" with numpy searchsorted instead of Python set/tuple -
        # in live measurement on a 1.4M point cloud, the Python set/tuple approach
        # constantly maxed out a single core to 100-110% (see tasks/lessons.md),
        # this vectorized approach does the same operation at C level, WITHOUT a
        # Python loop running over N.
        self._VOXEL_BITS = 21  # +/- ~1M voxels/axis - more than enough for warehouse size
        self._VOXEL_OFFSET = 1 << (self._VOXEL_BITS - 1)
        self._VOXEL_MASK = (1 << self._VOXEL_BITS) - 1
        self._neighbor_offsets = np.array(
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
            dtype=np.int64)

        # Precomputed HORIZONTAL DISK neighborhood for information gain (see
        # __init__ comment, info_gain_radius) - ONLY x/y change, z offset is
        # ALWAYS 0 (same layer as candidate's OWN altitude). (2r+1)^2 offsets,
        # calculated ONCE and reused in every planning loop.
        # Reason for using disk NOT cube: a candidate should never be
        # rewarded for unexplored volume ABOVE/BELOW it
        # (see __init__ comment, ceiling-climbing bug).
        r = self.info_gain_radius
        offsets_1d = np.arange(-r, r + 1)
        gx, gy = np.meshgrid(offsets_1d, offsets_1d, indexing='ij')
        gz = np.zeros_like(gx)
        self._info_gain_offsets = np.stack(
            [gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.int64)

        num_neighbors = self._info_gain_offsets.shape[0]
        self._ray_offsets = np.zeros((num_neighbors, r, 3), dtype=np.int64)
        self._ray_valid = np.zeros((num_neighbors, r), dtype=bool)

        for i, offset in enumerate(self._info_gain_offsets):
            dist = max(abs(offset[0]), abs(offset[1]))
            if dist == 0:
                continue
            for step in range(1, dist + 1):
                self._ray_offsets[i, step - 1] = np.round(offset * (step / dist)).astype(np.int64)
                self._ray_valid[i, step - 1] = True

        self.empty_points = None
        self.obs_points = None
        
        self.current_target = None
        
        self.create_subscription(
            PointCloud2, '/rtabmap/octomap_empty_space', self.empty_callback, 1)
        self.create_subscription(
            PointCloud2, '/rtabmap/octomap_obstacles', self.obs_callback, 1)
        self.create_subscription(
            PoseStamped, '/theta_star/unreachable_target', self.unreachable_target_callback, TARGET_QOS)
            
        self.target_pub = self.create_publisher(PoseStamped, '/local_planner/current_target', TARGET_QOS)
        
        self.create_timer(2.0, self.planning_loop)
        self.get_logger().info('NBV planner started.')

    def empty_callback(self, msg):
        raw = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if raw.shape[0] == 0:
            self.empty_points = np.zeros((0, 3))
        else:
            self.empty_points = np.stack([raw['x'], raw['y'], raw['z']], axis=-1).astype(np.float64)

    def obs_callback(self, msg):
        raw = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if raw.shape[0] == 0:
            self.obs_points = np.zeros((0, 3))
        else:
            self.obs_points = np.stack([raw['x'], raw['y'], raw['z']], axis=-1).astype(np.float64)

    def unreachable_target_callback(self, msg):
        target = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.blacklisted_targets.append(target)
        if self.current_target is not None and float(np.linalg.norm(target - self.current_target)) < 1e-3:
            self.get_logger().warn(
                f'Current target ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) '
                f'marked unreachable by Theta*. Clearing target and blacklisting.')
            self.current_target = None

    def get_current_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            pos = t.transform.translation
            return np.array([pos.x, pos.y, pos.z])
        except tf2_ros.TransformException:
            return None

    def _voxel_keys(self, voxel_coords):
        """Packs (N,3) int voxel coordinates into sortable (N,) int64 keys."""
        x = (voxel_coords[:, 0].astype(np.int64) + self._VOXEL_OFFSET) & self._VOXEL_MASK
        y = (voxel_coords[:, 1].astype(np.int64) + self._VOXEL_OFFSET) & self._VOXEL_MASK
        z = (voxel_coords[:, 2].astype(np.int64) + self._VOXEL_OFFSET) & self._VOXEL_MASK
        return (x << (2 * self._VOXEL_BITS)) | (y << self._VOXEL_BITS) | z

    def _is_known(self, voxel_coords, sorted_known_keys):
        """Returns vectorized boolean if each row in voxel_coords is in sorted_known_keys."""
        keys = self._voxel_keys(voxel_coords)
        idx = np.searchsorted(sorted_known_keys, keys)
        idx = np.clip(idx, 0, len(sorted_known_keys) - 1)
        return sorted_known_keys[idx] == keys

    def _unpack_voxel_keys(self, keys):
        """Inverse of _voxel_keys: (N,3) int voxel coordinates from (N,) int64 keys."""
        z = (keys & self._VOXEL_MASK) - self._VOXEL_OFFSET
        y = ((keys >> self._VOXEL_BITS) & self._VOXEL_MASK) - self._VOXEL_OFFSET
        x = ((keys >> (2 * self._VOXEL_BITS)) & self._VOXEL_MASK) - self._VOXEL_OFFSET
        return np.stack([x, y, z], axis=-1)

    def _unique_voxels(self, raw_points):
        """Reduces raw (N,3) float points to unique (M,3) int voxel coordinates.
        Because np.unique(..., axis=0) does 2D row-based sorting (~1.5s live with 1.4M points)
        first PACK then 1D unique - 1D unique is a much faster numpy primitive
        (~0.2s live, ~7x speedup)."""
        if raw_points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        int_coords = np.round(raw_points / self.voxel_size).astype(np.int64)
        keys = np.unique(self._voxel_keys(int_coords))
        return self._unpack_voxel_keys(keys), keys

    def _compute_info_gain(self, voxel_coords, known_keys, obs_keys):
        """For each voxel, returns vectorized number of UNKNOWN cells (not in known_keys)
        in the info_gain_radius disk neighborhood, provided the ray to them is not blocked by obs_keys."""
        n = voxel_coords.shape[0]
        m = self._info_gain_offsets.shape[0]
        r = self.info_gain_radius

        neighborhood = voxel_coords[:, None, :] + self._info_gain_offsets[None, :, :]
        flat_neighborhood = neighborhood.reshape(-1, 3)
        known_dest = self._is_known(flat_neighborhood, known_keys).reshape(n, m)
        unknown_dest = ~known_dest

        rays = voxel_coords[:, None, None, :] + self._ray_offsets[None, :, :, :]
        flat_rays = rays.reshape(-1, 3)
        is_obs_flat = self._is_known(flat_rays, obs_keys).reshape(n, m, r)
        
        is_obs_valid = is_obs_flat & self._ray_valid[None, :, :]
        ray_blocked = is_obs_valid.any(axis=2)
        
        valid_cells = unknown_dest & ~ray_blocked
        return valid_cells.sum(axis=1)

    def planning_loop(self):
        if self.empty_points is None or self.obs_points is None:
            self.get_logger().info('Waiting for map data...', throttle_duration_sec=5.0)
            return

        pose = self.get_current_pose()
        if pose is None:
            self.get_logger().warn('Waiting for TF...', throttle_duration_sec=5.0)
            return

        # Pose jump check (see __init__ comment). Rejected poses
        # go through ACCEPT-AFTER-CONFIRMATION mechanism: if they remain
        # consistently close (not chaotic), they are accepted and reference is updated.
        # EXPERIMENTAL: when pose_jump_check_enabled=False this block is COMPLETELY skipped.
        if self.pose_jump_check_enabled and self.last_known_good_pose is not None:
            jump = float(np.linalg.norm(pose - self.last_known_good_pose))
            if jump > self.pose_jump_threshold:
                if (self.pending_pose is not None and
                        float(np.linalg.norm(pose - self.pending_pose)) <= self.pose_jump_threshold):
                    self.pending_count += 1
                else:
                    self.pending_pose = pose.copy()
                    self.pending_count = 1

                if self.pending_count < self.pose_confirm_count:
                    self.get_logger().warn(
                        f'Pose jump detected ({jump:.2f}m) — target not calculated '
                        f'this loop, waiting until pose stabilizes '
                        f'({self.pending_count}/{self.pose_confirm_count}).',
                        throttle_duration_sec=2.0)
                    return
                self.get_logger().info(
                    f'New pose has been stable for {self.pending_count} loops - '
                    f'accepted (diff: {jump:.2f}m).')
            self.pending_pose = None
            self.pending_count = 0
        self.last_known_good_pose = pose.copy()

        # 1. Voxelize known space - COMPLETELY vectorized (see _unique_voxels).
        # In live test /rtabmap/octomap_empty_space can contain 1.4M points;
        # the old Python set(map(tuple, ...)) approach trying to process this
        # directly constantly maxed out a single core to 100-110%.
        ev, empty_keys = self._unique_voxels(self.empty_points)
        if ev.shape[0] == 0:
            return

        ov, obs_keys = self._unique_voxels(self.obs_points)
        known_keys = empty_keys
        if ov.shape[0] > 0:
            known_keys = np.unique(np.concatenate([empty_keys, obs_keys]))

        # Check current target status
        if self.current_target is not None:
            dist_to_target = np.linalg.norm(pose - self.current_target)

            if dist_to_target < self.goal_tolerance:
                self.get_logger().info('Target reached, searching for new frontier.')
                self.current_target = None
            else:
                # Check if still a frontier
                tv = np.round(self.current_target / self.voxel_size).astype(np.int64).reshape(1, 3)
                is_frontier = False
                if self._is_known(tv, empty_keys)[0]:
                    neighbor_coords = tv + self._neighbor_offsets
                    is_frontier = not self._is_known(neighbor_coords, known_keys).all()
                if not is_frontier:
                    self.get_logger().info('Current target is no longer a frontier (explored/obstacle), searching for new target.')
                    self.current_target = None

        if self.current_target is not None:
            # Check for staleness timeout
            if self.target_published_time is not None:
                elapsed = (self.get_clock().now() - self.target_published_time).nanoseconds / 1e9
                if elapsed > 120.0:
                    self.get_logger().warn(f'Target stalled for {elapsed:.1f}s, dropping and blacklisting.')
                    self.blacklisted_targets.append(self.current_target)
                    self.current_target = None
            if self.current_target is not None:
                return  # Wait until current target is reached or explored

        # 2. Find frontiers - vectorized: 6 neighbor directions for N voxels, NO
        # Python loop over N, just 6 vectorized numpy comparisons.
        frontier_mask = np.zeros(ev.shape[0], dtype=bool)
        for offset in self._neighbor_offsets:
            neighbor_coords = ev + offset
            frontier_mask |= ~self._is_known(neighbor_coords, known_keys)

        frontier_voxels = ev[frontier_mask]
        if frontier_voxels.shape[0] == 0:
            self.get_logger().info('Exploration complete, no frontiers left!', throttle_duration_sec=10.0)
            return

        frontier_world = frontier_voxels.astype(np.float64) * self.voxel_size

        # 3. Score frontiers (vectorized). Altitude limit: if a target conflicting with
        # local_planner's min_altitude/max_altitude safety clamp is selected,
        # the drone is constantly pushed towards a point it can never reach (infinite tug-of-war)
        # - therefore frontiers outside the range are not even considered as candidates.
        dx = frontier_world[:, 0] - pose[0]
        dy = frontier_world[:, 1] - pose[1]
        dz = frontier_world[:, 2] - pose[2]
        dist_xy = np.hypot(dx, dy)
        dist_3d = np.hypot(dist_xy, dz)

        valid = (
            (dist_3d >= self.goal_tolerance) &
            (frontier_world[:, 2] >= self.min_altitude) &
            (frontier_world[:, 2] <= self.max_altitude)
        )

        if len(self.blacklisted_targets) > 0:
            blacklisted = np.array(self.blacklisted_targets)
            dists_to_blacklisted = np.linalg.norm(
                frontier_world[:, None, :] - blacklisted[None, :, :], axis=-1)
            is_blacklisted = dists_to_blacklisted.min(axis=1) < (self.voxel_size * 2)
            valid &= ~is_blacklisted

        if not np.any(valid):
            self.get_logger().info('No frontier found at suitable distance/altitude.', throttle_duration_sec=5.0)
            return

        # Information gain (see __init__ comment): calculate ONLY for valid candidates
        # - fewer candidates, cheaper.
        info_gain = self._compute_info_gain(frontier_voxels[valid], known_keys, obs_keys)
        scores = (dist_xy[valid] + self.z_penalty * np.abs(dz[valid])
                  - self.info_gain_weight * info_gain)

        # Anti-stuck: add penalty to candidates within revisit_radius of any of
        # the recently visited targets - this prevents constantly choosing the
        # "nearest" frontier in the same region with small steps and getting stuck
        # (observed in live test), forcing transitions to distant/unexplored regions.
        # NOT a hard elimination (soft penalty) - if there is really no other option,
        # the least bad candidate is still chosen.
        if len(self.recent_targets) > 0:
            recent_xy = np.array(self.recent_targets)[:, :2]
            cand_xy = frontier_world[valid][:, :2]
            dists_to_recent = np.linalg.norm(
                cand_xy[:, None, :] - recent_xy[None, :, :], axis=-1)
            near_recent = dists_to_recent.min(axis=1) < self.revisit_radius
            scores = scores + np.where(near_recent, self.revisit_penalty, 0.0)

        best_idx = np.argmin(scores)
        best_frontier = frontier_world[valid][best_idx]
        best_score = scores[best_idx]

        self.recent_targets.append(best_frontier.copy())
        self.current_target = best_frontier
        self.target_published_time = self.get_clock().now()
        msg = PoseStamped()
        msg.header.stamp = self.target_published_time.to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = float(best_frontier[0])
        msg.pose.position.y = float(best_frontier[1])
        msg.pose.position.z = float(best_frontier[2])
        msg.pose.orientation.w = 1.0
        self.target_pub.publish(msg)
        self.get_logger().info(f'New NBV target published: {best_frontier[0]:.2f}, {best_frontier[1]:.2f}, {best_frontier[2]:.2f} (Score: {best_score:.2f})')

def main(args=None):
    rclpy.init(args=args)
    node = NBVPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
