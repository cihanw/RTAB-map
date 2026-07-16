#!/usr/bin/env python3
"""Reactive 3D local planner / controller.

Follows the waypoints published by the 2D global planner (theta_star_planner
via the goal_topic launch override) using an APF (artificial potential
field) reactive layer, and converts the resulting velocity into strict
TURN-THEN-GO unicycle commands.

Layers (each a small method; control_loop reads top to bottom):
  1. HEALTH  - odometry gate + hover/yaw-sweep recovery
  2. POSE    - TF lookup (map -> base_link)
  3. GOAL    - reached-latch lifecycle
  4. FORCES  - attractive (goal) + repulsive (obstacles), fully 3D
  5. SAFETY  - altitude geofence with active push-back
  6. MOTION  - turn-then-go state machine -> Twist

TURN-THEN-GO (user requirement, sacred): the drone has NO cornering ability
at all - not even car-like blended arcs. Rotation and translation are NEVER
commanded in the same Twist. A two-state machine with hysteresis alternates
pure in-place rotation and pure straight-line translation. Side benefit
measured live: pure-rotation / pure-translation segments are individually
easier for visual odometry than mixed motion (quality improved ~400-650 ->
700-770 when this replaced the blended cos-scaling model).

This node is fully 3D - the global planner is 2D and annotates every
waypoint with the goal's z; the attractive force here closes that z error
gradually along the route.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from scipy.spatial import cKDTree
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

# Goal subscription uses the latched profile: every publisher in the target
# chain (NBV and theta_star) is TRANSIENT_LOCAL, so a late-(re)started
# local_planner immediately receives the current waypoint instead of
# waiting for the next change. NOTE for manual debugging: `ros2 topic pub`
# to the goal topic must then pass --qos-durability transient_local or the
# subscription will not match.
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)

# Numerical dead-band (m/s), not a tuning knob: below this planar speed the
# APF direction is noise and the commanded heading must not chase it.
HEADING_UPDATE_MIN_SPEED = 0.05


class LocalPlanner(Node):

    def __init__(self):
        # TF data is sim-time stamped; wall clock would make every lookup an
        # "extrapolation into the past" error. use_sim_time is an rclpy
        # built-in -> parameter_overrides, not declare_parameter.
        super().__init__(
            'local_planner',
            parameter_overrides=[Parameter('use_sim_time', value=True)])

        self.declare_parameter('control_rate_hz', 10.0)
        # Kept low on purpose: gives odometry robustness margin and lets
        # mapping keep up with obstacles outside the camera's FOV.
        self.declare_parameter('max_linear_speed', 0.6)
        # VO-safety-tuned: at 1.0 rad/s odometry quality collapsed from
        # ~870 to 150-400 in live tests (60deg FOV camera). Do not raise
        # casually.
        self.declare_parameter('max_angular_speed', 0.35)
        self.declare_parameter('goal_tolerance', 0.3)
        # Repulsion only engages this close (1.5 -> 0.8 history: two
        # obstacles ~2m apart had overlapping 1.5m fields that sealed the
        # gap between them). 0.8 -> 1.0 (2026-07-12, close-proximity VO
        # death): frame-verified incident - the drone ended up pressed
        # against a roller door, camera filled with a blank close-up, VO
        # died WHILE STUCK (clipping caused the loss, not vice versa).
        # Earlier/stronger push-back margin. Gap-sealing risk at 1.0 is
        # far milder than the old 1.5 case: repulsion is now MEAN-
        # normalized (lesson 7), not a density-scaled sum.
        self.declare_parameter('influence_radius', 1.0)
        # Hard numerical floor for the repulsion magnitude - a safety
        # floor, not a "bravery" knob. 0.4 -> 0.5 (2026-07-12, same
        # incident): the camera needs ~0.4m (near-clip = real D455 min
        # range) to see ANYTHING - a 0.4m body floor let the lens get
        # close enough to blind itself. NOTE the ordering constraint:
        # theta_star's inflation_radius_m must stay >= this floor (raised
        # to 0.55 in the same change).
        self.declare_parameter('min_safe_distance', 0.5)
        # GNRON fade radius (see compute_repulsive_force): within this
        # distance of the current target, repulsion fades quadratically to
        # zero at the target so waypoints near obstacles are dockable.
        # Must be > waypoint_tolerance (0.5, theta_star side) or docking
        # still stalls at the equilibrium ring; > slow_radius (1.0) keeps
        # the fade fully covering the weak-attraction zone.
        self.declare_parameter('repulsion_fade_radius', 1.2)
        self.declare_parameter('k_att', 1.0)
        # 0.8 -> 1.0 after live "doesn't crash but too bold near obstacles"
        # observation; influence_radius deliberately untouched (narrow-gap
        # capability preserved, only push strength increased).
        self.declare_parameter('k_rep', 1.0)
        self.declare_parameter('k_yaw', 1.5)
        # Turn-then-go hysteresis. Two DIFFERENT thresholds on purpose: the
        # APF heading jitters a few degrees every cycle; a single threshold
        # chatters (align -> move -> 1deg drift -> stop -> realign ...).
        # Tight entry (~5deg) to start moving, loose exit (~15deg) to stop.
        self.declare_parameter('align_enter_threshold', 0.09)
        self.declare_parameter('align_exit_threshold', 0.26)
        # Attractive-force ramp radius (full speed beyond, linear inside).
        self.declare_parameter('slow_radius', 1.0)
        # OUTERMOST altitude geofence (NBV's own ceiling of 1.0 is the
        # conservative planner-side band; this is the last-resort clamp).
        self.declare_parameter('min_altitude', 0.3)
        self.declare_parameter('max_altitude', 5.0)
        # Geofence push-back speed. ACTIVE push, not just zeroing the
        # violating component - momentum overshot the bound when the push
        # was merely stopped (tested live). Was hardcoded 0.3.
        self.declare_parameter('geofence_recovery_speed', 0.3)
        # DEPTH GUARD (2026-07-16, after the third frame-verified wall-
        # contact incident): if the minimum valid depth in the camera's
        # forward ROI drops below this, forward translation is hard-zeroed
        # (rotation and vertical stay allowed - turning AWAY and climbing
        # are exactly the escape moves). Prevention-first per project
        # policy: this acts within one 30Hz frame, where the cloud->octomap
        # ->grid->APF chain reacted only seconds later. Note the camera
        # near-clip is 0.4 (real D455 min range), so readable values live
        # in [0.4, inf): 0.7 gives a 0.3m reaction band = ~0.5s at max
        # 0.6 m/s = 5 control ticks. Fail-open on stale/no depth data
        # (>0.5s old): guard silently inactive, old behavior preserved.
        self.declare_parameter('depth_guard_stop_m', 0.7)
        self.declare_parameter(
            'depth_topic',
            '/world/depot/model/drone/model/d455/link/link/sensor/'
            'realsense_d455/depth_image')
        self.declare_parameter('obstacle_topic', '/rtabmap/octomap_obstacles')
        # Overridden to /theta_star/next_waypoint in autonomous.launch.py.
        self.declare_parameter('goal_topic', '/local_planner/current_target')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_topic', '/rtabmap/odom')
        # Health gate: odometry is UNHEALTHY if covariance[0] exceeds this
        # (RTAB-Map publishes 9999 on tracking loss) ...
        self.declare_parameter('odom_cov_threshold', 1.0)
        # ... or nothing arrived for this long.
        self.declare_parameter('odom_timeout_sec', 1.5)
        # Anti-freeze (391s live incident: hovering stares at the same
        # feature-poor scene forever, VO never self-recovers): after this
        # long in recovery, rotate slowly in place to feed the camera new
        # content.
        self.declare_parameter('recovery_stall_threshold', 10.0)
        # ~43% of max_angular_speed: present new features GENTLY, not fast
        # enough to break VO further.
        self.declare_parameter('recovery_yaw_rate', 0.15)
        # POSE-TRUST GUARDS (2026-07-12, phantom-altitude incident): during
        # a VO flap storm (102 loss->restore cycles in one run), "no guess"
        # re-locks can place the pose ANYWHERE - the run showed believed
        # altitudes of 2-3m while every commanded waypoint was z<=0.8.
        # Acting on such poses either dives the drone into the floor
        # (believed high) or drives an unbounded physical climb via the
        # min-altitude geofence push (believed low/frozen). Two guards:
        # 1) sanity envelope: the mission's believable altitude range -
        #    a pose outside it is by definition a bad re-lock, never a
        #    real flight state (NBV band tops at 1.0, geofence at 5.0 is
        #    the actuator clamp, not a plausibility bound). Treat as
        #    unhealthy -> hover recovery until VO re-locks sanely.
        self.declare_parameter('pose_sane_z_min', 0.05)
        self.declare_parameter('pose_sane_z_max', 2.0)
        # 2) grace hover after ANY recovery exit: a freshly restored pose
        #    is the least trustworthy pose there is (it may BE the bad
        #    re-lock). Hover this long before resuming pursuit so
        #    odometry/graph corrections settle first.
        self.declare_parameter('recovery_grace_sec', 2.5)

        p = self.get_parameter
        self.control_rate_hz = p('control_rate_hz').value
        self.max_linear_speed = p('max_linear_speed').value
        self.max_angular_speed = p('max_angular_speed').value
        self.goal_tolerance = p('goal_tolerance').value
        self.influence_radius = p('influence_radius').value
        self.min_safe_distance = p('min_safe_distance').value
        self.repulsion_fade_radius = p('repulsion_fade_radius').value
        self.k_att = p('k_att').value
        self.k_rep = p('k_rep').value
        self.k_yaw = p('k_yaw').value
        self.align_enter_threshold = p('align_enter_threshold').value
        self.align_exit_threshold = p('align_exit_threshold').value
        self.slow_radius = p('slow_radius').value
        self.min_altitude = p('min_altitude').value
        self.max_altitude = p('max_altitude').value
        self.geofence_recovery_speed = p('geofence_recovery_speed').value
        self.depth_guard_stop_m = p('depth_guard_stop_m').value
        self.map_frame = p('map_frame').value
        self.base_frame = p('base_frame').value
        self.odom_cov_threshold = p('odom_cov_threshold').value
        self.odom_timeout_sec = p('odom_timeout_sec').value
        self.recovery_stall_threshold = p('recovery_stall_threshold').value
        self.recovery_yaw_rate = p('recovery_yaw_rate').value
        self.pose_sane_z_min = p('pose_sane_z_min').value
        self.pose_sane_z_max = p('pose_sane_z_max').value
        self.recovery_grace_sec = p('recovery_grace_sec').value
        self.grace_until_sec = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.obstacle_kdtree = None
        self.current_goal = None
        self.goal_reached = False
        self.last_odom_cov = None
        self.last_odom_time = None
        self.in_recovery = False
        self.recovery_reason = None
        self.recovery_start_time = None
        self.last_heading = 0.0
        # Starts in 'rotate' so even the very first target is faced before
        # any translation.
        self.motion_state = 'rotate'

        self.cmd_vel_pub = self.create_publisher(Twist, '/x500/cmd_vel', 10)
        self.debug_att_pub = self.create_publisher(
            Vector3, '/local_planner/debug/attractive_force', 10)
        self.debug_rep_pub = self.create_publisher(
            Vector3, '/local_planner/debug/repulsive_force', 10)

        self.create_subscription(
            PointCloud2, p('obstacle_topic').value, self.obstacle_callback, 10)
        self.create_subscription(
            PoseStamped, p('goal_topic').value, self.goal_callback,
            LATCHED_QOS)
        self.create_subscription(
            Odometry, p('odom_topic').value, self.odom_callback, 10)
        # DEPTH GUARD input (see depth_guard_stop_m). Raw depth frames at
        # 30Hz - the callback only slices a ROI and takes a min, cheap.
        self.min_forward_depth = float('inf')
        self.min_forward_depth_time = None
        self.create_subscription(
            Image, p('depth_topic').value, self.depth_callback, 1)

        self.create_timer(1.0 / self.control_rate_hz, self.control_loop)
        self.get_logger().info('Local planner started successfully.')

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def depth_callback(self, msg):
        """DEPTH GUARD (2026-07-16): fast forward-proximity sensing straight
        off the raw 32FC1 depth image, bypassing the slow cloud -> octomap
        -> grid pipeline whose latency let the drone reach physical wall
        contact three separate times (frame-verified incidents) before any
        map-based layer reacted. ROI = the upper-middle band of the frame:
        rows [h/4, h/2) so the floor (which legitimately reads < 1m in the
        lower rows at low flight altitudes) can never trigger it, cols
        [w/4, 3w/4) covering the direction of travel."""
        h, w = msg.height, msg.width
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        roi = depth[h // 4: h // 2, w // 4: 3 * w // 4]
        valid = roi[np.isfinite(roi) & (roi > 0.05)]
        self.min_forward_depth = float(valid.min()) if valid.size else float('inf')
        self.min_forward_depth_time = self.get_clock().now()

    def obstacle_callback(self, msg):
        raw = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if raw.shape[0] == 0:
            self.obstacle_kdtree = None
            return
        pts = np.stack(
            [raw['x'], raw['y'], raw['z']], axis=-1).astype(np.float64)
        self.obstacle_kdtree = cKDTree(pts)

    def goal_callback(self, msg):
        self.current_goal = np.array([msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z])
        self.goal_reached = False
        self.get_logger().info(
            f'New target received: ({self.current_goal[0]:.2f}, '
            f'{self.current_goal[1]:.2f}, {self.current_goal[2]:.2f})')

    def odom_callback(self, msg):
        self.last_odom_cov = msg.pose.covariance[0]
        self.last_odom_time = self.get_clock().now()

    # ------------------------------------------------------------------
    # 1. HEALTH
    # ------------------------------------------------------------------

    def is_odometry_healthy(self):
        if self.last_odom_time is None or self.last_odom_cov is None:
            return False
        if self.last_odom_cov > self.odom_cov_threshold:
            return False
        silence = (self.get_clock().now()
                   - self.last_odom_time).nanoseconds / 1e9
        return silence <= self.odom_timeout_sec

    def enter_recovery(self, reason, message):
        if not self.in_recovery or self.recovery_reason != reason:
            self.get_logger().warn(message)
            self.in_recovery = True
            self.recovery_reason = reason

    def clear_recovery(self, message):
        if self.in_recovery:
            self.get_logger().info(message)
            self.in_recovery = False
            self.recovery_reason = None
            self.recovery_start_time = None
            # Grace hover starts NOW (see pose-trust guard comments in
            # __init__): the freshly restored pose gets recovery_grace_sec
            # of hover before any pursuit command is issued on it.
            self.grace_until_sec = (
                self.get_clock().now().nanoseconds / 1e9
                + self.recovery_grace_sec)

    def publish_recovery_twist(self):
        """Hover; after recovery_stall_threshold, add a slow in-place yaw
        sweep. Rationale (391s live freeze): a hovering camera stares at
        the same feature-poor scene forever - VO has no reason to
        self-recover until it is shown something new."""
        now = self.get_clock().now()
        if self.recovery_start_time is None:
            self.recovery_start_time = now
        stalled = (now - self.recovery_start_time).nanoseconds / 1e9

        twist = Twist()
        if stalled > self.recovery_stall_threshold:
            twist.angular.z = self.recovery_yaw_rate
            self.get_logger().warn(
                f'Stuck for {stalled:.0f}s - applying active '
                f're-acquisition rotation.', throttle_duration_sec=3.0)
        self.cmd_vel_pub.publish(twist)

    # ------------------------------------------------------------------
    # 2. POSE
    # ------------------------------------------------------------------

    def get_current_pose(self):
        """Returns (xyz ndarray, yaw) or None."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except tf2_ros.TransformException as e:
            self.get_logger().warn(
                f'TF ({self.map_frame}->{self.base_frame}) not available: {e}',
                throttle_duration_sec=2.0)
            return None
        pos = t.transform.translation
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return np.array([pos.x, pos.y, pos.z]), yaw

    # ------------------------------------------------------------------
    # 4. FORCES
    # ------------------------------------------------------------------

    def compute_attractive_force(self, pose_xyz, goal_xyz):
        """Linear ramp toward the goal: full k_att beyond slow_radius,
        proportional inside, zero within goal_tolerance."""
        d_vec = goal_xyz - pose_xyz
        d = float(np.linalg.norm(d_vec))
        if d < self.goal_tolerance:
            return np.zeros(3), d
        scale = min(d, self.slow_radius) / self.slow_radius
        return self.k_att * (d_vec / d) * scale, d

    def compute_repulsive_force(self, pose_xyz, dist_to_goal):
        """Classic APF repulsion k_rep*(1/d - 1/rho0)/d^2 summed over
        obstacle points within influence_radius, with a hard distance
        floor. Vectorized over the neighbor set (no per-point Python
        loop - project discipline, see tasks/lessons.md).

        GNRON fix (Goals Non-Reachable with Obstacles Nearby, Ge & Cui
        2000): repulsion fades quadratically as the drone closes on its
        CURRENT target, reaching zero at the target itself. Without this,
        any waypoint near obstacles is physically undockable: Theta*'s
        any-angle paths hug the inflation boundary by construction, so
        waypoints sit ~inflation_radius from obstacles - inside the
        repulsion field - while the attractive ramp (slow_radius) weakens
        on approach. The force equilibrium then sits OUTSIDE
        waypoint_tolerance and the drone can never arrive (live
        2026-07-11: 127s dancing at an equilibrium ring around waypoint
        (6.6,-0.8) - rotate 10s / translate 1s / heading swings +-100 deg
        - until NBV's stall timeout). SAFETY: obstacle points at or
        inside min_safe_distance are exempt from fading - collision-range
        repulsion is always full strength no matter how close the goal.
        """
        if self.obstacle_kdtree is None:
            return np.zeros(3)
        idx = self.obstacle_kdtree.query_ball_point(
            pose_xyz, r=self.influence_radius)
        if not idx:
            return np.zeros(3)
        obs = self.obstacle_kdtree.data[idx]                 # (K,3)
        diff = pose_xyz[None, :] - obs                       # (K,3)
        dist = np.linalg.norm(diff, axis=1)                  # (K,)
        dist = np.maximum(dist, self.min_safe_distance)
        mag = self.k_rep * (1.0 / dist - 1.0 / self.influence_radius) / dist ** 2
        mag = np.maximum(mag, 0.0)                           # clip numeric negatives
        # Fade floor 0.2 -> 0 -> 0.2 RE-APPLIED (2026-07-12): briefly
        # dropped to 0 (plain GNRON fade) while chasing an unrelated
        # "behavior doesn't match baseline" mismatch, then restored once
        # the user pinned the exact pre-issue conversation moment and
        # confirmed 0.2 was already live there. Keeps SOME repulsion
        # active on final approach to any target - protection against
        # thin/sparsely-mapped obstacles near a waypoint (statue crash
        # fix, lesson 11). min_safe_distance's own full-strength
        # exemption below is separate and always applies regardless.
        goal_scale = max(0.2, min(
            1.0, (dist_to_goal / self.repulsion_fade_radius) ** 2))
        # dist was clamped upward to min_safe_distance, so equality below
        # identifies exactly the points at/inside the hard safety floor.
        mag = np.where(dist <= self.min_safe_distance, mag, mag * goal_scale)
        # MEAN over neighbors, not SUM (2026-07-11, same incident): the
        # obstacle cloud is a dense voxel map - a single shelf face puts
        # 100+ points inside influence_radius, so a raw sum made |f_rep|
        # 20-100x the attractive force with a density-weighted direction
        # that swung +-100 deg per half-meter of motion (the observed
        # rotate-10s/translate-1s dance). The mean is density-independent:
        # one wall repels like ONE wall regardless of how many cloud
        # points sample it, which is the sparse-obstacle scale k_rep=1.0
        # was originally tuned at.
        return (mag[:, None] * (diff / dist[:, None])).mean(axis=0)

    # ------------------------------------------------------------------
    # 6. MOTION (turn-then-go)
    # ------------------------------------------------------------------

    def compute_motion_command(self, v_world, yaw):
        """Strict turn-then-go: two-state machine with hysteresis;
        rotation and translation NEVER share a Twist."""
        planar_speed = math.hypot(v_world[0], v_world[1])
        if planar_speed > HEADING_UPDATE_MIN_SPEED:
            self.last_heading = math.atan2(v_world[1], v_world[0])
        yaw_error = math.atan2(math.sin(self.last_heading - yaw),
                               math.cos(self.last_heading - yaw))

        if self.motion_state == 'rotate':
            if abs(yaw_error) < self.align_enter_threshold:
                self.motion_state = 'translate'
                self.get_logger().info(
                    'Motion state: rotate -> translate (aligned).')
        else:
            if abs(yaw_error) > self.align_exit_threshold:
                self.motion_state = 'rotate'
                self.get_logger().info(
                    f'Motion state: translate -> rotate '
                    f'(yaw_error={math.degrees(yaw_error):.1f} deg).')

        twist = Twist()
        if self.motion_state == 'rotate':
            yaw_rate = self.k_yaw * yaw_error
            twist.angular.z = float(max(-self.max_angular_speed,
                                        min(self.max_angular_speed, yaw_rate)))
        else:
            twist.linear.x = float(planar_speed)
            # DEPTH GUARD enforcement (see depth_guard_stop_m): forward
            # translation only - rotation and vertical are untouched.
            now = self.get_clock().now()
            depth_fresh = (
                self.min_forward_depth_time is not None and
                (now - self.min_forward_depth_time).nanoseconds / 1e9 < 0.5)
            if depth_fresh and self.min_forward_depth < self.depth_guard_stop_m:
                twist.linear.x = 0.0
                self.get_logger().warn(
                    f'[DEPTH-GUARD] forward stop: obstacle at '
                    f'{self.min_forward_depth:.2f}m (< '
                    f'{self.depth_guard_stop_m:.2f}m).',
                    throttle_duration_sec=2.0)
        twist.linear.z = float(v_world[2])  # vertical is state-independent
        return twist

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def control_loop(self):
        # 3. GOAL lifecycle -------------------------------------------------
        if self.current_goal is None:
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info(
                'Waiting for goal...', throttle_duration_sec=2.0)
            return
        # Reached-latch: once inside goal_tolerance the state LATCHES until
        # a new goal message arrives. Without the latch, SLAM noise around
        # the threshold re-engaged APF and caused oscillation (live bug).
        if self.goal_reached:
            self.cmd_vel_pub.publish(Twist())
            return

        # 1. HEALTH ---------------------------------------------------------
        if not self.is_odometry_healthy():
            if self.last_odom_time is None:
                # Startup, not a loss - no recovery state, no yaw sweep.
                self.cmd_vel_pub.publish(Twist())
                self.get_logger().info(
                    'Waiting for odometry data...', throttle_duration_sec=2.0)
                return
            self.enter_recovery(
                'odom',
                'Odometry loss detected — entering hover mode, waiting '
                'for re-acquisition')
            self.publish_recovery_twist()
            return
        if self.recovery_reason == 'odom':
            self.clear_recovery('Odometry restored — continuing to target')

        # 2. POSE -----------------------------------------------------------
        pose = self.get_current_pose()
        if pose is None:
            self.enter_recovery(
                'tf', 'TF not available yet — entering hover mode')
            self.publish_recovery_twist()
            return
        if self.recovery_reason == 'tf':
            self.clear_recovery('TF available again — continuing to target')
        pose_xyz, yaw = pose

        # 2b. POSE SANITY (z envelope, see __init__ pose-trust comments) ----
        # DIRECTIONAL response, not hover (2026-07-12 deadlock fix): a pure
        # hover assumes the insane z is a phantom estimate - but when the
        # climb is REAL (measured live: ground truth 2.95m while chasing a
        # 0.4m target after VO under-reported the ascent), hovering holds
        # the drone at altitude FOREVER (124s+ observed). Gently drive
        # toward the sane band instead: works for both cases - a real
        # excursion physically returns to the band, and a phantom estimate
        # is dragged back into the envelope by its own (under-reporting)
        # motion tracking. In a static world with a hard 5.0m ceiling and
        # geofence floor, a slow vertical crawl toward the band is safe.
        if not (self.pose_sane_z_min <= pose_xyz[2] <= self.pose_sane_z_max):
            self.enter_recovery(
                'pose',
                f'Pose sanity: believed z={pose_xyz[2]:.2f}m is outside the '
                f'plausible envelope [{self.pose_sane_z_min:.2f}, '
                f'{self.pose_sane_z_max:.2f}] - driving gently back toward '
                f'the envelope.')
            twist = Twist()
            if pose_xyz[2] > self.pose_sane_z_max:
                twist.linear.z = -self.geofence_recovery_speed
            else:
                twist.linear.z = self.geofence_recovery_speed
            self.cmd_vel_pub.publish(twist)
            return
        if self.recovery_reason == 'pose':
            self.clear_recovery(
                'Pose estimate back inside the sane envelope - '
                'continuing to target')

        # 2c. POST-RECOVERY GRACE HOVER -------------------------------------
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec < self.grace_until_sec:
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info(
                'Post-recovery grace hover (pose settling)...',
                throttle_duration_sec=2.0)
            return

        # 4. FORCES ---------------------------------------------------------
        f_att, dist_to_goal = self.compute_attractive_force(
            pose_xyz, self.current_goal)
        if dist_to_goal < self.goal_tolerance:
            self.cmd_vel_pub.publish(Twist())
            self.goal_reached = True
            self.get_logger().info('Target reached.')
            return

        if self.obstacle_kdtree is None:
            f_rep = np.zeros(3)
            self.get_logger().warn(
                'No obstacle data yet, applying attraction only.',
                throttle_duration_sec=5.0)
        else:
            f_rep = self.compute_repulsive_force(pose_xyz, dist_to_goal)

        self.debug_att_pub.publish(
            Vector3(x=f_att[0], y=f_att[1], z=f_att[2]))
        self.debug_rep_pub.publish(
            Vector3(x=f_rep[0], y=f_rep[1], z=f_rep[2]))

        v_world = f_att + f_rep
        speed = float(np.linalg.norm(v_world))
        if speed > self.max_linear_speed:
            v_world = v_world / speed * self.max_linear_speed

        # 5. SAFETY: altitude geofence with ACTIVE push-back -----------------
        if pose_xyz[2] <= self.min_altitude:
            v_world[2] = max(v_world[2], self.geofence_recovery_speed)
        elif pose_xyz[2] >= self.max_altitude:
            v_world[2] = min(v_world[2], -self.geofence_recovery_speed)

        # 6. MOTION -----------------------------------------------------------
        self.cmd_vel_pub.publish(self.compute_motion_command(v_world, yaw))


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
