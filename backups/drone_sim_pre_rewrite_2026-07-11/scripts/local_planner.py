#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from scipy.spatial import cKDTree


class LocalPlanner(Node):
    def __init__(self):
        # TF data (rgbd_odometry/rtabmap) is stamped with simulation time;
        # if this node uses wall-clock by default, TF queries
        # will throw "extrapolation into the past" error - hence sim time is a must.
        # Since use_sim_time is a parameter automatically defined by rclpy,
        # it is provided via parameter_overrides, not declare_parameter.
        super().__init__(
            'local_planner',
            parameter_overrides=[Parameter('use_sim_time', value=True)])

        self.declare_parameter('control_rate_hz', 10.0)
        # Reduced from 1.0 upon user request - to leave an extra margin for both
        # odometry robustness and to allow mapping to catch up (for obstacles
        # outside the camera's field of view).
        self.declare_parameter('max_linear_speed', 0.6)
        # In live testing, it was observed that at 1.0 rad/s odometry quality dropped
        # from 870 to 150-400. D435 horizontal FOV is 60 degrees (1.0472 rad), camera
        # is 15 FPS - while a theoretical rotation of ~3.8 degrees per frame at 1.0 rad/s
        # seems reasonable, sudden direction changes/momentum overshoot can cause
        # the actual rotation to be higher. Safety margin increased by using 0.35 rad/s.
        self.declare_parameter('max_angular_speed', 0.35)
        self.declare_parameter('goal_tolerance', 0.3)
        # 1.5 -> 0.8 (2026-07-08, user request: "be braver about passing
        # between obstacles"): repulsive force now only engages when
        # much closer - if two objects were close to each other (e.g. ~2m gap),
        # with the old 1.5m radius the two repulsive fields would
        # overlap and completely block the passage in between. min_safe_distance
        # (0.4) is still preserved - this is a hard lower bound/numerical safety
        # floor, not a "bravery" setting.
        self.declare_parameter('influence_radius', 0.8)
        self.declare_parameter('min_safe_distance', 0.4)
        self.declare_parameter('k_att', 1.0)
        # 0.8 -> 1.0 (2026-07-10, user observation: "doesn't crash but is too
        # brave regarding proximity"): influence_radius (0.8) kept CONSTANT -
        # we are not changing when the repulsive force engages, but HOW
        # strongly it pushes when it does engage. This preserves the ability
        # to pass through narrow passages (the reason for lowering influence_radius
        # above), while keeping the drone slightly more decisively away from
        # the obstacle when inside the influence zone. A small step (25%) -
        # to avoid the risk of oscillation/stopping due to overreaction.
        self.declare_parameter('k_rep', 1.0)
        self.declare_parameter('k_yaw', 1.5)
        # Turn-then-go motion (2026-07-10, user request): the drone should have
        # NO cornering ability at all - not even car-like blended turning. The
        # old model scaled forward_speed by cos(yaw_error), so at small
        # misalignments it would drive an arc (turn and translate at once).
        # Replaced with an explicit two-state machine (rotate / translate) in
        # control_loop: while 'rotate', forward_speed is forced to zero and
        # only yaw_rate is commanded; while 'translate', yaw_rate is forced to
        # zero and only forward_speed is commanded - never both in the same
        # Twist. Two DIFFERENT thresholds (not one) are used deliberately:
        # APF recomputes the desired heading every cycle (10Hz) and it jitters
        # by a few degrees continuously as repulsive forces shift - a single
        # threshold would cause chattering (align -> start moving -> heading
        # drifts 1 degree past the threshold -> stop -> re-align -> forever).
        # align_enter_threshold (tight, ~5 deg) gates rotate->translate;
        # align_exit_threshold (loose, ~15 deg) gates translate->rotate - the
        # gap between them is the hysteresis band that absorbs normal jitter,
        # only forcing a re-turn when the heading has MEANINGFULLY changed
        # (e.g. a new waypoint or an obstacle deflecting the force vector).
        self.declare_parameter('align_enter_threshold', 0.09)
        self.declare_parameter('align_exit_threshold', 0.26)
        self.declare_parameter('slow_radius', 1.0)
        self.declare_parameter('min_altitude', 0.3)
        self.declare_parameter('max_altitude', 5.0)
        self.declare_parameter('obstacle_topic', '/rtabmap/octomap_obstacles')
        self.declare_parameter('goal_topic', '/local_planner/current_target')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_topic', '/rtabmap/odom')
        self.declare_parameter('odom_cov_threshold', 1.0)
        self.declare_parameter('odom_timeout_sec', 1.5)
        # Pose jump detection (2026-07-08, tasks/tracking_loss_roadmap.md Phase 0):
        # in live test, during extended tracking loss, EKF's IMU-only dead-reckoning
        # completely derailed the estimate (estimated z=-2.47m, BELOW the
        # floor). The moment VO gave a brief "quality>0", is_odometry_healthy()
        # said "healthy" but the FUSED TF had already teleported - the drone
        # was driven to the target calculated based on this corrupted pose, crashed
        # into an object/floor and FLIPPED. Solution: Even if TF says "available",
        # additionally check if the new pose has drifted unreasonably far from the last RELIABLE pose.
        # A fixed (NOT scaled to time) threshold is used - the moment the health
        # check is False, zero velocity is already commanded, meaning the drone
        # shouldn't ACTUALLY be moving, no matter how long the loss lasts.
        self.declare_parameter('pose_jump_threshold', 1.0)
        # Accept-after-confirmation (2026-07-08, live observation - see the same
        # mechanism in nbv_planner.py): the single-threshold design correctly
        # rejected CHAOTIC jumps, but it could FOREVER reject a CONSISTENT/persistent
        # small drift (e.g. a real SLAM correction). If consecutively rejected
        # poses remain consistent with each other (not chaotic), they are accepted
        # and the reference is updated.
        self.declare_parameter('pose_confirm_count', 3)
        # EXPERIMENTAL (2026-07-08, user request, TEMPORARY): For EKF vs embedded-VO
        # comparison, pose-jump detection + confirmation-mechanism is
        # COMPLETELY DISABLED - code NOT DELETED, only the below flag is False.
        # To revert: set to True (see full backup is also available,
        # backups/drone_sim_before_vo_experiment_2026-07-08/).
        self.declare_parameter('pose_jump_check_enabled', False)
        # Active re-acquisition rotation (2026-07-08, user observation: "permanent
        # freeze" - in EKF-less/embedded-VO test, the drone got stuck in hover and couldn't
        # recover on its own for 391 seconds, until a new image was shown to the camera
        # using a gamepad). Root cause: ZERO velocity is commanded during hover
        # (correct behavior - the drone REALLY stops), but this also means the camera
        # continues to look at the SAME scene - there is no reason for VO to recover
        # on its own (this is the SAME mechanism whether there is an EKF or not -
        # a similar "5-minute freeze" was also observed in the EKF version in this session).
        # Solution (proposed in tasks/tracking_loss_roadmap.md Phase 2):
        # if stuck for a certain period (recovery_stall_threshold), instead of passively
        # waiting, apply a VERY SLOW in-place rotation (recovery_yaw_rate) -
        # this shows a NEW image to the camera and gives VO a chance to re-acquire.
        # Chosen slowly (about 43% of normal max_angular_speed) -
        # the goal is not to FORCE VO, but to GENTLY present new features.
        self.declare_parameter('recovery_stall_threshold', 10.0)
        self.declare_parameter('recovery_yaw_rate', 0.15)

        p = self.get_parameter
        self.control_rate_hz = p('control_rate_hz').value
        self.max_linear_speed = p('max_linear_speed').value
        self.max_angular_speed = p('max_angular_speed').value
        self.goal_tolerance = p('goal_tolerance').value
        self.influence_radius = p('influence_radius').value
        self.min_safe_distance = p('min_safe_distance').value
        self.k_att = p('k_att').value
        self.k_rep = p('k_rep').value
        self.k_yaw = p('k_yaw').value
        self.align_enter_threshold = p('align_enter_threshold').value
        self.align_exit_threshold = p('align_exit_threshold').value
        self.slow_radius = p('slow_radius').value
        self.min_altitude = p('min_altitude').value
        self.max_altitude = p('max_altitude').value
        self.map_frame = p('map_frame').value
        self.base_frame = p('base_frame').value
        self.odom_topic = p('odom_topic').value
        self.odom_cov_threshold = p('odom_cov_threshold').value
        self.odom_timeout_sec = p('odom_timeout_sec').value
        self.pose_jump_threshold = p('pose_jump_threshold').value
        self.pose_confirm_count = p('pose_confirm_count').value
        self.pose_jump_check_enabled = p('pose_jump_check_enabled').value
        self.last_known_good_pose = None
        self.pending_pose = None
        self.pending_count = 0
        self.recovery_stall_threshold = p('recovery_stall_threshold').value
        self.recovery_yaw_rate = p('recovery_yaw_rate').value
        self.recovery_start_time = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.obstacle_kdtree = None
        self.current_goal = None
        self.goal_reached_logged = False
        self.last_heading = 0.0
        # Turn-then-go state machine (see align_enter/exit_threshold comment
        # above). Starts in 'rotate' so the drone turns to face even its
        # very first target before moving at all.
        self.motion_state = 'rotate'
        self.last_odom_cov = None
        self.last_odom_time = None
        self.in_recovery = False
        self.recovery_reason = None

        self.cmd_vel_pub = self.create_publisher(Twist, '/x500/cmd_vel', 10)
        self.debug_att_pub = self.create_publisher(Vector3, '/local_planner/debug/attractive_force', 10)
        self.debug_rep_pub = self.create_publisher(Vector3, '/local_planner/debug/repulsive_force', 10)

        self.create_subscription(
            PointCloud2, p('obstacle_topic').value, self.obstacle_callback, 10)
        self.create_subscription(
            PoseStamped, p('goal_topic').value, self.goal_callback, 10)
        self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10)

        self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.get_logger().info('Local planner started successfully.')

    def obstacle_callback(self, msg):
        raw = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if raw.shape[0] == 0:
            self.obstacle_kdtree = None
            return
        pts = np.stack([raw['x'], raw['y'], raw['z']], axis=-1).astype(np.float64)
        self.obstacle_kdtree = cKDTree(pts)

    def goal_callback(self, msg):
        self.current_goal = msg
        self.goal_reached_logged = False
        self.get_logger().info(
            f'New target received: ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})')

    def odom_callback(self, msg):
        self.last_odom_cov = msg.pose.covariance[0]
        self.last_odom_time = self.get_clock().now()

    def is_odometry_healthy(self):
        if self.last_odom_time is None or self.last_odom_cov is None:
            return False

        if self.last_odom_cov > self.odom_cov_threshold:
            return False

        time_since_last_odom = (self.get_clock().now() - self.last_odom_time).nanoseconds / 1e9
        if time_since_last_odom > self.odom_timeout_sec:
            return False

        return True

    def publish_recovery_twist(self):
        """Called during hover/recovery. For short-term losses, publishes pure zero
        velocity (current behavior). If the loss lasts longer than recovery_stall_threshold,
        initiates a very slow in-place rotation to show a new image to the camera
        (see __init__ comment)."""
        now = self.get_clock().now()
        if self.recovery_start_time is None:
            self.recovery_start_time = now
        stalled_sec = (now - self.recovery_start_time).nanoseconds / 1e9

        twist = Twist()
        if stalled_sec > self.recovery_stall_threshold:
            twist.angular.z = self.recovery_yaw_rate
            self.get_logger().warn(
                f'Stuck for {stalled_sec:.0f}s - active re-acquisition '
                f'rotation applied.', throttle_duration_sec=3.0)
        self.cmd_vel_pub.publish(twist)

    def get_current_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except tf2_ros.TransformException as e:
            self.get_logger().warn(
                f'TF ({self.map_frame}->{self.base_frame}) not available yet: {e}',
                throttle_duration_sec=2.0)
            return None

        pos = t.transform.translation
        q = t.transform.rotation
        # extract yaw from quaternion (ignore roll/pitch, only Z axis rotation is needed)
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return np.array([pos.x, pos.y, pos.z]), yaw

    def compute_attractive_force(self, pose_xyz, goal_xyz):
        d_vec = goal_xyz - pose_xyz
        d = np.linalg.norm(d_vec)
        if d < self.goal_tolerance:
            return np.zeros(3), d
        scale = min(d, self.slow_radius) / self.slow_radius
        return self.k_att * (d_vec / d) * scale, d

    def compute_repulsive_force(self, pose_xyz):
        if self.obstacle_kdtree is None:
            return np.zeros(3)
        indices = self.obstacle_kdtree.query_ball_point(pose_xyz, r=self.influence_radius)
        f_rep = np.zeros(3)
        for i in indices:
            obs = self.obstacle_kdtree.data[i]
            diff = pose_xyz - obs
            dist = max(np.linalg.norm(diff), self.min_safe_distance)
            magnitude = self.k_rep * (1.0 / dist - 1.0 / self.influence_radius) / (dist ** 2)
            if magnitude < 0:
                continue
            f_rep += magnitude * (diff / dist)
        return f_rep

    def control_loop(self):
        if self.current_goal is None:
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info('Waiting for goal...', throttle_duration_sec=2.0)
            return

        # "Reached" state is LATCHED: once reached, APF is not recalculated
        # again until a new target arrives. Reason: position estimation (SLAM)
        # can be noisy/drifting - recalculating the distance on EVERY loop and
        # checking it around the threshold caused the "reached" state to break
        # when the estimate fluctuated slightly, re-engaging the APF (especially
        # if the target is close to an obstacle) and causing oscillations
        # (observed in live test: drone continued to move AFTER "reached" was logged).
        if self.goal_reached_logged:
            self.cmd_vel_pub.publish(Twist())
            return

        if not self.is_odometry_healthy():
            if self.last_odom_time is None:
                # No /rtabmap/odom message has arrived yet - this is not a "loss",
                # but the initial state. If we don't use a separate log, we'd falsely
                # print a "loss detected" warning as soon as the first target arrives.
                # It is not a "stall" yet either, active rotation shouldn't be initiated.
                self.cmd_vel_pub.publish(Twist())
                self.get_logger().info('Waiting for odometry data...', throttle_duration_sec=2.0)
                return
            if not self.in_recovery:
                self.get_logger().warn('Odometry loss detected — entering hover mode, waiting for re-acquisition')
                self.in_recovery = True
                self.recovery_reason = 'odom'
            self.publish_recovery_twist()
            return
        else:
            if self.in_recovery and self.recovery_reason == 'odom':
                self.get_logger().info('Odometry restored — continuing to target')
                self.in_recovery = False
                self.recovery_reason = None
                self.recovery_start_time = None

        pose = self.get_current_pose()
        if pose is None:
            if not self.in_recovery:
                self.get_logger().warn('TF not available yet — entering hover mode')
                self.in_recovery = True
                self.recovery_reason = 'tf'
            self.publish_recovery_twist()
            return
        pose_xyz, yaw = pose

        # Pose jump check (see __init__ comment) - even if TF/odometry says "healthy",
        # a final check if the underlying estimate is REALLY reliable.
        # Without this, the drone could be driven to a target calculated
        # based on a wildly drifting estimate during tracking loss. Rejected poses
        # go through the ACCEPT-AFTER-CONFIRMATION mechanism (see __init__ comment).
        # EXPERIMENTAL: when pose_jump_check_enabled=False, this block is COMPLETELY skipped.
        if self.pose_jump_check_enabled and self.last_known_good_pose is not None:
            jump = float(np.linalg.norm(pose_xyz - self.last_known_good_pose))
            if jump > self.pose_jump_threshold:
                if (self.pending_pose is not None and
                        float(np.linalg.norm(pose_xyz - self.pending_pose)) <= self.pose_jump_threshold):
                    self.pending_count += 1
                else:
                    self.pending_pose = pose_xyz.copy()
                    self.pending_count = 1

                if self.pending_count < self.pose_confirm_count:
                    if self.current_goal is not None:
                        self.current_goal = None
                        self.goal_reached_logged = False
                    if not self.in_recovery or self.recovery_reason != 'jump':
                        self.get_logger().warn(
                            f'Pose jump detected ({jump:.2f}m > '
                            f'{self.pose_jump_threshold:.2f}m) — estimate not trusted, '
                            f'current target cancelled, hover until NBV gives a new target '
                            f'({self.pending_count}/{self.pose_confirm_count}).')
                        self.in_recovery = True
                        self.recovery_reason = 'jump'
                    self.publish_recovery_twist()
                    return
                self.get_logger().info(
                    f'New pose has been stable for {self.pending_count} loops - accepted '
                    f'(diff: {jump:.2f}m).')
            self.pending_pose = None
            self.pending_count = 0
            if self.in_recovery and self.recovery_reason == 'jump':
                self.get_logger().info('Pose stabilized — continuing.')
                self.in_recovery = False
                self.recovery_reason = None
                self.recovery_start_time = None

        self.last_known_good_pose = pose_xyz.copy()

        if self.in_recovery and self.recovery_reason == 'tf':
            self.get_logger().info('TF available again — continuing to target')
            self.in_recovery = False
            self.recovery_reason = None
            self.recovery_start_time = None

        goal = self.current_goal.pose.position
        goal_xyz = np.array([goal.x, goal.y, goal.z])

        f_att, dist_to_goal = self.compute_attractive_force(pose_xyz, goal_xyz)

        if dist_to_goal < self.goal_tolerance:
            self.cmd_vel_pub.publish(Twist())
            if not self.goal_reached_logged:
                self.get_logger().info('Target reached.')
                self.goal_reached_logged = True
            return

        if self.obstacle_kdtree is None:
            f_rep = np.zeros(3)
            self.get_logger().warn(
                'No obstacle data yet, only applying attractive force.',
                throttle_duration_sec=5.0)
        else:
            f_rep = self.compute_repulsive_force(pose_xyz)

        self.debug_att_pub.publish(Vector3(x=f_att[0], y=f_att[1], z=f_att[2]))
        self.debug_rep_pub.publish(Vector3(x=f_rep[0], y=f_rep[1], z=f_rep[2]))

        v_world = f_att + f_rep
        speed = np.linalg.norm(v_world)
        if speed > self.max_linear_speed:
            v_world = v_world / speed * self.max_linear_speed

        # Safety altitude clamp: if repulsive force (e.g. while avoiding an overhead shelf)
        # tries to push the drone below the floor/above the ceiling,
        # cancel the movement in the corresponding direction outside these bounds. Just
        # "stopping the push" caused the threshold to be slightly exceeded due to
        # physical momentum (tested) - instead, a constant recovery velocity that
        # ACTIVELY pushes back is applied. A real drone/flight controller
        # would also have this kind of "geofence" protection.
        recovery_speed = 0.3
        if pose_xyz[2] <= self.min_altitude:
            v_world[2] = max(v_world[2], recovery_speed)
        elif pose_xyz[2] >= self.max_altitude:
            v_world[2] = min(v_world[2], -recovery_speed)

        # "Unicycle" motion model: the drone can ONLY move forward in the direction
        # its nose (camera) is facing, NO lateral (left/right) sliding. In the previous
        # holonomic model (ability to slide in any world direction, even where the camera
        # isn't looking), it crashed into a box in a live test precisely because of this:
        # while the box was outside the camera's FOV, the drone slid in that
        # direction and went into it before the map was even generated. Now the drone first
        # turns to the target direction, moves forward as it aligns - the camera always sees
        # the direction of travel, no obstacle can approach from a "blind spot".
        planar_speed = math.hypot(v_world[0], v_world[1])
        if planar_speed > 0.05:
            self.last_heading = math.atan2(v_world[1], v_world[0])
        yaw_error = math.atan2(math.sin(self.last_heading - yaw), math.cos(self.last_heading - yaw))

        # Turn-then-go (2026-07-10, user request): NO cornering at all, not even
        # car-like blended turning - rotation and translation are NEVER
        # commanded in the same Twist. Explicit two-state machine with
        # hysteresis (see align_enter/exit_threshold comment in __init__):
        # 'rotate' zeroes forward_speed and only commands yaw_rate; 'translate'
        # zeroes yaw_rate and only commands forward_speed. The two different
        # thresholds (tight enter, loose exit) absorb APF's continuous small
        # heading jitter without chattering between states every cycle.
        if self.motion_state == 'rotate':
            if abs(yaw_error) < self.align_enter_threshold:
                self.motion_state = 'translate'
                self.get_logger().info('Motion state: rotate -> translate (aligned).')
        else:
            if abs(yaw_error) > self.align_exit_threshold:
                self.motion_state = 'rotate'
                self.get_logger().info(
                    f'Motion state: translate -> rotate (yaw_error={math.degrees(yaw_error):.1f} deg).')

        if self.motion_state == 'rotate':
            forward_speed = 0.0
            yaw_rate = self.k_yaw * yaw_error
            yaw_rate = max(-self.max_angular_speed, min(self.max_angular_speed, yaw_rate))
        else:
            forward_speed = planar_speed
            yaw_rate = 0.0

        vx_body = forward_speed
        vy_body = 0.0
        vz_body = v_world[2]

        twist = Twist()
        twist.linear.x = float(vx_body)
        twist.linear.y = float(vy_body)
        twist.linear.z = float(vz_body)
        twist.angular.z = float(yaw_rate)
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
