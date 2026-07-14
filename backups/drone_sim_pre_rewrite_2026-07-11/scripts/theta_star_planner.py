#!/usr/bin/env python3
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

from theta_star_grid import OccupancyGrid3D, theta_star, world_to_voxel

# nbv_planner only publishes when the target CHANGES (as long as the same target remains valid
# it won't publish again) - because this node starts AFTER nbv_planner,
# with the default "volatile" QoS it could miss the first (and most important) target
# locking the whole stack (verified in live test: local_planner forever
# logged "Waiting for goal"). TRANSIENT_LOCAL ensures we receive the last message
# nbv_planner published with the same QoS even if we subscribe late. Also used for
# our own waypoint_pub defensively for the same reason -
# so that if local_planner subscribes late (e.g. restarted) it doesn't
# face the same deadlock risk.
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)


class ThetaStarPlanner(Node):
    """Global path planner that sits between the frontier target chosen by NBV
    (potentially close to an obstacle) and local_planner's purely reactive APF.

    Root cause (diagnosed in live test): local_planner doesn't see the whole map,
    only sums the current attractive/repulsive forces. When NBV chose a target
    right next to an obstacle, these two forces balanced out and locked the drone
    in a local minimum (verified in live TF: 0.67m near the target,
    rotating with no progress for 5+ mins, yaw shifting ~16 deg/sec).

    This node builds a 3D obstacle grid from /rtabmap/octomap_obstacles, computes
    a path to NBV's target via 3D Theta*, and publishes the next waypoint on this
    path to /theta_star/next_waypoint. ZERO code changes in local_planner.py -
    in the launch file, only the goal_topic parameter is redirected to this
    topic (see autonomous.launch.py). local_planner has NO connection to the
    SOURCE of the target, it only cares about the PoseStamped message format -
    hence this integration is possible with zero code changes.
    """

    def __init__(self):
        super().__init__(
            'theta_star_planner',
            parameter_overrides=[Parameter('use_sim_time', value=True)])

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('obstacle_topic', '/rtabmap/octomap_obstacles')
        # Topic nbv_planner already publishes to - reused without touching
        # nbv_planner.py AT ALL.
        self.declare_parameter('nbv_goal_topic', '/local_planner/current_target')
        self.declare_parameter('waypoint_topic', '/theta_star/next_waypoint')
        self.declare_parameter('path_topic', '/theta_star/path')
        # Matches NBV's voxel_size (frontier targets are already snapped to
        # this resolution - to avoid systematic offsets); deliberately coarser
        # than RTAB-Map's raw Grid/CellSize (0.1) (8x fewer
        # voxels = limited search cost).
        self.declare_parameter('grid_resolution', 0.2)
        # Warehouse floor - obstacles don't spawn at z<0, no need for the search
        # to unnecessarily explore below the floor.
        self.declare_parameter('grid_z_min', 0.0)
        # FIX (2026-07-09, live observation): initial value was 2.0 ("1m headroom
        # above NBV's 1.0" reasoning) - but this allowed Theta*'s WAYPOINTS
        # to climb up to z=1.8 to bypass an obstacle over the top (even if the final NBV target is always <=1.0).
        # In live test, the drone climbed to z~1.8m precisely because of this -
        # JUST above the rich-feature band of objects (statues/vehicles/panels, z=0-1.5),
        # in a relatively feature-poor region - and lost visual tracking there,
        # failing to recover for 8+ mins. By reducing it to 1.5
        # (user request), we leave a 0.5m routing headroom above NBV's own ceiling (1.0) -
        # enough to cross over a narrow obstacle, but strictly remaining
        # at the border of the feature-rich band (z=0-1.5).
        # local_planner's hard safety geofence (max_altitude=
        # 5.0) is still the outermost last resort - this value should align
        # with NBV's own ceiling, not with that.
        self.declare_parameter('grid_z_max', 1.5)
        # Slightly tighter than local_planner's hard repulsive floor (min_safe_distance=0.4) -
        # so the two layers agree on what is "too close", but
        # not so conservative that Theta* completely blocks narrow passages
        # that the APF could already safely navigate.
        # 0.3 -> 0.6 (2026-07-10, user request: "global planner algorithm does not select goals too close to obstacles")
        self.declare_parameter('inflation_radius_m', 0.6)
        # Slower than NBV's 2.0s loop (full graph search is more expensive) but
        # frequent enough to catch newly discovered obstacles within a few seconds.
        self.declare_parameter('replan_interval_sec', 3.0)
        # Cheap (just norm+compare) TF-proximity check, much more frequent
        # than replan - makes waypoint advancement feel responsive.
        self.declare_parameter('waypoint_advance_hz', 5.0)
        # DELIBERATELY looser than local_planner's own goal_tolerance (0.3):
        # this is the "am I ready to move to the next waypoint" question,
        # not "have I exactly reached the destination" - that is still entirely local_planner's job
        # (with its own goal_tolerance, unchanged).
        self.declare_parameter('waypoint_tolerance', 0.5)
        # Limits worst-case (target enclosed/unreachable) latency - if exceeded
        # treated the same as "path not found".
        self.declare_parameter('max_search_iterations', 20000)
        # If the drone's own position is temporarily read as blocked (due to inflation margin),
        # a small, bounded search to the nearest free cell.
        self.declare_parameter('start_snap_radius_cells', 3)
        # Must be kept STRICTLY below nbv_planner's goal_tolerance (0.5) -
        # otherwise Theta*'s "snapped" final waypoint might stay too far
        # from NBV's own arrival detection, NBV might never say "reached"
        # and never change its current_target forever - this would be a NEW and
        # subtler deadlock mode.
        self.declare_parameter('goal_snap_radius_m', 0.4)

        p = self.get_parameter
        self.map_frame = p('map_frame').value
        self.base_frame = p('base_frame').value
        self.replan_interval_sec = p('replan_interval_sec').value
        self.waypoint_advance_hz = p('waypoint_advance_hz').value
        self.waypoint_tolerance = p('waypoint_tolerance').value
        self.max_search_iterations = p('max_search_iterations').value
        self.start_snap_radius_cells = p('start_snap_radius_cells').value
        self.goal_snap_radius_m = p('goal_snap_radius_m').value

        self.grid = OccupancyGrid3D(
            resolution=p('grid_resolution').value,
            inflation_radius_m=p('inflation_radius_m').value,
            z_min=p('grid_z_min').value,
            z_max=p('grid_z_max').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nbv_goal = None
        self.current_path = None
        self.path_index = 0
        # advance_and_publish runs at waypoint_advance_hz (5Hz) but shouldn't REPUBLISH
        # as long as the same waypoint hasn't changed - local_planner resets
        # goal_reached_logged on every message and logs 'New goal received'
        # (see local_planner.py goal_callback); constantly forwarding the same target
        # as "new" over and over caused both log spam and the "reached" lock to
        # unnecessarily open and close (observed in live test).
        # Only publish when it TRULY changes.
        self._last_published_target = None
        self.fail_counter = 0

        self.create_subscription(
            PointCloud2, p('obstacle_topic').value, self.obstacle_callback, 1)
        self.create_subscription(
            PoseStamped, p('nbv_goal_topic').value, self.goal_callback, LATCHED_QOS)

        self.waypoint_pub = self.create_publisher(
            PoseStamped, p('waypoint_topic').value, LATCHED_QOS)
        self.path_pub = self.create_publisher(Path, p('path_topic').value, 1)
        self.unreachable_target_pub = self.create_publisher(
            PoseStamped, '/theta_star/unreachable_target', LATCHED_QOS)

        self.create_timer(self.replan_interval_sec, self.do_replan)
        self.create_timer(1.0 / self.waypoint_advance_hz, self.advance_and_publish)

        self.get_logger().info('Theta* global planner started.')

    def obstacle_callback(self, msg):
        raw = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if raw.shape[0] == 0:
            pts = np.zeros((0, 3))
        else:
            pts = np.stack([raw['x'], raw['y'], raw['z']], axis=-1).astype(np.float64)
        self.grid.update_from_points(pts)
        self._invalidate_path_if_blocked()

    def _invalidate_path_if_blocked(self):
        """Checks the remaining waypoints of the path being followed against new
        obstacle data - if one is now blocked, IMMEDIATELY triggers a replan
        instead of waiting for its turn in the periodic loop (see plan Sec 4.1, item 3).
        Never keeps a known-blocked path as "valid" - first clears it
        (None), then tries to replan (Sec 4.2: "stale-but-valid
        is preferred, but known-blocked is NEVER preferred")."""
        if self.current_path is None:
            return
        remaining = self.current_path[self.path_index:]
        if len(remaining) == 0:
            return
        vox_coords = np.array(
            [world_to_voxel(wp, self.grid.resolution) for wp in remaining],
            dtype=np.int64)
        blocked = self.grid.is_blocked(vox_coords)
        if np.any(blocked):
            self.get_logger().warn(
                'Followed path invalidated by new obstacle data - '
                'path cleared, replan triggered.',
                throttle_duration_sec=2.0)
            self.current_path = None
            self.path_index = 0
            self.do_replan()

    def goal_callback(self, msg):
        new_goal = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if self.nbv_goal is None or float(np.linalg.norm(new_goal - self.nbv_goal)) > 1e-3:
            self.nbv_goal = new_goal
            self.fail_counter = 0
            self.get_logger().info(
                f'New NBV goal received: ({new_goal[0]:.2f}, {new_goal[1]:.2f}, '
                f'{new_goal[2]:.2f}) - replan triggered.')
            self.do_replan()

    def get_current_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            pos = t.transform.translation
            return np.array([pos.x, pos.y, pos.z])
        except tf2_ros.TransformException:
            return None

    def do_replan(self):
        if self.nbv_goal is None:
            return
        pose = self.get_current_pose()
        if pose is None:
            self.get_logger().warn('Waiting for TF, skipping replan...', throttle_duration_sec=5.0)
            return

        path = theta_star(
            self.grid, pose, self.nbv_goal, self.max_search_iterations,
            start_snap_radius_cells=self.start_snap_radius_cells,
            goal_snap_radius_m=self.goal_snap_radius_m)

        if path is None:
            # NEVER pass the raw NBV target directly (this brings back
            # the exact bug we fixed). If there is a previous valid path (and
            # it wasn't already cleared because it's blocked) continue to follow it
            # - see Sec 3.9/4.2.
            self.get_logger().warn(
                f'Theta* planning failed: valid path not found for target ({self.nbv_goal[0]:.2f}, '
                f'{self.nbv_goal[1]:.2f}, {self.nbv_goal[2]:.2f}).', throttle_duration_sec=3.0)
            
            self.fail_counter += 1
            if self.fail_counter >= 5:
                self.get_logger().warn(
                    f'Target ({self.nbv_goal[0]:.2f}, {self.nbv_goal[1]:.2f}, {self.nbv_goal[2]:.2f}) '
                    f'unreachable after {self.fail_counter} attempts. Publishing to unreachable_target.',
                    throttle_duration_sec=3.0)
                msg = PoseStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.map_frame
                msg.pose.position.x = float(self.nbv_goal[0])
                msg.pose.position.y = float(self.nbv_goal[1])
                msg.pose.position.z = float(self.nbv_goal[2])
                msg.pose.orientation.w = 1.0
                self.unreachable_target_pub.publish(msg)
            return

        self.fail_counter = 0
        self.current_path = path
        self.path_index = 0
        self.get_logger().info(
            f'Theta* path found: {len(path)} waypoints, target '
            f'({self.nbv_goal[0]:.2f}, {self.nbv_goal[1]:.2f}, {self.nbv_goal[2]:.2f}).')
        self._publish_path_viz()

    def advance_and_publish(self):
        if self.current_path is None or self.path_index >= len(self.current_path):
            return
        pose = self.get_current_pose()
        if pose is None:
            return

        target = self.current_path[self.path_index]
        if np.linalg.norm(pose - target) < self.waypoint_tolerance:
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                self.get_logger().info(
                    'End of Theta* path reached.', throttle_duration_sec=5.0)
                return
            target = self.current_path[self.path_index]

        if (self._last_published_target is None or
                float(np.linalg.norm(target - self._last_published_target)) > 1e-6):
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
        for wp in self.current_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = float(wp[2])
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
