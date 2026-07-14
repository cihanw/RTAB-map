import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_config import WORLD_NAME  # noqa: E402

# Fully scoped topic path of the camera in Gazebo (bridged under this name by ros_gz_bridge)
# World name comes from world_config.py (Phase 1: husarion_office).
# D435 -> D455 transition (2026-07-10, user request): IMU now comes from the embedded IMU
# of the D455, NOT from the x500's OWN sensor - under the SAME sub-model (d455) tree,
# under the same link with the camera (see models/d455/model.sdf).
CAMERA_BASE = f'/world/{WORLD_NAME}/model/drone/model/d455/link/link/sensor/realsense_d455'
IMU_TOPIC = f'/world/{WORLD_NAME}/model/drone/model/d455/link/link/sensor/imu_sensor/imu'


def generate_launch_description():
    pkg_drone_sim = get_package_share_directory('drone_sim')
    rtabmap_launch_path = os.path.join(
        get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py'
    )
    ekf_config_path = os.path.join(pkg_drone_sim, 'config', 'ekf.yaml')

    # Transformation between drone body (base_link) and camera's frame_id
    # (drone/d455/link/realsense_d455). Translation (0,0,0.15) is the physical
    # mount point (from camera_joint in model.sdf: d455 z=0.40 - x500 z=0.25 = 0.15).
    # FIX (2026-07-08, user complaint + VERIFIED with live test):
    # previously the physical mount yaw was 90° (model.sdf) and this comment
    # assumed that 90° was "canceled out" by the -90° yaw component in the
    # standard body->optical transform (roll=-90°, yaw=-90°)
    # (Rz(90°)*[Rz(-90°)*Rx(-90°)]=Rx(-90°)). This was mathematically CORRECT BUT
    # led to the wrong conclusion: the net transform (roll=-90°,yaw=0) shifted
    # the camera's TRUE viewing direction from the drone's +X (forward/travel)
    # axis to the +Y (side) axis - the logic "there is mount rotation, therefore
    # no extra yaw needed in TF" actually meant "camera physically looks sideways,
    # and TF correctly reflects this" (TF was consistent with itself, but the
    # mount ITSELF was wrong). Verified with live test: when the drone was stopped
    # and pose fixed, an object (SUV) at +90° (left) viewing angle appeared in the
    # image, while an object at 0° (forward) angle did not.
    # Solution: physical mount yaw in model.sdf was REMOVED (set to 0),
    # and here ONLY the standard body->optical transform (roll=-90°,
    # yaw=-90°) is applied - no mount rotation, no need to cancel.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_camera',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0.15',
            '--roll', '-1.5708', '--pitch', '0', '--yaw', '-1.5708',
            '--frame-id', 'base_link',
            '--child-frame-id', 'drone/d455/link/realsense_d455',
        ],
        parameters=[{'use_sim_time': True}],
    )

    # D435 -> D455 transition (2026-07-10, user request): IMU is no longer at
    # the base_link of x500, but defined in the SAME link/pose with the D455 camera
    # (see models/d455/model.sdf) - modeling the embedded IMU in the real D455 hardware.
    # Therefore z=0.15 (same physical mount height as the camera, same z as the
    # static_tf above), BUT rotation remains IDENTITY (unlike the camera):
    # IMU is a body-frame sensor, it does not need the camera's optical-frame
    # transform (roll=-90°, yaw=-90°). Even though on the real D455 the IMU is
    # offset by a few mm from the optical center, modeling that offset in this
    # drone-scale simulation has no practical benefit - IMU and camera are defined
    # in the same link (without offset) in model.sdf, and kept co-located here
    # with the same z=0.15 (with identity rotation). Two separate static transforms
    # (camera + IMU) are still required - even though they are on the same SDF link,
    # the ROS TF side cannot infer this automatically, explicit transforms are
    # required for both (just like in the old d435+x500-IMU setup, only the numbers/target
    # frame changed).
    static_tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_imu',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0.15',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'drone/d455/link/imu_sensor',
        ],
        parameters=[{'use_sim_time': True}],
    )

    # ============================================================
    # EXPERIMENTAL CHANGE (2026-07-08, user request, TEMPORARY):
    # EKF + standalone rgbd_odometry have been disabled, instead using
    # rtabmap_launch.launch.py's OWN embedded visual odometry
    # (visual_odometry:true, below) - IMU is still fed to it
    # (via imu_topic). Goal: to live-compare whether the EKF layer
    # (and the pose-jump/approval-mechanism defense it triggers) really
    # brings a net benefit, or if it's just added complexity/risk.
    # THE rgbd_odometry AND ekf_node node definitions BELOW WERE NOT
    # DELETED (see return statement) - they are just NOT INCLUDED in the launch
    # list, reverting when the experiment is over just requires adding them back
    # to the list (and reverting visual_odometry/odom_topic/wait_imu_to_init
    # in the rtabmap include). Also full backup: backups/drone_sim_before_vo_experiment_2026-07-08/
    # ============================================================
    # Visual odometry (rgbd_odometry) is no longer INSIDE rtabmap_launch.launch.py,
    # but defined here as a STANDALONE node. Reason: we NEED to DISABLE this
    # node's TF broadcast (publish_tf) for EKF integration - ekf_node will now
    # broadcast the odom->base_link TF. The official launch file's
    # "visual_odometry" mode does not allow controlling this separately
    # (TF ownership and topic name are tied to the same argument), so this
    # single node was extracted from the official launch file.
    # Vis/MaxFeatures 1000, Vis/MinInliers 18: proven by live CPU measurement
    # (see tasks/lessons.md) - 2000 features saturated rgbd_odometry at 90-124% CPU
    # on a single core, spending ~150ms per frame (~4.5x the 30 FPS budget).
    # Reducing to 1000 brought per-frame time down to ~25ms.
    # Updated to 1500/17 (2026-07-07, user request): observed in live test
    # that the same physical object (casual_female statue) was drawn as a "ghost"
    # in 2-3 different locations on the map - indicating that different passes of
    # that object failed the loop closure geometric verification (Vis/MinInliers)
    # (rejection due to "Not enough inliers" in logs). Increasing MaxFeatures
    # (1000->1500) provides a richer feature pool, increasing true match chances
    # without loosening the threshold; slightly lowering MinInliers (18->17)
    # directly loosens the threshold - a small step, but RGBD/OptimizeMaxError
    # already acts as a secondary safety net against false closures. After
    # observing consecutive healthy closures (12-66 node differences) with 1500/17
    # in live test, pushed a bit further in the same direction to 1700/16
    # (2026-07-07, user request).
    rgbd_odometry = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        namespace='rtabmap',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'publish_tf': False,
            'approx_sync': True,
            'subscribe_rgbd': False,
            'wait_imu_to_init': True,
            'use_sim_time': True,
        }],
        remappings=[
            ('rgb/image', f'{CAMERA_BASE}/image'),
            ('depth/image', f'{CAMERA_BASE}/depth_image'),
            ('rgb/camera_info', f'{CAMERA_BASE}/camera_info'),
            ('odom', 'odom'),
            ('imu', IMU_TOPIC),
        ],
        # MinInliers reduced from 16 -> 14 (2026-07-08, user request):
        # rtabmap.db analysis (337 nodes, 62 accepted/10 rejected) showed
        # two of the rejected closures (14/16, 15/16) were VERY close to the threshold -
        # the rest (0/16, 7/16 x2, 0/16 x3) were already way below it, more features/loosening
        # wouldn't save them. 14 is a small/targeted step just enough to save these two
        # clear near-misses.
        # MaxFeatures 1700 -> 2000, MinInliers 14 -> 15 (2026-07-09, user request):
        # with the camera reverting from 320x240@20fps -> 640x480@30fps, the original
        # (PRE-CPU-fix) MaxFeatures=2000 value was also restored.
        # MinInliers 15 -> 13 (2026-07-10, user request): live loop-closure
        # investigation (two full test runs) found ZERO surviving loop closures
        # in either run - most rejections sit at 0 inliers regardless of
        # threshold (genuine perceptual aliasing from the depot's repetitive
        # structure, not fixable by this parameter), but a consistent minority
        # cluster at 6-14/15 inliers with strong raw match counts (90-160,
        # including one at 14/15) - likely genuine revisits narrowly missing
        # the old threshold. 13 is a small step (same magnitude as the earlier
        # 16->14 move) to rescue those near-misses without touching the
        # 0-inlier aliasing cases.
        # !!! DEAD CODE WARNING (2026-07-11, learned the hard way): this
        # Node is NOT in the returned LaunchDescription (see the comment
        # block above line ~90) - the LIVE odometry is rtabmap_launch's
        # embedded one, configured via rtabmap_args + odom_args in the
        # IncludeLaunchDescription below. Parameters set HERE reach
        # nothing. A ResetCountdown fix added here silently never took
        # effect, while the SLAM-side MinInliers raise leaked into the
        # live odometry through the shared rtabmap_args (VO dropouts
        # spiked). Odometry-specific values belong in odom_args below.
        arguments=['--Vis/MaxFeatures 2000 --Vis/MinInliers 13'],
    )

    # EKF (robot_localization): fuses rgbd_odometry's raw output (/rtabmap/odom)
    # with the IMU's angular velocity + linear acceleration. The goal is to
    # continue estimating from the IMU for a few seconds instead of going completely
    # blind when visual tracking briefly drops (high covariance/"null" message) -
    # the exact same "IMU predicts at high rate, slow sensor corrects" pattern
    # used by real flight controllers (PX4/ArduPilot EKF2).
    # world_frame=odom (inside ekf.yaml) - EKF only improves the local estimate,
    # global loop-closure correction is still RTAB-Map's job.
    # See comments in config/ekf.yaml for details and rationale.
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_node',
        output='screen',
        parameters=[ekf_config_path, {'use_sim_time': True}],
        remappings=[
            ('odometry/filtered', '/rtabmap/odometry_filtered'),
        ],
    )

    # Including RTAB-Map's official launch file (starts the rtabmap SLAM node
    # + rviz internally) — instead of rewriting them manually, we reuse
    # the existing, maintained launch file.
    # visual_odometry:false - disables the internal rgbd_odometry
    # (we now launch it STANDALONE above). odom_topic ensures the
    # rtabmap SLAM node listens to the EKF's fused output
    # (/rtabmap/odometry_filtered) instead of the raw VO.
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rtabmap_launch_path),
        launch_arguments={
            'rgb_topic': f'{CAMERA_BASE}/image',
            'depth_topic': f'{CAMERA_BASE}/depth_image',
            'camera_info_topic': f'{CAMERA_BASE}/camera_info',
            'frame_id': 'base_link',
            'approx_sync': 'true',
            'use_sim_time': 'true',
            # --delete_db_on_start RE-ADDED (user request): want
            # ~/.ros/rtabmap.db cleared on every startup and RViz/map starting
            # empty - accumulation between sessions is no longer desired.
            # Grid/CellSize: default (~0.05m) is much finer than nbv_planner.py's
            # voxel_size parameter (0.2m) - in live measurement
            # /rtabmap/octomap_empty_space produced 2.46 MILLION points, DDS
            # deserialization of this massive message accounted for the majority
            # of nbv_planner's CPU load (even AFTER the vectorized algorithm fix).
            # 0.2 WAS TRIED but LOCKED UP exploration in live test: when the drone
            # starts stationary, it requires ray-tracing the 0.2m volume from top to
            # bottom to count the first cell as "completely empty", which never
            # happens in the first frame (RTAB-Map log: "Grid map is empty!") -
            # NBV finds zero frontiers, drone never moves, permanent lockup.
            # 0.1 VERIFIED by live test: "Grid map is empty!" warning was never
            # seen, NBV found its first goal in ~3 seconds, exploration continued
            # uninterrupted. Point count dropped from 1.4-2.46M to ~83K (~20-30x).
            # nbv_planner CPU is still 77-80% during active exploration (was pegging
            # 100-110% before) - not exactly "cheap" but system load average dropped
            # from 13.98 to ~10.2-10.7 (returned below 12-thread capacity), RViz and
            # rgbd_odometry remained healthy. NOTE: if voxel_size (0.2) in
            # nbv_planner.py changes, this value must be MATCHED MANUALLY.
            # RGBD/OptimizeMaxError: default 3.0 std dev. Gathered proof in live
            # test on refactored repo (Phase 1) - ALL rejected closures are now
            # rejected by a 14-33 DEGREE orientation difference (used to be thousands
            # of METERS, a completely different/catastrophic scale issue). This
            # shows that what remains after fixing the root cause (odometry collapse)
            # is a "borderline/ignorable" statistical threshold matter - loosened to
            # 5.0 (still a tight defense, but allows borderline correct closures
            # of 14-33 degrees to pass).
            # Vis/MaxFeatures 1500, Vis/MinInliers 17 (2026-07-07, user request):
            # rtabmap SLAM node was NEVER receiving these values BEFORE, running
            # with defaults (1000/20) - loop closure geometric verification was
            # operating with a STRICTER threshold than rgbd_odometry (18). This
            # is the direct cause of the "same object drawn 2-3 times as a ghost on
            # the map" issue observed in live test: passes from different angles
            # couldn't find enough consistent feature matches (>=20). Made
            # CONSISTENT with rgbd_odometry.
            # MaxFeatures 2000, MinInliers 15 (2026-07-09, user request) -
            # see rgbd_odometry comment above, reverted to 640x480@30fps camera.
            # MinInliers 15 -> 13 (2026-07-10, user request) - kept CONSISTENT
            # with rgbd_odometry's own MinInliers above (a past bug was caused
            # by exactly these two being out of sync - see the 2026-07-07 note
            # a few lines up). Same rationale as rgbd_odometry's comment above.
            # OptimizeMaxError 5 -> 8 -> 5 (2026-07-11, REVERTED same day):
            # the raise to 8 was based on one run's data showing genuine
            # closures at ratios 5.7-7.1 vs junk at 15-28 ("clean separation
            # at 8"). It did NOT generalize: the very next run silently
            # accepted TWO closures under the 8 gate, one of which (58<->171,
            # a depot shelving alias between two lookalike spots ~2m apart)
            # carried ~2.1m of graph inconsistency and instantly warped the
            # whole map into nonsense. A scalar consistency threshold cannot
            # separate genuine 20cm healing from an alias that happens to
            # look consistent at acceptance time - keep 5 and fight aliasing
            # at the earlier gates instead (LoopThr + MinInliers below).
            # Rtabmap/LoopThr 0.11 (default) -> 0.20 (2026-07-11): aliases
            # enter through GLOBAL place recognition (bayes filter). Raising
            # the hypothesis-acceptance posterior makes single-glance
            # lookalike matches fail while sustained genuine revisits still
            # pass. Proximity closures (spatially gated, correction capped
            # ~1m, structurally alias-immune) are unaffected and remain the
            # primary drift healer.
            # Vis/MinInliers 13 -> 15 SLAM-SIDE ONLY (2026-07-11, user
            # request): second gate against aliased closures. DELIBERATELY
            # different from the odometry's 13 - odometry needs tracking
            # sensitivity (lower = fewer VO dropouts); the SLAM node's copy
            # gates closure registration, where stricter = alias-safer.
            # CRITICAL PLUMBING NOTE: rtabmap_args feeds BOTH the SLAM node
            # AND the embedded odometry node. Odometry-side values must be
            # re-overridden in odom_args below (it "overwrites same
            # parameters in rtabmap_args" per rtabmap_launch docs) - when
            # this MinInliers raise first landed WITHOUT the odom_args
            # override, the live odometry silently inherited 15 and VO
            # dropouts spiked (5 loss episodes in 10 min vs 1 per run).
            # MinInliers 15 -> 20 SLAM-side (2026-07-11 night, user
            # decision): run 23:08 proved 15 still leaks - a proximity
            # closure (255<->275, blamed 17x afterwards) AND a global one
            # (303<->317, blamed 5x) were both accepted and warped the map
            # again. Historical inlier data says genuine closures here
            # cluster at 6-14, so at 20 closures will almost never fire -
            # deliberately chosen over disabling them outright to keep a
            # small chance for pristine high-overlap revisits. Every
            # closure accepted to date (3 across all runs) was harmful.
            # Loop-closure package (2026-07-12): quantity gates alone
            # (MinInliers 13/15/20 all tried) provably cannot separate
            # genuine closures from depot-shelving aliases - so gate on
            # QUALITY instead and fix the physical root (camera FOV 60->87
            # deg, see models/d455/model.sdf):
            # - Vis/MinInliers 20 -> 15: re-opens genuine closures +
            #   proximity healing; safe only together with the two new
            #   gates below.
            # - Vis/PnPReprojError 2 -> 1.5: genuine revisits re-observe
            #   the SAME physical 3D points (sub-pixel consistency);
            #   aliases match DIFFERENT similar-looking points (larger
            #   residuals). This discriminates genuine-vs-alias in a way
            #   inlier COUNT cannot. NOTE: re-pinned to default 2 in
            #   odom_args below - tightening odometry's own registration
            #   was NOT intended (lesson 9: shared args feed both nodes).
            # - Kp/MaxFeatures 500 (default) -> 1000: richer BoW place
            #   signatures, sharper discrimination, 1Hz so CPU-cheap.
            # - RGBD/LoopGravitySigma 0.3 set explicitly: IMU gravity
            #   consistency check on closures (secondary - depot aliases
            #   are mostly yaw-rotated, invisible to gravity - but free).
            # - LoopThr 0.20 / OptimizeMaxError 5: UNCHANGED, last-line
            #   defense proven at these values.
            # Grid/MinClusterSize 10 (default) -> 3 -> 5 -> 8 -> 10 -> 3
            # RE-APPLIED (2026-07-12): filters obstacle clusters under
            # this many cells as noise - at CellSize 0.1, 10 cells DELETES
            # thin obstacles entirely. Live crash: the Rescue Randy
            # statue's thin limbs never survived the filter, so theta*
            # inflation had nothing to inflate and APF repulsion nothing
            # to repel from; the planner routed waypoints straight through
            # the statue's true position at (-12,-6) and the drone hit it
            # and flipped. Was bisected up to 5/8/10 chasing an unrelated
            # "behavior doesn't match baseline" issue, then restored to 3
            # once the user pinned the exact pre-issue conversation
            # moment and confirmed 3 was already live there - this
            # parameter was never actually the cause of that mismatch.
            # Vis/MinInliers 15 -> 14 SLAM-side (2026-07-12, user request):
            # one small step toward the observed genuine-closure band
            # (candidates were scoring 10-12 vs the 15 gate) - the hope is
            # that on the drone's 2nd/3rd lap of the map, high-overlap
            # revisits clear 14. Quality gates (PnPReprojError 1.5,
            # LoopGravitySigma, LoopThr 0.20, OptimizeMaxError 5) unchanged.
            # Odometry keeps its own 13 via odom_args below.
            'rtabmap_args': '--delete_db_on_start --Grid/RayTracing true --Grid/CellSize 0.1 --Grid/MinClusterSize 3 --RGBD/OptimizeMaxError 5 --Rtabmap/LoopThr 0.20 --Vis/MaxFeatures 2000 --Vis/MinInliers 14 --Vis/PnPReprojError 1.5 --Kp/MaxFeatures 1000 --RGBD/LoopGravitySigma 0.3',
            # Odometry-specific overrides (odometry = rtabmap_args + these,
            # last occurrence wins):
            # - Vis/MinInliers back to 13: tracking sensitivity (see above).
            # - Odom/ResetCountdown 30: after a HARD VO loss (view changed
            #   instantly, e.g. rotation onto a featureless close-up), F2M
            #   can hover blind forever - its local feature map is from the
            #   lost place; re-lock quality flickered at 8-14 vs the 13
            #   gate for 3+ minutes in live test. 30 consecutive failed
            #   frames (~1-3s) now reset odometry in place; rtabmap detects
            #   the discontinuity, starts a new map session, and later loop
            #   closures can merge sessions. Bounded blindness.
            # Vis/PnPReprojError re-pinned to default 2: the 1.5 in
            # rtabmap_args is a loop-closure quality gate; odometry's own
            # frame-to-map registration must keep its tracking robustness.
            'odom_args': '--Vis/MinInliers 13 --Odom/ResetCountdown 30 --Vis/PnPReprojError 2',
            'rviz': 'true',
            # RViz already shows the same 3D map (MapCloud), no need to open
            # the rtabmap_viz window as well (unnecessary GPU/CPU load).
            'rtabmap_viz': 'false',
            # IMU for gravity constraint in SLAM graph optimization.
            # The same imu_topic is now fed BELOW to the embedded
            # rgbd_odometry as well (official launch file does this automatically).
            'imu_topic': IMU_TOPIC,
            # EXPERIMENTAL (see block comment above): 'false' -> 'true'.
            # The embedded VO's own Vis/MaxFeatures/MinInliers automatically take
            # the SAME values as rtabmap_args below, BECAUSE the default for
            # the 'args' launch argument IS rtabmap_args.
            'visual_odometry': 'true',
            # EXPERIMENTAL: odom_topic override REMOVED - using official
            # launch file's default ('odom'), so embedded rgbd_odometry AND
            # rtabmap SLAM node consistently use the same topic (and TF,
            # publish_tf_odom defaults to 'true').
            # EXPERIMENTAL: previously explicitly True in our standalone
            # rgbd_odometry node, official launch file defaults to 'false' -
            # explicitly giving 'true' back for behavioral consistency.
            'wait_imu_to_init': 'true',
        }.items(),
    )

    # EXPERIMENTAL: rgbd_odometry and ekf_node are no longer in the list - embedded
    # VO (inside the rtabmap include above) is used. To revert to the previous state:
    # [static_tf, static_tf_imu, rgbd_odometry, ekf_node, rtabmap]
    return LaunchDescription([static_tf, static_tf_imu, rtabmap])
