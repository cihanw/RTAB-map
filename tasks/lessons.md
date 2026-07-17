# Lessons (Notes to Self)

## 1. RViz freezing → root cause: sim-time jump backward (partial restart) + correct startup order

**What happened:** After adding the EKF, RViz froze repeatedly (100% CPU, stuck on the "R" spinner, not even responding to SIGTERM). My first diagnosis was "two GPU render clients (Gazebo+RViz) competing for 4GB VRAM" — WRONG. My "I opened RViz alone, it was healthy" test wasn't a clean control, because when I shut down Gazebo I had also torn down the SLAM data at the same time — two changes at once, which muddied the result.

**Elimination with evidence (each hypothesis tested separately, ONE variable at a time):**
- GPU contention → Empty-config RViz + Gazebo GUI side by side = HEALTHY (6-10%). Not GPU.
- EKF at 100Hz → clean baseline + 100Hz + RViz = HEALTHY. Not frequency.
- Restarting RViz → clean baseline + manual restart = HEALTHY. Not the restart itself.
- rgbd.rviz config → clean baseline, automatic + manual = HEALTHY. Not the config.

**Actual root cause (proven with a smoking gun):** With everything clean/healthy, I restarted Gazebo ALONE (out from under a live RViz+SLAM stack). The simulation clock jumped back to zero (a 562-second jump backward) and every node immediately logged:
```
[rviz2]    Detected jump back in time. Resetting RViz.
[ekf_node] Detected jump back in time. Clearing TF buffer.
[rgbd_odometry] Detected jump back in time of 562.303 sec...
```
After that, `odom→base_link` BROKE ("two unconnected trees"). While running with `use_sim_time`, RViz detects a backward clock jump and resets itself; this reset **sometimes** (probabilistically / a race condition) locks up at 100% CPU. So the problem wasn't EKF/GPU/config — it was **sim-time discontinuity.** "This never happened before EKF" simply because in earlier sessions a restart always meant shutting down and bringing up the ENTIRE stack together (RViz's clock started from 0, no backward jump); working on the EKF was the first time I started partially restarting Gazebo out from under a live stack.

**RULE 1 — Startup is always a single command, in the correct order:** `ros2 launch drone_sim bringup.launch.py`. This unified file enforces the order:
1. `sim.launch.py` (Gazebo + sensors + `/clock`) — t=0
2. `ros_gz_bridge` (GZ→ROS bridge) — t=5s (to give Gazebo topics time to be ready)
3. `slam.launch.py` (static TF + rgbd_odometry + ekf_node + rtabmap + RViz) — t=8s (once `/clock` and the camera are streaming)

**RULE 2 — NEVER restart Gazebo alone, out from under a live RViz/SLAM stack.** If a restart is needed, shut down the ENTIRE stack together and bring it back up with `bringup.launch.py`. Even if starting things manually piece by piece, the order above must be followed, and if Gazebo restarts, everything — including RViz — must restart with it.

**RULE 3 — If "RViz froze" happens again**, first check with `ros2 topic echo /tf --once` whether `odom→base_link` exists; if it's broken and the logs show "jump back in time", this is exactly this lesson — the fix is a full restart.

## 2. Fully killing Gazebo/ROS2 processes

**What happened:** When I cancelled (killed) the Gazebo (Ignition) simulator or ROS2 launch files running in the background (as a task), child processes (`gz sim server`, `gz sim gui`, `ruby`, `rtabmap`, etc.) survived and started piling up in the background (zombie processes).

**Rule:** After terminating simulations started with `ros2 launch`, ALWAYS make sure the processes are actually dead. Killing just the task is not enough. If needed, run this cleanup in the terminal:
`pkill -9 -f "gz sim" && pkill -9 -f "rtabmap" && pkill -9 -f "local_planner" && pkill -9 -f "nbv_planner"`
Before starting a new process, check with `ps aux | grep -iE 'gazebo|gz|ros|rtabmap|planner'` and confirm that old simulations are definitely cleaned up.

**Reinforced (2026-07-09):** During a rapid edit-relaunch-edit-relaunch loop (iterating on a bug fix, relaunching each time to verify), three full stacks ended up running simultaneously (each with its own Gazebo GUI + RViz instance, all competing for the same GPU) — the user had to point out "you're opening a new RViz without closing the old one, and it's not rendering anything." A cleanup+verify check run once, several tool-calls before the next launch, does NOT stay valid — a `ps aux` check is only trustworthy if run IMMEDIATELY before the next `ros2 launch`, in the same breath. **Rule addendum: every single relaunch, no matter how minor the code change or how confident you are the previous run was stopped, must be preceded by a fresh `ps aux` process check run right then — never rely on a "clean" result from earlier in the session.**

**Root cause found (same incident, deeper dig):** Even after adding `pkill -9 -f "gz sim"`/`"rtabmap"`/`"local_planner"`/`"nbv_planner"`/`"rviz2"` etc., orphaned `ros_gz_bridge`/`parameter_bridge` and `tf2_ros`/`static_transform_publisher` processes from THREE separate earlier launch attempts survived every cleanup pass, undetected, because none of my kill/grep patterns ever included those two node types. They kept relaying/publishing into the SAME topics (`/clock`, camera image/depth, `/x500/cmd_vel`) as the live stack, which is exactly what caused RViz to render only the grid+origin (no map/robot data) and the drone to behave erratically ("frozen at a wall") — not a GPU contention issue at all, a **multiple-conflicting-bridges-on-the-same-topics** issue. System load average hit 31 on a 12-core machine before this was caught. **Rule: name-based `pkill -f <node_name>` patterns are inherently incomplete — every time a new node type gets added to a launch file (as happened here with `theta_star_planner`, and as will happen with any future node), the kill-pattern list silently goes stale for whichever OLDER node types someone forgets to include (here: the bridge and static-TF nodes, present since the very first version of this project, simply never got added to the pattern list). The robust check going forward: verify with a PATH-based pattern that catches every ROS2-installed executable regardless of name, e.g. `pgrep -af "/opt/ros/jazzy/lib/|gz sim|ros2 launch drone_sim"` (matches rviz2, rtabmap, rgbd_odometry, parameter_bridge, static_transform_publisher, point_cloud_xyzrgb — everything under the ROS install tree — in one shot), rather than an ever-growing, easy-to-under-cover list of individual process names.**

## 3. Tuning Exploration Heuristics (Distance vs. Information Gain)

**What happened:** The drone persistently loitered in areas it had already explored instead of heading toward unexplored corners of the room. It was taking tiny steps to trivial local frontiers.
**Root cause found:** The `info_gain_weight` in `nbv_planner.py` was too low (0.25). A fully unexplored 2D disk of radius 2 holds a maximum of 25 cells. At `0.25` weight, the maximum possible reward for a giant unexplored room is 6.25 points. Because distance is penalized at 1.0 point per meter, any completely unexplored room that is more than 6.25 meters away will mathematically *always* be out-scored by a tiny 0-information-gain crack 1.0 meters away.
**Rule:** When tuning exploration scoring functions of the form `distance - weight * info_gain`, always calculate the absolute maximum `info_gain` achievable, multiply it by the weight, and compare it to the distance penalty. The max bonus must be greater than the maximum expected travel distance across the room, otherwise the drone will never break out of its local bubble to explore far-away frontiers.


## 4. Planning stack rewritten (2026-07-11): 2D global planner + multiplicative NBV utility

**What changed:** All four planner scripts were rewritten from scratch
(backup of the old versions: `backups/drone_sim_pre_rewrite_2026-07-11/`).
- The global planner (Theta*) is now **2D (xy only)** — obstacle points
  within a projection band (z in [0.0, 1.5]) are flattened onto the plane;
  the search never sees z. Rationale: vertical routing only ever caused harm
  here — a 3D waypoint at z=1.8m routed the drone above the feature-rich
  band and VO was lost for 8+ minutes. **Vertical terms must never re-enter
  the global search or the NBV scoring.** Waypoints carry the NBV goal's z;
  waypoint advance is XY-only (a 3D check with goal-z waypoints can carry a
  permanent z error bigger than the tolerance = a deadlock).
- NBV scoring is now the literature-standard **U = gain × exp(−λ·d_xy)**
  (λ = `distance_lambda` = 0.10). The additive-scale rule in lesson 3 is
  SUPERSEDED — the equivalent check is now the indifference inequality:
  for the far-room-vs-near-scrap pair, exploration escapes the local bubble
  iff **λ < ln(g_far/g_near) / Δd** (e.g. gain 25 @ 15m vs gain 5 @ 2m →
  λ < ln5/13 ≈ 0.124; 0.15 would re-create the stuck-in-corner bug, 0.10
  passes with margin). Larger environments → smaller λ.
- z_penalty / info_gain_weight / revisit_* are GONE (subsumed or made
  structurally redundant: visited areas have gain ≈ 0, so utility ≈ 0).
- Gain is occlusion-aware ONLY (unknown cells behind obstacle voxels never
  count) — this is the wall-attraction fix; the zero-gain degenerate case
  falls back to the nearest frontier (never "exploration complete").
- The dormant pose-jump mechanism was deleted from both nodes (only copy
  now lives in the backup above).

**Post-rewrite live bug (2026-07-11, first endurance run):** `do_replan`
reset `path_index = 0` on every replan, and `path[0]` is the drone's OWN
snapped start cell — which start-snapping can place just outside
`waypoint_tolerance`. Every 3s replan therefore republished the drone's
own position as the flight target, yanking the heading backward before
the 5Hz advance tick corrected it 200ms later. With turn-then-go and
max_angular 0.35 rad/s (90° ≈ 4.5s > time between yanks), the drone
could never finish aligning → oscillated in place ~2 min until NBV's
120s target timeout bailed it out. **Rule: never publish a fresh path's
start vertex as a target — hand off from `path[1]`. More generally,
whenever a periodic replanner resets its waypoint index, check what the
FIRST published target will be relative to the robot's current pose.**

**Follow-on regression from the fix above (2026-07-11, same session):** the
`path[1]` handoff fix introduced a NEW deadlock. `advance_and_publish`'s
arrival-check ("if already within `waypoint_tolerance` of the current
target, advance `path_index`") ran on every tick, including the very
first tick right after a fresh replan. For a 2-waypoint direct path
whose only real waypoint (`path[1]`) happens to sit within
`waypoint_tolerance` of the drone's OWN current position — common, since
nearby low-effort NBV targets are frequently picked first — the first
tick judged the leg already "arrived," advanced `path_index` straight
past the end of the array, logged "End of Theta* path reached," and
`return`ed WITHOUT EVER CALLING `_publish_waypoint`. Because the goal
never changes, `do_replan` repeated this identically every 3s forever:
"Theta* path found" logged on a loop while `local_planner` sat on
"Waiting for goal..." indefinitely — drone never moves, reproduced on
the very first NBV target of the very next launch after the path[1] fix
landed. **Rule: a "skip/advance if already close enough" check must never
run on the FIRST tick after a fresh plan/path is installed — that tick's
job is to publish the handoff target unconditionally, establishing what
"currently pursuing" even means, before any distance-to-target
comparison is meaningful. Added an explicit `_pending_fresh_publish`
flag for this rather than trying to infer freshness from state.**

## 6. "Drone looping in place at the exact same spot" — periodic replan discarding in-progress path (2026-07-11)

**What happened:** the deadlock in Lesson 5's follow-on was fixed, but a
separate, pre-existing bug then became visible live: `do_replan()` ran
unconditionally every `replan_interval_sec` (3.0s) via a plain timer and
ALWAYS threw away `current_path`, re-solving Theta* fresh from the
drone's live pose every single time - even when the existing path was
still perfectly valid and being actively followed. In a geometrically
tight/cluttered area, any-angle Theta* is sensitive to small differences
in the snapped start cell between one replan and the next, so consecutive
"fresh" solves legitimately picked different immediate next-hops (path
length stayed a stable 6-8 waypoints for 60+ seconds - never shrinking -
while the published waypoint bounced inside a ~1m cluster). Combined with
turn-then-go (every retarget needs a full rotate-then-translate cycle,
which a large yaw change can take 1-2+ seconds to complete), retargeting
faster than that cycle finishes left net displacement ~0: the drone kept
re-orienting toward a slightly different heading every few seconds
without ever completing a real translate leg.

**Why the periodic full-resolve was actually redundant:** the two real
reasons to replan are already event-driven and much faster than 3s:
`goal_callback` replans the instant the NBV goal changes; a companion
callback replans the instant new obstacle data actually blocks the
followed path. The periodic timer's stated purpose ("absorb newly mapped
obstacles") was already fully covered by the latter.

**Fix:** `do_replan(force=False)` - the periodic timer call is unforced
and now returns immediately (does nothing) if a valid, still-in-progress
path already exists (`current_path is not None and path_index <
len(current_path)`). The two event-driven callers pass `force=True`. The
timer becomes a pure safety net ("if for any reason there's no path,
try to get one") rather than a source of constant re-solving.

**Rule: a periodic replanner must never unconditionally discard
in-progress, still-valid state just because its timer fired - only
replan for an actual reason (goal changed / path now unsafe / no path
exists). "Refresh periodically just in case" sounds harmless but, when
the underlying solver is sensitive to small state perturbations (as
any-angle path search is) and the downstream controller is slow to
respond to retargeting (as turn-then-go is), it actively prevents
convergence. Always ask what event-driven trigger already covers the
periodic tick's stated purpose before trusting the tick itself.**

## 5. "RViz opens, drone never moves" (~1 in 3-4 launches) — Fuel network race (2026-07-11)

**What happened:** `models/x500_d455/model.sdf` included the x500 airframe
(which carries the `MulticopterVelocityControl` plugin) via a remote
`<uri>https://fuel.gazebosim.org/...</uri>`. Even with a populated local
Fuel cache (`~/.gz/fuel/...`), gz-fuel-tools re-resolves the model AND
every one of its 16 mesh/texture URIs (each ALSO an absolute
`https://fuel.gazebosim.org/.../files/...` reference inside the cached
model.sdf itself — not just the top-level include) at world-load time.
This is a variable-latency network step with no timeout guard, racing the
fixed wall-clock startup schedule (`bringup.launch.py`'s bridge@5s/slam@8s,
`autonomous.launch.py`'s local_planner@10s/nbv@12s/theta_star@14s — all
real time via `TimerAction`, unrelated to sim time or actual Gazebo
readiness). On a slow/flaky Fuel resolution, the drone model — and its
flight-control plugin — could still be loading when cmd_vel commands
started flowing: camera (a separate, already-local d455 include) and SLAM
come up looking healthy, but the airframe never receives commands.

**Fix:** vendored the model locally (`models/x500/`, copied from the Fuel
cache) and rewrote all 16 absolute mesh/texture URIs to the project's
existing relative-path convention (`meshes/foo.dae`, matching how
`models/d455/` already does it). `x500_d455/model.sdf`'s include now points
to `model://x500`. Zero network dependency at launch time now for either
onboard model.

Swept the rest of the world file for the same pattern and found two more:
`depot.sdf` also pulled "Rescue Randy" and "Standing person" (decorative
statues) straight from Fuel — same risk, since world load parses all
`<include>`s (drone + props) at the same synchronous stage, so a stall on
EITHER prop could equally delay everything downstream. Vendored both
(`models/rescue_randy/`, `models/standing_person/`), same relative-URI
rewrite. `depot.sdf` now has zero remote `<uri>` references anywhere.

**Rule: any `<uri>https://fuel.gazebosim.org/...>` include is a standing
launch-time flakiness risk in this project, REGARDLESS of local cache
state — vendor it into `models/` and rewrite internal mesh/texture URIs to
relative paths, the same way d455 was already done. Grep
`models/**/*.sdf` for `fuel.gazebosim.org` before trusting any new model
include to be launch-safe. This class of bug is easy to miss because it's
probabilistic and every symptom (RViz fine, SLAM fine, only the airframe
inert) points away from the actual cause.**

## 7. "Milling in a tiny area" — APF sum-over-dense-cloud + GNRON (2026-07-11)

**What happened (after lessons 5-6 were fixed):** targets/waypoints were
stable (no churn), yet the drone danced inside ~1m² for 127s at a time:
rotate ~10s → translate ~1s → desired heading swings ±20-100° → repeat,
until NBV's stall timeout. Two compounding causes in
`compute_repulsive_force` (local_planner.py):
1. **Raw SUM over a dense voxel cloud.** k_rep=1.0 was tuned when an
   "obstacle" was a handful of points. The rtabmap cloud samples a single
   shelf face with 100+ points inside influence_radius (0.8m), so |f_rep|
   hit 10-100× |f_att| the moment a surface entered range (measured in a
   60-point wall model: rep jumps 0 → 14.1 vs att 0.2 in one step), with a
   density-weighted direction that swung wildly per half-meter moved.
   **Fix: MEAN over neighbors, not sum — one wall repels like one wall no
   matter how many cloud points sample it.**
2. **GNRON (Goals Non-Reachable with Obstacles Nearby, Ge & Cui 2000):**
   Theta* any-angle waypoints hug the inflation boundary by construction,
   so every waypoint sits inside the repulsion field while the attractive
   ramp (slow_radius) weakens on approach — the force equilibrium sat
   OUTSIDE waypoint_tolerance → arrival never fired. **Fix: fade repulsion
   quadratically within `repulsion_fade_radius` (1.2m, must stay >
   waypoint_tolerance 0.5) of the CURRENT target, zero at the target;
   obstacle points at/inside min_safe_distance are exempt from the fade
   (hard safety floor always full strength).**

**Rule: when the obstacle representation changes density (sparse
hand-placed obstacles → dense SLAM cloud), any force SUMMED per-point
silently rescales by orders of magnitude — normalize by count. Any APF
controller whose goals can legitimately sit near obstacles needs a GNRON
term, or those goals are physically undockable at ANY gain tuning. And
verify force-model fixes numerically first (recreate the stall geometry
in a 10-line numpy model) — the equilibrium argument is checkable on
paper before burning a live run.**

## 8. Wrong loop closure accepted → instant map warp (2026-07-11)

**What happened:** to fix zero-accepted-closures, RGBD/OptimizeMaxError
was raised 5 → 8 based on one run's rejection data (genuine closures at
ratios 5.7-7.1 vs junk at 15-28 — looked cleanly separable at 8). The
very next run silently accepted two closures under the 8 gate; one
(58↔171, two lookalike shelving views ~2m apart) carried ~2.1m of graph
inconsistency and warped the entire map into nonsense in a single
optimization step. Distinct signatures: **gradual crookedness = drift
(odometry problem); sudden wholesale warp = accepted wrong closure
(graph problem).** Note acceptances are INFO-level and invisible in our
WARN-level logs — detect them indirectly: rejection messages that
reference `type=1` edges prove closures were accepted earlier.

**Why the threshold approach failed:** a scalar graph-consistency gate
cannot distinguish "genuine 20cm drift healing" from "alias that happens
to be consistent with a still-flexible graph at acceptance time." One
sample run's separation does not generalize.

**Layered defense adopted instead (all three, launch slam.launch.py):**
1. `RGBD/OptimizeMaxError` back to **5** (last-line warp preventer).
2. `Rtabmap/LoopThr` 0.11 → **0.20** — aliases enter via GLOBAL place
   recognition; a higher bayes posterior kills single-glance lookalike
   hypotheses before geometry is even attempted. Proximity closures
   (spatially gated, correction magnitude capped, structurally
   alias-immune) are untouched and remain the drift healer.
3. SLAM-side `Vis/MinInliers` 13 → **15**, DELIBERATELY stricter than
   rgbd_odometry's 13 (asymmetry documented at both ends): odometry
   needs sensitivity, closure registration needs strictness.

**Rule: never loosen a SLAM safety threshold based on one run's data —
the failure it guards against is rare and catastrophic, so a sample
without a catastrophe proves little. Prefer tightening the earlier,
mechanism-specific gates (hypothesis posterior, inlier count) over
loosening the last-line consistency check. And after ANY closure-related
tuning, watch for the sudden-warp signature specifically.**

## 9. Dead-code launch config + shared-args leak (2026-07-11)

**What happened:** two parameter changes silently went to the wrong
places at once. (1) `Odom/ResetCountdown` was added to the standalone
`rgbd_odometry` Node definition in slam.launch.py - which is DEAD CODE:
defined but not included in the returned LaunchDescription (the live
odometry is rtabmap_launch's embedded one). The fix never took effect,
and the next hard VO loss again hovered blind for 3+ minutes.
(2) The SLAM-side `Vis/MinInliers` 13→15 raise was made in
`rtabmap_args` - which rtabmap_launch feeds to BOTH the SLAM node AND
its embedded odometry. The live odometry silently inherited 15 and VO
dropouts spiked (5 loss episodes in 10 min vs 1 the run before).

**Fix:** odometry-specific values go in the `odom_args` launch argument
("More arguments for odometry (overwrite same parameters in
rtabmap_args)"): `--Vis/MinInliers 13 --Odom/ResetCountdown 30`. The
dead Node now carries a DEAD CODE warning header.

**Rule: before changing any launch parameter, verify WHICH definition
actually reaches the running process - check the live cmdline with
`ps aux | grep <node>` against the string you edited, IMMEDIATELY after
relaunch. A node definition existing in a launch file proves nothing
(this one was kept deliberately as documentation/fallback and its header
said so - but the warning was 30 lines up from where the edit landed).
And when one args string feeds multiple nodes, every per-node divergence
needs an explicit per-node override, not an edit to the shared string.**

## 10. 3.5h endurance run findings (2026-07-11 night)

Run with the corrected params (lesson 9) explored well for ~23 min
("most of the map created without issue" - user), then died in layers:
1. **Physical wedge** near (9.5, 2.5): drone contacted geometry (user
   witnessed the collision); local_planner then commanded max yaw for
   2.5h against the contact constraint. No wedge detection exists -
   recovery machinery only keys on odometry health, and VO stayed
   healthy throughout (quality 330+ while physically stuck).
   **A full detect-and-recover watchdog was built for this (sliding-window
   zero-displacement check + reverse-and-unstick response - design in
   `backups/todo_wedge_watchdog_ABANDONED_2026-07-11.md`, code backup at
   `backups/local_planner_pre_wedge_watchdog_2026-07-11.py` is the
   PRE-watchdog state) and then DELIBERATELY REMOVED same day (user
   decision): "there is absolutely no logic in adding a recovery system
   after it happens" - the world is static, so collisions are a
   PREVENTION failure, not something to detect and paper over
   afterward. Recovering from a wedge treats the symptom; the real bug
   is that the APF/inflation prevention layer let contact happen at
   all. Do not re-propose a reactive watchdog - if a wedge recurs,
   investigate WHY the repulsive force / Theta* inflation didn't stop
   it (e.g. is the obstacle cloud stale/missing near the contact point,
   did k_rep/influence_radius get overridden, was the drone off the
   planned path) rather than building a bigger safety net under the
   same hole.**
2. **Stale-path guard bug**: the lesson-6 livelock guard suppressed
   timer replans whenever ANY valid path existed - including the
   PREVIOUS goal's held path after a failed replan for the new goal.
   fail_counter froze at 1/5, unreachable never fired, NBV waited its
   slow 122s stall timeout five times in a row. FIXED: guard now also
   requires current_path to belong to the CURRENT goal (_path_goal
   tracking). **Rule: any "keep the old result" fallback must be
   goal/键-checked before it can suppress retries for a NEW request.**
3. **Blacklist starvation**: the wedged phase blacklisted every frontier
   in reach → "No valid frontier candidates" every 2s for 2.5 hours, no
   completion signal, no recovery. FIXED: on empty candidate set, clear
   the blacklist once (unreachables re-earn their spot in 15s each);
   only an empty set with an empty blacklist means genuinely complete
   (throttled COMPLETE log).

## 11. Statue crash: thin obstacles DELETED by the grid's noise filter (2026-07-12)

**What happened:** first run with the loop-closure package (FOV 87°) —
9 min of excellent VO (quality 700-1000), then the drone flew straight
into the Rescue Randy statue at (-12,-6) and FLIPPED (IMU roll 172°,
quality 459→0 in one frame). Theta* waypoints had converged exactly on
the statue's position: (-12.2,-6.8) → (-12.6,-6.4) → (-13.0,-6.2).

**Root cause (perception, not planning):** rtabmap's occupancy grid
drops obstacle clusters smaller than `Grid/MinClusterSize` (default 10
cells) as sensor noise. At CellSize 0.1, thin geometry — statue limbs,
shelf lips, rack posts — produces exactly such small clusters, so it
was being SYSTEMATICALLY DELETED from the obstacle map. Every
prevention layer consumes that same map: theta* inflation had nothing
to inflate, APF repulsion nothing to repel from. The GNRON fade
(reaching zero at the target) removed the残 remainder. The earlier
shelf-wedge crash likely shares this mechanism.

**Fixes:** `Grid/MinClusterSize 10 → 3` (keeps thin real geometry,
still kills single-cell speckle) + GNRON fade floor 0.2 (repulsion
never fully zero even at the target; min_safe full-strength exemption
unchanged).

**Rule: when EVERY safety layer misses the same obstacle, suspect the
shared PERCEPTION input before tuning any consumer. Check what the
occupancy pipeline filters out (cluster size, ray-tracing erasure,
projection bands) against the thinnest real obstacle in the world. And
"the planner routed INTO an object" is a map-content question first, a
cost-function question second.**

## 12. Crash prevention: stop targeting the unknown's edge (2026-07-12) — IMPLEMENTED THEN REVERTED SAME DAY

Third crash analysis (drone survived, no flip): all three crashes share
two traits - (1) flying at z=1.0, the exact top of the NBV altitude
band, which coincides with depot shelf-beam height; (2) closing on a
FRONTIER target. A frontier cell is by definition the boundary of
unknown space - unmapped structure begins exactly there, and no
map-based safety layer (inflation, repulsion, min_safe floor) can
protect against geometry that is not in the map yet. Textbook NBV
targets a VIEWPOINT observing the frontier, never the frontier itself.

**Fixes (built, unit-tested, run live, then REMOVED on user instruction
- "completely remove the current frontier system; revert the project to
its state before implementing that system"):** (a) `frontier_standoff_m`
0.8 in nbv_planner - the published flight target was pulled back toward
the drone into known space, frontier tracked separately
(`current_frontier`) for staleness/blacklisting. (b) `max_altitude` 1.0
-> 0.8 (itself reverted back to 1.0 first - see lesson 13 - the low
ceiling starved VO of features, 8 -> 103 loss events in one run).

The standoff system DID measurably help (unreachable-target rate,
reached-rate, zero crashes in its test run) but was fully reverted
regardless, per direct user decision - not a case of it being proven
wrong, but a design-direction call. `nbv_planner.py` is back to
publishing the frontier cell directly as the flight target;
`current_target` is once again the only target/frontier state variable.
**Rule: a fix being empirically effective does not override an explicit
"remove this" instruction - revert cleanly and completely, do not
half-keep pieces that "seemed to help." If the underlying crash
mechanism (frontier-boundary targets = unmapped-structure contact) recurs,
this design and its live evidence are the starting point, not the
altitude cap or loop-closure tuning.**

**Context note (still valid, unrelated to the revert):** ground truth vs
SLAM comparison measured 1.6m of accumulated drift in ~20 min (no
accepted loop closures to correct it). Drift beyond min_safe_distance
degrades ALL map-based collision protection on revisits; genuine closure
candidates score 10-12 vs the MinInliers 15 gate (FOV fix moved them up
from 6-14). User decision: keep MinInliers 15, prevention-first. If
crashes recur in REVISITED (not frontier) areas, drift is the prime
suspect and the loop-closure question reopens.

## 13. Phantom 2-3m altitudes = bad VO re-locks, not commands (2026-07-12)

**What happened:** during the 103-VO-loss run, the drone's believed
altitude reached 2-3m (user saw it in RViz) while EVERY commanded
waypoint z was 0.4-0.8 (verified: zero waypoints above 0.8 in the log).
The climbs were never commanded - they were phantom poses: 19 "no
guess" F2M re-locks during the flap storm can place the pose anywhere.
Acting on such poses either dives the drone (believed high) or drives a
real unbounded climb via the min-altitude geofence push (believed
low/frozen). **Fixes (local_planner health layer):** (1) pose z-sanity
envelope [0.05, 2.0] - outside it the pose is by definition a bad
re-lock (NBV band tops at 1.0), so hover-recover instead of acting;
(2) 2.5s grace hover after EVERY recovery exit - the freshly restored
pose is the least trustworthy one there is.

**Standoff edge-case audit (user-requested), all unit-tested offline:**
frontiers closer than standoff+goal_tolerance are excluded from
candidacy entirely (a pulled-back target would insta-reach-churn or
need to skip the standoff = the old crash vector); a standoff point
landing on a known obstacle slides toward the drone in voxel steps
(up to +1m) to the first free cell; a fully blocked line caps and
defers to theta*'s goal-snap/unreachable machinery. max_altitude
reverted 0.8 -> 1.0 (user decision: the low band starved VO of
features - 8 -> 103 loss events; shelf-beam risk is covered by the
standoff instead).

**Rule: when telemetry shows a state no command could have produced,
audit the ESTIMATOR before the controller - and never let a controller
act on an estimate that is outside the mission's physical envelope.**

## 14. Camera FOV 87° reverted back to 60° (2026-07-12)

After the frontier-standoff system was removed (lesson 12), the drone's
behavior still didn't match its pre-frontier-system baseline. User asked
whether something else had changed in the meantime. Audit found THREE
still-live changes from the same session that were never part of the
frontier system itself: camera FOV 60->87deg (lesson 8's loop-closure
package), `Grid/MinClusterSize` 10->3 (lesson 11, statue-crash fix), and
the GNRON repulsion floor 0->0.2 (lesson 11 companion fix). None were
requested to be reverted when the frontier system was removed - they
were separate, independently-justified fixes that happened to land in
the same session. User chose to revert only the FOV, back to
`horizontal_fov` 1.0472 rad / 60deg with the original D435-derived
intrinsics (fx=fy=554.25469). `Grid/MinClusterSize 3` and the GNRON
floor 0.2 are UNCHANGED - still live, still needed for the statue/thin-
obstacle crash fix documented in lesson 11.

**Rule: when a user says "revert the drone's behavior" after a
multi-fix session, that phrase can refer to ANY subset of the session's
changes, not just the one most recently discussed. Audit and list every
change since the last known-good state before assuming which one(s) are
in scope - the user may want some reverted and others kept, as happened
here (FOV reverted, MinClusterSize/GNRON explicitly kept).**

## 15. "Frozen" investigation: NBV/Theta* cycling near-duplicate unreachable targets (2026-07-13)

**What happened:** live-observed drone motionless for 80s, then again for
159s. Root cause traced via frame-sampler + log correlation, NOT a VO/
health issue: NBV picks a frontier, theta* fails 5x (~13s) and reports
unreachable, NBV blacklists that exact point and immediately picks
ANOTHER point in the same trapped pocket - observed literally
`(-5.80,7.60,0.40)` blacklisted then `(-5.80,7.60,0.80)` tried next
(same x,y, only z differs), failing identically 3s later.

**Two compounding bugs, both fixed:**
1. `nbv_planner.py`'s blacklist matched in full 3D Euclidean distance at
   radius `voxel_size*2` (0.4m) - a same-(x,y) candidate exactly one
   z-step away (0.4m) sat right on the boundary and slipped through
   unblacklisted (strict `<` at exactly 0.4m).
2. Even at a wider radius, 3D matching was conceptually wrong:
   theta_star_planner is a fully 2D planner (search happens entirely on
   the xy-projected grid; z is only ever tacked on afterward as flight
   altitude) - reachability at a given (x,y) is IDENTICAL for every z.
   If theta* fails at z=0.4 it is GUARANTEED to fail at z=0.6/0.8/1.0
   too, since the search never looks at z at all.

**Fix:** blacklist entries are now XY-only (`blacklisted_regions`,
renamed from `blacklisted_targets`), matched at a new
`blacklist_radius_m` parameter (0.55, matching theta_star's
`inflation_radius_m` by convention - not cross-node enforced). Verified
offline: the exact live failure case (`-5.80,7.60,0.80` after
`-5.80,7.60,0.40` blacklisted) is now correctly excluded. Also
`UNREACHABLE_FAIL_THRESHOLD` 5 -> 3 in theta_star_planner - with the
region blacklist now doing most of the "don't re-try the same obstacle"
work, the extra patience of 5 attempts bought little, so cutting to 3
shrinks the per-target cost of the (now rarer) genuine dead-end case.

**Rule: a per-POINT blacklist against a planner that doesn't actually
reason about all of a point's dimensions (here: z) will silently leak -
match the blacklist's dimensionality to what the downstream planner
actually consults, not to the full state vector of the rejected target.**

## 16. Depth-image forward guard — the close-proximity fix, third incident (2026-07-16)

**What happened:** despite the proximity-margin bumps (influence 1.0,
min_safe 0.5, inflation 0.55 — lesson from the roller-door incident),
the close-proximity failure recurred: gt-vs-SLAM logging caught the
drone physically wedged against a wooden crate TWICE in the same
endgame (gt x,y frozen to the millimeter at (1.343,-6.769) then
(0.584,-6.351)), at ~1.75m REAL altitude (contact slide pushed it above
its 1.0m command ceiling), camera half-filled with the crate at
point-blank range. VO limped at quality ~350 on the remaining open half
- the user's observed rule: the moment the camera turns fully into the
obstacle, VO is permanently lost.

**Why margins alone can't fix it:** the map-based layers (cloud ->
octomap -> grid -> APF/inflation) share one pipeline with SECONDS of
latency plus systematic holes (thin/unmapped/stale geometry). Every
contact incident had the same shape: the obstacle was under-represented
in the map at approach time, so no tuning of map-consuming layers helps.

**Fix: depth-image forward guard in local_planner.** Subscribes to the
raw 32FC1 depth stream (30Hz, bridged already for rgbd_odometry); ROI =
upper-middle band (rows [h/4,h/2) - floor CANNOT enter it at low flight
altitudes; cols [w/4,3w/4)); if min valid ROI depth <
`depth_guard_stop_m` (0.7), forward translation is hard-zeroed in the
translate branch of compute_motion_command. Rotation and vertical stay
allowed (turning away / climbing ARE the escape moves), so no new
deadlock class: APF heading keeps evolving and the state machine can
rotate out. Camera near-clip 0.4 (real D455 min range) means readable
values live in [0.4,inf): 0.7 = a 0.3m reaction band = ~5 control ticks
at max speed. Fail-open on stale (>0.5s) or all-invalid depth. Verified
offline: floor-only frame does not fire, wall-ahead at 0.65m fires,
all-nan frame safe.

**Rule: when a failure class survives repeated tuning of layers that all
consume the same delayed/imperfect intermediate representation, add ONE
guard on the rawest available signal instead of continuing to tune
downstream consumers.**

## 17. WARN-only log capture made accepted loop closures invisible (2026-07-13/16)

**What happened:** for the entire loop-closure investigation, "did any
closure get accepted?" was answered by grepping `sim_run.log` for
rejection messages and checking whether a rejection blamed an
already-in-the-graph edge. A run that showed a sudden map distortion but
ZERO such evidence was confidently declared "closure-free, must be pure
drift." Wrong: querying `~/.ros/rtabmap.db`'s `Link` table directly (the
actual committed pose graph, ground truth for what was ever added)
showed 8 GlobalLoopClosure + 1 ProximitySpace links had been accepted in
that exact run - invisible to the log entirely.

**Root cause:** RTAB-Map logs closure *rejections* at WARN level (what
our launch captures) but closure *acceptances* only at INFO level (never
captured). The rejection-blame heuristic is a real detector for SOME
accepted-but-later-conflicting closures, but it is not a substitute for
ground truth - a closure that never conflicts with anything later
produces zero log evidence of ever having existed, in either direction.

**Fix / standing rule:** any "was a closure accepted" question during
this investigation must be answered from `~/.ros/rtabmap.db` (`SELECT
type, from_id, to_id, transform FROM Link` - Python's builtin `sqlite3`
module works directly on the file, no server needed; `sqlite3` CLI is
NOT installed on this machine), not from log greps. The log is a real
diagnostic (rejection reasons, VO quality, timing) but was never a
complete record of graph mutations and should not have been treated as
one. **Rule: when a tool only surfaces one severity level of a system's
logging, "no WARN/ERROR seen" proves absence of *failures*, not absence
of *events* - before concluding "X never happened," check whether X is
even the kind of thing that would log at a level you're capturing.**

## 18. DB-polling closure monitor crashed rtabmap mid-run (2026-07-16)

**What happened:** a long, otherwise-healthy test run (bounded drift,
genuine closures healing it, zero VO loss) ended with the drone frozen
then vanishing from RViz. rtabmap had died: `DBDriverSqlite3.cpp
addStatisticsQuery() ... database is locked`, fatal abort (exit -6). The
process I built for lesson 17 (polling `rtabmap.db` every 15s with
Python's `sqlite3` to catch accepted closures live) was reading the same
file rtabmap was actively writing; a read caught the write lock and
rtabmap treats that as unrecoverable. Gazebo/the drone itself never
crashed - once rtabmap died, `map->odom` just stopped, so RViz lost the
ability to display it. Easy to misread as "the drone disappeared."

**Fix: stopped touching the database entirely.** `rtabmap_msgs/msg/Info`
(topic `/rtabmap/info`, confirmed present on `ros2 topic list`) publishes
`loop_closure_id` / `proximity_detection_id` (non-zero = accepted this
cycle) and the actual `loop_closure_transform` (a proper quaternion, no
need to reconstruct rotation from a raw matrix trace) live over ROS every
~1Hz processing cycle - the real-time signal this whole investigation
needed from the start, with zero file contention. `closure_monitor.py`
replaces `closure_poller.py`.

**Rule: a database file being actively written by a live process is not
a safe concurrent-read target, even for read-only queries, even with a
short poll interval - if the writer publishes the same information over
IPC (a topic, a service), use that instead of the file. Reserve direct
DB inspection for POST-MORTEM analysis after the writer has exited.**

## 19. Latched-goal deadlock bypassed EVERY safety layer for 43 minutes (2026-07-17)

**What happened:** drone reached a target near the east wall, latched
goal_reached, and waited for the next waypoint. Theta* could never
produce one: pinned against the wall, the drone's own grid cell sat
inside the 0.55m inflation zone and the start-snap radius (3 cells =
0.6m) barely out-reached it - every plan failed, 298+ unreachable
events churned, and the blacklist (maxlen 50) aged out and re-picked
regions ~6 times over. Meanwhile control_loop's goal-reached gate sat
ABOVE the health/pose/sanity blocks, so the zero-twist hover it
published every tick was never sanity-checked: a ~2mm/s contact-creep
against the wall accumulated to a REAL 5.4m altitude (believed 5.0m -
2.5x the sanity ceiling) with the guard structurally blind. VO stayed
healthy on wall brick texture throughout - none of the health layers
had anything to object to; the unchecked layers were the problem.

**Fixes:** (1) control_loop reordered - health, pose, z-sanity, grace
now run unconditionally BEFORE the goal-none/goal-reached pursuit
gates. Only pursuit is goal-gated; safety is per-tick. (2)
start_snap_radius_cells 3 -> 5 (1.0m) so a wall-pinned start can always
escape its own inflation: keep >= ceil(inflation/resolution)+2 if
either parameter changes.

**Rule: early-return gates are implicit priority declarations - any
check placed below one silently inherits "this can be skipped." Audit
what sits below every `return` in a safety-relevant loop; guards that
must ALWAYS run belong above every conditional exit, no matter how
innocent the gate above them looks ("we're parked, nothing to do" was
exactly the state in which the drone drifted 4 meters).**
