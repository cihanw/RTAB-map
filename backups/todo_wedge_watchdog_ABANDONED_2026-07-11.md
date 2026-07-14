# Collision / Wedge Watchdog — Implementation Plan (NOT yet executed)

(Previous task list — the unreachable-feedback loop — completed 2026-07-10
and documented in lessons.md; superseded here per todo.md convention.)

## Problem statement (evidence: lessons.md §10, 3.5h endurance run 2026-07-11)

The drone physically wedged against depot geometry near (9.5, 2.5) ~23 min
into the run. `local_planner` then commanded maximum yaw (0.35 rad/s)
against the contact constraint for **2.5 hours**. No existing mechanism can
see this failure:

- The recovery machinery keys ONLY on odometry health (covariance > 1.0 or
  silence > 1.5 s). VO stayed healthy the whole time (quality 330+) — the
  camera sees the scene fine from inside a wedge.
- NBV's stall timeout fired repeatedly and blacklisted every reachable
  frontier (now mitigated by blacklist amnesty, but the drone still can't
  move).
- Theta* failures describe unplannable goals, not an unmovable robot.

**The observable signature of a wedge** (from live TF/cmd_vel inspection of
the actual incident): sustained non-trivial `cmd_vel` + healthy pose stream
+ near-zero actual displacement AND near-zero actual yaw change. Every
element of that signature is already available inside `local_planner`.

## Design

### D1. Home: `local_planner.py` only

It already has, at 10 Hz: the pose (`get_current_pose()`), the command it
just published, the recovery-state machinery, and the motion state machine.
No other node changes. NBV/theta* churn during a wedge is already
self-managing (fast 5/5 unreachable loop + blacklist amnesty).

### D2. Detection — sliding-window zero-progress check

Keep a `deque` of `(t, x, y, z, yaw, cmd_was_active)` appended once per
control tick. `cmd_was_active` = the Twist published this tick had
`|linear| > 0.1 m/s` or `|angular.z| > 0.1 rad/s`.

Every tick (when armed — see D4), examine the window spanning
`wedge_window_sec`:

```
WEDGED iff (over the full window):
    active_fraction  >= wedge_cmd_activity_min   (drone kept TRYING)
AND xyz displacement <  wedge_min_displacement_m (didn't move)
AND |yaw change|     <  wedge_min_yaw_rad        (didn't turn either)
```

Why each term:
- `active_fraction` gate: excludes goal-reached hover and idle "Waiting
  for goal" (no commands → window never qualifies).
- displacement gate: 0.10 m over 8 s is far above RTAB-Map pose noise
  (~1–3 cm) and far below any genuine progress at commanded speeds.
- yaw gate: excludes the legitimate turn-then-go rotate phase. A 180°
  realignment at 0.35 rad/s takes ~9 s of zero displacement — but its yaw
  visibly CHANGES throughout. A blocked rotation shows < 8° over 8 s.
  This is the term that makes the watchdog compatible with turn-then-go.

### D3. Response — bounded escalation ladder

Stage 1 — **unstick burst** (on detection):
  - Abort pursuit for `unstick_duration_sec` (2.0 s) and publish, in body
    frame: `linear.x = -unstick_reverse_speed` (back straight out of the
    contact — rotation is exactly what a wedge blocks, so back out first,
    never turn first) and `linear.z = +unstick_climb_speed` (wedges are
    typically against shelf lips/skids; vertical is usually the free axis).
  - Respect the altitude geofence: cap the climb so z stays < max_altitude.
  - Turn-then-go is NOT violated: reverse translation along the current
    heading is straight-line motion, no rotation is mixed in.

Stage 2 — **re-check + alternate** (after each burst):
  - Clear the window, resume normal control. If WEDGED re-triggers within
    `wedge_retrigger_sec` (15 s), burst again, alternating the vertical
    component sign (+z, then −z, then +z…) in case the contact is above.
  - Count attempts.

Stage 3 — **hard-stuck** (after `unstick_max_attempts` = 3 failed bursts):
  - Publish zero Twist (hover), log CRITICAL
    `[WEDGE] HARD-STUCK after 3 unstick attempts - manual intervention
    required` (throttled, 30 s), and hold until displacement resumes for
    any reason (then auto-clear and resume). No automatic teardown — in
    sim the operator decides; on hardware this would be a land command,
    out of scope here.

### D4. Arming rules (false-positive safety)

The watchdog is DISARMED whenever:
- odometry-health recovery is active (that machinery owns the situation;
  healthy-odom is precisely the gap this watchdog fills), or
- `goal_reached` latch is set / no current goal, or
- an unstick burst is in progress (its own zero-displacement must not
  re-trigger detection), or
- fewer than `wedge_window_sec` of samples exist since the last arm/clear.

Pose-jump robustness: a SLAM map correction while wedged can show a fake
> 0.10 m "displacement". Consequence: the window resets and detection is
delayed by one window (~8 s). Accepted — the failure we're fixing lasted
2.5 hours; an extra 8 s is noise. No pose-jump special-casing (that
machinery was deliberately deleted in the rewrite).

### D5. Parameters (all declared, project comment style)

| Parameter                  | Default | Rationale |
|----------------------------|---------|-----------|
| `wedge_window_sec`         | 8.0     | > longest legit zero-displacement stretch (180° rotate ≈ 9 s is excluded by the yaw gate, not the window) |
| `wedge_min_displacement_m` | 0.10    | > SLAM noise (1–3 cm), << real progress (0.6 m/s × 8 s) |
| `wedge_min_yaw_deg`        | 8.0     | blocked rotation shows ~0°; legit rotate shows > 100° per window |
| `wedge_cmd_activity_min`   | 0.7     | ≥ 70 % of window ticks actively commanding |
| `unstick_reverse_speed`    | 0.3     | same magnitude class as geofence_recovery_speed |
| `unstick_climb_speed`      | 0.2     | gentle vertical escape, geofence-capped |
| `unstick_duration_sec`     | 2.0     | ~0.6 m of reverse — clears a prop-depth contact |
| `unstick_max_attempts`     | 3       | then hard-stuck (bounded, no infinite bursts) |
| `wedge_retrigger_sec`      | 15.0    | escape verification horizon |

### D6. Code placement (`scripts/local_planner.py`)

- New section header `# WEDGE WATCHDOG` between the HEALTH and POSE
  sections (renumber section comments accordingly).
- New state in `__init__`: `pose_history` (deque, maxlen ≈
  `wedge_window_sec × control_rate_hz + margin`), `wedge_state`
  (`'monitoring' | 'unsticking' | 'hard_stuck'`), `unstick_attempts`,
  `unstick_until` (time), `unstick_z_sign`.
- `control_loop` integration (reads top-to-bottom as prose, matching the
  rewrite style):
  1. existing health gates (unchanged),
  2. pose acquisition (unchanged),
  3. `if self.wedge_state == 'unsticking': publish burst twist / finish
     burst; return`,
  4. `self._feed_watchdog(pose, last_cmd_active)` +
     `if self._wedge_detected(): begin unstick / escalate; return`,
  5. existing forces/motion pipeline; record whether the published Twist
     was "active" for the next tick's feed.
- Log lines (grep-able, consistent with the monitoring workflow):
  `[WEDGE] detected (disp=..., dyaw=..., active=...%)`,
  `[WEDGE] unstick attempt k/3 (reverse+{up|down})`,
  `[WEDGE] escaped (displacement resumed)`,
  `[WEDGE] HARD-STUCK ...`.

### D7. Explicitly out of scope

- No lateral (`linear.y`) escape — preserves the unicycle/turn-then-go
  contract everywhere, including emergencies.
- No Gazebo contact-sensor dependency — pose+command only, so the design
  transfers to real hardware unchanged.
- No NBV/theta* changes.
- No automatic land/disarm on hard-stuck (sim context).

## Execution checklist (for the implementation session)

- [ ] 0. Backup: `cp scripts/local_planner.py backups/local_planner_pre_wedge_watchdog_<date>.py` (no git!)
- [ ] 1. Add parameters + state to `__init__` (D5, D6)
- [ ] 2. Implement `_feed_watchdog` / `_wedge_detected` (D2, D4)
- [ ] 3. Implement unstick burst + escalation (D3) in `control_loop`
- [ ] 4. `python3 -m py_compile` + offline detector test on synthetic
      traces (scratchpad script, mirrors the theta* test style):
      wedged trace → fires; healthy 180°-rotate trace → must NOT fire;
      hover-at-goal trace → must NOT fire; slow-crawl-near-goal trace →
      must NOT fire; pose-jump-mid-wedge trace → fires one window late.
- [ ] 5. Live false-positive run: ≥ 15 min normal exploration,
      `grep -c "\[WEDGE\]"` must be 0.
- [ ] 6. Live true-positive run: force a wedge (publish a manual goal into
      a rack corner on a test launch with `k_rep` temporarily 0 — a
      deliberate, documented, test-only degradation) → detection within
      ~10 s, unstick executes, normal flight resumes, log shows
      `[WEDGE] escaped`.
- [ ] 7. lessons.md §10 update: mark the open item CLOSED with a pointer
      to this design; note any tuning discovered in steps 5/6.

## Review

(to be filled after implementation)
