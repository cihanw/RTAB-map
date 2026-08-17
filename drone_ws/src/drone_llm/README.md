# drone_llm

Natural-language commander for the `drone_sim` stack, backed by a **local** LLM
(Qwen3-14B via Ollama). No cloud, no API key.

## What it does

- Turns plain English into typed function calls that drive the drone.
- Narrates live telemetry every ~5 s in one short sentence.
- Acts as a **goal arbiter** so a spoken command is not overwritten by autonomous
  exploration two seconds later.

## Architecture

```
nbv_planner ──/nbv/goal──┐                     (remapped only when llm:=true)
                         v
   /llm/user_input ─> llm_bridge ──/local_planner/current_target──> theta_star ─> local_planner
                         │
                         └──> /llm/response, /llm/narration ──> llm_console
```

`nbv_planner` republishes a frontier goal every 2 s. If the bridge simply published
alongside it, `navigate_to` would be overwritten almost immediately. Routing NBV
*through* the bridge makes goal ownership explicit via a mode:

| Mode | Goal source |
|---|---|
| `EXPLORE` | `/nbv/goal` forwarded unchanged (identical to non-LLM behaviour) |
| `NAVIGATE` | one bridge-published goal; NBV goals dropped |
| `PATTERN` | bridge-published waypoint sequence; NBV goals dropped |
| `IDLE` | current position held |

Nothing downstream was modified: `theta_star_planner` and `local_planner` keep their
existing topics, parameters and tuning. The only change to `drone_sim` is a
conditional remapping in `autonomous.launch.py`.

## Functions exposed to the model

| Function | Arguments | Effect |
|---|---|---|
| `navigate_to` | `x`, `y`, `z?` | Fly to a map-frame point |
| `draw_star` | `radius?`, `center_x?`, `center_y?`, `altitude?` | Trace a 5-point pentagram |
| `start_exploration` | — | Autonomous frontier exploration (NBV) |
| `stop_and_hover` | — | Cancel and hold position |
| `get_status` | — | Live telemetry snapshot |

Omitted centre/altitude default to the drone's current pose.

## Running

**1. Start Ollama on the host** (once per boot). GPU 1 is used so Gazebo keeps GPU 2:

```bash
CUDA_VISIBLE_DEVICES=1 OLLAMA_VULKAN=0 ~/.local/ollama/bin/ollama serve
```

`OLLAMA_VULKAN=0` matters: Vulkan ignores `CUDA_VISIBLE_DEVICES`, and without it
Ollama enumerates a second GPU and reports 72 GiB of VRAM.

**2. Launch the stack with the LLM** (inside the container):

```bash
ros2 launch drone_sim autonomous.launch.py llm:=true
```

**3. Open a client** in a second terminal — either works, and both can run at once:

```bash
ros2 run drone_llm llm_console   # terminal chat
ros2 run drone_llm llm_web       # browser UI at http://localhost:8080/
```

```
you> explore the area
drone> Starting frontier exploration; I'll map the space autonomously.
[telemetry] Exploring at (1.2, -0.4), 0.9 m up, odometry nominal, 2.1 m to goal.
you> draw a star with radius 3
drone> Tracing a 3 m star centred on my current position.
```

Both clients are separate processes on purpose: the bridge must keep flying the drone
whether or not a client is attached, and a closed browser tab or dropped SSH session
must not take the commander down. `llm_web` is stdlib-only (`http.server` +
Server-Sent Events) — no new dependency for a single page.

## Notes

- The bridge is resilient to the model being down: commands report an error, the
  drone keeps flying, and narration is skipped rather than crashing the node.
- `think: false` is set on every request. Qwen3 emits `<think>` blocks by default,
  which would blow the 5 s narration budget and corrupt tool-call parsing.
- All model access is serialised through one worker thread — there is a single GPU,
  so concurrent requests would only contend. User commands take priority; a stale
  telemetry snapshot is dropped rather than queued.
- Odometry health uses the *same* thresholds as `local_planner.py`
  (`covariance[0] > 1.0`, or no message for 1.5 s) so the narration agrees with what
  the controller actually believes.

## Tests

```bash
cd src/drone_llm && python3 -m pytest test/ -q
```

Covers the pentagram geometry: five distinct vertices on the requested circle, the
144° traversal that makes edges cross (rather than drawing a pentagon), centre and
altitude handling, and rejection of a non-positive radius.
