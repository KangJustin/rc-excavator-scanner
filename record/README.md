# Data format v2: 4-line controls + video

How one teleoperated excavator demonstration is stored, and how it becomes what Physical
Intelligence (PI) receives. Source of truth: `record/episode.py`, `record/controls.py`,
`record/rawlog.py`, `record/pi.py`.

The excavator is driven by four independent digital lines. Each is high (1.0) or low (0.0), with
no reverse, arm-down, or other-direction tilt/turn control yet.

| Line | Does |
|---|---|
| `W` | tracks forward (drive) |
| `O` | arm up |
| `L` | camera tilt |
| `A` | rotate tracks (steer; used together with `W` to turn while driving) |

The lines are independent: `W` and `A` together drive in an arcing turn, and `O` or `L` can be
held regardless of what the tracks are doing. (Nothing yet moves the tracks backward or the arm
down. If those get added later, they need a format change, since the vector length is part of the
PI contract.)

## Two layers

1. **Raw episode (local).** JPEG frames, a JSONL of controls, an edge log and a JSON of metadata.
   Easy to inspect, review, replay and edit. Independent of PI.
2. **PI shards (generated).** At upload each episode becomes two MCAP files. `pi.write_shards()`
   writes them. Never write them by hand.

PI cannot delete an uploaded episode, so everything is checked and reviewed locally first.

## Layer 1: the raw episode

```
episodes/<episode_id>/
    meta.json            episode info
    steps.jsonl          one line per step, aligned with a frame
    controls_raw.jsonl   every line's edges, timestamped (stays local, never uploaded)
    frames/000000.jpg    frame for step 0, then 000001.jpg ...
```

`<episode_id>` defaults to `<rig-id>-YYYYMMDD-HHMMSS`. Give each person or rig its own `--rig-id`
(or `$RIG_ID`) so ids never collide in PI.

**Step `i` is frame `i` is line `i`.** There is no separate video timeline to synchronise.

### `steps.jsonl`
```json
{"i": 3, "t_ns": 1790486491122028000, "state": [0.0, 0.0, 0.0, 0.0], "action": [1.0, 0.0, 0.0, 1.0]}
```
| field | meaning |
|---|---|
| `i` | step index from 0 |
| `t_ns` | epoch nanoseconds on the laptop clock, taken when the frame was received |
| `action` | `[W, O, L, A]` commanded this tick, held until the next frame |
| `state` | `[W, O, L, A]` commanded on the previous tick (zeros at step 0). **Not measured**: there is no feedback from the machine |

The order is always `record.controls.CONTROL_NAMES`. If that ever changes, `meta.json`'s
`control_names` records exactly what order a given episode used, and `load_episode` refuses to
load an episode whose `control_names` doesn't match the code's current order (see "Format guard"
below) rather than silently misreading the columns.

### `controls_raw.jsonl`
```json
{"seq": 0, "t_ns": 1790486490983815000, "ch": "W", "level": 0, "init": true}
{"seq": 4, "t_ns": 1790486493167085000, "ch": "W", "level": 1}
```
One line per edge on the same clock as `t_ns`, plus the starting level of each line (`init`).
`seq` orders lines that share a timestamp (more than one can change on the same tick, e.g. SPACE
drops everything at once). It is lossless, so the step rate can be changed later without
re-recording.

`rawlog.derive_actions()` turns it back into one action per step: a line is high for a step if it
was high for at least half of the window from this frame to the next (the last step uses one step
period). With the keyboard the two always agree, and a test checks that they re-derive identically.

### `meta.json`
```json
{"episode_id": "excavator-20260926-222130", "task": "orbit around the object", "fps": 10,
 "format_version": 2, "control_names": ["W", "O", "L", "A"], "complete": true, "steps": 101,
 "achieved_fps": 9.244, "max_gap_ms": 210.8, "frame_size": [640, 480], "review": "unreviewed"}
```
`complete` is false until the episode is saved. `format_version` plus `control_names` together
identify the format; `load_episode` requires an exact match on both (see "Format guard"). `review`
is `good` or `bad`, set by `python -m record review`; bad episodes are never uploaded.

### Format guard
`load_episode` rejects, with a clear error instead of a silent misread or a crash downstream:
- a `format_version` that doesn't match the code's (an older recording, e.g. the pre-line 10-value
  stick-and-button format),
- a `control_names` that doesn't match the code's `CONTROL_NAMES` exactly, even if
  `format_version` matches (this caught a real early recording that used `["A", "B"]` under
  `format_version: 2`, from before the design settled on the 4-line set above),
- an episode with 0 steps (seen once in the field: `complete: true` with nothing recorded, from an
  interrupted or crashed run; our own recorder never writes this shape, but load_episode checks
  for it directly rather than assuming).

`stats` and `review` never call `load_episode` on an episode that would fail this check: they
route it into `legacy_episodes()`/the incomplete list instead, so one bad file can't crash a
listing of everything else.

### Rules PI enforces (so the recorder checks them)
- **Steady rate.** `fps` is 10 and the contract's tolerance is +/-10% (`pi.STEP_TOLERANCE`, our
  value, not a PI limit). `run` warns when the achieved rate is off, and `stats` flags `RATE`.
- **No dropped frames.** Each step needs a frame, so a stale image is never repeated. `run` warns
  and `stats` flags `GAP` when the longest gap between frames is over 2 steps. Review those
  episodes before uploading.
- **Constant frame size** within an episode (even width and height for H.264). `record` defaults
  to 640x480.
- **Same fps as the contract** (`.pi/ids.v2.json`), or upload refuses.

## Layer 2: what PI receives

Two MCAP files per episode in `shards/<episode_id>/`: `light.mcap` (`system/step` and the two
arrays) and `camera.mcap` (H.264 video).

| topic | role | type |
|---|---|---|
| `system/step` | system | `pi.data.v1.SnapshotRef`, 10 Hz, tolerance 0.1 |
| `observation/camera/rgb/image` | observation | `pi.data.v1.VideoFrame` |
| `observation/robot/controls` | observation | float32 `(4,)`, our `state` |
| `action/robot/controls` | action | float32 `(4,)`, our `action` |

For step `i`, one `system/step` snapshot references the frame, the state array and the action
array by `sequence_id = i` and `capture_time_ns = t_ns`. That link is what ties video to controls.

Names for v2 (nothing has been created in PI under these yet, so the shape above can still change
freely; once `pi-setup` actually runs against PI, it is permanent):

| what | name |
|---|---|
| contract | `rc_excavator_v2` |
| training profile | `rc-excavator-training-v2` |
| dataset filter | `rc-excavator-orbit-demos-v2` |
| source dataset path | `rc-excavator/orbit-v2` |
| saved ids | `.pi/ids.v2.json` (v1's `.pi/ids.json`, a separate 10-value stick-and-button contract, is never touched) |

The training profile uses the camera as the base camera, the state and action arrays above, action
space `absolute`, and `keep_idle_actions=True`. That last one matters: all four lines low means
"doing nothing", and PI skips idle stretches by default, so the model would never see examples of
stopping. Trim long idle stretches at the start and end of each episode yourself.

## Recording

```bash
python -m record run --port $EXCAVATOR_PORT --camera 1 --rig-id yourname --task "orbit around the object"
python -m record run --dry-run --fake-camera        # no hardware at all
```

| key | action |
|---|---|
| `w` | toggle `W` (drive) |
| `o` | toggle `O` (arm up) |
| `l` | toggle `L` (camera tilt) |
| `a` | toggle `A` (steer) |
| SPACE | all lines low |
| `x` | save and end the episode |
| ESC | discard |

The window has no key-up events, so a line stays where you set it until you toggle it. The robot
always goes back to neutral (all lines low) on exit, on error and on Ctrl-C. The ESP32 watchdog
drops all lines after 1 second without a command, and the recorder sends a keepalive every tick.

Use one identical `--task` string for every episode. At inference the task text sent to the model
should match it.

## Workflow

```bash
python -m record stats                      # steps, Hz, idle share, flags, progress toward ~3000 steps
python -m record review                     # G good, B bad, N skip, Q quit
python -m record pi-setup                   # once: creates the v2 contract and profile. PERMANENT.
python -m record pi-upload episodes/<id>    # convert and upload. Uploaded episodes cannot be deleted.
python -m record pi-finetune <name> --wait  # train
```

Aim for about 3,000 usable steps (roughly 5 minutes at 10 Hz) before a fine-tune. Steps in
episodes marked bad, or with a rate problem, and idle steps, do not count towards that number in
`stats`.

## Not done yet
- The ESP32 firmware has `W`, `O`, `L`, `A` outputs, but their GPIO pins are still `-1`
  (unassigned). Fill them in from the wiring. Until then `SET_LINE` answers `ERR unassigned_pin`.
- Nothing has run against real hardware. The firmware logic was exercised on a computer with stub
  Arduino headers, not on an ESP32.
- The wireless camera (Raspberry Pi) is not built. `t_ns` is stamped when the laptop receives a
  frame, so a wifi delay of 100 to 300 ms would misalign frames and controls.
- The v2 contract has not been created in PI and no v2 episode has been uploaded.
- Whether `W`/`A` are wired into the transmitter or drive the motors directly is a decision for
  the hardware side, not this code; either way the software protocol (`SET_LINE <name> <0|1>`)
  stays the same.
