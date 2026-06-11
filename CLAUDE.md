# PuzzleBot Line-Follower — Claude Context

## What this project is

ROS 2 Humble package running on a **Jetson Orin Nano**. It drives an autonomous
PuzzleBot to follow a line, obey traffic lights, detect arrow/stop signs, and
execute open-loop crossing maneuvers at zebra crossings.

**Never push to a remote, never `colcon build` on behalf of the user — the
robot is tested on hardware directly.**

---

## Active files (touch these)

| File | Role |
|------|------|
| `my_opencv_demo/line_follower_controller.py` | Main node: Mamdani v2 FIS + full state machine |
| `my_opencv_demo/deteccion.py` | YOLOv8 detection node (signs + traffic lights) |
| `my_opencv_demo/line_follower_memoria.py` | Classical CV line detection, publishes `/line_error_px` |
| `my_opencv_demo/zebra_detector.py` | Zebra-stripe detector, publishes `/zebra_detected` |
| `my_opencv_demo/utils.py` | Shared `fresh_qos()` and `pub_img()` helpers |
| `curve_analysis.py` | Standalone analysis script (no ROS needed) — generates control curves + 3D surfaces |

## Reference / archive files (read-only context)

| File | Role |
|------|------|
| `my_opencv_demo/mamdani_v1.py` | First Mamdani (single-input, no state machine) |
| `my_opencv_demo/sugeno_v1.py` | Sugeno zero-order equivalent — same MFs as Mamdani, weighted-average defuzz |
| `backups/` | Snapshot before any refactoring |
| `backups_improved/` | Snapshot after optimization (before fuzzy work) |
| `backups_difuso/` | Snapshot before adding ΔE second input |

---

## State machine (line_follower_controller.py)

```
FOLLOW_LINE → SIGN_LATCHED (when YOLO sees an arrow)
SIGN_LATCHED → STOP_SIGN   (when stop sign seen)
SIGN_LATCHED → STOP_PRE    (when zebra_active + latch set)
STOP_PRE    → CROSSING
CROSSING    → STOP_POST
STOP_POST   → RECOVERING   ← _crossing_active released HERE (not on exit)
RECOVERING  → SIGN_LATCHED (when line re-acquired stably)
```

Key latches / flags:
- `latched_sign` — last arrow seen; default = `p_default_latch` (ros param, default `flecha_derecho`)
- `_crossing_active` — True while in STOP_PRE/CROSSING/STOP_POST/RECOVERING
- `_tl_latched_stop` — True after red, only cleared by green or yellow
- `zebra_consumed` — prevents re-triggering same zebra

Arrow confusion fix: when `flecha_derecho` (go straight) and a turn arrow are
both detected simultaneously, the turn arrow wins.

---

## Fuzzy controller — current version (v2, 2-input Mamdani)

**Inputs:**
- `E`  — line error in px × `error_gain`, clamped to [-320, 320]
- `DE` — filtered_error derivative (px/s), clamped to [-300, 300]

**Outputs:**
- `Velocidad_lineal`  [0, 1.5]  × `fuzzy_v_scale` → `cmd_vel.linear.x`
- `Velocidad_angular` [-3, 3]   × `fuzzy_w_scale` → `cmd_vel.angular.z`

**E memberships** (5): `mucho_error_negativo` (zmf), `poco_error_negativo`
(trimf), `sin_error` (trimf), `poco_error_positivo` (trimf),
`mucho_error_positivo` (smf)

**DE memberships** (3): `decreciendo` (zmf, -300→-20), `estable` (trimf,
-40/0/40), `creciendo` (smf, 20→300)

**15 rules** — key insight: `poco_error + DE_creciendo → mucho_w` (fixes dead
zone), `poco_error + DE_decreciendo → nada_w` (prevents overshoot)

**v1 (single-input, kept as `build_fuzzy_system()`)** is still compiled but not
used — `self.fuzzy_sim = build_fuzzy_system_v2()`.

---

## Parameter groups in line_follower_controller.py

| Group | Parameters | Purpose |
|-------|-----------|---------|
| 1 — Anti-oscillation | `error_filter_alpha`, `angular_slew_rate`, `max_angular_z` | Direct zigzag fix |
| 2 — Curve speed | `curve_slowdown_error`, `curve_min_v_factor` | Slow down on curves |
| 3 — Fuzzy scale | `fuzzy_v_scale`, `fuzzy_w_scale`, `error_gain` | Overall aggressiveness |
| 4 — Accel ramp | `accel_rate_up`, `accel_rate_down`, `soft_start_*` | Acceleration profile |
| 5 — Line lost | `line_lost_timeout`, `line_lost_speed` | Behaviour when line disappears |
| 6 — Traffic light | `traffic_light_enabled`, `yellow_slowdown_factor`, `tl_*` | Semaphore logic |
| 7 — Stop sign | `stop_sign_enabled`, `stop_sign_duration` | Stop sign pause |
| 8 — Crossing maneuvers | `cross_right_*`, `cross_left_*`, `cross_straight_*` | Open-loop turn params |
| 9 — Recovering | `recover_stable_count`, `recover_timeout` | Re-acquisition logic |
| 10 — Default latch | `default_latch` | Sign used when no YOLO detection |

Tuned values that work (do NOT change without hardware test):
- `error_filter_alpha = 0.45`
- `angular_slew_rate  = 1.40`
- `max_angular_z      = 0.55`
- `fuzzy_v_scale      = 0.10`
- `fuzzy_w_scale      = 0.20`
- `error_gain         = 0.75`

---

## Camera / image pipeline

- All nodes subscribe to `video_source/compressed` (CompressedImage) using
  `cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)`.
  **Do NOT switch to raw** — raw ran at ~4 Hz, compressed gives many more fps.
- QoS: `fresh_qos(depth=1)` = BEST_EFFORT + KEEP_LAST(1) everywhere.
- `utils.pub_img(node, publisher, img)` — shared helper to publish debug images.

---

## curve_analysis.py — what it does

Run standalone: `python3 curve_analysis.py`

**Part 1 (v1):** sweeps E [-320,320], plots MFs + control curves v(e) and w(e),
overlays sigmoid/tanh math fits. Saves `control_curves.png`.

**Part 2 (v2):** grid sweep over (E × DE), generates 3D surfaces, slice curves
at DE=-100/0/+100, heatmap of w(E,DE). Saves `control_surfaces_v2.png`.

Key v1 values (for report):
- `v(0) = 0.130 m/s`, `v(±150+) = 0.065 m/s` — step-like
- `w` saturates at `±0.50 rad/s` (= `max_angular_z * fuzzy_w_scale... no —
  max_angular_z` clamp is dead; real limit is `2.50 × 0.20 = 0.50`)
- Dead zone in w: ~0–100 px gives very small w due to MF overlap

---

## Academic report context

This is for a controls/robotics course. The assignment has two parts:
- **Part 1** (done): analyze v1 controller — control curves, math approximations,
  key values table.
- **Part 2** (done): add ΔE as second input to Mamdani → `build_fuzzy_system_v2()`,
  generate 3D control surface plots.

`sugeno_v1.py` is a reference implementation of a Sugeno zero-order equivalent
of the same controller — likely also needed for the report (comparison).

---

## What is NOT done / possible next steps

- `sugeno_v1.py` exists but is not wired into the state machine — it's a
  standalone controller class. If the report requires running Sugeno on the
  robot, it needs the full state machine wrapped around it (copy
  `line_follower_controller.py` structure, swap fuzzy for sugeno).
- `curve_analysis.py` only has curves for Mamdani v1 and v2. Sugeno curves
  could be added for comparison.
- Hardware test of v2 (ΔE input) has NOT been done yet. The robot last ran on
  v1 (single-input Mamdani). v2 is theoretically tuned but untested.
- `setup.py` entry_points intentionally incomplete — not production package.
