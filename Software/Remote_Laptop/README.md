# Remote_Laptop  

This folder contains the codebase which was used during the final mission as well as some prototype code used during development. It also contains the necessary documentation from the software development stage.

---

# Auto Navigation — Full Codebase Reference

This document covers every Python file in the `auto_nav` package, tracing the evolution from simple keyboard teleop all the way to the final autonomous explorer with A\* pathfinding, polar-arc docking, and multi-station mission logic.

---

## Table of Contents

1. [Repository Overview](#1-repository-overview)
2. [File Inventory](#2-file-inventory)
3. [Architecture Fundamentals](#3-architecture-fundamentals)
   - [Shared Building Blocks](#31-shared-building-blocks)
   - [State Machine Pattern](#32-state-machine-pattern)
   - [ROS2 Topic Map](#33-ros2-topic-map)
4. [Early / Utility Files](#4-early--utility-files)
   - [r2mover.py](#41-r2moverpy)
   - [r2moverotate.py](#42-r2moverotate)
   - [r2scanner.py](#43-r2scannerpy)
   - [r2e2.py](#44-r2e2py)
   - [r2occupancy.py & r2occupancy2.py](#45-r2occupancypy--r2occupancy2py)
5. [Navigation Foundations](#5-navigation-foundations)
   - [r2auto_nav.py](#51-r2auto_navpy)
6. [Core Components Deep Dive](#6-core-components-deep-dive)
   - [RegulatedPurePursuit](#61-regulatedpurepursuit)
   - [MapNode](#62-mapnode)
7. [Test / Prototype Progression](#7-test--prototype-progression)
   - [r2autopilot.py](#71-r2autopilotpy)
   - [r2fulltest.py](#72-r2fulltestpy)
   - [polardocking.py](#73-polardockingpy)
   - [r2livecode.py](#74-r2livecodepy)
   - [r2livecode(frontierscoring,wallinflation).py](#75-r2livecodefrontierscoringwallinflationpy)
   - [new.py](#76-newpy)
   - [3phasedocking.py](#77-3phasedockingpy)
8. [fulltest Series — Complete Analysis](#8-fulltest-series--complete-analysis)
   - [fulltest(astar).py](#81-fulltestastarpy)
   - [fulltest(polardocking).py](#82-fulltestpolardockingpy)
   - [fulltest(astarpolardocking).py](#83-fulltestastarpolar-dockingpy)
   - [fulltest(astar3phasedocking).py](#84-fullestastar3phasedockingpy)
9. [Final Evaluation Code](#9-final-evaluation-code)
10. [Algorithm Reference](#10-algorithm-reference)
    - [A\* Pathfinding](#101-a-pathfinding)
    - [BFS Frontier Exploration](#102-bfs-frontier-exploration)
    - [Polar-Arc Docking Controller](#103-polar-arc-docking-controller)
    - [3-Phase Docking Sequence](#104-3-phase-docking-sequence)
    - [Path Post-Processing Pipeline](#105-path-post-processing-pipeline)
11. [Parameter Reference Tables](#11-parameter-reference-tables)
12. [Evolution Diagram](#12-evolution-diagram)
13. [setup.py](#13-setuppy)

---

## 1. Repository Overview

The codebase is a ROS2 (Humble / Iron) autonomous navigation stack for a TurtleBot3-class robot. The robot must:

1. Map an unknown environment using SLAM.
2. Explore frontiers autonomously until it detects ArUco fiducial markers.
3. Dock at two stations (marker IDs 0 and 1) in sequence.
4. Signal a Raspberry Pi over GPIO, wait for an acknowledgement, then undock.
5. Continue exploring until the full map is covered.

The files in this folder represent the full development history — from the first keyboard teleop script right through to the final polished code.

---

## 2. File Inventory

| # | File | Category | Lines (approx) | Status |
|---|------|----------|----------------|--------|
| 1 | `r2mover.py` | Teleop | ~60 | Early prototype |
| 2 | `r2moverotate.py` | Teleop | ~100 | Early prototype |
| 3 | `r2scanner.py` | Sensor util | ~40 | Diagnostic |
| 4 | `r2e2.py` | ROS2 echo | ~25 | Test node |
| 5 | `r2occupancy.py` | Visualisation | ~60 | Diagnostic |
| 6 | `r2occupancy2.py` | Visualisation | ~90 | Diagnostic |
| 7 | `r2auto_nav.py` | Navigation | ~200 | Navigation v0 |
| 8 | `r2autopilot.py` | Navigation | ~600 | Navigation v1 |
| 9 | `r2fulltest.py` | Navigation | ~700 | Navigation v2 |
| 10 | `polardocking.py` | Docking | ~150 | Docking prototype |
| 11 | `r2livecode.py` | Navigation | ~800 | Navigation v3 |
| 12 | `r2livecode(frontierscoring,wallinflation).py` | Navigation | ~850 | Navigation v3.5 |
| 13 | `new.py` | Navigation | ~900 | 3-phase docking prototype |
| 14 | `3phasedocking.py` | Navigation | ~950 | 3-phase docking v2 |
| 15 | `fulltest(astar).py` | Navigation | ~900 | A\* + navigation |
| 16 | `fulltest(polardocking).py` | Navigation | ~950 | BFS + polar dock |
| 17 | `fulltest(astarpolardocking).py` | Navigation | ~1000 | A\* + polar dock |
| 18 | `fulltest(astar3phasedocking).py` | Navigation | ~1050 | A\* + 3-phase dock |
| 19 | `r2CDE2310_FINAL.py` | Navigation | ~1100 | **Final evaluation code** |
| 20 | `setup.py` | Build | ~30 | ROS2 package config |

---

## 3. Architecture Fundamentals

### 3.1 Shared Building Blocks

Every non-trivial navigation file shares three reusable components:

```
┌─────────────────────────────────────────────────────────┐
│                    AutoPilot(Node)                       │
│  ┌──────────────────┐   ┌──────────────────────────┐   │
│  │RegulatedPurePursuit│   │      MapNode              │   │
│  │  (path follower) │   │  (waypoint / graph node)  │   │
│  └──────────────────┘   └──────────────────────────┘   │
│                                                          │
│  Sensors: LiDAR scan → front/left/back/right arrays    │
│           OccupancyGrid → occ_pooled (3x downscale)    │
│           ArUco camera → marker_x, marker_z            │
│           TF2 map→base_link → cur_x, cur_y, cur_yaw   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 State Machine Pattern

All advanced files run a 10 Hz timer (`controller()`) that dispatches to the current state:

```
PLANNING → DRIVING → (marker seen) → DOCKING/DOCK_NAV → STATION
    ↑           |                                            |
    |      obstacle                                    WAITING_FOR_PI
    |           ↓                                            |
    |       RECOVERY → ESCAPING ──────────────────────► UNDOCKING
    |                                                        |
    └────────────────────────────────────────────────────────┘
```

### 3.3 ROS2 Topic Map

| Topic | Type | Direction | Purpose |
|-------|------|-----------|---------|
| `/cmd_vel` | `Twist` | publish | Drive commands |
| `/map` | `OccupancyGrid` | subscribe | SLAM map |
| `/scan` | `LaserScan` | subscribe | LiDAR ranges |
| `/usbcam1_markers` | `ArucoMarkers` | subscribe | Fiducial detections |
| `/rpi_response` | `String` | subscribe | Station ACK from Pi |
| `/gpio_commands` | `String` | publish | Signal Pi on dock |
| `/goal_marker` | `Marker` | publish | RViz goal sphere |
| `/planned_path` | `Path` | publish | RViz path line |
| `/lookahead_marker` | `Marker` | publish | RViz RPP lookahead |

---

## 4. Early / Utility Files

### 4.1 r2mover.py

The very first file. Pure keyboard teleoperation, no sensor feedback.

```
Keyboard → readKey() → Twist → /cmd_vel
```

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rotatechange` | `0.1` rad | Angular step per keypress |
| `speedchange` | `0.05` m/s | Linear step per keypress |

**Key bindings:** `W/↑` forward · `S/↓` backward · `A/←` turn left · `D/→` turn right · `Space` stop

No odometry. No feedback loop. Baseline for manual testing of the hardware.

---

### 4.2 r2moverotate.py

Extends `r2mover` with odometry-based heading tracking so the robot holds a requested heading rather than just applying a raw angular velocity.

**Heading arithmetic** uses complex numbers to avoid the ±π wraparound problem:

```python
c = complex(math.cos(heading), math.sin(heading))
c *= complex(math.cos(delta), math.sin(delta))
new_heading = math.atan2(c.imag, c.real)
```

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rotatechange` | `0.1` rad | Increment per keypress |
| `speedchange` | `0.05` m/s | Linear increment |

**Subscriptions:** `/odom` (Odometry)

---

### 4.3 r2scanner.py

Diagnostic node. Subscribes to `/scan`, finds the minimum distance reading, and logs it. Used during early LiDAR bring-up to verify the sensor was working and to understand the scan array indexing before implementing sector splitting.

---

### 4.4 r2e2.py

Minimal ROS2 pub/sub echo node from the official ROS2 beginner tutorial. Included to verify the ROS2 install and Python environment. No navigation logic.

---

### 4.5 r2occupancy.py & r2occupancy2.py

Both visualise the SLAM occupancy grid using matplotlib. Intended for debugging path planning decisions outside of RViz.

| Feature | r2occupancy.py | r2occupancy2.py |
|---------|---------------|-----------------|
| Robot-centric view | No | Yes |
| TF2 lookup | No | Yes (map→base_link) |
| Rotation correction | No | Yes (rotates image so robot faces up) |
| Image padding | No | Yes |

**Occupancy bins used:**

| Value | Meaning | Colour |
|-------|---------|--------|
| `-1` | Unknown | Grey |
| `0` | Free | White |
| `1–49` | Low confidence | Light grey |
| `50–100` | Wall / obstacle | Black |

---

## 5. Navigation Foundations

### 5.1 r2auto_nav.py

First attempt at fully autonomous navigation. No path planning — the robot picks the direction with the maximum LiDAR range and drives toward it until it hits something, then picks a new direction. Serves as a baseline to measure later improvements against.

**Algorithm:**

```
While running:
  1. Scan LiDAR → find direction of max range
  2. Rotate toward that direction
  3. Drive forward
  4. If front_dist < stop_distance → stop → goto 1
```

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stop_distance` | `0.25` m | Emergency stop threshold |
| `front_angle` | `30°` | Front cone for obstacle detection |
| `rotatechange` | `0.1` rad | Rotation step |
| `speedchange` | `0.05` m/s | Speed step |

**Limitation:** This greedy approach gets stuck in local minima (dead-ends, concave obstacles). All subsequent files replace it with map-based planning.

---

## 6. Core Components Deep Dive

### 6.1 RegulatedPurePursuit

Present in every advanced file. Implements the Pure Pursuit geometric controller with a speed regulation layer that slows the robot on tight curves.

**Control law:**

```
curve     = 2 * sin(angle_diff) / distance
reg_speed = max_speed / (1 + safety_factor * |curve|)
v = clamp(reg_speed, min_speed, max_speed)
w = clamp(v * curve, -max_angular_v, max_angular_v)
```

**Angle handling tiers:**

| Condition | Behaviour |
|-----------|-----------|
| `|angle_diff| ≤ 69°` | Normal arc following |
| `69° < |angle_diff| ≤ 149°` | Creep forward + aggressive steer (1.0 rad/s) |
| `|angle_diff| > 149°` | Return `None` → caller triggers in-place rotation |

**Parameters (final values):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lookaheaddist` | `0.35` m | Distance to look ahead on path |
| `max_speed` | `0.22` m/s | TurtleBot3 Burger max safe speed |
| `min_speed` | `0.04` m/s | Minimum creep speed |
| `max_angular_v` | `0.60` rad/s | Normal steering cap |
| `max_angular_v_hard` | `1.0` rad/s | Aggressive steering for tight turns |
| `safety_factor` | `3.0` | Speed reduction on curves |
| `slow_turn_threshold` | `1.20` rad (~69°) | Sharp-turn boundary |
| `rotate_threshold` | `2.60` rad (~149°) | Full-stop-and-rotate boundary |

---

### 6.2 MapNode

Lightweight waypoint struct used throughout A\* and BFS. Grid coordinates stored as floats to support sub-cell shifted positions from the wall-repulsion step.

```python
class MapNode:
    x: float          # column (pooled grid or map metres)
    y: float          # row
    parent: MapNode   # back-pointer for path reconstruction
```

`__eq__` and `__hash__` use `int(x), int(y)` so nodes snap to integer cells in sets and dicts regardless of floating-point sub-cell offsets.

`generate_neighbours()` returns 4-connected neighbours (N/S/E/W) used only in BFS frontier search. A\* uses the hardcoded `DIRECTIONS_8` constant for 8-connectivity.

---

## 7. Test / Prototype Progression

### 7.1 r2autopilot.py

**Role:** Navigation v1 — introduces BFS path planning, RPP path following, polar docking, and the full state machine. The skeleton that all later files are derived from.

**What's new over r2auto_nav:**
- BFS replaces the greedy direction-picker for proper frontier-based path planning.
- Regulated Pure Pursuit replaces raw `cmd_vel` for smooth path following.
- LiDAR is split into `front / left / back / right` sector arrays.
- A state machine (12 states) replaces the single-loop structure.
- Polar-arc docking controller introduced.
- RViz markers published for debugging.

**Polar docking gains:**

| Gain | Value | Effect |
|------|-------|--------|
| `k_rho` | `0.2` | Forward speed proportional to distance |
| `k_alpha` | `1.6` | Steering proportional to lateral angle |
| `k_beta` | `-0.40` | Heading correction (stabilises approach angle) |
| `target_dist` | `0.25` m | Stop distance from marker |

The β term adds a correction based on the desired final heading, making the robot arrive more perpendicular to the marker face. This was later dropped in favour of a simpler 2-gain controller.

---

### 7.2 r2fulltest.py

**Role:** Navigation v2 — BFS with frontier scoring.

**Key addition:** Instead of taking the first frontier BFS finds, up to `MAX_CANDIDATES = 30` frontier cells are collected and scored:

```python
score = unknown_neighbor_count + wall_distance * 0.5
```

This biases exploration toward large open frontiers and away from narrow gaps. Also adds binary dilation wall inflation before computing the wall-distance map.

---

### 7.3 polardocking.py

**Role:** Standalone docking prototype — no exploration, just the docking controller in isolation.

Subscribes to `/usbcam1_poses` as a `PoseArray` (pre-processed marker poses from a dedicated ArUco node) rather than the raw `ArucoMarkers` message used in the integrated files.

**3-state mini state machine:**

```
APPROACH → FINAL_ALIGN → DONE
```

**Control law gains:**

| Gain | Value |
|------|-------|
| `k_rho` | `0.30` |
| `k_alpha` | `0.80` |
| `k_beta` | `-0.15` |
| `max_v` | `0.12` m/s |
| `max_w` | `0.50` rad/s |
| `fov_loss_timeout` | `1.5` s |

**Lesson learned:** The β term at `-0.15` was too weak and caused oscillation on noisy pose estimates. Later files drop β entirely and use a cleaner 2-gain controller.

---

### 7.4 r2livecode.py

**Role:** Navigation v3 — first version deployed on the physical robot.

**Key changes from r2autopilot:**
- `STOP_DISTANCE` raised to `0.35` m for more conservative obstacle avoidance.
- `SIDE_THRESHOLD` reduced to `0.20` m.
- Docking transitions happen directly from `marker_callback` (not polled in controller loop).
- `SEARCHING` state added: before declaring no marker found, the robot spins a full 360° so the camera sweeps the area.

---

### 7.5 r2livecode(frontierscoring,wallinflation).py

**Role:** Navigation v3.5 — frontier scoring and wall inflation added to the live robot code.

Adds the frontier scoring from `r2fulltest.py` and wall inflation via `binary_dilation` from `scipy.ndimage`.

**New path-quality features:**
- Wall-repulsion shift: each waypoint is pushed away from nearby walls using an inverse-distance force, capped at `maxshift = 1.5` cells.
- Wall-aware smoothing: moving-average with a 5-point window, with each smoothed point validated against the pooled map before acceptance.

---

### 7.6 new.py

**Role:** 3-phase docking prototype using BFS.

Introduces the 3-phase docking alternative to the polar arc approach. The staging point is computed from the marker's quaternion pose:

```python
# Marker normal in camera frame = (0, 0, 1)
# Rotate by marker quaternion to get normal in map frame
nx, ny, _ = quat_rotate_vector(qx, qy, qz, qw, 0, 0, 1)
staging_x = marker_x + nx * 0.4
staging_y = marker_y + ny * 0.4
```

The robot navigates to this staging point, rotates to face the marker, then drives straight in.

---

### 7.7 3phasedocking.py

**Role:** 3-phase docking v2 — refined quaternion handling and progressive approach.

Extends `new.py` with more robust quaternion extraction from the ArUco message and better recovery when the marker is lost mid-dock (falls back to SEARCHING).

---

## 8. fulltest Series — Complete Analysis

The `fulltest(...)` files represent the final integration testing phase — each combines A\* pathfinding with one of the docking strategies to determine which performs best on hardware.

### Common fulltest Infrastructure

**Map downscaling (3× pooling):**

```
Raw OccupancyGrid (e.g. 480×480 @ 0.05m/cell)
        │  max-pool every 3×3 block
        ▼
Pooled grid (160×160 @ 0.15m/cell)
```

Using `max` pooling ensures any wall occupancy within a 3×3 block propagates to the pooled cell — conservative and safe.

**Wall distance computation:**

```python
binary_walls  = (occ_pooled >= WALL_THRESHOLD)
dilated_walls = binary_dilation(binary_walls, iterations=1)
wall_dist     = distance_transform_edt(1 - dilated_walls)
```

**LiDAR sector splitting:**

```
360° scan → split by angular proportions:
  front = ±40° around 0°  (front_fov = 80°)
  left  = 80° arc after front-left
  back  = 80° arc after left
  right = 80° arc before front-right
```

---

### 8.1 fulltest(astar).py

**Purpose:** Verify that A\* produces better, more direct paths than BFS frontier search alone.

**A\* cost function:**

```
f(n) = g(n) + h(n)

g(n) = move_cost        (1.0 straight, 1.414 diagonal)
     + proximity_penalty (5.0 / max(wall_dist, 0.5))

h(n) = Euclidean distance to goal
```

The `proximity_penalty` routes the path through corridor centres rather than scraping along walls. The factor `5.0` was tuned empirically.

**8-directional movement:** Diagonal moves cost `√2 ≈ 1.414`. Without diagonals, paths on open ground take Manhattan-geometry zigzags, increasing travelled distance by up to 41%.

**Path post-processing pipeline (all 4 steps):**

```
Raw A* path → LOS pruning → Wall-repulsion shift
           → Convert to map coordinates → Wall-aware smoothing
           → Final path (MapNode list, map-frame metres)
```

---

### 8.2 fulltest(polardocking).py

**Purpose:** BFS navigation combined with the full polar-arc docking controller.

**Polar-arc docking controller (2-gain version):**

The β term is dropped here. The final controller is:

```
rho   = sqrt(marker_x² + marker_z²)
alpha = atan2(marker_x, marker_z)

v = k_rho  * rho
w = -k_alpha * alpha

v = clamp(v, 0, dock_max_v)
w = clamp(w, -dock_max_w, dock_max_w)
```

**Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `dock_k_rho` | `0.30` | Forward gain |
| `dock_k_alpha` | `1.8` | Steering gain |
| `dock_max_v` | `0.12` m/s | Speed cap |
| `dock_max_w` | `0.50` rad/s | Angular cap |
| `dock_target_dist` | `0.30` m | Stop distance |
| `dock_lost_timeout` | `0.75` s | Abort if marker invisible |

**Marker gating logic:**

```python
alpha_check = atan2(marker_x, marker_z)
if abs(alpha_check) > radians(10):
    # Too wide — ignore, keep exploring
    return
```

This 10° gate prevents starting a docking approach from an oblique angle.

---

### 8.3 fulltest(astarpolardocking).py

**Purpose:** Replace BFS with A\* in the polar-docking pipeline for shorter, more direct paths.

| Feature | fulltest(polardocking) | fulltest(astarpolardocking) |
|---------|----------------------|---------------------------|
| Frontier search | BFS | A\* to BFS-found frontier |
| Goal navigation | BFS | A\* to specific goal |
| Path quality | Medium | High (LOS pruning + wall repulsion) |
| Docking | Polar arc | Same polar arc |

**A\* fallback strategy:**

```python
# Try A* with inflation first
path, found = _astar(..., use_inflation=True)

if not found:
    # Retry without inflation — handles narrow gaps
    path, found = _astar(..., use_inflation=False)
```

---

### 8.4 fulltest(astar3phasedocking).py

**Purpose:** A\* pathfinding combined with the 3-phase docking approach as a comparison point against polar arc.

**Phase 1 — DOCK_NAV:**

```python
# Transform marker pose to map frame via TF2
marker_map_x, marker_map_y = transform_to_map(marker_pose)

# Staging point 0.4m along marker normal
nx, ny, _ = quat_rotate_vector(qx, qy, qz, qw, 0, 0, 1)
staging_x = marker_map_x + nx * 0.4
staging_y = marker_map_y + ny * 0.4

# A* to staging point (allow_unknown=True)
self.dock_path = planroute(goal=staging_goal, allow_unknown=True)
```

`allow_unknown=True` is critical — the staging point is often in unvisited map area.

**Phase 2 — DOCK_ROTATE:** Rotate in place to face the marker until `|error| < 0.05` rad.

**Phase 3 — DOCK_STRAIGHT:** Drive forward at `0.08` m/s until front LiDAR ≤ `STOP_DISTANCE`.

**3-phase vs polar arc comparison:**

| Criterion | 3-Phase | Polar Arc |
|-----------|---------|-----------|
| Requires map frame marker pose | Yes | No |
| Works from any approach angle | No (needs staging) | Yes |
| Sensitivity to marker pose noise | High | Low |
| Final alignment quality | Excellent | Good |
| Recovery if marker lost | Hard | Easy |
| Complexity | High | Medium |
| Tuning required | Low (no gains) | Medium |

---

## 9. Final Evaluation Code

**File: `r2CDE2310_FINAL.py`**

This is the code that was run during the actual evaluation. It is based on `fulltest(astarpolardocking).py` with two key changes made on the day:

1. **`MISSION_COMPLETE` state removed** — the check for both stations being complete was commented out so the robot continues exploring after servicing both stations, ensuring full map coverage regardless. The robot only stops when no more frontiers are found.

2. **On-the-fly parameter tuning** — several navigation and docking parameters were fine-tuned during the final run (see v1.5.3 in CHANGELOG.md).

This means the final code **always undocks and resumes exploration** after each station, with no terminal state — a deliberate choice to maximise map coverage under time pressure.

---

## 10. Algorithm Reference

### 10.1 A\* Pathfinding

```
f(n) = g(n) + h(n)

g(n) = move_cost (1.0 cardinal / 1.414 diagonal)
     + proximity_penalty (5.0 / max(wall_dist, 0.5))

h(n) = Euclidean distance to goal

Traversability per cell:
  cell_val >= WALL_THRESHOLD (50)          → blocked
  wall_dist <= INFLATE_RADIUS + inflation  → skip
  cell_val == -1 (unknown) + no flag       → skip
```

---

### 10.2 BFS Frontier Exploration

```
Frontier cell = unknown cell (val=-1) adjacent to known free cell

Score = unknown_neighbor_count + wall_dist_of_parent * 0.5

Collect up to MAX_CANDIDATES → pick highest score.

Fallback: random known free cell at distance > 5 pooled cells.
```

---

### 10.3 Polar-Arc Docking Controller

```
rho   = sqrt(marker_x² + marker_z²)   [distance]
alpha = atan2(marker_x, marker_z)      [lateral bearing]

v = k_rho * rho
w = -k_alpha * alpha

Termination: rho <= dock_target_dist → STATION

Loss recovery (elapsed <= dock_lost_timeout):
    rotate toward last known alpha at 0.10 rad/s

Loss recovery (elapsed > dock_lost_timeout):
    abort → PLANNING
```

---

### 10.4 3-Phase Docking Sequence

```
DOCK_NAV:
  1. Transform marker pose to map frame
  2. Compute staging point = marker_pos + normal * 0.4m
  3. A* to staging point (allow_unknown=True)
  4. Follow path with mover(allow_state_change=False)
  → DOCK_ROTATE when close enough

DOCK_ROTATE:
  1. Compute desired_yaw from marker normal
  2. turn_in_place() until |error| < 0.05 rad
  → DOCK_STRAIGHT

DOCK_STRAIGHT:
  1. Drive forward at 0.08 m/s, angular.z = 0
  2. When front_dist <= STOP_DISTANCE → STATION
```

---

### 10.5 Path Post-Processing Pipeline

```
Step 1 — LOS pruning
  For each waypoint i, find furthest j with clear straight line.
  Drop all points between i and j.

Step 2 — Wall-repulsion shift
  For each waypoint, sum inverse-distance forces from nearby walls.
  Nudge waypoint away; cap at maxshift = 1.5 cells.

Step 3 — Convert to map metres
  mx = (pooled_x * 3 + 1.5) * resolution + origin.x

Step 4 — Wall-aware smoothing
  5-point moving average; keep original if smoothed point
  falls inside a wall or closer than 1.5 cells to one.
```

---

## 11. Parameter Reference Tables

### Navigation / Safety

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STOP_DISTANCE` | `0.30` m (final) | Front obstacle stop threshold |
| `SIDE_THRESHOLD` | `0.25` m (final) | Side obstacle nudge threshold |
| `GOAL_THRESHOLD` | `0.20` m | Distance to declare goal reached |
| `WALL_THRESHOLD` | `50` | OccupancyGrid wall cell value |
| `INFLATE_RADIUS` | `0` (final) / `2` (earlier) | Clearance cells around walls in A* |
| `front_fov` | `80°` normal, `110°` recovery | Front LiDAR cone width |

### Path Planning

| Parameter | Value | Description |
|-----------|-------|-------------|
| `proximity_penalty` | `5.0 / wall_dist` | A\* cost added near walls |
| `wallinfdist` | `3` cells | Wall repulsion search radius |
| `maxshift` | `1.5` cells | Max wall-repulsion displacement |
| Smoothing window | `5` waypoints | Moving average half-width |
| `MAX_CANDIDATES` | `8` (final) / `30` (earlier) | BFS frontier candidates to score |

### Regulated Pure Pursuit

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lookaheaddist` | `0.35` m | Lookahead distance |
| `max_speed` | `0.22` m/s | Maximum forward speed |
| `min_speed` | `0.04` m/s | Minimum creep speed |
| `max_angular_v` | `0.60` rad/s | Normal angular cap |
| `max_angular_v_hard` | `1.00` rad/s | Aggressive turn cap |
| `safety_factor` | `3.0` | Speed reduction coefficient |
| `slow_turn_threshold` | `1.20` rad | Threshold for aggressive steering |
| `rotate_threshold` | `2.60` rad | Threshold for full stop & rotate |

### Polar-Arc Docking

| Parameter | Final value | Earlier values | Description |
|-----------|------------|----------------|-------------|
| `dock_k_rho` | `0.30` | `0.20` | Forward gain |
| `dock_k_alpha` | `1.8` | `1.6` | Steering gain |
| `dock_k_beta` | — (dropped) | `-0.40` | Heading correction (removed) |
| `dock_max_v` | `0.12` m/s | `0.12` m/s | Speed cap |
| `dock_max_w` | `0.10` rad/s | `0.50` rad/s | Angular cap (tightened for final) |
| `dock_target_dist` | `0.30` m | `0.25` m | Stop distance from marker |
| `dock_lost_timeout` | `0.75` s | `1.5` s | Abort if marker invisible |

### Recovery / Escape

| Parameter | Value | Description |
|-----------|-------|-------------|
| `escape_duration` | `1.0` s | Forward escape duration after recovery |
| `escape_speed` | `0.15` m/s | Forward speed during escape |
| `turning_timeout` | `8.0` s | Max rotation time before replanning |
| `turn_angle_by` | `π/18` rad (10°) | Step for recovery direction selection |
| `boink` limit | `3` | Consecutive hits before goal drop |

### Station / Undocking

| Parameter | Value | Description |
|-----------|-------|-------------|
| `undock_duration` | `2.0` s | Reverse duration after station |
| `undock_speed` | `-0.08` m/s | Reverse speed |
| `valid_station_ids` | `[0, 1]` | ArUco IDs to dock at |

---

## 12. Evolution Diagram

```
Phase 1 — Bring-up & teleop
  r2e2.py, r2scanner.py, r2mover.py, r2moverotate.py
  r2occupancy.py, r2occupancy2.py

Phase 2 — Basic autonomy
  r2auto_nav.py       (greedy direction-picker, no map)

Phase 3 — Path planning introduction
  r2autopilot.py      (BFS + RPP + polar dock + state machine)
  r2fulltest.py       (BFS frontier scoring + wall inflation)

Phase 4 — Docking experiments
  polardocking.py     (standalone polar dock, PoseArray input)
  r2livecode.py       (first robot deployment)
  r2livecode(frontierscoring,wallinflation).py

Phase 5 — 3-phase docking experiments
  new.py              (3-phase docking v1)
  3phasedocking.py    (3-phase docking v2, refined quaternion)

Phase 6 — fulltest integration & comparison
  fulltest(astar).py
  fulltest(polardocking).py
  fulltest(astarpolardocking).py
  fulltest(astar3phasedocking).py

Phase 7 — Final evaluation
  r2CDE2310_FINAL.py  ← EVALUATION CODE (A* + polar arc, no MISSION_COMPLETE)
```

---

## 13. setup.py

Standard ROS2 Python package manifest. Registers the package as `auto_nav` with `ament_python` and declares all entry points so `ros2 run auto_nav <node>` works.

```python
entry_points={
    'console_scripts': [
        'r2mover      = auto_nav.r2mover:main',
        'r2moverotate = auto_nav.r2moverotate:main',
        'r2auto_nav   = auto_nav.r2auto_nav:main',
        # ... etc
    ],
},
```

After any change to `setup.py`, run `colcon build --symlink-install` in the workspace root to apply changes.

---
