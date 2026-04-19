# Remote_Laptop  

This folder contains the codebase which was used dueing the final mission as well as some prototype code used during development. It also contains the necessary documentations during the software development stage.

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
9. [Algorithm Reference](#9-algorithm-reference)
   - [A\* Pathfinding](#91-a-pathfinding)
   - [BFS Frontier Exploration](#92-bfs-frontier-exploration)
   - [Polar-Arc Docking Controller](#93-polar-arc-docking-controller)
   - [3-Phase Docking Sequence](#94-3-phase-docking-sequence)
   - [Path Post-Processing Pipeline](#95-path-post-processing-pipeline)
10. [Parameter Reference Tables](#10-parameter-reference-tables)
11. [Evolution Diagram](#11-evolution-diagram)
12. [setup.py](#12-setuppy)

---

## 1. Repository Overview

The codebase is a ROS2 (Humble / Iron) autonomous navigation stack for a TurtleBot3-class robot. The robot must:

1. Map an unknown environment using SLAM.
2. Explore frontiers autonomously until it detects ArUco fiducial markers.
3. Dock at two charging stations (marker IDs 0 and 1) in sequence.
4. Signal a Raspberry Pi over GPIO, wait for an acknowledgement, then undock.
5. Complete the mission when both stations are serviced.

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
| 16 | `fulltest(polardocking).py` | Navigation | ~950 | A\* + polar dock |
| 17 | `fulltest(astarpolardocking).py` | Navigation | ~1000 | A\* + polar v2 |
| 18 | `fulltest(astar3phasedocking).py` | Navigation | ~1050 | A\* + 3-phase dock |
| 19 | `setup.py` | Build | ~30 | ROS2 package config |

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
│  Sensors: LiDAR scan ─► front/left/back/right arrays   │
│           OccupancyGrid ─► occ_pooled (3× downscale)   │
│           ArUco camera ─► marker_x, marker_z           │
│           TF2 map→base_link ─► cur_x, cur_y, cur_yaw  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 State Machine Pattern

All advanced files run a 10 Hz timer (`controller()`) that dispatches to the current state:

```
                        ┌──────────┐
                        │ PLANNING │ ◄─────────────────────────────────┐
                        └────┬─────┘                                    │
                path found   │                                           │
                        ┌────▼─────┐     obstacle        ┌──────────┐  │
                        │ DRIVING  │ ──────────────────► │ RECOVERY │  │
                        └────┬─────┘                      └────┬─────┘  │
              large angle    │                    clear         │        │
                        ┌────▼─────┐                     ┌─────▼─────┐  │
                        │ ALIGNING │                      │ ESCAPING  │──┘
                        └────┬─────┘                      └───────────┘
              aligned        │
                        ┌────▼─────┐    marker seen
                        │ DRIVING  │ ──────────────────────────────────►
                        └──────────┘                                    │
                                                                        │
          ┌─────────────────────────────────────────────────────────── ┘
          │  (docking branch, varies by file — see §7-8)
          ▼
     SEARCHING / DOCKING / DOCK_NAV / DOCK_ROTATE / DOCK_STRAIGHT
          │
          ▼
     STATION ──► WAITING_FOR_PI ──► UNDOCKING ──► PLANNING (next station)
          │
          └────────────────────────────────────────────────────────────►
                                                                MISSION_COMPLETE
```

### 3.3 ROS2 Topic Map

```
┌─────────────────────────────────────────────────────────────┐
│  SLAM / Nav stack (external)                                │
│   /map (OccupancyGrid) ─────────────────────────────────►  │
│   /scan (LaserScan) ────────────────────────────────────►  │
│   TF2 map → base_link ──────────────────────────────────►  │
└─────────────────────────────────────────────────────────────┘
                 │                                     ▲
                 ▼                                     │
         ┌──────────────┐                    ┌────────────────┐
         │  AutoPilot   │──► /cmd_vel ──────►│  Robot (HW)   │
         │    Node      │──► /gpio_commands  │                │
         │              │◄── /rpi_response   └────────────────┘
         │              │◄── /usbcam1_markers
         │              │──► /goal_marker (RViz)
         │              │──► /planned_path (RViz)
         └──────────────┘──► /lookahead_marker (RViz)
```

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

```
Keyboard → rotatebot(delta) ──► target_heading
                                      │
odom_callback ─► current_heading ─────┤
                                      ▼
                               PD angular cmd → /cmd_vel
```

**Heading arithmetic** uses complex numbers to avoid the ±π wraparound problem:

```python
# Converts heading to unit vector, increments angle, reads back atan2
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
| Rotation correction | No | Yes (rotates image so robot is facing up) |
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

Present in every advanced file. Implements the Pure Pursuit geometric controller with a speed regulation layer that slows the robot on tight curves to prevent sliding.

```
         lookaheaddist
         ◄──────────►
Robot ●───────────────● target waypoint
      ▲
      │ cur_yaw
      │
      └──── angle_diff = target_angle − cur_yaw
```

**Control law:**

```
curve   = 2 * sin(angle_diff) / distance     # arc curvature κ
reg_speed = max_speed / (1 + safety_factor * |κ|)
v = clamp(reg_speed, min_speed, max_speed)
ω = clamp(v * κ, -max_angular_v, max_angular_v)
```

**Angle handling tiers:**

| Condition | Behaviour |
|-----------|-----------|
| `|angle_diff| ≤ slow_turn_threshold` (~69°) | Normal arc following |
| `slow_turn_threshold < |angle_diff| ≤ rotate_threshold` (~149°) | Creep forward + aggressive steer |
| `|angle_diff| > rotate_threshold` | Return `None` → caller triggers in-place rotation |

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

**Important:** `__eq__` and `__hash__` use `int(x), int(y)` so nodes snap to integer cells in sets and dicts regardless of floating-point sub-cell offsets.

`generate_neighbours()` returns 4-connected neighbours (N/S/E/W) used only in BFS frontier search. A\* uses the hardcoded `DIRECTIONS_8` constant for 8-connectivity.

---

## 7. Test / Prototype Progression

### 7.1 r2autopilot.py

**Role:** Navigation v1 — introduces A\*, RPP, polar docking, and the full state machine. The "skeleton" that all later files are derived from.

**What's new over r2auto_nav:**
- A\* replaces the greedy direction-picker for proper path planning.
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

The β term distinguishes this from the simpler 2-gain polar controller: it adds a correction based on the desired final heading, making the robot arrive perpendicular to the marker face rather than just centred on it.

---

### 7.2 r2fulltest.py

**Role:** Navigation v2 — BFS with frontier scoring.

**Key addition:** Instead of taking the first frontier BFS finds, up to `MAX_CANDIDATES = 30` frontier cells are collected and scored:

```python
score = unknown_neighbor_count + wall_distance * 0.5
```

This biases exploration toward large open frontiers (high unknown count) and away from narrow gaps (low wall distance). The robot wastes less time squeezing into dead-ends.

**Also adds:** Binary dilation wall inflation before computing the wall-distance map, which thickens single-pixel walls so the planner sees them correctly at the pooled resolution.

---

### 7.3 polardocking.py

**Role:** Standalone docking prototype — no exploration, just the docking controller.

Designed to be run separately from a navigation node to test the docking approach in isolation. Uses `/usbcam1_poses` (a `PoseStamped` topic) rather than `/usbcam1_markers`, so the pose computation is done upstream by a dedicated ArUco node.

**3-state mini state machine:**

```
APPROACH ──► FINAL_ALIGN ──► DONE
```

**Control law (same k_rho / k_alpha / k_beta as r2autopilot but different values):**

| Gain | Value |
|------|-------|
| `k_rho` | `0.30` |
| `k_alpha` | `0.80` |
| `k_beta` | `-0.15` |
| `max_v` | `0.12` m/s |
| `max_w` | `0.50` rad/s |
| `fov_loss_timeout` | `1.5` s |

**Lesson learned:** The β term at `-0.15` was too weak. Later files use `-0.40` or drop β entirely and switch to a 2-gain controller (k_rho + k_alpha only) which proved simpler to tune and more reliable.

---

### 7.4 r2livecode.py

**Role:** Navigation v3 — first version run on the physical robot.

**Key changes from r2autopilot:**
- `STOP_DISTANCE` raised from `0.30` to `0.35` m for more conservative obstacle avoidance.
- `SIDE_THRESHOLD` reduced from `0.25` to `0.20` m.
- Docking transitions happen from marker callback directly (not polled in controller loop).
- `SEARCHING` state added: before declaring "no marker here", the robot spins a full 360° at slow speed so the camera can sweep the area.

---

### 7.5 r2livecode(frontierscoring,wallinflation).py

**Role:** Navigation v3.5 — frontier scoring and wall inflation added to live code.

Adds the frontier scoring from `r2fulltest.py` (§7.2) to the live robot code. Also adds wall inflation via `binary_dilation` from `scipy.ndimage`.

**New path-quality features:**
- Wall-repulsion shift: each pooled-grid waypoint is pushed away from nearby walls using a 1/r² repulsion force, capped at `maxshift = 1.5` cells.
- Wall-aware smoothing: moving-average smoothing with a 5-point window, but each smoothed point is checked against the pooled map — if it falls inside a wall, the original unsmoothed point is kept.

---

### 7.6 new.py

**Role:** 3-phase docking prototype using A\*.

This file introduces the 3-phase docking alternative to the polar arc approach. The core idea: instead of letting the robot approach the marker from any direction and steer with a control law, compute a staging point in front of the marker and execute three clean manoeuvres:

```
Phase 1 — DOCK_NAV:     Navigate to staging point (0.4m in front of marker)
Phase 2 — DOCK_ROTATE:  Rotate in place to face the marker normal
Phase 3 — DOCK_STRAIGHT: Drive straight toward marker at slow speed
```

The staging point is computed from the marker's quaternion pose:

```python
# Marker normal in camera frame = (0, 0, 1)
# Rotate by marker quaternion to get normal in map frame
nx, ny, _ = quat_rotate_vector(qx, qy, qz, qw, 0, 0, 1)
staging_x = marker_x + nx * 0.4
staging_y = marker_y + ny * 0.4
```

A\* (not BFS) is used to navigate to the staging point because it's a specific target, not a frontier.

---

### 7.7 3phasedocking.py

**Role:** 3-phase docking v2 — refined quaternion handling and progressive approach.

Extends `new.py` with:
- More robust quaternion extraction from the ArUco message.
- Progressive straight approach: checks if the robot is still centred on the marker during DOCK_STRAIGHT; if it drifts too far laterally, it replans.
- Better recovery from losing the marker mid-dock (falls back to SEARCHING).

---

## 8. fulltest Series — Complete Analysis

The `fulltest(...)` files represent the final integration testing phase — each combines A\* pathfinding (proven in `new.py`) with one of the docking strategies to determine which performs best on hardware. They are the most complete and best-documented files in the codebase.

### Common fulltest Infrastructure

All four fulltest files share this infrastructure layer:

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
binary_walls = (occ_pooled >= WALL_THRESHOLD)
dilated_walls = binary_dilation(binary_walls, iterations=1)
wall_dist = distance_transform_edt(1 - dilated_walls)
# wall_dist[y,x] = Euclidean distance in cells to nearest wall
```

The dilation step thickens single-pixel walls before computing the EDT, preventing the planner from routing through gaps that are actually impassable at robot scale.

**LiDAR sector splitting:**

```
360° scan → split by angular proportions:
  front  = ±40° around 0°   (front_fov = 80°)
  left   = 80° arc after front-left
  back   = 80° arc after left
  right  = 80° arc before front-right
```

This is computed dynamically based on `total_points / 360.0` (points per degree), so it works correctly regardless of the LiDAR model's angular resolution.

---

### 8.1 fulltest(astar).py

**Purpose:** Verify that A\* produces better, more direct paths than BFS frontier search alone, before adding docking complexity.

**What makes this file special:**

#### A\* Implementation Details

```
Open set: min-heap keyed by f = g + h
g cost components:
  move_cost        = 1.0 (cardinal) or 1.414 (diagonal)
  proximity_penalty = 5.0 / max(wall_dist, 0.5)
h                  = Euclidean distance to goal

Total f = g + h
```

The `proximity_penalty` is the critical addition over vanilla A\*. It penalises cells close to walls, routing the path through the centre of corridors rather than scraping along one wall. The factor `5.0` was tuned empirically — too low and paths hug walls; too high and the planner takes unnecessarily long detours.

**8-directional movement:**

```python
DIRECTIONS_8 = [(-1,0),(1,0),(0,-1),(0,1),
                (-1,-1),(-1,1),(1,-1),(1,1)]
```

Diagonal moves cost `√2 ≈ 1.414`. Without diagonals, paths on open ground are forced to take Manhattan-geometry zigzags which look unnatural and increase travelled distance by up to 41% vs straight-line.

**Path post-processing pipeline (all 4 steps):**

```
Raw A* path (dense waypoints)
         │
         ▼  Step 1: LOS pruning
Remove intermediate waypoints where a straight line to a later
waypoint clears all walls (uses _line_of_sight() with wall_dist check)
         │
         ▼  Step 2: Wall-repulsion shift
Push each pooled-grid waypoint away from walls using 1/r² repulsion
within wallinfdist=3 cells, capped at maxshift=1.5 cells
         │
         ▼  Step 3: Convert to map coordinates
(sx * 3 + 1.5) * resolution + origin  (centre of pooled cell → metres)
         │
         ▼  Step 4: Wall-aware moving-average smoothing
5-point window average; keep original if smoothed point is inside wall
         │
         ▼
Final path (MapNode list, map-frame metres)
```

**State machine (this file uses 2-phase docking approach):**

| State | Trigger In | Trigger Out | Logic |
|-------|-----------|-------------|-------|
| `PLANNING` | init / goal reached / replan | path found → DRIVING | A\* or BFS |
| `DRIVING` | path found | goal reached / obstacle | RPP mover() |
| `ALIGNING` | large angle diff | aligned → DRIVING | turn_in_place() |
| `RECOVERY` | obstacle detected | turn complete | recoveryTurn() |
| `ESCAPING` | recovery done | escape_duration elapsed | drive forward |
| `SEARCHING` | escape after dock / marker not found | marker found / 360° done | spin slowly |
| `DOCKING_APPROACH` | marker detected during DRIVING | close enough → FINAL_ALIGN | polar arc |
| `DOCKING_FINAL_ALIGN` | rho ≤ threshold | aligned → STATION | fine heading align |
| `STATION` | docked | commandSent → WAITING | GPIO command |
| `WAITING_FOR_PI` | command sent | ACK received → UNDOCKING | wait rpi_response |
| `UNDOCKING` | ACK received | undock_duration elapsed | reverse |
| `MISSION_COMPLETE` | both stations done | — | stop |

---

### 8.2 fulltest(polardocking).py

**Purpose:** BFS (not A\*) navigation combined with the full polar-arc docking controller. This file is the direct predecessor of the production code.

**Polar-arc docking controller (2-gain version):**

The β term from earlier files is dropped here. Experiments showed it caused oscillation when the marker pose estimate was noisy. The final controller is:

```
rho   = sqrt(marker_x² + marker_z²)     [distance to marker]
alpha = atan2(marker_x, marker_z)        [bearing: +ve = marker right of heading]

v = k_rho  * rho                         [approach speed]
w = -k_alpha * alpha                     [steering: centring command]

v = clamp(v, 0, dock_max_v)
w = clamp(w, -dock_max_w, dock_max_w)
```

```
Camera frame:
    Z (forward)
    ▲
    │        ● marker
    │       /
    │      /  rho
    │     /
    │    / alpha
    │   /
    ●──────────────► X (right)
  robot
```

**Why camera frame (not map frame)?**

Using the camera frame directly avoids needing a calibrated camera-to-base transform and makes the controller independent of localisation accuracy. The controller steers so that the marker is centred in the camera frame — which means the robot is heading toward the marker face.

**Docking state logic:**

```
marker_visible AND marker_x is not None
  ├── rho ≤ dock_target_dist (0.30m) → STATION  ✓ docked
  ├── rho > 0.50m AND front obstacle → RECOVERY
  └── else → publish v, w

marker not visible AND elapsed ≤ dock_lost_timeout (0.75s)
  └── slow rotation toward last known alpha → re-acquire

marker not visible AND elapsed > dock_lost_timeout
  └── abort → PLANNING
```

**Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `dock_k_rho` | `0.30` | Forward gain |
| `dock_k_alpha` | `1.8` | Steering gain |
| `dock_max_v` | `0.12` m/s | Speed cap |
| `dock_max_w` | `0.10` rad/s | Angular cap (deliberately tight) |
| `dock_target_dist` | `0.30` m | Stop distance |
| `dock_lost_timeout` | `0.75` s | Abort if marker invisible for this long |

Note that `dock_max_w = 0.10` rad/s is deliberately tight — a fast angular velocity during the final approach causes the robot to overshoot and oscillate. The slow, deliberate steering is what makes the arc smooth.

**Marker gating logic (in `marker_callback`):**

```python
# Only trigger docking if alpha is small enough for a clean approach
alpha_check = atan2(marker_x, marker_z)
if abs(alpha_check) > radians(10):
    # Too wide — marker is in peripheral vision, not aligned
    # Ignore and keep exploring; SEARCHING will align us first
    return
```

This 10° gate prevents the robot from starting a docking approach from an oblique angle where the polar controller would struggle.

---

### 8.3 fulltest(astarpolardocking).py

**Purpose:** Replace BFS with A\* in the polar-docking pipeline. The hypothesis: A\* produces shorter, more direct paths to exploration goals, reducing mission time.

**What changes vs fulltest(polardocking).py:**

| Feature | fulltest(polardocking) | fulltest(astarpolardocking) |
|---------|----------------------|---------------------------|
| Frontier search | BFS | A\* to BFS-found frontier |
| Goal navigation | BFS | A\* to specific goal |
| Path quality | Medium | High (LOS pruning + wall repulsion) |
| Compute cost | Low | Medium (A\* heap operations) |
| Docking | Same polar arc | Same polar arc |

**A\* fallback strategy:**

```python
# Try A* with inflation (wall proximity penalties)
path, found = _astar(..., use_inflation=True)

if not found:
    # Retry without inflation — allows routing through narrow gaps
    path, found = _astar(..., use_inflation=False)
```

The two-pass strategy handles the common case where the only path to a goal passes through a gap that the inflation layer incorrectly marks as impassable.

**Proximity penalty tuning:**

```python
proximity_penalty = 5.0 / max(wall_dist[ny, nx], 0.5)
```

At `wall_dist = 1` cell (0.15m clearance): penalty = 5.0  
At `wall_dist = 2` cells (0.30m clearance): penalty = 2.5  
At `wall_dist = 5` cells (0.75m clearance): penalty = 1.0  
At `wall_dist = 10` cells (1.5m clearance): penalty = 0.5

This creates a smooth potential field that steers paths away from walls even when they're technically passable.

---

### 8.4 fulltest(astar3phasedocking).py

**Purpose:** A\* pathfinding combined with the 3-phase docking approach. Final comparison point against the polar-arc approach.

**The 3-phase docking sequence in detail:**

#### Phase 1 — DOCK_NAV (Navigate to staging point)

When a marker is detected:

```python
# Extract marker pose in map frame via TF2
marker_map_x, marker_map_y = transform_to_map(marker_pose)

# Compute staging point 0.4m along marker normal
qx, qy, qz, qw = marker_orientation
nx, ny, _ = quat_rotate_vector(qx, qy, qz, qw, 0, 0, 1)
staging_x = marker_map_x + nx * 0.4
staging_y = marker_map_y + ny * 0.4

# Plan A* path to staging point
staging_goal = MapNode(staging_x, staging_y)
self.dock_path = planroute(goal=staging_goal, allow_unknown=True)
self.state = 'DOCK_NAV'
```

```
          Marker face
        ┌───────────┐
        │           │  ◄── marker normal direction
        └─────●─────┘
              │  0.4 m
              │
              ● staging point
              │
           (robot navigates here via A*)
```

`allow_unknown=True` is critical: the staging point is often in a part of the map the robot hasn't visited yet. Without this flag, A\* refuses to route through unknown cells, making the staging point unreachable.

#### Phase 2 — DOCK_ROTATE (Rotate to face marker)

```python
# Compute desired heading = atan2 of marker normal in map frame
desired_yaw = atan2(ny, nx)  # opposite direction: face the marker
desired_yaw += pi            # point toward the marker, not away

# Rotate until |angle_diff| < 0.05 rad (~3°)
turn_in_place(desired_yaw, current_yaw)
```

The rotation precision threshold of `0.05` rad (≈3°) ensures the robot is well-aligned before the final straight approach. Getting this wrong causes the DOCK_STRAIGHT phase to drift sideways.

#### Phase 3 — DOCK_STRAIGHT (Drive straight to marker)

```python
# Drive straight at slow speed until front LiDAR detects the marker/wall
while front_dist > STOP_DISTANCE:
    cmd.linear.x = 0.08   # slow: 8 cm/s
    cmd.angular.z = 0.0   # no steering
    publish(cmd)

# Dock complete
state = 'STATION'
```

The straight approach avoids the need for any control gains — the robot just drives slowly forward until it hits the stop distance. This is highly reliable because it makes no assumptions about marker pose accuracy.

**3-phase vs polar arc — comparison:**

| Criterion | 3-Phase | Polar Arc |
|-----------|---------|-----------|
| Requires map frame marker pose | Yes | No |
| Works from any approach angle | No (needs staging point) | Yes |
| Sensitivity to marker pose noise | High (staging point calculation) | Low |
| Final alignment quality | Excellent (geometric) | Good (control-law) |
| Recovery if marker lost mid-dock | Hard | Easy (slow search rotation) |
| Complexity | High | Medium |
| Tuning required | Low (no gains) | Medium (k_rho, k_alpha) |

---

## 9. Algorithm Reference

### 9.1 A\* Pathfinding

```
Input:  start (sx,sy), goal (gx,gy), pooled occupancy grid
Output: list of (x,y) tuples, start→goal

f(n) = g(n) + h(n)
g(n) = cumulative cost from start:
       - move_cost: 1.0 (cardinal) or 1.414 (diagonal)
       - proximity_penalty: 5.0 / max(wall_dist, 0.5)
h(n) = Euclidean distance to goal

Traversability check per cell:
  cell_val >= WALL_THRESHOLD (50) → blocked
  wall_dist ≤ INFLATE_RADIUS AND use_inflation → skip
  cell_val == -1 (unknown) AND not allow_unknown → skip
```

**Complexity:** O(n log n) for n pooled cells. In practice the pooled grid is ~160×160 = 25,600 cells — very fast.

---

### 9.2 BFS Frontier Exploration

```
Input:  robot position (sx,sy), pooled occupancy grid
Output: path to best frontier cell

Frontier cell = unknown cell (val=-1) adjacent to a known free cell (0≤val<50)

Scoring: score = unknown_neighbor_count + wall_dist_of_parent * 0.5

Collect up to MAX_CANDIDATES best-scoring frontiers.
Pick highest score → reconstruct path via parent chain.

Fallback: if no frontier found, pick a random known free cell
          at distance > 5 pooled cells from robot.
```

The fallback wandering prevents the robot from stopping when the map is nearly fully explored but a few unknown cells remain in inaccessible locations.

---

### 9.3 Polar-Arc Docking Controller

```
Camera frame (ROS convention):
  X = right
  Z = forward

State variables:
  rho   = sqrt(marker_x² + marker_z²)    distance
  alpha = atan2(marker_x, marker_z)       lateral bearing

Control law (2-gain):
  v = k_rho * rho           approach speed (proportional to distance)
  w = -k_alpha * alpha      steering (centres marker in FOV)

Clamping:
  v = clamp(v, 0, dock_max_v)
  w = clamp(w, -dock_max_w, dock_max_w)

Termination:
  rho ≤ dock_target_dist → STATION (docked)

Loss recovery:
  marker invisible for ≤ dock_lost_timeout:
      rotate toward last known alpha at 0.10 rad/s
  marker invisible for > dock_lost_timeout:
      abort → PLANNING
```

---

### 9.4 3-Phase Docking Sequence

```
Precondition: marker detected in DRIVING/PLANNING state

DOCK_NAV:
  1. Transform marker pose to map frame via TF2
  2. Compute staging point = marker_pos + normal * 0.4m
  3. Run A* to staging point (allow_unknown=True)
  4. Follow path with mover(allow_state_change=False)
  5. When path exhausted → DOCK_ROTATE

DOCK_ROTATE:
  1. Compute desired_yaw = atan2(marker_normal) + π
  2. turn_in_place() until |error| < 0.05 rad
  3. → DOCK_STRAIGHT

DOCK_STRAIGHT:
  1. Drive forward at 0.08 m/s
  2. No angular command (angular.z = 0.0)
  3. When front_dist ≤ STOP_DISTANCE → STATION
```

---

### 9.5 Path Post-Processing Pipeline

```
Step 1 — Line-of-sight pruning:
  for each waypoint i:
    find furthest j > i with clear LOS (no wall, wall_dist > 2.0 everywhere)
    skip waypoints i+1 … j-1

Step 2 — Wall-repulsion shift:
  for each waypoint (gx, gy):
    shiftx = sum over wall cells in ±wallinfdist:  -dx * maxshift / dist²
    shifty = sum over wall cells in ±wallinfdist:  -dy * maxshift / dist²
    new pos = (gx + shiftx, gy + shifty)
    if new pos is free: use it; else use 25% shift as fallback

Step 3 — Convert to map metres:
  mx = (sx * 3 + 1.5) * resolution + origin.x
  my = (sy * 3 + 1.5) * resolution + origin.y

Step 4 — Wall-aware moving-average smoothing:
  for each interior waypoint i:
    average over window of 5 surrounding waypoints
    check averaged point against pooled map:
      if free AND wall_dist > 1.5: use averaged point
      else: keep original point
```

---

## 10. Parameter Reference Tables

### Navigation / Safety

| Parameter | Value | Files | Description |
|-----------|-------|-------|-------------|
| `STOP_DISTANCE` | `0.30` m (final), `0.35` m (livecode) | all | Front obstacle stop threshold |
| `SIDE_THRESHOLD` | `0.25` m (final), `0.20` m (livecode) | all | Side obstacle nudge threshold |
| `GOAL_THRESHOLD` | `0.20` m | all | Distance to declare goal reached |
| `WALL_THRESHOLD` | `50` | all | OccupancyGrid cell value for wall |
| `INFLATE_RADIUS` | `0–2` cells | all | Inflation clearance around walls |
| `front_fov` | `80°` normal, `110°` recovery | all | Front LiDAR cone width |

### Path Planning

| Parameter | Value | Description |
|-----------|-------|-------------|
| `INFLATE_RADIUS` | `0` (final) / `2` (earlier) | Pooled cells blocked near walls |
| `proximity_penalty` | `5.0 / wall_dist` | A\* cost added near walls |
| `wallinfdist` | `3` cells | Wall repulsion search radius |
| `maxshift` | `1.5` cells | Max wall-repulsion displacement |
| Smoothing window | `5` waypoints | Moving average half-width |
| `MAX_CANDIDATES` | `8`–`30` | BFS frontier candidates to score |

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

| Parameter | Value (final) | Earlier values | Description |
|-----------|--------------|----------------|-------------|
| `dock_k_rho` | `0.30` | `0.20` | Forward gain |
| `dock_k_alpha` | `1.8` | `1.6` | Steering gain |
| `dock_k_beta` | — (dropped) | `-0.40` | Heading correction (removed) |
| `dock_max_v` | `0.12` m/s | `0.12` m/s | Speed cap during dock |
| `dock_max_w` | `0.10` rad/s | `0.50` rad/s | Angular cap (tightened) |
| `dock_target_dist` | `0.30` m | `0.25` m | Stop distance from marker |
| `dock_lost_timeout` | `0.75` s | `1.5` s | Abort if marker invisible |

### Recovery / Escape

| Parameter | Value | Description |
|-----------|-------|-------------|
| `escape_duration` | `1.0`–`1.5` s | How long to drive after recovery turn |
| `escape_speed` | `0.15` m/s | Forward speed during escape |
| `turning_timeout` | `8.0` s | Max time to spend rotating before replanning |
| `turn_angle_by` | `π/18` rad (10°) | Step for recovery direction selection |
| `boink` limit | `3` | Consecutive obstacle hits before goal drop |

### Station / Undocking

| Parameter | Value | Description |
|-----------|-------|-------------|
| `undock_duration` | `2.0` s | How long to reverse after station |
| `undock_speed` | `-0.08` m/s | Reverse speed |
| `valid_station_ids` | `[0, 1]` | ArUco IDs to dock at |
| `dock_target_dist` | `0.30` m | Distance at which docking declared complete |

---

## 11. Evolution Diagram

```
Phase 1 — Bring-up & teleop
────────────────────────────
r2e2.py             (ROS2 echo test)
r2scanner.py        (LiDAR bring-up)
r2mover.py          (keyboard drive)
r2moverotate.py     (keyboard drive + heading hold)
r2occupancy.py      (map display)
r2occupancy2.py     (robot-centric map display)

Phase 2 — Basic autonomy
────────────────────────
r2auto_nav.py       (greedy direction-picker, no map)

Phase 3 — Path planning introduction
─────────────────────────────────────
r2autopilot.py      (A* + RPP + polar dock + state machine)
r2fulltest.py       (BFS frontier scoring + wall inflation)

Phase 4 — Docking experiments
──────────────────────────────
polardocking.py     (standalone polar dock with 3 gains)
r2livecode.py       (first robot deployment)
r2livecode          (frontier scoring + wall inflation on robot)
  (frontierscoring,
   wallinflation).py

Phase 5 — 3-phase docking experiments
──────────────────────────────────────
new.py              (3-phase docking v1 with A*)
3phasedocking.py    (3-phase docking v2, refined quaternion)

Phase 6 — fulltest integration & comparison
────────────────────────────────────────────
fulltest(astar).py              (A* quality benchmark)
fulltest(polardocking).py       (BFS + polar arc, baseline)
fulltest(astarpolardocking).py  (A* + polar arc, candidate final)
fulltest(astar3phasedocking).py (A* + 3-phase, candidate final)

Phase 7 — Final production code
──────────────────────────────────
r2livecode.py (current, merged with docking)   ← THIS IS THE FINAL CODE
```

---

## 12. setup.py

Standard ROS2 Python package manifest. Registers the package as `auto_nav` with `ament_python` and declares all entry points (console scripts) so `ros2 run auto_nav <node>` works.

Entry points map Python classes to ROS2 node names. Each navigation file that reaches Phase 5+ has a corresponding entry point.

```python
entry_points={
    'console_scripts': [
        'r2mover       = auto_nav.r2mover:main',
        'r2moverotate  = auto_nav.r2moverotate:main',
        'r2auto_nav    = auto_nav.r2auto_nav:main',
        'r2autopilot   = auto_nav.r2autopilot:main',
        # ... etc
    ],
},
```

After any change to `setup.py`, run `colcon build --symlink-install` in the workspace root to apply the changes.

---

*Generated from source on 2026-04-19. All parameter values are taken directly from the source files and reflect the state of the code at that date.*
