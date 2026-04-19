# Software  

This folder contains the codebase deployed on both the external laptop as well as the TurtleBot3 Raspberry Pi 4B.

# Path Planning — A*

When the robot has a specific goal (a BFS-found frontier cell or a docking staging point), A* finds the shortest safe path through the occupancy grid.

## Map Preprocessing

Before any search, the raw SLAM occupancy grid is preprocessed into a coarser, safer representation:

```
Raw OccupancyGrid  (e.g. 480 × 480 cells @ 0.05 m/cell)
         │
         │  3× max-pool (every 3×3 block → single cell, take max value)
         ▼
  Pooled grid  (~160 × 160 cells @ 0.15 m/cell)
         │
         │  binary_dilation(walls, iterations=1)
         │  — thickens single-pixel wall features that pooling might thin out
         ▼
  Dilated wall mask
         │
         │  distance_transform_edt(~dilated_walls)
         ▼
  wall_dist  — each cell holds its Euclidean distance (in cells) to the nearest wall
```

**Why max-pool?** Taking the maximum occupancy value per block means any wall touching a 3×3 region propagates to the pooled cell — conservative and safe. Free-space only registers as free if the entire block is clear.

**Why binary dilation before EDT?** Thin walls (single-pixel features common in SLAM maps) can disappear at the pooled resolution. Dilating before computing wall distances ensures the planner sees these features and routes away from them.

**Resolution trade-off:** A* on the 160×160 pooled grid is roughly 9× faster than running it on the full 480×480 grid, with negligible path quality loss since the robot itself is ~0.18 m wide and the 0.15 m pooled cell size is already sub-robot.

---

## Cost Function

```
f(n) = g(n) + h(n)

g(n) = move_cost + proximity_penalty
     = (1.0 or 1.414)  +  (5.0 / max(wall_dist, 0.5))

h(n) = Euclidean distance to goal   [admissible — never overestimates]
```

Move cost uses true Euclidean cost for diagonal moves:

| Move direction | move_cost |
|----------------|-----------|
| Cardinal (N/S/E/W) | 1.000 |
| Diagonal (NE/NW/SE/SW) | 1.414 (≈ √2) |

8-directional connectivity (`DIRECTIONS_8`) means paths can take diagonal shortcuts across open space, reducing total path length by up to ~30% compared to 4-connected Manhattan routing.

Proximity penalty is a soft repulsion from walls applied to every traversable cell:

| wall_dist (cells) | Approx clearance | Penalty |
|-------------------|-----------------|---------|
| 0.5 (minimum clamp) | 0.075 m | 10.0 |
| 1 | 0.15 m | 5.0 |
| 2 | 0.30 m | 2.5 |
| 3 | 0.45 m | 1.67 |
| 5 | 0.75 m | 1.0 |
| 10 | 1.50 m | 0.5 |

This creates a potential field that naturally centres paths in corridors. A path hugging a wall at `wall_dist = 1` pays +5.0 per step; the same path shifted one cell toward the centre drops to +2.5 — a 50% penalty reduction. The planner steers away from walls without ever explicitly forbidding near-wall cells.

---

## Wall Inflation (INFLATE_RADIUS)

On top of the soft penalty, a hard exclusion zone can be enabled:

```
INFLATE_RADIUS = 2   (pooled cells ≈ 0.30 m physical clearance)

During search:
  if wall_dist[ny, nx] ≤ INFLATE_RADIUS  AND  (ny, nx) ≠ goal:
      skip this cell entirely
```

This hard-excludes cells within 0.30 m of any wall, giving the robot a guaranteed clearance margin. The goal cell itself is exempted — otherwise goals near walls (like staging points 0.40 m from a dock) would be unreachable.

**Two-pass fallback:**

```
Attempt 1:  A* with inflation ON  (INFLATE_RADIUS = 2)
                │
                │ if no path found (narrow corridor too tight for hard exclusion)
                ▼
Attempt 2:  A* with inflation OFF  (proximity penalty still active)
```

The fallback handles the common case where the only valid route passes through a corridor that is physically navigable but falls inside the inflation radius on the coarse pooled grid.

---

## Unknown Space Handling

By default A* refuses to route through unknown cells (occupancy = −1), keeping the robot in explored territory. One exception: docking. When navigating to a staging point, `allow_unknown=True` is passed:

```python
self.planroute(goal=self.dock_staging_point, allow_unknown=True)
```

The staging point is always in front of the marker — often in a region the robot hasn't mapped yet (it just spotted the marker from across the room). Without this flag, the staging point would be permanently unreachable.

---

## Path Post-Processing

A* returns a dense sequence of pooled-grid cells. Four post-processing stages convert this into a smooth, wall-safe path in map coordinates:

```
Raw A* output:  [(x0,y0), (x1,y1), (x2,y2), ..., (xN,yN)]  (pooled cells)
        │
        ▼  ── Stage 1: LOS Pruning ──────────────────────────────────────
        │
        │   For each waypoint i, find the furthest waypoint j > i
        │   where the straight line i→j:
        │     • passes no cell with occupancy ≥ WALL_THRESHOLD (50)
        │     • passes no cell with wall_dist ≤ 2.0
        │   Drop all waypoints between i and j.
        │
        │   Sampled at 3× the segment length for sub-cell accuracy.
        │
        ▼  ── Stage 2: Wall-Repulsion Shift ─────────────────────────────
        │
        │   For each remaining waypoint (gx, gy):
        │     search all wall cells within wallinfdist = 3 cells
        │     for each wall cell at offset (dx, dy):
        │       dist      = sqrt(dx² + dy²)
        │       repulsion = maxshift / dist  (falls off with distance)
        │       push waypoint away: shiftx -= dx/dist * repulsion
        │                           shifty -= dy/dist * repulsion
        │     apply shift; if shifted position is inside a wall,
        │     fall back to 25% of the shift instead
        │
        ▼  ── Stage 3: Convert to Map Coordinates ────────────────────────
        │
        │   mx = (sx × 3 + 1.5) × resolution + origin.x
        │   my = (sy × 3 + 1.5) × resolution + origin.y
        │   (the +1.5 centres the coordinate within the pooled cell)
        │
        ▼  ── Stage 4: Wall-Safe Moving-Average Smoothing ─────────────────

        For each interior waypoint i:
          window = waypoints[i−2 : i+3]   (5-point window)
          candidate = average(window)
          check candidate position on pooled map:
            if free (0 ≤ val < 50) AND wall_dist > 1.5:  use candidate
            else:                                         keep original

Final output:  MapNode list in map-frame metres, ready for RPP
```

**Why keep the original during smoothing?** A naive moving average can pull waypoints into walls in tight corners. The wall-distance check ensures smoothing only applies where there is genuine clearance to move.

**Post-processing parameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| wallinfdist | 3 cells | Wall repulsion search radius |
| maxshift | 1.5 cells | Maximum repulsion displacement |
| Smoothing window | 5 waypoints | Moving average half-width |
| LOS wall_dist guard | 2.0 cells | Minimum clearance for straight-line shortcuts |

---

# Docking

Two independent docking strategies were implemented and evaluated. Both share the same trigger: when `marker_callback` detects a valid ArUco marker within 2.5 m, the robot stops, clears its current navigation path, and switches to the docking branch of the state machine.

## Marker Detection Gate

Both strategies apply the same angular gate before triggering:

```python
alpha = atan2(marker_x, marker_z)   # bearing in camera frame
if abs(alpha) > radians(10):
    # Marker is too far to the side — ignore, keep navigating
    return
```

If the bearing exceeds ±10°, the detection is ignored and the robot continues on its current navigation path. As the robot moves, the marker naturally enters the ±10° window when the robot is roughly facing it — at which point docking triggers cleanly. This is a passive gate, not active alignment — the robot does not deliberately turn toward the marker.

This prevents the polar arc controller from starting at a wide oblique angle where convergence is slow, and prevents the 3-phase approach from computing a staging point from noisy pose data.

---

## Polar Arc Docking

The polar arc controller drives a smooth converging curve from wherever the robot is standing directly to the dock face. No intermediate waypoints, no phases — just a continuous control law running at 10 Hz.

### Camera Frame & Polar Coordinates

```
         Z  (forward, camera axis)
         ▲
         │        ● ArUco marker
         │       ╱
         │  rho ╱
         │     ╱
         │    ╱  α (alpha, positive = marker to the right)
         │   ╱
         ●──────────────────────────► X (right)
       robot

rho   = sqrt(marker_x² + marker_z²)     distance to marker
alpha = atan2(marker_x, marker_z)        lateral bearing (0 = straight ahead)
```

All measurements come directly from the ArUco pose in the camera frame — no TF transform, no localisation dependency. The controller steers to zero alpha while reducing rho.

### Control Law

```
v =  dock_k_rho   × rho       →  approach speed, proportional to remaining distance
w = −dock_k_alpha × alpha     →  steer to centre marker (zero lateral offset)

Clamped:
  v = clamp(v, 0, dock_max_v)           always forward, never overshoot
  w = clamp(w, −dock_max_w, dock_max_w)
```

As rho decreases, v decreases proportionally — the robot naturally decelerates on approach. As alpha decreases, w decreases — steering self-corrects. The two channels are independent and stable.

### Why dock_max_w = 0.10 rad/s is deliberately slow

Early tests used `max_w = 0.50 rad/s`. At that rate, small noisy alpha readings caused aggressive angular corrections that made the robot weave and overshoot laterally in the final metre. Capping at 0.10 rad/s forces slow, deliberate steering — the robot takes a wider arc but arrives straight and centred.

Note that with `dock_k_alpha = 1.8` and `dock_max_w = 0.10 rad/s`, the angular output is clamped for virtually any non-zero alpha — the gain drives aggressive centring intent, but the clamp enforces the slow physical rate throughout the approach.

### Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `dock_k_rho` | 0.30 | 1 m away → 0.30 m/s forward; 0.5 m → 0.15 m/s |
| `dock_k_alpha` | 1.8 | High gain intent; always clamped to dock_max_w in practice |
| `dock_max_v` | 0.12 m/s | Below TurtleBot3 slippage threshold on smooth floor |
| `dock_max_w` | 0.10 rad/s | Suppresses jitter; forces smooth arc over weave |
| `dock_target_dist` | 0.30 m | Stop before physical contact; ArUco reads ~0.25–0.35 m at dock |
| `dock_lost_timeout` | 0.75 s | Short enough to abort quickly; long enough to survive a dropped frame |

### Loss Recovery

```
marker_visible = True
    → run control law as above

marker_visible = False,  elapsed ≤ 0.75 s
    → continuously publish slow rotation toward last known alpha direction
      (0.10 rad/s, sign from dock_last_alpha)
    → re-acquire marker as it re-enters camera FOV

marker_visible = False,  elapsed > 0.75 s
    → stopbot()
    → state = 'PLANNING'   (abort, resume exploration)
```

The last-known alpha direction is saved each frame so the recovery rotation turns the robot the correct way — toward where the marker was last seen rather than a random direction.

### State Transitions

```
DRIVING / PLANNING / SEARCHING
    │  marker_callback fires (|alpha| < 10°, dist < 2.5 m)
    ▼
  DOCKING ────────────────────────────────────────────────► RECOVERY
    │  rho ≤ 0.30 m                      (obstacle during approach)
    ▼
  STATION
```

---

## 3-Phase LiDAR Docking

An alternative approach that removes continuous camera dependency after initial detection by navigating to a precomputed staging point and then using LiDAR for the final stop.

### Phase 1 — DOCK_NAV: Navigate to Staging Point

On marker detection, the marker's Z-axis (normal to its face) is extracted from the ArUco quaternion and rotated into the map frame:

```python
nx, ny, _ = quat_rotate_vector(qx, qy, qz, qw,  0, 0, 1)  # marker Z-axis in map frame
staging_x = marker_map_x + 0.4 * nx
staging_y = marker_map_y + 0.4 * ny
```

```
    ┌──────────────────────┐   ← dock wall
    │       ArUco          │
    └──────────●───────────┘
               │ marker normal (Z-axis)
               │  0.4 m
               ●  staging point
               │
               │  A* path
               │
              ●  robot
```

A* navigates to the staging point with `allow_unknown=True`. The 0.40 m offset gives enough room for the robot to align before the final approach.

**Retry logic:** If A* cannot find a path to the exact staging point, the planner retries up to 5 times from progressively closer positions. If the robot is within approximately 0.30–0.35 m of the staging point and planning still fails, it proceeds to rotation anyway. The exact fallback threshold varies slightly across file versions.

### Phase 2 — DOCK_ROTATE: Align to Marker Normal

```python
dock_heading = atan2(ny, nx) + pi   # face toward the marker (normal points away from wall, +π flips it)
```

A proportional controller rotates the robot in place:

```
w = kp × angle_error     (kp = 2.0,  max_w = 0.15 rad/s)
Stop when |angle_error| < 0.08 rad  (~4.6°)
```

### Phase 3 — DOCK_STRAIGHT: Creep to Dock Face

```python
cmd.linear.x  = 0.08    # 8 cm/s — slow creep
cmd.angular.z = 0.0     # no steering by default

# Minor correction if marker still visible:
if marker_visible:
    alpha = atan2(marker_x, marker_z)
    cmd.angular.z = clamp(-0.3 × alpha, -0.15, 0.15)
```

A narrow ±10° LiDAR cone is used for stopping (not the full front sector) to measure the actual distance to the dock wall face rather than nearby side obstacles:

```
Stop condition:  min(center_cone LiDAR) ≤ 0.20 m  →  STATION
```

The 0.20 m stop distance (tighter than polar arc's 0.30 m) is possible because the straight approach is precisely aligned — there is no lateral arc to account for.

---

## Comparison

| | Polar Arc | 3-Phase LiDAR |
|--|-----------|---------------|
| Approach | Smooth continuous curve | Navigate → rotate → creep |
| Camera dependency | Continuous throughout | Only at initial detection |
| Works from any angle | Yes, within ±10° gate | Yes — computes staging point |
| Localisation required | No (camera frame only) | Yes (staging point needs map frame) |
| Tight corridor behaviour | May arc wide in narrow spaces | Predictable straight-line final approach |
| Stop mechanism | Distance from camera (rho ≤ 0.30 m) | LiDAR narrow cone (≤ 0.20 m) |
| Failure modes | Marker lost mid-arc, noisy alpha | Staging unreachable, rotation overshoot |
| Tuning required | k_rho, k_alpha, max_w | Staging distance, rotation tolerance |
| Implementation complexity | Low | High |
| **Used in final evaluation** | **Yes** | No |

The polar arc approach was selected for the final evaluation. It is simpler to reason about, has fewer states that can fail independently, and proved more reliable within the project timeline. The 3-phase approach introduced compounding failure modes: an unreachable staging point would abort the dock entirely, and even small rotation overshoots in Phase 2 cascaded into off-centre approaches in Phase 3. The polar arc controller handles these gracefully — any misalignment is continuously corrected by the alpha feedback term throughout the entire approach.
The polar arc approach was chosen for its simplicity and reliable smooth behaviour. The 3-phase approach introduced more failure modes (staging point unreachable, rotation overshoot) and was harder to tune reliably within the project timeline.
