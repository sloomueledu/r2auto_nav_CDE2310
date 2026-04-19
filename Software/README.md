# Software  

This folder contains the codebase deployed on both the external laptop as well as the TurtleBot3 Raspberry Pi 4B.

**Path Planning — A STAR**
When the robot has a specific goal (a docking staging point or a BFS-found frontier), A* is used to find the optimal path through the maze.
Cost function:

f(n) = g(n) + h(n)

g(n) = move_cost (1.0 straight, 1.414 diagonal)
     + proximity_penalty (5.0 / wall_dist)

h(n) = Euclidean distance to goal


The proximity_penalty makes cells close to walls expensive, so the path naturally hugs corridor centres rather than scraping along walls. 8-directional movement is used for smoother paths.

The map is first downscaled 3× (max-pool) to reduce compute cost, and walls are dilated by 1 cell using binary_dilation before computing wall distances — this thickens thin wall features that might otherwise slip through the pooled grid.

**Wall inflation** (controlled by INFLATE_RADIUS): 
Cells within INFLATE_RADIUS pooled cells of any wall are skipped entirely during search. If A* fails with inflation enabled, it retries without — useful for tight corridors.
After A* returns a raw path, it goes through post-processing: LOS pruning removes redundant intermediate waypoints (if a straight line between two non-adjacent waypoints is clear, all points between are dropped), followed by a wall-repulsion shift that nudges each waypoint toward the corridor centre, and finally a wall-safe moving-average smooth that averages nearby waypoints but keeps the original if the smoothed position lands in a wall.

**Docking**
Two docking approaches were implemented and tested.

Polar Arc Docking (used in final evaluation)
The robot uses ArUco pose data from the USB camera. In the camera frame, X = lateral offset and Z = forward distance. Two polar coordinates are computed each frame:

rho   = sqrt(x² + z²)   # distance to marker
alpha = atan2(x, z)      # bearing (0 = straight ahead)


Control law:

v =  k_rho   * rho      # slow as you get close
w = -k_alpha * alpha    # steer to centre marker in view


This produces a smooth converging arc from any starting position — the robot naturally curves toward the marker and decelerates as distance closes. No explicit phases or state transitions are needed within the docking sequence itself.
FOV constraint: docking only triggers if |alpha| < 10°. If the marker exits this window mid-dock, the last known alpha direction is saved and the robot slowly rotates back to re-acquire it. If the marker is lost for more than dock_lost_timeout = 0.75s, docking aborts and the robot resumes exploration.
Gains used:

dock_k_rho   = 0.30
dock_k_alpha = 1.8
dock_max_v   = 0.12 m/s
dock_max_w   = 0.10 rad/s   ← intentionally slow for precision
dock_target  = 0.30m        ← stop distance


3-Phase LiDAR Docking (tested, not used in final)

An alternative approach that navigates to a precomputed staging point rather than relying on continuous camera visibility.
Phase 1 — DOCK_NAV: On marker detection, the marker’s Z-axis normal is extracted from the ArUco orientation quaternion and transformed into the map frame. A staging point 0.4m in front of the marker face is computed:

staging_x = marker_map_x + 0.4 * normal_map_x
staging_y = marker_map_y + 0.4 * normal_map_y


A* navigates to this staging point with retry logic (up to 5 attempts, accepts closest reachable position if staging is unreachable).
Phase 2 — DOCK_ROTATE: At the staging point, the robot rotates in place to face the precomputed dock_heading angle using a slow proportional controller (max_w = 0.15 rad/s).
Phase 3 — DOCK_STRAIGHT: Drives straight at 0.08 m/s while monitoring a narrow ±10° LiDAR cone for wall proximity. Stops when front LiDAR reads ≤ 0.20m. Optional minor angular correction from the camera is applied during the creep.

Comparison



|                        |Polar Arc              |3-Phase LiDAR            |
|------------------------|-----------------------|-------------------------|
|Approach                |Smooth continuous curve|Navigate → rotate → creep|
|Camera needed           |Continuous             |Only at trigger          |
|Works from any angle    |✅ (within ±10° window) |✅ (navigates to staging) |
|Tight corridor behaviour|May arc wide           |More predictable         |
|Complexity              |Low                    |High                     |
|**Used in evaluation**  |**✅**                  |❌                        |

The polar arc approach was chosen for its simplicity and reliable smooth behaviour. The 3-phase approach introduced more failure modes (staging point unreachable, rotation overshoot) and was harder to tune reliably within the project timeline.
