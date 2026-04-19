import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from ros2_aruco_interfaces.msg import ArucoMarkers
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import numpy as np
import math
import random
import heapq
from collections import deque
from scipy.ndimage import distance_transform_edt, binary_dilation
import tf2_ros
from tf2_ros import TransformException, LookupException, ConnectivityException, ExtrapolationException

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
STOP_DISTANCE = 0.30
SIDE_THRESHOLD = 0.25
GOAL_THRESHOLD = 0.20
SCANFILE = 'lidar.txt'
MAPFILE = 'map.txt'
WALL_THRESHOLD = 50
INFLATE_RADIUS = 2          # pooled cells (~0.30m clearance at 0.05 res)
DIRECTIONS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z


# ─── REGULATED PURE PURSUIT ─────────────────────────────────────────────────
class RegulatedPurePursuit():
    def __init__(self):
        self.lookaheaddist = 0.35       # larger = smoother arcs
        self.max_speed = 0.22
        self.min_speed = 0.04
        self.max_angular_v = 0.60       # normal steering
        self.max_angular_v_hard = 1.0   # aggressive steering for sharp turns
        self.safety_factor = 3.0
        self.slow_turn_threshold = 1.20 # >69° = slow-and-turn (not full stop)
        self.rotate_threshold = 2.60    # >149° = full stop (near U-turns only)

    def findpoint(self, cur_x, cur_y, path):
        """Return the first waypoint beyond lookahead distance."""
        target = path[-1]
        for node in path:
            dist = math.sqrt((node.x - cur_x)**2 + (node.y - cur_y)**2)
            if dist > self.lookaheaddist:
                target = node
                break
        return target

    def command(self, cur_x, cur_y, cur_yaw, path):
        twist = Twist()
        if not path or len(path) < 1:
            return twist
        target = self.findpoint(cur_x, cur_y, path)
        dx = target.x - cur_x
        dy = target.y - cur_y
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - cur_yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # Only return None (full stop + rotate) for near-U-turns (>150°)
        if abs(angle_diff) > self.rotate_threshold:
            return None

        distance = math.sqrt(dx**2 + dy**2)

        # Sharp turn (70-150°): creep forward + aggressive steer
        if abs(angle_diff) > self.slow_turn_threshold:
            twist.linear.x = self.min_speed
            turn_sign = 1.0 if angle_diff > 0 else -1.0
            twist.angular.z = turn_sign * self.max_angular_v_hard
            return twist

        # Normal arc-following
        curve = 2.0 * math.sin(angle_diff) / max(0.01, distance)
        reg_speed = self.max_speed / (1.0 + self.safety_factor * abs(curve))
        twist.linear.x = max(self.min_speed, min(self.max_speed, reg_speed))
        reg_angular = twist.linear.x * curve
        twist.angular.z = max(-self.max_angular_v, min(self.max_angular_v, reg_angular))
        return twist


# ─── MAP NODE ────────────────────────────────────────────────────────────────
class MapNode():
    def __init__(self, x, y, parent=None):
        self.x = float(x)
        self.y = float(y)
        self.parent = parent

    def __eq__(self, other):
        if isinstance(other, MapNode):
            return int(self.x) == int(other.x) and int(self.y) == int(other.y)
        return False

    def __hash__(self):
        return hash((int(self.x), int(self.y)))

    def generate_neighbours(self, max_x, max_y):
        """4-connected neighbours (for BFS frontier exploration)."""
        neighbours = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = self.x + dx
            ny = self.y + dy
            if 0 <= nx < max_x and 0 <= ny < max_y:
                neighbours.append(MapNode(nx, ny, parent=self))
        return neighbours


# ─── AUTOPILOT NODE ──────────────────────────────────────────────────────────
class AutoPilot(Node):
    def __init__(self):
        super().__init__('autopilot_node')
        self.rpp_controller = RegulatedPurePursuit()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Publishers ──
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.gpio_pub = self.create_publisher(String, '/gpio_commands', 10)
        self.marker_publisher = self.create_publisher(Marker, 'goal_marker', 10)
        self.path_publisher = self.create_publisher(Path, 'planned_path', 10)
        self.lookahead_publisher = self.create_publisher(Marker, 'lookahead_marker', 10)

        # ── Navigation State ──
        self.path = []
        self.boink = 0
        self.goal = None
        self.state = 'PLANNING'
        self.rotation_start_time = None
        self.escape_direction_locked = None
        self.escape_start_time = None
        self.escape_duration = 1.0          # was 1.5 — quicker escapes
        self.escape_speed = 0.15
        self.front_fov = 90
        self.turning_timeout = 8.0          # was 20 — don't spin for ages
        self.recovery_angle = None
        self.turn_angle_by = (math.pi / 6)
        self.wallinfdist = 3
        self.maxshift = 1.5
        self.pre_recovery_state = None

        # ── Station Tracking ──
        self.valid_station_ids = [0, 1]
        self.responseHeard = False
        self.current_docking_id = None
        self.STATION_A_COMPLETED = False
        self.STATION_B_COMPLETED = False

        self.piResponseSub = self.create_subscription(
            String, 'rpi_response', self.pi_response_callback, 10)
        self.piResponseSub

        self.marker_sub = self.create_subscription(
            ArucoMarkers, '/usbcam1_markers', self.marker_callback, 10)
        self.marker_sub

        # ── Docking — Polar Arc Controller ──
        self.marker_x = None                # latest camera-frame X (right)
        self.marker_z = None                # latest camera-frame Z (forward)
        self.last_seen = self.get_clock().now()
        self.marker_visible = False
        self.dock_target_dist = 0.20        # stop this far from marker
        self.dock_k_rho = 0.30              # forward gain
        self.dock_k_alpha = 1.8             # steering gain (arc curvature)
        self.dock_max_v = 0.12              # max forward speed while docking
        self.dock_max_w = 0.50              # max angular speed while docking
        self.dock_lost_timeout = 3.0        # seconds before aborting if lost
        self.dock_last_alpha = None         # last known alpha for recovery

        # ── Undocking ──
        self.undock_start_time = None
        self.undock_duration = 2.0
        self.undock_speed = -0.08
        self.commandSent = False

        # ── Searching ──
        self.search_last_yaw = None
        self.search_yaw_accumulated = 0.0
        self.search_spin_speed = 0.08       # slow for camera ArUco detection

        # ── LiDAR Sectors ──
        self.front = np.array([])
        self.left = np.array([])
        self.back = np.array([])
        self.right = np.array([])

        # ── Map ──
        self.occ_subscription = self.create_subscription(
            OccupancyGrid, 'map', self.occ_callback, qos_profile_sensor_data)
        self.occ_subscription
        self.occdata = np.empty((0, 0))
        self.res = 0
        self.origin = 0

        # ── LiDAR ──
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.scan_subscription
        self.laser_range = np.array([])

        # ── Main Timer (10 Hz) ──
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.controller)

    # ─── CALLBACKS ───────────────────────────────────────────────────────────

    def pi_response_callback(self, msg):
        response = msg.data
        self.get_logger().info(f'Received response from RPI: {response}')
        if response == 'A COMPLETE':
            self.STATION_A_COMPLETED = True
            self.responseHeard = True
            self.get_logger().info('Station A has been marked as completed.')
        elif response == 'B COMPLETE':
            self.STATION_B_COMPLETED = True
            self.responseHeard = True
            self.get_logger().info('Station B has been marked as completed.')

    def occ_callback(self, msg):
        self.res = msg.info.resolution
        self.origin = msg.info.origin.position
        msgdata = np.array(msg.data)
        self.occdata = msgdata.reshape((msg.info.height, msg.info.width))
        np.savetxt(MAPFILE, self.occdata, fmt='%d')

    def scan_callback(self, msg):
        self.laser_range = np.array(msg.ranges)
        self.laser_range[self.laser_range == 0] = np.nan
        total_points = len(self.laser_range)
        np.savetxt(SCANFILE, self.laser_range)

        ppd = total_points / 360.0
        half_front = int((self.front_fov / 2) * ppd)
        front_right_scan = self.laser_range[-half_front:]
        front_left_scan = self.laser_range[0:half_front]
        self.front = np.concatenate((front_right_scan, front_left_scan))
        left_start = half_front
        left_end = left_start + int(80 * ppd)
        self.left = self.laser_range[left_start:left_end]
        back_start = left_end
        back_end = back_start + int(80 * ppd)
        self.back = self.laser_range[back_start:back_end]
        right_start = back_end
        right_end = total_points - half_front
        self.right = self.laser_range[right_start:right_end]

    def marker_callback(self, msg):
        """Detect ArUco markers.  On detection during exploration,
        immediately switch to DOCKING and let the polar arc controller
        drive a smooth curve to the marker."""
        if not msg.marker_ids:
            self.marker_visible = False
            return

        # Find closest valid (uncompleted) marker
        closest_idx = None
        min_distance = float('inf')
        chosen_id = None

        for i, detected_id in enumerate(msg.marker_ids):
            if detected_id in self.valid_station_ids:
                pose = msg.poses[i]
                dist = math.sqrt(pose.position.x**2 + pose.position.z**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_idx = i
                    chosen_id = detected_id

        if closest_idx is None:
            self.marker_visible = False
            return

        if min_distance > 2.5:
            self.marker_visible = False
            return

        target_pose = msg.poses[closest_idx]
        self.marker_x = target_pose.position.x
        self.marker_z = target_pose.position.z
        self.last_seen = self.get_clock().now()
        self.marker_visible = True
        self.current_docking_id = chosen_id

        # ── Trigger docking only from exploration states ──
        if self.state in ['PLANNING', 'DRIVING', 'ALIGNING', 'SEARCHING']:
            self.get_logger().warn(
                f'Station {chosen_id} detected at {min_distance:.2f}m! '
                f'Starting polar arc dock...')
            self.stopbot()
            self.path = []
            self.goal = None
            self.search_last_yaw = None
            self.dock_last_alpha = None
            self.state = 'DOCKING'

    # ─── UTILITY ─────────────────────────────────────────────────────────────

    @staticmethod
    def _quat_rotate_vector(qx, qy, qz, qw, vx, vy, vz):
        """Rotate vector (vx,vy,vz) by quaternion (qx,qy,qz,qw)."""
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return (vx + qw * tx + qy * tz - qz * ty,
                vy + qw * ty + qz * tx - qx * tz,
                vz + qw * tz + qx * ty - qy * tx)

    @staticmethod
    def _line_of_sight(x0, y0, x1, y1, occ_pooled, wall_dist):
        """Return True if a straight line between two pooled-grid cells is
        free of walls and stays at least 1 cell away from any wall."""
        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 0.5:
            return True
        steps = int(dist * 3) + 1
        for i in range(steps + 1):
            t = i / max(steps, 1)
            px = int(round(x0 + t * dx))
            py = int(round(y0 + t * dy))
            if 0 <= py < occ_pooled.shape[0] and 0 <= px < occ_pooled.shape[1]:
                if occ_pooled[py, px] >= WALL_THRESHOLD:
                    return False
                if wall_dist[py, px] <= 2.0:
                    return False
            else:
                return False
        return True

    def get_orientation(self):
        transform = self.tf_buffer.lookup_transform(
            'map', 'base_link', rclpy.time.Time())
        _, _, current_angle = euler_from_quaternion(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w)
        return (transform.transform.translation.x,
                transform.transform.translation.y,
                current_angle)

    def stopbot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)

    # ─── PATH PLANNING ───────────────────────────────────────────────────────

    def planroute(self, goal=None, allow_unknown=False):
        occ_grid = self.occdata
        if occ_grid.shape[0] == 0:
            return []

        # ── Downscale by 3 ──
        pad_y = (3 - occ_grid.shape[0] % 3) % 3
        pad_x = (3 - occ_grid.shape[1] % 3) % 3
        occ_grid = np.pad(occ_grid, ((0, pad_y), (0, pad_x)),
                          mode='constant', constant_values=-1)
        height, width = occ_grid.shape
        occ_pooled = occ_grid.reshape(
            height // 3, 3, width // 3, 3).max(axis=(1, 3))
        pooled_h, pooled_w = occ_pooled.shape

        # ── Wall distance — dilate walls first to catch thin features ──
        binary_walls = (occ_pooled >= WALL_THRESHOLD).astype(np.uint8)
        # Dilate walls by 1 cell — thickens thin walls so planner respects them
        dilated_walls = binary_dilation(binary_walls, iterations=1).astype(np.uint8)
        wall_dist = distance_transform_edt(1 - dilated_walls)

        # ── Locate robot on pooled map ──
        try:
            if not self.tf_buffer.can_transform(
                    'map', 'base_link', rclpy.time.Time()):
                self.get_logger().warn('TF not available. Retrying...')
                return []
            cur_x, cur_y, _ = self.get_orientation()
            obotloc_x = (cur_x - self.origin.x) / self.res
            obotloc_y = (cur_y - self.origin.y) / self.res
            pbotloc_x = max(0, min(int(obotloc_x) // 3, pooled_w - 1))
            pbotloc_y = max(0, min(int(obotloc_y) // 3, pooled_h - 1))

            # Adjust if robot appears inside a wall on the pooled map
            if occ_pooled[pbotloc_y, pbotloc_x] >= WALL_THRESHOLD:
                best, best_d = None, float('inf')
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        ny, nx = pbotloc_y + i, pbotloc_x + j
                        if (0 <= ny < pooled_h and 0 <= nx < pooled_w
                                and occ_pooled[ny, nx] < WALL_THRESHOLD):
                            d = math.sqrt(i**2 + j**2)
                            if d < best_d:
                                best_d = d
                                best = (ny, nx)
                if best:
                    pbotloc_y, pbotloc_x = best
                else:
                    self.get_logger().error(
                        'Robot stuck inside walls on pooled map.')
                    return []
        except (LookupException, ConnectivityException,
                ExtrapolationException) as e:
            self.get_logger().error(f'Transform Lookup Failed: {e}')
            return []

        self.get_logger().info(
            f'Pooled bot: ({pbotloc_x}, {pbotloc_y})  '
            f'Map bot: ({cur_x:.2f}, {cur_y:.2f})')

        # ── Compute goal on pooled map ──
        pgoal_x, pgoal_y = -1, -1
        if goal is not None:
            pgoal_x = max(0, min(
                int((goal.x - self.origin.x) / self.res) // 3, pooled_w - 1))
            pgoal_y = max(0, min(
                int((goal.y - self.origin.y) / self.res) // 3, pooled_h - 1))

            # Snap goal to nearest free cell if it lands on wall/unknown
            goal_val = occ_pooled[pgoal_y, pgoal_x]
            if goal_val >= WALL_THRESHOLD or goal_val == -1:
                best, best_d = None, float('inf')
                for i in range(-5, 6):
                    for j in range(-5, 6):
                        ny, nx = pgoal_y + i, pgoal_x + j
                        if (0 <= ny < pooled_h and 0 <= nx < pooled_w
                                and 0 <= occ_pooled[ny, nx] < WALL_THRESHOLD):
                            d = math.sqrt(i**2 + j**2)
                            if d < best_d:
                                best_d = d
                                best = (ny, nx)
                if best:
                    old_gx, old_gy = pgoal_x, pgoal_y
                    pgoal_y, pgoal_x = best
                    self.get_logger().warn(
                        f'Goal ({old_gx},{old_gy}) in wall/unknown '
                        f'(val={int(goal_val)}). '
                        f'Snapped to free cell ({pgoal_x},{pgoal_y})')
                else:
                    self.get_logger().error(
                        'No free cell found near goal on pooled map.')
                    return []

            self.get_logger().info(f'Pooled goal: ({pgoal_x}, {pgoal_y})')

        # ══════════════════════════════════════════════════════════════════════
        #  PATH SEARCH
        # ══════════════════════════════════════════════════════════════════════
        found_path = False
        raw_path = []       # list of (gx, gy) tuples in pooled coords

        if goal is not None:
            # ── A* to specific goal ──
            raw_path, found_path = self._astar(
                pbotloc_x, pbotloc_y, pgoal_x, pgoal_y,
                occ_pooled, wall_dist, pooled_w, pooled_h,
                use_inflation=True, allow_unknown=allow_unknown)

            if not found_path:
                self.get_logger().warn(
                    'A* failed with inflation.  Retrying without...')
                raw_path, found_path = self._astar(
                    pbotloc_x, pbotloc_y, pgoal_x, pgoal_y,
                    occ_pooled, wall_dist, pooled_w, pooled_h,
                    use_inflation=False, allow_unknown=allow_unknown)
        else:
            # ── BFS for nearest frontier ──
            raw_path, found_path = self._bfs_frontier(
                pbotloc_x, pbotloc_y,
                occ_pooled, wall_dist, pooled_w, pooled_h)

        if not found_path or len(raw_path) < 1:
            self.get_logger().info('No path found.')
            return []

        # ══════════════════════════════════════════════════════════════════════
        #  POST-PROCESSING  (all in pooled-grid space)
        # ══════════════════════════════════════════════════════════════════════

        # Step 1 — LOS pruning: remove unnecessary intermediate waypoints
        if len(raw_path) > 2:
            pruned = [raw_path[0]]
            i = 0
            while i < len(raw_path) - 1:
                best_j = i + 1
                for j in range(i + 2, len(raw_path)):
                    if self._line_of_sight(
                            raw_path[i][0], raw_path[i][1],
                            raw_path[j][0], raw_path[j][1],
                            occ_pooled, wall_dist):
                        best_j = j
                    else:
                        break
                pruned.append(raw_path[best_j])
                i = best_j
            raw_path = pruned

        # Step 2 — Wall-repulsion shift
        shifted = []
        for (gx, gy) in raw_path:
            shiftx, shifty = 0.0, 0.0
            for dy in range(-self.wallinfdist, self.wallinfdist + 1):
                for dx in range(-self.wallinfdist, self.wallinfdist + 1):
                    ny, nx = gy + dy, gx + dx
                    if (0 <= ny < pooled_h and 0 <= nx < pooled_w
                            and occ_pooled[ny, nx] >= WALL_THRESHOLD):
                        sd = max(0.1, math.sqrt(dy**2 + dx**2))
                        dirx = -dx / sd
                        diry = -dy / sd
                        mag = self.maxshift * (1.0 / sd)
                        shiftx += dirx * mag
                        shifty += diry * mag
            sx, sy = gx + shiftx, gy + shifty
            if (0 <= int(sy) < pooled_h and 0 <= int(sx) < pooled_w
                    and 0 <= occ_pooled[int(sy), int(sx)] < WALL_THRESHOLD):
                shifted.append((sx, sy))
            else:
                # Smaller fallback shift
                sx2, sy2 = gx + 0.25 * shiftx, gy + 0.25 * shifty
                shifted.append((sx2, sy2))

        # Step 3 — Convert to map coordinates
        map_path = []
        for (sx, sy) in shifted:
            mx = (sx * 3 + 1.5) * self.res + self.origin.x
            my = (sy * 3 + 1.5) * self.res + self.origin.y
            map_path.append(MapNode(mx, my))

        # Step 4 — Wall-safe moving-average smoothing
        if len(map_path) > 2:
            smooth = [map_path[0]]
            window = 5
            for i in range(1, len(map_path) - 1):
                ws = max(0, i - window // 2)
                we = min(len(map_path), i + window // 2 + 1)
                w = map_path[ws:we]
                avgx = sum(n.x for n in w) / len(w)
                avgy = sum(n.y for n in w) / len(w)
                # Check smoothed point isn't near a wall
                spx = max(0, min(int((avgx - self.origin.x) / self.res) // 3,
                                 pooled_w - 1))
                spy = max(0, min(int((avgy - self.origin.y) / self.res) // 3,
                                 pooled_h - 1))
                if (0 <= occ_pooled[spy, spx] < WALL_THRESHOLD
                        and wall_dist[spy, spx] > 1.5):
                    smooth.append(MapNode(avgx, avgy))
                else:
                    # Keep original (unsmoothed) point
                    smooth.append(map_path[i])
            smooth.append(map_path[-1])
            map_path = smooth

        if len(map_path) > 0:
            self.goalpoint = map_path[-1]
            self.get_logger().info(
                f'Goal: ({self.goalpoint.x:.2f}, {self.goalpoint.y:.2f})  '
                f'Waypoints: {len(map_path)}')
            self.publish_goal_marker(self.goalpoint.x, self.goalpoint.y)

        self.publish_planned_path(map_path)
        return map_path

    # ── A* Search ────────────────────────────────────────────────────────────

    def _astar(self, sx, sy, gx, gy, occ_pooled, wall_dist,
               pooled_w, pooled_h, use_inflation=True, allow_unknown=False):
        """A* from (sx,sy) to (gx,gy) on the pooled grid.
        Returns (raw_path, found)  where raw_path is [(x,y), ...] start→goal.
        """
        start_key = (sx, sy)
        goal_key = (gx, gy)
        counter = 0
        open_set = []
        g_score = {start_key: 0.0}
        came_from = {}
        h = math.sqrt((gx - sx)**2 + (gy - sy)**2)
        heapq.heappush(open_set, (h, counter, start_key))
        counter += 1
        closed = set()

        while open_set:
            _, _, current_key = heapq.heappop(open_set)
            if current_key in closed:
                continue
            closed.add(current_key)
            cx, cy = current_key

            if current_key == goal_key:
                # Reconstruct
                path = []
                k = goal_key
                while k in came_from:
                    path.append(k)
                    k = came_from[k]
                path.append(start_key)
                path.reverse()
                return path, True

            for ddx, ddy in DIRECTIONS_8:
                nx, ny = cx + ddx, cy + ddy
                nkey = (nx, ny)
                if nkey in closed:
                    continue
                if not (0 <= nx < pooled_w and 0 <= ny < pooled_h):
                    continue

                cell_val = occ_pooled[ny, nx]
                if cell_val >= WALL_THRESHOLD:
                    continue

                wd = wall_dist[ny, nx]
                # Inflation check (skip cells too close to walls)
                if (use_inflation and wd <= INFLATE_RADIUS
                        and nkey != goal_key):
                    continue

                # Don't route through unknown space (except the goal itself)
                # Unless allow_unknown is set (e.g. during docking)
                if cell_val == -1 and nkey != goal_key and not allow_unknown:
                    continue

                move_cost = 1.414 if (abs(ddx) + abs(ddy) == 2) else 1.0
                proximity_penalty = 5.0 / max(wd, 0.5)
                tentative_g = g_score[current_key] + move_cost + proximity_penalty

                if nkey not in g_score or tentative_g < g_score[nkey]:
                    g_score[nkey] = tentative_g
                    came_from[nkey] = current_key
                    h = math.sqrt((gx - nx)**2 + (gy - ny)**2)
                    heapq.heappush(open_set,
                                   (tentative_g + h, counter, nkey))
                    counter += 1

        return [], False

    # ── BFS Frontier Discovery ───────────────────────────────────────────────

    def _bfs_frontier(self, sx, sy, occ_pooled, wall_dist,
                      pooled_w, pooled_h):
        """BFS to find the best unexplored frontier cell.
        Prefers frontier cells with more unknown neighbours (bigger openings)
        over tiny single-cell cracks.
        Returns (raw_path, found)."""
        start = MapNode(sx, sy)
        frontier = deque([start])
        visited = set()
        visited.add(start)
        found_path = False

        # Collect candidate frontiers with scores
        candidates = []     # list of (score, node)
        MAX_CANDIDATES = 8  # stop early once we have enough

        while frontier and len(candidates) < MAX_CANDIDATES:
            check_node = frontier.popleft()
            cy, cx = int(check_node.y), int(check_node.x)
            cell_val = occ_pooled[cy, cx]

            # Found unexplored cell adjacent to free space?
            if cell_val == -1:
                if check_node.parent:
                    parent_val = occ_pooled[int(check_node.parent.y),
                                            int(check_node.parent.x)]
                    if 0 <= parent_val < WALL_THRESHOLD:
                        # Score: count unknown neighbours (prefer big openings)
                        unknown_count = 0
                        for ddx, ddy in DIRECTIONS_8:
                            nnx, nny = cx + ddx, cy + ddy
                            if (0 <= nnx < pooled_w and 0 <= nny < pooled_h
                                    and occ_pooled[nny, nnx] == -1):
                                unknown_count += 1
                        # Also prefer frontiers farther from walls
                        wd = wall_dist[int(check_node.parent.y),
                                       int(check_node.parent.x)]
                        score = unknown_count + wd * 0.5
                        candidates.append((score, check_node))
                        continue  # keep searching for more candidates

            if cell_val >= WALL_THRESHOLD:
                continue

            neighbours = check_node.generate_neighbours(pooled_w, pooled_h)
            neighbours.sort(key=lambda n: -wall_dist[int(n.y), int(n.x)])
            for nb in neighbours:
                if nb not in visited:
                    frontier.append(nb)
                    visited.add(nb)
                    nb.parent = check_node

        # Pick the best-scoring candidate
        check_node = None
        if candidates:
            candidates.sort(key=lambda c: -c[0])   # highest score first
            check_node = candidates[0][1]
            found_path = True
            self.get_logger().info(
                f'Frontier: picked best of {len(candidates)} candidates '
                f'(score={candidates[0][0]:.1f})')

        # Fallback: random known free spot
        if not found_path:
            self.get_logger().warn(
                'No frontier found.  Wandering to a random free spot...')
            valid = [n for n in visited
                     if (0 <= occ_pooled[int(n.y), int(n.x)] < WALL_THRESHOLD
                         and math.sqrt((n.x - sx)**2 + (n.y - sy)**2) > 5)]
            if valid:
                check_node = random.choice(valid)
                found_path = True
            else:
                return [], False

        # Extract path via parent chain → reverse
        raw_path = []
        node = check_node
        while node is not None:
            raw_path.append((int(node.x), int(node.y)))
            node = node.parent
        raw_path.reverse()
        return raw_path, True

    # ─── VISUALIZATION ───────────────────────────────────────────────────────

    def publish_goal_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'goal_point'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.marker_publisher.publish(marker)

    def publish_planned_path(self, path_nodes):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for node in path_nodes:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = float(node.x)
            pose.pose.position.y = float(node.y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_publisher.publish(path_msg)

    def publish_lookahead_marker(self, target_node):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'lookahead'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(target_node.x)
        marker.pose.position.y = float(target_node.y)
        marker.pose.position.z = 0.1
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        self.lookahead_publisher.publish(marker)

    # ─── OBSTACLE DETECTION ──────────────────────────────────────────────────

    def checkObstacles(self):
        """Return True if an obstacle is dangerously close in front.
        Side-effect: transitions to RECOVERY if obstacle detected."""
        if len(self.laser_range) == 0 or np.isnan(self.laser_range).all():
            return False
        front_dist = (np.nanmin(self.front)
                      if len(self.front) > 0
                      and not np.isnan(self.front).all()
                      else float('inf'))
        if front_dist <= STOP_DISTANCE:
            if self.state not in ['RECOVERY', 'ESCAPING']:
                self.pre_recovery_state = self.state
                self.stopbot()
                self.get_logger().info(
                    f'Obstacle at {front_dist:.2f}m while in {self.state}! '
                    f'→ RECOVERY')
                self.state = 'RECOVERY'
            return True
        return False

    # ─── TURNING ─────────────────────────────────────────────────────────────

    def turn_in_place(self, target_angle, current_angle):
        anglediff = target_angle - current_angle
        anglediff = (anglediff + math.pi) % (2 * math.pi) - math.pi
        twist = Twist()
        twist.linear.x = 0.0
        kp_yaw = 1.5       # was 0.8 — faster rotation
        turning_speed = anglediff * kp_yaw
        twist.angular.z = max(-self.rpp_controller.max_angular_v,
                              min(self.rpp_controller.max_angular_v,
                                  turning_speed))
        self.publisher_.publish(twist)

    def evaluate_escape_direction(self):
        left_dist = (np.nanmin(self.left)
                     if len(self.left) > 0
                     and not np.isnan(self.left).all() else 0.0)
        right_dist = (np.nanmin(self.right)
                      if len(self.right) > 0
                      and not np.isnan(self.right).all() else 0.0)
        self.get_logger().info(
            f'Clearance — Left: {left_dist:.2f}m  Right: {right_dist:.2f}m')
        MIN_SIDE_CLEARANCE = 0.35
        if left_dist >= MIN_SIDE_CLEARANCE and left_dist > right_dist:
            return self.turn_angle_by
        elif right_dist >= MIN_SIDE_CLEARANCE and right_dist >= left_dist:
            return -self.turn_angle_by
        elif left_dist >= right_dist:
            return self.turn_angle_by
        else:
            return -self.turn_angle_by

    # ─── RECOVERY ────────────────────────────────────────────────────────────

    def recoveryTurn(self):
        current_angle = self.get_orientation()[2]
        if self.recovery_angle is None:
            if self.escape_direction_locked is None:
                self.escape_direction_locked = self.evaluate_escape_direction()
            raw_angle = current_angle + self.escape_direction_locked
            self.recovery_angle = ((raw_angle + math.pi) % (2 * math.pi)
                                   - math.pi)
        angle_diff = self.recovery_angle - current_angle
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        if abs(angle_diff) <= 0.05:
            self.stopbot()
            self.get_logger().info('Recovery Turn Complete.')
            self.recovery_angle = None
            return True
        self.turn_in_place(self.recovery_angle, current_angle)
        return False

    def recoverySequence(self):
        if self.recoveryTurn():
            if not self.checkObstacles():
                self.get_logger().info('Front clear after recovery. Escaping…')
                self.stopbot()
                self.escape_direction_locked = None
                if self.boink >= 3:
                    self.get_logger().warn(
                        'Stuck loop detected.  Dropping current goal.')
                    self.goal = None
                    self.boink = 0
                self.escape_start_time = self.get_clock().now()
                self.state = 'ESCAPING'
            else:
                self.recoveryTurn()
        else:
            self.recoveryTurn()

    # ─── PATH FOLLOWING ──────────────────────────────────────────────────────

    def mover(self, allow_state_change=True):
        """Follow self.path using RPP.  Returns True when path is exhausted.
        Set allow_state_change=False when called from DOCK_NAV to prevent
        automatic transitions to PLANNING/SEARCHING."""
        if not self.path or len(self.path) == 0:
            self.stopbot()
            if allow_state_change:
                self.state = 'PLANNING'
            return True

        try:
            cur_x, cur_y, cur_yaw = self.get_orientation()
        except Exception as e:
            self.get_logger().warn(f'Pose unavailable: {e}')
            return False

        # ── Prune waypoints behind the robot ──
        heading_x = math.cos(cur_yaw)
        heading_y = math.sin(cur_yaw)
        while len(self.path) > 1:
            wp = self.path[0]
            dx = wp.x - cur_x
            dy = wp.y - cur_y
            dist = math.sqrt(dx**2 + dy**2)
            dot = dx * heading_x + dy * heading_y
            # Pop if within 0.20 m  OR  behind us and within 0.50 m
            if dist < 0.20 or (dot < 0 and dist < 0.50):
                self.path.pop(0)
            else:
                break

        # ── Goal reached? (only during normal DRIVING) ──
        if allow_state_change and getattr(self, 'goal', None) is not None:
            disttogoal = math.sqrt(
                (self.goal.x - cur_x)**2 + (self.goal.y - cur_y)**2)
            if disttogoal <= GOAL_THRESHOLD:
                self.get_logger().info('Goal Reached! Replanning immediately…')
                self.path = []
                self.goal = None
                self.boink = 0
                # Skip SEARCHING — markers are detected passively via
                # marker_callback during driving.  Go straight to PLANNING
                # so we don't waste time spinning in place.
                self.state = 'PLANNING'
                return True

        # ── Obstacle check ──
        if self.checkObstacles():
            self.boink += 1
            return False

        # ── RPP command ──
        cmd_vel = self.rpp_controller.command(cur_x, cur_y, cur_yaw, self.path)
        target = self.rpp_controller.findpoint(cur_x, cur_y, self.path)
        self.publish_lookahead_marker(target)

        if cmd_vel is None:
            # Very large angle → stop-and-rotate
            now = self.get_clock().now()
            if self.rotation_start_time is None:
                self.rotation_start_time = now
            elif ((now - self.rotation_start_time).nanoseconds / 1e9
                  > self.turning_timeout):
                self.get_logger().warn('Rotation timeout.  Replanning…')
                self.stopbot()
                if allow_state_change:
                    self.state = 'PLANNING'
                self.rotation_start_time = None
                return True
            self.get_logger().info('Large angle → ALIGNING')
            self.stopbot()
            if allow_state_change:
                self.state = 'ALIGNING'
            else:
                target_angle = math.atan2(target.y - cur_y, target.x - cur_x)
                self.turn_in_place(target_angle, cur_yaw)
            return False

        self.rotation_start_time = None     # reset on successful move

        # ── Side-clearance nudge (proportional, both-sides-aware) ──
        SIDE_NUDGE_DIST = 0.25
        left_min = (float(np.nanmin(self.left))
                    if len(self.left) > 0
                    and not np.isnan(self.left).all() else float('inf'))
        right_min = (float(np.nanmin(self.right))
                     if len(self.right) > 0
                     and not np.isnan(self.right).all() else float('inf'))
        # Only nudge if at least one side is close
        if left_min < SIDE_NUDGE_DIST or right_min < SIDE_NUDGE_DIST:
            left_pressure = max(0.0, (SIDE_NUDGE_DIST - left_min) / SIDE_NUDGE_DIST)
            right_pressure = max(0.0, (SIDE_NUDGE_DIST - right_min) / SIDE_NUDGE_DIST)
            nudge = 0.15 * (right_pressure - left_pressure)
            cmd_vel.angular.z += nudge

        self.publisher_.publish(cmd_vel)
        return False

    # ─── SEARCHING ───────────────────────────────────────────────────────────

    def search_marker_logic(self):
        self.get_logger().info('Searching for marker… spinning.')
        try:
            _, _, cur_yaw = self.get_orientation()
        except Exception:
            return
        if self.search_last_yaw is not None:
            delta = cur_yaw - self.search_last_yaw
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
            self.search_yaw_accumulated += abs(delta)
        self.search_last_yaw = cur_yaw
        if self.search_yaw_accumulated >= (2 * math.pi):
            self.get_logger().warn(
                '360° done.  No marker.  Resuming exploration.')
            self.stopbot()
            self.state = 'PLANNING'
            self.search_last_yaw = None
            self.search_yaw_accumulated = 0.0
            return
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = self.search_spin_speed
        self.publisher_.publish(twist)

    # ─── DOCKING — POLAR ARC CONTROLLER ───────────────────────────────────────

    def docking_logic(self):
        """Single-state polar arc dock.  Drives a smooth curve from any
        position so the robot arrives facing the marker perpendicularly.

        Camera frame: X = right, Z = forward.
        alpha = atan2(X, Z) = bearing to marker from robot heading.
        rho   = sqrt(X² + Z²) = distance to marker.

        Control law:
            v = k_rho  * rho          (approach: slow as we get close)
            w = -k_alpha * alpha      (steer to centre marker in view)
        This naturally produces a smooth converging arc."""

        elapsed = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9

        # ── Marker visible: drive the arc ──
        if self.marker_visible and self.marker_x is not None:
            x = self.marker_x       # lateral offset (+right)
            z = self.marker_z       # forward distance
            rho = math.sqrt(x**2 + z**2)
            alpha = math.atan2(x, z)
            self.dock_last_alpha = alpha     # save for recovery

            self.get_logger().info(
                f'[DOCK] rho={rho:.2f}m  α={math.degrees(alpha):+.1f}°',
                throttle_duration_sec=0.3)

            # ── Close enough → docked! ──
            if rho <= self.dock_target_dist:
                self.stopbot()
                self.state = 'STATION'
                self.get_logger().info(
                    f'✓ Docked!  rho={rho:.2f}m  α={math.degrees(alpha):+.1f}°')
                return

            # ── Obstacle check (ignore within 0.5m — that's the dock wall) ──
            if rho > 0.50:
                front_dist = (np.nanmin(self.front)
                              if len(self.front) > 0
                              and not np.isnan(self.front).all()
                              else float('inf'))
                if front_dist <= STOP_DISTANCE:
                    self.get_logger().warn(
                        f'Obstacle at {front_dist:.2f}m during dock! → RECOVERY')
                    self.pre_recovery_state = 'DOCKING'
                    self.stopbot()
                    self.state = 'RECOVERY'
                    return

            # ── Polar control law ──
            v = self.dock_k_rho * rho
            w = -self.dock_k_alpha * alpha

            # Clamp
            v = max(0.0, min(self.dock_max_v, v))   # always forward
            w = max(-self.dock_max_w, min(self.dock_max_w, w))

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.publisher_.publish(cmd)
            return

        # ── Marker lost: slow search to re-acquire ──
        if elapsed <= self.dock_lost_timeout:
            cmd = Twist()
            cmd.linear.x = 0.0
            # Rotate toward last known direction
            if self.dock_last_alpha is not None:
                cmd.angular.z = -math.copysign(0.10, self.dock_last_alpha)
            else:
                cmd.angular.z = 0.10
            self.publisher_.publish(cmd)
            self.get_logger().info(
                f'Marker lost {elapsed:.1f}s ago. Searching...',
                throttle_duration_sec=0.5)
            return

        # ── Lost too long → abort ──
        self.stopbot()
        self.dock_last_alpha = None
        self.get_logger().warn('Marker lost too long. Resuming exploration.')
        self.state = 'PLANNING'

    # ─── MAIN CONTROLLER (10 Hz) ─────────────────────────────────────────────

    def controller(self):
        # ── Exploration & Navigation ──
        if self.state == 'PLANNING':
            self.path = self.planroute(
                goal=None if self.goal is None else self.goal)
            if self.path and len(self.path) > 0:
                self.goal = self.path[-1]
                self.state = 'DRIVING'
            else:
                self.stopbot()
                self.goal = None
                self.boink = 0

        elif self.state == 'DRIVING':
            self.mover(allow_state_change=True)

        elif self.state == 'ALIGNING':
            try:
                cur_x, cur_y, cur_yaw = self.get_orientation()
            except Exception:
                return
            if not self.path or len(self.path) == 0:
                self.state = 'PLANNING'
                return
            target = self.rpp_controller.findpoint(cur_x, cur_y, self.path)
            target_angle = math.atan2(target.y - cur_y, target.x - cur_x)
            angle_diff = target_angle - cur_yaw
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) <= 0.30:
                self.stopbot()
                self.state = 'DRIVING'
                self.rotation_start_time = None
            else:
                now = self.get_clock().now()
                if (self.rotation_start_time is not None
                        and (now - self.rotation_start_time).nanoseconds / 1e9
                        > self.turning_timeout):
                    self.stopbot()
                    self.state = 'PLANNING'
                    self.rotation_start_time = None
                    return
                if (len(self.front) > 0
                        and np.nanmin(self.front) <= SIDE_THRESHOLD):
                    self.stopbot()
                    self.state = 'RECOVERY'
                    return
                self.turn_in_place(target_angle, cur_yaw)

        # ── Recovery & Escape ──
        elif self.state == 'RECOVERY':
            self.front_fov = 120
            self.recoverySequence()

        elif self.state == 'ESCAPING':
            now = self.get_clock().now()
            elapsed = (now - self.escape_start_time).nanoseconds / 1e9
            if not self.checkObstacles():
                if elapsed < self.escape_duration:
                    twist = Twist()
                    twist.linear.x = float(self.escape_speed)
                    twist.angular.z = 0.0
                    self.publisher_.publish(twist)
                    if (len(self.front) > 0
                            and np.nanmin(self.front) <= SIDE_THRESHOLD):
                        self.stopbot()
                        self.state = 'RECOVERY'
                else:
                    self.stopbot()
                    self.front_fov = 90
                    # Resume correct state after escape
                    if self.pre_recovery_state == 'DOCKING':
                        self.get_logger().info(
                            'Was docking. Resuming polar arc…')
                        self.state = 'DOCKING'
                    else:
                        self.get_logger().info(
                            'Resuming standard navigation.')
                        self.state = 'PLANNING'
                    self.pre_recovery_state = None
            else:
                self.stopbot()
                self.state = 'RECOVERY'

        # ── Marker Search ──
        elif self.state == 'SEARCHING':
            self.search_marker_logic()

        # ── Docking ──
        elif self.state == 'DOCKING':
            self.docking_logic()

        # ── Station & Mission ──
        elif self.state == 'STATION':
            self.stopbot()
            if not self.commandSent:
                self.get_logger().info(
                    f'Signaling payload for Station '
                    f'{self.current_docking_id} via GPIO…')
                if self.current_docking_id == 0:
                    self.gpio_pub.publish(String(data='A'))
                    self.get_logger().info(
                        'Command Sent for Station A.  Waiting for PI…')
                    self.commandSent = True
                    self.state = 'WAITING_FOR_PI'
                if self.current_docking_id == 1:
                    self.gpio_pub.publish(String(data='B'))
                    self.get_logger().info(
                        'Command Sent for Station B.  Waiting for PI…')
                    self.commandSent = True
                    self.state = 'WAITING_FOR_PI'

        elif self.state == 'WAITING_FOR_PI':
            self.stopbot()
            if self.responseHeard:
                if self.current_docking_id in self.valid_station_ids:
                    self.valid_station_ids.remove(self.current_docking_id)
                self.current_docking_id = None
                self.commandSent = False
                self.responseHeard = False
                # Clear docking state
                self.dock_last_alpha = None
                if self.STATION_A_COMPLETED and self.STATION_B_COMPLETED:
                    self.get_logger().info('MISSION COMPLETE!')
                    self.state = 'MISSION_COMPLETE'
                else:
                    self.get_logger().info(
                        'Station complete!  Undocking…')
                    self.undock_start_time = self.get_clock().now()
                    self.state = 'UNDOCKING'

        elif self.state == 'UNDOCKING':
            now = self.get_clock().now()
            elapsed = (now - self.undock_start_time).nanoseconds / 1e9
            back_dist = (np.nanmin(self.back)
                         if len(self.back) > 0
                         and not np.isnan(self.back).all()
                         else float('inf'))
            if back_dist <= 0.25:
                self.get_logger().warn(
                    f'Obstacle {back_dist:.2f}m behind!  Aborting reverse.')
                self.stopbot()
                self.undock_start_time = None
                self.state = 'PLANNING'
                return
            if elapsed < self.undock_duration:
                twist = Twist()
                twist.linear.x = float(self.undock_speed)
                twist.angular.z = 0.0
                self.publisher_.publish(twist)
            else:
                self.stopbot()
                self.get_logger().info(
                    'Undocking complete.  Resuming exploration…')
                self.undock_start_time = None
                self.state = 'PLANNING'

        elif self.state == 'MISSION_COMPLETE':
            self.stopbot()


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    autopilot_node = AutoPilot()

    print("\n--- Node Initialized ---")
    print("Configure RViz now. When ready, type 'yes' to start navigation.")

    user_input = ""
    while user_input.lower() != 'yes':
        user_input = input("Start planning? (yes/no): ")

    try:
        autopilot_node.get_logger().info('Starting navigation...')
        rclpy.spin(autopilot_node)
    except KeyboardInterrupt:
        autopilot_node.get_logger().info('Shutting Down')
        autopilot_node.stopbot()
    finally:
        autopilot_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
