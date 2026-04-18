import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import numpy as np
import math
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# ─── TUNING ───────────────────────────────────────────────────────────────────
GOAL_THRESHOLD      = 0.25   # m — switch from DRIVING to DOCKING
DOCK_DISTANCE       = 0.20   # m — stop creeping and fire
DOCK_SPEED          = 0.06   # m/s — approach speed
DOCK_ALIGN_TOL      = 0.08   # rad (~4.5°) — alignment tolerance

NAV_LINEAR_MAX      = 0.18   # m/s — top cruising speed
NAV_ANGULAR_MAX     = 1.2    # rad/s — top turn speed
NAV_KP_ANGULAR      = 1.8    # proportional heading gain
NAV_KP_LINEAR       = 0.6    # proportional speed-scaling on heading error

OBST_STOP_DIST      = 0.30   # m — emergency stop threshold (front)
OBST_SLOW_DIST      = 0.50   # m — begin braking
OBST_SIDE_DIST      = 0.25   # m — side clearance for wall avoidance

# Yaw the robot should face when docked (0=East, π/2=North, π=West, -π/2=South)
TARGET_STATION_YAW  = 0.0
# ──────────────────────────────────────────────────────────────────────────────


def euler_from_quaternion(x, y, z, w):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1), math.asin(t2), math.atan2(t3, t4)


def angle_diff(a, b):
    """Shortest signed angular difference a - b, result in (-π, π]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class SectorScan:
    """Holds processed LiDAR distances per sector."""
    def __init__(self):
        self.front      = 9.9
        self.front_left = 9.9
        self.front_right= 9.9
        self.left       = 9.9
        self.right      = 9.9
        self.rear       = 9.9


class AutoPilot(Node):
    def __init__(self):
        super().__init__('autopilot_node')

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pi_pub  = self.create_publisher(String, 'gpio_commands', 10)

        # QoS for map/nav topics that use transient local
        nav_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.goal_sub = self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_callback, 10)
        self.pi_sub = self.create_subscription(
            String, 'rpi_response', self.pi_response_callback, 10)

        # State
        self.state     = 'IDLE'      # IDLE | DRIVING | DOCKING | WAITING_FOR_PI
        self.goal_x    = None
        self.goal_y    = None
        self.scan      = SectorScan()
        self.pose_ok   = False       # becomes True once TF is available

        # Control timer — 10 Hz
        self.create_timer(0.1, self.controller)
        self.get_logger().info('AutoPilot ready. Waiting for /goal_pose ...')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def goal_callback(self, msg: PoseStamped):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.state  = 'DRIVING'
        self.get_logger().info(
            f'New goal received: ({self.goal_x:.2f}, {self.goal_y:.2f})'
        )

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=float)
        # Replace invalid readings with NaN
        ranges[np.isinf(ranges)] = np.nan
        ranges[ranges < msg.range_min] = np.nan

        n = len(ranges)
        if n == 0:
            return

        def sector_min(center_deg, half_width_deg):
            """Returns min range in a symmetric arc around center_deg."""
            hw = max(1, int(round((half_width_deg / 360.0) * n)))
            ci = int(round((center_deg / 360.0) * n)) % n
            indices = [(ci + i) % n for i in range(-hw, hw + 1)]
            vals = ranges[indices]
            return float(np.nanmin(vals)) if not np.all(np.isnan(vals)) else 9.9

        # TurtleBot3 convention: index 0 = front, increasing CCW
        self.scan.front       = sector_min(0,   15)
        self.scan.front_left  = sector_min(45,  15)
        self.scan.front_right = sector_min(315, 15)
        self.scan.left        = sector_min(90,  15)
        self.scan.right       = sector_min(270, 15)
        self.scan.rear        = sector_min(180, 20)

    def pi_response_callback(self, msg: String):
        if 'COMPLETE' in msg.data:
            self.get_logger().info('Pi confirmed COMPLETE. Returning to IDLE.')
            self.state  = 'IDLE'
            self.goal_x = None
            self.goal_y = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_pose(self):
        """Returns (x, y, yaw) from TF map→base_link, or None on failure."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            r = t.transform.rotation
            _, _, yaw = euler_from_quaternion(r.x, r.y, r.z, r.w)
            return (
                t.transform.translation.x,
                t.transform.translation.y,
                yaw,
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def stop(self):
        self.cmd_pub.publish(Twist())

    def publish_vel(self, linear, angular):
        t = Twist()
        t.linear.x  = float(np.clip(linear,  -NAV_LINEAR_MAX,  NAV_LINEAR_MAX))
        t.angular.z = float(np.clip(angular, -NAV_ANGULAR_MAX, NAV_ANGULAR_MAX))
        self.cmd_pub.publish(t)

    # ── Navigation ────────────────────────────────────────────────────────────

    def drive_to_goal(self):
        """
        Heading controller with reactive obstacle avoidance.
        1. Steer toward goal using proportional heading error.
        2. Scale speed down when heading error is large or obstacle is close.
        3. If frontal obstruction, attempt avoidance turn toward the clearer side.
        """
        pose = self.get_pose()
        if pose is None:
            self.get_logger().warn('TF not ready — waiting...')
            return

        cx, cy, yaw = pose

        # ── Check arrival ──────────────────────────────────────────────
        dist = math.hypot(self.goal_x - cx, self.goal_y - cy)
        if dist <= GOAL_THRESHOLD:
            self.stop()
            self.state = 'DOCKING'
            self.get_logger().info('Goal reached. Switching to DOCKING.')
            return

        # ── Desired heading ────────────────────────────────────────────
        desired_yaw = math.atan2(self.goal_y - cy, self.goal_x - cx)
        heading_err = angle_diff(desired_yaw, yaw)

        # ── Emergency stop ─────────────────────────────────────────────
        if self.scan.front < OBST_STOP_DIST:
            # Pick the clearer side to rotate
            if self.scan.front_left >= self.scan.front_right:
                self.publish_vel(0.0, NAV_ANGULAR_MAX)
            else:
                self.publish_vel(0.0, -NAV_ANGULAR_MAX)
            self.get_logger().warn(
                f'OBSTACLE @ {self.scan.front:.2f}m — evading')
            return

        # ── Obstacle-aware angular correction ──────────────────────────
        # If a side is too close, add a correction bias away from the wall
        wall_bias = 0.0
        if self.scan.front_left < OBST_SIDE_DIST:
            wall_bias -= 0.6   # nudge right
        if self.scan.front_right < OBST_SIDE_DIST:
            wall_bias += 0.6   # nudge left

        angular = NAV_KP_ANGULAR * heading_err + wall_bias

        # ── Speed scaling ──────────────────────────────────────────────
        # Slow down when heading far off target or near obstacle
        heading_factor = math.cos(heading_err)          # 1.0 when aligned, 0 at 90°
        heading_factor = max(0.0, heading_factor)

        obst_factor = 1.0
        if self.scan.front < OBST_SLOW_DIST:
            obst_factor = (self.scan.front - OBST_STOP_DIST) / \
                          (OBST_SLOW_DIST  - OBST_STOP_DIST)
            obst_factor = max(0.0, obst_factor)

        linear = NAV_LINEAR_MAX * heading_factor * obst_factor * \
                 NAV_KP_LINEAR / max(abs(heading_err) * NAV_KP_LINEAR, 1.0)
        # Simpler: just scale by cos of heading error
        linear = NAV_LINEAR_MAX * heading_factor * obst_factor

        self.publish_vel(linear, angular)

    # ── Docking ───────────────────────────────────────────────────────────────

    def execute_docking(self):
        """
        Phase 1 — Rotate to face docking station.
        Phase 2 — Creep forward until DOCK_DISTANCE.
        Phase 3 — Hand off to Pi.
        """
        pose = self.get_pose()
        if pose is None:
            return
        _, _, yaw = pose

        yaw_err = angle_diff(TARGET_STATION_YAW, yaw)

        if abs(yaw_err) > DOCK_ALIGN_TOL:
            # Phase 1: align
            self.get_logger().info(f'[DOCK] Aligning — yaw_err={math.degrees(yaw_err):.1f}°')
            self.publish_vel(0.0, 0.4 * np.sign(yaw_err))

        elif self.scan.front > DOCK_DISTANCE:
            # Phase 2: creep
            self.get_logger().info(
                f'[DOCK] Creeping — front_dist={self.scan.front:.3f}m')
            self.publish_vel(DOCK_SPEED, 0.0)

        else:
            # Phase 3: hand off
            self.stop()
            self.get_logger().info('[DOCK] Docked. Firing Pi → LAUNCH')
            self.pi_pub.publish(String(data='LAUNCH'))
            self.state = 'WAITING_FOR_PI'

    # ── Main state machine ────────────────────────────────────────────────────

    def controller(self):
        if self.state == 'IDLE':
            pass  # Waiting for a /goal_pose message

        elif self.state == 'DRIVING':
            if self.goal_x is None:
                self.state = 'IDLE'
                return
            self.drive_to_goal()

        elif self.state == 'DOCKING':
            self.execute_docking()

        elif self.state == 'WAITING_FOR_PI':
            self.stop()


def main(args=None):
    rclpy.init(args=args)
    node = AutoPilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()