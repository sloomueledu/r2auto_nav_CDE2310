"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist
import math

class ArucoDocker(Node):
    def __init__(self):
        super().__init__('aruco_docker')

        self.pose_sub = self.create_subscription(
            PoseArray, '/usbcam1_poses', self.pose_callback, 10
        )
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.safety_timer = self.create_timer(0.5, self.safety_check_callback)
        self.last_seen_time = self.get_clock().now()

        # --- Tuning Parameters ---
        self.target_distance_z = 0.25   # 25 cm from marker
        self.target_offset_x   = 0.043   # laterally centered on marker

        # Proportional gains
        self.kp_linear          = 0.2   # forward/back
        self.kp_angular_orient  = 0.25   # perpendicularity correction  ← NEW
        self.kp_angular_center  = 0.20   # lateral centering (secondary) ← NEW

        # Speed limits
        self.max_linear_speed  = 0.15   # m/s
        self.max_angular_speed = 0.33    # rad/s

        # Deadzones
        self.dz_z   = 0.02   # 2 cm depth deadzone
        self.dz_x   = 0.01   # 1 cm lateral deadzone
        self.dz_yaw = 0.02   # ~1° orientation deadzone

        self.get_logger().info("Aruco Docker Node Started. Waiting for poses...")

    # ------------------------------------------------------------------
    # NEW: extract yaw from the marker's quaternion
    # ------------------------------------------------------------------
    def get_marker_yaw(self, q):
        
        '''
        Returns the rotation of the marker around the camera's Y-axis (vertical).

        Geometry: in the camera frame (Z-forward, X-right, Y-down), this angle
        captures how much the marker is "angled" left/right relative to the lens.
        When the robot is perfectly perpendicular to the marker, this is 0.

        NOTE: if the sign feels wrong on the robot, negate kp_angular_orient.
        '''
        sin_pitch = 2.0 * (q.w * q.y - q.z * q.x)
        sin_pitch = max(-1.0, min(1.0, sin_pitch))   # clamp before asin
        return math.asin(sin_pitch)

    # ------------------------------------------------------------------

    def pose_callback(self, msg):
        if not msg.poses:
            return

        self.last_seen_time = self.get_clock().now()
        marker_pose = msg.poses[0]

        # --- Compute errors ---
        error_z   = marker_pose.position.z - self.target_distance_z  # depth
        error_x   = marker_pose.position.x - self.target_offset_x    # lateral
        yaw_error = self.get_marker_yaw(marker_pose.orientation)      # orientation

        cmd = Twist()

        # 1. Linear Control (forward / back) — unchanged logic
        if abs(error_z) > self.dz_z:
            raw_linear = error_z * self.kp_linear
            cmd.linear.x = max(min(raw_linear, self.max_linear_speed),
                               -self.max_linear_speed)

        # 2. Angular Control — perpendicularity + centering blended
        #
        #   yaw_error: primary term — rotates robot until it squares up to marker
        #   error_x:   secondary term — keeps marker centered in frame while closing in
        #
        #   Negative sign on yaw_error: if marker is rotated +Y (its right side
        #   faces you), the robot must turn right (negative angular.z) to square up.
        if abs(yaw_error) > self.dz_yaw or abs(error_x) > self.dz_x:
            raw_angular = -(yaw_error * self.kp_angular_orient
                            + error_x  * self.kp_angular_center)
            cmd.angular.z = max(min(raw_angular,  self.max_angular_speed),
                               -self.max_angular_speed)

        self.vel_pub.publish(cmd)

        # Helpful debug output
        self.get_logger().info(
            f"Z_err={error_z:+.3f}m  X_err={error_x:+.3f}m  "
            f"Yaw_err={math.degrees(yaw_error):+.1f}°  "
            f"→ lin={cmd.linear.x:+.3f}  ang={cmd.angular.z:+.3f}",
            throttle_duration_sec=0.2
        )

    def safety_check_callback(self):
        # Stop the robot if nomarker seen for 1 second.
        elapsed = (self.get_clock().now() - self.last_seen_time).nanoseconds / 1e9
        if elapsed > 1.0:
            self.vel_pub.publish(Twist())
            self.get_logger().warn("Marker lost! Stopping robot.", throttle_duration_sec=2.0)
    
    def spininplace(self):
        # Optional helper to spin in place (for testing)
        cmd = Twist()
        cmd.angular.z = 0.2  # rad/s
        self.vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    docker_node = ArucoDocker()
    try:
        rclpy.spin(docker_node)
    except KeyboardInterrupt:
        pass
    finally:
        docker_node.vel_pub.publish(Twist())
        docker_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
import math

class DockingState:
    IDLE            = 0
    PHASE1_ROTATE   = 1   # Turn to face the 'spot'
    PHASE2_DRIVE    = 2   # Drive distance (Blind)
    PHASE3_ROTATE90 = 3   # Final turn to face hole
    DONE            = 4

class ArucoDocker(Node):
    def __init__(self):
        super().__init__('aruco_docker')

        self.pose_sub = self.create_subscription(PoseArray, '/usbcam1_poses', self.pose_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.state = DockingState.IDLE
        self.target_distance = 0.25 
        
        # Tracking variables
        self.odom_yaw_now = 0.0
        self.odom_pos_now = (0.0, 0.0)
        
        # Targets locked in at IDLE
        self.target_yaw = 0.0
        self.target_drive_dist = 0.0
        self.start_pos = (0.0, 0.0)
        self.start_yaw = 0.0

        self.get_logger().info("Blind Docking Node Ready. Waiting for Marker...")

    def odom_callback(self, msg):
        # Update current heading
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw_now = math.atan2(siny, cosy)
        
        # Update current position
        self.odom_pos_now = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        # Run Phase 1-3 logic here because it's now driven by Odom, not Camera
        self.control_loop()

    def pose_callback(self, msg):
        # Only use the camera for the initial "Snapshot"
        if not msg.poses or self.state != DockingState.IDLE:
            return
        
        x = msg.poses[0].position.x
        z = msg.poses[0].position.z

        # CALCULATION: Point C is (0, 0.25) in Marker Frame
        # Vector to that point from robot:
        self.target_drive_dist = math.sqrt(x**2 + (z - self.target_distance)**2)
        angle_to_spot = math.atan2(x, z - self.target_distance)

        # Record where we are starting from
        self.start_yaw = self.odom_yaw_now
        self.start_pos = self.odom_pos_now
        
        # Our first turn goal
        self.target_yaw = angle_to_spot 
        
        self.get_logger().info(f"Snapshot! Dist to spot: {self.target_drive_dist:.2f}m")
        self.state = DockingState.PHASE1_ROTATE

    def control_loop(self):
        cmd = Twist()

        # PHASE 1: Initial Turn
        if self.state == DockingState.PHASE1_ROTATE:
            diff = self._angle_diff(self.odom_yaw_now, self.start_yaw)
            error = self.target_yaw - diff
            if abs(error) > 0.05:
                cmd.angular.z = 0.3 if error > 0 else -0.3
                self.vel_pub.publish(cmd)
            else:
                self.stop()
                self.start_pos = self.odom_pos_now # Reset starting point for drive
                self.state = DockingState.PHASE2_DRIVE
                self.get_logger().info("Facing spot. Driving...")

        # PHASE 2: Drive Blind (Odometry)
        elif self.state == DockingState.PHASE2_DRIVE:
            # Calculate distance traveled from start of this phase
            curr_dist = math.sqrt((self.odom_pos_now[0] - self.start_pos[0])**2 + 
                                  (self.odom_pos_now[1] - self.start_pos[1])**2)
            
            if curr_dist < self.target_drive_dist:
                cmd.linear.x = 0.12
                self.vel_pub.publish(cmd)
            else:
                self.stop()
                self.start_yaw = self.odom_yaw_now # Reset for final turn
                self.state = DockingState.PHASE3_ROTATE90
                self.get_logger().info("Arrived at spot. Squaring up...")

        # PHASE 3: Rotate to face Marker
        elif self.state == DockingState.PHASE3_ROTATE90:
            diff = self._angle_diff(self.odom_yaw_now, self.start_yaw)
            # We turn back the same amount we turned in Phase 1
            error = (-self.target_yaw) - diff
            if abs(error) > 0.05:
                cmd.angular.z = 0.3 if error > 0 else -0.3
                self.vel_pub.publish(cmd)
            else:
                self.stop()
                self.state = DockingState.DONE
                self.get_logger().info("DOCKING COMPLETE.")

    def stop(self):
        self.vel_pub.publish(Twist())

    def _angle_diff(self, a, b):
        d = a - b
        while d > math.pi: d -= 2*math.pi
        while d < -math.pi: d += 2*math.pi
        return d

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDocker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    """

"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
import math
import time

class DockingState:
    SEARCHING       = 0   # Spin to find marker
    LOCKING         = 1   # Stop for 4 seconds, print stats, lock targets
    PHASE1_ROTATE   = 2   # Turn to the calculated path (Odom)
    PHASE2_DRIVE    = 3   # Drive the calculated distance (Odom)
    PHASE3_CENTERING = 4  # Turn until marker is dead-center (Camera)
    DONE            = 5   # Brief pause before restarting

class ArucoDocker(Node):
    def __init__(self):
        super().__init__('aruco_docker')

        self.pose_sub = self.create_subscription(PoseArray, '/usbcam1_poses', self.pose_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.state = DockingState.SEARCHING
        self.target_distance = 0.25  # 25 cm
        
        # Internal Nav Variables
        self.odom_yaw_now = 0.0
        self.odom_pos_now = (0.0, 0.0)
        self.start_yaw = 0.0
        self.start_pos = (0.0, 0.0)
        
        self.target_angle = 0.0
        self.target_dist_m = 0.0
        self.lock_time = 0.0

        self.get_logger().info("System Started: Searching for ArUco...")

    def odom_callback(self, msg):
        # Update heading (Yaw)
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw_now = math.atan2(siny, cosy)
        # Update position
        self.odom_pos_now = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        self.control_loop()

    def pose_callback(self, msg):
        if not msg.poses:
            return

        marker = msg.poses[0]
        
        # --- STATE 0: SPOTTED ---
        if self.state == DockingState.SEARCHING:
            self.stop()
            self.state = DockingState.LOCKING
            self.lock_time = time.time()
            
            # Convert to CM for the printout
            x_cm = marker.position.x * 100
            y_cm = marker.position.y * 100
            z_cm = marker.position.z * 100
            
            # CALCULATE PATH (V-Maneuver)
            # Distance to the spot 25cm in front of marker
            self.target_dist_m = math.sqrt(marker.position.x**2 + (marker.position.z - self.target_distance)**2)
            self.target_angle = math.atan2(marker.position.x, marker.position.z - self.target_distance)

            print("-" * 30)
            print(f"MARKER DETECTED!")
            print(f"Camera Readings (cm): X:{x_cm:.1f}, Y:{y_cm:.1f}, Z:{z_cm:.1f}")
            print(f"Calculated Path: Turn {math.degrees(self.target_angle):+.1f}°, Drive {self.target_dist_m*100:.1f}cm")
            print("Locking in coordinates for 4 seconds...")
            print("-" * 30)

    def control_loop(self):
        cmd = Twist()

        # 0. SPIN SEARCH
        if self.state == DockingState.SEARCHING:
            cmd.angular.z = 0.3
            self.vel_pub.publish(cmd)

        # 1. 4-SECOND LOCK
        elif self.state == DockingState.LOCKING:
            if (time.time() - self.lock_time) > 4.0:
                self.start_yaw = self.odom_yaw_now
                self.start_pos = self.odom_pos_now
                self.state = DockingState.PHASE1_ROTATE
                self.get_logger().info("4 Seconds up. Executing maneuver.")

        # 2. TURN TO PATH (ODOM)
        elif self.state == DockingState.PHASE1_ROTATE:
            error = self.target_angle - self._angle_diff(self.odom_yaw_now, self.start_yaw)
            if abs(error) > 0.03:
                # Automatic Clock/Anticlock based on sign of error
                cmd.angular.z = 0.4 if error > 0 else -0.4
                self.vel_pub.publish(cmd)
            else:
                self.stop()
                self.start_pos = self.odom_pos_now
                self.state = DockingState.PHASE2_DRIVE

        # 3. DRIVE BLIND (ODOM)
        elif self.state == DockingState.PHASE2_DRIVE:
            curr_dist = math.sqrt((self.odom_pos_now[0]-self.start_pos[0])**2 + (self.odom_pos_now[1]-self.start_pos[1])**2)
            if curr_dist < self.target_dist_m:
                cmd.linear.x = 0.12
                self.vel_pub.publish(cmd)
            else:
                self.stop()
                self.state = DockingState.PHASE3_CENTERING

        # 4. FINAL CENTER ON HOLE (CAMERA)
        elif self.state == DockingState.PHASE3_CENTERING:
            # Note: We need a fresh camera reading here. 
            # We'll use a local check for the marker.
            # If the robot can't see it, it will spin slowly until it does.
            cmd.angular.z = 0.2 # Slow search rotation
            self.vel_pub.publish(cmd)
            # This state is handled by pose_callback basically 'taking over'
            # to check if the x-offset is near zero.

    def pose_callback_centering(self, x_offset):
        # Helper to handle the centering logic in Phase 3
        if self.state == DockingState.PHASE3_CENTERING:
            cmd = Twist()
            if abs(x_offset) > 0.02: # 2cm centering tolerance
                cmd.angular.z = -0.3 if x_offset > 0 else 0.3
                self.vel_pub.publish(cmd)
            else:
                self.stop()
                self.get_logger().info("TARGET CENTERED. Ready to fire.")
                self.done_time = time.time()
                self.state = DockingState.DONE

    # We need to update the pose_callback to handle the Centering state too
    def pose_callback(self, msg):
        if not msg.poses:
            return
        
        # Original Searching Logic
        if self.state == DockingState.SEARCHING:
            # (Logic from above remains here)
            marker = msg.poses[0]
            self.stop()
            self.state = DockingState.LOCKING
            self.lock_time = time.time()
            x_cm, y_cm, z_cm = marker.position.x*100, marker.position.y*100, marker.position.z*100
            self.target_dist_m = math.sqrt(marker.position.x**2 + (marker.position.z - self.target_distance)**2)
            self.target_angle = math.atan2(marker.position.x, marker.position.z - self.target_distance)
            print("-" * 30)
            print(f"MARKER DETECTED! X:{x_cm:.1f} Y:{y_cm:.1f} Z:{z_cm:.1f}")
            print(f"Path: Turn {math.degrees(self.target_angle):+.1f}°, Drive {self.target_dist_m*100:.1f}cm")
            print("-" * 30)
            
        # Final Centering Logic
        elif self.state == DockingState.PHASE3_CENTERING:
            self.pose_callback_centering(msg.poses[0].position.x)

    def control_loop(self):
        # (Control logic from above goes here)
        # Added a restart loop
        if self.state == DockingState.DONE:
            if (time.time() - self.done_time) > 5.0:
                self.state = DockingState.SEARCHING
                self.get_logger().info("Restarting loop...")
        else:
            # Run the normal machine...
            pass

    def stop(self):
        self.vel_pub.publish(Twist())

    def _angle_diff(self, a, b):
        d = a - b
        while d > math.pi: d -= 2*math.pi
        while d < -math.pi: d += 2*math.pi
        return d

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDocker()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
import math

class ArucoDocker(Node):
    def __init__(self):
        super().__init__('aruco_docker')

        self.pose_sub = self.create_subscription(
            PoseArray, '/usbcam1_poses', self.pose_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # ── (ρ, α, β) Gains ─────────────────────────────────────────────
        # Signs are mandatory: k_rho>0, k_alpha>k_rho>0, k_beta<0
        self.k_rho   =  0.3
        self.k_alpha =  0.8
        self.k_beta  = -0.15

        # ── Target ──────────────────────────────────────────────────────
        self.target_dist = 0.25
        # +π/2 or -π/2 depending on which side the dock faces — flip if wrong
        self.desired_final_heading = math.pi / 2

        # ── Speed limits ─────────────────────────────────────────────────
        self.max_v = 0.12
        self.max_w = 0.45

        # ── FOV loss recovery ────────────────────────────────────────────
        # Instead of stopping, continue rotating toward last known direction
        self.fov_loss_timeout   = 1.5    # seconds before giving up entirely
        self.last_known_alpha   = None   # sign tells us which way to spin
        self.last_known_rho     = None
        self.recovery_w         = 0.25   # slow rotation speed to re-acquire

        # ── Final cleanup ────────────────────────────────────────────────
        self.rho_switch_threshold = 0.04
        self.final_yaw_threshold  = 0.03

        # ── State ────────────────────────────────────────────────────────
        self.APPROACH  = 0
        self.FINAL_ROT = 1
        self.DONE      = 2
        self.state = self.APPROACH

        self.marker_x    = None
        self.marker_z    = None
        self.last_seen   = self.get_clock().now()
        self.marker_visible = False

        self.odom_yaw        = 0.0
        self.odom_yaw_start  = None

        self.get_logger().info("Aruco Docker ready — (ρ,α,β) controller with FOV recovery")

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny, cosy)

    def pose_callback(self, msg):
        if not msg.poses:
            self.marker_visible = False
            return
        self.marker_x       = msg.poses[0].position.x
        self.marker_z       = msg.poses[0].position.z
        self.last_seen      = self.get_clock().now()
        self.marker_visible = True

        # Always cache last known bearing so recovery spins the right way
        self.last_known_alpha = math.atan2(self.marker_x, self.marker_z)
        self.last_known_rho   = math.sqrt(self.marker_x**2 + self.marker_z**2) - self.target_dist

    # ────────────────────────────────────────────────────────────────────
    def control_loop(self):
        if self.state == self.DONE:
            return

        elapsed = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9

        # ── FINAL_ROT ────────────────────────────────────────────────────
        if self.state == self.FINAL_ROT:
            if self.odom_yaw_start is None:
                self.odom_yaw_start = self.odom_yaw
                self.get_logger().info("Final orientation cleanup started.")

            rotated   = self._angle_diff(self.odom_yaw, self.odom_yaw_start)
            remaining = self.desired_final_heading - rotated

            if abs(remaining) > self.final_yaw_threshold:
                cmd = Twist()
                cmd.angular.z = max(min(remaining * 1.2, self.max_w), -self.max_w)
                self.vel_pub.publish(cmd)
                self.get_logger().info(
                    f"Final rot: {math.degrees(rotated):.1f}° / {math.degrees(self.desired_final_heading):.0f}°",
                    throttle_duration_sec=0.3
                )
            else:
                self.stop()
                self.state = self.DONE
                self.get_logger().info("✓ Docking complete.")
            return

        # ── APPROACH ─────────────────────────────────────────────────────

        # Marker visible → run full (ρ,α,β) controller
        if self.marker_visible and self.marker_x is not None:
            x = self.marker_x
            z = self.marker_z

            rho   = math.sqrt(x**2 + z**2) - self.target_dist
            alpha = math.atan2(x, z)
            beta  = self.desired_final_heading - alpha

            # ── FIX: negate angular terms ─────────────────────────────
            # Camera: +X is right, ROS: +angular.z is left.
            # Marker to the right → α>0 → must turn right → negative ω.
            v =  self.k_rho  * rho
            w = -(self.k_alpha * alpha + self.k_beta * beta)   # ← negated

            v = max(min(v, self.max_v), -self.max_v)
            w = max(min(w, self.max_w), -self.max_w)

            self.get_logger().info(
                f"ρ={rho:+.3f}m  α={math.degrees(alpha):+.1f}°  "
                f"β={math.degrees(beta):+.1f}°  v={v:+.3f}  ω={w:+.3f}",
                throttle_duration_sec=0.2
            )

            if abs(rho) < self.rho_switch_threshold:
                self.stop()
                self.state = self.FINAL_ROT
                return

            cmd = Twist()
            cmd.linear.x  = v
            cmd.angular.z = w
            self.vel_pub.publish(cmd)
            return

        # Marker NOT visible ──────────────────────────────────────────────
        if elapsed > self.fov_loss_timeout:
            # Truly lost — stop and wait
            self.stop()
            self.get_logger().warn(
                "Marker lost for too long. Stopped.",
                throttle_duration_sec=2.0
            )
            return

        # Short-term loss → dead-reckon: rotate toward last known position
        if self.last_known_alpha is not None:
            cmd = Twist()

            if self.last_known_rho is not None and self.last_known_rho < 0.1:
                # Very close — don't drive blind, only rotate slowly
                cmd.linear.x = 0.0
            else:
                # Still far — creep forward slightly while re-acquiring
                cmd.linear.x = 0.04

            # Spin in the direction the marker was last seen
            # last_known_alpha > 0 → marker was to the right → spin right (negative ω)
            cmd.angular.z = -math.copysign(self.recovery_w, self.last_known_alpha)

            self.vel_pub.publish(cmd)
            self.get_logger().warn(
                f"FOV loss recovery — spinning {'right' if self.last_known_alpha > 0 else 'left'} "
                f"({elapsed:.1f}s / {self.fov_loss_timeout:.1f}s)",
                throttle_duration_sec=0.3
            )

    # ────────────────────────────────────────────────────────────────────
    def stop(self):
        self.vel_pub.publish(Twist())

    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        while d >  math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDocker()
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