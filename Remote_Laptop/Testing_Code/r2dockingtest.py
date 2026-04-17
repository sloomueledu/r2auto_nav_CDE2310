import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
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
        self.gpio_pub = self.create_publisher(String, '/gpio_commands', 10)  # For signaling docking status
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # ── (ρ, α, β) Gains ─────────────────────────────────────────────
        # Signs are mandatory: k_rho>0, k_alpha>k_rho>0, k_beta<0
        self.k_rho   =  0.3
        self.k_alpha =  0.8
        self.k_beta  = -0.15

        # ── Target ──────────────────────────────────────────────────────
        self.target_dist = 0.25
        self.desired_final_heading = 0.0

        # ── Speed limits ─────────────────────────────────────────────────
        self.max_v = 0.12
        self.max_w = 0.50

        # ── FOV loss recovery ────────────────────────────────────────────
        # Instead of stopping, continue rotating toward last known direction
        self.fov_loss_timeout   = 1.5    # seconds before giving up entirely
        self.last_known_alpha   = None   # sign tells us which way to spin
        self.last_known_rho     = None
        self.recovery_w         = 0.25   # slow rotation speed to re-acquire

        # ── Final cleanup ────────────────────────────────────────────────
        self.rho_switch_threshold = 0.04
        self.alpha_tolerance = 0.05 

        # ── State ────────────────────────────────────────────────────────
        self.APPROACH  = 0
        self.FINAL_ALIGN = 1
        self.DONE      = 2
        self.state = self.APPROACH

        self.commandSent = False

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
            self.stop()
            if not self.commandSent:
                # Publish a GPIO command to signal successful docking (e.g., turn on an LED)
                gpio_cmd = String()
                gpio_cmd = "LAUNCH"  # Example: linear.x=1.0 means "docking complete"
                self.gpio_pub.publish(String(data='LAUNCH'))
                self.commandSent = True
            return

        elapsed = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9

        # ── FINAL_ALIGN (Visual Centering) ──────────────────────────────
        if self.state == self.FINAL_ALIGN:
            if not self.marker_visible or self.marker_x is None:
                self.stop()
                self.get_logger().warn("Lost marker during final alignment! Stopping here.")
                self.state = self.DONE 
                return

            # Re-calculate the bearing to the marker
            alpha = math.atan2(self.marker_x, self.marker_z)
            
            if abs(alpha) > self.alpha_tolerance:
                cmd = Twist()
                # Turn toward the marker (negative sign because +X is right, but +Z angular is left)
                w = -(0.8 * alpha) # 0.8 is the proportional gain, tune if it spins too fast/slow
                cmd.angular.z = max(min(w, self.max_w), -self.max_w)
                self.vel_pub.publish(cmd)
                
                self.get_logger().info(
                    f"Centering... offset: {math.degrees(alpha):.1f}°",
                    throttle_duration_sec=0.3
                )
            else:
                self.stop()
                self.state = self.DONE
                self.get_logger().info("✓ Docking complete and centered.")
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
                # Check if we are off-centered
                if abs(alpha) > self.alpha_tolerance:
                    self.state = self.FINAL_ALIGN
                    self.get_logger().info(f"Target distance reached, but off-center by {math.degrees(alpha):.1f}°. Aligning...")
                else:
                    self.state = self.DONE
                    self.get_logger().info("✓ Docking complete. Perfectly aligned.")
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