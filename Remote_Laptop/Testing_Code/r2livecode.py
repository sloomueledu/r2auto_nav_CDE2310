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
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import tf2_ros
from tf2_ros import TransformException, LookupException, ConnectivityException, ExtrapolationException

#CONSTANTS
STOP_DISTANCE = 0.35
SIDE_THRESHOLD = 0.20
GOAL_THRESHOLD = 0.20
SCANFILE = 'lidar.txt'
MAPFILE = 'map.txt'

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
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

    return roll_x, pitch_y, yaw_z # in radians

#The class below is for Regulated Pure Pursuit:
class RegulatedPurePursuit():
    def __init__(self):
        #these are to be fine tuned once testing begins
        self.lookaheaddist = 0.3
        self.max_speed = 0.22
        self.min_speed = 0.05
        self.max_angular_v = 0.2
        self.safety_factor = 3.0
        self.rotate_threshold = 0.70
    
    def findpoint(self, cur_x, cur_y, path):
        for node in path:
            dist = math.sqrt((node.x - cur_x)**2 + (node.y - cur_y)**2)
            # We want the first point that is OUTSIDE the lookahead circle
            # This ensures the robot is always pulled forward
            if dist > self.lookaheaddist:
                target = node
                break
        return target
        
    def command(self, cur_x, cur_y, cur_yaw, path):
        twist = Twist()
        if not path or len(path) < 1:
            return twist
        #get the target point
        target = self.findpoint(cur_x, cur_y, path)
        
        #get the difference in distance and angle from the target point to the robot
        dx = target.x - cur_x
        dy = target.y - cur_y
        target_angle = math.atan2(dy, dx)
        angle_diff =  target_angle - cur_yaw

        #normalise thios to be between -pi and pi
        # This does the same thing as your while loops
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        #check if the angle that we need to turn to is greater than our threshold
        #if yes, then we should get the robot to turn first and then refind the path
        #this is important as the turn could be at an intersection

        if abs(angle_diff) > self.rotate_threshold:
            return None
        
        distance = math.sqrt(dx**2 + dy**2)
        curve = 2.0 * math.sin(angle_diff) / max(0.01, distance) #the 0.01 here is a safeguard against divide by 0

        #based on the curvature, calculate the regulated speed
        reg_speed = self.max_speed / (1.0 + self.safety_factor * abs(curve))
        twist.linear.x = max(self.min_speed, min(self.max_speed, reg_speed))
        reg_angular = twist.linear.x * curve
        twist.angular.z = max(-self.max_angular_v, min(self.max_angular_v, reg_angular))

        return twist

# The node below is to help us in our pathfinding algorithm
# essentially, this helps us to log positions on a map, and generate the neighbouring nodes around a source node
# to aid inb our bfs later
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
        neighbours = []
        """
        We are going to check for neighbours within a 3x3 grid around the parent node.
        How this works is this:

        [x-1,y-1]   [x-1,y]     [x-1,y+1]
        [x,y-1]     [PARENT]    [x,y+1]
        [x+1,y-1]   [x+1,y]     [x+1,y+1]

        for dx in [-1, 0, 1]:
            for dy in [-1, 0,1]:
                if dx == 0 and dy == 0:
                    continue
                nx = self.x + dx
                ny = self.y + dy
                #this check is to prevent reading off the map
                if 0 <= nx < max_x and 0 <= ny < max_y:
                    neighbours.append(MapNode(nx, ny, parent=self))
        return neighbours
        """
        neighbours = []
        # Strictly Up, Down, Left, Right (No diagonals to prevent wall clipping)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = self.x + dx
            ny = self.y + dy
            if 0 <= nx < max_x and 0 <= ny < max_y:
                neighbours.append(MapNode(nx, ny, parent=self))
        return neighbours

"""
this is where the fun begins

Game Plan:
1. Get the current position of the robot on the map
2. Using BFS, seasrch for a frontier for the robot to go to
2a. if frontiers fails, check for a free spot that is safe to navigate to
3. Plan a path for the robot to follow
4. send this path planned to the regulated pure pursuit controller for path execution
5. go to the goal point
6. repeat until all areeas of the map has been explored

Interrupt if:
1. Any of the onbaord cameras detect an ArUco marker
2. Robot is going to collide witgh a wall
3. any errors occur
"""

class AutoPilot(Node):
    def __init__(self):
        super().__init__('autopilot_node')
        self.rpp_controller = RegulatedPurePursuit() #this is our path executioner
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer,self)

        self.publisher_ = self.create_publisher(Twist,'cmd_vel',10)
        self.gpio_pub = self.create_publisher(String, '/gpio_commands', 10)  # For signaling docking status
        
        # This is to help us see our goal point on RVis
        self.marker_publisher = self.create_publisher(Marker,'goal_marker',10)

        # This is to help us see the planned path in RVis
        self.path_publisher = self.create_publisher(Path, 'planned_path', 10)

        # This is to help us see the lookahead point in RVis
        self.lookahead_publisher = self.create_publisher(Marker, 'lookahead_marker', 10)

        #variables initialisation
        self.path = []
        self.boink = 0
        self.goal = None
        self.state = 'PLANNING' #this is to keep track of the state of the robot
        self.rotation_start_time = None
        self.escape_direction_locked = None
        self.escape_start_time = None
        self.escape_duration = 1.5  # Seconds to drive blindly away from a trap
        self.escape_speed = 0.1     # Linear speed during escape (m/s)
        self.turning_timeout = 20.0 # seconds, this is to prevent the robot from getting stuck in a turning state for too long
        self.recovery_angle = None
        self.turn_angle_by = (math.pi / 9) # threshold to turn the robot by, this is to help the robot to get unstuck when it is trapped in a corner or narrow path
        self.wallinfdist = 3 # how far walls will influence the path, in terms of number of pooled cells. This is to help the robot to stay away from walls and navigate through narrow paths more effectively
        self.maxshift = 1.5 # the maximum number of cells that a point in the path can be shifted by to avoid walls. This is to prevent the path from being shifted so much that it becomes inefficient or goes off course

        # STATION TRACKERS
        self.valid_station_ids = [0, 1]
        self.responseHeard = False
        self.current_docking_id = None
        self.STATION_A_COMPLETED = False
        self.STATION_B_COMPLETED = False

        self.piResponseSub = self.create_subscription(String, 'rpi_response', self.pi_response_callback, 10)
        self.piResponseSub #prevent unused variable warning

        # ARUCO DOCKING
        self.marker_sub = self.create_subscription(
            ArucoMarkers, '/usbcam1_markers', self.marker_callback, 10
        )
        self.marker_sub #prevent unused variable warning

        # ── Docking Variables ──
        self.k_rho = 0.3
        self.k_alpha = 0.8
        self.k_beta = -0.15
        self.target_dist = 0.25
        self.desired_final_heading = 0.0
        self.docking_max_v = 0.12
        self.docking_max_w = 0.50
        
        self.fov_loss_timeout = 0.5
        self.last_known_alpha = None
        self.last_known_rho = None
        self.recovery_w = 0.25
        
        self.rho_switch_threshold = 0.04
        self.alpha_tolerance = 0.05 
        
        self.commandSent = False
        self.marker_x = None
        self.marker_z = None
        self.last_seen = self.get_clock().now()
        self.marker_visible = False
        self.pre_recovery_state = None

        # ── Undocking Variables ──
        self.undock_start_time = None
        self.undock_duration = 2.0  # Seconds to reverse
        self.undock_speed = -0.08   # Safe, slow reverse speed (m/s)

        # ── Searching State Variables ──
        self.search_last_yaw = None
        self.search_yaw_accumulated = 0.0
        self.search_spin_speed = 0.10  

        # Dividing up the LiDAR Data into 4 sections
        self.front = np.array([])
        self.left = np.array([])
        self.back = np.array([])
        self.right = np.array([])

        self.occ_subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.occ_callback,
            qos_profile_sensor_data)
        self.occ_subscription #prevent unused variable warning
        self.occdata = np.empty((0,0))
        self.res = 0
        self.origin = 0

        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data)
        self.scan_subscription #prevent unused variable warning
        self.laser_range = np.array([])
        
        # Define how often the control loop should run (in seconds)
        # 0.1 seconds = 10 Hz (10 times per second), which is standard for navigation
        timer_period = 0.1  
        
        # Create the timer that calls your state machine
        self.timer = self.create_timer(timer_period, self.controller)

        """
        to add in once this navigation part has been settled:
        1. OpenCV Control for ArUco Marker Detection and Alignment
        2. RPI Controller for GPIO Commands
        """

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

        if self.STATION_A_COMPLETED and self.STATION_B_COMPLETED:
            self.get_logger().info('MISSION COMPLETE!.')
            self.state = 'MISSION_COMPLETE'
            # Here you can set a new goal for returning to base or performing any final task

    def occ_callback(self,msg):
        # Get map metadata
        self.res = msg.info.resolution
        self.origin = msg.info.origin.position
        # 1. Convert to NumPy array
        msgdata = np.array(msg.data)
        # 2. Reshape to 2D (Height x Width)
        self.occdata = msgdata.reshape((msg.info.height, msg.info.width))
        # 3. Save to file
        np.savetxt(MAPFILE, self.occdata, fmt='%d')
    
    def scan_callback(self,msg):
        self.laser_range = np.array(msg.ranges)
        # replace out of range readings (0.0) with nan
        self.laser_range[self.laser_range == 0] = np.nan
        total_points = len(self.laser_range)
        # self.get_logger().info('Number of LaserScan points: %d' % total_points)
        np.savetxt(SCANFILE, self.laser_range)

        # 1. Calculate how many array indices represent 1 degree
        ppd = total_points / 360.0  # Points Per Degree

        # 2. Define your desired FOVs in degrees
        # Example setup: 120° Front, 80° Left, 80° Right, 80° Back
        front_fov = 100
        
        # 3. Calculate index boundaries based on angles
        # Front is split across the 0-degree mark (beginning and end of array)
        half_front = int((front_fov / 2) * ppd) 
        
        # FRONT: The last 'half_front' points and the first 'half_front' points
        front_right_scan = self.laser_range[-half_front:]
        front_left_scan = self.laser_range[0:half_front]
        self.front = np.concatenate((front_right_scan, front_left_scan))

        # LEFT: Starts where the front-left ends, spans 80 degrees
        left_start = half_front
        left_end = left_start + int(80 * ppd)
        self.left = self.laser_range[left_start:left_end]

        # BACK: Starts where Left ends, spans 80 degrees
        back_start = left_end
        back_end = back_start + int(80 * ppd)
        self.back = self.laser_range[back_start:back_end]

        # RIGHT: Starts where Back ends, goes up until the front-right scan begins
        right_start = back_end
        right_end = total_points - half_front
        self.right = self.laser_range[right_start:right_end]
    
    def marker_callback(self, msg):
        if not msg.marker_ids:
            self.marker_visible = False
            return

        closest_idx = None
        min_distance = float('inf')
        chosen_id = None

        # Greedy Distance: Find the closest VALID marker
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
        
        if dist > 2.5:
            self.marker_visible = False
            return
        
        target_pose = msg.poses[closest_idx]
        self.marker_x = target_pose.position.x
        self.marker_z = target_pose.position.z
        alpha_check = math.atan2(self.marker_x, self.marker_z)
        if abs(alpha_check) >= math.pi / 9: # if the angle to the marker is greater than 20 degrees, we consider it not visible for docking purposes
            self.marker_visible = False
            return
        self.last_seen = self.get_clock().now()
        self.marker_visible = True
        self.current_docking_id = chosen_id

        self.last_known_alpha = math.atan2(self.marker_x, self.marker_z)
        self.last_known_rho = math.sqrt(self.marker_x**2 + self.marker_z**2) - self.target_dist

        # Hijack Navigation if we spot a target marker
        if self.state in ['PLANNING', 'DRIVING', 'ALIGNING', 'SEARCHING'] and self.marker_visible:
            self.get_logger().warn(f'Target Station ({self.current_docking_id}) Detected! Taking over...')
            self.stopbot()
            self.path = [] 
            self.goal = None
            self.search_last_yaw = None 
            self.state = 'DOCKING_APPROACH'

    def get_orientation(self):
        transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        _, _, current_angle = euler_from_quaternion(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        )
        return transform.transform.translation.x, transform.transform.translation.y, current_angle

    def stopbot(self):
        self.get_logger().info('In stopbot')
        # publish to cmd_vel to move TurtleBot
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        # time.sleep(1)
        self.publisher_.publish(twist)
    
    # This function is to publish our found goal point which is to be marked out on RVis
    def publish_goal_marker(self, x, y):
        """
        Publishes a red sphere marker to RViz at the specified x, y coordinates.
        """
        marker = Marker()
        marker.header.frame_id = 'map' # Ensures it aligns with your map coordinates
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = 'goal_point'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set the position
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.0 # Keep it flat on the floor
        
        # Set orientation (needed even for spheres to prevent warnings)
        marker.pose.orientation.w = 1.0
        
        # Set the scale (size in meters)
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        # Set the color (Red: r=1.0, g=0.0, b=0.0) and Alpha (a=1.0 is fully opaque)
        marker.color.a = 1.0 
        marker.color.r = 1.0 
        marker.color.g = 0.0 
        marker.color.b = 0.0 
        self.marker_publisher.publish(marker)

    # This function helps us visualised the planned path in RVis
    def publish_planned_path(self, path_nodes):
        """
        Converts a list of MapNodes into a nav_msgs/Path message and publishes it for RViz.
        """
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for node in path_nodes:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp
            
            # Set waypoint coordinates
            pose.pose.position.x = float(node.x)
            pose.pose.position.y = float(node.y)
            pose.pose.position.z = 0.0
            
            # Default orientation pointing forward
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
        marker.pose.position.z = 0.1 # Slightly elevated so it doesn't clip into the floor
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0 
        marker.color.r = 0.0 
        marker.color.g = 0.5 
        marker.color.b = 1.0 # Bright Blue
        self.lookahead_publisher.publish(marker)

    # This function is to detect for obstacles all around the robot
    def checkObstacles(self):
        if len(self.laser_range) == 0 or np.isnan(self.laser_range).all():
            return False

        front_dist = np.nanmin(self.front) if len(self.front) > 0 and not np.isnan(self.front).all() else float('inf')

        if front_dist <= STOP_DISTANCE:
            # ── SAVE THE CURRENT STATE BEFORE SWITCHING ──
            self.pre_recovery_state = self.state 
            
            self.stopbot()
            self.get_logger().info(f'Obstacle detected while in {self.state}!')
            if self.state not in ['RECOVERY', 'ESCAPING']:
                self.pre_recovery_state = self.state 
                self.stopbot()
                self.get_logger().info(f'Recovering...')
                self.state = 'RECOVERY'
        return False

    def turn_in_place(self, target_angle, current_angle):
        # self.get_logger().info('Turning in place.')
        anglediff = target_angle - current_angle
        anglediff = (anglediff + math.pi) % (2 * math.pi) - math.pi

        twist = Twist()
        twist.linear.x = 0.0
        kp_yaw = 0.8
        turning_speed = anglediff * kp_yaw
        twist.angular.z = max(-self.rpp_controller.max_angular_v, min(self.rpp_controller.max_angular_v, turning_speed))
        self.publisher_.publish(twist)

    def evaluate_escape_direction(self):
        # Safely get the minimum distance for left and right
        left_dist = np.nanmin(self.left) if len(self.left) > 0 and not np.isnan(self.left).all() else 0.0
        right_dist = np.nanmin(self.right) if len(self.right) > 0 and not np.isnan(self.right).all() else 0.0

        self.get_logger().info(f'Clearance - Left: {left_dist:.2f}m, Right: {right_dist:.2f}m')

        MIN_SIDE_CLEARANCE = 0.35

        # 1. Check if Left has the required clearance and is more open than Right
        if left_dist >= MIN_SIDE_CLEARANCE and left_dist > right_dist:
            self.get_logger().info('Left side clear. Nudging Counter-Clockwise.')
            return self.turn_angle_by
            
        # 2. Check if Right has the required clearance
        elif right_dist >= MIN_SIDE_CLEARANCE and right_dist >= left_dist:
            self.get_logger().info('Right side clear. Nudging Clockwise.')
            return -self.turn_angle_by
            
        # 3. Fallback: Both sides are too tight (< 0.35m). Pick the "least bad" option.
        else:
            if left_dist >= right_dist:
                self.get_logger().info('Sides tight. Left is slightly better. Nudging Counter-Clockwise.')
                return self.turn_angle_by
            else:
                self.get_logger().info('Sides tight. Right is slightly better. Nudging Clockwise.')
                return -self.turn_angle_by

    def recoveryTurn(self):
        current_angle = self.get_orientation()[2]

        if self.recovery_angle is None:
            # 1. Check if we have already locked in an escape direction for this obstacle
            if self.escape_direction_locked is None:
                # We haven't picked a direction yet. Evaluate and LOCK it in.
                self.escape_direction_locked = self.evaluate_escape_direction()

            # 2. Use the locked direction to calculate the target angle
            raw_angle = current_angle + self.escape_direction_locked
            
            # 3. Normalize strictly between -pi and pi
            self.recovery_angle = (raw_angle + math.pi) % (2 * math.pi) - math.pi
        
        # Calculate the shortest angular distance to the target
        angle_diff = self.recovery_angle - current_angle
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        
        # Check if we are close enough to the target
        if abs(angle_diff) <= 0.05: 
            self.stopbot()
            self.get_logger().info('Recovery Turn Complete. Checking path...')
            self.recovery_angle = None
            return True 
        
        # Turn in place to the calculated recovery angle
        self.turn_in_place(self.recovery_angle, current_angle)
        return False

    def recoverySequence(self):
        #check if I have made a full turn already
        if self.recoveryTurn():

            #check if My Front is Clear
            if not self.checkObstacles():
                self.get_logger().info('Front Path is Clear after Recovery Turn. Escaping...')
                self.stopbot()

                # RESET THE DIRECTION LOCK HERE
                self.escape_direction_locked = None 

                # If we've hit walls 3 times trying to get to this frontier, drop it.
                if self.boink >= 3:
                    self.get_logger().warn('Stuck in a loop. Dropping current frontier goal...')
                    self.goal = None
                    self.boink = 0
                
                # Start the escape timer and change state
                self.escape_start_time = self.get_clock().now()
                self.state = 'ESCAPING'

            else:
                self.get_logger().info('Front Path is Still Blocked. Continuing Recovery Turn...')
                self.recoveryTurn()
        
        else:
            self.recoveryTurn()

    """
        Currently, there are 4 main states (more will be added once the camera integration has been fully completed)
        1. PLANNING
            - Path planning State
        2. DRIVING
            - Robot is moving to the goal via RPP
        3. ROTATE
            - RPP Controller determines angle is to bing and needs the robot to rotate in place
        4. RECOVERY
            - Robot has made an E-Stop due to an obstacle being detected infront and needs to back up

        Potential new states (to be added when ready):
        1. ALIGN
            - To align the robot with the ArUco Marker for Docking
        2. RPICONTROL
            - Control has been handed over to the RPI for Payload Delivery (Station A)
        3. TRACKING
            - Tracking ArUco Marker
        4. STATION
            - Robot is at a station and is completing it (might be a combination of RPICONTROL and TRACKING)
    """

    # This function is to make the robot actually move
    def mover(self):
        '''
        What this code is doing:
        1. Check if we are near the goal point
            a. if yes, stop and plan a new route
        2. Check if there is any obstacles
            a. if yes, exit to ther control loop
        
        if 1. and 2. are not satisfied, compute the rpp command
            a. if there is a need to spin on the spot, do so
            b. else, publish the command and move along the path
        '''
        # Safety Check: make sure that there is a valid path
        if not self.path or len(self.path) == 0:
            self.stopbot()
            self.state = 'PLANNING'
            return
        try:
            cur_x, cur_y, cur_yaw = self.get_orientation()
        except Exception as e:
            self.get_logger().warn(f'Robot Pose Not Found. Skipping control loop: {e}')
            return
        
        while len(self.path) > 1:
            dist_to_first = math.sqrt((self.path[0].x - cur_x)**2 + (self.path[0].y - cur_y)**2)
            if dist_to_first < 0.3: # If closer than 0.3m, it's "done"
                self.path.pop(0)
            else:
                break

        # Step 1: Check if we are near the goal point
        if getattr(self, 'goal', None) is not None:
            disttogoal = math.sqrt((self.goal.x - cur_x)**2 + (self.goal.y - cur_y)**2)

            if disttogoal <= GOAL_THRESHOLD:
                self.get_logger().info('Goal Reached!')
                self.stopbot()
                self.path = []
                self.goal = None
                self.boink = 0
                self.state = 'SEARCHING'
                return
        
        # Step 2: Check for obstacles
        obstacles = self.checkObstacles()
        if obstacles:
            self.boink += 1
            return
        
        # Step 3: Get the RPP Command
        cmd_vel = self.rpp_controller.command(cur_x, cur_y, cur_yaw, self.path)
        target = self.rpp_controller.findpoint(cur_x, cur_y, self.path)
        self.publish_lookahead_marker(target) # Visualise what we are aiming for

        # Step 3a: Check if there is a need to rotate on the spot
        if cmd_vel is None:
            now = self.get_clock().now()
            if self.rotation_start_time is None:
                self.rotation_start_time = now
            
            elif (now - self.rotation_start_time).nanoseconds / 1e9 > self.turning_timeout:
                self.get_logger().warn('Rotation timeout exceeded. Replanning new route...')
                self.stopbot()
                self.state = 'PLANNING'
                self.rotation_start_time = None
                return
            
            self.get_logger().info('Large Angle Detected. Switching to ALIGNING state.')
            self.stopbot()
            self.state = 'ALIGNING'
            return
            
        self.publisher_.publish(cmd_vel)

    # ── New Docking & Searching State Methods ───────────────────────────
    def search_marker_logic(self):
        self.get_logger().info('Searching for marker... Spinning in place.')
        try:
            _, _, cur_yaw = self.get_orientation()
        except Exception:
            return

        if self.search_last_yaw is not None:
            delta_yaw = cur_yaw - self.search_last_yaw
            delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi
            self.search_yaw_accumulated += abs(delta_yaw)

        self.search_last_yaw = cur_yaw

        if self.search_yaw_accumulated >= (2 * math.pi):
            self.get_logger().warn('Completed 360° search. No target marker found. Resuming exploration.')
            self.stopbot()
            self.state = 'PLANNING'
            self.search_last_yaw = None
            return

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = self.search_spin_speed
        self.publisher_.publish(twist)

    def docking_approach_logic(self):
        self.get_logger().info('Approaching Marker...')
        elapsed = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9

        current_dist_to_dock = float('inf')
        if self.marker_visible and self.marker_x is not None:
            current_dist_to_dock = self.marker_z

        # Collision avoidance integration
        ignore_obstacles_dist = 0.55
        if current_dist_to_dock > ignore_obstacles_dist:
            if self.checkObstacles():
                self.get_logger().warn('Obstacle blocking dock! Aborting approach -> RECOVERY.')
                self.current_docking_id = None 
                return

        # Docking Controller Math
        if self.marker_visible and self.marker_x is not None:
            x, z = self.marker_x, self.marker_z
            rho = math.sqrt(x**2 + z**2) - self.target_dist
            alpha = math.atan2(x, z)
            beta = self.desired_final_heading - alpha

            v = self.k_rho * rho
            w = -(self.k_alpha * alpha + self.k_beta * beta)

            v = max(min(v, self.docking_max_v), -self.docking_max_v)
            w = max(min(w, self.docking_max_w), -self.docking_max_w)

            if abs(rho) < self.rho_switch_threshold:
                if abs(alpha) > self.alpha_tolerance:
                    self.state = 'DOCKING_FINAL_ALIGN'
                    self.get_logger().info('Target distance reached. Final alignment...')
                else:
                    self.state = 'STATION'
                    self.get_logger().info('✓ Docking complete. Perfectly aligned.')
                return

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.publisher_.publish(cmd)
            return

        if elapsed > self.fov_loss_timeout:
            self.stopbot()
            self.get_logger().warn('Marker lost during approach. Dropping back to planning.')
            self.state = 'PLANNING' 
            return

        # FOV recovery
        if self.last_known_alpha is not None:
            cmd = Twist()
            if self.last_known_rho is not None and self.last_known_rho < 0.1:
                cmd.linear.x = 0.0
            else:
                cmd.linear.x = 0.04
            cmd.angular.z = -math.copysign(self.recovery_w, self.last_known_alpha)
            self.publisher_.publish(cmd)

    def docking_final_align_logic(self):
        self.get_logger().info('Final alignment with marker...')
        if not self.marker_visible or self.marker_x is None:
            self.stopbot()
            self.state = 'STATION'
            return

        alpha = math.atan2(self.marker_x, self.marker_z)
        
        if abs(alpha) > self.alpha_tolerance:
            cmd = Twist()
            w = -(0.8 * alpha) 
            cmd.angular.z = max(min(w, self.docking_max_w), -self.docking_max_w)
            self.publisher_.publish(cmd)
        else:
            self.stopbot()
            self.state = 'STATION'
            self.get_logger().info("✓ Final alignment complete.")

    # ── The Master Control Loop ─────────────────────────────────────────
    def controller(self):
        if self.state == 'PLANNING':
            self.path = self.planroute(goal=None if self.goal is None else self.goal)
            if self.path and len(self.path) > 0: 
                self.goal = self.path[-1] 
                self.state = "DRIVING"
            else:
                self.stopbot()
                self.goal = None 
                self.boink = 0   

        elif self.state == 'DRIVING':
            self.mover()
        
        elif self.state == 'ALIGNING':
            try:
                cur_x, cur_y, cur_yaw = self.get_orientation()
            except Exception: return

            if not self.path or len(self.path) == 0:
                self.state = 'PLANNING'
                return

            target = self.rpp_controller.findpoint(cur_x, cur_y, self.path)
            target_angle = math.atan2(target.y - cur_y, target.x - cur_x)
            angle_diff = target_angle - cur_yaw
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

            if abs(angle_diff) <= 0.15: 
                self.stopbot()
                self.state = 'DRIVING'
                self.rotation_start_time = None 
            else:
                now = self.get_clock().now()
                if self.rotation_start_time is not None and (now - self.rotation_start_time).nanoseconds / 1e9 > self.turning_timeout:
                    self.stopbot()
                    self.state = 'PLANNING'
                    self.rotation_start_time = None
                    return

                if len(self.front) > 0 and np.nanmin(self.front) <= SIDE_THRESHOLD:
                    self.stopbot()
                    self.state = 'RECOVERY'
                    return
                self.turn_in_place(target_angle, cur_yaw)

        elif self.state == 'RECOVERY':
            self.recoverySequence()     

        elif self.state == 'ESCAPING':
            now = self.get_clock().now()
            elapsed_time = (now - self.escape_start_time).nanoseconds / 1e9
            
            if elapsed_time < self.escape_duration:
                if not self.checkObstacles():
                    twist = Twist()
                    twist.linear.x = float(self.escape_speed)
                    twist.angular.z = 0.0
                    self.publisher_.publish(twist)
                    
                    if len(self.front) > 0 and np.nanmin(self.front) <= SIDE_THRESHOLD:
                        self.stopbot()
                        self.state = 'RECOVERY'
            else:
                self.stopbot()
                
                # ── REDIRECT BASED ON MEMORY ──
                if self.pre_recovery_state in ['DOCKING_APPROACH', 'DOCKING_FINAL_ALIGN']:
                    self.get_logger().info('Was docking before obstacle. Spinning to re-acquire marker...')
                    self.state = 'SEARCHING'
                    # Reset search tracking
                    self.search_yaw_accumulated = 0.0
                    try:
                        _, _, cur_yaw = self.get_orientation()
                        self.search_last_yaw = cur_yaw
                    except: pass
                else:
                    self.get_logger().info('Resuming standard navigation.')
                    self.state = 'PLANNING'
                
                self.pre_recovery_state = None # Clear memory

        elif self.state == 'SEARCHING':
            self.search_marker_logic()

        elif self.state == 'DOCKING_APPROACH':
            self.docking_approach_logic()
            
        elif self.state == 'DOCKING_FINAL_ALIGN':
            self.docking_final_align_logic()
            
        elif self.state == 'STATION':
            self.stopbot()
            if not self.commandSent:
                self.get_logger().info(f"Signaling payload for Station {self.current_docking_id} via GPIO...")
                self.gpio_pub.publish(String(data='LAUNCH'))
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

                # Tell the robot to resume its mission
                self.get_logger().info('Station complete! Resuming exploration...')
                self.undock_start_time = self.get_clock().now()
                self.state = 'UNDOCKING'
        
        elif self.state == 'UNDOCKING':
            now = self.get_clock().now()
            elapsed_time = (now - self.undock_start_time).nanoseconds / 1e9
            
            # ── Rear Collision Avoidance ──
            # Safely check the minimum distance behind the robot
            back_dist = np.nanmin(self.back) if len(self.back) > 0 and not np.isnan(self.back).all() else float('inf')
            
            # If an obstacle is closer than 0.25m to the rear, abort the reverse!
            if back_dist <= 0.25:
                self.get_logger().warn(f'Obstacle detected {back_dist:.2f}m behind! Aborting reverse.')
                self.stopbot()
                self.undock_start_time = None
                self.state = 'PLANNING' # Let the path planner figure out how to drive forward safely
                return

            # ── Normal Reversing ──
            if elapsed_time < self.undock_duration:
                # Path is clear, continue driving backward
                twist = Twist()
                twist.linear.x = float(self.undock_speed)
                twist.angular.z = 0.0
                self.publisher_.publish(twist)
            else:
                # Timer finished safely
                self.stopbot()
                self.get_logger().info('Undocking complete. Resuming frontier exploration...')
                self.undock_start_time = None
                self.state = 'PLANNING'

        elif self.state == 'MISSION_COMPLETE':
            self.stopbot()
    
def main(args=None):
    rclpy.init(args=args)
    autopilot_node = AutoPilot()

    # Allow RViz configuration time
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