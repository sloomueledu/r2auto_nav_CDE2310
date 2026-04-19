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
import tf2_geometry_msgs

#CONSTANTS
STOP_DISTANCE = 0.30
SIDE_THRESHOLD = 0.25
GOAL_THRESHOLD = 0.20
SCANFILE = 'lidar.txt'
MAPFILE = 'map.txt'

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

class RegulatedPurePursuit():
    def __init__(self):
        # CHANGE: slightly faster and bolder
        self.lookaheaddist = 0.20    # was 0.1 — 10cm was way too twitchy
        self.max_speed = 0.22
        self.min_speed = 0.05
        self.max_angular_v = 0.35    # was 0.2 — too slow to turn in maze
        self.safety_factor = 3.0
        self.rotate_threshold = 0.70  # anything >40 degrees
        # New Standoff Variables
        self.dock_ghost_x = None
        self.dock_ghost_y = None
        self.dock_ghost_yaw = None
        self.marker_yaw = None
        self.standoff_distance = 0.40 # Meters in front of dock
    
    def findpoint(self, cur_x, cur_y, path):
        target = path[-1]  # default to last point if all are inside lookahead
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

        if abs(angle_diff) > self.rotate_threshold:
            return None
        
        distance = math.sqrt(dx**2 + dy**2)
        curve = 2.0 * math.sin(angle_diff) / max(0.01, distance)
        reg_speed = self.max_speed / (1.0 + self.safety_factor * abs(curve))
        twist.linear.x = max(self.min_speed, min(self.max_speed, reg_speed))
        reg_angular = twist.linear.x * curve
        twist.angular.z = max(-self.max_angular_v, min(self.max_angular_v, reg_angular))
        return twist

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
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = self.x + dx
            ny = self.y + dy
            if 0 <= nx < max_x and 0 <= ny < max_y:
                neighbours.append(MapNode(nx, ny, parent=self))
        return neighbours

class AutoPilot(Node):
    def __init__(self):
        super().__init__('autopilot_node')
        self.rpp_controller = RegulatedPurePursuit()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.gpio_pub = self.create_publisher(String, '/gpio_commands', 10)
        self.marker_publisher = self.create_publisher(Marker, 'goal_marker', 10)
        self.path_publisher = self.create_publisher(Path, 'planned_path', 10)
        self.lookahead_publisher = self.create_publisher(Marker, 'lookahead_marker', 10)

        self.path = []
        self.boink = 0
        self.goal = None
        self.state = 'PLANNING'
        self.rotation_start_time = None
        self.escape_direction_locked = None
        self.escape_start_time = None
        self.escape_duration = 1.5
        self.escape_speed = 0.15
        self.front_fov = 90
        self.turning_timeout = 20.0
        self.recovery_angle = None
        self.turn_angle_by = (math.pi / 6)
        self.wallinfdist = 4
        self.maxshift = 2.0

        # STATION TRACKERS
        self.valid_station_ids = [0, 1]
        self.responseHeard = False
        self.current_docking_id = None
        self.STATION_A_COMPLETED = False
        self.STATION_B_COMPLETED = False

        self.piResponseSub = self.create_subscription(String, 'rpi_response', self.pi_response_callback, 10)
        self.piResponseSub

        self.marker_sub = self.create_subscription(
            ArucoMarkers, '/usbcam1_markers', self.marker_callback, 10
        )
        self.marker_sub

        # Docking Variables
        self.k_rho = 0.2
        self.k_alpha = 1.6
        self.k_beta = -0.40
        self.target_dist = 0.25
        self.desired_final_heading = 0.0
        self.docking_max_v = 0.12
        self.docking_max_w = 0.50
        self.fov_loss_timeout = 0.75
        self.last_known_alpha = None
        self.last_known_rho = None
        self.recovery_w = 0.25
        self.rho_switch_threshold = 0.1
        self.alpha_tolerance = 0.05
        self.commandSent = False
        self.marker_x = None
        self.marker_z = None
        self.last_seen = self.get_clock().now()
        self.marker_visible = False
        self.pre_recovery_state = None

        # Docking Ghost Goal (map frame, for navigation)
        self.dock_ghost_x = None
        self.dock_ghost_y = None
        self.dock_ghost_yaw = None

        # Undocking Variables
        self.undock_start_time = None
        self.undock_duration = 2.0
        self.undock_speed = -0.08

        # Searching State Variables
        self.search_last_yaw = None
        self.search_yaw_accumulated = 0.0
        self.search_spin_speed = 0.15   # CHANGE: was 0.10 — 360° in ~14s now

        self.front = np.array([])
        self.left = np.array([])
        self.back = np.array([])
        self.right = np.array([])

        self.occ_subscription = self.create_subscription(
            OccupancyGrid, 'map', self.occ_callback, qos_profile_sensor_data)
        self.occ_subscription
        self.occdata = np.empty((0, 0))
        self.res = 0
        self.origin = 0

        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.scan_subscription
        self.laser_range = np.array([])

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.controller)

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
        if not msg.marker_ids:
            self.marker_visible = False
            return

        for i, detected_id in enumerate(msg.marker_ids):
            if detected_id in self.valid_station_ids:
                # --- Camera-frame data (for visual servoing) ---
                self.marker_x = msg.poses[i].position.x
                self.marker_z = msg.poses[i].position.z
                _, _, cam_yaw = euler_from_quaternion(
                    msg.poses[i].orientation.x, msg.poses[i].orientation.y,
                    msg.poses[i].orientation.z, msg.poses[i].orientation.w
                )
                self.marker_yaw = cam_yaw  # relative to camera, used for heading trim in approach
                self.marker_visible = True
                self.last_seen = self.get_clock().now()

                # --- Map-frame data (for navigation to standoff) ---
                try:
                    marker_pose = PoseStamped()
                    marker_pose.header.frame_id = msg.header.frame_id
                    marker_pose.pose = msg.poses[i]
                    map_pose = self.tf_buffer.transform(marker_pose, 'map')
                    _, _, m_yaw = euler_from_quaternion(
                        map_pose.pose.orientation.x, map_pose.pose.orientation.y,
                        map_pose.pose.orientation.z, map_pose.pose.orientation.w
                    )
                    # Standoff: 40cm in front of the marker face in map frame
                    gx = map_pose.pose.position.x + 0.40 * math.cos(m_yaw)
                    gy = map_pose.pose.position.y + 0.40 * math.sin(m_yaw)
                    gyaw = (m_yaw + math.pi + math.pi) % (2 * math.pi) - math.pi  # face back toward marker

                    # Keep updating standoff while not yet navigating to it
                    if self.state not in ['DOCKING_ARC', 'DOCKING_PIVOT', 'DOCKING_APPROACH', 'DOCKING_FINAL_ALIGN', 'STATION', 'WAITING_FOR_PI']:
                        self.dock_ghost_x = gx
                        self.dock_ghost_y = gy
                        self.dock_ghost_yaw = gyaw

                except (TransformException):
                    pass

                # Trigger docking sequence
                if self.state in ['PLANNING', 'DRIVING', 'SEARCHING']:
                    self.get_logger().info(f'Marker {detected_id} acquired. Navigating to standoff...')
                    self.current_docking_id = detected_id
                    self.state = 'DOCKING_ARC'

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
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)

    #the fun begins
    def planroute(self, goal=None):
        occ_grid = self.occdata
        #check if it is empty
        if occ_grid.shape[0] == 0:
            return []
        found_path = False

        # We are going to downscale the map by 3 here to save on computational cost
        # Step 1: Pad the map
        pad_y = (3 - occ_grid.shape[0] % 3) % 3
        pad_x = (3 - occ_grid.shape[1] % 3) % 3
        occ_grid = np.pad(occ_grid, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=-1)

        # Step 2: Reshape the map
        height, width = occ_grid.shape
        occ_pooled_grid = occ_grid.reshape(height // 3, 3, width // 3, 3).max(axis=(1,3)) # // is used here since it returns an integer (by flooring the result)
        # at this point in the code, every 3x3 block in the original map has been downscaled into a single grid   

        # get an array of wall coordinates and their respective distances
        wall_distance = np.zeros_like(occ_pooled_grid, dtype=float)
        WALL_THRESHOLD = 50 # any cells that has a value of >=75 in the pooled map is considered a wall
        wall_cells = np.where(occ_pooled_grid >= WALL_THRESHOLD)
        
        # calculate the distance betweeen each cell to the nearest wall
        for y in range(occ_pooled_grid.shape[0]):
            for x in range(occ_pooled_grid.shape[1]):
                if occ_pooled_grid[y,x] >= WALL_THRESHOLD:
                    wall_distance[y,x] = 0 # This is a wall
                else:
                    min_dist = float('inf')
                    # we want to find the closest wall to the cell
                    for wy, wx in zip(wall_cells[0], wall_cells[1]):
                        dist = math.sqrt((y-wy)**2 + (x - wx)**2)
                        min_dist = min(min_dist, dist)
                    wall_distance[y,x] = min_dist

        # we need to locate where the robot is in the pooled map
        # note that becausew of the downscalling, the pooled coords might cause the software
        # to percive the bot as being inside a wall when it isn't. As such, we need to account for that and adjust accordingly
        
        # these are placeholders for the pooled bot location
        pbotloc_x = -1
        pbotloc_y = -1

        try:        
            # these are placeholders for the pooled goal coords
            pgoal_x = -1
            pgoal_y = -1

            # get the actual bot location in the original map
            if self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
                cur_x, cur_y, _ = self.get_orientation()
                obotloc_x = (cur_x - self.origin.x)/self.res
                obotloc_y = (cur_y - self.origin.y)/self.res
                
                pbotloc_x = max(0, min(int(obotloc_x) // 3, occ_pooled_grid.shape[1] - 1))
                pbotloc_y = max(0, min(int(obotloc_y) // 3, occ_pooled_grid.shape[0] - 1))
                potential_cells = [(pbotloc_y,pbotloc_x)]
                
                # we shall check if the pooled coords shows that the bot is in a wall here
                if occ_pooled_grid[pbotloc_y, pbotloc_x] >= WALL_THRESHOLD:
                    potential_cells = []
                    for i in range(-1,2):
                        for j in range(-1,2):
                            ny, nx = pbotloc_y + i, pbotloc_x + j
                            if 0 <= ny < occ_pooled_grid.shape[0] and 0 <= nx < occ_pooled_grid.shape[1]:
                                if occ_pooled_grid[ny, nx] < WALL_THRESHOLD:
                                    potential_cells.append((ny,nx))
                    
                    # we will take the closest pooled coord to be where the bot lies on the pooled map
                    min_dist = float('inf')
                    for cell in potential_cells:
                        dist = math.sqrt((cell[0] - pbotloc_y)**2 + (cell[1] - pbotloc_x)**2)
                        if dist < min_dist:
                            min_dist = dist
                            pbotloc_y, pbotloc_x = cell
                state = 'PLANNING' # go back to planning state to find the next station
                # at this point, we would have identified here the bot is in the pooled map
                self.get_logger().info(f'Robot Location on Pooled Map: x={pbotloc_x}, y={pbotloc_y}')
                self.get_logger().info(f'Robot Location on Actual Map: x={obotloc_x}, y={obotloc_y}')

                # get the actual goal coords in the original map
                if goal is not None:
                    ogoal_x, ogoal_y = goal.x, goal.y
                    pgoal_x = min(int((ogoal_x - self.origin.x) / self.res) // 3, occ_pooled_grid.shape[1] - 1)
                    pgoal_y = min(int((ogoal_y - self.origin.y) / self.res) // 3, occ_pooled_grid.shape[0] - 1)
                self.get_logger().info(f'Goal Coordinates on Pooled Map: x={pgoal_x}, y={pgoal_y}')
            else:
                self.get_logger().warn('Transform from "map" and "base_link" is not available. Retrying...')
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'Transform Lookup Failed: {e}')
        
        if pbotloc_x == -1 or pbotloc_y == -1:
            return [] # tells us that there is an error with finding the bot location
        
        """
        Recap of what we have done so far in the code:
        1. Downscaled the map by transforming every 3x3 grid in the original map into a single grid
           and assigned it the maximum value out of the 3x3 grid
        2. Found Where our Robot lies in the pooled map, adjusting if the original positions made it seem if the robot lies inside a wall
        3. Found where our goal lies in the pooled map
        4. Created a 'cost-map' of sorts that tells us where the walls are and the distance of cells from their respective nearest wall
        """
        # at this point, we are now ready to commence the BFS to
        # 1. Find the nearest frontier to go to (if no goal point)
        # 2. Get the path to the goal point

        # Setting up BFS
        self.get_logger().info(f'Starting BFS from x={pbotloc_x}, y={pbotloc_y}')
        start = MapNode(pbotloc_x, pbotloc_y)
        frontier = [start]
        visited = set() # this keeps track of points that has been checked before
        visited.add(start)

        if len(frontier) == 0:
            return []
        
        # main BFS Logic here
        check_node = None

        # Debug the starting cell
        start_val = occ_pooled_grid[int(start.y), int(start.x)]
        self.get_logger().info(f'Start cell (x={int(start.x)}, y={int(start.y)}) value: {start_val}')

        if start_val >= WALL_THRESHOLD:
            self.get_logger().warn('BFS failed: Robot is starting inside a wall on the pooled map!')
        
        while len(frontier) > 0:
            """
            HOW THIS IS GOING TO WORK:
            STEP 1: GET THE NODE THAT WE WANT TO EXPLORE
            STEP 2: THERE ARE 3 CASES:
                CASE 1: NODE IS A AN UNEXPLORED AREA (VALUE == -1)
                        A. CHECK IF THERE IS AN EXISTING GOAL
                        B. IF NO GOAL HAS BEEN SET PREVIOUSLY, SET THIS NEW POINT FOUND AS THE GOAL POINT
                        C. EXIT THE LOOP AS A PATH HAS BEEN FOUND
                IF A GOAL POINT HAS BEEN PRE-DEFINED:
                CASE 2: NODE IS THE GOAL POINT
                        A. BREAK OUT OF THE LOOP AS A PATH HAS BEEN FOUND
                CASE 3: NODE IS A WALL
                        A. SKIP THIS POINT AND STEP 3 AS WE CANNOT FIND A PATH OUT OF IT
            STEP 3: GENERATE A LIST OF NEIGHBOURS AROUND THE EXPLORED CELL
            """
            # Step 1:
            check_node = frontier.pop(0)
            visited.add(check_node)

            # Step 2:
            # case 1
            if goal is None:
                # 1. Is the current node unexplored?
                if occ_pooled_grid[int(check_node.y), int(check_node.x)] == -1:
                    # 2. Is the parent (the previous step) confirmed free space?
                    # We check if the parent value is between 0 and your WALL_THRESHOLD
                    if check_node.parent:
                        parent_val = occ_pooled_grid[int(check_node.parent.y), int(check_node.parent.x)]
                        if 0 <= parent_val < WALL_THRESHOLD: 
                            self.get_logger().info(f'Valid internal frontier found at {int(check_node.x)}, {int(check_node.y)}')
                            found_path = True
                            break
            else:
                # case 2:
                if check_node.y == pgoal_y and check_node.x == pgoal_x:
                    self.get_logger().info(f'Goal Point Found')
                    found_path = True
                    break
            
            # case 3
            cell_val = occ_pooled_grid[int(check_node.y), int(check_node.x)]
            if cell_val >= WALL_THRESHOLD:
                    continue
            
            # Prevent walking through the void when routing to a known goal
            if goal is not None and cell_val == -1:
                # Allow it ONLY if it is the exact goal node we are trying to reach
                if not (check_node.y == pgoal_y and check_node.x == pgoal_x):
                    continue

            # Step 3:
            neighbours = check_node.generate_neighbours(occ_pooled_grid.shape[1], occ_pooled_grid.shape[0])
            neighbours.sort(key=lambda n: -wall_distance[int(n.y), int(n.x)]) 
            # this sorts the neighbour list according to their distanced from the closest wall
            # we want to prioritise points that are further way from potential walls

            for neighbour in neighbours:
                if neighbour in visited:
                    continue # skip nodes that have been checked previously
                frontier.append(neighbour)
                visited.add(neighbour)
                neighbour.parent = check_node
        
        # --- NEW FALLBACK: RANDOM FREE SPOT ---
        # If we were exploring (goal is None) and failed to find a frontier
        if not found_path and goal is None:
            self.get_logger().warn('No frontier found! Wandering to a random known spot...')
            
            valid_random_spots = []
            for node in visited:
                val = occ_pooled_grid[int(node.y), int(node.x)]
                
                # Check if the node is confirmed free space
                if 0 <= val < WALL_THRESHOLD:
                    # Calculate distance from the robot (in pooled grid units)
                    dist = math.sqrt((node.x - pbotloc_x)**2 + (node.y - pbotloc_y)**2)
                    
                    # Ensure the spot is at least ~5 pooled cells away so it actually drives somewhere
                    if dist > 5.0:
                        valid_random_spots.append(node)
            
            # If we found safe spots, pick one at random
            if len(valid_random_spots) > 0:
                check_node = random.choice(valid_random_spots)
                found_path = True
                self.get_logger().info(f'Fallback successful: Heading to x={int(check_node.x)}, y={int(check_node.y)}')
            else:
                self.get_logger().error('Fallback failed: No safe open space found to wander to.')
        # --------------------------------------

        path = []
        if not found_path or check_node is None:
            self.get_logger().info('No Path Found')
            return []

        """
        RECAP:
        AT THIS STAGE, WE WOULD HAVE:
        1. FOUND A NEW GOAL POINT IF NO GOAL HAS BEEN FOUND PREVIOUSLY
        2. FOUND A PATH TO THE GOAL POINT BY BFS

        WHAT WE NEED TO DO NOW
        1. REFINE THE PATH TO ENSURE THE SAFETY OF THE ROBOT
        """
        path = []
        if not found_path or check_node is None:
            self.get_logger().info('No Path Found')
            return []
        
        # we will now adjust the path to make sure that the robot stays safe and away from walls
        while check_node is not None:
            y, x = int(check_node.y), int(check_node.x)
            shifty = 0
            shiftx = 0

            # we will now determine how much we need to shift each point in the path by and the shift direction
            for dy in range(-self.wallinfdist, self.wallinfdist + 1):
                for dx in range(-self.wallinfdist, self.wallinfdist + 1):
                    ny, nx = y + dy, x + dx

                    # check if we have gone out of bounds
                    if 0 <= ny < occ_pooled_grid.shape[0] and 0 <= nx < occ_pooled_grid.shape[1]:
                        # check if this new point is in a wall
                        if occ_pooled_grid[ny, nx] >= WALL_THRESHOLD:
                            # compute how much to shift by
                            shiftdist = max(0.1, math.sqrt(dy**2 + dx**2))
                            # the negative here means shift away
                            dirx = -dx / shiftdist
                            diry = -dy / shiftdist

                            # scale the shift pi_response'
                            # inverse relationship: closer you are to the wall, more you get pusahed away
                            magnitude = self.maxshift * (1.0 / shiftdist)

                            shiftx += dirx * magnitude
                            shifty += diry * magnitude
            # apply the shift
            shiftedx = x + shiftx
            shiftedy = y + shifty

            # double check to make sure we didnt shift into a new wall or go out of bounds
            if 0 <= shiftedy < occ_pooled_grid.shape[0] and 0 <= shiftedx < occ_pooled_grid.shape[1]:
                if 0 <= occ_pooled_grid[int(shiftedy), int(shiftedx)] < WALL_THRESHOLD:
                    check_node.x = shiftedx
                    check_node.y = shiftedy
                else:
                    # apply a smaller shift
                    check_node.x = x + 0.25 * shiftx # might need to fine tune the 0.25 here
                    check_node.y = y + 0.25 * shifty
            
            # we will need to convert back the pooled coords to the original map coords
            orimap_x = (check_node.x * 3 + 1.5) * self.res + self.origin.x
            orimap_y = (check_node.y * 3 + 1.5) * self.res + self.origin.y

            toAppend = MapNode(orimap_x, orimap_y)
            self.get_logger().info(f'Path waypoint: x={toAppend.x}, y={toAppend.y}')
            path.append(toAppend)
            check_node = check_node.parent
        
        if len(path) > 0:
            self.goalpoint = path[0]
            self.get_logger().info(f'Goal Point: x={self.goalpoint.x}, y={self.goalpoint.y}')
            self.publish_goal_marker(self.goalpoint.x, self.goalpoint.y)
        
        path.reverse() # we planned the path from the goal to the bot. what we want is opposite
        """
        Recap of what we have done so far:
        1. We have adjusted the path waypoints to ensure that the robot stays far away from walls

        What we need to do now:
        1. Smooth Out the path
        """
        if len(path) > 2: # only smooth paths if there are more than 2 waypoints inside
            smoothpath = [path[0]] # keep the first point
            smoothingwindow = 7 # this may need to be fine tuned
            for i in range(1, len(path) -1):
                window_start = max(0, i - smoothingwindow // 2)
                window_end = min(len(path), i + smoothingwindow // 2 + 1)
                window = path[window_start:window_end]

                avgx = sum(node.x for node in window) / len(window)
                avgy = sum(node.y for node in window) / len(window)
                smoothpath.append(MapNode(avgx,avgy))
            smoothpath.append(path[-1]) # keep the last goal point
            path = smoothpath
        
        # This function call helps to visualise the planned path in RVis
        self.publish_planned_path(path)
        return path

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

    def checkObstacles(self):
        if len(self.laser_range) == 0 or np.isnan(self.laser_range).all():
            return False
            
        # Ignore obstacles when in docking states so the station doesn't trip the lidar
        if self.state in ['DOCKING_ARC', 'DOCKING_PIVOT', 'DOCKING_APPROACH', 'DOCKING_FINAL_ALIGN']:
            return False

        front_dist = np.nanmin(self.front) if len(self.front) > 0 and not np.isnan(self.front).all() else float('inf')
        if front_dist <= STOP_DISTANCE:
            self.stopbot()
            if self.state not in ['RECOVERY', 'ESCAPING']:
                self.pre_recovery_state = self.state
                self.state = 'RECOVERY'
        return False

    def turn_in_place(self, target_angle, current_angle):
        anglediff = target_angle - current_angle
        anglediff = (anglediff + math.pi) % (2 * math.pi) - math.pi
        twist = Twist()
        twist.linear.x = 0.0
        kp_yaw = 0.8
        turning_speed = anglediff * kp_yaw
        twist.angular.z = max(-self.rpp_controller.max_angular_v, min(self.rpp_controller.max_angular_v, turning_speed))
        self.publisher_.publish(twist)

    def evaluate_escape_direction(self):
        left_dist  = np.nanmin(self.left)  if len(self.left)  > 0 and not np.isnan(self.left).all()  else 0.0
        right_dist = np.nanmin(self.right) if len(self.right) > 0 and not np.isnan(self.right).all() else 0.0
        self.get_logger().info(f'Clearance - Left: {left_dist:.2f}m, Right: {right_dist:.2f}m')
        MIN_SIDE_CLEARANCE = 0.35
        if left_dist >= MIN_SIDE_CLEARANCE and left_dist > right_dist:
            self.get_logger().info('Left side clear. Nudging Counter-Clockwise.')
            return self.turn_angle_by
        elif right_dist >= MIN_SIDE_CLEARANCE and right_dist >= left_dist:
            self.get_logger().info('Right side clear. Nudging Clockwise.')
            return -self.turn_angle_by
        else:
            if left_dist >= right_dist:
                self.get_logger().info('Sides tight. Left is slightly better.')
                return self.turn_angle_by
            else:
                self.get_logger().info('Sides tight. Right is slightly better.')
                return -self.turn_angle_by

    def recoveryTurn(self):
        current_angle = self.get_orientation()[2]
        if self.recovery_angle is None:
            if self.escape_direction_locked is None:
                self.escape_direction_locked = self.evaluate_escape_direction()
            raw_angle = current_angle + self.escape_direction_locked
            self.recovery_angle = (raw_angle + math.pi) % (2 * math.pi) - math.pi
        angle_diff = self.recovery_angle - current_angle
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        if abs(angle_diff) <= 0.05:
            self.stopbot()
            self.get_logger().info('Recovery Turn Complete. Checking path...')
            self.recovery_angle = None
            return True
        self.turn_in_place(self.recovery_angle, current_angle)
        return False

    def recoverySequence(self):
        if self.recoveryTurn():
            if not self.checkObstacles():
                self.get_logger().info('Front Path is Clear after Recovery Turn. Escaping...')
                self.stopbot()
                self.escape_direction_locked = None
                if self.boink >= 3:
                    self.get_logger().warn('Stuck in a loop. Dropping current frontier goal...')
                    self.goal = None
                    self.boink = 0
                self.escape_start_time = self.get_clock().now()
                self.state = 'ESCAPING'
            else:
                self.get_logger().info('Front Path is Still Blocked. Continuing Recovery Turn...')
                self.recoveryTurn()
        else:
            self.recoveryTurn()

    def mover(self):
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
            if dist_to_first < 0.3:
                self.path.pop(0)
            else:
                break

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

        obstacles = self.checkObstacles()
        if obstacles:
            self.boink += 1
            return

        cmd_vel = self.rpp_controller.command(cur_x, cur_y, cur_yaw, self.path)
        target = self.rpp_controller.findpoint(cur_x, cur_y, self.path)
        self.publish_lookahead_marker(target)

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
            self.get_logger().info('Large Angle Detected. Turning in place toward waypoint.')
            target = self.rpp_controller.findpoint(cur_x, cur_y, self.path)
            target_angle = math.atan2(target.y - cur_y, target.x - cur_x)
            self.turn_in_place(target_angle, cur_yaw)
            return

        # CHANGE: side clearance nudge for thin-edge / parallel wall clipping.
        # Front lidar (100° cone) can miss a wall edge that's exactly to the side.
        # If either side gets within 0.22m while driving, inject a small correction.
        SIDE_NUDGE_DIST = 0.22
        left_min  = float(np.nanmin(self.left))  if len(self.left)  > 0 and not np.isnan(self.left).all()  else float('inf')
        right_min = float(np.nanmin(self.right)) if len(self.right) > 0 and not np.isnan(self.right).all() else float('inf')
        if left_min < SIDE_NUDGE_DIST:
            cmd_vel.angular.z -= 0.25   # nudge right
        elif right_min < SIDE_NUDGE_DIST:
            cmd_vel.angular.z += 0.25   # nudge left

        self.publisher_.publish(cmd_vel)

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
    
    """
    def docking_approach_logic(self):
        self.get_logger().info('Approaching Marker...')
        elapsed = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9

        current_dist_to_dock = float('inf')
        if self.marker_visible and self.marker_x is not None:
            current_dist_to_dock = math.sqrt(self.marker_x**2 + self.marker_z**2)

        # Obstacle check — direct lidar, NOT checkObstacles() (which always returns False)
        # Disabled within 50cm: at that range the wall IS the dock
        ignore_obstacles_dist = 0.50
        if current_dist_to_dock > ignore_obstacles_dist:
            front_dist = (np.nanmin(self.front)
                          if len(self.front) > 0 and not np.isnan(self.front).all()
                          else float('inf'))
            if front_dist <= STOP_DISTANCE:
                self.get_logger().warn(f'Obstacle at {front_dist:.2f}m blocking dock path! → RECOVERY')
                
                self.stopbot()
                self.state = 'RECOVERY'
                return

        if self.marker_visible and self.marker_x is not None:
            x, z  = self.marker_x, self.marker_z
            rho   = math.sqrt(x**2 + z**2) - self.target_dist
            alpha = math.atan2(x, z)
            beta  = self.desired_final_heading - alpha

            # Pre-rotation phase: if angle > ~35°, rotate in place first.
            # Fixes large-angle (70°+) approach resulting in a slanted dock.
            pre_rotate_threshold = 0.60  # radians (~35°)
            if abs(alpha) > pre_rotate_threshold:
                cmd = Twist()
                cmd.linear.x  = 0.0
                cmd.angular.z = max(min(-alpha * self.k_alpha, self.docking_max_w), -self.docking_max_w)
                self.publisher_.publish(cmd)
                self.get_logger().info(
                    f'[DOCK] Pre-rotating: α={math.degrees(alpha):+.1f}° ω={cmd.angular.z:+.3f}',
                    throttle_duration_sec=0.2)
                return

            v = self.k_rho * rho
            w = -(self.k_alpha * alpha + self.k_beta * beta)
            heading_factor = math.cos(alpha)**2 # Slow down as we approach head-on to prevent overshooting
            v *= max(0.3,heading_factor) # Don't slow down too much to avoid getting stuck when very close
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
            cmd.linear.x  = v
            cmd.angular.z = w
            self.publisher_.publish(cmd)
            return

        if elapsed > self.fov_loss_timeout:
            self.stopbot()
            self.get_logger().warn('Marker lost during approach → PLANNING')
            self.state = 'PLANNING'
            return

        if self.last_known_alpha is not None:
            cmd = Twist()
            if self.last_known_rho is not None and self.last_known_rho < 0.1:
                cmd.linear.x = 0.0
            else:
                cmd.linear.x = 0.04
            cmd.angular.z = -math.copysign(self.recovery_w, self.last_known_alpha)
            self.publisher_.publish(cmd)
    """
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
            front_dist = (np.nanmin(self.front)
                          if len(self.front) > 0 and not np.isnan(self.front).all()
                          else float('inf'))
            if front_dist <= STOP_DISTANCE:
                self.get_logger().warn(f'Obstacle at {front_dist:.2f}m blocking dock path! → RECOVERY')
               
        else:
            self.stopbot()
            self.state = 'STATION'
            self.get_logger().info("✓ Final alignment complete.")


    def docking_arc_logic(self):
        """Phase 1: Drive to the standoff point (40cm in front of marker) using RPP."""
        try:
            rx, ry, ryaw = self.get_orientation()
        except:
            return

        # If standoff not yet computed (TF failed on first frame), fall back to camera-frame calc
        if self.dock_ghost_x is None:
            if self.marker_x is None or self.marker_z is None:
                self.get_logger().warn('No standoff point and no camera data. Waiting...')
                return
            dock_global_angle = ryaw + self.marker_yaw
            marker_map_x = rx + (self.marker_z * math.cos(ryaw) - self.marker_x * math.sin(ryaw))
            marker_map_y = ry + (self.marker_z * math.sin(ryaw) + self.marker_x * math.cos(ryaw))
            self.dock_ghost_x = marker_map_x + self.rpp_controller.standoff_distance * math.cos(dock_global_angle)
            self.dock_ghost_y = marker_map_y + self.rpp_controller.standoff_distance * math.sin(dock_global_angle)
            self.dock_ghost_yaw = (dock_global_angle + math.pi) % (2 * math.pi) - math.pi
            self.get_logger().info(f'Ghost Goal (camera fallback): ({self.dock_ghost_x:.2f}, {self.dock_ghost_y:.2f})')

        dist = math.sqrt((self.dock_ghost_x - rx)**2 + (self.dock_ghost_y - ry)**2)
        self.get_logger().info(f'[DOCKING_ARC] dist to standoff: {dist:.2f}m', throttle_duration_sec=0.5)

        if dist < 0.12:
            self.stopbot()
            self.get_logger().info('Standoff reached. Pivoting to face dock...')
            self.state = 'DOCKING_PIVOT'
            return

        tmp_path = [MapNode(self.dock_ghost_x, self.dock_ghost_y)]
        cmd = self.rpp_controller.command(rx, ry, ryaw, tmp_path)
        if cmd:
            self.publisher_.publish(cmd)
        else:
            # Large angle — turn in place toward standoff
            target_angle = math.atan2(self.dock_ghost_y - ry, self.dock_ghost_x - rx)
            self.turn_in_place(target_angle, ryaw)

    def docking_pivot_logic(self):
        """Phase 2: Pivot until facing the marker perfectly."""
        try:
            _, _, ryaw = self.get_orientation()
        except: return

        # We want our heading to match the dock_ghost_yaw (facing the dock)
        angle_err = self.dock_ghost_yaw - ryaw
        angle_err = (angle_err + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_err) < 0.05:
            self.stopbot()
            self.get_logger().info("Aligned. Commencing Final Approach.")
            self.state = 'DOCKING_APPROACH'
            return

        twist = Twist()
        twist.angular.z = max(min(angle_err * 0.8, 0.4), -0.4)
        self.publisher_.publish(twist)

    def docking_approach_logic(self):
        """Phase 3: Visual Servoing 'Rail' approach."""
        if not self.marker_visible:
            elapsed = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9
            if elapsed > 1.5:
                # Lost marker too long — stop and search again
                self.stopbot()
                self.get_logger().warn('Marker lost during approach. Re-searching...')
                self.search_yaw_accumulated = 0.0
                try:
                    _, _, yaw = self.get_orientation()
                    self.search_last_yaw = yaw
                except: pass
                self.state = 'SEARCHING'
                return
            # Very close — creep forward blind
            cmd = Twist()
            cmd.linear.x = 0.05
            self.publisher_.publish(cmd)
            return

        x = self.marker_x
        z = self.marker_z
        if x is None or z is None:
            return
        dist = math.sqrt(x**2 + z**2)

        k_lat = 1.2
        k_yaw = 0.6
        trim = (x * k_lat) + (self.marker_yaw * k_yaw)

        if dist <= 0.28:
            self.stopbot()
            self.dock_ghost_x = None
            self.dock_ghost_y = None
            self.dock_ghost_yaw = None
            self.get_logger().info('Docked! Transitioning to STATION.')
            self.state = 'STATION'
            return

        cmd = Twist()
        cmd.linear.x = 0.07
        cmd.angular.z = max(min(-trim, 0.3), -0.3)
        self.publisher_.publish(cmd)
    
    def controller(self):
        try:
            curr_x, curr_y, curr_yaw = self.get_orientation()
        except: return

        self.checkObstacles()
        twist = Twist()

        if self.state == 'DOCKING_ARC':
            self.docking_arc_logic()

        elif self.state == 'DOCKING_PIVOT':
            self.docking_pivot_logic()

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

        elif self.state == 'RECOVERY':
            self.front_fov = 120 #temporarily widen front fov to be more cautious during recovery
            self.recoverySequence()

        elif self.state == 'ESCAPING':
            now = self.get_clock().now()
            elapsed_time = (now - self.escape_start_time).nanoseconds / 1e9
            if not self.checkObstacles():
                if elapsed_time < self.escape_duration:
                    twist = Twist()
                    twist.linear.x = float(self.escape_speed)
                    twist.angular.z = 0.0
                    self.publisher_.publish(twist)
                    if len(self.front) > 0 and np.nanmin(self.front) <= SIDE_THRESHOLD:
                        self.stopbot()
                        self.state = 'RECOVERY'
                else:
                    self.stopbot()
                    if self.pre_recovery_state in ['DOCKING_APPROACH', 'DOCKING_FINAL_ALIGN']:
                        self.get_logger().info('Was docking before obstacle. Spinning to re-acquire marker...')
                        self.state = 'SEARCHING'
                        self.search_yaw_accumulated = 0.0
                        try:
                            _, _, cur_yaw = self.get_orientation()
                            self.search_last_yaw = cur_yaw
                        except:
                            pass
                    else:
                        self.get_logger().info('Resuming standard navigation.')
                        self.state = 'PLANNING'
                    self.front_fov = 90
                    self.pre_recovery_state = None
            else: 
                self.stopbot()
                self.state = 'RECOVERY'

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
                if self.current_docking_id == 0:
                    self.gpio_pub.publish(String(data='A'))
                    self.get_logger().info("Command Sent for Station A. Waiting for PI response...")
                    self.commandSent = True
                    self.state = 'WAITING_FOR_PI'
                if self.current_docking_id == 1:
                    self.gpio_pub.publish(String(data='B'))
                    self.get_logger().info("Command Sent for Station B. Waiting for PI response...")
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
                if self.STATION_A_COMPLETED and self.STATION_B_COMPLETED:
                    self.get_logger().info('MISSION COMPLETE!.')
                    self.state = 'MISSION_COMPLETE'
                else:    
                    self.get_logger().info('Station complete! Resuming exploration...')
                    self.undock_start_time = self.get_clock().now()
                    self.state = 'UNDOCKING'

        elif self.state == 'UNDOCKING':
            now = self.get_clock().now()
            elapsed_time = (now - self.undock_start_time).nanoseconds / 1e9
            back_dist = np.nanmin(self.back) if len(self.back) > 0 and not np.isnan(self.back).all() else float('inf')
            if back_dist <= 0.25:
                self.get_logger().warn(f'Obstacle detected {back_dist:.2f}m behind! Aborting reverse.')
                self.stopbot()
                self.undock_start_time = None
                self.state = 'PLANNING'
                return
            if elapsed_time < self.undock_duration:
                twist = Twist()
                twist.linear.x = float(self.undock_speed)
                twist.angular.z = 0.0
                self.publisher_.publish(twist)
            else:
                self.stopbot()
                self.get_logger().info('Undocking complete. Resuming frontier exploration...')
                self.undock_start_time = None
                self.state = 'PLANNING'

        elif self.state == 'MISSION_COMPLETE':
            self.stopbot()


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