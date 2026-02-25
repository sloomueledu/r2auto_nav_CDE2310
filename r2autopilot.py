import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
import numpy as np
import math
import random
# import cmath
# import time
import tf2_ros
from tf2_ros import TransformException, LookupException, ConnectivityException, ExtrapolationException
# import cv2
# from cv_bridge import CvBridge

#CONSTANTS
STOP_DISTANCE = 0.3
SIDE_THRESHOLD = 0.15
GOAL_THRESHOLD = 0.3
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
        self.lookaheaddist = 0.2
        self.max_speed = 0.22
        self.min_speed = 0.05
        self.max_angular_v = 0.2
        self.safety_factor = 3.0
        self.rotate_threshold = 0.5 #anything >28 degrees
    
    def findpoint(self, cur_x, cur_y, path):
    # Default to the very last point if we can't find one in range
        target = path[-1]
        
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

        """
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
        self.turning_timeout = 10.0 # seconds, this is to prevent the robot from getting stuck in a turning state for too long

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
        #replace out of range readings (0.0) with nan
        self.laser_range[self.laser_range==0] = np.nan
        np.savetxt(SCANFILE, self.laser_range)

        # Divide the laser_range array into 4 distinct parts:
        # FRONT: -45 to 45
        # LEFT: 45 to 135
        # RIGHT: -135 to -45
        # BACK: 135 to -135
        if len(self.laser_range) == 360:
            front_right = self.laser_range[315:]
            front_left = self.laser_range[0:45]
            self.front = np.concatenate((front_right, front_left))
            self.left = self.laser_range[45:135]
            self.back = self.laser_range[135:225]
            self.right = self.laser_range[225:315]

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
            timeout = rclpy.time.Duration(seconds=2.0) #implement a timeout to ensure that the robot doesn't get stuck here
            
            # these are placeholders for the pooled goal coords
            pgoal_x = -1
            pgoal_y = -1

            # get the actual bot location in the original map
            if self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time(), timeout):
                cur_x, cur_y, _ = self.get_orientation()
                obotloc_x = (cur_x - self.origin.x)/self.res
                obotloc_y = (cur_y - self.origin.y)/self.res
                
                pbotloc_x = min(int(obotloc_x) // 3, occ_pooled_grid.shape[1] - 1)
                pbotloc_y = min(int(obotloc_y) // 3, occ_pooled_grid.shape[0] - 1)
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
        
        # ... (End of the while len(frontier) > 0: loop) ...
        
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
            
            # to adjust if necessary
            maxshift = 1 # this is the maximum a node will be shifted by
            wallinfdist  = 2 # how far walls will influence the path

            # we will now determine how much we need to shift each point in the path by and the shift direction
            for dy in range(-wallinfdist, wallinfdist + 1):
                for dx in range(-wallinfdist, wallinfdist + 1):
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

                            # scale the shift 
                            # inverse relationship: closer you are to the wall, more you get pusahed away
                            magnitude = maxshift * (1.0 / shiftdist)

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
        # If there is an obstacle detected, return True, else, return false

        if len(self.laser_range) == 0 or np.isnan(self.laser_range).all():
            self.get_logger().info(f'No Valid LiDAR data')
            return False

        # Front Check
        if not np.isnan(self.front).all() and np.nanmin(self.front) <= STOP_DISTANCE:
            self.stopbot()
            self.get_logger().info(f'Obstacle detected infront of the robot!')
            # check if can back up
            if not np.isnan(self.back).all() and np.nanmin(self.back) > STOP_DISTANCE:
                self.stopbot()
                self.get_logger().info(f'Attempting to back up')
                self.state = 'RECOVERY'
                return True
            else:
                self.get_logger().info(f'Stuck. Replanning new route...')
                self.state = 'PLANNING'
                return True

        if not np.isnan(self.left).all() and np.nanmin(self.left) <= SIDE_THRESHOLD:
            self.stopbot()
            self.get_logger().info(f'Obstacle detected to the left of the robot! Replanning new route...')
            self.state = 'PLANNING'
            return True

        if not np.isnan(self.back).all() and np.nanmin(self.back) <= SIDE_THRESHOLD:
            self.stopbot()
            self.get_logger().info(f'Obstacle detected behind the robot! Replanning New Route...')
            self.state = 'PLANNING'
            return True
        
        if not np.isnan(self.right).all() and np.nanmin(self.right) <= SIDE_THRESHOLD:
            self.stopbot()
            self.get_logger().info(f'Obstacle detected to the right of the robot! Replanning new route...')
            self.state = 'PLANNING'
            return True
        
        return False
        
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
                self.state = 'PLANNING'
                return
        
        # Step 2: Check for obstacles
        obstacles = self.checkObstacles()
        if obstacles:
            self.boink += 1
            return
        
        # Step 3: Get the RPP Command:

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
            # pretty much the same code as finding angle and normalising
            self.get_logger().info('Large Angle Detected. Rotating on the spot to align with path.')
            target_angle = math.atan2(target.y - cur_y, target.x - cur_x)
            anglediff = target_angle - cur_yaw
            anglediff = (anglediff + math.pi) % (2 * math.pi) - math.pi

            twist = Twist()
            twist.linear.x = 0.0
            # Define a proportional gain (Kp)
            # 1.0 to 2.0 is usually a good starting point for TurtleBots
            kp_yaw = 0.6
            
            # Calculate a proportional speed
            turning_speed = anglediff * kp_yaw
            
            # Cap the speed so it doesn't exceed your max limit
            twist.angular.z = max(-self.rpp_controller.max_angular_v, min(self.rpp_controller.max_angular_v, turning_speed))
            self.publisher_.publish(twist)
            return
        self.publisher_.publish(cmd_vel)
        
    def recovery(self):
            if self.boink > 3:
                self.get_logger().warn('Path or Goal bad. Replanning new route...')
                self.stopbot()
                self.state = 'PLANNING'
                self.goal = None
                self.boink = 0
                return
            """
            Reactive state to back away from obstacles safely.
            """
            twist = Twist()
            
            # 1. Safely get the minimum distances (default to infinity if array is empty or all NaNs)
            front_dist = np.nanmin(self.front) if len(self.front) > 0 and not np.isnan(self.front).all() else float('inf')
            left_dist  = np.nanmin(self.left)  if len(self.left) > 0  and not np.isnan(self.left).all()  else float('inf')
            right_dist = np.nanmin(self.right) if len(self.right) > 0 and not np.isnan(self.right).all() else float('inf')

            # 2. Have we backed up enough? (Add a 0.15m safety buffer)
            if front_dist > (STOP_DISTANCE + 0.15):
                self.get_logger().info('Clear space found. Resuming PLANNING.')
                self.stopbot()
                self.state = 'PLANNING'
                return

            # 3. If we are still too close, back up slowly
            twist.linear.x = -0.08  # Negative speed means reverse
            
            # 4. Turn away from the side that has the closest obstacle
            if left_dist < right_dist:
                twist.angular.z = 0.5  # Obstacle on left, swing rear to the left
            else:
                twist.angular.z = -0.5   # Obstacle on right, swing rear to the right

            self.publisher_.publish(twist)

    # This function is to control the state of the robot
    def controller(self):

        """
        In here, we will focus on 3 Main State:
        1. PLANNING
        2. DRIVING
        3. RECOVERY
        """

        if self.state == 'PLANNING':
            if self.goal is None:
                self.path = self.planroute(goal=None)
            else:
                self.path = self.planroute(goal=self.goal)
            
            if self.path and len(self.path) > 0: # This is a safety check to make sure a valid path has been found
                self.goal = self.path[-1] # If a goal point has already been established, This WILL NOT override it!
                self.state = "DRIVING"
                self.get_logger().info('Planning Complete! Switching to DRIVING state')
            else:
                self.get_logger().warn('No valid path found. Retrying...')
                self.stopbot()

        elif self.state == 'DRIVING':
            self.mover()
        
        elif self.state == 'RECOVERY':
            self.recovery()          
    
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