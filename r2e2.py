import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import numpy as np

# Define Constants
stop_dist = 1.0
front_angle = 30
front_angle_range = range(-front_angle, front_angle + 1,1)  # -30 to 30 degrees

class LengthDetect(Node):
    def __init__(self):
        super().__init__('length_detect')
        
        # Subscribe to LIDAR Topic
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.callback,
            qos_profile_sensor_data)
        self.subscription #prevent unused varibale warning

        # Publisher for GPIO commands
        self.publisher_ = self.create_publisher(String, 'gpio_commands', 10) 
        self.get_logger().info('LengthDetect node initialized')
    
    def callback(self, msg):
        
        # Storage for LIDAR data
        self.laser_range = np.array(msg.ranges) #create an array to hold LIDAR ranges

        # Replace 0.0 (out of range) with NaN 
        self.laser_range[self.laser_range == 0] = np.nan
        
        if np.isnan(self.laser_range).all(): #handles cases whereby all values are OUT OF RANGE (all nan) or EMPTY ARRAY
            self.get_logger().info('No LIDAR data received')
            pub_msg = String()
            pub_msg.data = "0"
            self.publisher_.publish(pub_msg)
            return
        
        else:
            lri = (self.laser_range[front_angle_range]<=float(stop_dist)).nonzero() #finds the distance of the closest object
            self.get_logger().info('Distances: %s' % str(lri))
            pub_msg = String() #initialise a message to be sent to RPI

            if len(lri[0]) > 0: #if there are obstacles in front

                self.get_logger().info('Obstacle detected at distance: %.2f meters at angle %d degrees' % (self.laser_range[lri[0][0]], np.nanargmin(self.laser_range[front_angle_range]) - front_angle)) #log the distance and angle of the closest obstacle in front
                pub_msg.data = "1"
                self.publisher_.publish(pub_msg)
            else:
                self.get_logger().info('No obstacle detected infront within %.2f meters' % stop_dist)
                pub_msg.data = "0"
                self.publisher_.publish(pub_msg)

def main(args=None):
    length_detect = None #initialize to None for exception handling

    try:
        rclpy.init(args=args)
        length_detect = LengthDetect()
        rclpy.spin(length_detect)

    except KeyboardInterrupt: #handle Ctrl+C exception to allow clean shutdown
        print("Shutting down LengthDetect node.")

    except Exception as e: #handle other exceptions
        print(f"An error occurred: {e}")

    finally:
        #graceful shutdown sequence
        if length_detect: #checks if the node was created successfully
            length_detect.destroy_node()
        
        # Only shutdown if rclpy is actually active
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()