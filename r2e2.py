import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import numpy as np

# Define Constants
DETECT_DIST = 1.0

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
        laser_range = np.array(msg.ranges) #create an array to hold LIDAR ranges

        # Replace 0.0 (out of range) with NaN 
        laser_range[laser_range == 0] = np.nan
        
        if np.isnan(laser_range).all(): #handles cases whereby all values are OUT OF RANGE (all nan) or EMPTY ARRAY
            self.get_logger().info('No LIDAR data received')
            pub_msg = String()
            pub_msg.data = "0"
            self.publisher_.publish(pub_msg)
            return
        
        else:
            lr2i = np.nanargmin(laser_range) #finds and saves the index of the closest object
            pub_msg = String() #initialise a message to be sent to RPI

            if laser_range[lr2i] <= DETECT_DIST:    
                self.get_logger().info('Obstacle detected at distance: %.2f meters' % laser_range[lr2i])
                pub_msg.data = "1"
                self.publisher_.publish(pub_msg)
            else:
                self.get_logger().info('No obstacle detected within %.2f meters' % DETECT_DIST)
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