"""
This node locates Aruco AR markers in images and publishes their ids and poses.
It also displays an OpenCV window showing the camera feed and drawn markers.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_poses (geometry_msgs.msg.PoseArray)
    /aruco_markers (ros2_aruco_interfaces.msg.ArucoMarkers)
    
Parameters:
    marker_size - size of the markers in meters (default .0625)
    aruco_dictionary_id - dictionary that was used to generate markers
                          (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/image_raw)
    camera_info_topic - camera info topic to subscribe to
                         (default /camera/camera_info)

Author: Nathan Sprague
Version: 10/26/2020
Note: Slightly Modified to work with OpenCV 4.10.0 and added visualisation
"""

import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class USB1ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("usb1_aruco_node")

        # Declare and read parameters
        self.declare_parameter(
            name="marker_size",
            value=0.0625,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of the markers in meters.",
            ),
        )

        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_250",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )

        self.declare_parameter(
            name="image_topic",
            value="/hdcam0/image_raw/compressed",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_info_topic",
            value="/hdcam0/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )

        self.marker_size = (
            self.get_parameter("marker_size").get_parameter_value().double_value
        )
        self.get_logger().info(f"Marker size: {self.marker_size}")

        dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        )
        self.get_logger().info(f"Marker type: {dictionary_id_name}")

        image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image topic: {image_topic}")

        info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image info topic: {info_topic}")

        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )

        # Around line 118 in aruco_node.py
        self.create_subscription(CompressedImage, image_topic, self.image_callback, qos_profile_sensor_data)

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(
                "bad aruco_dictionary_id: {}".format(dictionary_id_name)
            )
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error("valid options: {}".format(options))

        # Set up subscriptions
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data
        )

        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, "usbcam1_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "usbcam1_markers", 10)

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        # OpenCV 4.10.0 API Setup
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
        self.bridge = CvBridge()

    def estimate_pose_single_marker(self, corners, marker_size, mtx, distortion):
        """
        Calculates the pose of the marker via Perspective-n-Point.
        """
        marker_points = np.array([
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0]
        ], dtype=np.float32)

        rvecs = []
        tvecs = []
        for c in corners:
            _, rvec, tvec = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(rvec)
            tvecs.append(tvec)
            
        return rvecs, tvecs

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        # Use compressed_imgmsg_to_cv2 to decode the stream directly to BGR
        cv_image_display = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        
        # Create a grayscale copy for the ArUco detector
        cv_image = cv2.cvtColor(cv_image_display, cv2.COLOR_BGR2GRAY)

        # ... (the rest of your detection and drawing code remains the same)

        markers = ArucoMarkers()
        pose_array = PoseArray()
        
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        # Detect markers
        corners, marker_ids, rejected = self.detector.detectMarkers(cv_image)
        
        if marker_ids is not None:
            # Draw the square borders and IDs
            cv2.aruco.drawDetectedMarkers(cv_image_display, corners, marker_ids)
            
            rvecs, tvecs = self.estimate_pose_single_marker(
                corners, self.marker_size, self.intrinsic_mat, self.distortion
            )
            
            for i, marker_id in enumerate(marker_ids):
                # Draw the 3D axes (Red=X, Green=Y, Blue=Z)
                cv2.drawFrameAxes(
                    cv_image_display, 
                    self.intrinsic_mat, 
                    self.distortion, 
                    np.array(rvecs[i]), 
                    np.array(tvecs[i]), 
                    self.marker_size / 2
                )

                pose = Pose()
                pose.position.x = float(tvecs[i][0][0])
                pose.position.y = float(tvecs[i][1][0])
                pose.position.z = float(tvecs[i][2][0])

                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i]))[0]
                quat = tf_transformations.quaternion_from_matrix(rot_matrix)

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id[0])

            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        # --- Show the live OpenCV window ---
        cv2.imshow("USB CAM 1", cv_image_display)
        cv2.waitKey(1)  # 1ms delay is required for OpenCV to update the window


def main():
    rclpy.init()
    usb1_aruco_node = USB1ArucoNode()
    rclpy.spin(usb1_aruco_node)

    usb1_aruco_node.destroy_node()
    cv2.destroyAllWindows() # Clean up windows on exit
    rclpy.shutdown()


if __name__ == "__main__":
    main()