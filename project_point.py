#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage
import cv2
from tf.transformations import quaternion_matrix

class PointProjector:
    def __init__(self):
        rospy.init_node('point_projection')

        self.K = None
        self.camera_info_received = False
        self.point_received = False
        self.robot_pose = None  # Pose from /odometry/filtered
        self.in_front = False

        # rospy.Subscriber('/camera/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        rospy.Subscriber('/object_coordinates', PointStamped, self.point_callback)

        self.img_pub = rospy.Publisher('/point_img/compressed', CompressedImage, queue_size=10)
        self.pixel_point = rospy.Publisher('/2d_point', PointStamped, queue_size=10)

        np.set_printoptions(precision=3, suppress=True)

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.K = np.array(msg.K).reshape(3, 3)
            self.width = msg.width
            self.height = msg.height
            self.camera_info_received = True
            rospy.loginfo("Camera intrinsics received.")
            rospy.loginfo(f'Camera Width: {self.width}, Camera Height {self.height}')

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def point_callback(self, point_msg):
        if not self.camera_info_received or self.robot_pose is None:
            rospy.logwarn("Waiting for camera intrinsics and odometry...")
            return

        self.point_received = True

        # Extract robot pose
        position = self.robot_pose.position
        orientation = self.robot_pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]

        # Get rotation matrix from world to camera
        self.Rc = quaternion_matrix(quat)[:3,:3]

        # Get robot position
        self.p = np.array([
            position.x,
            position.y,
            position.z
        ])

        # Get the 3D coordinates of the point 
        self.point_pos = np.array([
            point_msg.point.x,
            point_msg.point.y,
            point_msg.point.z,
            1.0
        ]) 

    def projection(self, point):
        projected_point = self.K @ point 

        return projected_point

    def make_img(self, point):
        x, y = int(point[0]), int(point[1])
        blank_image = 255 * np.ones((self.height, self.width), dtype=np.uint8)

        if 0 <= x and x <= self.width and 0 <= y and y <= self.height and self.in_front:
            cv2.circle(blank_image, (x, y), 10, (0, 0, 0), -1)
        
        success, encoded_image = cv2.imencode('.jpg', blank_image)
        if not success:
            rospy.logerr("Failed to encode image")
            return

        # Create CompressedImage message
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = encoded_image.tobytes()

        self.img_pub.publish(msg)

    def run(self):
        rospy.sleep(0.5) # sleep to ensure that all the callbacks ran
        self.rate = rospy.Rate(12)
        R = np.array([
            [0,  0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])

        try:
            while not rospy.is_shutdown():
                if self.point_received:
                    # Set the transformation matrix from world to camera
                    Gc = np.eye(4)
                    Gc[:3, :3] = self.Rc
                    Gc[:3, 3] = self.p

                    # 3D point from the perspective of the camera
                    point_c = np.linalg.inv(Gc)[:3, :] @ self.point_pos
                    optical_point = R.T @ point_c # Point from camera optical frame

                    # Check if the point is in front of the camera 
                    if optical_point[2] >= 0:
                        self.in_front = True
                    else:
                        self.in_front = False 

                    normalized_point = np.array([optical_point[0] / optical_point[2], optical_point[1] / optical_point[2], 1])
                    # print(optical_point)

                    # Project 3D point to a 2D pixel point
                    pixel = self.projection(normalized_point)[:2]

                    pixel_msg = PointStamped() 
                    pixel_msg.header.stamp = rospy.Time.now()
                    pixel_msg.point.x = pixel[0]                # Pixel x
                    pixel_msg.point.y = pixel[1]                # Pixel y
                    pixel_msg.point.z = optical_point[2]        # Depth z optical point 
                    
                    self.pixel_point.publish(pixel_msg)
                    
                    # Make and publish the image with the point
                    self.make_img(pixel)

                self.rate.sleep() 

        except rospy.ROSInterruptException:
            print('\n[ FINISHED ]') 

if __name__ == '__main__':
    try:
        point = PointProjector()
        point.run()
    except rospy.ROSInterruptException:
        pass
