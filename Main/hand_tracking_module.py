#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyrealsense2 as rs2
import time

class Hand_Tracking():
    def __init__(self, rate = 15):
        #Initialize ROS Node
        rospy.init_node("Hand_tracking_node", anonymous=True)

        self.color_image = None
        self.depth_image = None
        self.intrinsics = None
        self.palm_center = None
        self.palm_distance = None
        self.width = None
        self.height = None
        self.results = None
        self.gesture = None
        self.point_3d = None
        self.brake = True

        """
            Benchmark Variables
        """
        self.processing_time = None
        self.running_sum = 0
        self.iterations = 1 
        self.avg = None

        self.Rco = np.array([
            [0,  0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])

        # self.mp_hands = mp.solutions.hands
        # self.mp_drawing = mp.solutions.drawing_utils

        # self.hands = self.mp_hands.Hands(
        #     static_image_mode=False,
        #     max_num_hands=1,
        #     min_detection_confidence=0.5,
        #     min_tracking_confidence=0.5,
        #     model_complexity=0  # Use simplest model for speed
        # )

        # self.custom_landmark_style = mp.solutions.drawing_utils.DrawingSpec(
        #     color=(0, 0, 255),  
        #     thickness=2,
        #     circle_radius=2
        # )

        # self.custom_connection_style = mp.solutions.drawing_utils.DrawingSpec(
        #     color=(255, 255, 255),  
        #     thickness=2
        # )

        # Set up gesture recognition
        base_options = python.BaseOptions(model_asset_path='/home/vasilis/cat_ws/src/robot_pkg/src/gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=1,  # Detect up to 1 hand  
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)


        # Ros Topics Subscribers
        rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.color_callback)
        rospy.Subscriber('/camera/depth/image_raw/compressed', CompressedImage, self.depth_callback)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.depth_info_callback)

        # Ros Topics Publishers
        self.point_pub = rospy.Publisher('/object_coordinates', PointStamped, queue_size=10)
        self.brake_pub = rospy.Publisher('/brake', Bool, queue_size=10)
        self.pixel_pub = rospy.Publisher('/2d_point', PointStamped, queue_size=10)

        self.rate = rospy.Rate(rate)

    def depth_info_callback(self, msg):
        self.intrinsics = rs2.intrinsics()

        self.intrinsics.width = msg.width
        self.intrinsics.height = msg.height
        self.intrinsics.ppx = msg.K[2]  # Principal point x
        self.intrinsics.ppy = msg.K[5]  # Principal point y
        self.intrinsics.fx = msg.K[0]   # Focal length x
        self.intrinsics.fy = msg.K[4]   # Focal length y

        # Set distortion model and coefficients
        if msg.distortion_model == "plumb_bob":
            self.intrinsics.model = rs2.distortion.brown_conrady
        else:
            self.intrinsics.model = rs2.distortion.none
        
        self.intrinsics.coeffs = [i for i in msg.D[:5]]

    def depth_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    
    def color_callback(self, msg):
        start_time = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        self.getImgDimentions()

        """
        1. Process the Image through the neural network
        2. Calculate the center of the palm if it exists
        3. Get the distance of that point 
        4. Convert to 3D point
        """
        self.mediapipe_process()
        self.findCenterOfPalm()
        end_time = time.time()
        self.getDistance()
        self.point_3d = self.pixel_to_point()
        self.plotImage()

        self.processing_time = (end_time - start_time) * 1000
        self.running_sum += self.processing_time 
        self.avg = self.running_sum / self.iterations

        self.iterations += 1

    def getRate(self):
        return self.rate
    
    def getImgDimentions(self):
        self.height, self.width, _ = self.color_image.shape
        return self.width, self.height

    ## V1 ##
    # def mediapipe_process(self):
    #     scale_factor = 0.5 
    #     scaled_img = cv2.resize(self.color_image, (0, 0), fx=scale_factor, fy=scale_factor)
    
    #     # Process the scaled image with MediaPipe
    #     scaled_img_rgb = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
    #     self.results = self.hands.process(scaled_img_rgb)

    def mediapipe_process(self):
        # Process the scaled image with MediaPipe
        img_rgb = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.flip(img_rgb, 0)
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=img_rgb)
        self.results = self.recognizer.recognize(mp_image)
    
    def drawHandLandmarks(self):
        """
            Don't use this method.
            Need's changes
        """
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw hand landmarks

                self.mp_drawing.draw_landmarks(
                    self.color_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.custom_landmark_style,
                    connection_drawing_spec=self.custom_connection_style
                )

    ## V1 ##            
    # def findCenterOfPalm(self):
    #     if self.results.multi_hand_landmarks:
    #         for hand_landmarks in self.results.multi_hand_landmarks:
    #             # Select key palm landmarks
    #             palm_landmarks = [
    #                 hand_landmarks.landmark[0],   # WRIST
    #                 hand_landmarks.landmark[1],   # THUMB_CMC
    #                 hand_landmarks.landmark[5],   # INDEX_FINGER_MCP
    #                 hand_landmarks.landmark[9],   # MIDDLE_FINGER_MCP
    #                 hand_landmarks.landmark[13],  # RING_FINGER_MCP
    #                 hand_landmarks.landmark[17]   # PINKY_MCP
    #             ]

    #             # Compute the average position of the selected landmarks
    #             cx = int(sum(lm.x for lm in palm_landmarks) / len(palm_landmarks) * self.width)
    #             cy = int(sum(lm.y for lm in palm_landmarks) / len(palm_landmarks) * self.height)

    #             self.palm_center = (cx, cy)

    #             # Draw the center of the palm
    #             cv2.circle(self.color_image, (cx, cy), 5, (255, 0, 0), -1)
    #     else:
    #         self.palm_center = None

    def findCenterOfPalm(self):
        if self.results.gestures:
            top_gesture = self.results.gestures[0][0]
            self.gesture = top_gesture.category_name
            cv2.putText(self.color_image,
                        f'{self.gesture}',
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if self.gesture == 'Closed_Fist':
                self.brake = True
            elif self.gesture == 'Pointing_Up':
                self.brake = False

        if self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                palm_indices = [0, 1, 5, 9, 13, 17]  # WRIST, THUMB_CMC, etc.

                palm_points = []
                for index in palm_indices:
                    if index < len(hand_landmarks):  # Ensure index is within bounds
                        lm = hand_landmarks[index]
                        palm_points.append((lm.x * self.width, lm.y * self.height))

                if palm_points:  # Compute center only if landmarks exist
                    cx = int(sum(p[0] for p in palm_points) / len(palm_points))
                    cy = int(sum(p[1] for p in palm_points) / len(palm_points))

                self.palm_center = (cx, self.height - 1 - cy)

                # Draw the center of the palm
                cv2.circle(self.color_image, self.palm_center, 5, (255, 0, 0), -1)
        else:
            self.palm_center = None

    def getDistance(self):
        if self.depth_image is not None and self.palm_center is not None:
            palm_center_x = self.palm_center[0]
            palm_center_y = self.palm_center[1]

            distance = self.depth_image[palm_center_y, palm_center_x]
            self.palm_distance = distance

            cv2.putText(self.color_image, f'{distance} mm', (10,20),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)

    def pixel_to_point(self, optical=False):
        if  self.palm_center is not None:
            point = rs2.rs2_deproject_pixel_to_point(self.intrinsics, 
                                                    self.palm_center,
                                                    self.palm_distance / 1000)
            point = np.array(point)
            # If optical = True return point from optical frame
            if not optical:
                point = self.Rco @ point + np.array([0, 0.044, 0.35]) 

            return point

        return None

    def pointPublisher(self, point):
        brake_msg = Bool()
        brake_msg.data = self.brake

        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"

        if point is not None:
            msg.point.x, msg.point.y, msg.point.z = point[0], point[1], point[2]
        else:
            msg.point.x, msg.point.y, msg.point.z = 0, 0, -1

        pixel_msg = PointStamped()
        pixel_msg.header.stamp = rospy.Time.now()
        if self.palm_center == None:
            pixel_msg.point.x, pixel_msg.point.y, pixel_msg.point.z = 0, 0, 0
        else:
            pixel_msg.point.x, pixel_msg.point.y, pixel_msg.point.z = self.palm_center[0], self.palm_center[1], 0


        self.pixel_pub.publish(pixel_msg)
        self.brake_pub.publish(brake_msg)
        self.point_pub.publish(msg) 

    def plotImage(self):
        cv2.imshow("D145 Camera", self.color_image)
        cv2.waitKey(1)       
        

if __name__ == '__main__':
    hd = Hand_Tracking()

    rate = hd.getRate()

    try:
        while not rospy.is_shutdown():
            hd.pointPublisher(hd.point_3d)

            # rospy.loginfo(f'Processing time: {hd.processing_time}ms')
            # rospy.loginfo(f'Average processing time: {hd.avg}ms')

            rate.sleep()

    except rospy.ROSInterruptException:
        print('something went wrong')

        cv2.destroyAllWindows()