#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
import numpy as np
import cv2

def fake_camera_node():
    rospy.init_node('fake_compressed_camera_publisher')

    rgb_pub = rospy.Publisher('/camera/rgb/image_raw/compressed', CompressedImage, queue_size=10)
    depth_pub = rospy.Publisher('/camera/depth/image_raw/compressed', CompressedImage, queue_size=10)
    camera_info_pub = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=10)

    rate = rospy.Rate(15)  # 15 Hz

    # Camera info setup
    camera_info = CameraInfo()
    camera_info.width = 640
    camera_info.height = 480
    camera_info.K = [525.0, 0.0, 319.5,
                     0.0, 525.0, 239.5,
                     0.0, 0.0, 1.0]
    camera_info.P = [525.0, 0.0, 319.5, 0.0,
                     0.0, 525.0, 239.5, 0.0,
                     0.0, 0.0, 1.0, 0.0]
    camera_info.distortion_model = "plumb_bob"

    while not rospy.is_shutdown():
        timestamp = rospy.Time.now()

        # --- Create fake RGB image (random noise) ---
        rgb_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # Encode to JPEG
        success_rgb, jpeg_data = cv2.imencode('.jpg', rgb_array)
        if not success_rgb:
            rospy.logerr("Failed to encode RGB image")
            continue

        rgb_msg = CompressedImage()
        rgb_msg.header.stamp = timestamp
        rgb_msg.header.frame_id = "camera_link"
        rgb_msg.format = "jpeg"
        rgb_msg.data = np.array(jpeg_data).tobytes()

        # --- Create fake depth image (random depth) ---
        depth_array = np.random.uniform(0.0, 5.0, (480, 640)).astype(np.float32)
        # Normalize depth to 0-255 for PNG 8-bit
        depth_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        # Encode to PNG
        success_depth, png_data = cv2.imencode('.png', depth_uint8)
        if not success_depth:
            rospy.logerr("Failed to encode depth image")
            continue

        depth_msg = CompressedImage()
        depth_msg.header.stamp = timestamp
        depth_msg.header.frame_id = "camera_link"
        depth_msg.format = "png"
        depth_msg.data = np.array(png_data).tobytes()

        # CameraInfo header
        camera_info.header.stamp = timestamp
        camera_info.header.frame_id = "camera_link"

        # Publish
        rgb_pub.publish(rgb_msg)
        depth_pub.publish(depth_msg)
        camera_info_pub.publish(camera_info)

        rate.sleep()

if __name__ == '__main__':
    try:
        fake_camera_node()
    except rospy.ROSInterruptException:
        pass
