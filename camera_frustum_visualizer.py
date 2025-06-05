#!/usr/bin/env python3

import rospy
import math
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class CameraFrustumVisualizer:
    def __init__(self):
        rospy.init_node('camera_frustum_visualizer')

        self.parent_frame = rospy.get_param('~parent_frame', 'base_link')  # frustum frame
        self.depth = rospy.get_param('~depth', 5)  # how far to draw frustum

        self.marker_pub = rospy.Publisher('camera_frustum_marker', Marker, queue_size=1)

        # rospy.Subscriber('/camera/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        # rospy.loginfo("Camera Frustum Visualizer started in frame: %s", self.parent_frame)
        rospy.spin()

    def camera_info_callback(self, msg):
        fx = msg.K[0]
        fy = msg.K[4]
        width = msg.width
        height = msg.height

        # Calculate FoV
        hfov = 2 * math.atan(width / (2 * fx))
        vfov = 2 * math.atan(height / (2 * fy))
        # rospy.loginfo("FoV: hfov=%.2f deg, vfov=%.2f deg", math.degrees(hfov), math.degrees(vfov))

        # Far plane dimensions
        far_width = 2 * math.tan(hfov / 2) * self.depth
        far_height = 2 * math.tan(vfov / 2) * self.depth

        # Define points in base_link coordinates
        origin = Point(0, 0, 0)
        tl = Point(self.depth, far_width/2, far_height/2)    # top-left
        tr = Point(self.depth, -far_width/2, far_height/2)   # top-right
        br = Point(self.depth, -far_width/2, -far_height/2)  # bottom-right
        bl = Point(self.depth, far_width/2, -far_height/2)   # bottom-left

        # Create marker
        marker = Marker()
        marker.header.frame_id = self.parent_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_frustum"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)

        # Add lines: from origin to each corner
        marker.points += [origin, tl]
        marker.points += [origin, tr]
        marker.points += [origin, br]
        marker.points += [origin, bl]

        # Rectangle at far plane
        marker.points += [tl, tr]
        marker.points += [tr, br]
        marker.points += [br, bl]
        marker.points += [bl, tl]

        # Publish
        self.marker_pub.publish(marker)
        # rospy.loginfo("Frustum marker published in %s", self.parent_frame)

if __name__ == '__main__':
    try:
        CameraFrustumVisualizer()
    except rospy.ROSInterruptException:
        pass
