#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PointStamped 
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import threading
import numpy as np
from numpy import sin, cos, pi
import math

class InteractivePoint:
    def __init__(self):
        rospy.init_node("interactive_point", anonymous=True)
        self.point_pub = rospy.Publisher("/object_coordinates", PointStamped, queue_size=10)
        self.odom_sub = rospy.Subscriber("/odometry/filtered", Odometry, self.__getPoseInfo)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.__getCameraInfo)
        self.rate = rospy.Rate(30)
        self.x = 0
        self.y = 0
        self.z = 0.0  # Default Z value, adjustable via slider
        self.theta = 0
        self.dots = None
        self.fov_patch = None
        self.x_data = [0]
        self.y_data = [0]
        self.cam_fov_horizontal = 60 * (pi/180)
        self.cam_range = 4.0
        self.robot_length = 0.5
        self.robot_width = 0.3
        self.robot_patch = None

    def __getCameraInfo(self, data):
        fx = data.K[0]
        width = data.width
        self.cam_fov_horizontal = 2 * math.atan2(width/2, fx)

    def __getPoseInfo(self, data):
        position = data.pose.pose.position
        self.X, self.Y = position.x, position.y
        orientation = data.pose.pose.orientation 
        self.theta = 2 * math.atan2(orientation.z, orientation.w)

    def create_robot_box(self, x, y, theta):
        l, w = self.robot_length, self.robot_width
        corners = np.array([
            [-l/2, -l/2,  l/2,  l/2, -l/2],
            [-w/2,  w/2,  w/2, -w/2, -w/2]
        ])
        R = np.array([
            [cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]
        ])
        transformed = R @ corners + np.array([[x], [y]])
        return list(zip(transformed[0, :], transformed[1, :]))

    def publisher(self):
        try:
            while not rospy.is_shutdown():
                point_msg = PointStamped()
                point_msg.header.frame_id = 'odom'
                point_msg.header.stamp = rospy.Time.now()
                point_msg.point.x = self.x 
                point_msg.point.y = self.y 
                point_msg.point.z = self.z  # Controlled by slider
                self.point_pub.publish(point_msg)
                self.rate.sleep()
        except rospy.ROSInterruptException:
            pass

    def start_publishing(self):
        self.t = threading.Thread(target=self.publisher)
        self.t.start()
        return self.t

    def on_move(self, event):
        if event.inaxes == self.ax:  # only update if mouse is in the main plot area
            if event.xdata is not None and event.ydata is not None:
                self.x_data[0] = event.xdata
                self.y_data[0] = event.ydata
                self.dots.set_data(self.x_data, self.y_data)
                self.x = event.xdata
                self.y = event.ydata
                self.fig.canvas.draw_idle()

    def rotz(self, theta):
        return np.array([
            [cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]
        ])

    def update_fov(self, event):
        base_points = np.array([
            [0, self.cam_range * cos(-self.cam_fov_horizontal/2), self.cam_range * cos(self.cam_fov_horizontal/2)],
            [0, self.cam_range * sin(-self.cam_fov_horizontal/2), self.cam_range * sin(self.cam_fov_horizontal/2)]
        ])
        rotated_points = self.rotz(self.theta) @ base_points + np.array([[self.X], [self.Y]])
        new_vertices = list(zip(rotated_points[0, :], rotated_points[1, :]))
        self.fov_patch.set_xy(new_vertices)
        self.fig.canvas.draw_idle()
        robot_vertices = self.create_robot_box(self.X, self.Y, self.theta)
        self.robot_patch.set_xy(robot_vertices)

    def plot(self):

        self.fig, ax = plt.subplots()
        self.ax = ax
        plt.subplots_adjust(right=0.85)  # Room for vertical slider on the right

        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-6.5, 6.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X -> (m)')
        ax.set_ylabel('Y -> (m)')
        ax.grid()

        base_points = np.array([
            [0, self.cam_range * cos(-self.cam_fov_horizontal/2), self.cam_range * cos(self.cam_fov_horizontal/2)],
            [0, self.cam_range * sin(-self.cam_fov_horizontal/2), self.cam_range * sin(self.cam_fov_horizontal/2)]
        ])
        rotated_points = self.rotz(self.theta) @ base_points + np.array([[self.X], [self.Y]]) 
        polygon_vertices = list(zip(rotated_points[0, :], rotated_points[1, :]))

        self.fov_patch = patches.Polygon(polygon_vertices, closed=True, alpha=0.3)
        ax.add_patch(self.fov_patch)

        self.dots, = ax.plot([0], [0], 'ro')
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        robot_vertices = self.create_robot_box(self.X, self.Y, self.theta)
        self.robot_patch = patches.Polygon(robot_vertices, closed=True, color='black', alpha=1)
        ax.add_patch(self.robot_patch)

        # Vertical slider for Z on the right
        ax_z = plt.axes([0.88, 0.2, 0.03, 0.6])  # [left, bottom, width, height]
        z_slider = Slider(ax_z, 'Z', -4.0, 4.0, valinit=self.z, orientation='vertical')

        def update_z(val):
            self.z = z_slider.val

        z_slider.on_changed(update_z)


        timer = self.fig.canvas.new_timer(interval=100)
        timer.add_callback(self.update_fov, None)
        timer.start()

        try:
            plt.show()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    ip = InteractivePoint()
    t = ip.start_publishing()
    ip.plot()
    t.join()
