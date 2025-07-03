#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import numpy as np
import math
import matplotlib.pyplot as plt

class MovingPoint:
    def __init__(self):
        rospy.init_node('moving_point')
        
        self.rate = rospy.Rate(15)
        self.size = 6
        self.buffer_x = []
        self.buffer_y = []
        self.buffer_z = []

        self.ok_flag = [False, False]
        self.pci = np.array([0, 0, 0])
        self.x1, self.x2, self.x3 = 0, 0, 0
        self.log = []
        self.filtered_log = []
        self.time_log = []

        rospy.Subscriber('/object_coordinates', PointStamped, self.point_callback)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odometry_callback)

        self.vel_pub = rospy.Publisher('/moving/point_vel', Vector3, queue_size=10)

        while  not (self.ok_flag[0] and self.ok_flag[1]):
            rospy.loginfo('waiting for all callbacks to run at least on time...')

        self.run()


    def point_callback(self, data):
        """
        The point from the topic is refrence from base_link of the robot
        so with the following we get the point coordinates refrenced from
        the optical point of the camera.
        """
    
        pi = np.array([data.point.x, data.point.y, data.point.z])
        
        if pi[0] == 0 and pi[1] == 0 and pi[2] == -1:
            self.out_of_bounds = True
        else: 
            self.out_of_bounds = False

        Rrc = np.array([
            [0,  0, 1], 
            [-1, 0, 0], 
            [0, -1, 0]
        ])
        
        self.pci = Rrc.T @ (pi - np.array([0, 0.044, 0.35]))
        """"""""""""""""""""""""""""""""""""""""""""""""
        """"""""""""""""""""""""""""""""""""""""""""""""

        self.ok_flag[0] = True
        

    def odometry_callback(self, data):
        position = data.pose.pose.position
        self.x1 = position.x
        self.x2 = position.y

        orientation = data.pose.pose.orientation
        self.x3 = 2 * math.atan2(orientation.z, orientation.w)

        self.ok_flag[1] = True

    def filter(self, buffer, new_sample):
        buffer.append(new_sample)
        if len(buffer) > self.size:
            buffer.pop(0)
        return sum(buffer) / len(buffer)

    def run(self):
        previous_time = rospy.Time().now()

        time = 0
        pci = self.pci
        x1, x2, x3 = self.x1, self.x2, self.x3

        previous_pos = np.array([
            [x1*np.sin(x3) - x2*np.cos(x3) + pci[0]],
            [pci[1] - 0.35],
            [x1*np.cos(x3) + x2*np.sin(x3) + pci[2]]
        ])
        
        rospy.sleep(0.05)

        try:
            while not rospy.is_shutdown():
                pci = self.pci
                x1, x2, x3 = self.x1, self.x2, self.x3
                
                # calculate the position of the interest from the inertial frame 
                p0i_c = np.array([
                    [x1*np.sin(x3) - x2*np.cos(x3) + pci[0]],
                    [pci[1] - 0.35],
                    [x1*np.cos(x3) + x2*np.sin(x3) + pci[2]]
                ])

                # Keep track of time
                current_time = rospy.Time().now()
                dt = (current_time - previous_time).to_sec()
                time += dt

                # calculate the numerical derivative
                p_dot = (p0i_c - previous_pos) / dt
                # Filter point
                p_dot_filtered = np.array([
                    [self.filter(self.buffer_x, p_dot[0])],
                    [self.filter(self.buffer_y, p_dot[1])],
                    [self.filter(self.buffer_z, p_dot[2])]
                ])

                # Create Vector3 message
                vel = Vector3()

                if self.out_of_bounds:
                    vel.x = 0
                    vel.y = 0
                    vel.z = 0
                else: 
                    vel.x = p_dot_filtered[0]
                    vel.y = p_dot_filtered[1]
                    vel.z = p_dot_filtered[2]

                # Publish speed vector
                self.vel_pub.publish(vel)

                # Updating variables
                previous_pos = p0i_c
                previous_time = current_time
                
                # Loging values
                # self.log.append(p_dot.flatten())
                # self.filtered_log.append(p_dot_filtered.flatten())
                # self.time_log.append(time)

                self.rate.sleep()

        except rospy.ROSInterruptException:
            pass
    
    def plot(self):
        p_dot = np.array(self.log)
        p_dot_filterd = np.array(self.filtered_log)
        time = np.array(self.time_log)

        fig, ax = plt.subplots(3, 1)

        ax[0].plot(time, p_dot[:, 0])
        ax[0].set_ylim(-1.5, 1.5)
        ax[0].set_ylabel('x vel --> (m/s)')
        ax[0].set_xlabel('time --> (sec)')
        ax[0].grid()

        ax[1].plot(time, p_dot[:, 1])
        ax[1].set_ylim(-1.5, 1.5)
        ax[1].set_ylabel('y vel --> (m/s)')
        ax[1].set_xlabel('time --> (sec)')
        ax[1].grid()

        ax[2].plot(time, p_dot[:, 2])
        ax[2].set_ylim(-1.5, 1.5)
        ax[2].set_ylabel('z vel --> (m/s)')
        ax[2].set_xlabel('time --> (sec)')
        ax[2].grid()

        fig.suptitle("Speed")
        fig.tight_layout()

        fig1, ax1 = plt.subplots(3, 1)

        ax1[0].plot(time, p_dot_filterd[:, 0])
        ax1[0].set_ylim(-1.5, 1.5)
        ax1[0].set_ylabel('x vel --> (m/s)')
        ax1[0].set_xlabel('time --> (sec)')
        ax1[0].grid()

        ax1[1].plot(time, p_dot_filterd[:, 1])
        ax1[1].set_ylim(-1.5, 1.5)
        ax1[1].set_ylabel('y vel --> (m/s)')
        ax1[1].set_xlabel('time --> (sec)')
        ax1[1].grid()

        ax1[2].plot(time, p_dot_filterd[:, 2])
        ax1[2].set_ylim(-1.5, 1.5)
        ax1[2].set_ylabel('z vel --> (m/s)')
        ax1[2].set_xlabel('time --> (sec)')
        ax1[2].grid()
        fig1.suptitle("Speed Filtered")
        fig1.tight_layout()

        plt.show()





if __name__ == "__main__":
    mp = MovingPoint()
    # mp.plot()