#!/usr/bin/env python3

import csv
import os
import rospy
import numpy as np
from jackal_controll import Jackal
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped, Vector3
import matplotlib.pyplot as plt
from scipy import ndimage
from tools import *

class FOVController:
    def __init__(self):
        # Callback variables
        self.pixel = np.array([0, 0])
        self.p0i_dot = np.array([0,0,0]) 
        self.depth = 0
        self.dimensions = False
        self.out_of_bounds = False
        self.w = 0 
        self.h = 0 
        self.fx = 0
        self.fy = 0
        # Visual servoing variables
        self.Lx  = np.zeros((2,2))
        # APF variables
        self.k1 = 8e6 # on x axis
        self.k2 = 3e6 # on y axis
        self.d1 = 210 
        self.d2 = 100
        self.a = 80

        # Log measurments
        self.position_log = []
        self.theta_log = []
        self.pixel_log = []
        self.time_log = []
        self.dx_log = []
        self.dy_log = []

        # Jackal Control
        self.robot = Jackal(30)
        self.rate = self.robot.getRate()

        # ROS Topic Subscribers
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_callback)
        rospy.Subscriber('/2d_point', PointStamped, self.pixel_callback)
        rospy.Subscriber('/object_coordinates', PointStamped, self.point_callback)
        rospy.Subscriber('/moving/point_vel', Vector3, self.velocity_callback)

        np.set_printoptions(precision=3, suppress=True)
    
    def point_callback(self, data):
        # the point is being published from the perspective of the camera

        pi = np.array([data.point.x, data.point.y, data.point.z])

        Rrc = np.array([
            [0,  0, 1], 
            [-1, 0, 0], 
            [0, -1, 0]
        ])
        
        pci = Rrc.T @ (pi - np.array([0, 0.044, 0.35])) # Finally the point is refrenced from the camera optical frame
        self.depth = pci[2] 

    def pixel_callback(self, data):
        # If the pixel is inside the field of view update the variables 
        if -1 < data.point.x  and data.point.x <= self.w and -1 < data.point.y  and data.point.y <= self.h:
            self.pixel = np.array([data.point.x, data.point.y])
            self.out_of_bounds = False
        else: # If its not inside the field of view zero the values of the variables
            self.pixel = np.array([0, 0])
            self.out_of_bounds = True

    def camera_callback(self, data):
        # Get Dimentions of image
        self.w, self.h = data.width, data.height
        self.fx, self.fy = data.K[0], data.K[4]
        self.cx, self.cy = data.K[2], data.K[5]

        self.dimensions = True
    
    def velocity_callback(self,data):
        self.p0i_dot = np.array([data.x, data.y, data.z])

    def create_potential(self, width, height):
        x = np.linspace(0.1, width - 0.1, width)
        y = np.linspace(0.1, height - 0.1, height)

        X, Y = np.meshgrid(x, y) # Create a meshgrid

        # Clip to avoid division by zero
        x_left = np.clip(X, 1e-6, None)
        x_right = np.clip(self.w - X, 1e-6, None)
        y_bottom = np.clip(Y, 1e-6, None)
        y_top = np.clip(self.h - Y, 1e-6, None)

        # Compute Artificial Potentials
        V1 = np.where((X > 0) & (X <= self.d1) & (Y > 0) & (Y < self.h),
                    (self.k1/2) * (1/x_left - 1/self.d1)**2, 0)

        V2 = np.where((X >= self.w - self.d1) & (X < self.w) & (Y > 0) & (Y < self.h),
                    (self.k1/2) * (1/x_right - 1/self.d1)**2, 0)

        V3 = np.where((Y > 0) & (Y <= self.d2) & (X > 0) & (X < self.w),
                    (self.k2/2) * (1/y_bottom - 1/self.d2)**2, 0)

        V4 = np.where((Y >= self.h - self.d2) & (Y < self.h) & (X > 0) & (X < self.w),
                    (self.k2/2) * (1/y_top - 1/self.d2)**2, 0)

        V_total = V1 + V2 + V3 + V4
        V_total = np.clip(V_total, None, 1e6)

        return V_total
    
    def create_potential_shifted(self, width, height):
        x = np.linspace(0.1, width - 0.1, width)
        y = np.linspace(0.1, height - 0.1, height)
        X, Y = np.meshgrid(x, y)
        
        # Clip to avoid division by zero based on your original expressions
        x_minus_a = np.clip(X - self.a, 1e-6, None)
        w_minus_x = np.clip(self.w - X, 1e-6, None)
        y_clipped = np.clip(Y, 1e-6, None)
        h_minus_y = np.clip(self.h - Y, 1e-6, None)
        
        # V1: (k1/2) * (1/(x-a) - 1/(d1-a))^2 for {a <= x <= d1} & {0 <= y <= h}
        V1 = np.where((X >= self.a) & (X <= self.d1) & (Y >= 0) & (Y <= self.h),
                     (self.k1/2) * (1/x_minus_a - 1/(self.d1 - self.a))**2, 0)
        
        # V2: (k1/2) * (1/(w-x) - 1/(d1-a))^2 for {w >= x >= w+a-d1} & {0 <= y <= h}
        V2 = np.where((X <= self.w) & (X >= self.w + self.a - self.d1) & (Y >= 0) & (Y <= self.h),
                     (self.k1/2) * (1/w_minus_x - 1/(self.d1 - self.a))**2, 0)
        
        # V3: 0 for {d1 < x < w - d1 + a} & {0 <= y <= h} (already zero)
        V3 = np.zeros_like(X)
        
        # V4: (k2/2) * (1/y - 1/d2)^2 for {0 <= y <= d2} & {a <= x <= w}
        V4 = np.where((Y >= 0) & (Y <= self.d2) & (X >= self.a) & (X <= self.w),
                     (self.k2/2) * (1/y_clipped - 1/self.d2)**2, 0)
        
        # V5: (k2/2) * (1/(h-y) - 1/d2)^2 for {h >= y >= h-d2} & {a <= x <= w}
        V5 = np.where((Y <= self.h) & (Y >= self.h - self.d2) & (X >= self.a) & (X <= self.w),
                     (self.k2/2) * (1/h_minus_y - 1/self.d2)**2, 0)
        
        # V6: 0 for {d2 < y < h - d2} & {a <= x <= w} (already zero)
        V6 = np.zeros_like(X)
        
        V_total = V1 + V2 + V3 + V4 + V5 + V6
        V_total = np.clip(V_total, None, 1e7)
        
        return V_total
    """
            Get the Potential field and its partial derivatives 
    """
    def calculate_potential(self):
        self.V = self.create_potential_shifted(self.w, self.h)
        self.dx = ndimage.sobel(self.V, axis=1)
        self.dy = ndimage.sobel(self.V, axis=0)

    """
            Normilize values from [0, 1] for visualization purposes only
    """
    def normalize_array(self, array):
        # If the array is a numpy array, we handle it using numpy functions
        if isinstance(array, np.ndarray):
            min_val = np.min(array)
            max_val = np.max(array)
            return (array - min_val) / (max_val - min_val)
        
        # Otherwise, we assume it's a list (1D or 2D)
        if isinstance(array[0], list):  # 2D array
            min_val = min(min(row) for row in array)
            max_val = max(max(row) for row in array)
            return [[(x - min_val) / (max_val - min_val) for x in row] for row in array]
        else:  # 1D array
            min_val = min(array)
            max_val = max(array)
            return [(x - min_val) / (max_val - min_val) for x in array]
    
    """
            Visualize the artificial potential field and its partials derivative
    """
    def visualize_potential(self):
        V = self.normalize_array(self.V)
        dx = self.normalize_array(self.dx)
        dy = self.normalize_array(self.dy)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(V, cmap='gray')

        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        ax1.imshow(dx, cmap='gray', )

        fig2 = plt.figure()
        ax2 = fig2.add_subplot()
        ax2.imshow(dy, cmap='gray')

        plt.show()

    """
            Calculation of interaction matrix 
    """
    def calculate_interaction_matrix(self, x, y):
        if not self.out_of_bounds: # if out of bounds 
            fx = self.fx
            cx, cy = self.cx, self.cy
            
            z = self.depth

            self.Lx = np.array([ # The interaction matrix according to my notes
                [(x - cx) / z,  fx + ((x - cx)**2) / fx],
                [(y - cy) / z,  ((x - cx) * (y - cy)) / fx]
            ])  

        else:
            self.Lx = np.eye(2) 

        return self.Lx
    
    def calculate_new_term(self, x, y):
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        z = self.depth

        p0i_dot = self.p0i_dot # Interest point velocity 

        ds_dp = np.array([
            [fx / z,      0,   -(x - cx) / z],
            [     0,  fy / z,  -(y - cy) / z]
        ])

        return ds_dp @ p0i_dot
        
    def run(self):
        previous_time = rospy.Time.now()
        time=0

        rospy.sleep(0.1)
        self.calculate_potential() # Calculation of the artificial potential field
        dx = self.dx.T # lookup table of the partial derivative
        dy = self.dy.T # lookup table of the partial derivative 
        
        # self.visualize_potential()

        input("press enter to continue")

        try:#####------------------------------------CONTROL LOOP---------$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            while not rospy.is_shutdown():
                # Keep track of time
                current_time = rospy.Time.now()
                dt = (current_time - previous_time).to_sec()
                time += dt
                previous_time = current_time

                # if time > 15:
                #     raise rospy.ROSInterruptException

                P = self.robot.getPosition()
                theta = self.robot.getTheta()

                # Get the pixel detected
                x = int(self.pixel[0])  # x coordinate of pixel
                y = int(self.pixel[1])  # y coordinate of pixel

                if not self.out_of_bounds: # if pixel is out of bound then no potential so no gradient
                    dV = np.array([dx[x, y], dy[x, y]])
                else:
                    dV = np.array([0, 0])

                # Calculation of the controller
                Lx = self.calculate_interaction_matrix(x, y)
                if not dV[0] and not dV[1]:#########################---eimaste sto plateu tou potential---$$$$$$$$$$$
                    new_term = np.array([0, 0])
                else:
                    new_term = self.calculate_new_term(x, y)
                
                if np.linalg.det(Lx) != 0 :
                    Lx_inv = np.linalg.inv(Lx)
                    Vc = -Lx_inv @ (dV + 0*new_term) #---CONTROL SIGNAL
                else:
                    Vc = np.array([0, 0])

                vx = Vc[0] 
                wz = Vc[1] 

                # Saturate speeds 
                vx = min(max(vx, -0.18), 0.18)
                wz = min(max(wz, -0.75), 0.75)
                print(f'[x,y]: [{x}, {y}], [dx, dy]: [{dV[0]}, {dV[1]}]')
                print(f'vx: {vx:.3f}, wz: {wz:.3f}')
                print(Lx)
                print(Lx_inv)

                # Set Robot Speed
                self.robot.setLinearSpeed(vx)
                self.robot.setAngularSpeed(wz)
                self.robot.setRobotSpeed()

                # Log measurements
                self.position_log.append(P)
                self.pixel_log.append([x,y])
                self.time_log.append(time)
                self.theta_log.append(theta)
                self.dx_log.append(dx[x,y])
                self.dy_log.append(dy[x,y])
            
                self.rate.sleep()

        except rospy.ROSInterruptException:
            pass

    def get_next_filename(self, path='/home/csrl/catkin_ws/src/my_robot_controller/src/results/', base_name="exp", extension=".csv"):
        """
        Generate the next available filename with incremental numbering
        """

        counter = 1
        while True:
            filename = f"{base_name}_{counter}{extension}"
            filename = path + filename
            if not os.path.exists(filename):
                return filename
            counter += 1

    def save_data(self):
        """
        Save logged data to CSV file with automatic numbering
        """
        if not self.time_log:
            print("No data to save!")
            return
        
        filename = self.get_next_filename()
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['time', 'position_x', 'position_y', 'theta', 'pixel_x', 'pixel_y', 'dV / dx', 'dV / dy'])
                
                # Write data rows
                for i in range(len(self.time_log)):
                    row = [
                        self.time_log[i],
                        self.position_log[i][0],
                        self.position_log[i][1],
                        self.theta_log[i],
                        self.pixel_log[i][0],
                        self.pixel_log[i][1],
                        self.dx_log[i],
                        self.dy_log[i]
                    ]
                    writer.writerow(row)
            
            print(f"Data saved successfully to {filename}")
            
        except Exception as e:
            print(f"Error saving data: {e}")


    def plot_measure(self):
        time = np.array(self.time_log)
        pixels = np.array(self.pixel_log)
        position = np.array(self.position_log)
        theta = np.array(self.theta_log)
        dx = np.array(self.dx_log)
        dy = np.array(self.dy_log)

        fig1, ax1 = plt.subplots()
        ax1.plot(time, pixels[:, 0])
        ax1.set_xlabel('time -> (sec)')
        ax1.set_ylabel('x coordinate')
        ax1.set_ylim([0, self.w])
        ax1.grid()

        fig2, ax2 = plt.subplots()
        ax2.plot(time, pixels[:, 1])
        ax2.set_xlabel('time -> (sec)')
        ax2.set_ylim([0, self.h])
        ax2.grid()

        fig3, ax3 = plt.subplots()
        ax3.plot(pixels[:, 0], pixels[:, 1])
        ax3.plot(pixels[0, 0], pixels[0, 1], 'go', markersize=8, label='Start')
        ax3.plot(pixels[-1, 0], pixels[-1, 1], 'ro', markersize=8, label='End')
        ax3.set_xlabel('x coordinate')
        ax3.set_ylabel('y coordinate')
        ax3.set_xlim([0, self.w])
        ax3.set_ylim([0, self.h])
        ax3.legend()
        ax3.grid()

        # fig4, ax4 = plt.subplots(2,1)
        # ax4[0].plot(position[:, 0], position[:, 1], 'b-', linewidth=2, label='Robot Path')
        # ax4[0].plot(position[0, 0], position[0, 1], 'go', markersize=8, label='Start')
        # ax4[0].plot(position[-1, 0], position[-1, 1], 'ro', markersize=8, label='End')
        # ax4[0].set_xlabel('X Position (m)')
        # ax4[0].set_ylabel('Y Position (m)')
        # ax4[0].set_title('Robot Trajectory')
        # ax4[0].grid(True, alpha=0.3)
        # ax4[0].set_aspect('equal')
        # ax4[0].legend()

        # ax4[1].plot(time, np.rad2deg(theta))
        # ax4[1].set_xlabel('time -> (sec)')
        # ax4[1].set_ylabel('Angle -> (degrees)')
        # ax4[1].grid()
        # ax4[1].set_aspect('equal')
        # fig4.tight_layout()

        fig5, ax5 = plt.subplots(2,1)
        ax5[0].plot(time, dx, label='dV/dx')
        ax5[0].set_xlabel('time -> (sec)')
        ax5[0].set_ylabel('Potential on x axis')
        ax5[0].legend()
        ax5[0].grid()

        ax5[1].plot(time, dy, label='dV/dy')
        ax5[1].set_xlabel('time -> (sec)')
        ax5[1].set_ylabel('Potential on y axis')
        ax5[1].legend()
        ax5[1].grid()
        fig5.tight_layout()

        plt.show()

if __name__ == '__main__':
    controller = FOVController()

    controller.run()
    controller.plot_measure()
    ans = input('\nSave the data (y/n):')

    if ans.lower() in ['yes', 'y', '']:
        controller.save_data()

    
