#!/usr/bin/env python3

import rospy
import numpy as np
from jackal_controll import Jackal
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped
import matplotlib.pyplot as plt
from scipy import ndimage
from tools import *

class FOVController:
    def __init__(self):
        # Callback variables
        self.pcx = np.array([0, 0, 0])
        self.pixel = np.array([0, 0]) 
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
        self.k1 = 3e7 # on x axis
        self.k2 = 3e7 # on y axis
        self.d0 = 200 
        self.d1 = 150

        # Jackal Control
        self.robot = Jackal(30)
        self.rate = self.robot.getRate()

        # ROS Topic Subscribers
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_callback)
        # rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_callback)
        rospy.Subscriber('/2d_point', PointStamped, self.pixel_callback)
        rospy.Subscriber('/object_coordinates', PointStamped, self.point_callback)

        np.set_printoptions(precision=3, suppress=True)
    
    def point_callback(self, data):
        """ sim """
        # # Get the position detected by the camera
        # px = np.array([data.point.x, data.point.y, data.point.z])

        # # The position should be in refrence to the camera
        # # In the simulation the position returned its in refrence to the world frame
        # # So it needs to be refrenced to the camera frame
        # pc = self.robot.getPosition()
        # p = (px - pc)
        # R0r = self.robot.getRotationMatrix()
        # Rrc = np.array([
        #     [0,  0, 1], 
        #     [-1, 0, 0], 
        #     [0, -1, 0]
        # ])
        # R0c = R0r @ Rrc

        # self.pcx = R0c.T @ p 

        """ real """
        px = np.array([data.point.x, data.point.y, data.point.z])
        if px[0] == 0 and px[1] == 0 and px[2] == -1:
            self.out_of_bounds = True
        else:
            self.out_of_bounds = False

        Rrc = np.array([
            [0,  0, 1], 
            [-1, 0, 0], 
            [0, -1, 0]
        ])
        self.pcx = Rrc.T @ (px - + np.array([0, 0.044, 0.35]))

    def pixel_callback(self, data):
        # If the pixel is inside the field of view update the variables 
        if 0 <= data.point.x  and data.point.x <= self.w and 0 <= data.point.y  and data.point.y <= self.h and data.point.z >= 0:
            self.pixel = np.array([data.point.x, data.point.y])
            self.depth = data.point.z
            # self.out_of_bounds = False
        else: # If its not inside the field of view zero the values of the variables
            self.pixel = np.array([0, 0])
            self.depth = 0
            # self.out_of_bounds = True

    def camera_callback(self, data):
        # Get Dimentions of image
        self.w, self.h = data.width, data.height
        self.fx, self.fy = data.K[0], data.K[4]
        self.dimensions = True

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
        V1 = np.where((X > 0) & (X <= self.d0) & (Y > 0) & (Y < self.h),
                    (self.k1/2) * (1/x_left - 1/self.d0)**2, 0)

        V2 = np.where((X >= self.w - self.d0) & (X < self.w) & (Y > 0) & (Y < self.h),
                    (self.k1/2) * (1/x_right - 1/self.d0)**2, 0)

        V3 = np.where((Y > 0) & (Y <= self.d1) & (X > 0) & (X < self.w),
                    (self.k2/2) * (1/y_bottom - 1/self.d1)**2, 0)

        V4 = np.where((Y >= self.h - self.d1) & (Y < self.h) & (X > 0) & (X < self.w),
                    (self.k2/2) * (1/y_top - 1/self.d1)**2, 0)

        V_total = V1 + V2 + V3 + V4
        V_total = np.clip(V_total, None, 1e6)

        return V_total

    """
            Get the Potential field and its partial derivatives 
    """
    def calculate_potential(self):
        self.V = self.create_potential(self.w, self.h)
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
    def calculate_interaction_matrix(self):
        if not self.out_of_bounds: # if out of bounds 
            fx = self.fx
            fy = self.fy
            px, py, pz = self.pcx[0], self.pcx[1], self.pcx[2] 
            print(f'px: {px:.3f}, py: {py:.3f}, pz: {pz:.3f}')

            self.Lx = np.array([ # The interaction matrix according to my notes
                [(px * fx) / pz**2, fx - ((fx * px**2) / pz**2)],
                [(py * fy) / pz**2, (py * px * fy) / pz**2]
            ])  
        else:
            self.Lx = np.eye(2) 

        return self.Lx
        
    def run(self):
        previous_time = rospy.Time.now()
        vx_log = []
        wz_log = []
        time_log = []
        time=0

        rospy.sleep(0.1)
        self.calculate_potential() # Calculation of the artificial potential field
        dx = self.dx.T # lookup table of the partial derivative
        dy = self.dy.T # lookup table of the partial derivative 
        
        self.visualize_potential()

        try:
            while not rospy.is_shutdown():
                # Keep track of time
                current_time = rospy.Time.now()
                dt = (current_time - previous_time).to_sec()
                time += dt
                previous_time = current_time

                # Get the pixel detected
                x = int(self.pixel[0])  # x coordinate of pixel
                y = int(self.pixel[1])  # y coordinate of pixel

                if not self.out_of_bounds: # if pixel is out of bound then no potential so no gradient
                    dV = np.array([dx[x, y], dy[x, y]])
                else:
                    dV = np.array([0, 0])

                # Calculation of the controller
                Lx = self.calculate_interaction_matrix()
                if np.linalg.det(Lx) != 0 :
                    Vc = -np.linalg.inv(Lx) @ dV
                else:
                    Vc = np.array([0, 0])

                vx = Vc[0] 
                wz = Vc[1] 

                # Saturate speeds 
                vx = min(max(vx, -0.18), 0.18)
                wz = min(max(wz, -0.7), 0.7)
                print(f'[x,y]: [{x}, {y}], [dx, dy]: [{dV[0]}, {dV[1]}]')
                print(f'vx: {vx:.3f}, wz: {wz:.3f}')
                print(Lx)

                # Set Robot Speed
                self.robot.setLinearSpeed(vx)
                self.robot.setAngularSpeed(wz)
                # self.robot.setLinearSpeed(0)
                # self.robot.setAngularSpeed(0)
                self.robot.setRobotSpeed()

                # Log speeds
                vx_log.append(vx)
                wz_log.append(wz)
                time_log.append(time)

                self.rate.sleep()

        except rospy.ROSInterruptException:
            print('\n[ FINISHED ]') 

if __name__ == '__main__':
    FOVController().run() 
