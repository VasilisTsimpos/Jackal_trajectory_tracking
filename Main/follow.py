#!/usr/bin/env python3

import rospy
import numpy as np
from jackal_controll import Jackal  # Ensure this module exists
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from tools import getError, getLinearError, getAngularError
import matplotlib.pyplot as plt


P_point = np.zeros((3,))
brake = True
running = False
stop = False

# Callback Method   
def getPoint(data): 
    global P_point, running, stop
    running = True

    stop = False
    if data.point.z == -1:
        stop = True
    # print(f'x: {data.point.x},y: {data.point.y}') 
    P_point[:] = data.point.x, data.point.y, 0.0

def brakeCallBack(msg):
    global brake 
     
    brake = msg.data

if __name__ == '__main__':
    robot = Jackal(15)
    rate = robot.getRate()
    
    I = np.eye(3)

    kp, kd = 2, 1.5 

    # point_sub = rospy.Subscriber('/minimum_point_pub', PointStamped, getPoint)  # For simulation
    point_sub = rospy.Subscriber('/object_coordinates', PointStamped, getPoint)  
    brake_sub = rospy.Subscriber('/brake', Bool, brakeCallBack)

    while not running:
        pass
    
    try:
        while not rospy.is_shutdown():
            Pb = robot.getPosition()
            theta = robot.getTheta()
            R = robot.getRotationMatrix()
            # P = R @ P_point + Pb
            P = P_point

            if not brake and not stop:
                ########### Angular ###########
                e = getError(Pb, P, R)

                e_ang = getAngularError(e)
                # print(f'ang_e: {np.rad2deg(e_ang)}')
                v_w = kp * e_ang + kd * (np.sin(e_ang) * np.cos(e_ang))

                ########### Linear ###########
                # P = R @ (P_point - [0.7, 0.0, 0.0]) + Pb 
                P = (P_point - (R @ [1.25, 0.0, 0.0]))
                e = getError(Pb, P, R)

                v_l = 2.3 * getLinearError(e)
                v_l = min(max(v_l, -0.35), 0.35)

                if np.linalg.norm(e) > 1.8:
                    v_l = 0
                    v_w = 0
            else:
                v_l = 0
                v_w = 0

            ########### Command Speed ###########
            # robot.setLinearSpeed(v_l)
            # robot.setAngularSpeed(v_w)
            robot.setLinearSpeed(0.2)
            robot.setAngularSpeed(0.5)
            robot.setRobotSpeed()

            rate.sleep()
    
    except rospy.exceptions.ROSInterruptException:
        print('\n[ FINISHED ]')
