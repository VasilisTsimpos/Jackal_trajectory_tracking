#!/usr/bin/env python3

import rospy
import numpy as np
from jackal_controll import Jackal  # Ensure this module exists
from geometry_msgs.msg import PointStamped
from tools import getError, getAngularError
import matplotlib.pyplot as plt


P_point = np.zeros((3,))
running = False


def getPoint(data):
    global P_point, running
    running = True
    
    P_point[:] = data.point.x, data.point.y, 0.0


if __name__ == '__main__':
    robot = Jackal(10)
    rate = robot.getRate()
    
    I = np.eye(3)
    t = 0

    v_l_track = []
    time = []

    kp, kd = 0.5, 0.0

    # point_sub = rospy.Subscriber('/minimum_point_pub', PointStamped, getPoint)  # Fixed topic name
    point_sub = rospy.Subscriber('/object_coordinates', PointStamped, getPoint)  # Fixed topic name

    while not running:
        pass
    
    try:
        while not rospy.is_shutdown():
            Pb = robot.getPosition()
            theta = robot.getTheta()
            R = robot.getRotationMatrix()
            P = R @ P_point + Pb

            e = getError(Pb, P, R)
            r = np.linalg.norm(e)

            e_ang = getAngularError(e)


            # if r < 1.5:
            #     v_l = -0.15 / r**2
            # else:
            #     v_l = 0
            #     raise Exception
            
            v_l = -0.15 / r**2


            v_l = max(-0.15, v_l)
            
            v_w = kp * e_ang + kd * (np.sin(e_ang) * np.cos(e_ang))

            # robot.setLinearSpeed(v_l)
            robot.setAngularSpeed(v_w)
            robot.setRobotSpeed()

            t += 0.1

            v_l_track.append(v_l)
            time.append(t)


            rate.sleep()
    
    except rospy.exceptions.ROSInterruptException:
        print('\n[ FINISHED ]')

    # except Exception:
    #     time = np.array(time)
    #     v_l_track = np.array(v_l_track)

    #     plt.plot(time, v_l_track)
    #     plt.xlabel('time -> (sec)')
    #     plt.ylabel('commanded speed -> (m/s)')
    #     plt.grid()
    #     plt.show()