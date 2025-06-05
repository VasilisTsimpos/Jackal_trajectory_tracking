#!/usr/bin/env python3

import rospy
import numpy as np
from jackal_controll import Jackal
import matplotlib.pyplot as plt
from tools import *

if __name__ == "__main__":
    try:
        robot = Jackal(10)
        time = 0

        rate = robot.getRate()
        
        #Robots orientation
        R0b = robot.getRotationMatrix()
        
        #Robots position from inertial frame
        P0b = robot.getPosition()

        #Robots desired position from inertial frame
        P0d = [0.0, 0.0, 0.0]

        while not rospy.is_shutdown():
            e = getError(robot.getPosition(), P0d, robot.getRotationMatrix()) # From robots frame 

            lin_vel = 1 * getLinearError(e)
            ang_vel = 1.5 * getAngularError(e)
            
            if np.linalg.norm(e[:2]) < 0.001:
                lin_vel = 0
                ang_vel = 0
                raise rospy.ROSInterruptException

            robot.setLinearSpeed(lin_vel)
            robot.setAngularSpeed(ang_vel)
            robot.setRobotSpeed()   

            time += 0.1
            rate.sleep()
            
    except rospy.ROSInterruptException:
        pass