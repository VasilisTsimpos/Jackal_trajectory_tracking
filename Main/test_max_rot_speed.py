#!/usr/bin/env python3

import rospy
import numpy as np
from jackal_controll import Jackal
import matplotlib.pyplot as plt
from tools import *

if __name__ == '__main__':
    time_log = []
    ang_speed_log = []
    ang_vel_log = []
    
    try:
        time, speed = 0, 0

        robot = Jackal(10)

        rate = robot.getRate()

        while not rospy.is_shutdown():
            if time >= 10:
                raise rospy.ROSInterruptException

            vel = robot.calcAngularVel()

            robot.setAngularSpeed(speed)
            robot.setRobotSpeed()

            speed += 0.1
            time += 0.1

            time_log.append(time)
            ang_speed_log.append(speed)
            ang_vel_log.append(vel)

            rate.sleep()
        
    except rospy.ROSInterruptException:
        # plt.plot(np.array(time_log), np.array(ang_speed_log))
        # plt.plot(np.array(time_log), np.array(ang_vel_log), ls = 'dotted')

        # plt.xlabel('Time (sec)')
        # plt.ylabel('Angular Speed (rad/s)')

        # # plt.legend(['Commanded Angular Speed', 'Calculated Angular Speed'])
        # plt.grid()
        
        # plt.show()
        pass