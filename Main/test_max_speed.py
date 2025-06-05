#!/usr/bin/env python3

import rospy
import numpy as np
from jackal_controll import Jackal
import matplotlib.pyplot as plt
from tools import *

if __name__ == '__main__':
    time_log = []
    speed_log = []
    vel_log = []
    try:
        time, speed = 0, 0

        robot = Jackal(10)

        rate = robot.getRate()

        while not rospy.is_shutdown():
            if time >= 10:
                raise rospy.ROSInterruptException

            vel = robot.calcLinearVel()

            robot.setLinearSpeed(speed)
            robot.setRobotSpeed()

            speed += 0.05
            time += 0.1

            time_log.append(time)
            speed_log.append(speed)
            vel_log.append(vel)

            rate.sleep()
        
    except rospy.ROSInterruptException:
        plt.plot(np.array(time_log), np.array(speed_log))
        plt.plot(np.array(time_log), np.array(vel_log), ls = 'dotted')

        plt.xlabel('Time (sec)')
        plt.ylabel('Speed (m/s)')

        plt.legend(['Commanded Speed', 'Calculated Speed'])
        plt.grid()

        plt.show()