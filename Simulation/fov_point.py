import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy import sin, cos, pi

def rotz(theta):
    R = np.array([
        [cos(theta), -sin(theta)],
        [sin(theta),  cos(theta)]
    ])

    return R

fig, ax = plt.subplots()
ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)

# Draw polygon
points = np.array([
    [0,  5,  -5],
    [0, 10, 10]
])

points = rotz(pi/1.5)  @ points

# print(points[0,:])
ax.fill(points[0, :], points[1, :])

# Store the scatter plot
dots, = ax.plot([], [], 'ro')  # 'ro' = red dots
x_data, y_data = [0], [0]

def on_move(event):
    if event.inaxes is not None:
        x_data.pop(0)
        x_data.append(event.xdata)
        y_data.pop(0)
        y_data.append(event.ydata)
        dots.set_data(x_data, y_data)
        fig.canvas.draw_idle()

# Connect the click event to the handler
fig.canvas.mpl_connect('motion_notify_event', on_move)
ax.set_aspect('equal')
plt.show()
