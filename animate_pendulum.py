import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from time import time
import os


###  Load simulation data from file  ###
this_dir = os.path.dirname(os.path.abspath(__file__))

angles = np.load(this_dir + "/angle_list.npy")
magnets_prop = np.load(this_dir + "/magnets.npy")
if len(magnets_prop) > 0:
    magnets_pos = magnets_prop[:, 0:2]
    magnets_pos[:, 1] *= -1
    magnets_mom = magnets_prop[:, 2:]
    magnets_mom[:, 1] *= -1
equil_points = np.load(this_dir + "/critical_points.npy")

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)

# Initialize scene objects
line, = ax.plot([], [], 'o-', lw=2)
pendulum, = ax.plot([], [], 'o', lw=2)
equil_lines = []
magnets = []

if len(magnets_prop) > 0:
    for i in range(magnets_pos.shape[0]):
        magnet_pos, = ax.plot([], [], 'o', lw=2)
        magnet_mom, = ax.plot([], [], '--', lw=2, color=magnet_pos.get_color(), alpha=0.5)
        
        magnets.append([magnet_pos, magnet_mom])

for _ in equil_points:
    equil_lines.append(ax.plot([], [], '.', lw=2, color='black')[0])


def init():
    line.set_data([], [])
    pendulum.set_data([], [])
    # equil_lines.set_data([], [])
    if len(magnets_prop) > 0:
        for magnet in magnets:
            magnet[0].set_data([], [])  # position
            magnet[1].set_data([], [])  # momentum

    for equil_line in equil_lines:
        equil_line.set_data([], [])

    return line, pendulum

# Update animation frame
def update(frame):
    # Redraw pendulum
    x = np.sin(angles[frame])
    y = -np.cos(angles[frame])
    line.set_data([0, x], [0, y])
    pendulum.set_data([x], [y])
    
    for equil_angle, equil_line in zip(equil_points, equil_lines):
        x = np.sin(equil_angle)
        y = -np.cos(equil_angle)
        equil_line.set_data([x], [y])
        # equil_line.set_data([0, x], [0, y])
    
    # Redraw magnet(s)
    for i, magnet in enumerate(magnets):
        pos = magnets_pos[i]
        mom_vec = magnets_mom[i] + pos
        magnet[0].set_data([pos[0]], [pos[1]])
        magnet[1].set_data([pos[0], mom_vec[0]], [pos[1], mom_vec[1]])
    
    artists = [line, pendulum]
    
    for equil_line in equil_lines:
        artists.append(equil_line)
    
    for magnet in magnets:
        artists.extend(magnet)
        
    return artists

# Show the animation
ani = FuncAnimation(fig, update, frames=angles.shape[0], init_func=init, blit=True, interval=6)
plt.show()

