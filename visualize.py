import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np


# Load the data
def read_data():
    timestep_data = []
    current_data = []
    with open("particles.csv", "r") as file:
        for line in file:
            if line.startswith("Timestep"):
                if current_data:
                    timestep_data.append(
                        pd.DataFrame(current_data, columns=["x", "y", "z", "radius"])
                    )
                    current_data = []
            elif line.strip():
                current_data.append(list(map(float, line.split(","))))
    if current_data:
        timestep_data.append(
            pd.DataFrame(current_data, columns=["x", "y", "z", "radius"])
        )
    return timestep_data


timestep_data = read_data()

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# Define the sphere
def draw_sphere(ax, radius):
    u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 50j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_surface(x, y, z, color="b", alpha=0.1)


# Draw the sphere with the defined radius
SPHERE_RADIUS = 1000.0
draw_sphere(ax, SPHERE_RADIUS)


def update_graph(num):
    data = timestep_data[num]
    graph._offsets3d = (data["x"], data["y"], data["z"])  # Update positions
    graph.set_sizes(data["radius"])  # Update sizes directly from radii
    title.set_text("3D Particle Simulation, timestep={}".format(num))


# Initialize scatter plot
data = timestep_data[0]
graph = ax.scatter(
    data["x"],
    data["y"],
    data["z"],
    s=data["radius"],
    c="r",
    alpha=0.6,
    edgecolors="w",
)
title = ax.set_title("3D Particle Simulation")

# Create animation
ani = FuncAnimation(
    fig, update_graph, frames=len(timestep_data), interval=50, repeat=False
)
plt.show()
