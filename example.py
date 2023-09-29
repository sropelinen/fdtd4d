
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from fdtd4d import FDTD
from boundaries import ABC


# Setup
fdtd = FDTD((50, 50, 1, 50))
fdtd.E_init[25, 25, 0, 25, 3] = 1
fdtd.add_BC(ABC(10, 10, 0, 10))

# Run
E, H = fdtd.run(70)

# Plot
energy = np.sum(np.sum(E**2 + H**2, -1)[:, 10:-10, 10:-10, :, 10:-10]**.5, -1)

plot = plt.imshow(energy[0, ...],
                  interpolation="bicubic",
                  cmap="hot",
                  vmin=0,
                  vmax=np.max(energy))
plt.axis("equal")
plt.gca().set_axis_off()
plt.tight_layout()

def animate(i):
    plot.set_array(energy[i, ...])
anim = FuncAnimation(plt.gcf(), animate, 70, interval=100)
plt.show()
