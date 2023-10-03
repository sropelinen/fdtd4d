
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from fdtd4d import FDTD


fdtd = FDTD((70, 70, 1, 70))
fdtd.E_init[35, 35, 0, 35, 2] = 1
E, H = fdtd.run(80)

energy = np.sum(E**2 + H**2, -1)**.5
energy = np.sum(energy, 0)[20:-20, 20:-20, :, 20:-20]
plot = plt.imshow(energy[..., 0, 0],
                  cmap="hot",
                  vmin=0,
                  vmax=np.max(energy))
plt.axis("equal")
plt.gca().set_axis_off()
plt.tight_layout()

def animate(i):
    plot.set_array(energy[..., 0, i])
anim = FuncAnimation(plt.gcf(), animate, 30, interval=100)
plt.show()
