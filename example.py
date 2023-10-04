
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from fdtd4d import FDTD


fdtd = FDTD((20, 20, 1, 20), (1, 1, 0, 1))
fdtd.E_init[10, 10, 0, 10, 2] = 1
E, H = fdtd.run(50)

energy = np.sum(np.sum(E**2 + H**2, -1)**.5, 0)
plot = plt.imshow(energy[..., 0, 0],
                  cmap="hot",
                  vmin=0,
                  vmax=np.max(energy))
plt.axis("equal")
plt.gca().set_axis_off()
plt.tight_layout()

def animate(i):
    plot.set_array(energy[..., 0, i])
anim = FuncAnimation(plt.gcf(), animate, 20, interval=100)
plt.show()
