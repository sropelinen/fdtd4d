
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from fdtd4d import FDTD


fdtd = FDTD((50, 50, 1, 50), (1, 1, 0, 1))
fdtd.E_init[25, 25, 0, 25, 3] = 1
E, H = fdtd.run(50)

energy = np.sum(E**2 + H**2, -1)**.5
plot = plt.imshow(energy[..., 0],
                  cmap="hot",
                  vmin=0,
                  vmax=np.max(energy))
plt.axis("equal")
plt.gca().set_axis_off()
plt.tight_layout()

def animate(i):
    plot.set_array(energy[..., i])
anim = FuncAnimation(plt.gcf(), animate, 50, interval=100)
plt.show()
