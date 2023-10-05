
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from fdtd4d import FDTD


fdtd = FDTD((50, 50, 1, 50), (1, 1, 0, 1))

X, Y = np.meshgrid(np.arange(50), np.arange(50))
fdtd.E_init[:, :, 0, 25] = np.exp(-((X - 25)**2 + (Y - 25)**2) / 5)[..., None]

E, H = fdtd.run(200)

energy = np.sum(E**2, -1)**.5 + np.sum(H**2, -1)**.5
energy = np.sum(energy, 0)

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
