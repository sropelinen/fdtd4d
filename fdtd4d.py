
import numpy as np


class FDTD:

    def __init__(self, shape):
        self.shape = np.array(shape + (4,))

        self.cn = 0.5

        self.E_init = np.zeros(self.shape)
        self.H_init = np.zeros(self.shape)

    def run(self, steps):
        self.E = np.zeros(np.insert(self.shape, 0, steps + 1))
        self.H = np.zeros(np.insert(self.shape, 0, steps + 1))
        self.E[0] += self.E_init
        self.H[0] += self.H_init

        c, a, border = self._get_boundary(20, 2, 0.1)

        for t in range(steps):
            self.E[t + 1] = self.E[t] + self.cn * self.dE(self.H[t]) * c
            self.E[t + 1] *= a
            self.E[t + 1][border] = 0
            self.H[t + 1] = self.H[t] + self.cn * self.dH(self.E[t + 1]) * c
            self.H[t + 1] *= a
            self.H[t + 1][border] = 0

        return self.E, self.H

    def dE(self, H):
        d = np.zeros(H.shape)
        d[1:, :, :, :, 0] -= H[1:, :, :, :, 3] - H[:-1, :, :, :, 3]
        d[:, 1:, :, :, 0] += H[:, 1:, :, :, 2] - H[:, :-1, :, :, 2]
        d[:, :, 1:, :, 0] -= H[:, :, 1:, :, 1] - H[:, :, :-1, :, 1]
        d[:, :, :, 1:, 0] -= H[:, :, :, 1:, 0] - H[:, :, :, :-1, 0]
        d[1:, :, :, :, 1] -= H[1:, :, :, :, 2] - H[:-1, :, :, :, 2]
        d[:, 1:, :, :, 1] -= H[:, 1:, :, :, 3] - H[:, :-1, :, :, 3]
        d[:, :, 1:, :, 1] += H[:, :, 1:, :, 0] - H[:, :, :-1, :, 0]
        d[:, :, :, 1:, 1] -= H[:, :, :, 1:, 1] - H[:, :, :, :-1, 1]
        d[1:, :, :, :, 2] += H[1:, :, :, :, 1] - H[:-1, :, :, :, 1]
        d[:, 1:, :, :, 2] -= H[:, 1:, :, :, 0] - H[:, :-1, :, :, 0]
        d[:, :, 1:, :, 2] -= H[:, :, 1:, :, 3] - H[:, :, :-1, :, 3]
        d[:, :, :, 1:, 2] -= H[:, :, :, 1:, 2] - H[:, :, :, :-1, 2]
        d[1:, :, :, :, 3] += H[1:, :, :, :, 0] - H[:-1, :, :, :, 0]
        d[:, 1:, :, :, 3] += H[:, 1:, :, :, 1] - H[:, :-1, :, :, 1]
        d[:, :, 1:, :, 3] += H[:, :, 1:, :, 2] - H[:, :, :-1, :, 2]
        d[:, :, :, 1:, 3] -= H[:, :, :, 1:, 3] - H[:, :, :, :-1, 3]
        return d

    def dH(self, E):
        d = np.zeros(E.shape)
        d[:-1, :, :, :, 0] += E[1:, :, :, :, 3] - E[:-1, :, :, :, 3]
        d[:, :-1, :, :, 0] -= E[:, 1:, :, :, 2] - E[:, :-1, :, :, 2]
        d[:, :, :-1, :, 0] += E[:, :, 1:, :, 1] - E[:, :, :-1, :, 1]
        d[:, :, :, :-1, 0] -= E[:, :, :, 1:, 0] - E[:, :, :, :-1, 0]
        d[:-1, :, :, :, 1] += E[1:, :, :, :, 2] - E[:-1, :, :, :, 2]
        d[:, :-1, :, :, 1] += E[:, 1:, :, :, 3] - E[:, :-1, :, :, 3]
        d[:, :, :-1, :, 1] -= E[:, :, 1:, :, 0] - E[:, :, :-1, :, 0]
        d[:, :, :, :-1, 1] -= E[:, :, :, 1:, 1] - E[:, :, :, :-1, 1]
        d[:-1, :, :, :, 2] -= E[1:, :, :, :, 1] - E[:-1, :, :, :, 1]
        d[:, :-1, :, :, 2] += E[:, 1:, :, :, 0] - E[:, :-1, :, :, 0]
        d[:, :, :-1, :, 2] += E[:, :, 1:, :, 3] - E[:, :, :-1, :, 3]
        d[:, :, :, :-1, 2] -= E[:, :, :, 1:, 2] - E[:, :, :, :-1, 2]
        d[:-1, :, :, :, 3] -= E[1:, :, :, :, 0] - E[:-1, :, :, :, 0]
        d[:, :-1, :, :, 3] -= E[:, 1:, :, :, 1] - E[:, :-1, :, :, 1]
        d[:, :, :-1, :, 3] -= E[:, :, 1:, :, 2] - E[:, :, :-1, :, 2]
        d[:, :, :, :-1, 3] -= E[:, :, :, 1:, 3] - E[:, :, :, :-1, 3]
        return d

    def _get_boundary(self, w, p, a):
        border = np.zeros(self.shape, dtype=bool)
        if self.shape[0] > 2:
            border[0] = 1
            border[-1] = 1
        if self.shape[1] > 2:
            border[:, 0] = 1
            border[:, -1] = 1
        if self.shape[2] > 2:
            border[:, 0] = 1
            border[:, -1] = 1
        if self.shape[3] > 2:
            border[:, 0] = 1
            border[:, -1] = 1

        d = self.shape[:-1] > w * 2
        s = np.pad(np.ones(self.shape[:-1][d] - w * 2), w, "linear_ramp")
        if not d[0]:
            s = s[None]
        if not d[1]:
            s = s[:, None]
        if not d[2]:
            s = s[:, :, None]
        if not d[3]:
            s = s[:, :, :, None]
        s = np.repeat(s[..., None], 4, -1)
        c = 1 - (1 - s) ** p

        return c, 1 + (c - 1) * a, border
