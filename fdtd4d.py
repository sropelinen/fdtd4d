
import numpy as np


class FDTD:

    def __init__(self, shape, boundary):

        # Courant number
        self.cn = 0.5
        # Absorbing boundary thickness
        self.bt = 30

        # Shape of the measured fields
        self.shape = np.insert(np.array(shape), 4, 4)
        # Shape of the fdtd grid
        self.bshape = np.copy(self.shape)
        self.boundary = np.array(boundary, dtype=bool)
        self.bshape[:-1] += self.boundary * 2 * self.bt

        # Slice of the measured part of the fdtd grid
        a = [slice(self.bt, -self.bt)] * 4
        for i in range(4):
            if not self.boundary[i]:
                a[i] = slice(None)
        self.slice = tuple(a)

        # Initial conditions for the fields
        self.E_init = np.zeros(self.shape)
        self.H_init = np.zeros(self.shape)

    def run(self, steps):

        # "Electric" field
        self.E = np.zeros(np.insert(self.bshape, 0, steps + 1))
        # "Magnetic" field
        self.H = np.zeros(np.insert(self.bshape, 0, steps + 1))

        # Insert initial conditions
        self.E[(0,) + self.slice] += self.E_init
        self.H[(0,) + self.slice] += self.H_init

        # Parameters for the absorbing boundary
        ds, loss = self._setup_boundary()

        # Run
        for t in range(steps):
            self.E[t + 1] = loss * (self.E[t]
                                    + self.cn * ds * self.dE(self.H[t]))
            self.H[t + 1] = loss * (self.H[t]
                                    + self.cn * ds * self.dH(self.E[t + 1]))

        # Return the measured area of the fields
        return self.E[(slice(None),) + self.slice], \
            self.H[(slice(None),) + self.slice]

    def dE(self, H):
        # Ds E = curl H - Dt H - grad Ht
        # Ds Et = div H - Dt Ht
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
        # Ds H = -curl E - Dt E + grad Et
        # Ds Ht = -div E - Dt Et
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

    def _setup_boundary(self):

        # Measured area is 1, boundaries go linearly to 0
        linear = np.pad(np.ones(self.shape[:-1][self.boundary]),
                        self.bt, "linear_ramp")
        for i in range(4):
            if not self.boundary[i]:
                a = [slice(None)] * (i + 1)
                a[i] = None
                linear = linear[tuple(a)]
        linear = np.repeat(linear[..., None], 4, -1)

        ds = 1 - (1 - linear)**2
        loss = 1 - (1 - linear)**2 * 0.1

        # Set edges to zero for stability
        for i in range(4):
            if self.boundary[i]:
                a = [slice(None)] * 4
                a[i] = 0
                loss[tuple(a)] = 0
                a[i] = -1
                loss[tuple(a)] = 0

        return ds, loss
