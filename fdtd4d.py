
import numpy as np


class FDTD:

    def __init__(self, shape, boundary):

        # Floating point accuracy
        self.ftype = np.float32

        # Courant number
        self.cn = 0.5
        # Absorbing boundary thickness
        self.bt = 30

        # Shape of the measured fields
        self.shape = np.insert(np.array(shape, dtype=np.int16), 4, 4)
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

        # Initial conditions
        self.E_init = np.zeros(self.shape, dtype=self.ftype)
        self.H_init = np.zeros(self.shape, dtype=self.ftype)

    def run(self, steps):

        # "Electric" and "magnetic" fields
        self.E = [np.zeros(self.bshape, dtype=self.ftype)]
        self.H = [np.zeros(self.bshape, dtype=self.ftype)]
        # [var] allows passing var by reference

        # Apply initial conditions
        self.E[0][self.slice] = self.E_init
        self.H[0][self.slice] = self.H_init

        # Parameters for the absorbing boundary
        ds, loss = self._setup_boundary()

        # Run
        for s in range(steps):
            self.next_E(self.E[0], self.cn * ds * self.H[0])
            self.E[0] *= loss
            self.next_H(self.H[0], self.cn * ds * self.E[0])
            self.H[0] *= loss

        return self.E[0][self.slice], self.H[0][self.slice]

    def next_E(self, E, H):
        # Ds E = curl H - Dt H - grad Ht
        # Ds Et = div H - Dt Ht
        E[1:, :, :, :, 0] -= H[1:, :, :, :, 3] - H[:-1, :, :, :, 3]
        E[:, 1:, :, :, 0] += H[:, 1:, :, :, 2] - H[:, :-1, :, :, 2]
        E[:, :, 1:, :, 0] -= H[:, :, 1:, :, 1] - H[:, :, :-1, :, 1]
        E[1:, :, :, :, 1] -= H[1:, :, :, :, 2] - H[:-1, :, :, :, 2]
        E[:, 1:, :, :, 1] -= H[:, 1:, :, :, 3] - H[:, :-1, :, :, 3]
        E[:, :, 1:, :, 1] += H[:, :, 1:, :, 0] - H[:, :, :-1, :, 0]
        E[1:, :, :, :, 2] += H[1:, :, :, :, 1] - H[:-1, :, :, :, 1]
        E[:, 1:, :, :, 2] -= H[:, 1:, :, :, 0] - H[:, :-1, :, :, 0]
        E[:, :, 1:, :, 2] -= H[:, :, 1:, :, 3] - H[:, :, :-1, :, 3]
        E[1:, :, :, :, 3] += H[1:, :, :, :, 0] - H[:-1, :, :, :, 0]
        E[:, 1:, :, :, 3] += H[:, 1:, :, :, 1] - H[:, :-1, :, :, 1]
        E[:, :, 1:, :, 3] += H[:, :, 1:, :, 2] - H[:, :, :-1, :, 2]
        E[:, :, :, 1:] -= H[:, :, :, 1:] - H[:, :, :, :-1]

    def next_H(self, H, E):
        # Ds H = -curl E - Dt E + grad Et
        # Ds Ht = -div E - Dt Et
        H[:-1, :, :, :, 0] += E[1:, :, :, :, 3] - E[:-1, :, :, :, 3]
        H[:, :-1, :, :, 0] -= E[:, 1:, :, :, 2] - E[:, :-1, :, :, 2]
        H[:, :, :-1, :, 0] += E[:, :, 1:, :, 1] - E[:, :, :-1, :, 1]
        H[:-1, :, :, :, 1] += E[1:, :, :, :, 2] - E[:-1, :, :, :, 2]
        H[:, :-1, :, :, 1] += E[:, 1:, :, :, 3] - E[:, :-1, :, :, 3]
        H[:, :, :-1, :, 1] -= E[:, :, 1:, :, 0] - E[:, :, :-1, :, 0]
        H[:-1, :, :, :, 2] -= E[1:, :, :, :, 1] - E[:-1, :, :, :, 1]
        H[:, :-1, :, :, 2] += E[:, 1:, :, :, 0] - E[:, :-1, :, :, 0]
        H[:, :, :-1, :, 2] += E[:, :, 1:, :, 3] - E[:, :, :-1, :, 3]
        H[:-1, :, :, :, 3] -= E[1:, :, :, :, 0] - E[:-1, :, :, :, 0]
        H[:, :-1, :, :, 3] -= E[:, 1:, :, :, 1] - E[:, :-1, :, :, 1]
        H[:, :, :-1, :, 3] -= E[:, :, 1:, :, 2] - E[:, :, :-1, :, 2]
        H[:, :, :, :-1] -= E[:, :, :, 1:] - E[:, :, :, :-1]
        return H

    def _setup_boundary(self):

        # Measured area is 1, boundaries go linearly to 0
        linear = np.pad(np.ones(self.shape[:-1][self.boundary],
                                dtype=self.ftype),
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
