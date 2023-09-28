
import numpy as np


class BC:

    def __init__(self, *args):
        return

    def init(self, parent):
        return

    def pre_update_E(self, H):
        return

    def pre_update_H(self, E):
        return

    def post_update_E(self, E):
        return E

    def post_update_H(self, H):
        return H


class PBC(BC):

    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def post_update_E(self, E):
        if self.x:
            E[0] = E[-1]
        if self.y:
            E[:, 0] = E[:, -1]
        if self.z:
            E[:, :, 0] = E[:, :, -1]
        if self.t:
            E[:, :, :, 0] = E[:, :, :, -1]
        return E

    def post_update_H(self, H):
        if self.x:
            H[-1] = H[0]
        if self.y:
            H[:, -1] = H[:, 0]
        if self.z:
            H[:, :, -1] = H[:, :, 0]
        if self.t:
            H[:, :, :, -1] = H[:, :, :, 0]
        return H


class ABC(BC):

    def __init__(self, x, y, z, t):
        self.boundaries = []
        for h in range(2):
            if x:
                self.boundaries.append(_ABC(h, x=x))
            if y:
                self.boundaries.append(_ABC(h, y=y))
            if z:
                self.boundaries.append(_ABC(h, z=z))
            if t:
                self.boundaries.append(_ABC(h, t=t))

    def init(self, parent):
        for b in self.boundaries:
            b.init(parent)

    def pre_update_E(self, H):
        for b in self.boundaries:
            b.pre_update_E(H)

    def pre_update_H(self, E):
        for b in self.boundaries:
            b.pre_update_H(E)

    def post_update_E(self, E):
        for b in self.boundaries:
            E = b.post_update_E(E)
        return E

    def post_update_H(self, H):
        for b in self.boundaries:
            H = b.post_update_H(H)
        return H


class _ABC(BC):

    def __init__(self, high, x=0, y=0, z=0, t=0):
        assert (x > 0) + (y > 0) + (z > 0) + (t > 0) == 1
        self.w = x + y + z + t
        self.high = high
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def init(self, parent):
        self.cn = parent._cn
        self.shape = parent.shape
        
        self.mask = np.zeros(self.shape + (4,))
        self.sigmaE = np.zeros(self.shape + (4,))
        self.sigmaH = np.zeros(self.shape + (4,))

        self.phi_E = np.zeros(self.shape + (4,))
        self.phi_H = np.zeros(self.shape + (4,))
        self.psi_E = np.zeros((4,) + self.shape + (4,))
        self.psi_H = np.zeros((4,) + self.shape + (4,))

        if self.high:
            s1 = self._sigma(0.5, self.w + 0.5, 1, self.w)
            s2 = self._sigma(1, self.w, 1, self.w)
            if self.x:
                self.mask[-self.w:] = 1
                self.sigmaE[-self.w:, :, :, :, 0] = s1[:, None, None, None]
                self.sigmaH[-self.w:-1, :, :, :, 0] = s2[:, None, None, None]
            elif self.y:
                self.mask[:, -self.w:] = 1
                self.sigmaE[:, -self.w:, :, :,  1] = s1[None, :, None, None]
                self.sigmaH[:, -self.w:-1, :, :, 1] = s2[None, :, None, None]
            elif self.z:
                self.mask[:, :, -self.w:] = 1
                self.sigmaE[:, :, -self.w:, :, 2] = s1[None, None, :, None]
                self.sigmaH[:, :, -self.w:-1, :, 2] = s2[None, None, :, None]
            elif self.t:
                self.mask[:, :, :, -self.w:] = 1
                self.sigmaE[:, :, :, -self.w:, 3] = s1[None, None, None, :]
                self.sigmaH[:, :, :, -self.w:-1, 3] = s2[None, None, None, :]
        else:
            s1 = self._sigma(self.w - 0.5, -0.5, -1, self.w)
            s2 = self._sigma(self.w - 1, 0, -1, self.w)
            if self.x:
                self.mask[:self.w] = 1
                self.sigmaE[:self.w, :, :, :, 0] = s1[:, None, None, None]
                self.sigmaH[:self.w - 1, :, :, :, 0] = s2[:, None, None, None]
            elif self.y:
                self.mask[:, :self.w] = 1
                self.sigmaE[:, :self.w, :, :, 1] = s1[None, :, None, None]
                self.sigmaH[:, :self.w - 1, :, :, 1] = s2[None, :, None, None]
            elif self.z:
                self.mask[:, :, :self.w] = 1
                self.sigmaE[:, :, :self.w, :, 2] = s1[None, None, :, None]
                self.sigmaH[:, :, :self.w - 1, :, 2] = s2[None, None, :, None]
            elif self.t:
                self.mask[:, :, :self.w] = 1
                self.sigmaE[:, :, :, :self.w, 3] = s1[None, None, None, :]
                self.sigmaH[:, :, :, :self.w - 1, 3] = s2[None, None, None, :]

        self.bE = np.exp(-(self.sigmaE + 1e-8) * self.cn) * self.mask
        self.cE = (self.bE - 1) * self.sigmaE / (self.sigmaE + 1e-8)
        self.bH = np.exp(-(self.sigmaH + 1e-8) * self.cn) * self.mask
        self.cH = (self.bH - 1) * self.sigmaH / (self.sigmaH + 1e-8)

    def pre_update_E(self, H):
        self.psi_E *= self.bE
        c = self.cE
        Hm = H * self.mask

        self.psi_E[0, 1:, :, :, :, 0] += (Hm[1:, :, :, :, 3] - Hm[:-1, :, :, :, 3]) * c[1:, :, :, :, 0]
        self.psi_E[0, :, 1:, :, :, 1] += (Hm[:, 1:, :, :, 2] - Hm[:, :-1, :, :, 2]) * c[:, 1:, :, :, 1]
        self.psi_E[0, :, :, 1:, :, 2] += (Hm[:, :, 1:, :, 1] - Hm[:, :, :-1, :, 1]) * c[:, :, 1:, :, 2]
        self.psi_E[0, :, :, :, 1:, 3] += (Hm[:, :, :, 1:, 0] - Hm[:, :, :, :-1, 0]) * c[:, :, :, 1:, 3]
        self.psi_E[1, 1:, :, :, :, 0] += (Hm[1:, :, :, :, 2] - Hm[:-1, :, :, :, 2]) * c[1:, :, :, :, 0]
        self.psi_E[1, :, 1:, :, :, 1] += (Hm[:, 1:, :, :, 3] - Hm[:, :-1, :, :, 3]) * c[:, 1:, :, :, 1]
        self.psi_E[1, :, :, 1:, :, 2] += (Hm[:, :, 1:, :, 0] - Hm[:, :, :-1, :, 0]) * c[:, :, 1:, :, 2]
        self.psi_E[1, :, :, :, 1:, 3] += (Hm[:, :, :, 1:, 1] - Hm[:, :, :, :-1, 1]) * c[:, :, :, 1:, 3]
        self.psi_E[2, 1:, :, :, :, 0] += (Hm[1:, :, :, :, 1] - Hm[:-1, :, :, :, 1]) * c[1:, :, :, :, 0]
        self.psi_E[2, :, 1:, :, :, 1] += (Hm[:, 1:, :, :, 0] - Hm[:, :-1, :, :, 0]) * c[:, 1:, :, :, 1]
        self.psi_E[2, :, :, 1:, :, 2] += (Hm[:, :, 1:, :, 3] - Hm[:, :, :-1, :, 3]) * c[:, :, 1:, :, 2]
        self.psi_E[2, :, :, :, 1:, 3] += (Hm[:, :, :, 1:, 2] - Hm[:, :, :, :-1, 2]) * c[:, :, :, 1:, 3]
        self.psi_E[3, 1:, :, :, :, 0] += (Hm[1:, :, :, :, 0] - Hm[:-1, :, :, :, 0]) * c[1:, :, :, :, 0]
        self.psi_E[3, :, 1:, :, :, 1] += (Hm[:, 1:, :, :, 1] - Hm[:, :-1, :, :, 1]) * c[:, 1:, :, :, 1]
        self.psi_E[3, :, :, 1:, :, 2] += (Hm[:, :, 1:, :, 2] - Hm[:, :, :-1, :, 2]) * c[:, :, 1:, :, 2]
        self.psi_E[3, :, :, :, 1:, 3] += (Hm[:, :, :, 1:, 3] - Hm[:, :, :, :-1, 3]) * c[:, :, :, 1:, 3]

        self.phi_E[..., 0] = -self.psi_E[0, ..., 0] + self.psi_E[0, ..., 1] - self.psi_E[0, ..., 2] - self.psi_E[0, ..., 3]
        self.phi_E[..., 1] = -self.psi_E[1, ..., 0] - self.psi_E[1, ..., 1] + self.psi_E[1, ..., 2] - self.psi_E[1, ..., 3]
        self.phi_E[..., 2] =  self.psi_E[2, ..., 0] - self.psi_E[2, ..., 1] - self.psi_E[2, ..., 2] - self.psi_E[2, ..., 3]
        self.phi_E[..., 3] =  self.psi_E[3, ..., 0] + self.psi_E[3, ..., 1] + self.psi_E[3, ..., 2] - self.psi_E[3, ..., 3]

    def pre_update_H(self, E):
        self.psi_H *= self.bH
        c = self.cH
        Em = E * self.mask

        self.psi_H[0, :-1, :, :, :, 0] += (Em[1:, :, :, :, 3] - Em[:-1, :, :, :, 3]) * c[:-1, :, :, :, 0]
        self.psi_H[0, :, :-1, :, :, 1] += (Em[:, 1:, :, :, 2] - Em[:, :-1, :, :, 2]) * c[:, :-1, :, :, 1]
        self.psi_H[0, :, :, :-1, :, 2] += (Em[:, :, 1:, :, 1] - Em[:, :, :-1, :, 1]) * c[:, :, :-1, :, 2]
        self.psi_H[0, :, :, :, :-1, 3] += (Em[:, :, :, 1:, 0] - Em[:, :, :, :-1, 0]) * c[:, :, :, :-1, 3]
        self.psi_H[1, :-1, :, :, :, 0] += (Em[1:, :, :, :, 2] - Em[:-1, :, :, :, 2]) * c[:-1, :, :, :, 0]
        self.psi_H[1, :, :-1, :, :, 1] += (Em[:, 1:, :, :, 3] - Em[:, :-1, :, :, 3]) * c[:, :-1, :, :, 1]
        self.psi_H[1, :, :, :-1, :, 2] += (Em[:, :, 1:, :, 0] - Em[:, :, :-1, :, 0]) * c[:, :, :-1, :, 2]
        self.psi_H[1, :, :, :, :-1, 3] += (Em[:, :, :, 1:, 1] - Em[:, :, :, :-1, 1]) * c[:, :, :, :-1, 3]
        self.psi_H[2, :-1, :, :, :, 0] += (Em[1:, :, :, :, 1] - Em[:-1, :, :, :, 1]) * c[:-1, :, :, :, 0]
        self.psi_H[2, :, :-1, :, :, 1] += (Em[:, 1:, :, :, 0] - Em[:, :-1, :, :, 0]) * c[:, :-1, :, :, 1]
        self.psi_H[2, :, :, :-1, :, 2] += (Em[:, :, 1:, :, 3] - Em[:, :, :-1, :, 3]) * c[:, :, :-1, :, 2]
        self.psi_H[2, :, :, :, :-1, 3] += (Em[:, :, :, 1:, 2] - Em[:, :, :, :-1, 2]) * c[:, :, :, :-1, 3]
        self.psi_H[3, :-1, :, :, :, 0] += (Em[1:, :, :, :, 0] - Em[:-1, :, :, :, 0]) * c[:-1, :, :, :, 0]
        self.psi_H[3, :, :-1, :, :, 1] += (Em[:, 1:, :, :, 1] - Em[:, :-1, :, :, 1]) * c[:, :-1, :, :, 1]
        self.psi_H[3, :, :, :-1, :, 2] += (Em[:, :, 1:, :, 2] - Em[:, :, :-1, :, 2]) * c[:, :, :-1, :, 2]
        self.psi_H[3, :, :, :, :-1, 3] += (Em[:, :, :, 1:, 3] - Em[:, :, :, :-1, 3]) * c[:, :, :, :-1, 3]

        self.phi_H[..., 0] =  self.psi_H[0, ..., 0] - self.psi_H[0, ..., 1] + self.psi_H[0, ..., 2] - self.psi_H[0, ..., 3]
        self.phi_H[..., 1] =  self.psi_H[1, ..., 0] + self.psi_H[1, ..., 1] - self.psi_H[1, ..., 2] - self.psi_H[1, ..., 3]
        self.phi_H[..., 2] = -self.psi_H[2, ..., 0] + self.psi_H[2, ..., 1] + self.psi_H[2, ..., 2] - self.psi_H[2, ..., 3]
        self.phi_H[..., 3] = -self.psi_H[3, ..., 0] - self.psi_H[3, ..., 1] - self.psi_H[3, ..., 2] - self.psi_H[3, ..., 3]

    def post_update_E(self, E):
        return E + self.cn * self.phi_E

    def post_update_H(self, H):
        return H + self.cn * self.phi_H
    
    def _sigma(self, a, b, c, t):
        return 40 * np.arange(a, b, c)**3 / (t + 1)**4
