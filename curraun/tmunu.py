"""
    A model for computing all components of the energy momentum tensor as a function of the transverse
    coordinate at some fixed proper time. This can be used to perform Landau matching to hydrodynamics
    e.g. MUSIC.
"""

from curraun.numba_target import myjit, prange, my_parallel_loop, use_cuda
from numba import jit
import numpy as np
import curraun.lattice as l
import curraun.su as su
import curraun.kappa as kappa
if use_cuda:
    import numba.cuda as cuda


class EnergyMomentumTensor:
    def __init__(self, s):
        self.s = s
        self.n = s.n

        # 10 independent components of T_\mu\nu in the following order
        # diagonal, energy flux, shear
        # T00, T11, T22, T33, T01, T02, T03, T12, T13, T23
        # where 0 corresponds to \tau, 1,2 correspond to transverse coordinates and 3 is \eta.
        self.t_munu = np.zeros((self.n ** 2, 10), dtype=su.GROUP_TYPE_REAL)

        self.d_t_munu = self.t_munu

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_t_munu = cuda.to_device(self.t_munu)

    def copy_to_host(self):
        self.d_t_munu.copy_to_host(self.t_munu)
        self.t_munu /= self.s.g ** 2

    def compute(self):
        # compute contributions in 2d
        u0 = self.s.d_u0

        aeta0 = self.s.d_aeta0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_parallel_loop(tmunu_kernel, n ** 2, n, u0, aeta0, peta1, peta0, pt1, pt0, t, self.d_t_munu)


# kernels
@myjit
def tmunu_kernel(xi, n, u0, aeta0, peta1, peta0, pt1, pt0, tau, t_munu):
    # Compute correctly averaged field strength components
    # Electric components: spatial and temporal (for Ex, Ey) and temporal (Ez)
    # Magnetic components: only spatial averaging (one direction for Bx, By, two for Bz)
    i = 0
    Ex = su.zero()
    Ex = su.add(Ex, pt1[xi, i])
    Ex = su.add(Ex, pt0[xi, i])
    xs = l.shift(xi, i, -1, n)
    b1 = l.act(su.dagger(u0[xs, i]), pt1[xs, i])
    Ex = su.add(Ex, b1)
    b1 = l.act(su.dagger(u0[xs, i]), pt0[xs, i])
    Ex = su.add(Ex, b1)
    Ex = su.mul_s(Ex, 0.25 / tau)
    
    i = 1
    Ey = su.zero()
    Ey = su.add(Ey, pt1[xi, i])
    Ey = su.add(Ey, pt0[xi, i])
    xs = l.shift(xi, i, -1, n)
    b1 = l.act(su.dagger(u0[xs, i]), pt1[xs, i])
    Ey = su.add(Ey, b1)
    b1 = l.act(su.dagger(u0[xs, i]), pt0[xs, i])
    Ey = su.add(Ey, b1)
    Ey = su.mul_s(Ey, 0.25 / tau)

    Ez = su.zero()
    Ez = l.add_mul(Ez, peta1[xi], 0.5)
    Ez = l.add_mul(Ez, peta0[xi], 0.5)
    # Ez = su.mul_s(Ez, tau)

    b1 = l.transport(aeta0, u0, xi, 1, +1, n)
    b2 = l.transport(aeta0, u0, xi, 1, -1, n)
    b2 = l.add_mul(b1, b2, -1.0)
    Bx = su.mul_s(b2, -0.5 / tau) # check these signs

    b1 = l.transport(aeta0, u0, xi, 0, +1, n)
    b2 = l.transport(aeta0, u0, xi, 0, -1, n)
    b2 = l.add_mul(b1, b2, -1.0)
    By = su.mul_s(b2, +0.5 / tau) # check these signs

    bf1 = su.zero()
    b1 = l.plaq(u0, xi, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, xi, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xi, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, xi, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    Bz = l.add_mul(bf1, b2, +0.25)

    # 0-3: Diagonal components
    eEx = dot(Ex, Ex)
    eEy = dot(Ey, Ey)
    eEz = dot(Ez, Ez)

    eBx = dot(Bx, Bx)
    eBy = dot(By, By)
    eBz = dot(Bz, Bz)

    # 0: Energy density
    t_munu[xi, 0] = eEx + eEy + eEz + eBx + eBy + eBz

    # 1: Transverse pressure x
    t_munu[xi, 1] = (- eEx + eEy + eEz - eBx + eBy + eBz)

    # 2: Transverse pressure y
    t_munu[xi, 2] = (+ eEx - eEy + eEz + eBx - eBy + eBz)

    # 3: Longitudinal pressure
    t_munu[xi, 3] = tau ** 2 * (eEx + eEy - eEz + eBx + eBy - eBz)

    # 4-6: Energy flux
    # 4: S_x
    t_munu[xi, 4] = 2*(dot(Ey, Bz) - dot(Ez, By))

    # 5: S_y
    t_munu[xi, 5] = 2*(dot(Ez, Bx) - dot(Ex, Bz))

    # 6: S_\eta
    t_munu[xi, 6] = 2*(dot(Ex, By) - dot(Ey, Bx)) * tau

    # 6-10: Shear stress
    # 7: s_xy
    t_munu[xi, 7] = 2*(- dot(Ex, Ey) - dot(Bx, By))

    # 8: s_xz
    t_munu[xi, 8] = 2*(- dot(Ex, Ez) - dot(Bx, Bz)) * tau

    # 9: s_yz
    t_munu[xi, 9] = 2*(- dot(Ey, Ez) - dot(By, Bz)) * tau


@myjit
def dot(a0, a1):
    return su.tr(su.mul(a0, su.dagger(a1))).real


def convert_to_matrix(t_munu, tau):
    n = int(np.sqrt(t_munu.shape[0]))
    t_munu = t_munu.reshape(n, n, 10)
    t_matrix = np.zeros((n, n, 4, 4))

    # fill non-diagonal elements
    t_matrix[:, :, 0, 1] = t_munu[:, :, 4]
    t_matrix[:, :, 0, 2] = t_munu[:, :, 5]
    t_matrix[:, :, 0, 3] = t_munu[:, :, 6]

    t_matrix[:, :, 1, 2] = t_munu[:, :, 7]
    t_matrix[:, :, 1, 3] = t_munu[:, :, 8]
    t_matrix[:, :, 2, 3] = t_munu[:, :, 9]

    # fill the other diagonal
    t_matrix = t_matrix + np.transpose(t_matrix, axes=[0, 1, 3, 2])

    # fill in diagonal
    t_matrix[:, :, 0, 0] = t_munu[:, :, 0]
    t_matrix[:, :, 1, 1] = t_munu[:, :, 1]
    t_matrix[:, :, 2, 2] = t_munu[:, :, 2]
    t_matrix[:, :, 3, 3] = t_munu[:, :, 3]

    # Pull up both indices
    t_matrix[:, :, 0, :] *= +1.0
    t_matrix[:, :, 1, :] *= -1.0
    t_matrix[:, :, 2, :] *= -1.0
    t_matrix[:, :, 3, :] *= -1.0 / tau ** 2

    t_matrix[:, :, :, 0] *= +1.0
    t_matrix[:, :, :, 1] *= -1.0
    t_matrix[:, :, :, 2] *= -1.0
    t_matrix[:, :, :, 3] *= -1.0 / tau ** 2

    return t_matrix


"""
    Landau matching 
"""


def landau(t_matrix):
    d = 4
    n = t_matrix.shape[0]
    t_matrix = t_matrix.reshape((n * n, d, d))
    w, v = np.linalg.eig(t_matrix)

    w = w.reshape((n, n, d))
    v = v.reshape((n, n, d, d))

    return w, v
