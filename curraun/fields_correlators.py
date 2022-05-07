"""
    Computation of the Glasma electric and magnetic fields correlators at different times, averaged over the transverse plane.
"""

from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda


class FieldsCorrelators:
    def __init__(self, s):
        self.s = s
        self.n = s.n

        self.E = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_E = self.E

        self.EformE = np.zeros((self.n ** 2, 3), dtype=su.GROUP_TYPE_REAL)
        self.d_EformE = self.EformE

        self.B = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_B = self.B

        self.BformB = np.zeros((self.n ** 2, 3), dtype=su.GROUP_TYPE_REAL)
        self.d_BformB = self.BformB

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_EformE = cuda.to_device(self.EformE)
        self.d_BformB = cuda.to_device(self.BformB)

    def copy_to_host(self):
        self.d_EformE.copy_to_host(self.EformE)
        self.d_BformB.copy_to_host(self.BformB)
        # self.EformE /= self.s.g ** 2
        # self.BformB /= self.s.g ** 2

    def compute_elfield(self):
        u0 = self.s.d_u0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_parallel_loop(compute_E, n ** 2, n, u0, peta1, peta0, pt1, pt0, t, self.d_E)

        if use_cuda:
            self.copy_to_host()

        return self.d_E

    def compute_magfield(self):
        u0 = self.s.d_u0
        aeta0 = self.s.d_aeta0

        t = self.s.t
        n = self.n

        my_parallel_loop(compute_B, n ** 2, n, u0, aeta0, t, self.d_B)

        if use_cuda:
            self.copy_to_host()

        return self.d_B

    def compute_elcorr(self, Eform):
        u0 = self.s.d_u0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_parallel_loop(compute_EformE_kernel, n**2, n, t, u0, pt1, pt0, peta1, peta0, self.d_E, Eform, self.d_EformE)

        if use_cuda:
            self.copy_to_host()
    
        return np.mean(self.EformE, axis=0)

    def compute_magcorr(self, Bform):
        u0 = self.s.d_u0
        aeta0 = self.s.d_aeta0

        t = self.s.t
        n = self.n

        my_parallel_loop(compute_BformB_kernel, n**2, n, t, u0, aeta0, self.d_B, Bform, self.d_BformB)

        if use_cuda:
            self.copy_to_host()
    
        return np.mean(self.BformB, axis=0)

# kernels
@myjit
def compute_E(xi, n, u0, peta1, peta0, pt1, pt0, tau, E):

    for i in range(2):
        Ei = su.zero()
        Ei = su.add(Ei, pt1[xi, i])
        Ei = su.add(Ei, pt0[xi, i])
        xs = l.shift(xi, i, -1, n)
        b1 = l.act(su.dagger(u0[xs, i]), pt1[xs, i])
        Ei = su.add(Ei, b1)
        b1 = l.act(su.dagger(u0[xs, i]), pt0[xs, i])
        Ei = su.add(Ei, b1)
        Ei = su.mul_s(Ei, 0.25 / tau)

        su.store(E[xi, i], Ei)
    
    Ez = su.zero()
    Ez = l.add_mul(Ez, peta1[xi], 0.5)
    Ez = l.add_mul(Ez, peta0[xi], 0.5)

    su.store(E[xi, 2], Ez)

    return E

@myjit
def compute_EformE_kernel(xi, n, t, u0, pt1, pt0, peta1, peta0, E, Eform, EformE):

    E = compute_E(xi, n, u0, peta1, peta0, pt1, pt0, t, E)

    for d in range(3):
        EformE[xi, d] = su.tr(su.mul(Eform[xi, d], su.dagger(E[xi, d]))).real

@myjit
def compute_B(xi, n, u0, aeta0, tau, B):

    for i in range(2):
        b1 = l.transport(aeta0, u0, xi, (i+1)%2, +1, n)
        b2 = l.transport(aeta0, u0, xi, (i+1)%2, -1, n)
        b2 = l.add_mul(b1, b2, -1.0)
        Bi = su.mul_s(b2, -0.5 / tau)

        su.store(B[xi, i], Bi)

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

    su.store(B[xi, 2], Bz)

    return B

@myjit
def compute_BformB_kernel(xi, n, t, u0, aeta0, B, Bform, BformB):

    B = compute_B(xi, n, u0, aeta0, t, B)

    for d in range(3):
        BformB[xi, d] = su.tr(su.mul(Bform[xi, d], su.dagger(B[xi, d]))).real