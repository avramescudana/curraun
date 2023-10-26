from curraun.numba_target import myjit, prange, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
import os

NC = su.NC

if use_cuda:
    import numba.cuda as cuda

"""
    A module for computing energy density components
"""

# set precision of variable
su_precision = os.environ.get('PRECISION', 'double')
if su_precision == 'single':
    DTYPE = np.float32
elif su_precision == 'double':
    DTYPE = np.float64


class Energy():
    def __init__(self, s):
        self.s = s

        self.E = np.zeros((s.n ** 2, 3), dtype=DTYPE)
        self.B = np.zeros((s.n ** 2, 3), dtype=DTYPE)

        self.d_E = self.E
        self.d_B = self.B

        if use_cuda:
            self.copy_to_device()

        self.E_mean = np.zeros(3)
        self.B_mean = np.zeros(3)

        # self.energy_density = 0.0
        # self.pL = 0.0
        # self.pT = 0.0

    def copy_to_device(self):
        self.d_E = cuda.to_device(self.E)
        self.d_B = cuda.to_device(self.B)

    def copy_to_host(self):
        self.d_E.copy_to_host(self.E)
        self.d_B.copy_to_host(self.B)

    def compute(self):
        # compute contributions in 2d
        u0 = self.s.d_u0
        # u1 = self.s.d_u1
        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1
        aeta0 = self.s.d_aeta0
        # aeta1 = self.s.d_aeta1
        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        # dt = self.s.dt
        # dth = dt / 2.0
        t = self.s.t

        E = self.d_E
        B = self.d_B

        n = self.s.n

        my_parallel_loop(fields_kernel, n ** 2, n, u0, pt0, pt1, aeta0, peta0, peta1, t, E, B)

        # compute means (very inefficient, but want to keep the arrays intact)
        if use_cuda:
            self.copy_to_host()

        self.E_mean = np.mean(E, axis=0) 
        self.B_mean = np.mean(B, axis=0) 

        # # compute density and pressures
        # self.energy_density = (self.EL_mean + self.BL_mean + self.ET_mean + self.BT_mean) / self.s.t
        # self.pL = (self.ET_mean + self.BT_mean - (self.EL_mean + self.BL_mean)) / self.s.t
        # self.pT = (self.EL_mean + self.BL_mean) / self.s.t


@myjit
def fields_kernel(xi, n, u0, pt0, pt1, aeta0, peta0, peta1, t, E, B):
    # Ez
    bf = su.add(peta0[xi], peta1[xi])
    bf = su.mul_s(bf, 0.5)
    E[xi, 2] = su.sq(bf) 

    # Ex & Ey
    for d in range(2):
        bf = su.add(pt0[xi, d], pt1[xi, d])
        bf = su.mul_s(bf, 0.5 / t)
        E[xi, d] = su.sq(bf) 

    # Bz
    bf = su.ah(l.plaq_pos(u0, xi, 0, 1, n))
    B[xi, 2] = su.sq(bf)

    # Bx & By
    for d in range(2):
        b1 = l.transport(aeta0, u0, xi, 0, +1, n)
        b2 = l.transport(aeta0, u0, xi, 0, -1, n)
        b1 = l.add_mul(b1, b2, -1.0)
        bf = su.mul_s(b1, 0.5 / t)
        B[xi, d] = su.sq(bf) 
