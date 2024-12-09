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

        self.EL = np.zeros(shape=s.n ** 2, dtype=DTYPE)
        self.BL = np.zeros(shape=s.n ** 2, dtype=DTYPE)
        self.ET = np.zeros(shape=s.n ** 2, dtype=DTYPE)
        self.BT = np.zeros(shape=s.n ** 2, dtype=DTYPE)

        self.d_EL = self.EL
        self.d_BL = self.BL
        self.d_ET = self.ET
        self.d_BT = self.BT

        if use_cuda:
            self.copy_to_device()

        self.EL_mean = 0.0
        self.BL_mean = 0.0
        self.ET_mean = 0.0
        self.BT_mean = 0.0

        self.energy_density = 0.0
        self.pL = 0.0
        self.pT = 0.0

    def copy_to_device(self):
        self.d_EL = cuda.to_device(self.EL)
        self.d_BL = cuda.to_device(self.BL)
        self.d_ET = cuda.to_device(self.ET)
        self.d_BT = cuda.to_device(self.BT)

    def copy_to_host(self):
        self.d_EL.copy_to_host(self.EL)
        self.d_BL.copy_to_host(self.BL)
        self.d_ET.copy_to_host(self.ET)
        self.d_BT.copy_to_host(self.BT)

    def compute(self, s):
        # compute contributions in 2d
        u0 = s.d_u0
        u1 = s.d_u1
        pt1 = s.d_pt1
        aeta0 = s.d_aeta0
        aeta1 = s.d_aeta1
        peta1 = s.d_peta1

        dt = s.dt
        dth = dt / 2.0
        t = s.t

        EL = self.d_EL
        BL = self.d_BL
        ET = self.d_ET
        BT = self.d_BT

        n = self.s.n

        my_parallel_loop(fields_kernel, n ** 2, n, u0, u1, pt1, aeta0, aeta1, peta1, dt, dth, t, EL, BL, ET, BT)

        # compute means (very inefficient, but want to keep the arrays intact)
        if use_cuda:
            self.copy_to_host()

        self.EL_mean = np.mean(EL) / self.s.g ** 2
        self.BL_mean = np.mean(BL) / self.s.g ** 2
        self.ET_mean = np.mean(ET) / self.s.g ** 2
        self.BT_mean = np.mean(BT) / self.s.g ** 2

        # compute density and pressures
        self.energy_density = (self.EL_mean + self.BL_mean + self.ET_mean + self.BT_mean) / self.s.t
        self.pL = (self.ET_mean + self.BT_mean - (self.EL_mean + self.BL_mean)) / self.s.t
        self.pT = (self.EL_mean + self.BL_mean) / self.s.t


@myjit
def fields_kernel(xi, n, u0, u1, pt1, aeta0, aeta1, peta1, dt, dth, t, EL, BL, ET, BT):
    # longitudinal electric field at t + dth
    EL[xi] = su.sq(peta1[xi]) * (t + dth)

    # transverse electric field at t + dth
    ET[xi] = su.sq(pt1[xi, 0]) / (t + dth) + su.sq(pt1[xi, 1]) / (t + dth)

    # longitudinal magnetic field at t + dth (averaged)
    #BL[xi] = (NC - su.tr(l.plaq_pos(u0, xi, 0, 1, n)).real) * t + (NC - su.tr(l.plaq_pos(u1, xi, 0, 1, n)).real) * (t + dt)
    BL[xi] = 0.5 * (su.sq(su.ah(l.plaq_pos(u0, xi, 0, 1, n))) * t + su.sq(su.ah(l.plaq_pos(u1, xi, 0, 1, n))) * (t+dt))

    # transverse magnetic field at t + dth (averaged)
    d = 0
    buffer1 = l.transport(aeta0, u0, xi, d, 1, n)
    buffer1 = l.add_mul(buffer1, aeta0[xi], -1)
    BT[xi] = su.sq(buffer1) / 2 / t

    buffer1 = l.transport(aeta1, u1, xi, d, 1, n)
    buffer1 = l.add_mul(buffer1, aeta1[xi], -1)
    BT[xi] += su.sq(buffer1) / 2 / (t + dt)

    d = 1
    buffer1 = l.transport(aeta0, u0, xi, d, 1, n)
    buffer1 = l.add_mul(buffer1, aeta0[xi], -1)
    BT[xi] += su.sq(buffer1) / 2 / t

    buffer1 = l.transport(aeta1, u1, xi, d, 1, n)
    buffer1 = l.add_mul(buffer1, aeta1[xi], -1)
    BT[xi] += su.sq(buffer1) / 2 / (t + dt)