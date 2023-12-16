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


class Energy_GaugeFields():
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

    def copy_to_device(self):
        self.d_E = cuda.to_device(self.E)
        self.d_B = cuda.to_device(self.B)

    def copy_to_host(self):
        self.d_E.copy_to_host(self.E)
        self.d_B.copy_to_host(self.B)

    def compute(self):
        # compute contributions in 2d
        u0 = self.s.d_u0
        u1 = self.s.d_u1
        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1
        aeta0 = self.s.d_aeta0
        aeta1 = self.s.d_aeta1
        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        dt = self.s.dt
        # dth = dt / 2.0
        t = self.s.t

        E = self.d_E
        B = self.d_B

        n = self.s.n

        # my_parallel_loop(fields_kernel, n ** 2, n, u0, pt0, pt1, aeta0, peta0, peta1, t, E, B)

        my_parallel_loop(gaugefields_kernel, n ** 2, n, u0, u1, aeta0, aeta1, t, dt, E, B)
        # my_parallel_loop(gaugefields_kernel, n ** 2, n, u0, u1, aeta0, aeta1, t, dt, E)

        # compute means (very inefficient, but want to keep the arrays intact)
        if use_cuda:
            self.copy_to_host()

        self.E_mean = np.mean(E, axis=0) 
        self.B_mean = np.mean(B, axis=0) 

@myjit
def gaugefields_kernel(xi, n, u0, u1, aeta0, aeta1, t, dt, E, B):
# def gaugefields_kernel(xi, n, u0, u1, aeta0, aeta1, t, dt, E):
    for d in range(2):
        # Extract Ai as logarithms
        Ai0 = su.mlog(u0[xi, d, :])
        Ai1 = su.mlog(u1[xi, d, :])

        # ET
        bf = l.add_mul(Ai1, Ai0, -1.0)
        Ftaui = su.mul_s(bf, 1.0/dt)
        E[xi, d] = su.sq(Ftaui)
        
        # BT
        xip1 = l.shift(xi, d, 1, n)
        xim1 = l.shift(xi, d, -1, n)
        bf1 = l.add_mul(aeta0[xip1], aeta0[xim1], -1.0)
        bf1 = su.mul_s(bf1, 0.5)
        bf2 = l.comm(Ai0, aeta0[xi])
        # bf2 = su.mul_s(bf2, 1j)
        bf = l.add_mul(bf1, bf2, +1.0)
        Fetai = su.mul_s(bf, 1.0/t)
        B[xi, d] = su.sq(Fetai)

    # EL
    bf = su.mul_s(aeta0[xi], -1.0)
    bf = su.add(aeta1[xi], bf)
    Ftaueta = su.mul_s(bf, 1.0/dt/t)
    E[xi, 2] = su.sq(Ftaueta)

    # BL 
    # Gauge fields
    # xip1i = l.shift(xi, 0, 1, n)
    # xim1i = l.shift(xi, 0, -1, n)
    # Aj0xip1i = su.mlog(u0[xip1i, 1, :])
    # Aj0xim1i = su.mlog(u0[xim1i, 1, :])
    # bf1 = l.add_mul(Aj0xip1i, Aj0xim1i, -1.0)
    # bf1 = su.mul_s(bf1, 0.5)

    # xip1j = l.shift(xi, 1, 1, n)
    # xim1j = l.shift(xi, 1, -1, n)
    # Ai0xip1j = su.mlog(u0[xip1j, 0, :])
    # Ai0xim1j = su.mlog(u0[xim1j, 0, :])
    # bf2 = l.add_mul(Ai0xip1j, Ai0xim1j, -1.0)
    # bf2 = su.mul_s(bf2, 0.5)

    # bf = l.add_mul(bf1, bf2, -1.0)
    # Ai0 = su.mlog(u0[xi, 0, :])
    # Aj0 = su.mlog(u0[xi, 1, :])
    # bf3 = l.comm(Ai0, Aj0)
    # Fij = l.add_mul(bf, bf3, +1.0)
    # B[xi, 2] = su.sq(Fij)

    # BL
    # Plaquette
    bf = su.ah(l.plaq_pos(u0, xi, 0, 1, n))
    B[xi, 2] = su.sq(bf)