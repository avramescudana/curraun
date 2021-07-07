"""
    Computation of the Glasma electric field correlators at different times, averaged over transverse simulation box.
"""

from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda


class ElectricFieldsCorrelators:
    def __init__(self, s):
        self.s = s
        self.n = s.n

        self.elxformelx = np.zeros((self.n ** 2), dtype=su.GROUP_TYPE_REAL)
        self.elyformely = np.zeros((self.n ** 2), dtype=su.GROUP_TYPE_REAL)
        self.elzformelz = np.zeros((self.n ** 2), dtype=su.GROUP_TYPE_REAL)
        self.d_elxformelx = self.elxformelx
        self.d_elyformely = self.elyformely
        self.d_elzformelz = self.elzformelz

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_elxformelx = cuda.to_device(self.elxformelx)
        self.d_elyformely = cuda.to_device(self.elyformely)
        self.d_elzformelz = cuda.to_device(self.elzformelz)

    def copy_to_host(self):
        self.d_elxformelx.copy_to_host(self.elxformelx)
        self.d_elyformely.copy_to_host(self.elyformely)
        self.d_elzformelz.copy_to_host(self.elzformelz)
        self.elxformelx /= self.s.g ** 2
        self.elyformely /= self.s.g ** 2
        self.elzformelz /= self.s.g ** 2

    def compute(self, elxform, elyform, elzform):
        u0 = self.s.d_u0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_parallel_loop(compute_Ex_correlation_kernel, n ** 2, n, u0, pt1, pt0, t, elxform, self.d_elxformelx)
        my_parallel_loop(compute_Ey_correlation_kernel, n ** 2, n, u0, pt1, pt0, t, elyform, self.d_elyformely)
        my_parallel_loop(compute_Ez_correlation_kernel, n ** 2, peta1, peta0, elzform, self.d_elzformelz)

        if use_cuda:
            self.copy_to_host()
    
        return np.mean(self.elxformelx), np.mean(self.elyformely), np.mean(self.elzformelz)

# kernels
@myjit
def compute_Ex_correlation_kernel(xi, n, u0, pt1, pt0, tau, elxform, elxformelx):
    
    elx = compute_Ex(xi, n, u0, pt1, pt0, tau)
    elxformelx[xi] = su.tr(su.mul(elxform[xi], su.dagger(elx))).real


@myjit
def compute_Ex(xi, n, u0,  pt1, pt0, tau):

    Ex = su.zero()

    Ex = su.add(Ex, pt1[xi, 0])
    Ex = su.add(Ex, pt0[xi, 0])
    xs = l.shift(xi, 0, -1, n)
    b1 = l.act(su.dagger(u0[xs, 0]), pt1[xs, 0])
    Ex = su.add(Ex, b1)
    b1 = l.act(su.dagger(u0[xs, 0]), pt0[xs, 0])
    Ex = su.add(Ex, b1)
    Ex = su.mul_s(Ex, 0.25 / tau)
    
    return Ex

@myjit
def compute_Ey_correlation_kernel(xi, n, u0, pt1, pt0, tau, elyform, elyformely):
    
    ely = compute_Ey(xi, n, u0, pt1, pt0, tau)
    elyformely[xi] = su.tr(su.mul(elyform[xi], su.dagger(ely))).real

@myjit
def compute_Ey(xi, n, u0, pt1, pt0, tau):

    Ey = su.zero()

    Ey = su.add(Ey, pt1[xi, 1])
    Ey = su.add(Ey, pt0[xi, 1])
    xs = l.shift(xi, 1, -1, n)
    b1 = l.act(su.dagger(u0[xs, 1]), pt1[xs, 1])
    Ey = su.add(Ey, b1)
    b1 = l.act(su.dagger(u0[xs, 1]), pt0[xs, 1])
    Ey = su.add(Ey, b1)
    Ey = su.mul_s(Ey, 0.25 / tau)
    
    return Ey


@myjit
def compute_Ez_correlation_kernel(xi, peta1, peta0, elzform, elzformelz):
    
    elz = compute_Ez(xi, peta1, peta0)
    elzformelz[xi] = su.tr(su.mul(elzform[xi], su.dagger(elz))).real

@myjit
def compute_Ez(xi, peta1, peta0):

    Ez = su.zero()

    Ez = l.add_mul(Ez, peta1[xi], 0.5)
    Ez = l.add_mul(Ez, peta0[xi], 0.5)
    
    return Ez
