"""
    Computation of the electric fields, longitudinal and transverse, at a given time. 
    These will be used in evaluating the Glasma chromoelectric field correlators at different times, averaged over the transverse box.
"""

from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda

class ElectricFields:
    def __init__(self, s):
        self.s = s
        self.n = s.n

        # Electric fields, stored as [Ex, Ey, Ez] evaluated at a given time

        self.elfields = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_elfields = self.elfields

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_elfields = cuda.to_device(self.elfields)

    def copy_to_host(self):
        self.d_elfields.copy_to_host(self.elfields)

    def compute(self):
        u0 = self.s.d_u0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_parallel_loop(elfields_kernel, n ** 2, n, u0, peta1, peta0, pt1, pt0, t, self.d_elfields)

        if use_cuda:
            self.copy_to_host()

        return self.d_elfields


# kernels
@myjit
def elfields_kernel(xi, n, u0, peta1, peta0, pt1, pt0, tau, elfields):

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

    su.store(elfields[xi, 0], Ex)
    su.store(elfields[xi, 1], Ey)
    su.store(elfields[xi, 2], Ez)
