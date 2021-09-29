"""
    Extraction of the electric and magnetic fields, which are then used to evaluate the correlators of color Lorentz force along HQs' trajectories.
"""

from curraun.numba_target import mycudajit, use_cuda, my_cuda_loop
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda


class ElectricFields:
    def __init__(self, s):
        self.s = s
        self.n = s.n

        # Electric fields, stored as [Ex, Ey, Ez], evaluated at a given time, in the lattice point where the HQ resides

        self.elfields = np.zeros((3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_elfields = self.elfields

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_elfields = cuda.to_device(self.elfields)

    def copy_to_host(self):
        self.d_elfields.copy_to_host(self.elfields)

    def compute(self, xhq, yhq):
        u0 = self.s.d_u0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_cuda_loop(elfields_kernel, self.d_elfields, xhq, yhq, n, u0, peta1, peta0, pt1, pt0, t)

        if use_cuda:
            self.copy_to_host()

        return self.d_elfields

# kernels
@mycudajit
def elfields_kernel(elfields, xhq, yhq, n, u0, peta1, peta0, pt1, pt0, tau):

    poshq = l.get_index(xhq, yhq, n)

    i = 0
    Ex = su.zero()
    Ex = su.add(Ex, pt1[poshq , i])
    Ex = su.add(Ex, pt0[poshq , i])
    xs = l.shift(poshq, i, -1, n)
    b1 = l.act(su.dagger(u0[xs, i]), pt1[xs, i])
    Ex = su.add(Ex, b1)
    b1 = l.act(su.dagger(u0[xs, i]), pt0[xs, i])
    Ex = su.add(Ex, b1)
    Ex = su.mul_s(Ex, 0.25 / tau)
    
    i = 1
    Ey = su.zero()
    Ey = su.add(Ey, pt1[poshq, i])
    Ey = su.add(Ey, pt0[poshq, i])
    xs = l.shift(poshq, i, -1, n)
    b1 = l.act(su.dagger(u0[xs, i]), pt1[xs, i])
    Ey = su.add(Ey, b1)
    b1 = l.act(su.dagger(u0[xs, i]), pt0[xs, i])
    Ey = su.add(Ey, b1)
    Ey = su.mul_s(Ey, 0.25 / tau)

    Ez = su.zero()
    Ez = l.add_mul(Ez, peta1[poshq], 0.5)
    Ez = l.add_mul(Ez, peta0[poshq], 0.5)


    su.store(elfields[0, :], Ex)
    su.store(elfields[1, :], Ey)
    su.store(elfields[2, :], Ez)

class LorentzForce:
    def __init__(self, s):
        self.s = s
        self.n = s.n

        # Lorentz force, stored as [Fx, Fy, Fz], evaluated at a given time, in the lattice point where the HQ resides
        self.lorentzforce = np.zeros((3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_lorentzforce = self.lorentzforce

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_lorentzforce = cuda.to_device(self.lorentzforce)

    def copy_to_host(self):
        self.d_lorentzforce.copy_to_host(self.lorentzforce)

    def compute(self, xhq, yhq, ptauhq, pxhq, pyhq, petahq, current_t):
        u0 = self.s.d_u0

        aeta0 = self.s.d_aeta0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_cuda_loop(lorentzforce_kernel, self.d_lorentzforce, xhq, yhq, ptauhq, pxhq, pyhq, petahq, current_t, n, u0, aeta0, peta1, peta0, pt1, pt0, t)

        if use_cuda:
            self.copy_to_host()

        return self.d_lorentzforce

# kernels
@mycudajit
def lorentzforce_kernel(lorentzforce, xhq, yhq, ptauhq, pxhq, pyhq, petahq, current_t, n, u0, aeta0, peta1, peta0, pt1, pt0, tau):

    poshq = l.get_index(xhq, yhq, n)

    i = 0
    Ex = su.zero()
    Ex = su.add(Ex, pt1[poshq , i])
    Ex = su.add(Ex, pt0[poshq , i])
    xs = l.shift(poshq, i, -1, n)
    b1 = l.act(su.dagger(u0[xs, i]), pt1[xs, i])
    Ex = su.add(Ex, b1)
    b1 = l.act(su.dagger(u0[xs, i]), pt0[xs, i])
    Ex = su.add(Ex, b1)
    Ex = su.mul_s(Ex, 0.25 / tau)
    
    i = 1
    Ey = su.zero()
    Ey = su.add(Ey, pt1[poshq, i])
    Ey = su.add(Ey, pt0[poshq, i])
    xs = l.shift(poshq, i, -1, n)
    b1 = l.act(su.dagger(u0[xs, i]), pt1[xs, i])
    Ey = su.add(Ey, b1)
    b1 = l.act(su.dagger(u0[xs, i]), pt0[xs, i])
    Ey = su.add(Ey, b1)
    Ey = su.mul_s(Ey, 0.25 / tau)

    Ez = su.zero()
    Ez = l.add_mul(Ez, peta1[poshq], 0.5)
    Ez = l.add_mul(Ez, peta0[poshq], 0.5)

    b1 = l.transport(aeta0, u0, poshq, 1, +1, n)
    b2 = l.transport(aeta0, u0, poshq, 1, -1, n)
    b2 = l.add_mul(b1, b2, -1.0)
    Bx = su.mul_s(b2, -0.5 / tau)
    
    b1 = l.transport(aeta0, u0, poshq, 0, +1, n)
    b2 = l.transport(aeta0, u0, poshq, 0, -1, n)
    b2 = l.add_mul(b1, b2, -1.0)
    By = su.mul_s(b2, +0.5 / tau)

    bf1 = su.zero()
    b1 = l.plaq(u0, poshq, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, poshq, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, poshq, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, poshq, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    Bz = l.add_mul(bf1, b2, +0.25)

    b1 = su.mul_s(Bz, pyhq/ptauhq)
    b2 = su.add(Ex, b1)
    b3 = su.mul_s(By, -current_t*petahq/ptauhq)
    Fx = su.add(b2, b3)

    b1 = su.mul_s(Bz, -pxhq/ptauhq)
    b2 = su.add(Ey, b1)
    b3 = su.mul_s(Bx, current_t*petahq/ptauhq)
    Fy = su.add(b2, b3)

    b1 = su.mul_s(By, pxhq/ptauhq)
    b2 = su.add(Ez, b1)
    b3 = su.mul_s(Bx, -pyhq/ptauhq)
    Fz = su.add(b2, b3)

    su.store(lorentzforce[0, :], Fx)
    su.store(lorentzforce[1, :], Fy)
    su.store(lorentzforce[2, :], Fz)

class ForceCorrelators:
    def __init__(self, s):
        self.s = s
        self.n = s.n

        self.fformf = np.zeros(3, dtype=su.GROUP_TYPE_REAL)
        self.w = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        my_cuda_loop(init_wilson_line_kernel, self.w)

        self.d_fformf = self.fformf
        self.d_w = self.w

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):

        self.d_fformf = cuda.to_device(self.fformf)
        self.d_w = cuda.to_device(self.w)

    def copy_to_host(self):

        self.d_fformf.copy_to_host(self.fformf)
        self.d_w.copy_to_host(self.w)

    def compute(self, tag, Fform, F, xhq, xhq0, yhq, yhq0, delta_etahq):

        u0 = self.s.d_u0
        n = self.n

        aeta0 = self.s.d_aeta0

        if tag=='naive':
            my_cuda_loop(force_correlator_naive_kernel, Fform, F, self.d_fformf)

        if (xhq>xhq0):
            my_cuda_loop(Ux_up, self.d_w, xhq, yhq, n, u0)
        elif (xhq<xhq0):
            my_cuda_loop(Ux_down, self.d_w, xhq, yhq, n, u0)

        if (yhq>yhq0):
            my_cuda_loop(Uy_up, self.d_w, xhq, yhq, n, u0)
        elif (yhq<yhq0):
            my_cuda_loop(Uy_down, self.d_w, xhq, yhq, n, u0)

        my_cuda_loop(Ueta, self.d_w, xhq, yhq, n, delta_etahq, aeta0)

        if tag=='transported':
            my_cuda_loop(force_correlator_transported_kernel, Fform, F, self.d_fformf, self.d_w)

        if use_cuda:
            self.copy_to_host()

@mycudajit
def init_wilson_line_kernel(w):
    su.store(w, su.unit())

@mycudajit
def Ux_up(w, xhq, yhq, n, u0):
    xs = l.get_index(xhq, yhq, n)
    su.store(w, su.mul(w, u0[xs, 0]))

@mycudajit
def Ux_down(w, xhq, yhq, n, u0):
    xs = l.get_index(xhq, yhq, n)
    su.store(w, su.mul(w, su.dagger(u0[xs, 0])))

@mycudajit
def Uy_up(w, xhq, yhq, n, u0):
    xs = l.get_index(xhq, yhq, n)
    su.store(w, su.mul(w, u0[xs, 1]))

@mycudajit
def Uy_down(w, xhq, yhq, n, u0):
    xs = l.get_index(xhq, yhq, n)
    su.store(w, su.mul(w, su.dagger(u0[xs, 1])))

@mycudajit
def Ueta(w, xhq, yhq, n, delta_etahq, aeta0):
    xs = l.get_index(xhq, yhq, n)
    buf = su.mexp(su.mul_s(aeta0[xs], delta_etahq))
    su.store(w, su.mul(w, buf))

@mycudajit
def force_correlator_naive_kernel(Fform, F, fformf):
    for i in range(3):
        fformf[i] = su.tr(su.mul(F[i], su.dagger(Fform[i]))).real

@mycudajit
def force_correlator_transported_kernel(Fform, F, fformf, w):
    for i in range(3):
        UdagFformU = l.act(w, Fform[i])
        fformf[i] = su.tr(su.mul(F[i], su.dagger(UdagFformU))).real