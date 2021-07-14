"""
    Extraction of the electric and magnetic fields, which are then used to evaluate the correlators of color Lorentz force along HQs' trajectories.
"""

from curraun.numba_target import mycudajit, use_cuda
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

        threadsperblock = 32
        blockspergrid = (su.GROUP_ELEMENTS + (threadsperblock - 1)) // threadsperblock
        elfields_kernel[blockspergrid, threadsperblock](xhq, yhq, n, u0, peta1, peta0, pt1, pt0, t, self.d_elfields)

        if use_cuda:
            self.copy_to_host()

        return self.d_elfields

# kernels
@mycudajit
def elfields_kernel(xhq, yhq, n, u0, peta1, peta0, pt1, pt0, tau, lorentzforce):

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


    su.store(lorentzforce[0, :], Ex)
    su.store(lorentzforce[1, :], Ey)
    su.store(lorentzforce[2, :], Ez)

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

        threadsperblock = 32
        blockspergrid = (su.GROUP_ELEMENTS + (threadsperblock - 1)) // threadsperblock
        lorentzforce_kernel[blockspergrid, threadsperblock](xhq, yhq, ptauhq, pxhq, pyhq, petahq, current_t, n, u0, aeta0, peta1, peta0, pt1, pt0, t, self.d_lorentzforce)

        if use_cuda:
            self.copy_to_host()

        return self.d_lorentzforce

# kernels
@mycudajit
def lorentzforce_kernel(xhq, yhq, ptauhq, pxhq, pyhq, petahq, current_t, n, u0, aeta0, peta1, peta0, pt1, pt0, tau, lorentzforce):

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
    b3 = su.mul_s(Bx, current_t*pyhq/ptauhq)
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

        self.fxformfx = np.zeros(1, dtype=su.GROUP_TYPE_REAL)
        self.fyformfy = np.zeros(1, dtype=su.GROUP_TYPE_REAL)
        self.fzformfz = np.zeros(1, dtype=su.GROUP_TYPE_REAL)

        self.Uxhq = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.Uyhq = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)

        self.d_fxformfx = self.fxformfx
        self.d_fyformfy = self.fyformfy
        self.d_fzformfz = self.fzformfz

        self.d_Uxhq = self.Uxhq
        self.d_Uyhq = self.Uyhq

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_fxformfx = cuda.to_device(self.fxformfx)
        self.d_fyformfy = cuda.to_device(self.fyformfy)
        self.d_fzformfz = cuda.to_device(self.fzformfz)

        self.d_Uxhq = cuda.to_device(self.Uxhq)
        self.d_Uyhq = cuda.to_device(self.Uyhq)

    def copy_to_host(self):
        self.d_fxformfx.copy_to_host(self.fxformfx)
        self.d_fyformfy.copy_to_host(self.fyformfy)
        self.d_fzformfz.copy_to_host(self.fzformfz)

        self.d_Uxhq.copy_to_host(self.Uxhq)
        self.d_Uyhq.copy_to_host(self.Uyhq)

    def compute(self, tag, Fxform, Fyform, Fzform, Fx, Fy, Fz, Uxhq0, Uyhq0, xhq, xhq0, yhq, yhq0, delta_etahq):

        u0 = self.s.d_u0
        n = self.n

        aeta0 = self.s.d_aeta0

        threadsperblock = 32
        blockspergrid = (Fx.size + (threadsperblock - 1)) // threadsperblock
        if tag=='naive':
            force_correlator_naive_kernel[blockspergrid, threadsperblock](Fxform, Fyform, Fzform, Fx, Fy, Fz, self.d_fxformfx, self.d_fyformfy, self.d_fzformfz)

        blockspergrid = (Uxhq0.size + (threadsperblock - 1)) // threadsperblock

        if (xhq>xhq0):
            Ux_up[blockspergrid, threadsperblock](Uxhq0, xhq, yhq, n, u0, self.d_Uxhq)
        if (xhq<xhq0):
            Ux_down[blockspergrid, threadsperblock](Uxhq0, xhq, yhq, n, u0, self.d_Uxhq)
        if (xhq==xhq0):
            U_same[blockspergrid, threadsperblock](Uxhq0, self.d_Uxhq)

        if (yhq>yhq0):
            Uy_up[blockspergrid, threadsperblock](Uyhq0, xhq, yhq, n, u0, self.d_Uyhq)
        if (yhq<yhq0):
            Uy_down[blockspergrid, threadsperblock](Uyhq0, xhq, yhq, n, u0, self.d_Uxhq)
        if (yhq==yhq0):
            U_same[blockspergrid, threadsperblock](Uyhq0, self.d_Uyhq)

        blockspergrid = (Fx.size + (threadsperblock - 1)) // threadsperblock
        if tag=='UxUy':
            force_correlator_UxUy_kernel[blockspergrid, threadsperblock](Fxform, Fyform, Fzform, Fx, Fy, Fz, self.d_Uxhq, self.d_Uyhq, self.d_fxformfx, self.d_fyformfy, self.d_fzformfz)
        elif tag=='UxUyAeta':
            force_correlator_UxUyAeta_kernel[blockspergrid, threadsperblock](Fxform, Fyform, Fzform, Fx, Fy, Fz, self.d_Uxhq, self.d_Uyhq, xhq, yhq, n, delta_etahq, aeta0, self.d_fxformfx, self.d_fyformfy, self.d_fzformfz)
        elif tag=='AetaUxUy':
            force_correlator_AetaUxUy_kernel[blockspergrid, threadsperblock](Fxform, Fyform, Fzform, Fx, Fy, Fz, self.d_Uxhq, self.d_Uyhq, xhq, yhq, n, delta_etahq, aeta0, self.d_fxformfx, self.d_fyformfy, self.d_fzformfz)
        
        if use_cuda:
            self.copy_to_host()

        # return [self.d_fxformfx, self.d_fyformfy, self.d_fzformfz]

@mycudajit
def U_same(U0, U):
    su.store(U, U0)

@mycudajit
def Ux_up(Uxhq0, xhq, yhq, n, u0, Uxhq):
    xs = l.get_index(xhq, yhq, n)
    su.store(Uxhq, su.mul(Uxhq0, u0[xs, 0]))

@mycudajit
def Ux_down(Uxhq0, xhq, yhq, n, u0, Uxhq):
    xs = l.get_index(xhq, yhq, n)
    su.store(Uxhq, su.mul(Uxhq0, su.dagger(u0[xs, 0])))

@mycudajit
def Uy_up(Uyhq0, xhq, yhq, n, u0, Uyhq):
    xs = l.get_index(xhq, yhq, n)
    su.store(Uyhq, su.mul(Uyhq0, u0[xs, 1]))

@mycudajit
def Uy_down(Uyhq0, xhq, yhq, n, u0, Uyhq):
    xs = l.get_index(xhq, yhq, n)
    su.store(Uyhq, su.mul(Uyhq0, su.dagger(u0[xs, 1])))

@mycudajit
def force_correlator_naive_kernel(Fxform, Fyform, Fzform, Fx, Fy, Fz, fxformfx, fyformfy, fzformfz):
    
    fxformfx[:] = su.tr(su.mul(Fx, su.dagger(Fxform))).real
    fyformfy[:] = su.tr(su.mul(Fy, su.dagger(Fyform))).real
    fzformfz[:] = su.tr(su.mul(Fz, su.dagger(Fzform))).real

@mycudajit
def force_correlator_UxUy_kernel(Fxform, Fyform, Fzform, Fx, Fy, Fz, Uxhq, Uyhq, fxformfx, fyformfy, fzformfz):
    
    FxformUxUy = l.act(su.mul(Uxhq, Uyhq), Fxform)
    fxformfx[:] = su.tr(su.mul(Fx, su.dagger(FxformUxUy))).real

    FyformUxUy = l.act(su.mul(Uxhq, Uyhq), Fyform)
    fyformfy[:] = su.tr(su.mul(Fy, su.dagger(FyformUxUy))).real

    FzformUxUy = l.act(su.mul(Uxhq, Uyhq), Fzform)
    fzformfz[:] = su.tr(su.mul(Fz, su.dagger(FzformUxUy))).real


@mycudajit
def force_correlator_UxUyAeta_kernel(Fxform, Fyform, Fzform, Fx, Fy, Fz, Uxhq, Uyhq, xhq, yhq, n, delta_etahq, aeta0, fxformfx, fyformfy, fzformfz):
    
    xs = l.get_index(xhq, yhq, n)
    Uetahq = su.mexp(su.mul_s(aeta0[xs, :], delta_etahq))
    UxUyUetahq = su.mul(su.mul(Uxhq, Uyhq), Uetahq)

    FxformUxUyAeta = l.act(UxUyUetahq, Fxform)
    fxformfx[:] = su.tr(su.mul(Fx, su.dagger(FxformUxUyAeta))).real

    FyformUxUyAeta = l.act(UxUyUetahq, Fyform)
    fyformfy[:] = su.tr(su.mul(Fy, su.dagger(FyformUxUyAeta))).real

    FzformUxUyAeta = l.act(UxUyUetahq, Fzform)
    fzformfz[:] = su.tr(su.mul(Fz, su.dagger(FzformUxUyAeta))).real


@mycudajit
def force_correlator_AetaUxUy_kernel(Fxform, Fyform, Fzform, Fx, Fy, Fz, Uxhq, Uyhq, xhq, yhq, n, delta_etahq, aeta0, fxformfx, fyformfy, fzformfz):
    
    xs = l.get_index(xhq, yhq, n)
    Uetahq = su.mexp(su.mul_s(aeta0[xs, :], delta_etahq))
    UxUyUetahq = su.mul(Uetahq, su.mul(Uxhq, Uyhq))

    FxformAetaUxUy = l.act(UxUyUetahq, Fxform)
    fxformfx[:] = su.tr(su.mul(Fx, su.dagger(FxformAetaUxUy))).real

    FyformAetaUxUy = l.act(UxUyUetahq, Fyform)
    fyformfy[:] = su.tr(su.mul(Fy, su.dagger(FyformAetaUxUy))).real

    FzformAetaUxUy = l.act(UxUyUetahq, Fzform)
    fzformfz[:] = su.tr(su.mul(Fz, su.dagger(FzformAetaUxUy))).real