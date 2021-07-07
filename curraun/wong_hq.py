"""
    Computes various quantities used in solving Wong's equations for a HQ immersed in the fields of the Glasma.
"""
from curraun.numba_target import myjit, prange, my_parallel_loop, use_cuda, mycudajit
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda

class WongFields:
    # Computes Tr{QE} and Tr{QB}
    def __init__(self, s):
        self.s = s
        self.n = s.n

        # Tr{QE} with electric fields, stored as [Ex, Ey, Eeta], evaluated at a given time
        self.trQE = np.zeros(3, dtype=su.GROUP_TYPE_REAL)
        self.d_trQE = self.trQE

        # Tr{QB} with magnetic fields, stored as [Bx, By, Beta], evaluated at a given time
        self.trQB = np.zeros(3, dtype=su.GROUP_TYPE)
        self.d_trQB = self.trQB

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_trQE = cuda.to_device(self.trQE)
        self.d_trQB = cuda.to_device(self.trQB)

    def copy_to_host(self):
        self.d_trQE.copy_to_host(self.trQE)
        self.d_trQB.copy_to_host(self.trQB)

    def compute(self, Q0, xhq, yhq):
        u0 = self.s.d_u0

        aeta0 = self.s.d_aeta0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        threadsperblock = 32
        blockspergrid = (Q0.size + (threadsperblock - 1)) // threadsperblock
        fields_kernel[blockspergrid, threadsperblock](Q0, xhq, yhq, n, u0, aeta0, peta1, peta0, pt1, pt0, t, self.d_trQE, self.d_trQB)

        if use_cuda:
            self.copy_to_host()


# kernels
@mycudajit
def fields_kernel(Q0, xhq, yhq, n, u0, aeta0, peta1, peta0, pt1, pt0, tau, trQE, trQB):

    # Position of the HQ
    poshq = l.get_index(xhq, yhq, n)

    # Electric fields
    Ex = su.add(pt1[poshq, 0, :], pt1[poshq, 0, :])
    Ex = su.zero()
    Ex = su.add(Ex, pt1[poshq, 0, :])
    Ex = su.add(Ex, pt0[poshq, 0, :])
    xs = l.shift(poshq, 0, -1, n)
    b1 = l.act(su.dagger(u0[xs, 0, :]), pt1[xs, 0, :])
    Ex = su.add(Ex, b1)
    b1 = l.act(su.dagger(u0[xs, 0]), pt0[xs, 0, :])
    Ex = su.add(Ex, b1)
    Ex = su.mul_s(Ex, 0.25 / tau)
    
    Ey = su.zero()
    Ey = su.add(Ey, pt1[poshq, 1, :])
    Ey = su.add(Ey, pt0[poshq, 1, :])
    xs = l.shift(poshq, 1, -1, n)
    b1 = l.act(su.dagger(u0[xs, 1, :]), pt1[xs, 1, :])
    Ey = su.add(Ey, b1)
    b1 = l.act(su.dagger(u0[xs, 1, :]), pt0[xs, 1, :])
    Ey = su.add(Ey, b1)
    Ey = su.mul_s(Ey, 0.25 / tau)

    Eeta = su.zero()
    Eeta = l.add_mul(Eeta, peta1[poshq, :], 0.5)
    Eeta = l.add_mul(Eeta, peta0[poshq, :], 0.5)

    # Magnetic fields
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
    Beta = l.add_mul(bf1, b2, +0.25)

    trQE[0] = su.tr(su.mul(Q0, Ex)).real
    trQE[1] = su.tr(su.mul(Q0, Ey)).real
    trQE[2] = su.tr(su.mul(Q0, Eeta)).real

    trQB[0] = su.tr(su.mul(Q0, Bx)).real
    trQB[1] = su.tr(su.mul(Q0, By)).real
    trQB[2] = su.tr(su.mul(Q0, Beta)).real


class WongPotentials:
    # Computes Ax, Ay and Aeta
    def __init__(self, s):
        self.s = s

        # Gauge fields stored as [Ax, Ay, Aeta], evaluated at a given time
        self.Ax = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Ax = self.Ax
        self.Ay = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Ay = self.Ay
        self.Aeta = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Aeta = self.Aeta

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_Ax = cuda.to_device(self.Ax)
        self.d_Ay = cuda.to_device(self.Ay)
        self.d_Aeta = cuda.to_device(self.Aeta)

    def copy_to_host(self):
        self.d_Ax.copy_to_host(self.Ax)
        self.d_Ay.copy_to_host(self.Ay)
        self.d_Aeta.copy_to_host(self.Aeta)

    def compute(self, tag, xhq, yhq):
        u0 = self.s.d_u0
        aeta0 = self.s.d_aeta0
        n = self.s.n

        threadsperblock = 32
        blockspergrid = (su.GROUP_ELEMENTS + (threadsperblock - 1)) // threadsperblock
        if tag=='x':
            Ax_kernel[blockspergrid, threadsperblock](xhq, yhq, n, u0, self.d_Ax)
        elif tag=='y':
            Ay_kernel[blockspergrid, threadsperblock](xhq, yhq, n, u0, self.d_Ay)
        elif tag=='eta':
            Aeta_kernel[blockspergrid, threadsperblock](xhq, yhq, n, aeta0, self.d_Aeta)

        if use_cuda:
            self.copy_to_host()

@mycudajit
def Ax_kernel(xhq, yhq, n, u0, Ax):
    poshq = l.get_index(xhq, yhq, n)
    su.store(Ax, su.mlog(u0[poshq, 0, :]))

@mycudajit
def Ay_kernel(xhq, yhq, n, u0, Ay):
    poshq = l.get_index(xhq, yhq, n)
    su.store(Ay, su.mlog(u0[poshq, 1, :]))

@mycudajit
def Aeta_kernel(xhq, yhq, n, aeta0, Aeta):
    poshq = l.get_index(xhq, yhq, n)
    su.store(Aeta, aeta0[poshq, :])


class InitialColorCharge:
    # Computes the Lie algebra valued color charge from color components of initial color charge
    def __init__(self, s):
        self.s = s

        self.Q = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Q = self.Q
        self.Q2 = np.zeros(1, dtype=su.GROUP_TYPE)
        self.d_Q2 = self.Q2

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_Q = cuda.to_device(self.Q)
        self.d_Q2 = cuda.to_device(self.Q2)

    def copy_to_host(self):
        self.d_Q.copy_to_host(self.Q)
        self.d_Q2.copy_to_host(self.Q2)

    def compute(self, q0):

        threadsperblock = 32
        blockspergrid = (q0.size + (threadsperblock - 1)) // threadsperblock
        initial_charge_kernel[blockspergrid, threadsperblock](q0, self.d_Q, self.d_Q2)

        if use_cuda:
            self.copy_to_host()

@mycudajit
def initial_charge_kernel(q0, Q, Q2):
    Q[:] = su.get_algebra_element(q0)
    Q2[:] = su.sq(Q[:]).real

class ColorChargeEvolve:
    # Evolves the color charge and computes the quadratic Casimir, for both SU(2) and SU(3)
    #TODO: Compute the cubic Casimir, for SU(3)
    def __init__(self, s):
        self.s = s

        self.Q = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Q = self.Q
        self.Q2 = np.zeros(su.ALGEBRA_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Q2 = self.Q2

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_Q = cuda.to_device(self.Q)
        self.d_Q2 = cuda.to_device(self.Q2)

    def copy_to_host(self):
        self.d_Q.copy_to_host(self.Q)
        self.d_Q2.copy_to_host(self.Q2)

    def compute(self, Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta):

        threadsperblock = 32
        blockspergrid = (Q0.size + (threadsperblock - 1)) // threadsperblock
        evolve_charge_kernel[blockspergrid, threadsperblock](Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta, self.d_Q, self.d_Q2)

        if use_cuda:
            self.copy_to_host()


@mycudajit
def evolve_charge_kernel(Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta, Q, Q2):
    commQAx = l.comm(Q0, Ax)
    commQAy = l.comm(Q0, Ay)
    commQAeta = l.comm(Q0, Aeta)
    
    prodx = su.mul_s(commQAx, px0)
    prody = su.mul_s(commQAy, py0)
    prodeta = su.mul_s(commQAeta, peta0)
    sum1 = su.add(prodx, prody)
    sum2 = su.add(sum1, prodeta)
    res1 = su.mul_s(sum2, tau_step / ptau0)
    Q[:] = su.add(Q0, res1)
    Q2[:] = su.sq(Q[:]).real
