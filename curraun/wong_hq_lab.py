"""
    Computes various quantities used in solving Wong's equations for a HQ immersed in the fields of the Glasma.
"""

from curraun.numba_target import use_cuda, mycudajit
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda
import math
from scipy.interpolate import griddata

class WongFields:
    # Computes Tr{QE} and Tr{QB}
    def __init__(self, s):
        self.s = s
        self.n = s.n

        # Tr{QE} with electric fields, stored as [Exlab, Eylab, Ez], evaluated at a given time
        self.trQE = np.zeros(3, dtype=su.GROUP_TYPE_REAL)
        self.d_trQE = self.trQE

        # Tr{QB} with magnetic fields, stored as [Bxlab, Bylab, Bz], evaluated at a given time
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

    def compute(self, Q0, xhq, yhq, sheta, cheta):
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
        fields_kernel[blockspergrid, threadsperblock](Q0, xhq, yhq, sheta, cheta, n, u0, aeta0, peta1, peta0, pt1, pt0, t, self.d_trQE, self.d_trQB)

        if use_cuda:
            self.copy_to_host()


# kernels
@mycudajit
def fields_kernel(Q0, xhq, yhq, sheta, cheta, n, u0, aeta0, peta1, peta0, pt1, pt0, tau, trQE, trQB):

    # Position of the HQ
    poshq = l.get_index(xhq, yhq, n)

    # Electric fields
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

    Ez= su.zero()
    Ez = l.add_mul(Ez, peta1[poshq, :], 0.5)
    Ez = l.add_mul(Ez, peta0[poshq, :], 0.5)

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
    Bz = l.add_mul(bf1, b2, +0.25)

    # Fields in the lab frame
    b1 = su.mul_s(Ex, cheta)
    b2 = su.mul_s(By, sheta)
    Exlab = su.add(b1, b2)

    b1 = su.mul_s(Ey, cheta)
    b2 = su.mul_s(Bx, sheta)
    Eylab = l.add_mul(b1, b2, -1)

    b1 = su.mul_s(Ey, -sheta)
    b2 = su.mul_s(Bx, cheta)
    Bxlab = l.add_mul(b2, b1, -1)

    b1 = su.mul_s(Ex, sheta)
    b2 = su.mul_s(By, cheta)
    Bylab = su.add(b1, b2)

    trQE[0] = su.tr(su.mul(Q0, Exlab)).real
    trQE[1] = su.tr(su.mul(Q0, Eylab)).real
    trQE[2] = su.tr(su.mul(Q0, Ez)).real

    trQB[0] = su.tr(su.mul(Q0, Bxlab)).real
    trQB[1] = su.tr(su.mul(Q0, Bylab)).real
    trQB[2] = su.tr(su.mul(Q0, Bz)).real


class WongPotentials:
    # Computes Ax, Ay and Aeta
    def __init__(self, s):
        self.s = s

        # Gauge fields stored as [Ax, Ay, Aeta], evaluated at a given time
        self.Ax = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Ax = self.Ax
        self.Ay = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Ay = self.Ay
        self.Az = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Az = self.Az

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_Ax = cuda.to_device(self.Ax)
        self.d_Ay = cuda.to_device(self.Ay)
        self.d_Az = cuda.to_device(self.Az)

    def copy_to_host(self):
        self.d_Ax.copy_to_host(self.Ax)
        self.d_Ay.copy_to_host(self.Ay)
        self.d_Az.copy_to_host(self.Az)

    def compute(self, tag, xhq, yhq, cheta, current_tau):
        u0 = self.s.d_u0
        aeta0 = self.s.d_aeta0
        n = self.s.n

        threadsperblock = 32
        blockspergrid = (su.GROUP_ELEMENTS + (threadsperblock - 1)) // threadsperblock
        if tag=='x':
            Ax_kernel[blockspergrid, threadsperblock](xhq, yhq, n, u0, self.d_Ax)
        elif tag=='y':
            Ay_kernel[blockspergrid, threadsperblock](xhq, yhq, n, u0, self.d_Ay)
        elif tag=='z':
            Az_kernel[blockspergrid, threadsperblock](xhq, yhq, cheta, current_tau, n, aeta0, self.d_Az)

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
def Az_kernel(xhq, yhq, cheta, current_tau, n, aeta0, Az):
    poshq = l.get_index(xhq, yhq, n)
    su.store(Az, su.mul_s(aeta0[poshq, :], cheta/current_tau))


class ColorChargeEvolve:
    # Evolves the color charge and computes the quadratic Casimir, for both SU(2) and SU(3)
    #TODO: Compute the cubic Casimir, for SU(3)
    def __init__(self, s):
        self.s = s

        self.Q = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Q = self.Q
        self.Q2 = np.zeros(su.ALGEBRA_ELEMENTS, dtype=su.GROUP_TYPE)
        self.d_Q2 = self.Q2
        self.Q3 = np.zeros(1, dtype=su.GROUP_TYPE)
        self.d_Q3 = self.Q3

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_Q = cuda.to_device(self.Q)
        self.d_Q2 = cuda.to_device(self.Q2)
        self.d_Q3 = cuda.to_device(self.Q3)

    def copy_to_host(self):
        self.d_Q.copy_to_host(self.Q)
        self.d_Q2.copy_to_host(self.Q2)
        self.d_Q3.copy_to_host(self.Q3)

    def compute(self, su_group, Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta):

        threadsperblock = 32
        blockspergrid = (Q0.size + (threadsperblock - 1)) // threadsperblock
        if su_group=='su3':
            evolve_charge_su3_kernel[blockspergrid, threadsperblock](Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta, self.d_Q, self.d_Q2, self.d_Q3)
        else:
            evolve_charge_su2_kernel[blockspergrid, threadsperblock](Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta, self.d_Q, self.d_Q2)

        if use_cuda:
            self.copy_to_host()


@mycudajit
def evolve_charge_su3_kernel(Q0, t_step, pt0, px0, py0, pz0, Ax, Ay, Az, Q, Q2, Q3):
    commQAx = l.comm(Q0, Ax)
    commQAy = l.comm(Q0, Ay)
    commQAz = l.comm(Q0, Az)
    
    prodx = su.mul_s(commQAx, px0)
    prody = su.mul_s(commQAy, py0)
    prodz = su.mul_s(commQAz, pz0)
    sum1 = su.add(prodx, prody)
    sum2 = su.add(sum1, prodz)
    res1 = su.mul_s(sum2, t_step / pt0)
    Q[:] = su.add(Q0, res1)
    Q2[:] = su.sq(Q[:]).real
    Q3[:] = su.tr(su.mul(Q[:], su.mul(Q[:], su.dagger(Q[:])))).imag

@mycudajit
def evolve_charge_su2_kernel(Q0, t_step, pt0, px0, py0, pz0, Ax, Ay, Az, Q, Q2):
    commQAx = l.comm(Q0, Ax)
    commQAy = l.comm(Q0, Ay)
    commQAz = l.comm(Q0, Az)
    
    prodx = su.mul_s(commQAx, px0)
    prody = su.mul_s(commQAy, py0)
    prodz = su.mul_s(commQAz, pz0)
    sum1 = su.add(prodx, prody)
    sum2 = su.add(sum1, prodz)
    res1 = su.mul_s(sum2, t_step / pt0)
    Q[:] = su.add(Q0, res1)
    Q2[:] = su.sq(Q[:]).real

"""
    Update positions and momenta using basic Euler and Boris push to solve the associated Wong's equations
"""

def update_coords(x0, y0, z0, pt0, px0, py0, pz0, t_step):
    x1 = x0 + px0 / pt0 * t_step
    y1 = y0 + py0 / pt0 * t_step
    z1 = z0 + pz0 / pt0 * t_step

    return x1, y1, z1

def update_momenta(pt0, px0, py0, pz0, t_step, trQE, trQB, mass, E0):
    # Dynkin index from Tr{T^aT^b}=T_R\delta^{ab} in fundamental representation R=F
    # The minus sign comes from Tr{QF} from simulation being -Tr{QF} from the analytic formulas
    tr = -1/2

    p0 = [px0, py0, pz0]
    p1, p2, p3, p = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

    for i in range(3):
        p1[i] = p0[i] + t_step / 2 / tr * trQE[i]
    for i in range(3):
        p2[i] = p1[i] + t_step / 2 / tr * (p1[(i+1)%3] * trQB[(i+2)%3]-p1[(i+2)%3] * trQB[(i+1)%3]) / pt0
    for i in range(3):
        p3[i] = p2[i] + t_step / tr * (p2[(i+1)%3] * trQB[(i+2)%3]-p2[(i+2)%3] * trQB[(i+1)%3]) / pt0
    for i in range(3):
        p[i] = p3[i] + t_step / 2 / tr * trQE[i] 
    pt1 = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2 + (mass/E0) ** 2)

    return pt1, p[0], p[1], p[2]