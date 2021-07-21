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

    def compute(self, su_group, q0):

        threadsperblock = 32
        blockspergrid = (q0.size + (threadsperblock - 1)) // threadsperblock
        if su_group=='su3':
            initial_charge_su3_kernel[blockspergrid, threadsperblock](q0, self.d_Q, self.d_Q2, self.d_Q3)
        else:
            initial_charge_su2_kernel[blockspergrid, threadsperblock](q0, self.d_Q, self.d_Q2)

        if use_cuda:
            self.copy_to_host()

@mycudajit
def initial_charge_su3_kernel(q0, Q, Q2, Q3):
    Q[:] = su.get_algebra_element(q0)
    Q2[:] = su.sq(Q[:]).real
    Q3[:] = su.tr(su.mul(Q[:],su.mul(Q[:], su.dagger(Q[:])))).imag

@mycudajit
def initial_charge_su2_kernel(q0, Q, Q2):
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
def evolve_charge_su3_kernel(Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta, Q, Q2, Q3):
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
    Q3[:] = su.tr(su.mul(Q[:], su.mul(Q[:], su.dagger(Q[:])))).imag

@mycudajit
def evolve_charge_su2_kernel(Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta, Q, Q2):
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

"""
    Initialize positions, momenta and color charges
""" 


def initial_coords(p):
    L = p['L']
    # x0, y0 = np.random.uniform(0, L), np.random.uniform(0, L)
    # TODO: Impose periodic boundary conditions
    # Since the HQ doesn't move too much, it suffices to place it somewhere within the interior of the simulation plane
    x0, y0 = np.random.uniform(L/3, 2*L/3), np.random.uniform(L/3, 2*L/3)
    eta0 = 0
    xmu0 = [x0, y0, eta0]
    return xmu0

# TODO: Initialise with FONLL pQCD distribution in momentum of HQs
def initial_momenta(p):
    pT = p['PT']
    tau_form = p['TFORM']
    mass = p['MASS']
    px0 = np.random.uniform(0, pT)
    py0 = np.sqrt(pT ** 2 - px0 ** 2)
    peta0 = 0
    ptau0 = np.sqrt(px0 ** 2 + py0 ** 2 + (tau_form * peta0) ** 2 + mass ** 2)
    pmu0 = [ptau0, px0, py0, peta0]
    return pmu0

def initial_charge(p):
    su_group = p['GROUP']
    if su_group=='su3':
        # TODO: Find the correct way to initialise the SU(3) color charges
        # Values used to compute the SU(3) and SU(2) Casimirs
        # J1, J2 = 1, 0
        J1, J2 = 2.84801, 1.00841
        # Angle Darboux variables
        phi1, phi2, phi3 = np.random.uniform(), np.random.uniform(), np.random.uniform()
        # phi1, phi2, phi3 = np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)
        # Momenta Darboux variables
        # Fixed J1 and J2
        # pi3 = np.random.uniform(0, (J1+J2)/2)
        # pi2 = np.random.uniform((J2-J1)/np.sqrt(3), (J1-J2)/(2*np.sqrt(3)))
        # pi1 = np.random.uniform(-pi3, pi3)
        # A = np.sqrt(((J1-J2)/3+pi3+pi2/np.sqrt(3))*((J1+2*J2)/3+pi3+pi2/np.sqrt(3))*((2*J1+J2)/3-pi3-pi2/np.sqrt(3)))/(2*pi3)
        # B = np.sqrt(((J2-J1)/3+pi3-pi2/np.sqrt(3))*((J1+2*J2)/3-pi3+pi2/np.sqrt(3))*((2*J1+J2)/3+pi3-pi2/np.sqrt(3)))/(2*pi3)

        search = True
        while search:
            pi2, pi3 = np.random.uniform(), np.random.uniform()
            pi1 = np.random.uniform(-pi3, pi3)

            # numA = 20-np.sqrt(3)*pi2**3+36*pi3-9*pi2**2*pi3-9*pi3**3-12*np.sqrt(3)*pi2-9*np.sqrt(3)*pi2*pi3**2
            # numB = -20+np.sqrt(3)*pi2**3+36*pi3-9*pi2**2*pi3-9*pi3**3-12*np.sqrt(3)*pi2+9*np.sqrt(3)*pi2*pi3**2
            numA = ((J1-J2)/3+pi3+pi2/np.sqrt(3))*((J1+2*J2)/3+pi3+pi2/np.sqrt(3))*((2*J1+J2)/3-pi3-pi2/np.sqrt(3))
            numB = ((J2-J1)/3+pi3-pi2/np.sqrt(3))*((J1+2*J2)/3-pi3+pi2/np.sqrt(3))*((2*J1+J2)/3+pi3-pi2/np.sqrt(3))

            if (numA>0) & (numB>0):
                search = False

        # A = np.sqrt(numA)/(6*pi3)
        # B = np.sqrt(numB)/(6*pi3)
        A = np.sqrt(numA)/(2*pi3)
        B = np.sqrt(numB)/(2*pi3)

        pip, pim = np.sqrt(pi3+pi1), np.sqrt(pi3-pi1)
        Cpp, Cpm, Cmp, Cmm = np.cos((phi1+np.sqrt(3)*phi2+phi3)/2), np.cos((phi1+np.sqrt(3)*phi2-phi3)/2), np.cos((-phi1+np.sqrt(3)*phi2+phi3)/2), np.cos((-phi1+np.sqrt(3)*phi2-phi3)/2)
        Spp, Spm, Smp, Smm = np.sin((phi1+np.sqrt(3)*phi2+phi3)/2), np.sin((phi1+np.sqrt(3)*phi2-phi3)/2), np.sin((-phi1+np.sqrt(3)*phi2+phi3)/2), np.sin((-phi1+np.sqrt(3)*phi2-phi3)/2)

        # Color charges
        Q1 = np.cos(phi1) * pip * pim
        Q2 = np.sin(phi1) * pip * pim
        Q3 = pi1
        Q4 = Cpp * pip * A + Cpm * pim * B
        Q5 = Spp * pip * A + Spm * pim * B
        Q6 = Cmp * pim * A - Cmm * pip * B
        Q7 = Smp * pim * A - Smm * pip * B
        Q8 = pi2
        q0 = np.array([Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8])
    elif su_group=='su2':
        # J = 1
        # J = 2
        J = 1.5
        phi, pi = np.random.uniform(), np.random.uniform(-J, J)
        # Bounded angles (0,2pi)
        # phi, pi = np.random.uniform(0, 2*np.pi), np.random.uniform(-J, J)
        Q1 = np.cos(phi) * np.sqrt(J**2 - pi**2)
        Q2 = np.sin(phi) * np.sqrt(J**2 - pi**2)
        Q3 = pi
        q0 = np.array([Q1, Q2, Q3])
    return q0

"""
    Interpolate fields
"""

def interpfield(x0, y0, Q0, fields):
    # Interpolate the values of trQE and trQB by finding the lattice cell in which the HQ is located and using the values of the neighbouring lattice points,
    # where the fields are evaluated, to extract the corresponding value where the HQ is at a given time
    x_low, x_high = math.floor(x0), math.ceil(x0)
    y_low, y_high = math.floor(y0), math.ceil(y0)
    xyhq = ([np.round(x0, decimals=6), np.round(y0, decimals=6)])
    points = np.array([[x_low, y_low], [x_low, y_high], [x_high, y_low], [x_high, y_high]])

    trQEx, trQEy, trQEeta = [], [], []
    trQBx, trQBy, trQBeta = [], [], []

    for point in points:
        fields.compute(Q0, point[0], point[1])
        trQEx.append(fields.trQE.real[0])
        trQEy.append(fields.trQE.real[1])
        trQEeta.append(fields.trQE.real[2])
        trQBx.append(fields.trQB.real[0])
        trQBy.append(fields.trQB.real[1])
        trQBeta.append(fields.trQB.real[2])

    trQEx_interp, trQEy_interp, trQEeta_interp = griddata(points, np.array(trQEx), xyhq,  method='cubic'), griddata(points, np.array(trQEy), xyhq,  method='cubic'), griddata(points, np.array(trQEeta), xyhq,  method='cubic')
    trQBx_interp, trQBy_interp, trQBeta_interp = griddata(points, np.array(trQBx), xyhq,  method='cubic'), griddata(points, np.array(trQBy), xyhq,  method='cubic'), griddata(points, np.array(trQBeta), xyhq,  method='cubic')

    return [trQEx_interp[0], trQEy_interp[0], trQEeta_interp[0]], [trQBx_interp[0], trQBy_interp[0], trQBeta_interp[0]]

def interppotential(x0, y0, axis, potentials):
    # Interpolate the values of the gauge potentials to where the HQ is, in a similar way as done for the electric and magnetic fields
    # Since Ax and Ay are extracted from lnUx and lnUy, they are evaluated at (x-a/2, y) and (x, y-a/2), whereas Aeta is computed at (x, y)
    if axis=='x':
        x_low = math.floor(x0-1/2)
        x_high = math.ceil(x0-1/2)
        y_low = math.floor(y0)
        y_high = math.ceil(y0)
        xyhq = ([np.round(x0-1/2, decimals=6), np.round(y0, decimals=6)])
    elif axis=='y':
        x_low = math.floor(x0)
        x_high = math.ceil(x0)
        y_low = math.floor(y0-1/2)
        y_high = math.ceil(y0-1/2)
        xyhq = ([np.round(x0, decimals=6), np.round(y0-1/2, decimals=6)])
    elif axis=='eta':
        x_low = math.floor(x0)
        x_high = math.ceil(x0)
        y_low = math.floor(y0)
        y_high = math.ceil(y0)
        xyhq = ([np.round(x0, decimals=6), np.round(y0, decimals=6)])

    points = np.array([[x_low, y_low], [x_low, y_high], [x_high, y_low], [x_high, y_high]])

    A = []
    for point in points:
        potentials.compute(axis, point[0], point[1])
        if axis=='x':
            A.append(potentials.Ax)
        elif axis=='y':
            A.append(potentials.Ay)
        elif axis=='eta':
            A.append(potentials.Aeta)

    A_interp = griddata(points, np.array(A), xyhq,  method='cubic')

    return A_interp[0]

# TODO: Impose boundary conditions
# def boundary(x1, y1, L, a):
#     if x1<0:
#         x1 = L-a
#     elif x1>(L-a):
#         x1 = a
#     if y1<0:
#         y1 = L-a
#     elif y1>(L-a):
#         y1 = a
#     return x1, y1

"""
    Update positions and momenta using basic Euler to solve the associated Wong's equations
"""

def update_coords(x0, y0, eta0, ptau0, px0, py0, peta0, tau_step):
    x1 = x0 + px0 / ptau0 * tau_step
    y1 = y0 + py0 / ptau0 * tau_step
    eta1 = eta0 + peta0 / ptau0 * tau_step

    return x1, y1, eta1

def update_momenta(ptau0, px0, py0, peta0, tau_step, current_tau, trQE, trQB, mass, E0):
    # Dynkin index from Tr{T^aT^b}=T_R\delta^{ab} in fundamental representation R=F
    # The minus sign comes from Tr{QF} from simulation being -Tr{QF} from the analytic formulas
    tr = -1/2

    px1 = px0 + tau_step / tr * (trQE[0] + trQB[2] * py0 / ptau0 - trQB[1] * peta0 * current_tau / ptau0)
    py1 = py0 + tau_step / tr * (trQE[1] - trQB[2] * px0 / ptau0 + trQB[0] * peta0 * current_tau / ptau0)
    peta1 = peta0 + tau_step * ((trQE[2] * ptau0 - trQB[0] * py0 + trQB[1] * px0) / tr  - 2 * peta0 * ptau0) / (current_tau * ptau0)
    ptau1 = np.sqrt(px1 ** 2 + py1 ** 2 + (current_tau * peta1) ** 2 + (mass/E0) ** 2)
    ptau2 = ptau0 + tau_step / tr * ((trQE[2]*peta0*current_tau + trQE[0]*px0 + trQE[1]*py0) - peta0 ** 2 * current_tau) / ptau0

    return ptau1, ptau2, px1, py1, peta1