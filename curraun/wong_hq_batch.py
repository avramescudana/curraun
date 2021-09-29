"""
    Computes various quantities used in solving Wong's equations for a HQ immersed in the fields of the Glasma.
"""

from curraun.numba_target import use_cuda, mycudajit, my_cuda_loop
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda
from curraun.wong_force_correlators import ElectricFields, LorentzForce, ForceCorrelators
import numpy as np
import logging, sys
from scipy.stats import unitary_group
# Supress Numba warnings
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
# Format logging messages, set level=logging.DEBUG or logging.INFO for more information printed out, logging.WARNING for basic info
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format='%(message)s')

# Define hbar * c in units of GeV * fm
hbarc = 0.197326 

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

    def compute(self, q0, xhq, yhq):
        u0 = self.s.d_u0

        aeta0 = self.s.d_aeta0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_cuda_loop(fields_kernel, q0, xhq, yhq, n, u0, aeta0, peta1, peta0, pt1, pt0, t, self.d_trQE, self.d_trQB)

        if use_cuda:
            self.copy_to_host()


# kernels
@mycudajit
def fields_kernel(q0, xhq, yhq, n, u0, aeta0, peta1, peta0, pt1, pt0, tau, trQE, trQB):

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

    Q0 = su.get_algebra_element(q0)

    trQE[0] = su.tr(su.mul(Q0, Ex)).real
    trQE[1] = su.tr(su.mul(Q0, Ey)).real
    trQE[2] = su.tr(su.mul(Q0, Eeta)).real

    trQB[0] = su.tr(su.mul(Q0, Bx)).real
    trQB[1] = su.tr(su.mul(Q0, By)).real
    trQB[2] = su.tr(su.mul(Q0, Beta)).real


class ColorChargeGaugePotentials:
    def __init__(self, s, q0):
        self.s = s

        self.Q = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.C = np.zeros(su.CASIMIRS, dtype=su.GROUP_TYPE)
        self.q = np.zeros(su.ALGEBRA_ELEMENTS, dtype=su.GROUP_TYPE)
        my_cuda_loop(initial_charge_kernel, q0, self.Q, self.C)
        self.d_Q, self.d_C, self.d_q = self.Q, self.C, self.q

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
        self.d_Q, self.d_C, self.d_q = cuda.to_device(self.Q), cuda.to_device(self.C), cuda.to_device(self.q)
        self.d_Ax = cuda.to_device(self.Ax)
        self.d_Ay = cuda.to_device(self.Ay)
        self.d_Aeta = cuda.to_device(self.Aeta)

    def copy_to_host(self):
        self.d_Q.copy_to_host(self.Q)
        self.d_C.copy_to_host(self.C)
        self.d_q.copy_to_host(self.q)
        self.d_Ax.copy_to_host(self.Ax)
        self.d_Ay.copy_to_host(self.Ay)
        self.d_Aeta.copy_to_host(self.Aeta)

    def evolve(self, q0, tau_step, ptau0, px0, py0, peta0, xhq, yhq, xahq, yahq):
        u0 = self.s.d_u0
        aeta0 = self.s.d_aeta0
        n = self.s.n

        my_cuda_loop(Ax_kernel, self.d_Ax, xahq, yhq, n, u0)
        my_cuda_loop(Ay_kernel, self.d_Ay, xhq, yahq, n, u0)
        my_cuda_loop(Aeta_kernel, self.d_Aeta, xhq, yhq, n, aeta0)

        my_cuda_loop(evolve_charge_gauge_fields_kernel, q0, tau_step, ptau0, px0, py0, peta0, self.d_Ax, self.d_Ay, self.d_Aeta, self.d_Q, self.d_C, self.d_q)

        if use_cuda:
            self.copy_to_host()

@mycudajit
def Ax_kernel(Ax, xhq, yhq, n, u0):
    poshq = l.get_index(xhq, yhq, n)
    su.store(Ax, su.mlog(u0[poshq, 0, :]))

@mycudajit
def Ay_kernel(Ay, xhq, yhq, n, u0):
    poshq = l.get_index(xhq, yhq, n)
    su.store(Ay, su.mlog(u0[poshq, 1, :]))

@mycudajit
def Aeta_kernel(Aeta, xhq, yhq, n, aeta0):
    poshq = l.get_index(xhq, yhq, n)
    su.store(Aeta, aeta0[poshq, :])


@mycudajit
def evolve_charge_gauge_fields_kernel(q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta, Q, C, q1):
    Q0 = su.get_algebra_element(q0)
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
    C[:] = su.casimir(Q[:])
    q1[:] = su.get_algebra_factors_from_group_element_approximate(Q[:])


class ColorChargeWilsonLines:
    def __init__(self, s, q0):
        self.s = s

        self.Q = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        self.C = np.zeros(su.CASIMIRS, dtype=su.GROUP_TYPE)
        self.q = np.zeros(su.ALGEBRA_ELEMENTS, dtype=su.GROUP_TYPE)
        my_cuda_loop(initial_charge_kernel, q0, self.Q, self.C)

        self.w = np.zeros(su.GROUP_ELEMENTS, dtype=su.GROUP_TYPE)
        my_cuda_loop(init_wilson_line_kernel, self.w)

        self.d_Q, self.d_C = self.Q, self.C
        self.d_w = self.w
        self.d_q = self.q

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_Q, self.d_C = cuda.to_device(self.Q), cuda.to_device(self.C)
        self.d_w = cuda.to_device(self.w)
        self.d_q = cuda.to_device(self.q)

    def copy_to_host(self):
        self.d_Q.copy_to_host(self.Q)
        self.d_C.copy_to_host(self.C)
        self.d_w.copy_to_host(self.w)
        self.d_q.copy_to_host(self.q)

    def evolve(self, q0, xhq, xhq0, yhq, yhq0, delta_etahq):

        u0 = self.s.d_u0
        n = self.s.n
        aeta0 = self.s.d_aeta0

        my_cuda_loop(init_wilson_line_kernel, self.d_w)

        if (xhq>xhq0):
            my_cuda_loop(Ux_up, self.d_w, xhq, yhq, n, u0)
        elif (xhq<xhq0):
            my_cuda_loop(Ux_down, self.d_w, xhq, yhq, n, u0)

        if (yhq>yhq0):
            my_cuda_loop(Uy_up, self.d_w, xhq, yhq, n, u0)
        elif (yhq<yhq0):
            my_cuda_loop(Uy_down, self.d_w, xhq, yhq, n, u0)

        my_cuda_loop(Ueta, self.d_w, xhq, yhq, n, delta_etahq, aeta0)

        my_cuda_loop(evolve_charge_wilson_lines_kernel, self.d_w, q0, self.d_Q, self.d_C, self.d_q)

        if use_cuda:
            self.copy_to_host()

@mycudajit
def initial_charge_kernel(q0, Q, C):
    Q[:] = su.get_algebra_element(q0)
    C[:] = su.casimir(Q[:])

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
def evolve_charge_wilson_lines_kernel(w, q0, Q, C, q1):
    Q0 = su.get_algebra_element(q0)
    buf = l.act(su.dagger(w), Q0)
    su.store(Q, buf)
    C[:] = su.casimir(Q[:])
    q1[:] = su.get_algebra_factors_from_group_element_approximate(Q[:])


"""
    Initialize positions, momenta and color charges
""" 


def initial_coords(p):
    L = p['L']
    a = L / p['N']
    DT = 1.0 / p['DTS']
    formt = int(p['TFORM'] / a * p['DTS'])
    tau0 = formt * DT
    x0, y0 = np.random.uniform(0, L-a), np.random.uniform(0, L-a)
    eta0 = 0
    xmu0 = [tau0, x0, y0, eta0]
    return xmu0

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

def fonll(p, pt):
    quark = p['QUARK']
    if quark=='charm':
        x0, x1, x2, x3 = 20.2837, 1.95061, 3.13695, 0.0751663
    elif quark=='beauty':
        x0, x1, x2, x3 = 0.467997, 1.83805, 3.07569, 0.0301554

    fonll = 2*np.pi*x0/(1+x3*pt**x1)**x2
    return fonll

def initial_momenta_fonll(p):
    nq, ntp = p['NQ'], p['NTP']
    Npairs = nq * ntp
    ptbins, ptmax = 20, 12
    pt = np.linspace(0, ptmax, ptbins)
    fonll_pt = fonll(p, pt)
    prob_pt = fonll_pt/sum(fonll_pt)
    N = []
    for prob in prob_pt:
        N.append(int(round(Npairs*prob)))
    p['PTFONLL'], p['NFONLL'] = pt, N

# gell-mann matrices
gm = [
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1/ np.sqrt(3), 0, 0], [0, 1/ np.sqrt(3), 0], [0, 0, -2/ np.sqrt(3)]]
]

T = np.array(gm) / 2.0

def initial_charge(p):
    su_group = p['GROUP']
    if su_group=='su3':
        """
        Step 1: create a random SU(3) charge with the correct q2 and q3 values.
        Note: This method is biased and leads to <Q> != 0.
        """
        # J1, J2 = 2.84801, 1.00841
        J1, J2 = 3, 3
        K1, K2 = (2*J1+J2)/3, (2*J2+J1)/3
        x, y = np.random.uniform(K2-K1, K1), np.random.uniform(K1-K2, K2)
        A1, A2, A3 = K1-K2+x, K2+x, K1-x
        B1, B2, B3 = K2-K1+y, K2-y, K1+y
        numA, numB = A1*A2*A3, B1*B2*B3
        pi2, pi3 = (x-y)*np.sqrt(3)/2, (x+y)/2
        pi1 = np.random.uniform(-pi3, pi3)
        A, B = np.sqrt(numA)/(2*pi3), np.sqrt(numB)/(2*pi3)

        # Angle Darboux variables
        # phi1, phi2, phi3 = np.random.uniform(), np.random.uniform(), np.random.uniform()
        # Bounded angles (0,2pi)
        phi1, phi2, phi3 = np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)

        pip, pim = np.sqrt(pi3+pi1), np.sqrt(pi3-pi1)
        Cpp, Cpm, Cmp, Cmm = np.cos((+phi1+np.sqrt(3)*phi2+phi3)/2), np.cos((+phi1+np.sqrt(3)*phi2-phi3)/2), \
                            np.cos((-phi1+np.sqrt(3)*phi2+phi3)/2), np.cos((-phi1+np.sqrt(3)*phi2-phi3)/2)
        Spp, Spm, Smp, Smm = np.sin((+phi1+np.sqrt(3)*phi2+phi3)/2), np.sin((+phi1+np.sqrt(3)*phi2-phi3)/2), \
                            np.sin((-phi1+np.sqrt(3)*phi2+phi3)/2), np.sin((-phi1+np.sqrt(3)*phi2-phi3)/2)

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
        
        # convert to matrix
        Q0 = np.einsum('ijk,i', T, q0)
        
        
        """
            Step 2: create a random SU(3) matrix to rotate Q.
        """
        V = unitary_group.rvs(3)
        detV = np.linalg.det(V)
        U = V / detV ** (1/3)
        Ud = np.conj(U).T
        
        Q = np.einsum('ab,bc,cd', U, Q0, Ud)
        
        """
            Step 3: Project onto color components
        """
        
        q = 2*np.einsum('ijk,kj', T, Q)
        q0 = np.real(q)

        # """
        #     New step 1: specific random color vector
        # """
        # q = [0., 0., 0., 0., -1.69469, 0., 0., -1.06209]
        # Q0 = np.einsum('ijk,i', T, q)
        
        
        
        # """
        #     Step 2: create a random SU(3) matrix to rotate Q.
        # """
        # V = unitary_group.rvs(3)
        # detV = np.linalg.det(V)
        # U = V / detV ** (1/3)
        # Ud = np.conj(U).T
        
        # Q = np.einsum('ab,bc,cd', U, Q0, Ud)
        
        # """
        #     Step 3: Project onto color components
        # """
        
        # q0 = 2*np.einsum('ijk,kj', T, Q)

    elif su_group=='su2':
        J = np.sqrt(1.5)
        # phi, pi = np.random.uniform(), np.random.uniform(-J, J)
        # Bounded angles (0,2pi)
        phi, pi = np.random.uniform(0, 2*np.pi), np.random.uniform(-J, J)
        Q1 = np.cos(phi) * np.sqrt(J**2 - pi**2)
        Q2 = np.sin(phi) * np.sqrt(J**2 - pi**2)
        Q3 = pi
        q0 = np.array([Q1, Q2, Q3])
    return q0


"""
    Update positions and momenta using basic Euler to solve the associated Wong's equations
"""

def update_coords(p, x0, y0, eta0, ptau0, px0, py0, peta0, tau_step):
    N = p['N']
    x1 = x0 + px0 / ptau0 * tau_step
    y1 = y0 + py0 / ptau0 * tau_step
    eta1 = eta0 + peta0 / ptau0 * tau_step
    x1, y1 = boundary(x1, y1, N)

    return x1, y1, eta1

def boundary(x1, y1, N):
    if x1<0:
        x1 = N-1
    elif x1>(N-1):
        x1 = 0
    if y1<0:
        y1 = N-1
    elif y1>(N-1):
        y1 = 0
    return x1, y1

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


"""
    Routine to numerically solve Wong's equations for positions, momenta and color charge for a quark or heavy quark test particle
"""

def solve_wong(s, p, t, xmu0, pmu0, q0, xmu, pmu, fields, charge, tag, constraint, casimirs, correlators, fieldsform, electric_fields, lorentz_force, force_correlators):

    a = p['L'] / p['N']
    DT = 1.0 / p['DTS']
    E0 = p['N'] / p['L'] * hbarc
    formt = int(p['TFORM'] / a * p['DTS'])
    mass = p['MASS']

    current_tau = t * DT
    tau_step = DT

    x0, y0, eta0 = xmu0[1]/a, xmu0[2]/a, xmu0[3]
    ptau0, px0, py0, peta0 = pmu0[0]/E0, pmu0[1]/E0, pmu0[2]/E0, pmu0[3]*a/E0
    xhq0, yhq0 = int(round(x0)), int(round(y0))


    if t==formt:
        fields[tag] = WongFields(s)
        if p['NUM_CHECKS']:
            constraint[tag], casimirs[tag] = [], []
        if p['FORCE_CORR']:
            tags_corr = ['naive', 'transported']
            for tag_corr in tags_corr:
                correlators['EformE'][tag_corr][tag], correlators['FformF'][tag_corr][tag] = [], []
            electric_fields[tag] = ElectricFields(s)
            fieldsform['E'][tag] = electric_fields[tag].compute(xhq0, yhq0)
            lorentz_force[tag] = LorentzForce(s)
            fieldsform['F'][tag] = lorentz_force[tag].compute(xhq0, yhq0, ptau0, px0, py0, peta0, current_tau)
            force_correlators[tag] = ForceCorrelators(s)
        
        xmu[tag], pmu[tag] = [], []
        xmu[tag].append([a*current_tau, a*x0, a*y0, eta0])
        pmu[tag].append([E0*ptau0, E0*px0, E0*py0, E0/a*peta0])
        logging.debug("Coordinates: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(a*current_tau, a*x0, a*y0, eta0))
        logging.debug("Momenta: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(E0*ptau0, E0*px0, E0*py0, E0/a*peta0))

        if p['SOLVEQ']=='gauge potentials':
            charge[tag] = ColorChargeGaugePotentials(s, q0)
        else:
            charge[tag] = ColorChargeWilsonLines(s, q0)

        if p['NUM_CHECKS']:
            C = charge[tag].C.real                        
            logging.debug("Quadratic Casimir: {:3.5f}".format(C[0]))
            if p['GROUP']=='su2':
                casimirs[tag].append(C[0])
            elif p['GROUP']=='su3':
                casimirs[tag].append([C[0], C[1]])
                logging.debug("Cubic Casimir: {:3.5f}".format(C[1]))

    elif t>formt:
        # Solve Wong's equations using basic Euler
        # Update positions
        x1, y1, eta1 = update_coords(p, x0, y0, eta0, ptau0, px0, py0, peta0, tau_step)

        # Convert to physical units
        xmu[tag].append([a*current_tau, a*x1, a*y1, eta1])

        # Approximate the position of the quark with closest lattice point
        # Locations where transverse gauge fields extracted from gauge links are evaluated, in the middle of lattice sites
        xhq0, yhq0 = int(round(xmu[tag][t-formt-1][1]/a)), int(round(xmu[tag][t-formt-1][2]/a))
        xhq, yhq = int(round(x1)), int(round(y1))

        fields[tag].compute(q0, xhq, yhq)
        trQE, trQB = fields[tag].trQE.real, fields[tag].trQB.real

        # Update momenta using Euler, with fields evaluated at nearest lattice points
        ptau1, ptau2, px1, py1, peta1 = update_momenta(ptau0, px0, py0, peta0, tau_step, current_tau, trQE, trQB, mass, E0)

        # Convert to physical units
        pmu[tag].append([E0*ptau1, E0*px1, E0*py1, E0/a*peta1])
        if p['NUM_CHECKS']:
            constraint[tag].append(E0*(ptau2-ptau1))

        if p['SOLVEQ']=='gauge potentials':
            xahq, yahq = int(round(x1-1/2)), int(round(y1-1/2))
            charge[tag].evolve(q0, tau_step, ptau0, px0, py0, peta0, xhq, yhq, xahq, yahq)
        else:
            delta_etahq = eta1-eta0
            charge[tag].evolve(q0, xhq, xhq0, yhq, yhq0, delta_etahq)

        q1 = charge[tag].q
        if p['NUM_CHECKS']:
            C = charge[tag].C.real
            logging.debug("Quadratic Casimir: {:3.5f}".format(C[0]))
            if p['GROUP']=='su2':
                casimirs[tag].append(C[0])
            elif p['GROUP']=='su3':
                casimirs[tag].append([C[0], C[1]])
                logging.debug("Cubic Casimir: {:3.5f}".format(C[1]))

        if p['FORCE_CORR']:
            # [(GeV / fm) ** 2]
            units = (E0 ** 2 / hbarc) ** 2 / p['G'] ** 2

            E = electric_fields[tag].compute(xhq, yhq)
            F = lorentz_force[tag].compute(xhq, yhq, ptau0, px0, py0, peta0, current_tau)

            EformE, FformF  = {}, {}
            tags_corr = ['naive', 'transported']
            for tag_corr in tags_corr:
                force_correlators[tag].compute(tag_corr, fieldsform['E'][tag], E, xhq, xhq0, yhq, yhq0, delta_etahq)
                EformE[tag_corr] = force_correlators[tag].fformf * units
                correlators['EformE'][tag_corr][tag].append(EformE[tag_corr][0]+EformE[tag_corr][1]+EformE[tag_corr][2])

                force_correlators[tag].compute(tag_corr, fieldsform['F'][tag], F, xhq, xhq0, yhq, yhq0, delta_etahq)
                FformF[tag_corr] = force_correlators[tag].fformf * units
                correlators['FformF'][tag_corr][tag].append(FformF[tag_corr][0]+FformF[tag_corr][1]+FformF[tag_corr][2])

        # Convert to physical units
        logging.debug("Coordinates: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(a*current_tau, a*x1, a*y1, eta1))
        logging.debug("Momenta: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(E0*ptau1, E0*px1, E0*py1, E0/a*peta1))
        logging.debug("Ptau constraint: {:.3e}".format(E0*ptau2-E0*ptau1)) 

        pT0, pT = np.sqrt(pmu[tag][0][1]**2+pmu[tag][0][2]**2), E0*np.sqrt(px1**2+py1**2)
        xT0, xT = np.sqrt(xmu[tag][0][1]**2+xmu[tag][0][2]**2), a*np.sqrt(x1**2+y1**2)
        logging.debug("Transverse coordinate variance: {:.3e}".format((xT-xT0)**2))
        logging.debug("Transverse momentum variance: {:3.3f}".format((pT-pT0)**2))

        xmu1 = [current_tau, a*x1, a*y1, eta1]
        pmu1 = [E0*ptau1, E0*px1, E0*py1, E0*peta1/a]

        return xmu1, pmu1, q1