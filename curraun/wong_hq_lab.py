"""
    Computes various quantities used in solving Wong's equations for a HQ immersed in the fields of the Glasma.
"""

from curraun.numba_target import use_cuda, mycudajit, my_cuda_loop
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda
from curraun.wong_hq_batch import ColorChargeWilsonLines
import numpy as np
import logging, sys
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

    def compute(self, q0, xhq, yhq, sheta, cheta):
        u0 = self.s.d_u0

        aeta0 = self.s.d_aeta0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_cuda_loop(fields_kernel, q0, xhq, yhq, sheta, cheta, n, u0, aeta0, peta1, peta0, pt1, pt0, t, self.d_trQE, self.d_trQB)

        if use_cuda:
            self.copy_to_host()


# kernels
@mycudajit
def fields_kernel(q0, xhq, yhq, sheta, cheta, n, u0, aeta0, peta1, peta0, pt1, pt0, tau, trQE, trQB):

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

    Q0 = su.get_algebra_element(q0)

    trQE[0] = su.tr(su.mul(Q0, Exlab)).real
    trQE[1] = su.tr(su.mul(Q0, Eylab)).real
    trQE[2] = su.tr(su.mul(Q0, Ez)).real

    trQB[0] = su.tr(su.mul(Q0, Bxlab)).real
    trQB[1] = su.tr(su.mul(Q0, Bylab)).real
    trQB[2] = su.tr(su.mul(Q0, Bz)).real


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
    x0, y0 = np.random.uniform(L/3, 2*L/3), np.random.uniform(L/3, 2*L/3)
    z0 = 0
    xmu0 = [tau0, x0, y0, z0]
    return xmu0

# TODO: Initialise with FONLL pQCD distribution in momentum of HQs
def initial_momenta(p):
    pT = p['PT']
    tau_form = p['TFORM']
    mass = p['MASS']
    px0 = np.random.uniform(0, pT)
    py0 = np.sqrt(pT ** 2 - px0 ** 2)
    pz0 = 0
    pt0 = np.sqrt(px0 ** 2 + py0 ** 2 + (tau_form * pz0) ** 2 + mass ** 2)
    pmu0 = [pt0, px0, py0, pz0]
    return pmu0

def initial_charge(p):
    su_group = p['GROUP']
    if su_group=='su3':
        # Values used to compute the SU(3) Casimirs
        # J1, J2 = 1, 0
        J1, J2 = 2.84801, 1.00841
        # Angle Darboux variables
        phi1, phi2, phi3 = np.random.uniform(), np.random.uniform(), np.random.uniform()
        # phi1, phi2, phi3 = np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)

        search = True
        while search:
            pi2, pi3 = np.random.uniform(), np.random.uniform()
            pi1 = np.random.uniform(-pi3, pi3)

            numA = ((J1-J2)/3+pi3+pi2/np.sqrt(3))*((J1+2*J2)/3+pi3+pi2/np.sqrt(3))*((2*J1+J2)/3-pi3-pi2/np.sqrt(3))
            numB = ((J2-J1)/3+pi3-pi2/np.sqrt(3))*((J1+2*J2)/3-pi3+pi2/np.sqrt(3))*((2*J1+J2)/3+pi3-pi2/np.sqrt(3))

            if (numA>0) & (numB>0):
                search = False

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

def update_coords(p, x0, y0, z0, pt0, px0, py0, pz0, t_step):
    N = p['N']
    x1 = x0 + px0 / pt0 * t_step
    y1 = y0 + py0 / pt0 * t_step
    z1 = z0 + pz0 / pt0 * t_step
    x1, y1 = boundary(x1, y1, N)

    return x1, y1, z1

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
    pt2 = pt0 + t_step/tr * (trQE[0]*px0 + trQE[1]*py0 + trQE[2]*pz0)/pt0

    return pt1, pt2, p[0], p[1], p[2]

"""
    Routine to numerically solve Wong's equations for positions, momenta and color charge for a quark or heavy quark test particle
"""

def solve_wong(s, p, t, xmu0, pmu0, q0, xmu, pmu, fields, charge, tag, constraint, casimirs):

    a = p['L'] / p['N']
    DT = 1.0 / p['DTS']
    E0 = p['N'] / p['L'] * hbarc
    formt = int(p['TFORM'] / a * p['DTS'])
    mass = p['MASS']

    current_tau = t * DT
    tau_step = DT


    x0, y0, z0 = xmu0[1]/a, xmu0[2]/a, xmu0[3]/a
    pt0, px0, py0, pz0 = pmu0[0]/E0, pmu0[1]/E0, pmu0[2]/E0, pmu0[3]/E0

    current_t = np.sqrt(current_tau**2+z0**2)
    previous_t = np.sqrt((current_tau-tau_step)**2+z0**2)
    t_step = current_t - previous_t

    eta0 = np.log((previous_t+z0)/(previous_t-z0))/2

    if t==formt:
        fields[tag] = WongFields(s)
        if p['NUM_CHECKS']:
            constraint[tag], casimirs[tag] = [], []
        
        xmu[tag], pmu[tag] = [], []
        xmu[tag].append([a*current_tau, a*x0, a*y0, a*z0])
        pmu[tag].append([E0*pt0, E0*px0, E0*py0, E0*pz0])
        logging.debug("Coordinates: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(a*current_tau, a*x0, a*y0, a*z0))
        logging.debug("Momenta: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(E0*pt0, E0*px0, E0*py0, E0*pz0))

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
        x1, y1, z1 = update_coords(p, x0, y0, z0, pt0, px0, py0, pz0, t_step)
        eta1 = np.log((current_t+z1)/(current_t-z1))/2

        # Convert to physical units
        xmu[tag].append([a*current_tau, a*x1, a*y1, a*z1])

        # Approximate the position of the quark with closest lattice point
        # Locations where transverse gauge fields extracted from gauge links are evaluated, in the middle of lattice sites
        xhq0, yhq0 = int(round(xmu[tag][t-formt-1][1]/a)), int(round(xmu[tag][t-formt-1][2]/a))
        xhq, yhq = int(round(x1)), int(round(y1))

        fields[tag].compute(q0, xhq, yhq, np.sinh(eta1), np.cosh(eta1))
        trQE, trQB = fields[tag].trQE.real, fields[tag].trQB.real

        # Update momenta using Euler, with fields evaluated at nearest lattice points
        pt1, pt2, px1, py1, pz1 = update_momenta(pt0, px0, py0, pz0, t_step, trQE, trQB, mass, E0)
        pmu[tag].append([E0*pt1, E0*px1, E0*py1, E0*pz1])
        if p['NUM_CHECKS']:
            constraint[tag].append(E0*(pt2-pt1))


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

        # Convert to physical units
        logging.debug("Coordinates: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(a*current_tau, a*x1, a*y1, a*z1))
        logging.debug("Momenta: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(E0*pt1, E0*px1, E0*py1, E0*pz1))

        pT0, pT = np.sqrt(pmu[tag][0][1]**2+pmu[tag][0][2]**2), E0*np.sqrt(px1**2+py1**2)
        xT0, xT = np.sqrt(xmu[tag][0][1]**2+xmu[tag][0][2]**2), a*np.sqrt(x1**2+y1**2)
        logging.debug("Transverse coordinate variance: {:.3e}".format((xT-xT0)**2))
        logging.debug("Transverse momentum variance: {:3.3f}".format((pT-pT0)**2))

        xmu1 = [current_t, a*x1, a*y1, a*z1]
        pmu1 = [E0*pt1, E0*px1, E0*py1, E0*pz1]

        return xmu1, pmu1, q1