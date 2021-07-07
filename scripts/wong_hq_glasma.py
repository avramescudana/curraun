import numpy as np
import math
from scipy.interpolate import griddata
import time
import argparse

"""
    Default simulation parameters chosen for Pb-Pb at 5.02 TeV
"""

# Numerical parameters
L = 10      # Length of simulation box [fm]
N = 512     # Number of lattice sites
tau_s = 2.0     # Simulation time [fm/c]
DTS = 8     # Time step

# Heavy quark related parameters, chosen here for a charm quark
mass = 1.5      # Heavy quark mass [GeV]
tau_form = 0.06     # Formation time [fm/c]
nq = 15     # Number of heavy quarks

# MV model parameters for Pb-Pb at 5.02 TeV
A = 207     # Mass number
sqrts = 5020        # Center-of-mass energy [GeV]
ns = 50     # Number of color sheets
factor = 0.8        # Ratio between Qs/g^2\mu for Ns = 50 color sheets
Qs = np.sqrt(0.13 * A**(1/3) * sqrts**0.25)         # Saturation momentum [GeV]	
g = np.pi * np.sqrt(1 / np.log(Qs / 0.2))           # Running coupling constant		
mu = Qs / (g**2 * factor)           # MV model parameter	
m = 0.1 * g**2 * mu         # Infrared regulator [GeV]


"""
    Dictionary with standard MV model paramaters
"""
p = {
    # Parameters for simulation box
    'L':    L,           # transverse size [fm]
    'N':    N,            # lattice size
    'DTS':  DTS,             # time steps per transverse spacing
    'TMAX': tau_s,          # max. proper time (tau) [fm/c]

    # Parameters for MV model
    'G':    g,            # YM coupling constant
    'MU':   mu,             # MV model parameter [GeV]
    'M':    m,              # IR regulator [GeV]
    'UV':   10.0,           # UV regulator [GeV]
    'NS':   ns,             # number of color sheets
    
    # Parameters for heavy quarks
    'MASS': mass,           # mass of HQ [GeV]
    'TFORM': tau_form,       # formation time of the HQ [fm/c]
}

"""
    Argument parsing
"""
parser = argparse.ArgumentParser(description='Compute momentum broadening of HQs in the Glasma.')

# parser.add_argument('-L',    type=float, help="Transverse lattice size [fm]")
# parser.add_argument('-N',    type=int,   help="Number of lattice sites")
# parser.add_argument('-DTS',  type=int,   help="Time steps per transverse spacing")
# parser.add_argument('-TMAX', type=float, help="Maximum proper time [fm/c]")
# parser.add_argument('-G',    type=float, help="YM coupling constant")
# parser.add_argument('-MU',   type=float, help="MV model parameter [GeV]")
# parser.add_argument('-M',    type=float, help="IR regulator [GeV]")
# parser.add_argument('-UV',   type=float, help="UV regulator [GeV]")
# parser.add_argument('-NS',   type=int,   help="Number of color sheets")
# parser.add_argument('-MASS',   type=float,   help="Mass of heavy quark [GeV]")
# parser.add_argument('-TFORM',   type=float,   help="Formation time of heavy quark [fm/c]")

# # Parse argumentss and update parameters dictionary
# args = parser.parse_args()
# data = args.__dict__
# for d in data:
#     if data[d] is not None:
#         p[d] = data[d]

parser.add_argument('-su', type=str, help="Gauge group", default='su2')
parser.add_argument('-pT', type=float, help="Initial transverse momentum [GeV]", default=0.5)
# TODO: Initialise with FONLL distribution in momentum of HQs
parser.add_argument('-interp', type=str, help="Interpolate fields.", default='yes')
parser.add_argument('-noffset', type=int, help="Test particles offset", default=0)
parser.add_argument('-ntp', type=int, help="Number of test particles", default=10)
parser.add_argument('-system',   type=str,   help="Collision system", default='pb+pb_5020gev')
args = parser.parse_args()
su_group, pT, interp, noffset, ntp, system = args.su, args.pT, args.interp, args.noffset, args.ntp, args.system

# Set environment variables 
import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["PRECISION"] = "double"
if su_group == 'su2':
    os.environ["GAUGE_GROUP"] = 'su2_complex'
    Ng = 4
elif su_group == 'su3':
    os.environ["GAUGE_GROUP"] = su_group
    Ng = 9
Nm = np.int(np.sqrt(Ng))

# Import relevant modules
import curraun.core as core
import curraun.mv as mv
import curraun.initial as initial
initial.DEBUG = False
from curraun.numba_target import use_cuda, use_numba
if use_cuda:
    from numba import cuda
from curraun.wong_hq import WongFields, WongPotentials, InitialColorCharge, ColorChargeEvolve

def initial_coords():
    # x0, y0 = np.random.uniform(0, L), np.random.uniform(0, L)
    # TODO: Impose periodic boundary conditions
    # Since the HQ doesn't move too much, it suffices to place it somewhere within the interior of the simulation plane
    x0, y0 = np.random.uniform(L/3, 2*L/3), np.random.uniform(L/3, 2*L/3)
    eta0 = 0
    xmu0 = [x0, y0, eta0]
    return xmu0

def initial_momenta():
    px0 = np.random.uniform(0, pT)
    py0 = np.sqrt(pT ** 2 - px0 ** 2)
    peta0 = 0
    ptau0 = np.sqrt(px0 ** 2 + py0 ** 2 + (tau_form * peta0) ** 2 + mass ** 2)
    pmu0 = [ptau0, px0, py0, peta0]
    return pmu0

def initial_charge():
    if su_group=='su3':
        # TODO: Find the correct way to initialise the SU(3) color charges
        # Values used to compute the SU(3) and SU(2) Casimirs
        # J1, J2 = 1, 0
        J1, J2 = 2.84801, 1.00841
        # Angle Darboux variables
        phi1, phi2, phi3 = np.random.uniform(), np.random.uniform(), np.random.uniform()
        # phi1, phi2, phi3 = np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)
        # Momenta Darboux variables
        pi3 = np.random.uniform(0, (J1+J2)/2)
        pi2 = np.random.uniform((J2-J1)/np.sqrt(3), (J1-J2)/(2*np.sqrt(3)))
        pi1 = np.random.uniform(-pi3, pi3)

        pip, pim = np.sqrt(pi3+pi1), np.sqrt(pi3-pi1)
        Cpp, Cpm, Cmp, Cmm = np.cos((phi1+np.sqrt(3)*phi2+phi3)/2), np.cos((phi1+np.sqrt(3)*phi2-phi3)/2), np.cos((-phi1+np.sqrt(3)*phi2+phi3)/2), np.cos((-phi1+np.sqrt(3)*phi2-phi3)/2)
        Spp, Spm, Smp, Smm = np.sin((phi1+np.sqrt(3)*phi2+phi3)/2), np.sin((phi1+np.sqrt(3)*phi2-phi3)/2), np.sin((-phi1+np.sqrt(3)*phi2+phi3)/2), np.sin((-phi1+np.sqrt(3)*phi2-phi3)/2)
        A = np.sqrt(((J1-J2)/3+pi3+pi2/np.sqrt(3))*((J1+2*J2)/3+pi3+pi2/np.sqrt(3))*((2*J1+J2)/3-pi3-pi2/np.sqrt(3)))/(2*pi3)
        B = np.sqrt(((J2-J1)/3+pi3-pi2/np.sqrt(3))*((J1+2*J2)/3-pi3+pi2/np.sqrt(3))*((2*J1+J2)/3+pi3-pi2/np.sqrt(3)))/(2*pi3)

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
        J = 1
        phi, pi = np.random.uniform(), np.random.uniform(-J, J)
        # Bounded angles (0,2pi)
        # phi, pi = np.random.uniform(0, 2*np.pi), np.random.uniform(-J, J)
        Q1 = np.cos(phi) * np.sqrt(J**2 - pi**2)
        Q2 = np.sin(phi) * np.sqrt(J**2 - pi**2)
        Q3 = pi
        q0 = np.array([Q1, Q2, Q3])
    return q0

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

# Define hbar * c in units of GeV * fm
hbarc = 0.197326 

# Simulation function
def simulate(p, xmu0, pmu0, q0):    
    # Derived parameters
    a = p['L'] / p['N']
    E0 = p['N'] / p['L'] * hbarc
    p['E0'] = E0
    DT = 1.0 / p['DTS']
    maxt = int(p['TMAX'] / a * p['DTS'])
    formt = int(p['TFORM'] / a * p['DTS'])
    mass = p['MASS']

    # Normal units: x/y [fm], eta [1], px/py [GeV], peta [GeV/fm]
    # Lattice units
    x0, y0, eta0 = xmu0[0]/a, xmu0[1]/a, xmu0[2]
    ptau0, px0, py0, peta0 = pmu0[0]/E0, pmu0[1]/E0, pmu0[2]/E0, pmu0[3]*a/E0

    print("Initializating ...")

    s = core.Simulation(p['N'], DT, p['G'])
    va = mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
    vb = mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
    initial.init(s, va, vb)

    fields = WongFields(s)
    potentials = WongPotentials(s)
    charge_initial = InitialColorCharge(s)
    charge_evolve = ColorChargeEvolve(s)

    if use_cuda:
        s.copy_to_device()

    xmu, pmu, constraint, qsq = [], [], [], []

    for t in range(maxt):
        core.evolve_leapfrog(s)

        if t>=formt:  
            current_tau = t * DT
            tau_step = DT
            
            if t==formt:
                xmu.append([a*current_tau, a*x0, a*y0, eta0])
                pmu.append([E0*ptau0, E0*px0, E0*py0, E0/a*peta0])
                # print('xmu ', [a*current_tau, a*x0, a*y0, eta0])
                # print('pmu ', [E0*ptau0, E0*px0, E0*py0, E0/a*peta0])

                charge_initial.compute(q0)
                Q0 = charge_initial.Q
                Qsq0 = charge_initial.Q2[0].real
                qsq.append(Qsq0)
                # print('Qsq ', Qsq0)

            # Solve Wong's equations using basic Euler
            # Update positions
            x1 = x0 + px0 / ptau0 * tau_step
            y1 = y0 + py0 / ptau0 * tau_step
            eta1 = eta0 + peta0 / ptau0 * tau_step

            # Convert to physical units
            xmu.append([a*current_tau, a*x1, a*y1, eta1])

            # Dynkin index from Tr{T^aT^b}=T_R\delta^{ab} in fundamental representation R=F
            tr = -1/2
            if interp=='yes':
                # print('Interpolating fields...')
                
                [trQEx_interp, trQEy_interp, trQEeta_interp], [trQBx_interp, trQBy_interp, trQBeta_interp] = interpfield(x0, y0, Q0, fields)
                Ax_interp, Ay_interp, Aeta_interp = interppotential(x0, y0, 'x', potentials), interppotential(x0, y0, 'y', potentials), interppotential(x0, y0, 'eta', potentials)

                px1 = px0 + tau_step / tr * (trQEx_interp + trQBeta_interp * py0 / ptau0 - trQBy_interp * peta0 * current_tau / ptau0)
                py1 = py0 + tau_step / tr * (trQEy_interp - trQBeta_interp * px0 / ptau0 + trQBx_interp * peta0 * current_tau / ptau0)
                peta1 = peta0 + tau_step * ((trQEeta_interp * ptau0 - trQBx_interp * py0 + trQBy_interp * px0) / tr  - 2 * peta0 * ptau0) / (current_tau * ptau0)
                ptau1 = np.sqrt(px1 ** 2 + py1 ** 2 + (current_tau * peta1) ** 2 + (mass/E0) ** 2)
                ptau2 = ptau0 + tau_step / tr * ((trQEeta_interp*peta0*current_tau + trQEx_interp*px0 + trQEy_interp*py0) - peta0 ** 2 * current_tau) / ptau0

                # Convert to physical units
                pmu.append([E0*ptau1, E0*px1, E0*py1, E0/a*peta1])
                constraint.append(E0*(ptau2-ptau1))

                charge_evolve.compute(Q0, tau_step, ptau0, px0, py0, peta0, Ax_interp, Ay_interp, Aeta_interp)
                Q1 = charge_evolve.Q
                Qsq = charge_evolve.Q2[0].real
                # print('qconstraint ', Qsq-Qsq0)
                qsq.append(Qsq)
                # print('Qsq ', Qsq)
               
            elif interp=='no':
                # Approximate the position of the quark with closest lattice point
                # Locations where transverse gauge fields extracted from gauge links are evaluated, in the middle of lattice sites
                xhq, yhq = int(round(x0)), int(round(y0))
                xahq, yahq = int(round(x0-1/2)), int(round(y0-1/2))

                fields.compute(Q0, xhq, yhq)
                trQEx, trQEy, trQEeta = fields.trQE.real[0], fields.trQE.real[1], fields.trQE.real[2]
                trQBx, trQBy, trQBeta = fields.trQB.real[0], fields.trQB.real[1], fields.trQB.real[2]

                potentials.compute('x', xahq, yhq)
                Ax = potentials.Ax
                potentials.compute('y', xhq, yahq)
                Ay = potentials.Ay
                potentials.compute('eta', xhq, yhq)
                Aeta = potentials.Aeta

                px1 = px0 + tau_step / tr * (trQEx + trQBeta * py0 / ptau0 - trQBy * peta0 * current_tau / ptau0)
                py1 = py0 + tau_step / tr * (trQEy - trQBeta * px0 / ptau0 + trQBx * peta0 * current_tau / ptau0)
                peta1 = peta0 + tau_step * ((trQEeta * ptau0 - trQBx * py0 + trQBy * px0) / tr - 2 * peta0 * ptau0) / (current_tau * ptau0)
                ptau1 = np.sqrt(px1 ** 2 + py1 ** 2 + (current_tau * peta1) ** 2 + (mass/E0) ** 2)
                ptau2 = ptau0 + tau_step / tr * ((trQEeta * peta0 * current_tau + trQEx * px0 + trQEy * py0) - peta0 ** 2 * current_tau) / ptau0

                # Convert to physical units
                pmu.append([E0*ptau1, E0*px1, E0*py1, E0/a*peta1])
                constraint.append(E0*(ptau2-ptau1))

                charge_evolve.compute(Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta)
                Q1 = charge_evolve.Q
                Qsq = charge_evolve.Q2[0].real
                # print('qconstraint ', Qsq-Qsq0)
                qsq.append(Qsq)
                # print('Qsq ', Qsq)

            # Convert to physical units
            # print('xmu ', [a*current_tau, a*x1, a*y1, eta1])
            # print('pmu ', [E0*ptau1, E0*px1, E0*py1, E0/a*peta1, current_tau*E0*peta1])
            # print('tauconstraint ', E0*ptau2-E0*ptau1)
            # pT0, pT = np.sqrt(pmu0[1]**2+pmu0[2]**2), E0*np.sqrt(px1**2+py1**2)
            # xT0, xT = np.sqrt(xmu0[0]**2+xmu0[1]**2), a*np.sqrt(x1**2+y1**2)
            # print('sigmaxT ', (xT-xT0)**2)
            # print('sigmapT ', (pT-pT0)**2)
            # print('sigmax ', (a**2*(x1-xmu0[0]/a)**2+a**2*(y1-xmu0[1]/a)**2)/2)
            # print('sigmap ', (E0**2*(px1-pmu0[1]/E0)**2+E0**2*(py1-pmu0[2]/E0)**2)/2)

            # Swap initial x, p, Q for next time step
            x0, y0, eta0 = x1, y1, eta1
            px0, py0, peta0, ptau0 = px1, py1, peta1, ptau1
            Q0 = Q1

    if use_cuda:
        s.copy_to_host()
        cuda.current_context().deallocations.clear()

    print("Simulation complete!")

    return xmu, pmu, constraint, qsq

current_path = os.getcwd() 
results_folder = 'results'
check_results_folder = os.path.isdir(results_folder)
if not check_results_folder:
    os.makedirs(results_folder)
    print("Created folder : ", results_folder)
else:
    print(results_folder, "folder already exists.")
results_path = current_path + '/' + results_folder + '/'
os.chdir(results_path)

wong_folder = 'wong_hq_' + system 
check_wong_folder = os.path.isdir(wong_folder)
if not check_wong_folder:
    os.makedirs(wong_folder)
    print("Created folder : ", wong_folder)
else:
    print(wong_folder, "folder already exists.")
wong_path = results_path + '/' + wong_folder + '/'
os.chdir(wong_path)

for qi in range(1, nq+1):

    xmu0 = initial_coords()
    pmu0 = initial_momenta()

    # Quark
    q0 = initial_charge()
    for tp in range(1+noffset, ntp+noffset+1):
        time_i = time.time()

        xmu, pmu, constraint, qsq = simulate(p, xmu0, pmu0, q0)

        filename = su_group + '_pT_' + str(pT) + '_interp_' + interp + '_q_' + str(qi) + '_tp_' + str(tp) + '.npz'
        np.savez(filename, xmu=xmu, pmu=pmu, constraint=constraint, qsq=qsq)

        time_f = time.time()
        print('Simulation time for a single quark test particle: ', time_f-time_i)

    # Antiquark
    q0 = initial_charge()
    pmu0 = [pmu0[0], -pmu0[1], -pmu0[2], pmu0[3]]
    for tp in range(1, ntp+1):
        time_i = time.time()

        xmu, pmu, constraint, qsq = simulate(p, xmu0, pmu0, q0)

        filename = su_group + '_pT_' + str(pT) + '_interp_' + interp + '_aq_' + str(qi) + '_tp_' + str(tp) + '.npz'
        np.savez(filename, xmu=xmu, pmu=pmu, constraint=constraint, qsq=qsq)

        time_f = time.time()
        print('Simulation time for a single antiquark test particle: ', time_f-time_i)