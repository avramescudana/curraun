import numpy as np
import time
import argparse
from tqdm import tqdm
import pickle
import logging, sys
# Supress Numba warnings
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
# Format logging messages, set level=logging.DEBUG or logging.INFO for more information printed out, logging.WARNING for basic info
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format='%(message)s')

"""
    Default simulation parameters chosen for Pb-Pb at 5.02 TeV
"""

# General parameters
su_group = 'su2'
folder = 'pb+pb_5020gev_su2_interp_pT_0.5'

# Simulation box parameters
L = 10      # Length of simulation box [fm]
N = 512     # Number of lattice sites
tau_s = 2.0     # Simulation time [fm/c]
DTS = 8     # Time step

# MV model parameters for Pb-Pb at 5.02 TeV
A = 207     # Mass number
sqrts = 5020        # Center-of-mass energy [GeV]
ns = 50     # Number of color sheets
factor = 0.8        # Ratio between Qs/g^2\mu for Ns = 50 color sheets
Qs = np.sqrt(0.13 * A**(1/3) * sqrts**0.25)         # Saturation momentum [GeV]	
g = np.pi * np.sqrt(1 / np.log(Qs / 0.2))           # Running coupling constant		
mu = Qs / (g**2 * factor)           # MV model parameter	
m = 0.1 * g**2 * mu         # Infrared regulator [GeV]
uv = 10.0           # Ultraviolet regulator [GeV]

# Heavy quark related parameters, chosen here for a charm quark
mass = 1.5      # Heavy quark mass [GeV]
tau_form = 0.06     # Formation time [fm/c]
nq = 1     # Number of heavy quarks
pT = 0.5    # Initial transverse momentum [GeV]
ntp = 1    # Number of test particles

# Other numerical parameters
nevents = 1    # Number of Glasma events
#TODO: remove option to interpolate
interp = 'no'     # Interpolate fields or use nearest lattice points


"""
    Dictionary with standard MV model paramaters
"""

p = {
    # General parameters
    'GROUP':    su_group,       # SU(2) or SU(3) group
    'FOLDER':   folder ,         # results folder

    # Parameters for simulation box
    'L':    L,           # transverse size [fm]
    'N':    N,            # lattice size
    'DTS':  DTS,             # time steps per transverse spacing
    'TMAX': tau_s,          # max. proper time (tau) [fm/c]

    # Parameters for MV model
    'G':    g,            # YM coupling constant
    'MU':   mu,             # MV model parameter [GeV]
    'M':    m,              # IR regulator [GeV]
    'UV':   uv,           # UV regulator [GeV]
    'NS':   ns,             # number of color sheets
    
    # Parameters for heavy quarks
    'MASS': mass,           # mass of HQ [GeV]
    'TFORM': tau_form,       # formation time of the HQ [fm/c]
    'PT': pT,           # transverse momentum of HQs [GeV]
    'NQ': nq,         # number of heavy quarks
    'NTP': ntp,         # number of test particles

    # Numerical parameters
    'NEVENTS': nevents,     # number of Glasma events
    'INTERP': interp,       # interpolate fields or use nearest lattice points

}

"""
    Argument parsing for running scripts with different parameters
"""

parser = argparse.ArgumentParser(description='Compute momentum broadening of HQs in the Glasma.')

parser.add_argument('-GROUP', type=str, help="Gauge group", default=su_group)
parser.add_argument('-FOLDER', type=str, help="Folder in which results are saved", default=folder)

parser.add_argument('-L', type=float, help="Transverse lattice size [fm]", default=L)
parser.add_argument('-N', type=int, help="Number of lattice sites", default=N)
parser.add_argument('-DTS', type=int, help="Time steps per transverse spacing", default=DTS)
parser.add_argument('-TMAX', type=float, help="Maximum proper time [fm/c]", default=tau_s)
parser.add_argument('-G', type=float, help="YM coupling constant", default=g)
parser.add_argument('-MU', type=float, help="MV model parameter [GeV]", default=mu)
parser.add_argument('-M', type=float, help="IR regulator [GeV]", default=m)
parser.add_argument('-UV', type=float, help="UV regulator [GeV]", default=uv)
parser.add_argument('-NS', type=int, help="Number of color sheets", default=ns)

parser.add_argument('-MASS', type=float, help="Mass of heavy quark [GeV]", default=mass)
parser.add_argument('-TFORM', type=float, help="Formation time of heavy quark [fm/c]", default=tau_form)
parser.add_argument('-PT', type=float, help="Initial transverse momentum [GeV]", default=pT)
parser.add_argument('-NQ', type=int, help="Number of heavy quarks", default=nq)
parser.add_argument('-NTP', type=int, help="Number of test particles", default=ntp)

parser.add_argument('-NEVENTS', type=int, help="Number of events", default=nevents)
parser.add_argument('-INTERP', type=str, help="Interpolate fields.", default=interp)

# Parse argumentss and update parameters dictionary
args = parser.parse_args()
data = args.__dict__
for d in data:
    if data[d] is not None:
        p[d] = data[d]

# Set environment variables 
import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["PRECISION"] = "double"
if p['GROUP'] == 'su2':
    os.environ["GAUGE_GROUP"] = 'su2_complex'
    id0 = (1, 0, 0, 1)
elif p['GROUP'] == 'su3':
    os.environ["GAUGE_GROUP"] = p['GROUP']
    id0 = (1, 0, 0, 0, 1, 0, 0, 0, 1)

# Import relevant modules
import curraun.core as core
import curraun.mv as mv
import curraun.initial as initial
initial.DEBUG = False
from curraun.numba_target import use_cuda
if use_cuda:
    from numba import cuda
from curraun.wong_hq import WongFields, WongPotentials, InitialColorCharge, ColorChargeEvolve, initial_coords, initial_momenta, initial_charge, interpfield, interppotential, update_coords, update_momenta
# from curraun.wong_force_correlators import ElectroMagneticFields
from curraun.wong_force_correlators import ElectricFields, LorentzForce, ForceCorrelators

# Define hbar * c in units of GeV * fm
hbarc = 0.197326 

"""
    Simulation function
"""

def simulate(p, xmu0, pmu0, q0, seed):    
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

    logging.info('Initializating ...')

    s = core.Simulation(p['N'], DT, p['G'])
    mv.set_seed(seed)
    va = mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
    vb = mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
    initial.init(s, va, vb)

    fields = WongFields(s)
    potentials = WongPotentials(s)
    charge_initial = InitialColorCharge(s)
    charge_evolve = ColorChargeEvolve(s)

    # elmag_fields = ElectroMagneticFields(s)
    electric_fields = ElectricFields(s)
    lorentz_force = LorentzForce(s)
    force_correlators = ForceCorrelators(s)

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
                logging.debug("Coordinates: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(a*current_tau, a*x0, a*y0, eta0))
                logging.debug("Momenta: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(E0*ptau0, E0*px0, E0*py0, E0/a*peta0))

                charge_initial.compute(q0)
                Q0 = charge_initial.Q
                Qsq0 = charge_initial.Q2[0].real
                qsq.append(Qsq0)
                logging.debug("Quadratic Casimir: {:3.3f}".format(Qsq0))

                if interp=='yes':
                    logging.info('Interpolating fields...')
                else:
                    logging.info('Approximating by nearest lattice points...')

                xhq0, yhq0 = int(round(x0)), int(round(y0))
                Eform = electric_fields.compute(xhq0, yhq0)
                Fform = lorentz_force.compute(xhq0, yhq0, ptau0, px0, py0, peta0, current_tau)

                # print('Fform=', Fform)

                Uxhq0, Uyhq0 = np.array(id0), np.array(id0)

            # Solve Wong's equations using basic Euler
            # Update positions
            x1, y1, eta1 = update_coords(x0, y0, eta0, ptau0, px0, py0, peta0, tau_step)
            delta_eta = eta1-eta0

            # Convert to physical units
            xmu.append([a*current_tau, a*x1, a*y1, eta1])

            if interp=='yes':                
                trQE_interp, trQB_interp = interpfield(x0, y0, Q0, fields)
                Ax_interp, Ay_interp, Aeta_interp = interppotential(x0, y0, 'x', potentials), interppotential(x0, y0, 'y', potentials), interppotential(x0, y0, 'eta', potentials)

                # Update momenta using Euler, with interpolated fields
                ptau1, ptau2, px1, py1, peta1 = update_momenta(ptau0, px0, py0, peta0, tau_step, current_tau, trQE_interp, trQB_interp, mass, E0)

                # Convert to physical units
                pmu.append([E0*ptau1, E0*px1, E0*py1, E0/a*peta1])
                constraint.append(E0*(ptau2-ptau1))

                charge_evolve.compute(Q0, tau_step, ptau0, px0, py0, peta0, Ax_interp, Ay_interp, Aeta_interp)
                Q1 = charge_evolve.Q
                Qsq = charge_evolve.Q2[0].real
                qsq.append(Qsq)
                logging.debug("Quadratic Casimir: {:3.3f}".format(Qsq0))
               
            elif interp=='no':
                # Approximate the position of the quark with closest lattice point
                # Locations where transverse gauge fields extracted from gauge links are evaluated, in the middle of lattice sites
                xhq, yhq = int(round(x0)), int(round(y0))
                xahq, yahq = int(round(x0-1/2)), int(round(y0-1/2))

                fields.compute(Q0, xhq, yhq)
                trQE, trQB = fields.trQE.real, fields.trQB.real

                potentials.compute('x', xahq, yhq)
                Ax = potentials.Ax
                potentials.compute('y', xhq, yahq)
                Ay = potentials.Ay
                potentials.compute('eta', xhq, yhq)
                Aeta = potentials.Aeta

                # Update momenta using Euler, with fields evaluated at nearest lattice points
                ptau1, ptau2, px1, py1, peta1 = update_momenta(ptau0, px0, py0, peta0, tau_step, current_tau, trQE, trQB, mass, E0)

                # Convert to physical units
                pmu.append([E0*ptau1, E0*px1, E0*py1, E0/a*peta1])
                constraint.append(E0*(ptau2-ptau1))

                charge_evolve.compute(Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta)
                Q1 = charge_evolve.Q
                Qsq = charge_evolve.Q2[0].real
                qsq.append(Qsq)
                logging.debug("Quadratic Casimir: {:3.3f}".format(Qsq0))

                # [(GeV / fm) ** 2]
                units = (E0 ** 2 / hbarc) ** 2 / p['G'] ** 2

                E = electric_fields.compute(xhq, yhq)
                F = lorentz_force.compute(xhq, yhq, ptau0, px0, py0, peta0, current_tau)

                force_correlators.compute('naive', Eform[0], Eform[1], Eform[2], E[0], E[1], E[2], Uxhq0, Uyhq0, xhq, xhq0, yhq, yhq0, delta_eta)
                ExformEx, EyformEy, EzformEz = force_correlators.fxformfx, force_correlators.fyformfy, force_correlators.fzformfz

                print('EformE=', (ExformEx+EyformEy+EzformEz)*units)
                # print('ExformEx=', ExformEx*units)
                # print('EyformEy=', EyformEy*units)
                # print('EzformEz=', EzformEz*units)
            
                Uxhq, Uyhq = force_correlators.Uxhq, force_correlators.Uyhq

                force_correlators.compute('naive', Fform[0], Fform[1], Fform[2], F[0], F[1], F[2], Uxhq0, Uyhq0, xhq, xhq0, yhq, yhq0, delta_eta)
                FxformFx, FyformFy, FzformFz = force_correlators.fxformfx, force_correlators.fyformfy, force_correlators.fzformfz

                # print('FformF=', (FxformFx+FyformFy+FzformFz)*units)
                print('FxformFx=', FxformFx*units)
                # print('FyformFy=', FyformFy*units)
                print('FzformFz=', FzformFz*units)


            # Convert to physical units
            logging.debug("Coordinates: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(a*current_tau, a*x1, a*y1, eta1))
            logging.debug("Momenta: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(E0*ptau1, E0*px1, E0*py1, E0/a*peta1))
            logging.debug("Ptau constraint: {:.3e}".format(E0*ptau2-E0*ptau1)) 

            pT0, pT = np.sqrt(pmu0[1]**2+pmu0[2]**2), E0*np.sqrt(px1**2+py1**2)
            xT0, xT = np.sqrt(xmu0[0]**2+xmu0[1]**2), a*np.sqrt(x1**2+y1**2)
            logging.debug("Transverse coordinate variance: {:.3e}".format((xT-xT0)**2))
            logging.debug("Transverse momentum variance: {:3.3f}".format((pT-pT0)**2))

            if (xhq!=xhq0):
                xhq0=xhq
            if (yhq!=yhq0):
                yhq0=yhq

            # Swap initial x, p, Q for next time step
            x0, y0, eta0 = x1, y1, eta1
            px0, py0, peta0, ptau0 = px1, py1, peta1, ptau1
            Q0 = Q1
            Uxhq0, Uyhq0 = Uxhq, Uyhq

    if use_cuda:
        s.copy_to_host()
        cuda.current_context().deallocations.clear()

    logging.info("Simulation complete!")

    return xmu, pmu, constraint, qsq

"""
    Create folders to store the files resulting from the simulations
"""

current_path = os.getcwd() 
results_folder = 'results'
check_results_folder = os.path.isdir(results_folder)
if not check_results_folder:
    os.makedirs(results_folder)
    logging.info("Creating folder " + results_folder)
else:
    logging.info(results_folder + " folder already exists.")
results_path = current_path + '/' + results_folder + '/'
os.chdir(results_path)

wong_folder = p['FOLDER']
check_wong_folder = os.path.isdir(wong_folder)
if not check_wong_folder:
    os.makedirs(wong_folder)
    logging.info("Creating folder " + wong_folder)
else:
    logging.info(wong_folder + " folder already exists.")
wong_path = results_path + '/' + wong_folder + '/'
os.chdir(wong_path)

# Save parameters dictionary to file
with open('parameters.pickle', 'wb') as handle:
    pickle.dump(p, handle)

"""
    Simulate multiple Glasma events, each event with 15 quarks and 15 antiquarks, produced at the same positions as the quarks, having opposite momenta and random charge
    The number of quarks or antiquarks in enlarged by a given number of test particles
"""

for ne in tqdm(range(p['NEVENTS']), desc="Events", position=0):
    logging.info("Simulating event {}/{}".format(ne, nevents))
    # Fixing the seed in a certain event
    seed = ne

    for qi in tqdm(range(p['NQ']), desc="Quark antiquark pairs", position=1, leave=bool(ne == (p['NEVENTS']-1))):

        xmu0 = initial_coords(p)
        pmu0 = initial_momenta(p)

        # Quark
        q0 = initial_charge(p)
        logging.info("Simulating quark {}/{}".format(qi, p['NQ']))

        for tp in tqdm(range(p['NTP']), desc="Quark test particles", position=2, leave=bool(ne == (p['NQ']-1))):
            time_i = time.time()
            logging.info("Simulating test particle {}/{}".format(tp, p['NTP']))

            xmu, pmu, constraint, qsq = simulate(p, xmu0, pmu0, q0, seed)

            filename = 'q_' + str(qi+1) + '_tp_' + str(tp+1) + '.npz'
            np.savez(filename, xmu=xmu, pmu=pmu, constraint=constraint, qsq=qsq)

            time_f = time.time()
            logging.info('Simulation time for a single quark test particle: {:5.2f}s'.format(time_f-time_i))
            time.sleep(0.1)

        # Antiquark
        q0 = initial_charge(p)
        pmu0 = [pmu0[0], -pmu0[1], -pmu0[2], pmu0[3]]
        logging.info("Simulating antiquark {}/{}".format(qi, nq))
        for tp in tqdm(range(p['NTP']), desc="Antiquark test particles", position=2, leave=bool(ne == (p['NQ']-1))):
            time_i = time.time()
            logging.info("Simulating test particle {}/{}".format(tp, p['NTP']))

            xmu, pmu, constraint, qsq = simulate(p, xmu0, pmu0, q0, seed)

            filename = 'aq_' + str(qi+1) + '_tp_' + str(tp+1) + '.npz'
            np.savez(filename, xmu=xmu, pmu=pmu, constraint=constraint, qsq=qsq)

            time_f = time.time()
            logging.info('Simulation time for a single antiquark test particle: {:5.2f}s'.format(time_f-time_i))
            time.sleep(0.1)
        
        time.sleep(0.1)