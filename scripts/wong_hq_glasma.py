import numpy as np
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
su_group = 'su2'        # Gauge group
folder = 'pb+pb_5020gev_su2_interp_pT_0.5'      # Results folder

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
nq = 2     # Number of heavy quarks
pT = 0.5    # Initial transverse momentum [GeV]
ntp = 2    # Number of test particles

# Other numerical parameters
nevents = 2    # Number of Glasma events
interp = 'no'     # Interpolate fields or use nearest lattice points
solveq = 'wilson lines'     # Solve the equation for the color charge with Wilson lines or gauge potentials


"""
    Dictionary with standard MV model paramaters
"""

p = {
    # General parameters
    'GROUP':    su_group,       # SU(2) or SU(3) group
    'FOLDER':   folder,         # results folder

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
    'SOLVEQ': solveq,       # method used to solve the equation for color charge

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
parser.add_argument('-SOLVEQ', type=str, help="Method to evolve color charge.", default=solveq)

# Parse argumentss and update parameters dictionary
args = parser.parse_args()
data = args.__dict__
for d in data:
    if data[d] is not None:
        p[d] = data[d]

if p['SOLVEQ']=='wilson lines':
    p['INTERP']='no'

# Set environment variables 
import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["PRECISION"] = "double"
if p['GROUP'] == 'su2':
    os.environ["GAUGE_GROUP"] = 'su2_complex'
elif p['GROUP'] == 'su3':
    os.environ["GAUGE_GROUP"] = p['GROUP']

NUM_CHECKS = True
FORCE_CORR = True
if FORCE_CORR:
    p['INTERP']='no'

# Import relevant modules
import curraun.core as core
import curraun.mv as mv
import curraun.initial as initial
initial.DEBUG = False
from curraun.numba_target import use_cuda
if use_cuda:
    from numba import cuda
from curraun.wong_hq import WongFields, ColorChargeWilsonLines, WongPotentials, ColorChargeGaugePotentials, initial_coords, initial_momenta, initial_charge, interpfield, interppotential, update_coords, update_momenta
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
    if p['SOLVEQ']=='gauge potentials':
        potentials = WongPotentials(s)  
    if FORCE_CORR:
        electric_fields = ElectricFields(s)
        lorentz_force = LorentzForce(s)
        force_correlators = ForceCorrelators(s)

    if use_cuda:
        s.copy_to_device()

    output = {}
    output['xmu'], output['pmu'] = [], []
    if NUM_CHECKS:
        output['constraint'], output['casimirs'] = [], []

    if FORCE_CORR:
        tags = ['naive', 'transported']
        all_EformE, all_FformF = {}, {}
        for tag in tags:
            all_EformE[tag], all_FformF[tag] = [], []

    for t in range(maxt):
        core.evolve_leapfrog(s)

        if t>=formt:  
            current_tau = t * DT
            tau_step = DT
            
            if t==formt:
                output['xmu'].append([a*current_tau, a*x0, a*y0, eta0])
                output['pmu'].append([E0*ptau0, E0*px0, E0*py0, E0/a*peta0])
                logging.debug("Coordinates: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(a*current_tau, a*x0, a*y0, eta0))
                logging.debug("Momenta: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(E0*ptau0, E0*px0, E0*py0, E0/a*peta0))

                if p['SOLVEQ']=='gauge potentials':
                    charge = ColorChargeGaugePotentials(s, q0)
                else:
                    charge = ColorChargeWilsonLines(s, q0)
                    xhq0, yhq0 = int(round(x0)), int(round(y0))

                Q0 = charge.Q
                if NUM_CHECKS:
                    C = charge.C.real
                    output['casimirs'].append(C)
                    logging.debug("Quadratic Casimir: {:3.3f}".format(C[0]))
                    if p['GROUP']=='su3':
                        logging.debug("Cubic Casimir: {:3.3f}".format(C[1]))

                if p['INTERP']=='yes':
                    logging.info('Interpolating fields...')
                else:
                    logging.info('Approximating by nearest lattice points...')

                if FORCE_CORR:
                    Eform = electric_fields.compute(xhq0, yhq0)
                    Fform = lorentz_force.compute(xhq0, yhq0, ptau0, px0, py0, peta0, current_tau)

            # Solve Wong's equations using basic Euler
            # Update positions
            x1, y1, eta1 = update_coords(x0, y0, eta0, ptau0, px0, py0, peta0, tau_step)

            # Convert to physical units
            output['xmu'].append([a*current_tau, a*x1, a*y1, eta1])

            if p['INTERP']=='yes':                
                trQE_interp, trQB_interp = interpfield(x0, y0, Q0, fields)
                Ax_interp, Ay_interp, Aeta_interp = interppotential(x0, y0, 'x', potentials), interppotential(x0, y0, 'y', potentials), interppotential(x0, y0, 'eta', potentials)

                # Update momenta using Euler, with interpolated fields
                ptau1, ptau2, px1, py1, peta1 = update_momenta(ptau0, px0, py0, peta0, tau_step, current_tau, trQE_interp, trQB_interp, mass, E0)

                # Convert to physical units
                output['pmu'].append([E0*ptau1, E0*px1, E0*py1, E0/a*peta1])
                if NUM_CHECKS:
                    output['constraint'].append(E0*(ptau2-ptau1))

                charge.evolve(Q0, tau_step, ptau0, px0, py0, peta0, Ax_interp, Ay_interp, Aeta_interp)
                Q1 = charge.Q 
                if NUM_CHECKS:
                    C = charge.C.real
                    output['casimirs'].append(C)
                    logging.debug("Quadratic Casimir: {:3.3f}".format(C[0]))
                    if p['GROUP']=='su3':
                        logging.debug("Cubic Casimir: {:3.3f}".format(C[1]))

            else:
                # Approximate the position of the quark with closest lattice point
                # Locations where transverse gauge fields extracted from gauge links are evaluated, in the middle of lattice sites
                xhq, yhq = int(round(x0)), int(round(y0))

                fields.compute(Q0, xhq, yhq)
                trQE, trQB = fields.trQE.real, fields.trQB.real

                if p['SOLVEQ']=='gauge potentials':
                    xahq, yahq = int(round(x0-1/2)), int(round(y0-1/2))
                    potentials.compute('x', xahq, yhq)
                    Ax = potentials.Ax
                    potentials.compute('y', xhq, yahq)
                    Ay = potentials.Ay
                    potentials.compute('eta', xhq, yhq)
                    Aeta = potentials.Aeta

                # Update momenta using Euler, with fields evaluated at nearest lattice points
                ptau1, ptau2, px1, py1, peta1 = update_momenta(ptau0, px0, py0, peta0, tau_step, current_tau, trQE, trQB, mass, E0)

                # Convert to physical units
                output['pmu'].append([E0*ptau1, E0*px1, E0*py1, E0/a*peta1])
                if NUM_CHECKS:
                    output['constraint'].append(E0*(ptau2-ptau1))

                if p['SOLVEQ']=='gauge potentials':
                    charge.evolve(Q0, tau_step, ptau0, px0, py0, peta0, Ax, Ay, Aeta)
                else:
                    delta_etahq = eta1-eta0
                    charge.evolve(xhq, xhq0, yhq, yhq0, delta_etahq)
                    if (xhq!=xhq0):
                        xhq0=xhq
                    if (yhq!=yhq0):
                        yhq0=yhq

                Q1 = charge.Q
                if NUM_CHECKS:
                    C = charge.C.real
                    output['casimirs'].append(C)
                    logging.debug("Quadratic Casimir: {:3.3f}".format(C[0]))
                    if p['GROUP']=='su3':
                        logging.debug("Cubic Casimir: {:3.3f}".format(C[1]))

                if FORCE_CORR:
                    # [(GeV / fm) ** 2]
                    units = (E0 ** 2 / hbarc) ** 2 / p['G'] ** 2

                    E = electric_fields.compute(xhq, yhq)
                    F = lorentz_force.compute(xhq, yhq, ptau0, px0, py0, peta0, current_tau)

                    EformE, FformF  = {}, {}
                    for tag in tags:
                        force_correlators.compute(tag, Eform, E, xhq, xhq0, yhq, yhq0, delta_etahq)
                        EformE[tag] = force_correlators.fformf * units
                        all_EformE[tag].append(EformE[tag][0]+EformE[tag][1]+EformE[tag][2])

                        force_correlators.compute(tag, Fform, F, xhq, xhq0, yhq, yhq0, delta_etahq)
                        FformF[tag] = force_correlators.fformf * units
                        all_FformF[tag].append(FformF[tag][0]+FformF[tag][1]+FformF[tag][2])

            # Convert to physical units
            logging.debug("Coordinates: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(a*current_tau, a*x1, a*y1, eta1))
            logging.debug("Momenta: [{:3.3f}, {:3.3f}, {:3.3f}, {:3.3f}]".format(E0*ptau1, E0*px1, E0*py1, E0/a*peta1))
            logging.debug("Ptau constraint: {:.3e}".format(E0*ptau2-E0*ptau1)) 

            pT0, pT = np.sqrt(pmu0[1]**2+pmu0[2]**2), E0*np.sqrt(px1**2+py1**2)
            xT0, xT = np.sqrt(xmu0[0]**2+xmu0[1]**2), a*np.sqrt(x1**2+y1**2)
            logging.debug("Transverse coordinate variance: {:.3e}".format((xT-xT0)**2))
            logging.debug("Transverse momentum variance: {:3.3f}".format((pT-pT0)**2))

            # Swap initial x, p, Q for next time step
            x0, y0, eta0 = x1, y1, eta1
            px0, py0, peta0, ptau0 = px1, py1, peta1, ptau1
            Q0 = Q1

    if use_cuda:
        s.copy_to_host()
        cuda.current_context().deallocations.clear()

    if FORCE_CORR:
        output['EformE'], output['FformF'] = all_EformE, all_FformF

    logging.info("Simulation complete!")

    return output

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

# Initializing progress bar objects
# Source: https://stackoverflow.com/questions/60928718/python-how-to-replace-tqdm-progress-bar-by-next-one-in-nested-loop
outer_loop=tqdm(range(p['NEVENTS']), desc="Event", position=0)
mid_loop=tqdm(range(p['NQ']), desc="Quark antiquark pair", position=1)
inner_loop=tqdm(range(p['NTP']), desc="Test particle", position=2)


for ev in range(len(outer_loop)):
    logging.info("\nSimulating event {}/{}".format(ev+1, nevents))
    # Fixing the seed in a certain event
    seed = ev

    mid_loop.refresh() 
    mid_loop.reset() 
    outer_loop.update() 

    for q in range(len(mid_loop)):
        logging.info("\nSimulating quark antiquark pair {}/{}".format(q+1, p['NQ']))

        inner_loop.refresh()  
        inner_loop.reset()  
        mid_loop.update()  

        for tp in range(len(inner_loop)):
            xmu0 = initial_coords(p)
            pmu0 = initial_momenta(p)

            # Quark
            logging.info("\nSimulating quark test particle {}/{}".format(tp+1, p['NTP']))
            q0 = initial_charge(p)

            filename = 'ev_' + str(ev+1) + '_q_' + str(q+1) + '_tp_' + str(tp+1) + '.pickle'
            output = simulate(p, xmu0, pmu0, q0, seed)
            with open(filename, 'wb') as handle:
                pickle.dump(output, handle)

            # Antiquark having opposite momentum and random color charge
            logging.info("\nSimulating antiquark test particle {}/{}".format(tp+1, p['NTP']))
            q0 = initial_charge(p)
            pmu0 = [pmu0[0], -pmu0[1], -pmu0[2], pmu0[3]]

            filename = 'ev_' + str(ev+1) + '_aq_' + str(q+1) + '_tp_' + str(tp+1) + '.pickle'
            output = simulate(p, xmu0, pmu0, q0, seed)
            with open(filename, 'wb') as handle:
                pickle.dump(output, handle)

            inner_loop.update()    