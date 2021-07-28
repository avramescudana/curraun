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
nq = 15     # Number of heavy quark antiquark pairs
pT = 0.5    # Initial transverse momentum [GeV]
ntp = 10   # Number of test particles

# Other numerical parameters
nevents = 1    # Number of Glasma events
solveq = 'wilson_lines'     # Solve the equation for the color charge with Wilson lines or gauge potentials
frame = 'milne'
NUM_CHECKS = True
FORCE_CORR = True


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
    'SOLVEQ': solveq,       # method used to solve the equation for color charge
    'FRAME': frame,         # laboratory of Milne frame
    'NUM_CHECK': NUM_CHECKS,    # perform numerical checks
    'FORCE_CORR': FORCE_CORR,   # compute correlator of Lorentz force
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
parser.add_argument('-SOLVEQ', type=str, help="Method to evolve color charge.", default=solveq)
parser.add_argument('-FRAME', type=str, help="Laboratory or Milne frame.", default=frame)
parser.add_argument('-NUM_CHECKS', type=bool, help="Perform numerical checks.", default=NUM_CHECKS)
parser.add_argument('-FORCE_CORR', type=bool, help="Compute correlator of Lorentz force.", default=FORCE_CORR)


# Parse argumentss and update parameters dictionary
args = parser.parse_args()
data = args.__dict__
for d in data:
    if data[d] is not None:
        p[d] = data[d]

# Option to solve in laboratory frame with Boris push only for solution of color charge equation with Wilson lines, with no numerical checks or force correlator computation
if p['FRAME']=='lab':
    p['SOLVEQ']='wilson lines'
    p['FORCE_CORR']=False
   
# Set environment variables 
import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["PRECISION"] = "double"
if p['GROUP'] == 'su2':
    os.environ["GAUGE_GROUP"] = 'su2_complex'
elif p['GROUP'] == 'su3':
    os.environ["GAUGE_GROUP"] = p['GROUP']


# Import relevant modules
import curraun.core as core
import curraun.mv as mv
import curraun.initial as initial
initial.DEBUG = False
from curraun.numba_target import use_cuda
if use_cuda:
    from numba import cuda
if p['FRAME']=='milne':
    from curraun.wong_hq_batch import initial_coords, initial_momenta, initial_charge, initial_wong, solve_wong
elif p['FRAME']=='lab':
    from curraun.wong_hq_lab import initial_coords, initial_momenta, initial_charge, initial_wong, solve_wong

# Define hbar * c in units of GeV * fm
hbarc = 0.197326 

"""
    Simulation function
"""

def simulate(p, ev, inner_loop): 
   
    xmu, pmu = {}, {}
    xmu0, pmu0, q0, xmu1, pmu1, q1 = {}, {}, {}, {}, {}, {}
    fields, charge = {}, {}
    # If NUM_CHECKS = True
    constraint, casimirs = {}, {}
    # If FORCE_CORR = True
    correlators = {}
    correlators['EformE'], correlators['FformF'] = {}, {}
    tags_corr = ['naive', 'transported']
    for tag in tags_corr:
        correlators['EformE'][tag], correlators['FformF'][tag] = {}, {}
    fieldsform = {}
    fieldsform['E'], fieldsform['F'] = {}, {}
    electric_fields, lorentz_force, force_correlators = {}, {}, {}

    # Derived parameters
    a = p['L'] / p['N']
    E0 = p['N'] / p['L'] * hbarc
    p['E0'] = E0
    DT = 1.0 / p['DTS']
    formt = int(p['TFORM'] / a * p['DTS'])

    logging.info('Initializating ...')

    s = core.Simulation(p['N'], DT, p['G'])
    va = mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
    vb = mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
    initial.init(s, va, vb)

    if use_cuda:
        s.copy_to_device()

    for t in range(len(inner_loop)):
        logging.debug("Time: {:3.5f}".format(t))
        core.evolve_leapfrog(s)

        if t>=formt:  
            for q in range(p['NQ']):
                for tp in range(p['NTP']):

                    tagq = 'ev_' + str(ev+1) + '_q_' + str(q+1) + '_tp_' + str(tp+1)
                    tagaq = 'ev_' + str(ev+1) + '_aq_' + str(q+1) + '_tp_' + str(tp+1)

                    if t==formt:
                        # Initialize quark
                        xmu0[tagq] = initial_coords(p)
                        pmu0[tagq] = initial_momenta(p)
                        q0[tagq] = initial_charge(p)
                        solve_wong(s, p, t, xmu0[tagq], pmu0[tagq], q0[tagq], xmu, pmu, fields, charge, tagq, constraint, casimirs, correlators, fieldsform, electric_fields, lorentz_force, force_correlators)

                        # Initialize antiquark
                        xmu0[tagaq] = xmu0[tagq]
                        pmu0[tagaq] = [pmu0[tagq][0], -pmu0[tagq][1], -pmu0[tagq][2], pmu0[tagq][3]]
                        q0[tagaq] = initial_charge(p)
                        solve_wong(s, p, t, xmu0[tagaq], pmu0[tagaq], q0[tagaq], xmu, pmu, fields, charge, tagaq, constraint, casimirs, correlators, fieldsform, electric_fields, lorentz_force, force_correlators)

                    elif t>formt:
                        xmu1[tagq], pmu1[tagq], q1[tagq] = solve_wong(s, p, t, xmu0[tagq], pmu0[tagq], q0[tagq], xmu, pmu, fields, charge, tagq, constraint, casimirs, correlators, fieldsform, electric_fields, lorentz_force, force_correlators)
                        # Swap x, p, q for next time step
                        xmu0[tagq], pmu0[tagq], q0[tagq] = xmu1[tagq], pmu1[tagq], q1[tagq]

                        xmu1[tagaq], pmu1[tagaq], q1[tagaq] = solve_wong(s, p, t, xmu0[tagaq], pmu0[tagaq], q0[tagaq], xmu, pmu, fields, charge, tagaq, constraint, casimirs, correlators, fieldsform, electric_fields, lorentz_force, force_correlators)
                        # Swap x, p, q for next time step
                        xmu0[tagaq], pmu0[tagaq], q0[tagaq] = xmu1[tagaq], pmu1[tagaq], q1[tagaq]

        inner_loop.update()

    if use_cuda:
        s.copy_to_host()
        cuda.current_context().deallocations.clear()

    for q in range(p['NQ']):
        for tp  in range(p['NTP']):
            output = {}
            tagq = 'ev_' + str(ev+1) + '_q_' + str(q+1) + '_tp_' + str(tp+1)
            output['xmu'], output['pmu'] = xmu[tagq], pmu[tagq]
            if NUM_CHECKS:
                output['constraint'], output['casimirs'] = constraint[tagq], casimirs[tagq]
            output['correlators'] = {}
            if FORCE_CORR:
                types_corr = ['EformE', 'FformF']
                tags_corr = ['naive', 'transported']
                for type_corr in types_corr:
                    output['correlators'][type_corr] = {}
                    for tag_corr in tags_corr:
                        output['correlators'][type_corr][tag_corr] = {}
                        output['correlators'][type_corr][tag_corr] = correlators[type_corr][tag_corr][tagq]                

            filename = tagq + '.pickle'
            with open(filename, 'wb') as handle:
                pickle.dump(output, handle)

            # output = {}
            tagaq = 'ev_' + str(ev+1) + '_aq_' + str(q+1) + '_tp_' + str(tp+1)
            output['xmu'], output['pmu'] = xmu[tagaq], pmu[tagaq]
            if NUM_CHECKS:
                output['constraint'], output['casimirs'] = constraint[tagaq], casimirs[tagaq]
            if FORCE_CORR:
                types_corr = ['EformE', 'FformF']
                tags_corr = ['naive', 'transported']
                for type_corr in types_corr:
                    for tag_corr in tags_corr:
                        output['correlators'][type_corr][tag_corr] = correlators[type_corr][tag_corr][tagq]   

            filename = tagaq + '.pickle'
            with open(filename, 'wb') as handle:
                pickle.dump(output, handle)

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

outer_loop=tqdm(range(p['NEVENTS']), desc="Event", position=0)
inner_loop=tqdm(range(int(p['TMAX'] / (p['L'] / p['N']) * p['DTS'])), desc="Time", position=1)

for ev in range(len(outer_loop)):
    logging.info("\nSimulating event {}/{}".format(ev+1, nevents))
    simulate(p, ev, inner_loop)
    inner_loop.refresh()  
    inner_loop.reset()
    outer_loop.update() 

    