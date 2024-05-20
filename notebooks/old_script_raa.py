import numpy as np
import argparse

# hbar * c [GeV * fm]
hbarc = 0.197326 

# Simulation box 
L = 10      
N = 512 
tau_sim = 1.5     
DTS = 8     

# Glasma
su_group = 'su3'
Qs = 1.4      
ns = 50      
factor = 0.8        
uv = 10.0 

# Wong
quark = 'charm'    
mass = 1.275       

# tau_form = 0
initialization = 'fonll'         
ntp = 10**5  

nevents_offset = 50
nevents = 150    

representation = 'quantum fundamental'      
boundary = 'periodic'   

# Store relevant parameters in a dictionary
p = {
    'QUARK': quark,
    'MASS': mass,   
    'QS': Qs,            
    'NEVENTS': nevents,
    'NOFFSET': nevents_offset,   
    'NTP': ntp,      
    }

"""
    Argument parsing
"""
parser = argparse.ArgumentParser()

parser.add_argument('-QUARK',   type=str,   help="Quark name")

parser.add_argument('-MASS',    type=float, help="Quark mass [GeV]")
parser.add_argument('-QS',    type=float,   help="Saturation momentum [GeV]")
parser.add_argument('-NEVENTS',    type=int,   help="Number of events")
parser.add_argument('-NOFFSET',    type=int,   help="Events offset")

# parse args and update parameters dict
args = parser.parse_args()
data = args.__dict__
for d in data:
    if data[d] is not None:
        p[d] = data[d]

# rest of parameters  
g2mu = p["QS"] / factor     
g = np.pi * np.sqrt(1 / np.log(p["QS"] / 0.2))          		
mu = g2mu / g**2          	
ir = 0.1 * g2mu     

tau_form = 1/(2*p["MASS"])*hbarc
tau_sim += tau_form

# Results folder
folder = 'RAA_' + p["QUARK"] + '_fonll_Qs_' + str(p["QS"])

import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["PRECISION"] = "double"
os.environ["GAUGE_GROUP"] = su_group

# Import relevant modules
import sys
sys.path.append('..')

# Glasma modules
import curraun.core as core
import curraun.mv as mv
import curraun.initial as initial
initial.DEBUG = False

import curraun.su as su
from curraun.numba_target import use_cuda
if use_cuda:
    from numba import cuda

# Wong modules
from curraun import wong
wong.BOUNDARY = boundary
wong.WONG_TO_HOST = True
from curraun.wong import init_pos, init_charge, init_mom_toy, init_mom_fonll
from curraun.particles_correlators import TransMom

import pickle
from tqdm import tqdm

current_path = os.getcwd() 
results_folder = 'results'
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)
results_path = current_path + '/' + results_folder + '/'

def simulate(p, ev): 
    os.chdir(results_path)

    output = {}
    output['parameters'] = p.copy()

    # Derived parameters
    a = L/N
    E0 = N/L * hbarc
    DT = 1.0 / DTS
    formt = int(tau_form / a * DTS)
    maxt = int(tau_sim / a * DTS)

    # Initialize Glasma fields
    s = core.Simulation(N, DT, g)
    va = mv.wilson(s, mu=mu / E0, m=ir / E0, uv=uv / E0, num_sheets=ns)
    vb = mv.wilson(s, mu=mu / E0, m=ir / E0, uv=uv / E0, num_sheets=ns)
    initial.init(s, va, vb)

    # fonll
    pTs_fonll, ntp_fonll = init_mom_fonll(p)
    ntp = ntp_fonll

    # Initialize the Wong solver
    wong_solver = wong.WongSolver(s, ntp)
    x0s, p0s, q0s = np.zeros((ntp, 3)), np.zeros((ntp, 5)), np.zeros((ntp, su.ALGEBRA_ELEMENTS))
    masses = p["MASS"] / E0 * np.ones(ntp)

    for i in range(ntp):
        # fonll
        pT = pTs_fonll[i] / E0
        p0 = init_mom_toy('pT', pT)

        p0s[i, :] = np.array(p0, dtype=object)

        x0, q0 = init_pos(s.n), init_charge(representation)
        x0s[i, :], q0s[i, :] = x0, q0
    
    wong_solver.initialize(x0s, p0s, q0s, masses)

    pTs = np.zeros((maxt-formt, ntp))
    compute_pT = TransMom(wong_solver, ntp)

    with tqdm(total=maxt) as pbar:
        for t in range(maxt):
            # Evolve Glasma fields
            core.evolve_leapfrog(s)

            # Solve Wong's equations
            if t>=formt:  
                compute_pT.compute()
                pTs[t-formt] = compute_pT.pT.copy() * E0
                
                wong_solver.evolve()

            pbar.set_description("Event " + str(ev+1))
            pbar.update(1)

    tau = np.linspace(0, tau_sim-tau_form, maxt-formt)
    output['tau'], output['pTs'] = tau, pTs
    output["pTs_fonll"], output["ntp_fonl"] = pTs_fonll, ntp_fonll

    wong_folder = folder
    if not os.path.isdir(wong_folder):
        os.makedirs(wong_folder)
    wong_path = results_path + '/' + wong_folder + '/'
    os.chdir(wong_path)

    filename = 'event_' + str(ev+1) + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle)
    return output

print(p['QUARK'].capitalize() + " quark")

for ev in range(p["NOFFSET"], p["NEVENTS"]+p["NOFFSET"]):
    simulate(p, ev)