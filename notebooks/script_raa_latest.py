import numpy as np

# hbar * c [GeV * fm]
hbarc = 0.197326 

# Simulation box 
L = 10      
N = 512 
tau_sim = 0.4        
DTS = 8     

# Glasma
su_group = 'su3'
Qs = 2.0     
ns = 50      
factor = 0.8        
uv = 10.0 

# Wong
quark = 'charm'    
mass = 1.275   
tau_form = 1/(2*mass)*hbarc    

# tau_sim += tau_form

initialization = 'pT'         
ntp = 10**4 

nevents = 2

representation = 'quantum fundamental'     
# representation = 'test'     
# representation = 'fundamental' 
# representation = 'adjoint' 
boundary = 'periodic'  
form_time = 'm'

# pTs = [0, 0.5, 1, 5]
# binning = 'pT2'
binning = 'pT'
npTbins = 21 
pTmax = 10
if binning=='pT':
    pTbins = np.linspace(0, pTmax, npTbins)
elif binning=='pT2':
    pTbins = np.linspace(0, pTmax**2, npTbins)
deltapTbin = pTbins[1] - pTbins[0]

# Store relevant parameters in a dictionary
p = {
    'QUARK': quark,
    'MASS': mass,   
    'TFORM': tau_form,
    'TSIM': tau_sim,
    'QS': Qs,            
    'NEVENTS': nevents,
    'NTP': ntp,   
    'PTS': pTbins,
    'NPTBINS': npTbins,
    'PTMAX': pTmax,
    'REPR': representation,
    'SUGROUP': su_group,
    'FORMTIME': form_time,
    'BINNING': binning,
    }

import argparse

"""
    Argument parsing
"""
parser = argparse.ArgumentParser()

parser.add_argument('-QUARK',   type=str,   help="Quark name")

parser.add_argument('-MASS',    type=float, help="Quark mass [GeV]")
parser.add_argument('-QS',    type=float,   help="Saturation momentum [GeV]")
parser.add_argument('-NEVENTS',    type=int,   help="Number of events")
parser.add_argument('-NPTBINS',    type=int,   help="Number of pT bins")
parser.add_argument('-PTMAX',    type=int,   help="Maximum value of pT [GeV]")
parser.add_argument('-REPR',    type=str,   help="Representation")
parser.add_argument('-SUGROUP',    type=str,   help="SU(2) or SU(3)")
parser.add_argument('-FORMTIME',    type=float,   help="Formation time propto 1/2m or 1/2mT")
parser.add_argument('-BINNING',    type=str,   help="Type of binning")

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

if representation == 'quantum fundamental':
    repr_name = 'qfund'
elif representation == 'fundamental':
    repr_name = 'fund'
elif representation == 'adjoint':
    repr_name = 'adj'
elif representation == 'test':
    repr_name = 'test'

# Results folder
folder = 'test_binning_RAA_' + p["QUARK"] + '_fonll_Qs_' + str(p["QS"]) + '_' + repr_name + '_' + p["SUGROUP"] + '_formt_' + p["FORMTIME"]

# filename = 'all_pTs_' + binning + '_bins.pickle'

import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["PRECISION"] = "double"
os.environ["GAUGE_GROUP"] = p["SUGROUP"]

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
from curraun.wong import init_pos, init_charge, init_mom_toy
from curraun.particles_correlators import TransMom

import pickle
from tqdm import tqdm

current_path = os.getcwd() 
results_folder = 'results'
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)
results_path = current_path + '/' + results_folder + '/'

def simulate(p, ev, pTbin, deltapTbin): 
    # Derived parameters
    a = L/N
    E0 = N/L * hbarc
    DT = 1.0 / DTS

    if p["FORMTIME"]=='m':
        tau_form = 1/(2*p["MASS"])*hbarc
    elif p["FORMTIME"]=='mT':
        # transverse mass
        mT = np.sqrt(p["MASS"]**2+pT**2)   
        # formation time given by inverse of transverse mass
        tau_form = 1/(2*mT)*hbarc

    p["TFORM"] = tau_form

    tau_sim = p["TSIM"] + tau_form 

    formt = int(tau_form / a * DTS)
    maxt = int(tau_sim / a * DTS)

    # Initialize Glasma fields
    s = core.Simulation(N, DT, g)
    va = mv.wilson(s, mu=mu / E0, m=ir / E0, uv=uv / E0, num_sheets=ns)
    vb = mv.wilson(s, mu=mu / E0, m=ir / E0, uv=uv / E0, num_sheets=ns)
    initial.init(s, va, vb)

   
    # Initialize the Wong solver
    wong_solver = wong.WongSolver(s, ntp)
    x0s, p0s, q0s = np.zeros((ntp, 3)), np.zeros((ntp, 5)), np.zeros((ntp, su.ALGEBRA_ELEMENTS))
    masses = p["MASS"] / E0 * np.ones(ntp)

    # pT bins
    binning=p["BINNING"]
    pTbinlow, pTbinhigh = pTbin, pTbin+deltapTbin
    uniform_pTbins = np.sort(np.random.uniform(pTbinlow, pTbinhigh, ntp))
    if binning=='pT':
        initial_pTbins = uniform_pTbins
    elif binning=='pT2':
        initial_pTbins = np.sqrt(uniform_pTbins)

    for i in range(ntp):
        # x0, p0, q0 = init_pos(s.n), init_mom_toy('pT', pT / E0), init_charge(representation)
        x0, p0, q0 = init_pos(s.n), init_mom_toy('pT', initial_pTbins[i] / E0), init_charge(representation)
        x0s[i, :], p0s[i, :], q0s[i, :] = x0, p0, q0
    
    wong_solver.initialize(x0s, p0s, q0s, masses)

    final_pTbins = np.zeros((maxt-formt, ntp))
    compute_pT = TransMom(wong_solver, ntp)

    with tqdm(total=maxt) as pbar:
        for t in range(maxt):
            # Evolve Glasma fields
            core.evolve_leapfrog(s)

            # Solve Wong's equations
            if t>=formt:  
                compute_pT.compute()
                pTs = compute_pT.pT.copy() * E0
                if binning=='pT':
                    final_pTbins[t-formt] = pTs
                elif binning=='pT2':
                    final_pTbins[t-formt] = pTs ** 2

                # wong_solver.compute_mom_broad()
                
                wong_solver.evolve()

            pbar.set_description("Event " + str(ev+1))
            pbar.update(1)

    # output[str(pT)]['pTs_event_'+ str(ev+1)] = pTs
    # output[str(pT)]['initial_pTs_event_'+ str(ev+1)] = initial_pTs

    tau = np.linspace(0, tau_sim-tau_form, maxt-formt)
    # output['tau'] = tau

    return tau, final_pTbins, uniform_pTbins


print(p['QUARK'].capitalize() + " quark")
os.chdir(results_path)

# output = {}
# output['parameters'] = p.copy()
# output['pTs'] = p['PTS']
# output['nevents'] = p["NEVENTS"]

wong_folder = folder
if not os.path.isdir(wong_folder):
    os.makedirs(wong_folder)
wong_path = results_path + '/' + wong_folder + '/'
os.chdir(wong_path)

for ipT, pT in enumerate(p['PTS']):
    print("pT = " + str(pT) + " GeV")

    # output[str(pT)] = {}
    for iev, ev in enumerate(range(0, p["NEVENTS"])):
        # print("event " + str(iev+1))
        tau, final_pTs, initial_pTs = simulate(p, ev, pT, deltapTbin)

        filename_pT_ev = p["BINNING"] + '_bin_' + str(ipT+1) + '_ev_' + str(iev+1)
        np.savez_compressed(filename_pT_ev, pTs=final_pTs, initial_pTs=initial_pTs)

p["TAU"] = tau

with open('parameters_'+ p["BINNING"] +'_binning.pickle', 'wb') as handle:
    pickle.dump(p, handle)