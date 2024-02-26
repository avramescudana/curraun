import numpy as np

# hbar * c [GeV * fm]
hbarc = 0.197326 

# Simulation box 
L = 10      
N = 512    
tau_sim = 10   
DTS = 8     

# Glasma
su_group = 'su3'
Qs = 2.0          
ns = 50    
factor = 0.8        
g2mu = Qs / factor     
g = np.pi * np.sqrt(1 / np.log(Qs / 0.2))          		
mu = g2mu / g**2          	
ir = 0.1 * g2mu         
uv = 10.0           

# Wong
quark = 'charm'     
mass = 1.27     
# tau_form = 1/(2*mass)*hbarc   
pT = 2 
mT = np.sqrt(mass*2+pT**2)
tau_form = 1/(2*mT)*hbarc 

ntp = 10**5  
nevents = 2    
representation = 'quantum fundamental'      
boundary = 'periodic'
initialization = 'toy'       

# Store relevant parameters in a dictionary
p = {
    'QS': Qs,
    'TAU_SIM': tau_sim,
    'QUARK': quark,
    'MASS': mass,           
    'TFORM': tau_form,       
    'PT': pT,       
    'NTP' : ntp,
    }

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
from curraun.wong import init_pos, init_charge, init_mom_toy

# Kappa module
import curraun.kappa as kappa

import pickle
from tqdm import tqdm

current_path = os.getcwd() 
results_folder = 'results'
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)
results_path = current_path + '/' + results_folder + '/'

# Simulation routine
def simulate(p, ev): 
    Qs = p['QS']
    tau_sim = p['TAU_SIM']*3/Qs
    g2mu = Qs / 0.8     
    g = np.pi * np.sqrt(1 / np.log(Qs / 0.2))          		
    mu = g2mu / g**2          	
    ir = 0.1 * g2mu 

    mass = p['MASS']
    pT = p['PT']
    tau_form = p['TFORM']

    # Derived parameters
    a = L / N
    E0 = N / L * hbarc
    DT = 1.0 / DTS
    formt = int(tau_form / a * DTS)
    maxt = int(tau_sim / a * DTS)

    tau = np.linspace(0, tau_sim, maxt)
    # tau = np.linspace(0, tau_sim + tau_form, maxt+formt)

    # Initialize Glasma fields
    s = core.Simulation(N, DT, g)
    va = mv.wilson(s, mu=mu / E0, m=ir / E0, uv=uv / E0, num_sheets=ns)
    vb = mv.wilson(s, mu=mu / E0, m=ir / E0, uv=uv / E0, num_sheets=ns)
    initial.init(s, va, vb)

    # Initialize the Wong solver
    wong_solver = wong.WongSolver(s, ntp)
    x0s, p0s, q0s = np.zeros((ntp, 3)), np.zeros((ntp, 5)), np.zeros((ntp, su.ALGEBRA_ELEMENTS))
    masses = mass / E0 * np.ones(ntp)

    for i in range(ntp):
        x0, p0, q0 = init_pos(s.n), init_mom_toy('pT', pT / E0), init_charge(representation)
        x0s[i, :], p0s[i, :], q0s[i, :] = x0, p0, q0

    wong_solver.initialize(x0s, p0s, q0s, masses)

    # Momentum broadening from the Wong solver
    mom_broad = np.zeros((maxt, 4))

    with tqdm(total=maxt+formt) as pbar:
        for t in range(maxt+formt):
            # Solve Wong's equations
            if t>=formt:  
                # Compute momentum broadening from Wong solver
                mom_broad[t-formt] = wong_solver.p_sq_mean * E0**2
                wong_solver.evolve()
                wong_solver.compute_mom_broad()

            # Evolve Glasma fields
            core.evolve_leapfrog(s)

            pbar.set_description("Event " + str(ev+1))
            pbar.update(1)

    tau = np.linspace(0, tau_sim-tau_form, maxt-formt)
    # output['quark'], output['Qs']= p['quark'], p['Qs']
    output = {}
    output['parameters'] = p.copy()
    output['mom_broad'], output['tau'] = mom_broad, tau

    os.chdir(results_path)
    wong_folder = p['FOLDER']
    if not os.path.isdir(wong_folder):
        os.makedirs(wong_folder)
    wong_path = results_path + wong_folder + '/'
    os.chdir(wong_path)

    filename = 'event_' + str(ev+1) + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle)

    return mom_broad, tau

quarks = ['charm', 'beauty']
quark_masses = [1.27, 4.18]

# pTs = [0, 2, 5, 10]
Qss = [1, 1.5, 2, 2.5, 3]
# formation_times = [0.02, 0.08]

mom_broad, tau = {}, {}
for iq in range(len(quarks)):
    print(quarks[iq].capitalize() + " quark")
    p['QUARK'], p['MASS'] = quarks[iq], quark_masses[iq]
    mT = np.sqrt(p['MASS']**2+pT**2)
    p['TFORM'] = 1/(2*mT)*hbarc 
    # p['TFORM'] = formation_times[iq]

    for Qs in Qss:
        print('Saturation momentum', Qs, 'GeV')
        p['QS'] = Qs
        tag = quarks[iq] + '_Qs_' + str(Qs)

        p['FOLDER'] = 'mom_broad_' + tag

        mom_broads = []
        for ev in range(nevents):
            mom_broad_ev, tau_wong = simulate(p, ev)
            mom_broads.append(mom_broad_ev)

        mom_broad[tag] = np.mean(mom_broads, axis=0)
        tau[tag] = tau_wong

    output = {}
    output['quarks'], output['Qs'], output['masses'] = quarks, Qss, quark_masses
    output['mom_broad'], output['tau'] = mom_broad, tau
    filename = results_path + 'mom_broad_Qs_dep_' + quarks[iq] + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle)

output = {}
output['quarks'], output['Qs'], output['masses'] = quarks, Qss, quark_masses
output['mom_broad'], output['tau'] = mom_broad, tau
filename = results_path + 'mom_broad_Qs_dep_charm_beauty.pickle'
with open(filename, 'wb') as handle:
    pickle.dump(output, handle)