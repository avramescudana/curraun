# Set environment variables 
import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["GAUGE_GROUP"] = "su3"
os.environ["PRECISION"] = "double"
su_group = os.environ["GAUGE_GROUP"]
quark = 'infmass'

# Import curraun and other packages
import sys
sys.path.append('..')

import curraun.core as core
import curraun.mv as mv
import curraun.kappa as kappa
import curraun.initial as initial
initial.DEBUG = False

from curraun.numba_target import use_cuda, use_numba
if use_cuda:
    from numba import cuda

import numpy as np

# define hbar * c in units of GeV * fm
hbarc = 0.197326 

def compute(p):    

    # derived parameters
    a = p['L'] / p['N']
    E0 = p['N'] / p['L'] * hbarc
    DT = 1.0 / p['DTS']
    maxt = int(p['TMAX'] / a) * p['DTS']
    formt = int(p['TFORM'] / a) * p['DTS']

    all_px, all_py, all_pz = [], [], []
    
    # event loop
    for e in range(p['NE']):
        s = core.Simulation(p['N'], DT, p['G'])
        va = mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
        vb = mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
        initial.init(s, va, vb)
        
        kappa_tforce = kappa.TransportedForce(s)
        px, py, pz = [], [], []
        tau = []

        if use_cuda:
            s.copy_to_device()
            kappa_tforce.copy_to_device()
            meminfo = cuda.current_context().get_memory_info()
        
        for t in range(maxt+1):
            core.evolve_leapfrog(s)
            
            if t>= formt:
                if t % p['DTS'] == 0:
                    if use_cuda:
                        kappa_tforce.copy_mean_to_host()

                    # unit factors (GeV^2)
                    units = E0 ** 2 / (s.g ** 2)

                    # color factors (for quarks)
                    Nc = core.su.NC
                    f = 2 * s.g ** 2 / (2 * Nc)
                    px.append(kappa_tforce.p_perp_mean[0] * units * f)
                    py.append(kappa_tforce.p_perp_mean[1] * units * f)
                    pz.append(kappa_tforce.p_perp_mean[2] * units * f)

                    tau.append(t*a/p['DTS'])
                    
                    if use_cuda:
                        kappa_tforce.copy_mean_to_device()

                kappa_tforce.compute()
        
        px = np.array(px)
        py = np.array(py)
        pz = np.array(pz)

        filename = 'tauform_' + str(p['TFORM']) + '_ev_' + str(e) + '.npz'
        np.savez(filename, px=px, py=py, pz=pz, tau=tau)

        all_px.append(px)
        all_py.append(py)
        all_pz.append(pz)

    px_mean, px_std = np.mean(all_px, axis=0), np.std(all_px, axis=0)
    py_mean, py_std = np.mean(all_py, axis=0), np.std(all_py, axis=0)
    pz_mean, pz_std = np.mean(all_pz, axis=0), np.std(all_pz, axis=0)

    filename = 'tauform_' + str(p['TFORM']) + '_mean.npz'
    np.savez(filename, px_mean=px_mean, px_std=px_std, py_mean=py_mean, py_std=py_std, pz_mean=pz_mean, pz_std=pz_std, tau=tau)
    
    if use_cuda:
        cuda.current_context().deallocations.clear()

current_path = os.getcwd() 
results_folder = 'results'
check_results_folder = os.path.isdir(results_folder)
if not check_results_folder:
    os.makedirs(results_folder)
results_path = current_path + '/' + results_folder + '/'
os.chdir(results_path)

kappa_folder = 'pb+pb_5020gev_' + su_group + '_kappa'
check_kappa_folder = os.path.isdir(kappa_folder)
if not check_kappa_folder:
    os.makedirs(kappa_folder)
kappa_path = results_path + '/' + kappa_folder + '/'
os.chdir(kappa_path)

def run_simulation():
    p = {
        'L':    L,               # transverse size [fm]
        'N':    N,            # lattice size
        'DTS':  DTS,              # time steps per transverse spacing
        'TMAX': tau_s,         # max. proper time (tau) [fm/c]

        'G':    g,               # YM coupling constant
        'MU':   mu,              # MV model parameter [GeV]
        'M':    m,               # IR regulator [GeV]
        'UV':   10.0,       # UV regulator [GeV]
        'NS':   ns,              # number of color sheets

        'NE':   20,              # number of events
        'TFORM': tau_form,      # formation time
    }
    
    compute(p)

# Length of simulation box [fm]
L = 10
# Number of lattice sites
N = 512
# Simulation time [fm/c]
tau_s = 2.0
# Time step
DTS = 8
# Formation time [fm/c]
if quark=='charm':
    tau_form = 0.06
elif quark=='beauty':
    tau_form = 0.02
elif quark=='infmass':
    tau_form = 0.00

# MV model parameters for Pb-Pb 5.02 TeV
ns = 50
factor = 0.8
A = 207
sqrts = 5020

Qs = np.sqrt(0.13 * A**(1/3) * sqrts**0.25)		
g = np.pi * np.sqrt(1 / np.log(Qs / 0.2))		
mu = Qs / (g**2 * factor)	
m = 0.1 * g**2 * mu   

run_simulation()