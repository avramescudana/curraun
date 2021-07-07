# Set environment variables 
import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["GAUGE_GROUP"] = "su3"
os.environ["PRECISION"] = "double"

# Import curraun and other packages
import sys
sys.path.append('..')

import curraun.core as core
import curraun.mv as mv
import curraun.initial as initial
initial.DEBUG = False
from curraun.electric_fields import ElectricFields
from curraun.electric_fields_correlators import ElectricFieldsCorrelators

from curraun.numba_target import use_cuda, use_numba
if use_cuda:
    from numba import cuda

import numpy as np
import pickle
from tqdm import tqdm

# Define hbar*c [GeV*fm]
hbarc = 0.197326 

def simulate(p):    
    # Derived parameters
    a = p['L'] / p['N']
    E0 = p['N'] / p['L'] * hbarc
    p['E0'] = E0
    DT = 1.0 / p['DTS']
    maxt = int(p['TMAX'] / a) * p['DTS']
    formt = int(p['TFORM'] / a) * p['DTS']

    s = core.Simulation(p['N'], DT, p['G'])
    va = mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
    vb = mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
    initial.init(s, va, vb)

    elfields = ElectricFields(s)
    elfieldscorr = ElectricFieldsCorrelators(s)

    if use_cuda:
        s.copy_to_device()

    tau = []
    corr_ex, corr_ey, corr_ez  = [], [], []

    for t in range(maxt):
        core.evolve_leapfrog(s)

        if t==formt:  
            elfields = elfields.compute()
            Exform, Eyform, Ezform = elfields[:, 0], elfields[:, 1], elfields[:, 2]

        elif t>=formt:
            ExformEx, EyformEy, EzformEz = elfieldscorr.compute(Exform, Eyform, Ezform)
            # Fields correlator units [GeV / fm^3]
            units = E0 ** 4 / hbarc ** 3
            corr_ex.append(ExformEx * units)
            corr_ey.append(EyformEy * units)
            corr_ez.append(EzformEz * units)

            current_tau = t / p['DTS'] * a
            tau.append(current_tau)

    if use_cuda:
        s.copy_to_host()

    return p, s, tau, corr_ex, corr_ey, corr_ez


# Number of color sheets
ns = 50
# Ratio Qs/g^2\mu
factor = 0.8
# Mass number for Pb
A = 207
# Center-of-mass energy for Pb-Pb at 5.02 TeV [GeV]
sqrts = 5020
# Saturation scale [GeV]
Qs = np.sqrt(0.13 * A**(1/3) * sqrts**0.25)		
# Running coupling constant
g = np.pi * np.sqrt(1 / np.log(Qs / 0.2))		
# MV model parameter [GeV]
mu = Qs / (g**2 * factor)	
# Infrared regulator [GeV]
m = 0.1 * g**2 * mu   

# Formation times [GeV/fm]
taus_form = [0, 0.05, 0.1]
# Simulation time [GeV/fm]
tau_s = 0.3

# Length of simulation box [fm]
L = 15
# Number of discretization points
N = 1024
# Numerical time interval
DTS = 16
# Number of events
n_events = 10

all_corr_ex, all_corr_ey, all_corr_ez = [], [], []

for tau_form in taus_form:
    print("Formation time = {:1.2f} [GeV/fm]".format(tau_form))

    # Standard MV model paramaters
    p = {
        # Parameters for simulation box
        'L':    L,           # transverse size [fm]
        'N':    N,            # lattice size
        'DTS':  DTS,             # time steps per transverse spacing
        'TMAX': tau_s,          # max. proper time (tau) [fm/c]
        'TFORM': tau_form,       # formation time of the HQ [fm/c]

        # Parameters for MV model
        'G':    g,            # YM coupling constant
        'MU':   mu,             # MV model parameter [GeV]
        'M':    m,              # IR regulator [GeV]
        # 'UV':   10000.0,      # UV regulator [GeV]
        'UV':   10.0, 
        'NS':   ns,             # number of color sheets
    }

    with tqdm(total=n_events, file=sys.stdout) as pbar:
        for nev in range(0, n_events):
            p, s, tau, mean_corr_ex, mean_corr_ey, mean_corr_ez = simulate(p)

            all_corr_ex.append(mean_corr_ex)
            all_corr_ey.append(mean_corr_ey)
            all_corr_ez.append(mean_corr_ez)

            pbar.set_description('Event {} / {}'.format(nev+1, n_events))
            pbar.update(1)

    avg_corr_ex, avg_corr_ey, avg_corr_ez = np.mean(all_corr_ex, axis = 0), np.mean(all_corr_ey, axis = 0), np.mean(all_corr_ez, axis = 0)

    std_corr_ex, std_corr_ey, std_corr_ez = np.std(all_corr_ex, axis = 0), np.std(all_corr_ey, axis = 0), np.std(all_corr_ez, axis = 0)

    output = {}
    output["avg_corr_ex"], output["std_corr_ex"] = avg_corr_ex, std_corr_ex
    output["avg_corr_ey"], output["std_corr_ey"] = avg_corr_ey, std_corr_ey
    output["avg_corr_ez"], output["std_corr_ez"] = avg_corr_ez, std_corr_ez

    fname = "./elfieldscorr_tauform_{}.pickle".format(tau_form)
    pickle.dump(output, open(fname, "wb"))
    print("Output: {}".format(fname))
