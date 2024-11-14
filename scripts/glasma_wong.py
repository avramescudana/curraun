###############################################
########### import general packages ###########
###############################################

import numpy as np
import os
import sys
import pickle
from tqdm import tqdm

import argparse

#############################################
########### simulation parameters ###########
#############################################

# hbar * c [GeV * fm]
hbarc = 0.197326 

# Simulation box 
L = 10      
N = 512    
tau_switch = 0.3    
DTS = 8     

# Glasma
su_group = 'su3'
Qs = 2.0        
#TODO: Qs mapping to sqrts
central = True
#TODO: b-MV model for centrality dependence
ns = 50    
factor = 0.8            
grun = True   
g = np.pi * np.sqrt(1 / np.log(Qs / 0.2))   
uv = 10.0           
nevents = 1

# Wong
quark = 'charm'     
mass = 1.275     
form_type = 'fixed'
#TODO: more formation time options
tau_form = 1/(2*mass)*hbarc   
pT = 0.5    
ntp = 10**5  
representation = 'quantum fundamental'      
boundary = 'periodic'
# initialization = 'toy'      
xT_distrib = 'random'
# pT_distrib = 'toy'
pT_distrib = 'fonll'
x_fonll = [20.2837, 1.95061, 3.13695, 0.0751663]

outfile = 'output'

# Store relevant parameters in a dictionary
p = {
    # Glasma - field parameters
    'out':       outfile,         # output file name
    'central':   central,         # centrality dependence
    'L':         L,               # transverse size [fm]
    'N':         N,               # lattice size
    'DTS':       DTS,             # time steps per transverse spacing
    'tswitch':   tau_switch,      # switching proper time (tau) [fm/c]
    'Qs':        Qs,              # saturation scale [GeV]
    'grun':      grun,            # running coupling constant
    'g':         g,               # coupling constant
    'factor':    factor,          # factor for ratio g2mu/Qs   
    'uv':        uv,              # UV regulator [GeV]
    'ns':        ns,              # number of color sheets
    'nevents':   nevents,         # number of events
    # Wong - particle parameters
    'quark':     quark,           # heavy quark type
    'mass':      mass,            # quark mass [GeV]         
    'formtype':  form_type,       # formation time type
    'tform':     tau_form,        # formation time [fm/c]       
    'ntp' :      ntp,             # number of test particles
    'bound':     boundary,        # boundary conditions
    # 'init':      initialization,  # initialization method
    'pt':        pT,              # particle transverse momentum [GeV]
    'xtdistrib': xT_distrib,      # transverse position distribution
    'ptdistrib': pT_distrib,      # transverse momentum distribution
    'xfonll':    x_fonll          # FONLL fit parameters
    }

########################################
########### argument parsing ###########
########################################

parser = argparse.ArgumentParser(description='Run the glasma + wong simulations for heavy quarks')

parser.add_argument('-out',       type=str,   help="Output file name") 
parser.add_argument('-central',   type=bool,  help="Centrality dependence")
parser.add_argument('-L',         type=float, help="Transverse size [fm]")
parser.add_argument('-N',         type=int,   help="Lattice size")
parser.add_argument('-DTS',       type=int,   help="Time steps per transverse spacing")
parser.add_argument('-tswitch',   type=float, help="Switching proper time (tau) [fm/c]")
parser.add_argument('-Qs',        type=float, help="Saturation scale [GeV]")
parser.add_argument('-grun',      type=bool,  help="Running coupling constant")
parser.add_argument('-g',         type=float, help="Coupling constant")
parser.add_argument('-factor',    type=float, help="Factor for ratio g2mu/Qs")
parser.add_argument('-uv',        type=float, help="UV regulator [GeV]")
parser.add_argument('-ns',        type=int,   help="Number of color sheets")
parser.add_argument('-nevents',   type=int,   help="Number of events")
parser.add_argument('-quark',     type=str,   help="Heavy quark type")
parser.add_argument('-mass',      type=float, help="Quark mass [GeV]")
parser.add_argument('-formtype',  type=str,   help="Formation time type")
parser.add_argument('-tform',     type=float, help="Formation time [fm/c]")
parser.add_argument('-ntp',       type=int,   help="Number of test particles")
parser.add_argument('-bound',     type=str,   help="Boundary conditions")
# parser.add_argument('-init',      type=str,   help="Initialization method")
parser.add_argument('-pt',        type=float, help="Particle transverse momentum [GeV]")
parser.add_argument('-xtdistrib', type=str,   help="Transverse position distribution")
parser.add_argument('-ptdistrib', type=str,   help="Transverse momentum distribution")
parser.add_argument('-xfonll',    type=float, nargs='+', help="FONLL fit parameters")

# Parse arguments and update parameters dictionary
args = parser.parse_args()
data = args.__dict__
for d in data:
    if data[d] is not None:
        p[d] = data[d]

# Derived parameters
if p['grun']:
    g = np.pi * np.sqrt(1 / np.log(Qs / 0.2))   
g2mu = Qs / factor 
mu = g2mu / g**2          	
ir = 0.1 * g**2 * mu
if p['central']:
    p['bound'] = 'periodic'
    # p['init'] = 'toy'
    p['pt_distrib'] = 'toy'
if p['formtype'] == 'fixed':
    p['tform'] = 1/(2*p['mass'])*hbarc   
#TODO: particle initialization for off-central collisions

####################################################
############ set environment variables #############
####################################################

# Run on specific GPU, comment this at the end #TODO: remove this
os.environ["CUDA_VISIBLE_DEVICES"]="4"

os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["PRECISION"] = "double"
os.environ["GAUGE_GROUP"] = su_group

####################################################
########### import glasma + wong modules ###########
####################################################

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
# Important parameter, used to compute (x,y,z) and (px,py,pz) for each test particle
wong.SWITCH =  True
from curraun.wong import init_pos, init_charge, init_mom_toy, init_mom_fonll

####################################################
######### glasma + wong simulation routine #########
####################################################

# Simulate glasma event ev with parameters p and ntp heavy quarks
def simulate(p, ev): 
    mass = p['mass']
    tau_form = p['tform']
    L = p['L']
    N = p['N']
    DTS = p['DTS']
    tau_sim = p['tswitch']
    g = p['g']
    ns = p['ns']
    uv = p['uv']
    ntp = p['ntp']
    pT = p['pt']

    # Derived parameters
    a = L / N
    E0 = N / L * hbarc
    DT = 1.0 / DTS
    formt = int(tau_form / a * DTS)
    maxt = int(tau_sim / a * DTS)

    # Initialize Glasma fields
    s = core.Simulation(N, DT, g)
    va = mv.wilson(s, mu = mu / E0, m = ir / E0, uv = uv / E0, num_sheets = ns)
    vb = mv.wilson(s, mu = mu / E0, m = ir / E0, uv = uv / E0, num_sheets = ns)
    initial.init(s, va, vb)

    # Initialize the Wong solver
    wong_solver = wong.WongSolver(s, ntp)
    x0s, p0s, q0s = np.zeros((ntp, 3)), np.zeros((ntp, 5)), np.zeros((ntp, su.ALGEBRA_ELEMENTS))
    masses = mass / E0 * np.ones(ntp)

    # if p['init'] == 'toy':
    if p['pt_distrib'] == 'toy':
        # Initialize test particles with toy model
        for i in range(ntp):
            x0, p0, q0 = init_pos(s.n), init_mom_toy('pT', pT / E0), init_charge(representation)
            x0s[i, :], p0s[i, :], q0s[i, :] = x0, p0, q0
    #TODO: FONLL pT distribution
    if p['pt_distrib'] == 'fonll':
        x_fonll = p['xfonll']
        # Initialize test particles with toy model
        for i in range(ntp):
            p0s, ntp_fonll = init_mom_fonll(p, x_fonll)
            # Number of test particle slightly changes when sampling the FONLL pT distribution
            ntp = ntp_fonll
            x0, q0 = init_pos(ntp), init_charge(representation)
            x0s[i, :], q0s[i, :] = x0, q0

    wong_solver.initialize(x0s, p0s, q0s, masses)

    # Coordinate and positions at switch time
    xs, ps = np.zeros((ntp, 3)), np.zeros((ntp, 3))

    with tqdm(total=maxt+formt) as pbar:
        for t in range(maxt+formt):
            # Solve Wong's equations
            if t>=formt:  
                wong_solver.evolve()
            
            if t==maxt+formt-1:
                 # Compute momentum broadening from Wong solver
                wong_solver.compute_xp_switch()
                xs = wong_solver.xs * a
                ps = wong_solver.ps * E0

            # Evolve Glasma fields
            core.evolve_leapfrog(s)

            pbar.set_description("Event " + str(ev+1))
            pbar.update(1)

    return xs, ps

####################################################
########## run for multiple glasma events ##########
####################################################

collect_xs, collect_ps = [], []
for ev in range(nevents):
    xs_ev, ps_ev = simulate(p, ev)
    collect_xs.append(xs_ev)
    collect_ps.append(ps_ev)

####################################################
############# output results to files ##############
####################################################
