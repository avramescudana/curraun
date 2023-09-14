import numpy as np

# hbar * c [GeV * fm]
hbarc = 0.197326 

# Simulation box 
L = 10      
N = 512  
# N = 1024  
tau_sim = 1.5
# tau_sim = 1
DTS = 8     

# Glasma fields
su_group = 'su3'

Qs = 2        
ns = 50    
factor = 0.8        
g2mu = Qs / factor     
g = np.pi * np.sqrt(1 / np.log(Qs / 0.2))          		
mu = g2mu / g**2          	
ir = 0.1 * g**2 * mu  
 
uv = 10.0           

# Wong solver 
quark = 'any'
mass = 1   
tau_form = 0   
pT = 0   
ntp = 10**5  
nevents = 10    
boundary = 'periodic'       

# Store relevant parameters in a dictionary
p = {
    'QUARK': quark,
    'MASS': mass, 
    'TFORM': tau_form,
    'TSIM': tau_sim,
    'QS': Qs,            
    'NEVENTS': nevents,
    'NTP': ntp,   
    'PT': pT,
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
wong.CUB_MOM = False
from curraun.wong import init_mom_toy, init_pos

from scipy.stats import unitary_group

# gell-mann matrices

gm = [
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(3), 0], [0, 0, -2 / np.sqrt(3)]]
]

T = np.array(gm) / 2.0

def init_charge(q2):
    """
        Step 1: specific random color vector
    """
    # Here q_3=0
    q0 = [np.sqrt(q2), 0., 0., 0., 0., 0., 0., 0.]
    Q0 = np.einsum('ijk,i', T, q0)

    """
        Step 2: create a random SU(3) matrix to rotate Q.
    """
    
    V = unitary_group.rvs(3)
    detV = np.linalg.det(V)
    U = V / detV ** (1 / 3)
    Ud = np.conj(U).T

    Q = np.einsum('ab,bc,cd', U, Q0, Ud)

    """
        Step 3: Project onto color components
    """

    q = 2 * np.einsum('ijk,kj', T, Q)
    return np.real(q)

import pickle
from tqdm import tqdm

# Simulation routine
def simulate(p, ev): 
    q2 = p["q2"]

    output = {}
    output["parameters"] = p.copy()

    tau_form = p["TFORM"]
    tau_sim = p["TSIM"] + tau_form 
    mass = p["MASS"]
    pT = p["PT"]

    # Derived parameters
    a = L / N
    E0 = N / L * hbarc
    DT = 1.0 / DTS
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
    masses = mass / E0 * np.ones(ntp)

    for i in range(ntp):
        x0, p0, q0 = init_pos(s.n), init_mom_toy('pT', pT / E0), init_charge(q2)
        x0s[i, :], p0s[i, :], q0s[i, :] = x0, p0, q0

    wong_solver.initialize(x0s, p0s, q0s, masses)

    psq = np.zeros((maxt-formt, 4)) 

    with tqdm(total=maxt) as pbar:
        for t in range(maxt):
            # Evolve Glasma fields
            core.evolve_leapfrog(s)

            # Solve Wong's equations
            if t>=formt:  
                psq_wong = wong_solver.p_sq_mean
                psq[t-formt, :] = psq_wong * E0**2

                wong_solver.evolve()
                wong_solver.compute_mom_broad()

            pbar.set_description("Event " + str(ev+1))
            pbar.update(1)

    formt, maxt = int(tau_form / L * N * DTS), int(tau_sim / L * N * DTS)
    tau = np.linspace(0, tau_sim, maxt-formt)

    output["psq"], output["tau"] = psq, tau

    return output

# q2s = [4/3, 4.0]
q2s = np.linspace(0, 6, 19)

quarks = ['infmass', 'beauty', 'charm']
quark_masses = [10**6, 4.18, 1.27]

# quarks = ['beauty', 'charm']
# quark_masses = [4.18, 1.27]

# quarks = ['infmass']
# quark_masses = [10**6]

psq, tau = {}, {}
for iq, quark in enumerate(quarks):
    print(quarks[iq].capitalize() + " quark")
    p['QUARK'], p['MASS'] = quark, quark_masses[iq]
    # transverse mass
    mT = np.sqrt(p["MASS"]**2+p["PT"]**2)   
    p['TFORM'] = 1/(2*mT)*hbarc
    if quark=='infmass':
        p['TFORM'] = 0.

    psq[quark], tau[quark] = {}, {}

    for q2 in q2s:
        p["q2"] =  q2
        print("q2 =", p["q2"])

        psqs = []

        
        for ev in range(nevents):
            output = simulate(p, ev)
            psqs.append(output["psq"])

        psq[quark][str(q2)] = np.mean(psqs, axis=0)

    tau[quark] = output["tau"]
    

output = {}
output["psq"], output["tau"] = psq, tau
output["quarks"], output["q2s"] = quarks, q2s

filename = 'mom_broad_q2_dep_more_events.pickle'
with open(filename, 'wb') as handle:
    pickle.dump(output, handle)