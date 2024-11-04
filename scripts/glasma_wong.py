###############################################
########### import general packages ###########
###############################################

import numpy as np
import os
import sys
import pickle
from tqdm import tqdm

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
tau_form = 1/(2*mass)*hbarc   
pT = 0.5    
ntp = 10**5  
representation = 'quantum fundamental'      
boundary = 'periodic'
initialization = 'toy'      

outfile = 'output'

# Store relevant parameters in a dictionary
p = {
    # Glasma - field parameters
    'out': outfile,         # output file name
    'central': central,    # centrality dependence
    'L':    L,           # transverse size [fm]
    'N':    N,            # lattice size
    'DTS':  DTS,              # time steps per transverse spacing
    'tswitch': tau_switch,            # switching proper time (tau) [fm/c]
    'Qs':   Qs,            # saturation scale [GeV]
    'grun':    grun,            # running coupling constant
    'g':    g,            # coupling constant
    'factor': factor,           # factor for ratio g2mu/Qs   
    'uv':   uv,           # UV regulator [GeV]
    'ns':   ns,             # number of color sheets
    'nevents':   nevents,             # number of events
    # Wong - particle parameters
    'quark': quark,        # heavy quark type
    'mass': mass,         # quark mass [GeV]           
    'tform': tau_form,          # formation time [fm/c]       
    'ntp' : ntp,           # number of test particles
    'bound': boundary,        # boundary conditions
    'init': initialization,     # initialization method
    }

# Derived parameters
if p['GRUN']:
    g = np.pi * np.sqrt(1 / np.log(Qs / 0.2))   
g2mu = Qs / factor 
mu = g2mu / g**2          	
ir = 0.1 * g**2 * mu
if p['central']:
    p['bound'] = 'periodic'
    p['init'] = 'toy'
#TODO: particle initialization for off-central collisions


####################################################
########### import glasma + wong modules ###########
####################################################

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

########################################
########### argument parsing ###########
########################################

import argparse
