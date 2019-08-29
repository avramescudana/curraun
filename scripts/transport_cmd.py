########################################
## Environment variables that can be set
import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"
# os.environ["NUMBA_NUM_THREADS"] = "1"
# os.environ["NUMBA-WARNINGS"] = "1"
# os.environ["NUMBA_DEBUG_ARRAY_OPT_STATS"] = "1"
# os.environ["NUMBA_OPT"] = "3"  # LLVM optimziation level
# os.environ["NUMBA_DUMP_OPTIMIZED"] = "1"
# os.environ["NUMBA_DEBUG_ARRAY_OPT_STATS"] = "1"
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"  # enable CUDA simulator

## Custom environment variables
# os.environ["MY_NUMBA_TARGET"] = "cuda"    # Target GPU using CUDA
# os.environ["MY_NUMBA_TARGET"] = "numba"     # Target CPU
# os.environ["MY_NUMBA_TARGET"] = "python"  # Pure Python version

# os.environ["GAUGE_GROUP"] = "su3"
# os.environ["GAUGE_GROUP"] = "su2"
########################################


import time
from collections import deque
import datetime

from curraun.numba_target import use_cuda, use_numba
if use_cuda:
    from numba import cuda
import curraun
import curraun.core as core
import curraun.mv as mv
import curraun.initial as initial
import numpy as np
import curraun.kappa as kappa
import curraun.qhat as qhat
import argparse
from scipy import stats

"""
    Standard parameters (can be overwritten via passing arguments)
"""

# filename
p = {
    'FN': "output",         # file basename

    'L':    10.0,           # transverse size [fm]
    'N':    64,            # lattice size
    'DTS':  2,              # time steps per transverse spacing
    'TMAX': 10.0,            # max. proper time (tau) [fm/c]

    'G':    2.0,            # YM coupling constant
    'MU':   0.5,            # MV model parameter [GeV]
    'M':    0.2,            # IR regulator [GeV]
    'UV':   10.0,           # UV regulator [GeV]
    'NS':   1,             # number of color sheets

    'NE':   50,             # number of events
}

"""
    Argument parsing
"""
parser = argparse.ArgumentParser(description='Compute momentum broadening in the 2+1D Glasma.')

parser.add_argument('-FN',   type=str,   help="Filename")

parser.add_argument('-L',    type=float, help="transverse size [fm]")
parser.add_argument('-N',    type=int,   help="lattice size")
parser.add_argument('-DTS',  type=int,   help="time steps per transverse spacing")
parser.add_argument('-TMAX', type=float, help="max. proper time [fm/c]")

parser.add_argument('-G',    type=float, help="YM coupling constant")
parser.add_argument('-MU',   type=float, help="MV model parameter [GeV]")
parser.add_argument('-M',    type=float, help="IR regulator [GeV]")
parser.add_argument('-UV',   type=float, help="UV regulator [GeV]")
parser.add_argument('-NS',   type=int,   help="Number of color sheets")

parser.add_argument('-NE',   type=int,   help="Number of events")

# parse args and update parameters dict
args = parser.parse_args()
data = args.__dict__
for d in data:
    if data[d] is not None:
        p[d] = data[d]

# derived parameters
a = p['L'] / p['N']
E0 = p['N'] / p['L'] * 0.197326
DT = 1.0 / p['DTS']
maxt = int(p['TMAX'] / a) * p['DTS']

# convert basename to filenames
fn_kappa = p['FN'] + "_kappa.dat"
fn_qhat = p['FN'] + "_qhat.dat"

fn_mean = p['FN'] + "_mean.dat"

# output arrays
"""
    Format description for x_kappa.dat and x_qhat.dat

number of columns: number of time steps
number of rows: 3*(number of events) + 1

rows: tau or p_i^2
columns: time index

row 0: tau (to translate time index into tau)

row 1: p_x^2 of event 0
row 2: p_y^2 of event 0
row 3: p_z^2 of event 0

row 4: p_x^2 of event 1
row 5: p_y^2 of event 1
row 6: p_z^2 of event 1

...

Averaging over events must be performed afterwards using these files.

    Format description for x_kappa_mean.dat and x_qhat_mean.dat

rows: tau or p_i^2
columns: time index

row 0: tau


row 1: mean of p_x^2 (qhat)
row 2: mean of p_y^2 (qhat)
row 3: mean of p_z^2 (qhat)

row 4: standard deviation of p_x^2 (qhat)
row 5: standard deviation of p_y^2 (qhat)
row 6: standard deviation of p_z^2 (qhat)

repeat for kappa

"""

results_kappa = np.zeros((3 * p['NE'] + 1, int(maxt / p['DTS'])))
results_kappa[0, :] = np.linspace(0.0, int(maxt / p['DTS']) * a, num=int(maxt / p['DTS']))

results_qhat = np.zeros((3 * p['NE'] + 1, int(maxt / p['DTS'])))
results_qhat[0, :] = np.linspace(0.0, int(maxt / p['DTS']) * a, num=int(maxt / p['DTS']))

progress_list = deque([[0.0, 0.0]])
eta_output = 0

init_time = time.time()


# event loop
for e in range(p['NE']):
    # initialization
    s = curraun.core.Simulation(p['N'], DT, p['G'])
    va = curraun.mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
    vb = curraun.mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
    curraun.initial.init(s, va, vb)

    print("Memory of data: {} GB".format(s.get_ngb()))

    # save simulation data
    kappa_tforce = kappa.TransportedForce(s)
    qhat_tforce = qhat.TransportedForce(s)

    if use_cuda:
        s.copy_to_device()
        kappa_tforce.copy_to_device()
        qhat_tforce.copy_to_device()

        meminfo = cuda.current_context().get_memory_info()
        print("CUDA free memory: {:.2f} GB of {:.2f} GB.".format(meminfo[0] / 1024 ** 3, meminfo[1] / 1024 ** 3))

    for t in range(maxt):
        if t % p['DTS'] == 0:
            if use_cuda:
                # Copy data from device to host
                kappa_tforce.copy_mean_to_host()
                qhat_tforce.copy_mean_to_host()

            # tau in fm/c
            tau = s.t * a
            progress = (tau/p['TMAX'] + float(e)) / p['NE']
            cur_time = time.time() - init_time
            progress_list.append([progress, cur_time])

            if t > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(progress_list)
                eta = intercept + slope * 1.0 - cur_time
                try:
                    eta_output = str(datetime.timedelta(seconds=eta))
                except ValueError:
                    eta_output = "---"

            if len(progress_list) >= 100:
                progress_list.popleft()

            print("tau = {:.3}, tau_max = {:.3}, event = {} of {}, {}%, ETA: {}".format(tau, p['TMAX'], e+1, p['NE'], int(100 * tau / p['TMAX']), eta_output))

            # unit factors (GeV^2)
            units = E0 ** 2 / (s.g ** 2)

            # color factors (for quarks)
            Nc = curraun.core.su.NC
            f = 2 * s.g ** 2 / (2 * Nc)

            # p_perp components for kappa
            for d in range(3):
                results_kappa[3 * e + d + 1, int(t / p['DTS'])] = kappa_tforce.p_perp_mean[d] * units * f
                results_qhat[3 * e + d + 1, int(t / p['DTS'])] = qhat_tforce.p_perp_mean[d] * units * f

            if use_cuda:
                # Copy data back to device
                kappa_tforce.copy_mean_to_device()
                qhat_tforce.copy_mean_to_device()

        kappa_tforce.compute()
        qhat_tforce.compute()
        curraun.core.evolve_leapfrog(s)

    np.savetxt(fname=fn_kappa, X=results_kappa)
    np.savetxt(fname=fn_qhat, X=results_qhat)

np.savetxt(fname=fn_kappa, X=results_kappa)
np.savetxt(fname=fn_qhat, X=results_qhat)

"""
    Statistical averages
"""

if p['NE'] > 1:
    tau = results_kappa[0, :]
    num_tau = len(tau)

    results = np.zeros((num_tau, 1+6+6))
    results[:, 0] = tau

    results_kappa = results_kappa[1:]
    results_qhat = results_qhat[1:]

    for t in range(num_tau):
        # qhat averages
        means = results_qhat[:, t].reshape(p['NE'], 3).mean(axis=0)
        stds = results_qhat[:, t].reshape(p['NE'], 3).std(axis=0)

        for d in range(3):
            results[t, 1+d] = means[d]

        for d in range(3):
            results[t, 1+3+d] = stds[d]

        # kappa averages
        means = results_kappa[:, t].reshape(p['NE'], 3).mean(axis=0)
        stds = results_kappa[:, t].reshape(p['NE'], 3).std(axis=0)

        for d in range(3):
            results[t, 1+6+d] = means[d]

        for d in range(3):
            results[t, 1+9+d] = stds[d]

    np.savetxt(fname=fn_mean, X=results)
else:
    print("Skipping statistical averages because only a single event was computed.")