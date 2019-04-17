########################################
## Environment variables that can be set
import os

## Custom environment variables
# os.environ["MY_NUMBA_TARGET"] = "cuda"    # Target GPU using CUDA
# os.environ["MY_NUMBA_TARGET"] = "numba"     # Target CPU
# os.environ["MY_NUMBA_TARGET"] = "python"  # Pure Python version

# os.environ["GAUGE_GROUP"] = "su3"
# os.environ["GAUGE_GROUP"] = "su2"
########################################


import time
import datetime

from curraun.numba_target import use_cuda, use_numba
if use_cuda:
    from numba import cuda
import curraun
import curraun.core as core
import curraun.mv as mv
import curraun.initial as initial
import numpy as np

"""
    Standard parameters (can be overwritten via passing arguments)
"""

p = {
    'L':    40.0, # 10.0,           # transverse size [fm]
    'N':    1024, # 64,            # lattice size
    'DTS':  32, # 2,              # time steps per transverse spacing
    'TMAX': 10., # 1.0,            # max. proper time (tau) [fm/c]

    'G':    2.0,            # YM coupling constant
    'MU':   0.5,            # MV model parameter [GeV]
    'M':    0.2,            # IR regulator [GeV]
    'UV':   10.0,           # UV regulator [GeV]
    'NS':   1,             # number of color sheets

    'NE':   50,             # number of events
}

# derived parameters
a = p['L'] / p['N']
E0 = p['N'] / p['L'] * 0.197326
DT = 1.0 / p['DTS']
maxt = int(p['TMAX'] / a) * p['DTS']

np.random.seed(1)

# initialization
s = curraun.core.Simulation(p['N'], DT, p['G'])
va = curraun.mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
vb = curraun.mv.wilson(s, mu=p['MU'] / E0, m=p['M'] / E0, uv=p['UV'] / E0, num_sheets=p['NS'])
curraun.initial.init(s, va, vb)

print("Memory of data: {} GB".format(s.get_ngb()))

if use_cuda:
    s.copy_to_device()

    meminfo = cuda.current_context().get_memory_info()
    print("CUDA free memory: {:.2f} GB of {:.2f} GB.".format(meminfo[0] / 1024 ** 3, meminfo[1] / 1024 ** 3))

# Evolve once for Just-In-Time compilation
curraun.core.evolve_leapfrog(s)

init_time = time.time()

for t in range(maxt):
    curraun.core.evolve_leapfrog(s)
    if use_cuda:
        cuda.synchronize()
    total_time = time.time() - init_time
    if total_time > 20:
        break

if use_cuda:
    # s.copy_to_host()
    cuda.synchronize()

total_time = time.time() - init_time
seconds_per_step = total_time / t
estimated_total_time = seconds_per_step * maxt * p['NE']
estimated_total_time_formatted = str(datetime.timedelta(seconds=estimated_total_time))

print("Number of steps calculated: {}".format(t))
print("Total time: {}".format(total_time))
print("Seconds per simulation step: {}".format(seconds_per_step))
print("Estimated total time: {}".format(estimated_total_time_formatted))