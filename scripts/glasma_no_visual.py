########################################
## Environment variables that can be set
import os
#os.environ["NUMBA_DISABLE_JIT"] = "1"
#os.environ["NUMBA_NUM_THREADS"] = "4"
#os.environ["NUMBA-WARNINGS"] = "1"
#os.environ["NUMBA_DEBUG_ARRAY_OPT_STATS"] = "1"
#os.environ["NUMBA_OPT"] = "3"  # LLVM optimziation level
#os.environ["NUMBA_DUMP_OPTIMIZED"] = "1"
#os.environ["NUMBA_DEBUG_ARRAY_OPT_STATS"] = "1"
#os.environ["NUMBA_ENABLE_CUDASIM"] = "1"  # enable CUDA simulator

## Custom environment variables
# os.environ["CURRAUN_TARGET"] = "cuda"    # Target GPU using CUDA
# os.environ["CURRAUN_TARGET"] = "numba"     # Target CPU
# os.environ["CURRAUN_TARGET"] = "python"  # Pure Python version
########################################

# environment
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["GAUGE_GROUP"] = "su3"
os.environ["PRECISION"] = "double"

import time
from curraun.numba_target import use_cuda
import curraun.mv as mv
import numpy as np
import curraun.core as core
import curraun.initial as initial
import curraun.energy as energy

# simulation parameters
L = 12.9
M = 0.0001
MU = 0.5
G = 2.0
N = 1024 # 64 # 32 # 256
DT = 0.125
UV = 100.0
NUMS = 1

# initialization
initial.DEBUG = True
hbarc = 0.197326
E0 = N / L * hbarc
s = core.Simulation(N, DT, G)
va = mv.wilson(s, mu=MU / E0, m=M / E0, uv=UV / E0, num_sheets=NUMS)
vb = mv.wilson(s, mu=MU / E0, m=M / E0, uv=UV / E0, num_sheets=NUMS)
initial.init(s, va, vb)

print("Memory of data: {} GB".format(s.get_ngb()))

start_time = time.time()
last_time = start_time
count_time = 0

if use_cuda:
    s.copy_to_device()

time_max = 100 # 1000
time_output = 10
timing_ignore = 3  # ignore first 3 steps when averaging

energy_computation = energy.Energy(s)

for t in range(time_max):
    core.evolve_leapfrog(s)

    diff_time = time.time() - last_time
    total_time = time.time() - start_time
    count_time += 1
    average_time = total_time / count_time
    last_time = time.time()
    if t < timing_ignore:
        # reset timer
        count_time = 0
        start_time = last_time
    print("count: {}, average: {}, current: {}".format(t, average_time, diff_time))
    if (t + 1) % time_output == 0:
        energy_computation.compute()
        print("tau = {:4.2f} fm/c, energy density = {:4.2f} GeV/fm^3".format(s.t / E0 * hbarc,energy_computation.energy_density * E0 ** 4 / hbarc ** 3))
        tau = s.t * L / N

if use_cuda:
    s.copy_to_host()
