import os
import time
import datetime
import importlib

from curraun.numba_target import use_cuda, use_numba
if use_cuda:
    from numba import cuda
import curraun
import curraun.core as core
import curraun.su2
import curraun.su3
import numpy as np
import numba

MAX_TIMEOUT = 10

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
DT = 1.0 / p['DTS']
maxt = int(p['TMAX'] / a) * p['DTS']

def time_simulation():
    # Reload libraries
    importlib.reload(curraun.numba_target)
    importlib.reload(curraun)
    importlib.reload(curraun.core)
    importlib.reload(curraun.su2)
    importlib.reload(curraun.su3)
    importlib.reload(curraun.su)
    importlib.reload(curraun.leapfrog)
    importlib.reload(curraun.lattice)

    from curraun.numba_target import use_cuda, use_numba
    if use_cuda:
        from numba import cuda

    np.random.seed(1)

    # initialization
    s = curraun.core.Simulation(p['N'], DT, p['G'])

    print("Memory of data: {} GB".format(s.get_ngb()))

    if use_cuda:
        s.copy_to_device()

        meminfo = cuda.current_context().get_memory_info()
        print("CUDA free memory: {:.2f} GB of {:.2f} GB.".format(meminfo[0] / 1024 ** 3, meminfo[1] / 1024 ** 3))

    if not use_numba:
        # Evolve once for Just-In-Time compilation
        curraun.core.evolve_leapfrog(s)

    init_time = time.time()

    for t in range(maxt):
        curraun.core.evolve_leapfrog(s)
        if use_cuda:
            cuda.synchronize()
        total_time = time.time() - init_time
        if total_time > MAX_TIMEOUT:
            break

    if use_cuda:
        # s.copy_to_host()
        cuda.synchronize()

    total_time = time.time() - init_time
    number_of_steps = t + 1
    seconds_per_step = total_time / number_of_steps
    estimated_total_time = seconds_per_step * maxt * p['NE']
    estimated_total_time_formatted = str(datetime.timedelta(seconds=estimated_total_time))

    print("Number of steps calculated: {}".format(number_of_steps))
    print("Total time: {}".format(total_time))
    print("Seconds per simulation step: {}".format(seconds_per_step))
    print("Estimated total time: {}".format(estimated_total_time_formatted))
    print("---------------------------------------")

def time_python_numba_cuda():
    #os.environ["MY_NUMBA_TARGET"] = "python"
    #time_simulation()

    print("Number of threads: 1")
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["MY_NUMBA_TARGET"] = "numba"
    time_simulation()

    del os.environ["NUMBA_NUM_THREADS"]
    print("Number of threads: {}".format(numba.config.NUMBA_DEFAULT_NUM_THREADS))
    os.environ["MY_NUMBA_TARGET"] = "numba"
    time_simulation()

    os.environ["MY_NUMBA_TARGET"] = "cuda"
    time_simulation()

print("---------------------------------------")

# SU(2) Double
os.environ["GAUGE_GROUP"] = "su2"
os.environ["PRECISION"] = "double"
time_python_numba_cuda()

# SU(2) Single
os.environ["PRECISION"] = "single"
time_python_numba_cuda()

# SU(3) Double
os.environ["GAUGE_GROUP"] = "su3"
os.environ["PRECISION"] = "double"
time_python_numba_cuda()

# SU(3) Single
os.environ["PRECISION"] = "single"
time_python_numba_cuda()
