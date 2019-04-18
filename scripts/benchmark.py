import os
import time
import datetime
import importlib
import platform

from curraun.numba_target import use_cuda, use_numba
if use_cuda:
    from numba import cuda
import curraun
import curraun.core as core
import curraun.su2
import curraun.su3
import numpy as np
import numba

FILENAME = "benchmark.dat"
MAX_TIMEOUT = 10
SU3_EXP_MIN_TERMS = 20   # Simulate more work (since our simulation only contains zero)

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

speedup_base = -1

def time_simulation(target_string, spec_string):
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

    curraun.su3.EXP_MIN_TERMS = SU3_EXP_MIN_TERMS

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

    global speedup_base
    if speedup_base == -1:
        speedup_base = seconds_per_step
    speedup = speedup_base / seconds_per_step
    print("Target: {}".format(target_string))
    print("Specification: {}".format(spec_string))
    print("Number of steps calculated: {}".format(number_of_steps))
    print("Total time: {:12.6f}".format(total_time))
    print("Seconds per simulation step: {:12.6f}".format(seconds_per_step))
    print("Speedup: {:8.3f}".format(speedup))
    print("Estimated total time: {}".format(estimated_total_time_formatted))
    print("---------------------------------------")

    with open(FILENAME, "a") as myfile:
        myfile.write("{0: <14},".format(target_string))
        myfile.write("{},".format(spec_string))
        myfile.write("{:12.6f},".format(seconds_per_step))
        myfile.write("{:8.3f},".format(speedup))
        myfile.write('"{}"\n'.format(estimated_total_time_formatted))

def time_python_numba_cuda(spec_string):
    #os.environ["MY_NUMBA_TARGET"] = "python"
    #time_simulation("Python")

    print("Number of threads: 1")
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["MY_NUMBA_TARGET"] = "numba"
    time_simulation("Numba 1 CPU", spec_string)

    del os.environ["NUMBA_NUM_THREADS"]
    print("Number of threads: {}".format(numba.config.NUMBA_DEFAULT_NUM_THREADS))
    os.environ["MY_NUMBA_TARGET"] = "numba"
    time_simulation("Numba {} CPUs".format(numba.config.NUMBA_DEFAULT_NUM_THREADS), spec_string)

    global cuda_supported
    if cuda_supported:
        os.environ["MY_NUMBA_TARGET"] = "cuda"
        time_simulation("CUDA", spec_string)

current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
hostname = platform.node()
import numba.cuda as cuda
cuda_supported = True
try:
    gpu_name = cuda.get_current_device().name.decode('UTF-8')
    compute_capability = cuda.get_current_device().compute_capability
    compute_capability_string = str(compute_capability[0]) + "." + str(compute_capability[1])
except:
    cuda_supported = False


print("---------------------------------------")
print("Date: {}".format(current_date))
print("Hostname: {}".format(hostname))
if cuda_supported:
    print("GPU Name: {}".format(gpu_name))
    print("Compute capability: {}".format(compute_capability_string))
else:
    print("No CUDA capable graphics card found")
print("---------------------------------------")

with open(FILENAME, "a") as myfile:
    myfile.write("---------------------------------------\n")
    myfile.write("Date: {}\n".format(current_date))
    myfile.write("Hostname: {}\n".format(hostname))
    if cuda_supported:
        myfile.write("GPU Name: {}\n".format(gpu_name))
        myfile.write("Compute capability: {}\n".format(compute_capability_string))
    else:
        myfile.write("No CUDA capable graphics card found\n")
    myfile.write("---------------------------------------\n")

# SU(2) Double
os.environ["GAUGE_GROUP"] = "su2"
os.environ["PRECISION"] = "double"
time_python_numba_cuda("su2-double")

# SU(2) Single
os.environ["PRECISION"] = "single"
time_python_numba_cuda("su2-single")

# SU(3) Double
os.environ["GAUGE_GROUP"] = "su3"
os.environ["PRECISION"] = "double"
time_python_numba_cuda("su3-double")

# SU(3) Single
os.environ["PRECISION"] = "single"
time_python_numba_cuda("su3-single")

# # For debugging types:
# # Numba SU(3) Single
# os.environ["MY_NUMBA_TARGET"] = "numba"
# os.environ["GAUGE_GROUP"] = "su3"
# os.environ["PRECISION"] = "single"
# time_simulation("NUMBA","su3-single")
#
# # For debugging types:
# import curraun.su as su
# if use_numba and (su.GROUP_TYPE == np.float32 or su.GROUP_TYPE == np.complex64):  # TODO: Remove debugging code
#     print("Debugging single precision types:")
#     su.store.inspect_types()
#     su.mexp.inspect_types()
#     su.get_algebra_element.inspect_types()
#     su.mul.inspect_types()
#     su.ah.inspect_types()
#     su.sq.inspect_types()