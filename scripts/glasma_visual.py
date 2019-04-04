#######################################
# ### DISABLE NUMBA JIT
import os
#os.environ["NUMBA_DISABLE_JIT"] = "1"
#os.environ["NUMBA_NUM_THREADS"] = "1"
#os.environ["NUMBA-WARNINGS"] = "1"
#os.environ["NUMBA_DEBUG_ARRAY_OPT_STATS"] = "1"
#os.environ["NUMBA_OPT"] = "0"  # LLVM optimziation level. Default = 3
#os.environ["NUMBA_DUMP_OPTIMIZED"] = "1"
#######################################

#import curraun
import time

import matplotlib.pyplot as plt
import numpy as np
import curraun.core as core

import curraun.initial as initial
import curraun.leapfrog as leapfrog
import curraun.mv as mv

# simulation parameters
L = 6.0
M = 0.0
MU = 0.5
G = 2.0
N = 256 # 64 # 32 # 256
DT = 0.5
UV = 10.0
NUMS = 1

# initialization
E0 = N / L * 0.197326
s = core.Simulation(N, DT, G, leapfrog)
va = mv.wilson(s, mu=MU / E0, m=M / E0, uv=UV / E0, num_sheets=NUMS)
vb = mv.wilson(s, mu=MU / E0, m=M / E0, uv=UV / E0, num_sheets=NUMS)
initial.init(s, va, vb)

# evolution and visualization
plt.ion()

# first step to initialize view
core.evolve_leapfrog(s)
el, bl, et, bt = leapfrog.fields2d(s)
E = np.max(el + bl + et + bt)
fig, axes = plt.subplots(ncols=3, nrows=2)
el_v = axes[0, 0].imshow(el / E, vmin=0.0, vmax=0.5, interpolation='none', cmap=plt.get_cmap('inferno'))
bl_v = axes[0, 1].imshow(bl / E, vmin=0.0, vmax=0.5, interpolation='none', cmap=plt.get_cmap('inferno'))
et_v = axes[1, 0].imshow(et / E, vmin=0.0, vmax=0.5, interpolation='none', cmap=plt.get_cmap('inferno'))
bt_v = axes[1, 1].imshow(bt / E, vmin=0.0, vmax=0.5, interpolation='none', cmap=plt.get_cmap('inferno'))

start_time = time.time()
last_time = start_time
count_time = 0

for t in range(1000):
    core.evolve_leapfrog(s)

    diff_time = time.time() - last_time
    total_time = time.time() - start_time
    count_time += 1
    average_time = total_time / count_time
    last_time = time.time()
    print("count: {}, average: {}, current: {}".format(count_time, average_time, diff_time))

    el, bl, et, bt = leapfrog.fields2d(s)
    E = np.max(el + bl + et + bt) / 2.0

    el_v.set_data(el / E)
    bl_v.set_data(bl / E)
    et_v.set_data(et / E)
    bt_v.set_data(bt / E)

    el_s = np.sum(el) * E0
    bl_s = np.sum(bl) * E0
    et_s = np.sum(et) * E0
    bt_s = np.sum(bt) * E0

    e_s = (el_s + bl_s + et_s + bt_s)

    tau = s.t * L / N

    f = 1.0
    axes[0, 2].scatter(tau, el_s * f, c='r')
    axes[0, 2].scatter(tau, bl_s * f, c='g')
    axes[0, 2].scatter(tau, et_s * f, c='b')
    axes[0, 2].scatter(tau, bt_s * f, c='m')

    f = 1.0 / (L ** 2 * tau)
    axes[1, 2].scatter(tau, e_s * f, c='r')
    # axes[1, 2].scatter(tau, e_s, c='r')

    plt.pause(0.001)
