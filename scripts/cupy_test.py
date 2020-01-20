"""
    Just a simple test script for the new cupy-capable MV module.
"""

import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["GAUGE_GROUP"] = "su2"
os.environ["PRECISION"] = "double"
os.environ["CUDA_PATH"] = "/usr/local/cuda-9.0"

import curraun.core as core
import curraun.mv as mv
from time import time

# simulation parameters
L = 6.0
M = 0.5
MU = 0.5
G = 2.0
N = 512
DT = 0.5
UV = 10.0
NUMS = 50
E0 = N / L * 0.197326

# Numpy/Numba test
mv.use_cupy = False
mv.set_seed(1)

tt1 = time()
s = core.Simulation(N, DT, G)
for i in range(3):
    va = mv.wilson(s, mu=MU / E0, m=M / E0, uv=UV / E0, num_sheets=NUMS)
tt2 = time()

print("Numpy/Numba time: {:3.2f} s".format(tt2 - tt1))

# CuPy test
mv.use_cupy = True
mv.set_seed(1)

tt1 = time()
s = core.Simulation(N, DT, G)
for i in range(3):
    va = mv.wilson(s, mu=MU / E0, m=M / E0, uv=UV / E0, num_sheets=NUMS)
tt2 = time()

print("CuPy time: {:3.2f} s".format(tt2 - tt1))
