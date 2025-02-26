from curraun.numba_target import myjit, my_parallel_loop, use_cuda, mynonparjit
import numpy as np
import curraun.lattice as l
import curraun.su as su
import os

if use_cuda:
    import numba.cuda as cuda

"""
    A module for computing energy density components
"""

# set precision of variable
su_precision = os.environ.get('PRECISION', 'double')
if su_precision == 'single':
    DTYPE = np.float32
elif su_precision == 'double':
    DTYPE = np.float64


class Lyapunov():
    def __init__(self, s, sprime):
        self.s = s
        self.sprime = sprime

        # if use_cuda:
        #     self.copy_to_device()

    # def copy_to_device(self):

    # def copy_to_host(self):

    def change_el(self, alpha):
        peta1 = self.sprime.d_peta1

        n = self.sprime.n

        my_parallel_loop(change_el_kernel, n ** 2, peta1, alpha)

        # if use_cuda:
        #     self.copy_to_host()

#TODO: Add Gaussian noise with parameter alpha
@mynonparjit
def change_el_kernel(xi, peta1, alpha):
    buf1 = su.mul_s(peta1[xi], 1.1)
    peta1[xi] = buf1
