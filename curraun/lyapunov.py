from curraun.numba_target import myjit, my_parallel_loop, use_cuda, mynonparjit
import numpy as np
import curraun.lattice as l
import curraun.su as su
import os
random_np = np.random.RandomState()

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

        n = self.s.n
        self.tr_sq_el = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.tr_sq_dif = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.d_tr_sq_el = self.tr_sq_el
        self.d_tr_sq_dif = self.tr_sq_dif

        self.ratio_dif = 0.0

        # if use_cuda:
        #     self.copy_to_device()

    # def copy_to_device(self):

    # def copy_to_host(self):

    def change_el(self, alpha):
        peta1 = self.sprime.d_peta1

        n = self.sprime.n

        eta = random_np.normal(loc=0.0, scale=alpha, size=(n ** 2 * su.GROUP_ELEMENTS))
        eta = eta.reshape((n * n, su.GROUP_ELEMENTS))
        my_parallel_loop(change_el_kernel, n ** 2, peta1, eta)

    def compute_change_el(self):
        peta1s = self.s.d_peta1
        peta1sprime = self.sprime.d_peta1

        n = self.s.n

        my_parallel_loop(compute_change_el_kernel, n ** 2, peta1s, peta1sprime, self.d_tr_sq_el, self.d_tr_sq_dif)

        dif_avg = np.mean(self.d_tr_sq_dif)
        el_avg = np.mean(self.d_tr_sq_el)

        # self.ratio_dif = dif_avg / el_avg
        self.ratio_dif = dif_avg


        # if use_cuda:
        #     self.copy_to_host()

#TODO: Add Gaussian noise with parameter alpha
@mynonparjit
def change_el_kernel(xi, peta1, eta):
    # buf1 = su.mul_s(peta1[xi], 1.1)
    buf1 = su.add(peta1[xi], eta[xi])
    # eta = random_np.normal(loc=0.0, scale=alpha, size=(su.GROUP_ELEMENTS))
    # buf1 = su.add(peta1[xi], eta)
    peta1[xi] = buf1

    #field = random_np.normal(loc=0.0, scale=g ** 2 * mu / math.sqrt(num_sheets), size=(n ** 2 * su.ALGEBRA_ELEMENTS))
     #       field = field.reshape((n * n, su.ALGEBRA_ELEMENTS))

    # alpha = 0.0001
    # pert =  random_np.normal(loc=0.0, scale=alpha,  size=())

@mynonparjit
def compute_change_el_kernel(xi, peta1s, peta1sprime, tr_sq_el, tr_sq_dif):
    buf1 = l.add_mul(peta1sprime[xi], peta1s[xi], -1)
    tr_sq_dif[xi] = su.sq(buf1) 

    buf2 = peta1s[xi]
    tr_sq_el[xi] = su.sq(buf2)

