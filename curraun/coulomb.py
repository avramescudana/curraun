from curraun.numba_target import myjit, use_cuda, my_parallel_loop, my_cuda_sum, mycudajit
import numpy as np
import curraun.lattice as l
import curraun.su as su
from numpy.fft import rfft2, irfft2
if use_cuda:
    import numba.cuda as cuda

"""
    A module for performing the Coulomb gauge transformation on the lattice.
    This is done at a fixed \tau time step and at a certain iteration.
    The only external parameter is the \alpha convergence parameter.
    #TODO: perform the guage transformation iterations of GPU
"""

DEBUG = True
#TODO: copy all objects only in debug mode

class CoulombGaugeTransf:
    def __init__(self, s, alpha):
        self.s = s
        self.n = s.n
        nn = self.n ** 2

        # gauge transformation at previous iteration
        self.g0 = np.zeros((self.nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        #TODO: initialize with identity matrix

        # gauge transformation at current iteration
        self.g1 = np.zeros((self.nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # gauge links at previous iteration
        self.ug0 = np.zeros((self.nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # gauge links at current iteration
        self.ug1 = np.zeros((self.nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # divergence of the gauge field at previous iteration
        self.delta0 = np.zeros(self.nn, su.GROUP_ELEMENTS, dtype=np.double)

        # divergence of the gauge field at current iteration
        self.delta1 = np.zeros(self.nn, su.GROUP_ELEMENTS, dtype=np.double)

        # convergence criterion
        self.theta = 0.0

        # Memory on the CUDA device:
        self.d_g0 = self.g0
        self.d_g1 = self.g1
        self.d_ug0 = self.ug0
        self.d_ug1 = self.ug1
        self.d_delta0 = self.delta0
        self.d_delta1 = self.delta1
        self.d_theta = self.theta

    def copy_to_device(self):
        self.d_g0 = cuda.to_device(self.g0)
        self.d_g1 = cuda.to_device(self.g1)
        self.d_ug0 = cuda.to_device(self.ug0)
        self.d_ug1 = cuda.to_device(self.ug1)
        self.d_delta0 = cuda.to_device(self.delta0)
        self.d_delta1 = cuda.to_device(self.delta1)
        self.d_theta = cuda.to_device(self.theta)

    def copy_to_host(self):
        self.d_g0.copy_to_host(self.g0)
        self.d_g1.copy_to_host(self.g1)
        self.d_ug0.copy_to_host(self.ug0)   
        self.d_ug1.copy_to_host(self.ug1)
        self.d_delta0.copy_to_host(self.delta0)
        self.d_delta1.copy_to_host(self.delta1)
        self.d_theta.copy_to_host(self.theta)

    def copy_theta_to_device(self, stream=None):
        self.d_theta = cuda.to_device(self.p_theta, stream)

    def copy_theta_to_host(self, stream=None):
        self.d_theta.copy_to_host(self.theta, stream)

    def compute(self):
            # compute delta with previous iteration
            compute_delta(self.s, self.d_g0, self.d_ug0, self.d_delta0)

def compute_delta(s, g0, delta0):
    u0 = s.d_u0
    u1 = s.d_u1

    n = s.n

    my_parallel_loop(compute_delta_kernel, n * n, n, u0, ug0, g0, delta0)

@myjit
def compute_delta_kernel(xi, n, u0, ug0, g0, delta0):
    # Delta = \sum_i [(U_x-i,i - U_x,i) - hc - trace] with i=x,y 

    delta0 = su.zero()

    for d in range(2):
        xiplus = l.shift(xi, d, +1, n)
        ug0[xi, d] = l.dact(g0[xi], g0[xiplus], u0[xi, d])

        ximinus = l.shift(xi, d, -1, n)
        temp1 = su.add_mul(ug0[ximinus, d], ug0[xi, d], -1)
        temp2 = su.hc(temp1)
        temp3 = su.add_mul(temp1, temp2, -1)
        temp4 = su.trace(temp3)
        temp5 = su.add_mul(bf, temp4, -1)

        su.add(delta0, temp5)