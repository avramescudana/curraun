from curraun.numba_target import myjit, use_cuda, my_parallel_loop, my_cuda_sum, mycudajit
import numpy as np
import math
import curraun.lattice as l
import curraun.su as su
from numpy.fft import rfft2, irfft2
# if use_cuda:
#     import numba.cuda as cuda

"""
    A module for performing the Coulomb gauge transformation on the lattice.
    This is done at a fixed \tau time step and at a certain iteration.
    The only external parameter is the \alpha convergence parameter.
    #TODO: perform the guage transformation iterations of GPU
"""

# Use cupy only if cuda is available
# cupy can be turned off by changing 'use_cupy'
if use_cuda:
    use_cupy = True
    import numba.cuda as cuda
    import cupy
else:
    use_cupy = False

DEBUG = True
#TODO: copy all objects only in debug mode

class CoulombGaugeTransf:
    def __init__(self, s, alpha):
        self.s = s
        self.n = s.n
        nn = self.n ** 2

        # gauge transformation at previous iteration
        self.g0 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        #TODO: initialize with identity matrix

        # gauge transformation at current iteration
        self.g1 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # gauge links at previous iteration
        self.ug0 = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # gauge links at current iteration
        self.ug1 = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # divergence of the gauge field at previous iteration
        self.delta0 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # divergence of the gauge field at current iteration
        self.delta1 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # foruier accelerated coulomb gauge condition
        self.c = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

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
        self.d_c = self.c

    def copy_to_device(self):
        self.d_g0 = cuda.to_device(self.g0)
        self.d_g1 = cuda.to_device(self.g1)
        self.d_ug0 = cuda.to_device(self.ug0)
        self.d_ug1 = cuda.to_device(self.ug1)
        self.d_delta0 = cuda.to_device(self.delta0)
        self.d_delta1 = cuda.to_device(self.delta1)
        self.d_theta = cuda.to_device(self.theta)
        self.d_c = cuda.to_device(self.c)

    def copy_to_host(self):
        self.d_g0.copy_to_host(self.g0)
        self.d_g1.copy_to_host(self.g1)
        self.d_ug0.copy_to_host(self.ug0)   
        self.d_ug1.copy_to_host(self.ug1)
        self.d_delta0.copy_to_host(self.delta0)
        self.d_delta1.copy_to_host(self.delta1)
        self.d_theta.copy_to_host(self.theta)
        self.d_c.copy_to_host(self.c)

    def copy_theta_to_device(self, stream=None):
        self.d_theta = cuda.to_device(self.p_theta, stream)

    def copy_theta_to_host(self, stream=None):
        self.d_theta.copy_to_host(self.theta, stream)

def gauge_transform(self, alpha):
        # initialize gauge transformation with identity matrix
        #TODO: separate this into a function, gauge_transform will be called iteratively
        init_transf(self.s, self.d_g0)

        # compute delta with previous iteration
        compute_delta(self.s, self.d_g0, self.d_ug0, self.d_delta0)

        # apply fourier acceleration to delta
        fourier_acceleration(self.s, self.d_delta0, alpha, self.d_c)

def init_transf(s, g0):
    n = s.n
    
    my_parallel_loop(init_transf_kernel, n * n, n, g0)

@myjit
def init_transf_kernel(xi, n, g0):
    g0[xi] = su.unit()

def compute_delta(s, g0, ug0, delta0):
    u0 = s.d_u0
    # u1 = s.d_u1

    n = s.n

    my_parallel_loop(compute_delta_kernel, n * n, n, u0, ug0, g0, delta0)

@myjit
def compute_delta_kernel(xi, n, u0, ug0, g0, delta0):
    # Delta = \sum_i [(U_x-i,i - U_x,i) - hc - trace] with i=x,y 

    buf = su.zero()

    for d in range(2):
        xiplus = l.shift(xi, d, +1, n)
        ug0[xi, d] = l.dact(g0[xi], g0[xiplus], u0[xi, d])

        ximinus = l.shift(xi, d, -1, n)
        temp1 = l.add_mul(ug0[ximinus, d], ug0[xi, d], -1)
        temp2 = su.dagger(temp1)
        temp3 = l.add_mul(temp1, temp2, -1)
        temp4 = su.mul_s(su.unit(), su.tr(temp3)/su.NC)

        temp5 = l.add_mul(temp3, temp4, -1)
        buf = su.add(buf, temp5)

    su.store(delta0[xi], buf)


def fourier_acceleration(s, delta0, alpha, c):
    n = s.n
    n_fft = (n // 2 + 1) if n % 2 == 0 else (n + 1) // 2

    # fourier transform to momentum space
    delta0_reshape = cupy.reshape(cupy.array(delta0), (n, n, su.GROUP_ELEMENTS))
    # delta0_fft = cupy.fft.rfft2(delta0_reshape, axes=(0, 1))
    delta0_fft = cupy.fft.fft2(delta0_reshape, axes=(0, 1))

    # fourier accelerate with alpha
    # delta0_fft_reshape = cupy.reshape(delta0_fft, (n*n_fft, su.GROUP_ELEMENTS))
    delta0_fft_reshape = cupy.reshape(delta0_fft, (n*n, su.GROUP_ELEMENTS))

    # my_parallel_loop(fourier_acceleration_kernel, n , n, n_fft, cupy.asnumpy(delta0_fft_reshape), alpha)
    my_parallel_loop(complex_fourier_acceleration_kernel, n*n , n, cupy.asnumpy(delta0_fft_reshape), alpha)

    # inverse fourier transform to position space
    # delta0_accfft_reshape = cupy.reshape(cupy.array(delta0_fft_reshape), (n, n_fft, su.GROUP_ELEMENTS))
    delta0_accfft_reshape = cupy.reshape(cupy.array(delta0_fft_reshape), (n, n, su.GROUP_ELEMENTS))

    # delta0_accfft = cupy.fft.irfft2(delta0_accfft_reshape, axes=(0, 1), s=(n, n))
    delta0_accfft = cupy.fft.ifft2(delta0_accfft_reshape, axes=(0, 1), s=(n, n))
    
    c_fft = cupy.asnumpy(cupy.reshape(delta0_accfft, (n * n, su.GROUP_ELEMENTS)))

    my_parallel_loop(store_c_fft_kernel, n*n , c_fft, c)

@myjit  
def fourier_acceleration_kernel(x, n, n_fft, delta0_fft, alpha):
    for y in range(n_fft):
        if (x > 0 or y > 0):
            # extract p^2 and p^2_max
            psq = psq_latt(x, y, n)
            psqmax = (2*np.pi/n)**2    

            buf = su.mul_s(delta0_fft[x], alpha * psqmax / psq)
            su.store(delta0_fft[x], buf)

@myjit  
def complex_fourier_acceleration_kernel(xi, n, delta0_fft, alpha):
    x, y = l.get_point(xi, n)

    # extract p^2 and p^2_max
    psq = psq_latt(x, y, n)
    psqmax = (2*np.pi/n)**2    

    buf = su.mul_s(delta0_fft[xi], alpha * psqmax / psq)
    su.store(delta0_fft[xi], buf)

@myjit
def psq_latt(x, y, n):
    result = 4.0 * (math.sin((np.pi * x) / n) ** 2 + math.sin((np.pi * y) / n) ** 2)
    return result

@myjit
def store_c_fft_kernel(xi, c_fft, c):
    su.store(c[xi], c_fft[xi])