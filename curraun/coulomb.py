from curraun.numba_target import myjit, use_cuda, my_parallel_loop, my_cuda_sum, mycudajit
import numpy as np
import math
import curraun.lattice as l
import curraun.su as su
from numpy.fft import rfft2, irfft2
from scipy.stats import unitary_group
# if use_cuda:
#     import numba.cuda as cuda

"""
    A module for performing the Coulomb gauge transformation on the lattice.
    This is done at a fixed \tau time step and at a certain iteration.
    The only external parameter is the \alpha convergence parameter.
    #TODO: perform the guage transformation iterations on GPU
    #TODO: iterate until convergence is reached
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
    def __init__(self, s):
        self.s = s
        self.n = s.n
        nn = self.n ** 2

        # gauge transformation at previous iteration
        self.g0 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        # gauge transformation at current iteration
        self.g1 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # gauge links at previous iteration
        self.ug0 = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        # gauge links at current iteration
        self.ug1 = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        #TODO: move this to separate debug function
        # check unitarity of the gauge transformation
        self.gunit = np.zeros(nn, dtype=su.GROUP_TYPE_REAL)
        # check unitarity of the gauge links
        self.ugunit = np.zeros((nn, 2), dtype=su.GROUP_TYPE_REAL)

        # divergence of the gauge field at previous iteration
        self.delta = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        # foruier accelerated coulomb gauge condition
        self.c = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # convergence criterion
        self.thetax = np.zeros(nn, dtype=su.GROUP_TYPE)
        self.theta = 0.0

        # Memory on the CUDA device:
        self.d_g0 = self.g0
        self.d_g1 = self.g1
        self.d_ug0 = self.ug0
        self.d_ug1 = self.ug1
        self.d_delta = self.delta
        self.d_theta = self.theta
        self.d_c = self.c
        self.d_thetax = self.thetax

        #TODO: move this to separate debug function
        self.d_ugunit = self.ugunit
        self.d_gunit = self.gunit

    def copy_to_device(self):
        self.d_g0 = cuda.to_device(self.g0)
        self.d_g1 = cuda.to_device(self.g1)
        self.d_ug0 = cuda.to_device(self.ug0)
        self.d_ug1 = cuda.to_device(self.ug1)
        self.d_delta = cuda.to_device(self.delta)
        self.d_theta = cuda.to_device(self.theta)
        self.d_c = cuda.to_device(self.c)
        self.d_thetax = cuda.to_device(self.thetax)

        #TODO: move this to separate debug function
        self.d_ugunit = cuda.to_device(self.ugunit)
        self.d_gunit = cuda.to_device(self.gunit)

    def copy_to_host(self):
        self.d_g0.copy_to_host(self.g0)
        self.d_g1.copy_to_host(self.g1)
        self.d_ug0.copy_to_host(self.ug0)   
        self.d_ug1.copy_to_host(self.ug1)
        self.d_delta.copy_to_host(self.delta)
        self.d_theta.copy_to_host(self.theta)
        self.d_c.copy_to_host(self.c)
        self.d_thetax.copy_to_host(self.thetax)

        #TODO: move this to separate debug function
        self.d_ugunit.copy_to_host(self.ugunit)
        self.d_gunit.copy_to_host(self.gunit)

    def copy_theta_to_device(self, stream=None):
        self.d_theta = cuda.to_device(self.p_theta, stream)

    def copy_theta_to_host(self, stream=None):
        self.d_theta.copy_to_host(self.theta, stream)

def init_gauge_transform(self):
    n = self.s.n
    nn = n ** 2

    # initialize gauge transformation with identity matrix
    init_transf(self.s, self.d_g0)
   
    # initialize gauge links with the glasma ones
    my_parallel_loop(init_gauge_links_kernel, nn, self.s.d_u0, self.d_ug0)

def init_transf(s, g0):
    n = s.n
    
    my_parallel_loop(init_transf_kernel, n * n, g0)

@myjit
def init_transf_kernel(xi, g0):
    g0[xi] = su.unit()

@myjit
def init_gauge_links_kernel(xi, u0, ug0):
    for d in range(2):
        su.store(ug0[xi, d], u0[xi, d])

def iterate(self):
    # iterate pointers to CUDA device memory
    self.d_g0, self.d_g1 = self.d_g1, self.d_g0
    self.d_ug0, self.d_ug1 = self.d_ug1, self.d_ug0
    
    self.g0, self.g1 = self.g1, self.g0
    self.ug0, self.ug1 = self.ug1, self.ug0

def init_transf_random(s, g0):
    n = s.n
    
    for xi in range(n*n):
        g0[xi] = unitary_group.rvs(su.NC).reshape(su.NC*su.NC)

def gauge_transform(self, alpha):
    # apply the gauge transformation to the gauge links
    gauge_transf_links(self.s, self.d_g0, self.d_ug0, self.d_ug1)

    # check unitarity of the gauge links
    #TODO: separate the unitarity check as debugging
    # check_ugunit(self.s, self.d_ugunit, self.d_ug1)

    # compute delta with previous iteration
    compute_delta(self.s, self.d_ug1, self.d_delta)

    compute_thetax(self.s, self.d_delta, self.d_thetax)
    self.d_theta = cupy.mean(cupy.array(self.d_thetax))

    self.theta = np.mean(self.d_thetax).real

    # apply fourier acceleration
    fourier_acceleration(self.s, self.d_delta, self.d_c)

    # update the gauge transformation
    update_gauge_transf(self.s, self.d_g0, self.d_c, self.d_g1, alpha)

    # check unitarity of the gauge transformation
    #TODO: separate the unitarity check as debugging
    # check_gunit(self.s, self.d_gunit, self.d_g1)

    #TODO: apply gauge transformation to aeta, peta and pi

    iterate(self)

def check_ugunit(s, ugunit, ug):
    n = s.n
    nn = n ** 2

    my_parallel_loop(check_ugunit_kernel, nn, ugunit, ug)

@myjit  
def check_ugunit_kernel(xi, ugunit, ug):
    for d in range(2):
        buf = su.mul(su.dagger(ug[xi, d]), ug[xi, d])
        ugunit[xi, d] = (su.sq(l.add_mul(su.unit(), buf, -1)))

def check_gunit(s, gunit, g):
    n = s.n
    nn = n ** 2

    my_parallel_loop(check_gunit_kernel, nn, gunit, g)

@myjit  
def check_gunit_kernel(xi, gunit, g):
    buf = su.mul(su.dagger(g[xi]), g[xi])
    gunit[xi] = (su.sq(l.add_mul(su.unit(), buf, -1)))

def gauge_transf_links(s, g1, ug0, ug1):
    n = s.n
    nn = n ** 2

    my_parallel_loop(gauge_transf_links_kernel, nn, n, g1, ug0, ug1)

@myjit
def gauge_transf_links_kernel(xi, n, g1, ug0, ug1):
    for d in range(2):
        xiplus = l.shift(xi, d, +1, n)
        ug1[xi, d] = l.dact(g1[xi], g1[xiplus], ug0[xi, d])

def update_gauge_transf(s, g0, c, g1, alpha):
    n = s.n
    nn = n ** 2

    my_parallel_loop(update_gauge_transf_kernel, nn, n, g0, c, g1, alpha)

@myjit
def update_gauge_transf_kernel(xi, n, g0, c, g1, alpha):
    buf = su.mul_s(su.mexp(c[xi]), alpha)
    g1[xi] = su.mul(buf, g0[xi])

def compute_thetax(s, delta, thetax):
    n = s.n
    nn = n ** 2 

    my_parallel_loop(compute_thetax_kernel, nn, delta, thetax)

@myjit
def compute_thetax_kernel(xi, delta, thetax):
    thetax[xi] = su.tr(su.mul(delta[xi], su.dagger(delta[xi])))

def compute_delta(s, ug0, delta):
    n = s.n

    my_parallel_loop(compute_delta_kernel, n * n, n, ug0, delta)

@myjit
def compute_delta_kernel(xi, n, ug0, delta):
    # Delta = \sum_i [(U_x-i,i - U_x,i) - hc - trace] with i=x,y 
    buf = su.zero()

    for d in range(2):
        ximinus = l.shift(xi, d, -1, n)
        temp1 = l.add_mul(ug0[xi, d], ug0[ximinus, d], -1)
        temp2 = su.dagger(temp1)
        temp3 = l.add_mul(temp1, temp2, -1)
        temp4 = su.mul_s(su.unit(), su.tr(temp3)/su.NC)
        temp5 = l.add_mul(temp3, temp4, -1)
        buf = su.add(buf, temp5)

    su.store(delta[xi], buf)

def fourier_acceleration(s, delta, c):
    n = s.n

    # fourier transform to momentum space
    delta_reshape = cupy.reshape(cupy.array(delta), (n, n, su.GROUP_ELEMENTS))
    delta_fft = cupy.fft.fft2(delta_reshape, axes=(0, 1))

    # fourier accelerate with alpha
    delta_fft_reshape = cupy.reshape(delta_fft, (n*n, su.GROUP_ELEMENTS))

    my_parallel_loop(complex_fourier_acceleration_kernel, n*n , n, cupy.asnumpy(delta_fft_reshape))

    # inverse fourier transform to position space
    delta_accfft_reshape = cupy.reshape(cupy.array(delta_fft_reshape), (n, n, su.GROUP_ELEMENTS))

    delta_accfft = cupy.fft.ifft2(delta_accfft_reshape, axes=(0, 1), s=(n, n))
    c_fft = cupy.asnumpy(cupy.reshape(delta_accfft, (n * n, su.GROUP_ELEMENTS)))
    my_parallel_loop(store_c_fft_kernel, n*n , c_fft, c)

@myjit  
def complex_fourier_acceleration_kernel(xi, n, delta_fft):
    x, y = l.get_point(xi, n)

    # extract p^2 and p^2_max
    psq = psq_latt(x, y, n)
    psqmax = psq_latt(n-1, n-1, n)

    buf = su.mul_s(delta_fft[xi], psqmax / psq)
    su.store(delta_fft[xi], buf)

@myjit
def psq_latt(x, y, n):
    result = 4.0 * (math.sin((np.pi * x) / n) ** 2 + math.sin((np.pi * y) / n) ** 2)
    return result

@myjit
def store_c_fft_kernel(xi, c_fft, c):
    su.store(c[xi], c_fft[xi])