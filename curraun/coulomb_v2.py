from curraun.numba_target import myjit, use_cuda, my_parallel_loop, my_cuda_sum, mycudajit
import numpy as np
import math
import curraun.lattice as l
import curraun.su as su
from numpy.fft import rfft2, irfft2

"""
    A module for performing the Coulomb gauge transformation on the lattice.
    This is done at a fixed \tau time step and at a certain iteration.
    The only external parameter is the \alpha convergence parameter.
    #TODO: perform the guage transformation iterations on GPU
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

        # gauge transformation 
        self.g = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # gauge links
        self.ug = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # convergence criterion
        self.thetax = np.zeros(nn, dtype=su.GROUP_TYPE)
        self.theta = 0.0

        # Memory on the CUDA device:
        self.d_g = self.g
        self.d_ug = self.ug
        self.d_thetax = self.thetax

    def copy_to_device(self):
        self.d_g = cuda.to_device(self.g)
        self.d_ug = cuda.to_device(self.ug)
        self.d_thetax = cuda.to_device(self.thetax)
        self.d_theta = cuda.to_device(self.theta)

    def copy_to_host(self):
        self.d_g.copy_to_host(self.g)
        self.d_ug.copy_to_host(self.ug)   
        self.d_thetax.copy_to_host(self.thetax)
        self.d_theta.copy_to_host(self.theta)

def init_gauge_transform(self):
    n = self.s.n
    nn = n ** 2

    my_parallel_loop(init_transf_kernel, nn, self.d_g, self.s.d_u0, self.d_ug)

    # initialize gauge links with the glasma ones
    # my_parallel_loop(init_gauge_links_kernel, nn, self.s.d_u0, self.d_ug)

@myjit
def init_transf_kernel(xi, g, u0, ug):
    # initialize gauge transformation with identity matrix
    g[xi] = su.unit()

    # initialize gauge links with the glasma ones
    for d in range(2):
        su.store(ug[xi, d], u0[xi, d])

def gauge_transform(self):
    n = self.s.n
    nn = n ** 2

    my_parallel_loop(gauge_transform_kernel, nn, n, self.d_g, self.d_ug, self.d_thetax)
    self.theta = np.mean(self.d_thetax).real


@myjit
def gauge_transform_kernel(xi, n, g, ug, thetax):
    # gauge transform the gauge links

    # x component
    xi_xplus = l.shift(xi, 0, +1, n)
    ugx_xi = l.dact(g[xi], g[xi_xplus], ug[xi, 0])
    xi_xminus = l.shift(xi, 0, -1, n)
    ugx_xi_xminus = l.dact(g[xi_xminus], g[xi], ug[xi_xminus, 0])    

    # y component
    xi_yplus = l.shift(xi, 1, +1, n)
    ugy_xi = l.dact(g[xi], g[xi_yplus], ug[xi, 1])
    xi_yminus = l.shift(xi, 1, -1, n)
    ugy_xi_yminus = l.dact(g[xi_yminus], g[xi], ug[xi_yminus, 1])

    su.store(ug[xi, 0], ugx_xi)
    su.store(ug[xi, 1], ugy_xi)

    # compute delta 
    buf1x = l.add_mul(ugx_xi, ugx_xi_xminus, -1)
    buf2x = su.get_algebra_factors_from_group_element_approximate(buf1x)

    buf1y = l.add_mul(ugy_xi, ugy_xi_yminus, -1)
    buf2y = su.get_algebra_factors_from_group_element_approximate(buf1y)

    buf1 = su.add_algebra(buf2x, buf2y)
    delta = su.mul_algebra(buf1, -0.5)

    buf2 = su.get_algebra_element(delta)
    thetax[xi] = su.tr(su.mul(buf2, su.dagger(buf2)))

    # update gauge transformation
    g0 = g[xi]
    # buf3 = su.mexp(buf2)
    buf3 = su.mexp(su.mul_s(buf2, -1/(4*su.NC*n**2)))

    buf4 = su.mul(buf3, g0)
    su.store(g[xi], buf4)


def update_gauge_transf(s, g0, c, g1):
    n = s.n
    nn = n ** 2

    my_parallel_loop(update_gauge_transf_kernel, nn, g0, c, g1)

@myjit
def update_gauge_transf_kernel(xi, g0, c, g1):
    buf = su.mexp(c[xi])
    g1[xi] = su.mul(buf, g0[xi])

#TODO: algebra color elements debugging
def update_gauge_transf_alg(s, g0, c, g1):
    n = s.n
    nn = n ** 2

    my_parallel_loop(update_gauge_transf_alg_kernel, nn, g0, c, g1)

@myjit
def update_gauge_transf_alg_kernel(xi, g0, c, g1):
    buf1 = su.get_algebra_element(c[xi])
    buf2 = su.mexp(buf1)
    g1[xi] = su.mul(buf2, g0[xi])

def compute_thetax(s, delta, thetax):
    n = s.n
    nn = n ** 2 

    my_parallel_loop(compute_thetax_kernel, nn, delta, thetax)

@myjit
def compute_thetax_kernel(xi, delta, thetax):
    thetax[xi] = su.tr(su.mul(delta[xi], su.dagger(delta[xi])))

#TODO: algebra color elements debugging
def compute_thetax_alg(s, delta, thetax):
    n = s.n
    nn = n ** 2 

    my_parallel_loop(compute_thetax_alg_kernel, nn, delta, thetax)

@myjit
def compute_thetax_alg_kernel(xi, delta, thetax):
    buf = su.get_algebra_element(delta[xi])
    thetax[xi] = su.tr(su.mul(buf, su.dagger(buf)))


# def compute_delta(s, g0, ug0, delta):
def compute_delta(s, ug0, delta):
    # u0 = s.d_u0
    # u1 = s.d_u1

    n = s.n

    # my_parallel_loop(compute_delta_kernel, n * n, n, u0, ug0, g0, delta)
    my_parallel_loop(compute_delta_kernel, n * n, n, ug0, delta)

@myjit
# def compute_delta_kernel(xi, n, u0, ug0, g0, delta):
def compute_delta_kernel(xi, n, ug0, delta):
    # Delta = \sum_i [(U_x-i,i - U_x,i) - hc - trace] with i=x,y 

    buf = su.zero()

    for d in range(2):
        # xiplus = l.shift(xi, d, +1, n)
        # ug0[xi, d] = l.dact(g0[xi], g0[xiplus], u0[xi, d])

        ximinus = l.shift(xi, d, -1, n)
        # temp1 = l.add_mul(ug0[ximinus, d], ug0[xi, d], -1)
        temp1 = l.add_mul(ug0[xi, d], ug0[ximinus, d], -1)
        temp2 = su.dagger(temp1)
        temp3 = l.add_mul(temp1, temp2, -1)
        # temp4 = su.mul_s(su.unit(), su.tr(temp3)/su.NC)
        temp4 = su.mul_s(su.unit(), su.tr(temp3))

        temp5 = l.add_mul(temp3, temp4, -1)
        buf = su.add(buf, temp5)

    su.store(delta[xi], buf)

#TODO: algebra color elements debugging
def compute_delta_alg(s, ug0, delta):
    n = s.n

    my_parallel_loop(compute_delta_alg_kernel, n * n, n, ug0, delta)

#TODO: algebra color elements debugging
@myjit
def compute_delta_alg_kernel(xi, n, ug0, delta):
    ximinusx = l.shift(xi, 0, -1, n)
    buf1x = l.add_mul(ug0[xi, 0], ug0[ximinusx, 0], -1)
    buf2x = su.get_algebra_factors_from_group_element_approximate(buf1x)

    ximinusy = l.shift(xi, 1, -1, n)
    buf1y = l.add_mul(ug0[xi, 1], ug0[ximinusy, 1], -1)
    buf2y = su.get_algebra_factors_from_group_element_approximate(buf1y)

    buf = su.add_algebra(buf2x, buf2y)
    delta[xi] = su.mul_algebra(buf, -0.5)

def fourier_acceleration(s, delta, alpha, c):
    n = s.n
    # n_fft = (n // 2 + 1) if n % 2 == 0 else (n + 1) // 2

    # fourier transform to momentum space
    delta_reshape = cupy.reshape(cupy.array(delta), (n, n, su.GROUP_ELEMENTS))
    # delta_fft = cupy.fft.rfft2(delta_reshape, axes=(0, 1))
    delta_fft = cupy.fft.fft2(delta_reshape, axes=(0, 1))

    # fourier accelerate with alpha
    # delta_fft_reshape = cupy.reshape(delta_fft, (n*n_fft, su.GROUP_ELEMENTS))
    delta_fft_reshape = cupy.reshape(delta_fft, (n*n, su.GROUP_ELEMENTS))

    # my_parallel_loop(fourier_acceleration_kernel, n , n, n_fft, cupy.asnumpy(delta_fft_reshape), alpha)
    my_parallel_loop(complex_fourier_acceleration_kernel, n*n , n, cupy.asnumpy(delta_fft_reshape), alpha)

    # inverse fourier transform to position space
    # delta_accfft_reshape = cupy.reshape(cupy.array(delta_fft_reshape), (n, n_fft, su.GROUP_ELEMENTS))
    delta_accfft_reshape = cupy.reshape(cupy.array(delta_fft_reshape), (n, n, su.GROUP_ELEMENTS))

    # delta_accfft = cupy.fft.irfft2(delta_accfft_reshape, axes=(0, 1), s=(n, n))
    delta_accfft = cupy.fft.ifft2(delta_accfft_reshape, axes=(0, 1), s=(n, n))
    
    c_fft = cupy.asnumpy(cupy.reshape(delta_accfft, (n * n, su.GROUP_ELEMENTS)))

    my_parallel_loop(store_c_fft_kernel, n*n , c_fft, c)

@myjit  
def fourier_acceleration_kernel(x, n, n_fft, delta_fft, alpha):
    for y in range(n_fft):
        if (x > 0 or y > 0):
            # extract p^2 and p^2_max
            psq = psq_latt(x, y, n)
            # psqmax = (2*np.pi/n)**2    
            psqmax = psq_latt(n-1, n-1, n)

            buf = su.mul_s(delta_fft[x], alpha * psqmax / psq)
            # buf = su.mul_s(delta_fft[x], alpha / psq)
            su.store(delta_fft[x], buf)

@myjit  
def complex_fourier_acceleration_kernel(xi, n, delta_fft, alpha):
    x, y = l.get_point(xi, n)

    # extract p^2 and p^2_max
    psq = psq_latt(x, y, n)
    # psqmax = (2*np.pi/n)**2    
    psqmax = psq_latt(n-1, n-1, n)

    buf = su.mul_s(delta_fft[xi], alpha * psqmax / psq)
    su.store(delta_fft[xi], buf)

@myjit
def psq_latt(x, y, n):
    result = 4.0 * (math.sin((np.pi * x) / n) ** 2 + math.sin((np.pi * y) / n) ** 2)
    return result

@myjit
def store_c_fft_kernel(xi, c_fft, c):
    su.store(c[xi], c_fft[xi])