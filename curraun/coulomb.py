from curraun.numba_target import myjit, use_cuda, my_parallel_loop, my_cuda_sum, mycudajit
import numpy as np
import math
import curraun.lattice as l
import curraun.su as su
from scipy.stats import unitary_group
if use_cuda:
    use_cupy = True
    import numba.cuda as cuda
    import cupy
else:
    use_cupy = False

"""
    A module for performing the Coulomb gauge transformation on the lattice.
    This is done at a fixed \tau time step in the glasma simulation.
    The \alpha convergence parameter is fixed accordint to its value for the Abelian case.
    The Coulomb gauge fixing is done until convergence is reached.
"""

DEBUG = True
TRANSF_GLASMA = True

max_iters = 1000
# max_iters = 1
if su.su_precision == 'single':
    # coulomb_accuracy = 1e-7
    coulomb_accuracy = 1e-6
elif su.su_precision == 'double':
    # coulomb_accuracy = 1e-17
    coulomb_accuracy = 1e-15
else:
    print("Unsupported precision: " + su.su_precision)

class CoulombGaugeTransf:
    def __init__(self, s):
        self.s = s
        self.n = s.n
        nn = self.n ** 2

        self.alpha = psq_latt_cpu(self.n-1, self.n-1, self.n) 
        # self.alpha = 0.02

        if DEBUG:
            print("alpha:", self.alpha)

        # gauge transformation at previous iteration
        self.g0 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        # gauge transformation at current iteration
        self.g1 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # gauge links at previous iteration
        self.ug0 = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        # gauge links at current iteration
        self.ug1 = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        if DEBUG:
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

        if TRANSF_GLASMA:
            # fields (times after evolve())
            self.u0 = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) 
            self.u1 = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) 
            self.pt1 = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) 
            self.pt0 = np.zeros((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) 

            self.aeta0 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) 
            self.aeta1 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) 
            self.peta1 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) 
            self.peta0 = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) 

            self.dt = s.dt
            self.t = s.t
            self.g = s.g

        # Memory on the CUDA device:
        self.d_g0 = self.g0
        self.d_g1 = self.g1
        self.d_ug0 = self.ug0
        self.d_ug1 = self.ug1
        self.d_delta = self.delta
        # self.d_theta = self.theta
        self.d_c = self.c
        self.d_thetax = self.thetax

        if DEBUG:
            self.d_ugunit = self.ugunit
            self.d_gunit = self.gunit

        if TRANSF_GLASMA:
            self.d_u0 = self.u0
            self.d_u1 = self.u1
            self.d_pt1 = self.pt1
            self.d_pt0 = self.pt0
            self.d_aeta0 = self.aeta0
            self.d_aeta1 = self.aeta1
            self.d_peta1 = self.peta1
            self.d_peta0 = self.peta0

    def copy_to_device(self):
        self.d_g0 = cuda.to_device(self.g0)
        self.d_g1 = cuda.to_device(self.g1)
        self.d_ug0 = cuda.to_device(self.ug0)
        self.d_ug1 = cuda.to_device(self.ug1)
        self.d_delta = cuda.to_device(self.delta)
        # self.d_theta = cuda.to_device(self.theta)
        self.d_c = cuda.to_device(self.c)
        self.d_thetax = cuda.to_device(self.thetax)

        if DEBUG:
            self.d_ugunit = cuda.to_device(self.ugunit)
            self.d_gunit = cuda.to_device(self.gunit)

        if TRANSF_GLASMA:   
            self.d_u0 = cuda.to_device(self.u0)
            self.d_u1 = cuda.to_device(self.u1)
            self.d_pt1 = cuda.to_device(self.pt1)
            self.d_pt0 = cuda.to_device(self.pt0)
            self.d_aeta0 = cuda.to_device(self.aeta0)
            self.d_aeta1 = cuda.to_device(self.aeta1)
            self.d_peta1 = cuda.to_device(self.peta1)
            self.d_peta0 = cuda.to_device(self.peta0)

    def copy_to_host(self):
        self.d_g0.copy_to_host(self.g0)
        self.d_g1.copy_to_host(self.g1)
        self.d_ug0.copy_to_host(self.ug0)   
        self.d_ug1.copy_to_host(self.ug1)
        self.d_delta.copy_to_host(self.delta)
        # self.d_theta.copy_to_host(self.theta)
        self.d_c.copy_to_host(self.c)
        self.d_thetax.copy_to_host(self.thetax)

        if DEBUG:
            self.d_ugunit.copy_to_host(self.ugunit)
            self.d_gunit.copy_to_host(self.gunit)

        if TRANSF_GLASMA:   
            self.d_u0.copy_to_host(self.u0)
            self.d_u1.copy_to_host(self.u1)
            self.d_pt1.copy_to_host(self.pt1)
            self.d_pt0.copy_to_host(self.pt0)
            self.d_aeta0.copy_to_host(self.aeta0)
            self.d_aeta1.copy_to_host(self.aeta1)
            self.d_peta1.copy_to_host(self.peta1)
            self.d_peta0.copy_to_host(self.peta0)

    # def copy_theta_to_device(self, stream=None):
    #     self.d_theta = cuda.to_device(self.theta, stream)

    # def copy_theta_to_host(self, stream=None):
    #     self.d_theta.copy_to_host(self.theta, stream)

# def apply_gauge_transf(self):
#     n = self.s.n
#     nn = n ** 2

#     # apply the coulomb gauge transformation to glasma fields
#     # namely u0, u1, aeta0, aeta1 and conjugate momenta peta0, peta1, pt0, pt1
#     my_parallel_loop(apply_gauge_links_kernel, nn, n, self.d_g1, self.d_u0, self.s.d_u0, self.d_u1, self.s.d_u1, self.d_aeta0, self.s.d_aeta0, self.d_aeta1, self.s.d_aeta1, self.d_peta0, self.s.d_peta0, self.d_peta1, self.s.d_peta1, self.d_pt0, self.s.d_pt0, self.d_pt1, self.s.d_pt1)

# @myjit
# def apply_gauge_links_kernel(xi, n, g1, u0_coul, u0_glasma, u1_coul, u1_glasma, aeta0_coul, aeta0_glasma, aeta1_coul, aeta1_glasma, peta0_coul, peta0_glasma, peta1_coul, peta1_glasma, pt0_coul, pt0_glasma, pt1_coul, pt1_glasma):
#     for d in range(2):
#         xiplus = l.shift(xi, d, +1, n)
#         u0_coul[xi, d] = l.dact(g1[xi], g1[xiplus], u0_glasma[xi, d])
#         u1_coul[xi, d] = l.dact(g1[xi], g1[xiplus], u1_glasma[xi, d])

#         pt0_coul[xi, d] = l.act(g1[xi], pt0_glasma[xi, d])
#         pt1_coul[xi, d] = l.act(g1[xi], pt1_glasma[xi, d])

#     aeta0_coul[xi] = l.act(g1[xi], aeta0_glasma[xi])
#     aeta1_coul[xi] = l.act(g1[xi], aeta1_glasma[xi])

#     peta0_coul[xi] = l.act(g1[xi], peta0_glasma[xi])
#     peta1_coul[xi] = l.act(g1[xi], peta1_glasma[xi])

def apply_gauge_transf(self):
    n = self.s.n
    nn = n ** 2

    # apply the coulomb gauge transformation to glasma fields
    # namely u0, u1, aeta0, aeta1 and conjugate momenta peta0, peta1, pt0, pt1
    my_parallel_loop(apply_gauge_links_kernel, nn, n, self.d_g1, self.d_u0, self.d_u1, self.d_aeta0, self.d_aeta1, self.d_peta0, self.d_peta1, self.d_pt0, self.d_pt1)

@myjit
def apply_gauge_links_kernel(xi, n, g1, u0, u1, aeta0, aeta1, peta0, peta1, pt0, pt1):
    for d in range(2):
        u0_prev, u1_prev = u0[xi, d], u1[xi, d]
        pt0_prev, pt1_prev = pt0[xi, d], pt1[xi, d]

        xiplus = l.shift(xi, d, +1, n)
        su.store(u0[xi, d], l.dact(g1[xi], g1[xiplus], u0_prev))
        u1[xi, d] = l.dact(g1[xi], g1[xiplus], u1_prev)

        pt0[xi, d] = l.act(g1[xi], pt0_prev)
        pt1[xi, d] = l.act(g1[xi], pt1_prev)

    aeta0_prev, aeta1_prev = aeta0[xi], aeta1[xi]
    peta0_prev, peta1_prev = peta0[xi], peta1[xi]

    aeta0[xi] = l.act(g1[xi], aeta0_prev)
    aeta1[xi] = l.act(g1[xi], aeta1_prev)

    peta0[xi] = l.act(g1[xi], peta0_prev)
    peta1[xi] = l.act(g1[xi], peta1_prev)

def iter_gauge_transf(self):
    for iter in range(max_iters):
        gauge_transform(self)
        apply_gauge_transf(self)

        # if DEBUG:
        #     print('iter:', iter+1, 'theta:', self.theta)

        theta_accuracy = int(np.floor(np.log10(abs(self.theta))))

        if self.theta <= coulomb_accuracy:
            if DEBUG:
                print(f"Coulomb gauge condition reached in {iter + 1} iterations with accuracy 1e{theta_accuracy}")
            break
    else:
        if DEBUG:
            print(f"Maximum iterations reached with accuracy: 1e{theta_accuracy}")

def init_gauge_transf(self):
    n = self.s.n
    nn = n ** 2

    # initialize gauge transformation with identity matrix
    init_transf(self.s, self.d_g0)
   
    # initialize gauge links with the glasma ones
    my_parallel_loop(init_gauge_links_kernel, nn, self.s.d_u0, self.d_ug0)

    if TRANSF_GLASMA:
        # initialize all glasma objects
        my_parallel_loop(init_glasma_fields_kernel, nn, self.d_u0, self.s.d_u0, self.d_u1, self.s.d_u1, self.d_aeta0, self.s.d_aeta0, self.d_aeta1, self.s.d_aeta1, self.d_peta0, self.s.d_peta0, self.d_peta1, self.s.d_peta1, self.d_pt0, self.s.d_pt0, self.d_pt1, self.s.d_pt1)

@myjit
def init_glasma_fields_kernel(xi, u0, u0_glasma, u1, u1_glasma, aeta0, aeta0_glasma, aeta1, aeta1_glasma, peta0, peta0_glasma, peta1, peta1_glasma, pt0, pt0_glasma, pt1, pt1_glasma):
    for d in range(2):
        su.store(u0[xi, d], u0_glasma[xi, d])
        su.store(u1[xi, d], u1_glasma[xi, d])
        su.store(pt0[xi, d], pt0_glasma[xi, d])
        su.store(pt1[xi, d], pt1_glasma[xi, d])

    su.store(aeta0[xi], aeta0_glasma[xi])
    su.store(aeta1[xi], aeta1_glasma[xi])
    su.store(peta0[xi], peta0_glasma[xi])
    su.store(peta1[xi], peta1_glasma[xi])

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

#TODO: remove this
def init_transf_random(s, g0):
    n = s.n
    
    for xi in range(n*n):
        g0[xi] = unitary_group.rvs(su.NC).reshape(su.NC*su.NC)

def gauge_transform(self):
    # apply the gauge transformation to the gauge links
    gauge_transf_links(self.s, self.d_g0, self.d_ug0, self.d_ug1)

    # check unitarity of the gauge links
    if DEBUG:
        check_ugunit(self.s, self.d_ugunit, self.d_ug1)

    # compute delta with previous iteration
    compute_delta(self.s, self.d_ug1, self.d_delta)

    compute_thetax(self.s, self.d_delta, self.d_thetax)
    #TODO: mean using cupy
    # if use_cupy:
    #     self.theta = cupy.mean(cupy.array(self.d_thetax))
    # else:
    #     self.theta = np.mean(self.d_thetax)

    self.theta = np.mean(self.d_thetax).real
    self.theta /= su.NC

    # apply fourier acceleration
    fourier_acceleration(self.s, self.d_delta, self.d_c)

    # update the gauge transformation
    update_gauge_transf(self.s, self.d_g0, self.d_c, self.d_g1, self.alpha)

    #TODO: remove this debug without fourier acceleration
    # update_gauge_transf(self.s, self.d_g0, self.d_delta, self.d_g1, self.alpha)

    # check unitarity of the gauge transformation
    if DEBUG:
        check_gunit(self.s, self.d_gunit, self.d_g1)

    iterate(self)

    # return self.theta

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

    my_parallel_loop(update_gauge_transf_kernel, nn, g0, c, g1, alpha)

@myjit
def update_gauge_transf_kernel(xi, g0, c, g1, alpha):
    buf = su.mexp(su.mul_s(c[xi], alpha))
    su.store(g1[xi], buf)

    # buf1 = su.mul(buf, g0[xi])
    # su.store(g1[xi], buf1)

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
        temp1 = l.add_mul(ug0[ximinus, d], ug0[xi, d], -1)
        temp2 = su.dagger(temp1)
        temp3 = l.add_mul(temp1, temp2, -1)
        temp4 = su.mul_s(su.unit(), su.tr(temp3)/su.NC)
        temp5 = l.add_mul(temp3, temp4, -1)
        buf = su.add(buf, temp5)

    su.store(delta[xi], buf)

def fourier_acceleration(s, delta, c):
    n = s.n

    if use_cupy:
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
    else:
        # fourier transform to momentum space
        delta_reshape = np.reshape(delta, (n, n, su.GROUP_ELEMENTS))
        delta_fft = np.fft.fft2(delta_reshape, axes=(0, 1))

        # fourier accelerate with alpha
        delta_fft_reshape = np.reshape(delta_fft, (n*n, su.GROUP_ELEMENTS))

        my_parallel_loop(complex_fourier_acceleration_kernel, n*n , n, delta_fft_reshape)

        # inverse fourier transform to position space
        delta_accfft_reshape = np.reshape(delta_fft_reshape, (n, n, su.GROUP_ELEMENTS))

        delta_accfft = np.fft.ifft2(delta_accfft_reshape, axes=(0, 1), s=(n, n))
        c_fft = np.reshape(delta_accfft, (n * n, su.GROUP_ELEMENTS))
        my_parallel_loop(store_c_fft_kernel, n*n , c_fft, c)


@myjit  
def complex_fourier_acceleration_kernel(xi, n, delta_fft):
    x, y = l.get_point(xi, n)

    # extract p^2 and p^2_max
    if (x > 0 or y > 0):
        psq = psq_latt(x, y, n)
        psqmax = psq_latt(n-1, n-1, n)

        buf = su.mul_s(delta_fft[xi], psqmax / psq)
        su.store(delta_fft[xi], buf)

@myjit
def psq_latt(x, y, n):
    result = 4.0 * (math.sin((np.pi * x) / n) ** 2 + math.sin((np.pi * y) / n) ** 2)
    return result

def psq_latt_cpu(x, y, n):
    result = 4.0 * (math.sin((np.pi * x) / n) ** 2 + math.sin((np.pi * y) / n) ** 2)
    return result

@myjit
def store_c_fft_kernel(xi, c_fft, c):
    su.store(c[xi], c_fft[xi])

