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

max_iters = 500
if su.su_precision == 'single':
    coulomb_accuracy = 1e-6
elif su.su_precision == 'double':
    coulomb_accuracy = 1e-12
else:
    print("Unsupported precision: " + su.su_precision)

class CoulombGaugeTransf:
    def __init__(self, s, alpha=0.08, max_iters=max_iters, accuracy=None):
        self.s = s
        self.n = s.n
        nn = self.n ** 2

        # self.alpha = psq_latt_cpu(self.n-1, self.n-1, self.n)
        self.alpha = alpha
        self.max_iters = max_iters

        # Set accuracy based on precision if not specified
        if accuracy is None:
            self.accuracy = coulomb_accuracy
        else:
            self.accuracy = accuracy

        if DEBUG:
            print("alpha:", self.alpha)
            print("accuracy:", self.accuracy)
            print("max_iters:", self.max_iters)

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
        self.d_c = self.c
        self.d_thetax = self.thetax

        if DEBUG:
            self.d_ugunit = self.ugunit
            self.d_gunit = self.gunit

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
        self.d_c = cuda.to_device(self.c)
        self.d_thetax = cuda.to_device(self.thetax)

        if DEBUG:
            self.d_ugunit = cuda.to_device(self.ugunit)
            self.d_gunit = cuda.to_device(self.gunit)

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
        self.d_c.copy_to_host(self.c)
        self.d_thetax.copy_to_host(self.thetax)

        if DEBUG:
            self.d_ugunit.copy_to_host(self.ugunit)
            self.d_gunit.copy_to_host(self.gunit)

        self.d_u0.copy_to_host(self.u0)
        self.d_u1.copy_to_host(self.u1)
        self.d_pt1.copy_to_host(self.pt1)
        self.d_pt0.copy_to_host(self.pt0)
        self.d_aeta0.copy_to_host(self.aeta0)
        self.d_aeta1.copy_to_host(self.aeta1)
        self.d_peta1.copy_to_host(self.peta1)
        self.d_peta0.copy_to_host(self.peta0)

"""
    Initialize the objects in the gauge transformation
"""

def init_gauge_transf(self):
    n = self.s.n
    nn = n ** 2

    # initialize gauge transformation with identity matrix
    my_parallel_loop(init_transf_kernel, nn, self.d_g0)
   
    # initialize gauge links with the glasma ones
    my_parallel_loop(init_gauge_links_kernel, nn, self.s.d_u0, self.d_ug0)

    # initialize all glasma objects
    my_parallel_loop(init_glasma_fields_kernel, nn, self.d_u0, self.s.d_u0, self.d_u1, self.s.d_u1, self.d_aeta0, self.s.d_aeta0, self.d_aeta1, self.s.d_aeta1, self.d_peta0, self.s.d_peta0, self.d_peta1, self.s.d_peta1, self.d_pt0, self.s.d_pt0, self.d_pt1, self.s.d_pt1)

@myjit
def init_transf_kernel(xi, g0):
    g0[xi] = su.unit()

@myjit
def init_gauge_links_kernel(xi, u0, ug0):
    for d in range(2):
        su.store(ug0[xi, d], u0[xi, d])

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


"""
    Perform the Coulomb gauge transformation iteratively until convergence is reached
"""

def gauge_fix(c, auto_extend=True, qeik_tforce=None):
    """
    Convenience function to perform complete Coulomb gauge transformation.

    This combines initialization and iteration into a single call.

    Args:
        c: CoulombGaugeTransf object
        auto_extend: If True, automatically extend max_iters if convergence not reached
        qeik_tforce: Optional KineticCanonicCheck object to initialize a0 after gauge transformation

    Example:
        c = coulomb.CoulombGaugeTransf(s, alpha=0.08, accuracy=1e-10)
        coulomb.gauge_fix(c)
    """

    if use_cuda:
        c.copy_to_device()

    init_gauge_transf(c)
    iter_gauge_transf(c, auto_extend=auto_extend)

    if use_cuda:
        c.copy_to_host()

    # Initialize a0 if qeik_tforce is provided
    if qeik_tforce is not None:
        import curraun.qhat_qeik as qeik
        t = round(c.s.t - 1E-8)
        if use_cuda:
            c.s.copy_to_device()
            qeik_tforce.copy_to_device()

        qeik.compute_ai(c.s, qeik_tforce.d_a0, t)
        qeik_tforce.a0_initialized = True

        if use_cuda:
            qeik_tforce.copy_to_host()
            c.s.copy_to_host()

def iter_gauge_transf(self, auto_extend=True):
    """
    Perform gauge fixing iterations until convergence.

    Args:
        auto_extend: If True, automatically extend max_iters if convergence not reached
    """
    total_iters = 0
    theta_prev = 1e10

    while True:
        for iter in range(self.max_iters):
            gauge_transform(self)
            total_iters += 1

            # Compute accuracy based on absolute value of theta
            theta_accuracy = int(np.floor(np.log10(abs(self.theta)))) if self.theta > 0 else -20

            if DEBUG:
                print('iter:', total_iters, 'theta:', self.theta, 'accuracy:', theta_accuracy)

            # Check absolute convergence
            if self.theta <= self.accuracy:
                print(f"Coulomb gauge condition reached in {total_iters} iterations with theta={self.theta:.2e}")
                break

            # Check if convergence has stalled (theta stops decreasing)
            if self.theta > theta_prev and self.theta < 1e-6:
                print(f"Convergence stalled at {total_iters} iterations with theta={self.theta:.2e}")
                break

            theta_prev = self.theta
        else:
            # max_iters reached without convergence
            if auto_extend and self.theta > self.accuracy:
                if DEBUG:
                    print(f"Max iterations ({self.max_iters}) reached with theta={self.theta:.2e}")
                    print(f"Extending by {self.max_iters} more iterations...")
                continue  # Continue to next batch of iterations
            else:
                if DEBUG:
                    print(f"Maximum iterations reached with theta={self.theta:.2e}")
                break

        # Break out of while loop if we broke out of for loop (converged or stalled)
        break

    # Copy the iterated gauge links ug0 to u0 before copying to simulation
    # The iteration operates on ug0, so ug0 contains the final Coulomb gauge links
    # Note: aeta0, peta0, pt0, etc. are now transformed iteratively in gauge_transform()
    # so they are consistent with ug0. No need to call apply_gauge_transf().
    copy_ug_to_u(self)
    copy_to_simulation(self)


def gauge_transform(self):
    # compute delta from current gauge links
    compute_delta(self.s, self.d_ug0, self.d_delta)

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

    # compute the incremental gauge transformation (not accumulated)
    update_gauge_transf_incremental(self.s, self.d_c, self.d_g1, self.alpha)

    # check unitarity of the gauge transformation
    if DEBUG:
        check_gunit(self.s, self.d_gunit, self.d_g1)

    # apply the incremental gauge transformation to the gauge links
    gauge_transf_links(self.s, self.d_g1, self.d_ug0, self.d_ug1)

    # apply the incremental gauge transformation to all other fields
    # This ensures aeta0, peta0, pt0, etc. are transformed consistently with ug0
    gauge_transf_fields(self.s, self.d_g1, self.d_aeta0, self.d_aeta1,
                        self.d_peta0, self.d_peta1, self.d_pt0, self.d_pt1)

    # accumulate the gauge transformation
    accumulate_gauge_transf(self.s, self.d_g1, self.d_g0)

    # check unitarity of the gauge links
    if DEBUG:
        check_ugunit(self.s, self.d_ugunit, self.d_ug1)

    iterate(self)

def gauge_transf_links(s, g1, ug0, ug1):
    n = s.n
    nn = n ** 2

    my_parallel_loop(gauge_transf_links_kernel, nn, n, g1, ug0, ug1)

@myjit
def gauge_transf_links_kernel(xi, n, g1, ug0, ug1):
    for d in range(2):
        xiplus = l.shift(xi, d, +1, n)
        ug1[xi, d] = l.dact(g1[xi], g1[xiplus], ug0[xi, d])

def gauge_transf_fields(s, g1, aeta0, aeta1, peta0, peta1, pt0, pt1):
    """Apply incremental gauge transformation to all fields (except links)."""
    n = s.n
    nn = n ** 2

    my_parallel_loop(gauge_transf_fields_kernel, nn, g1, aeta0, aeta1, peta0, peta1, pt0, pt1)

@myjit
def gauge_transf_fields_kernel(xi, g1, aeta0, aeta1, peta0, peta1, pt0, pt1):
    # Transform aeta and peta with adjoint action
    aeta0[xi] = l.act(g1[xi], aeta0[xi])
    aeta1[xi] = l.act(g1[xi], aeta1[xi])
    peta0[xi] = l.act(g1[xi], peta0[xi])
    peta1[xi] = l.act(g1[xi], peta1[xi])

    # Transform pt with adjoint action
    for d in range(2):
        pt0[xi, d] = l.act(g1[xi], pt0[xi, d])
        pt1[xi, d] = l.act(g1[xi], pt1[xi, d])

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

def compute_thetax(s, delta, thetax):
    n = s.n
    nn = n ** 2 

    my_parallel_loop(compute_thetax_kernel, nn, delta, thetax)

@myjit
def compute_thetax_kernel(xi, delta, thetax):
    thetax[xi] = su.tr(su.mul(delta[xi], su.dagger(delta[xi])))

def fourier_acceleration(s, delta, c):
    n = s.n

    if use_cupy:
        # Convert numba CUDA array to numpy, then to cupy
        delta_host = np.empty((n*n, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        delta.copy_to_host(delta_host)
        delta_cupy = cupy.array(delta_host)

        # fourier transform to momentum space
        delta_reshape = cupy.reshape(delta_cupy, (n, n, su.GROUP_ELEMENTS))
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

def update_gauge_transf_incremental(s, c, g, alpha):
    """Compute incremental gauge transformation g = exp(alpha * c)"""
    n = s.n
    nn = n ** 2

    my_parallel_loop(update_gauge_transf_incremental_kernel, nn, c, g, alpha)

@myjit
def update_gauge_transf_incremental_kernel(xi, c, g, alpha):
    buf = su.mexp(su.mul_s(c[xi], alpha))
    buf = su.reunitarize(buf)
    su.store(g[xi], buf)

def accumulate_gauge_transf(s, g_inc, g_acc):
    """Accumulate gauge transformation: g_acc = g_inc * g_acc"""
    n = s.n
    nn = n ** 2

    my_parallel_loop(accumulate_gauge_transf_kernel, nn, g_inc, g_acc)

@myjit
def accumulate_gauge_transf_kernel(xi, g_inc, g_acc):
    buf = su.mul(g_inc[xi], g_acc[xi])
    buf = su.reunitarize(buf)
    su.store(g_acc[xi], buf)

def update_gauge_transf(s, g0, c, g1, alpha):
    n = s.n
    nn = n ** 2

    my_parallel_loop(update_gauge_transf_kernel, nn, g0, c, g1, alpha)

@myjit
def update_gauge_transf_kernel(xi, g0, c, g1, alpha):
    buf = su.mexp(su.mul_s(c[xi], alpha))
    buf = su.reunitarize(buf)
    buf1 = su.mul(buf, g0[xi])
    su.store(g1[xi], buf1)

def iterate(self):
    # iterate pointers to CUDA device memory
    self.d_g0, self.d_g1 = self.d_g1, self.d_g0
    self.d_ug0, self.d_ug1 = self.d_ug1, self.d_ug0
    
    self.g0, self.g1 = self.g1, self.g0
    self.ug0, self.ug1 = self.ug1, self.ug0

"""
    Apply the coulomb gauge transformation to glasma fields
    Namely u0, u1, aeta0, aeta1 and conjugate momenta peta0, peta1, pt0, pt1
"""

def apply_gauge_transf(self):
    n = self.s.n
    nn = n ** 2

    my_parallel_loop(apply_gauge_transf_kernel, nn, n, self.d_g1, self.d_u0, self.d_u1, self.d_aeta0, self.d_aeta1, self.d_peta0, self.d_peta1, self.d_pt0, self.d_pt1)

@myjit
def apply_gauge_transf_kernel(xi, n, g1, u0, u1, aeta0, aeta1, peta0, peta1, pt0, pt1):
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

"""
    Copy iterated gauge links back to internal arrays
"""

def copy_ug_to_u(self):
    """Copy the iterated gauge links ug0 to u0 and ug1 to u1"""
    n = self.n
    nn = n ** 2

    my_parallel_loop(copy_ug_to_u_kernel, nn, self.d_ug0, self.d_u0, self.d_ug1, self.d_u1)

@myjit
def copy_ug_to_u_kernel(xi, ug0, u0, ug1, u1):
    for d in range(2):
        su.store(u0[xi, d], ug0[xi, d])
        su.store(u1[xi, d], ug1[xi, d])

"""
    Copy transformed glasma fields back to simulation object
"""

def copy_to_simulation(self):
    """Copy the gauge-transformed fields back to the simulation object s"""
    n = self.s.n
    nn = n ** 2

    my_parallel_loop(copy_to_simulation_kernel, nn, self.d_u0, self.s.d_u0, self.d_u1, self.s.d_u1,
                     self.d_aeta0, self.s.d_aeta0, self.d_aeta1, self.s.d_aeta1,
                     self.d_peta0, self.s.d_peta0, self.d_peta1, self.s.d_peta1,
                     self.d_pt0, self.s.d_pt0, self.d_pt1, self.s.d_pt1)

@myjit
def copy_to_simulation_kernel(xi, u0_src, u0_dst, u1_src, u1_dst,
                               aeta0_src, aeta0_dst, aeta1_src, aeta1_dst,
                               peta0_src, peta0_dst, peta1_src, peta1_dst,
                               pt0_src, pt0_dst, pt1_src, pt1_dst):
    for d in range(2):
        su.store(u0_dst[xi, d], u0_src[xi, d])
        su.store(u1_dst[xi, d], u1_src[xi, d])
        su.store(pt0_dst[xi, d], pt0_src[xi, d])
        su.store(pt1_dst[xi, d], pt1_src[xi, d])

    su.store(aeta0_dst[xi], aeta0_src[xi])
    su.store(aeta1_dst[xi], aeta1_src[xi])
    su.store(peta0_dst[xi], peta0_src[xi])
    su.store(peta1_dst[xi], peta1_src[xi])

"""
    Check the unitarity of the gauge transformation and the gauge links
"""

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


"""
    Compute the gauge potential A_i from the gauge links U_i and A_eta
"""

def compute_ai(s, u0, aeta0, t, ai):
    n = s.n
    nn = n ** 2

    my_parallel_loop(compute_ai_kernel, nn, u0, aeta0, t, ai)

@myjit
def compute_ai_kernel(xi, u0, aeta0, t, ai):
    ax = su.mlog(u0[xi, 0])
    ay = su.mlog(u0[xi, 1])
    az = su.mul_s(aeta0[xi], 1.0 / t)

    su.store(ai[xi, 0], ax)
    su.store(ai[xi, 1], ay)
    su.store(ai[xi, 2], az)