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
    def __init__(self, s, alpha='auto', max_iters=max_iters, accuracy=None):
        self.s = s
        self.n = s.n
        nn = self.n ** 2

        if alpha == 'auto':
            # Start with a moderately aggressive alpha.
            # The adaptive scheme in iter_gauge_transf will reduce it if oscillation occurs.
            self.alpha = 0.08
        else:
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
        # total accumulated gauge transformation (g_total = g_N * g_{N-1} * ... * g_1)
        self.g_total = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

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
        self.thetax = np.zeros(nn, dtype=su.GROUP_TYPE_REAL)
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

        # Precompute Fourier acceleration ratio (p^2_max / p^2) for GPU
        # This is the same for every iteration, so we cache it
        if use_cuda:
            x = cupy.arange(self.n)
            y = cupy.arange(self.n)
            xx, yy = cupy.meshgrid(x, y, indexing='xy')
            psq = 4.0 * (cupy.sin(cupy.pi * xx / self.n) ** 2 + cupy.sin(cupy.pi * yy / self.n) ** 2)
            psqmax = 4.0 * (cupy.sin(cupy.pi * (self.n // 2) / self.n) ** 2 + cupy.sin(cupy.pi * (self.n // 2) / self.n) ** 2)
            psq[0, 0] = 1.0  # Avoid division by zero
            self.d_fourier_ratio = psqmax / psq
            self.d_fourier_ratio[0, 0] = 1.0  # Don't modify zero mode
        else:
            self.d_fourier_ratio = None

        # Memory on the CUDA device:
        self.d_g0 = self.g0
        self.d_g1 = self.g1
        self.d_g_total = self.g_total
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
        self.d_g_total = cuda.to_device(self.g_total)
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
        self.d_g_total.copy_to_host(self.g_total)
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
    # initialize total accumulated gauge transformation with identity
    my_parallel_loop(init_transf_kernel, nn, self.d_g_total)

    # initialize gauge links with the glasma ones
    # NOTE: We only initialize ug0 from s.d_u0 because the iteration operates on ug0.
    # The transformation of u1 is handled separately via apply_gauge_transf_to_u1
    # after the iteration converges.
    my_parallel_loop(init_gauge_links_kernel, nn, self.s.d_u0, self.d_ug0)

    # initialize all glasma objects
    my_parallel_loop(init_glasma_fields_kernel, nn, self.d_u0, self.s.d_u0, self.d_u1, self.s.d_u1, self.d_aeta0, self.s.d_aeta0, self.d_aeta1, self.s.d_aeta1, self.d_peta0, self.s.d_peta0, self.d_peta1, self.s.d_peta1, self.d_pt0, self.s.d_pt0, self.d_pt1, self.s.d_pt1)

@myjit
def init_transf_kernel(xi, g0):
    su.store(g0[xi], su.unit())

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

def gauge_fix(c, auto_extend=True, qeik_tforce=None, apply_to_sim=True):
    """
    Convenience function to perform complete Coulomb gauge transformation.

    This combines initialization and iteration into a single call.

    Args:
        c: CoulombGaugeTransf object
        auto_extend: If True, automatically extend max_iters if convergence not reached
        qeik_tforce: Optional KineticCanonicCheck object to initialize a0 after gauge transformation
        apply_to_sim: If True (default), apply the transformation to the simulation fields.
                      If False, only compute g_total without modifying the simulation.

    Example:
        c = coulomb.CoulombGaugeTransf(s, alpha=0.08, accuracy=1e-10)
        coulomb.gauge_fix(c)
    """

    if use_cuda:
        c.copy_to_device()

    init_gauge_transf(c)
    iter_gauge_transf(c, auto_extend=auto_extend, apply_to_sim=apply_to_sim)

    if use_cuda:
        c.copy_to_host()

    # Initialize a0 if qeik_tforce is provided
    if qeik_tforce is not None:
        import curraun.qhat_qeik as qeik
        t = round(c.s.t - 1E-8)
        if use_cuda:
            # NOTE: Do NOT call c.s.copy_to_device() here!
            # The simulation's device arrays (s.d_aeta0, etc.) already contain
            # the gauge-transformed fields from copy_to_simulation().
            # Calling copy_to_device() would overwrite them with the old host data.
            qeik_tforce.copy_to_device()

        qeik.compute_ai(c.s, qeik_tforce.d_a0, t)
        qeik_tforce.a0_initialized = True

        if use_cuda:
            qeik_tforce.copy_to_host()
            # Copy the transformed fields from device back to host
            c.s.copy_to_host()

def iter_gauge_transf(self, auto_extend=True, apply_to_sim=True):
    """
    Perform gauge fixing iterations until convergence.

    Uses adaptive alpha:
    - If theta increases (oscillation), alpha is halved immediately.
    - If convergence is slow (theta ratio close to 1), alpha is increased by 10%.
    This finds the optimal alpha dynamically throughout the iteration.

    Args:
        auto_extend: If True, automatically extend max_iters if convergence not reached
        apply_to_sim: If True, copy the transformed fields back to simulation object
    """
    total_iters = 0
    theta_prev = 1e10
    monotone_count = 0

    while True:
        for iter in range(self.max_iters):
            gauge_transform(self)
            total_iters += 1

            # Compute accuracy based on absolute value of theta
            theta_accuracy = int(np.floor(np.log10(abs(self.theta)))) if self.theta > 0 else -20

            if DEBUG:
                print('iter:', total_iters, 'theta:', self.theta, 'accuracy:', theta_accuracy, 'alpha:', self.alpha)

            # Check absolute convergence
            if self.theta <= self.accuracy:
                print(f"Coulomb gauge condition reached in {total_iters} iterations with theta={self.theta:.2e}")
                break

            # Adaptive alpha
            # Use a tolerance for oscillation detection to avoid reacting to
            # floating-point noise when theta is small
            if self.theta > theta_prev * 1.05:
                # Genuine oscillation: reduce alpha
                self.alpha *= 0.5
                monotone_count = 0
                if DEBUG:
                    print(f"Oscillation detected, reducing alpha to {self.alpha:.6e}")
            else:
                ratio = self.theta / theta_prev if theta_prev > 0 else 0
                if ratio > 0.99:
                    # Convergence is very slow: increase alpha
                    monotone_count += 1
                    if monotone_count >= 5:
                        self.alpha *= 1.1
                        monotone_count = 0
                        if DEBUG:
                            print(f"Slow convergence, increasing alpha to {self.alpha:.6e}")
                elif ratio > 0.95:
                    # Convergence is slow: increase alpha gently
                    monotone_count += 1
                    if monotone_count >= 10:
                        self.alpha *= 1.05
                        monotone_count = 0
                        if DEBUG:
                            print(f"Slow convergence, increasing alpha to {self.alpha:.6e}")
                else:
                    monotone_count = 0

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

    # Copy the iterated gauge links ug0 to u0
    # The iteration operates on ug0, so ug0 contains the final Coulomb gauge links for u0
    copy_ug0_to_u0(self)

    # Apply the total gauge transformation to u1
    # The u1 field was NOT transformed during iteration (only ug0 was iterated).
    # We need to apply g_total to the original u1 to get the properly transformed u1.
    # This ensures energy density (which uses both u0 and u1) remains gauge invariant.
    apply_gauge_transf_to_u1(self)

    # Only copy to simulation if requested
    if apply_to_sim:
        copy_to_simulation(self)


def gauge_transform(self):
    # compute delta from current gauge links
    compute_delta(self.s, self.d_ug0, self.d_delta)

    compute_thetax(self.s, self.d_delta, self.d_thetax)

    # Compute mean - for CUDA, copy to host first then use numpy
    # (cupy.asarray doesn't work reliably with numba device arrays in older versions)
    if use_cuda:
        thetax_host = self.d_thetax.copy_to_host()
        self.theta = np.mean(thetax_host).real / su.NC
    else:
        self.theta = np.mean(self.d_thetax).real / su.NC

    # apply fourier acceleration
    fourier_acceleration(self.s, self.d_delta, self.d_c, self.d_fourier_ratio)

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

    # accumulate the gauge transformation (used internally for iteration)
    accumulate_gauge_transf(self.s, self.d_g1, self.d_g0)

    # accumulate into g_total: g_total = g1 * g_total
    # This keeps track of the TOTAL gauge transformation from the original fields
    accumulate_gauge_transf(self.s, self.d_g1, self.d_g_total)

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
        su.store(ug1[xi, d], l.dact(g1[xi], g1[xiplus], ug0[xi, d]))

def gauge_transf_fields(s, g1, aeta0, aeta1, peta0, peta1, pt0, pt1):
    """Apply incremental gauge transformation to all fields (except links)."""
    n = s.n
    nn = n ** 2

    my_parallel_loop(gauge_transf_fields_kernel, nn, g1, aeta0, aeta1, peta0, peta1, pt0, pt1)

@myjit
def gauge_transf_fields_kernel(xi, g1, aeta0, aeta1, peta0, peta1, pt0, pt1):
    # Transform aeta and peta with adjoint action
    # IMPORTANT: Copy to local variables first to avoid read-write aliasing issues on GPU
    # (same pattern as apply_gauge_transf_kernel)
    aeta0_prev = aeta0[xi]
    aeta1_prev = aeta1[xi]
    peta0_prev = peta0[xi]
    peta1_prev = peta1[xi]

    su.store(aeta0[xi], l.act(g1[xi], aeta0_prev))
    su.store(aeta1[xi], l.act(g1[xi], aeta1_prev))
    su.store(peta0[xi], l.act(g1[xi], peta0_prev))
    su.store(peta1[xi], l.act(g1[xi], peta1_prev))

    # Transform pt with adjoint action
    for d in range(2):
        pt0_prev = pt0[xi, d]
        pt1_prev = pt1[xi, d]
        su.store(pt0[xi, d], l.act(g1[xi], pt0_prev))
        su.store(pt1[xi, d], l.act(g1[xi], pt1_prev))

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
    # Take real part explicitly - trace of delta*delta^dagger is real but may have small imaginary noise
    thetax[xi] = su.tr(su.mul(delta[xi], su.dagger(delta[xi]))).real

def fourier_acceleration(s, delta, c, fourier_ratio=None):
    n = s.n

    if use_cupy:
        # Copy numba CUDA array to host, then to cupy
        # (cupy.asarray doesn't work reliably with numba device arrays in older versions)
        delta_host = delta.copy_to_host()
        delta_cupy = cupy.array(delta_host)

        # fourier transform to momentum space (on GPU)
        delta_reshape = cupy.reshape(delta_cupy, (n, n, su.GROUP_ELEMENTS))
        delta_fft = cupy.fft.fft2(delta_reshape, axes=(0, 1))

        # Apply cached Fourier acceleration ratio (p^2_max / p^2)
        delta_fft *= fourier_ratio[:, :, cupy.newaxis]

        # inverse fourier transform to position space (on GPU)
        delta_accfft = cupy.fft.ifft2(delta_fft, axes=(0, 1), s=(n, n))
        c_cupy = cupy.reshape(delta_accfft, (n * n, su.GROUP_ELEMENTS))

        # Copy result back to numba device array via host
        c_host = cupy.asnumpy(c_cupy)
        cuda.to_device(c_host, to=c)
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
        psqmax = psq_latt(n // 2, n // 2, n)

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
    # buf = su.mexp_cayham(su.mul_s(c[xi], alpha))
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
    # buf = su.mexp_cayham(su.mul_s(c[xi], alpha))
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
        su.store(u1[xi, d], l.dact(g1[xi], g1[xiplus], u1_prev))

        su.store(pt0[xi, d], l.act(g1[xi], pt0_prev))
        su.store(pt1[xi, d], l.act(g1[xi], pt1_prev))

    aeta0_prev, aeta1_prev = aeta0[xi], aeta1[xi]
    peta0_prev, peta1_prev = peta0[xi], peta1[xi]

    su.store(aeta0[xi], l.act(g1[xi], aeta0_prev))
    su.store(aeta1[xi], l.act(g1[xi], aeta1_prev))

    su.store(peta0[xi], l.act(g1[xi], peta0_prev))
    su.store(peta1[xi], l.act(g1[xi], peta1_prev))

"""
    Copy iterated gauge links back to internal arrays
"""

def copy_ug0_to_u0(self):
    """Copy the iterated gauge links ug0 to u0 only.

    Note: u1 is handled separately by apply_gauge_transf_to_u1 because
    the iteration only operates on ug0, not ug1.
    """
    n = self.n
    nn = n ** 2

    my_parallel_loop(copy_ug0_to_u0_kernel, nn, self.d_ug0, self.d_u0)

@myjit
def copy_ug0_to_u0_kernel(xi, ug0, u0):
    for d in range(2):
        su.store(u0[xi, d], ug0[xi, d])

def apply_gauge_transf_to_u1(self):
    """Apply the total accumulated gauge transformation g_total to u1.

    The Coulomb gauge iteration only operates on ug0 (which becomes u0).
    The u1 field (gauge links at time t+dt) must be transformed separately
    using the total gauge transformation g_total that was accumulated during
    the iteration. This ensures that both u0 and u1 are in the same gauge,
    which is necessary for gauge-invariant quantities like energy density.

    Gauge transformation for links: U'_i(x) = g(x) * U_i(x) * g(x+i)^dag
    """
    n = self.n
    nn = n ** 2

    my_parallel_loop(apply_gauge_transf_to_u1_kernel, nn, n, self.d_g_total, self.d_u1)

@myjit
def apply_gauge_transf_to_u1_kernel(xi, n, g_total, u1):
    for d in range(2):
        xiplus = l.shift(xi, d, +1, n)
        u1_prev = u1[xi, d]
        su.store(u1[xi, d], l.dact(g_total[xi], g_total[xiplus], u1_prev))

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


"""
    Compute A^tau from the gauge transformation.

    In temporal gauge A^tau = 0. After a gauge transformation g(x):
    A'^tau = (i/g_coupling) * (d_tau g) * g^dagger

    Since we perform Coulomb gauge fixing at discrete times, we compute
    the time derivative numerically: d_tau g ≈ (g_new - g_old) / delta_tau

    Then A^tau = (i/g_coupling) * (g_new - g_old) / delta_tau * g_new^dagger

    And <(A^tau)^2> is the lattice average of Tr(A^tau * A^tau^dagger)
"""

class AtauComputer:
    """Helper class to compute A^tau from consecutive gauge transformations.

    This class tracks the TOTAL accumulated gauge transformation g_total_acc
    across all time steps. Each call to gauge_fix gives an incremental g_total,
    and we accumulate: g_total_acc(n+1) = g_total(n+1) * g_total_acc(n)

    This way we track the full gauge transformation from the original temporal
    gauge fields to the current Coulomb gauge fields.
    """

    def __init__(self, s):
        self.s = s
        self.n = s.n
        nn = self.n ** 2

        # Total accumulated gauge transformation across ALL time steps
        # g_total_acc = g_total(n) * g_total(n-1) * ... * g_total(0)
        self.g_total_acc = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        # Initialize to identity
        for xi in range(nn):
            self.g_total_acc[xi] = su.unit()

        # Store the previous accumulated transformation (for computing d_tau g)
        self.g_prev = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.g_prev_initialized = False

        # A^tau at each lattice site
        self.atau = np.zeros((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        # (A^tau)^2 trace at each lattice site
        self.atau_sq = np.zeros(nn, dtype=su.GROUP_TYPE_REAL)

        # Device memory
        self.d_g_total_acc = self.g_total_acc
        self.d_g_prev = self.g_prev
        self.d_atau = self.atau
        self.d_atau_sq = self.atau_sq

    def copy_to_device(self):
        self.d_g_total_acc = cuda.to_device(self.g_total_acc)
        self.d_g_prev = cuda.to_device(self.g_prev)
        self.d_atau = cuda.to_device(self.atau)
        self.d_atau_sq = cuda.to_device(self.atau_sq)

    def copy_to_host(self):
        self.d_g_total_acc.copy_to_host(self.g_total_acc)
        self.d_g_prev.copy_to_host(self.g_prev)
        self.d_atau.copy_to_host(self.atau)
        self.d_atau_sq.copy_to_host(self.atau_sq)


def compute_atau(atau_comp, coulomb_transf, delta_tau):
    """
    Compute A^tau from the total accumulated gauge transformation.

    In temporal gauge A^tau = 0. After a gauge transformation g(x,tau) to Coulomb gauge:
        A'^tau = (i/g_coupling) * (∂_τ g) * g^†

    Each call to gauge_fix produces g_total for that step. We accumulate these
    in atau_comp.g_total_acc to track the TOTAL transformation from the original
    temporal gauge fields:
        g_total_acc(n) = g_total(n) * g_total(n-1) * ... * g_total(0)

    By comparing g_total_acc at consecutive times, we compute ∂_τ g.

    Args:
        atau_comp: AtauComputer object
        coulomb_transf: CoulombGaugeTransf object with the current gauge transformation
        delta_tau: Time step between consecutive calls (in lattice units)

    Returns:
        atau_sq_mean: Lattice average of Tr(A^tau * A^tau^dagger) / Nc
    """
    n = atau_comp.n
    nn = n ** 2

    # Get the gauge transformation from this step
    g_step = coulomb_transf.g_total

    # Accumulate into g_total_acc: g_total_acc = g_step * g_total_acc
    # This builds up the total transformation from the original fields
    my_parallel_loop(accumulate_g_kernel, nn, g_step, atau_comp.g_total_acc)

    if not atau_comp.g_prev_initialized:
        # First call: store the initial accumulated transformation
        # We can't compute A^tau yet - need two time steps
        my_parallel_loop(store_g_kernel, nn, atau_comp.g_total_acc, atau_comp.g_prev)
        atau_comp.g_prev_initialized = True
        if DEBUG:
            print("First Coulomb gauge fixing - storing g_total_acc, skipping A^tau computation")
        return 0.0

    if DEBUG:
        # Check that g_total_acc is significantly different from identity
        identity_check = np.zeros(nn, dtype=su.GROUP_TYPE_REAL)
        for xi in range(nn):
            diff = atau_comp.g_total_acc[xi] - su.unit()
            identity_check[xi] = np.sqrt(np.sum(np.abs(diff)**2))
        mean_deviation = np.mean(identity_check)
        max_deviation = np.max(identity_check)
        print(f"Mean |g_total_acc - 1| = {mean_deviation:.6e}, Max |g_total_acc - 1| = {max_deviation:.6e}")

        # Check difference between consecutive g_total_acc values
        diff_check = np.zeros(nn, dtype=su.GROUP_TYPE_REAL)
        for xi in range(nn):
            diff = atau_comp.g_total_acc[xi] - atau_comp.g_prev[xi]
            diff_check[xi] = np.sqrt(np.sum(np.abs(diff)**2))
        mean_diff = np.mean(diff_check)
        max_diff = np.max(diff_check)
        print(f"Mean |g_new - g_old| = {mean_diff:.6e}, Max |g_new - g_old| = {max_diff:.6e}")

    # Compute A^tau from the difference of consecutive accumulated transformations
    # A^tau = (i/g_coupling) * (g_new - g_old) / delta_tau * g_new^dag
    # In the glasma code convention with -ig absorbed into A fields:
    # -ig A^tau = (g_new - g_old) / delta_tau * g_new^dag
    my_parallel_loop(compute_atau_kernel, nn, atau_comp.g_total_acc, atau_comp.g_prev,
                     atau_comp.atau, delta_tau)

    # Compute (A^tau)^2 = Tr(A^tau * A^tau^dagger)
    my_parallel_loop(compute_atau_sq_kernel, nn, atau_comp.atau, atau_comp.atau_sq)

    # Store current accumulated transformation for next iteration
    my_parallel_loop(store_g_kernel, nn, atau_comp.g_total_acc, atau_comp.g_prev)

    # Compute lattice average
    atau_sq_mean = np.mean(atau_comp.atau_sq).real / su.NC

    if DEBUG:
        print(f"atau_sq_mean = {atau_sq_mean}")

    return atau_sq_mean


@myjit
def store_g_kernel(xi, g_src, g_dst):
    su.store(g_dst[xi], g_src[xi])


@myjit
def accumulate_g_kernel(xi, g_step, g_acc):
    """Accumulate gauge transformation: g_acc = g_step * g_acc"""
    buf = su.mul(g_step[xi], g_acc[xi])
    buf = su.reunitarize(buf)
    su.store(g_acc[xi], buf)


@myjit
def compute_atau_from_g_kernel(xi, g, atau, delta_tau):
    # Compute A^tau from the incremental gauge transformation g
    #
    # The gauge transformation g restores Coulomb gauge after time evolution.
    # g ≈ exp(i g_coupling A^tau delta_tau) ≈ 1 + i g_coupling A^tau delta_tau
    #
    # So: A^tau ≈ (g - 1) / (i g_coupling delta_tau) * g^dag
    #
    # In the glasma code convention with -ig absorbed into A fields:
    # -ig A^tau = (g - 1) / delta_tau * g^dag

    # Compute (g - 1) / delta_tau
    g_minus_1 = l.add_mul(g[xi], su.unit(), -1.0)
    diff = su.mul_s(g_minus_1, 1.0 / delta_tau)

    # Multiply by g^dagger from the right
    g_dag = su.dagger(g[xi])
    result = su.mul(diff, g_dag)

    su.store(atau[xi], result)


@myjit
def compute_atau_kernel(xi, g_new, g_old, atau, delta_tau):
    # In temporal gauge A^tau = 0. After gauge transformation g(x,tau):
    # A'^tau = (d_tau g) g^dag
    #
    # In the glasma code, fields have -ig absorbed: A -> -igA
    # So we store: -ig A'^tau = (d_tau g) g^dag
    #
    # Numerically: d_tau g ≈ (g_new - g_old) / delta_tau

    # Compute (g_new - g_old) / delta_tau
    diff = l.add_mul(g_new[xi], g_old[xi], -1.0)
    diff = su.mul_s(diff, 1.0 / delta_tau)

    # Multiply by g_new^dagger from the right: (d_tau g) * g^dag
    g_new_dag = su.dagger(g_new[xi])
    result = su.mul(diff, g_new_dag)

    su.store(atau[xi], result)


@myjit
def compute_atau_sq_kernel(xi, atau, atau_sq):
    # Tr(A^tau * A^tau^dagger) = Tr(A^tau * A^tau) since A^tau is anti-Hermitian
    # But we compute Tr(A^tau * A^tau^dagger) for generality
    atau_dag = su.dagger(atau[xi])
    product = su.mul(atau[xi], atau_dag)
    atau_sq[xi] = su.tr(product).real
