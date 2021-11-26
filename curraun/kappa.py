from curraun.numba_target import myjit, use_cuda, my_parallel_loop, my_cuda_sum, mycudajit
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda

"""
    A module for various calculations related to momentum broadening and the \hat{q} parameter.
"""

class TransportedForce:
    def __init__(self, s):
        self.s = s
        self.n = s.n
        self.dtstep = round(1.0 / s.dt)

        # transported force
        self.f = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # integrated force
        self.fi = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # single components
        self.p_perp_x = np.zeros(self.n ** 2, dtype=np.double)
        self.p_perp_y = np.zeros(self.n ** 2, dtype=np.double)
        self.p_perp_z = np.zeros(self.n ** 2, dtype=np.double)

        # mean values
        self.p_perp_mean = np.zeros(3, dtype=np.double)
        if use_cuda:
            # use pinned memory for asynchronous data transfer
            self.p_perp_mean = cuda.pinned_array(3, dtype=np.double)
            self.p_perp_mean[0:3] = 0.0

        # time counter
        self.t = 0

        # Memory on the CUDA device:
        self.d_f = self.f
        self.d_fi = self.fi
        self.d_p_perp_x = self.p_perp_x
        self.d_p_perp_y = self.p_perp_y
        self.d_p_perp_z = self.p_perp_z
        self.d_p_perp_mean = self.p_perp_mean

    def copy_to_device(self):
        self.d_f = cuda.to_device(self.f)
        self.d_fi = cuda.to_device(self.fi)
        self.d_p_perp_x = cuda.to_device(self.p_perp_x)
        self.d_p_perp_y = cuda.to_device(self.p_perp_y)
        self.d_p_perp_z = cuda.to_device(self.p_perp_z)
        self.d_p_perp_mean = cuda.to_device(self.p_perp_mean)

    def copy_to_host(self):
        self.d_f.copy_to_host(self.f)
        self.d_fi.copy_to_host(self.fi)
        self.d_p_perp_x.copy_to_host(self.p_perp_x)
        self.d_p_perp_y.copy_to_host(self.p_perp_y)
        self.d_p_perp_z.copy_to_host(self.p_perp_z)
        self.d_p_perp_mean.copy_to_host(self.p_perp_mean)

    def copy_mean_to_device(self, stream=None):
        self.d_p_perp_mean = cuda.to_device(self.p_perp_mean, stream)

    def copy_mean_to_host(self, stream=None):
        self.d_p_perp_mean.copy_to_host(self.p_perp_mean, stream)

    def compute(self, stream=None):
        tint = round(self.s.t / self.s.dt)
        if tint % self.dtstep == 0 and tint >= 1:
            # compute un-transported f (temporal gauge does not need any transport)
            compute_f(self.s, self.d_f, stream)

            # integrate f
            integrate_f(self.d_f, self.d_fi, self.s.n, 1.0, stream)

            # integrate perpendicular momentum
            compute_p_perp(self.d_fi, self.d_p_perp_x, self.d_p_perp_y, self.d_p_perp_z, self.s.n, stream)

            # calculate mean
            compute_mean(self.d_p_perp_x, self.d_p_perp_y, self.d_p_perp_z, self.d_p_perp_mean, stream)

"""
    Correctly aligned calculation of the force for a resting particle (kappa).
    The particle is only affected by electric fields because it does not move.
"""
def compute_f(s, f, stream):
    u0 = s.d_u0
    u1 = s.d_u1
    pt1 = s.d_pt1
    aeta0 = s.d_aeta0
    aeta1 = s.d_aeta1
    peta1 = s.d_peta1
    pt0 = s.d_pt0
    peta0 = s.d_peta0

    n = s.n
    dt = s.dt
    dth = s.dt / 2.0
    tau = s.t

    my_parallel_loop(compute_f_kernel, n * n, n, u0, peta1, peta0, pt1, pt0, f, tau, stream=stream)

@myjit
def compute_f_kernel(xi, n, u0, peta1, peta0, pt1, pt0, f, tau):
    #### F_X & F_Y

    for d in range(2):
        # f_1 = E_1 (index 0)
        # f_2 = E_2 (index 1)
        bf = su.zero()

        # quadratically accurate +Ex
        # quadratically accurate +Ey
        bf = su.add(bf, pt1[xi, d])
        bf = su.add(bf, pt0[xi, d])

        xs = l.shift(xi, d, -1, n)
        b1 = l.act(su.dagger(u0[xs, d]), pt1[xs, d])
        bf = su.add(bf, b1)
        b1 = l.act(su.dagger(u0[xs, d]), pt0[xs, d])
        bf = su.add(bf, b1)
        bf = su.mul_s(bf, 0.25 / tau)
        su.store(f[xi, d], bf)

    #### F_Z

    # f_3 = E_3 (index 2)
    bf = su.zero()

    # Accurate +E_z
    bf = su.add(bf, peta1[xi])
    bf = su.add(bf, peta0[xi])
    bf = su.mul_s(bf, 0.5)
    # Is this correct? Ez=\tau*E_\eta
    bf = su.mul_s(bf, 1/tau)
    su.store(f[xi, 2], bf)


def integrate_f(f, fi, n, dt, stream):
    my_parallel_loop(integrate_f_kernel, n * n, f, fi, dt, stream=stream)

@myjit
def integrate_f_kernel(xi, f, fi, dt):
    for d in range(3):
        bfi = l.add_mul(fi[xi, d], f[xi, d], dt)
        su.store(fi[xi, d], bfi)


def compute_p_perp(fi, p_perp_x, p_perp_y, p_perp_z, n, stream):
    my_parallel_loop(compute_p_perp_kernel, n * n, fi, p_perp_x, p_perp_y, p_perp_z, stream=stream)


@myjit
def compute_p_perp_kernel(xi, fi, p_perp_x, p_perp_y, p_perp_z):
    #p_perp[xi] = 0
    p_perp_x[xi] = su.sq(fi[xi, 0])
    p_perp_y[xi] = su.sq(fi[xi, 1])
    p_perp_z[xi] = su.sq(fi[xi, 2])


def compute_mean(p_perp_x, p_perp_y, p_perp_z, p_perp_mean, stream):
    if use_cuda:
        # # Unfortunately, this version is blocking:
        # p_perp_mean[0] = sum_reduce(p_perp_x) / p_perp_x.size
        # p_perp_mean[1] = sum_reduce(p_perp_y) / p_perp_y.size
        # p_perp_mean[2] = sum_reduce(p_perp_z) / p_perp_z.size

        # TODO: use GPU summation which keeps the array intact

        # TODO: use Cupy.sum: https://docs-cupy.chainer.org/en/stable/reference/generated/cupy.sum.html#cupy.sum

        # CUDA version which does not require copy to host
        # It leaves the sum in array[0] and leaves the rest
        # of the array in an undefined state.
        my_cuda_sum(p_perp_x, stream)
        my_cuda_sum(p_perp_y, stream)
        my_cuda_sum(p_perp_z, stream)
        collect_results[1, 1, stream](p_perp_mean, p_perp_x, p_perp_y, p_perp_z)
    else:
        p_perp_mean[0] = np.mean(p_perp_x)
        p_perp_mean[1] = np.mean(p_perp_y)
        p_perp_mean[2] = np.mean(p_perp_z)

# @cuda.reduce
# def sum_reduce(a, b):
#    return a + b


@mycudajit  # @cuda.jit
def collect_results(p_perp_mean, p_perp_x, p_perp_y, p_perp_z):
    p_perp_mean[0] = p_perp_x[0] / p_perp_x.size
    p_perp_mean[1] = p_perp_y[0] / p_perp_y.size
    p_perp_mean[2] = p_perp_z[0] / p_perp_z.size
