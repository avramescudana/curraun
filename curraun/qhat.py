from curraun.numba_target import myjit, prange, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
import curraun.kappa as kappa
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

        # light-like wilson lines
        self.v = np.zeros((self.n ** 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        my_parallel_loop(reset_wilsonfield, self.n ** 2, self.v)

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

        # time counter
        self.t = 0

        # Memory on the CUDA device:
        self.d_v = self.v
        self.d_f = self.f
        self.d_fi = self.fi
        self.d_p_perp_x = self.p_perp_x
        self.d_p_perp_y = self.p_perp_y
        self.d_p_perp_z = self.p_perp_z
        self.d_p_perp_mean = self.p_perp_mean

    def copy_to_device(self):
        self.d_v = cuda.to_device(self.v)
        self.d_f = cuda.to_device(self.f)
        self.d_fi = cuda.to_device(self.fi)
        self.d_p_perp_x = cuda.to_device(self.p_perp_x)
        self.d_p_perp_y = cuda.to_device(self.p_perp_y)
        self.d_p_perp_z = cuda.to_device(self.p_perp_z)
        self.d_p_perp_mean = cuda.to_device(self.p_perp_mean)

    def copy_to_host(self):
        self.d_v.copy_to_host(self.v)
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

    def compute(self,stream=None):
        tint = round(self.s.t / self.s.dt)
        if tint % self.dtstep == 0 and tint >= 1:
            # compute un-transported f
            compute_f(self.s, self.d_f, round(self.s.t - 10E-8), stream)

            # apply parallel transport
            apply_v(self.d_f, self.d_v, self.s.n, stream)

            # integrate f
            integrate_f(self.d_f, self.d_fi, self.s.n, 1.0, stream)

            # integrate perpendicular momentum
            compute_p_perp(self.d_fi, self.d_p_perp_x, self.d_p_perp_y, self.d_p_perp_z, self.s.n, stream)

            # calculate mean
            compute_mean(self.d_p_perp_x, self.d_p_perp_y, self.d_p_perp_z, self.d_p_perp_mean, stream)

        if tint % self.dtstep == self.dtstep / 2:
            # update v
            update_v(self.s, self.d_v, round(self.s.t - 10E-8), stream)



@myjit
def reset_wilsonfield(x, wilsonfield):
    su.store(wilsonfield[x], su.unit())


"""
    "Update" the light-like Wilson line.
    Adds a single link to the Wilson line.
"""


def update_v(s, v, t, stream):
    u = s.d_u0
    n = s.n

    my_parallel_loop(update_v_kernel, n * n, u, v, t, n, stream=stream)

@myjit
def update_v_kernel(xi, u, v, t, n):
    xs = l.shift(xi, 0, t, n)

    b1 = su.mul(v[xi], u[xs, 0])
    su.store(v[xi], b1)
    #l.normalize(v[xi])


"""
    Computes the correctly aligned (in the sense of lattice
    sites) force acting on a light-like trajectory particle.
"""


def compute_f(s, f, t, stream):
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
    tau = s.t # TODO: use tau_inverse = 1/s.t to avoid division in kernel? (measurable effect?)
    sign = +1.0 # TODO: can this constant be removed?

    my_parallel_loop(compute_f_kernel, n * n, n, u0, aeta0, aeta1, peta1, peta0, pt1, pt0, f, t, tau,
                     stream=stream)

@myjit
def compute_f_kernel(xi, n, u0, aeta0, aeta1, peta1, peta0, pt1, pt0, f, t, tau):

    # f_1 = E_1 (index 0)

    xs = l.shift(xi, 0, t, n)

    bf0 = su.zero()

    bf0 = su.add(bf0, pt1[xs, 0])
    bf0 = su.add(bf0, pt0[xs, 0])

    xs2 = l.shift(xs, 0, -1, n)
    b1 = l.act(su.dagger(u0[xs2, 0]), pt1[xs2, 0])
    bf0 = su.add(bf0, b1)
    b1 = l.act(su.dagger(u0[xs2, 0]), pt0[xs2, 0])
    bf0 = su.add(bf0, b1)
    bf0 = su.mul_s(bf0, 0.25 / tau)
    su.store(f[xi, 0], bf0)

    # f_2 = E_2 - B_3 (index 1)

    xs = l.shift(xi, 0, t, n)
    xs2 = l.shift(xs, 1, -1, n)

    bf1 = su.zero()

    # quadratically accurate +Ey
    bf1 = l.add_mul(bf1, pt1[xs, 1], 0.25 / tau)
    bf1 = l.add_mul(bf1, pt0[xs, 1], 0.25 / tau)
    xs3 = l.shift(xs, 1, -1, n)
    b1 = l.act(su.dagger(u0[xs3, 1]), pt1[xs2, 1])
    bf1 = l.add_mul(bf1, b1, 0.25 / tau)
    b1 = l.act(su.dagger(u0[xs3, 1]), pt0[xs2, 1])
    bf1 = l.add_mul(bf1, b1, 0.25 / tau)

    # quadratically accurate -Bz
    b1 = l.plaq(u0, xs, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xs, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, xs, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xs, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    su.store(f[xi, 1], bf1)

    # f_3 = E_3 + B_2 (index 2)
    bf2 = su.zero()

    # Accurate +E_z
    bf2 = l.add_mul(bf2, peta1[xs], 0.5)
    bf2 = l.add_mul(bf2, peta0[xs], 0.5)

    # Quadratically accurate +B_y
    b1 = l.transport(aeta0, u0, xs, 0, +1, n)
    b2 = l.transport(aeta0, u0, xs, 0, -1, n)
    b1 = l.add_mul(b1, b2, -1.0)
    bf2 = l.add_mul(bf2, b1, 0.5 / tau)

    su.store(f[xi, 2], bf2)


"""
    Applies the Wilson line to the untransported force.
    This is important for gauge covariance.
"""


def apply_v(f, v, n, stream):
    my_parallel_loop(apply_v_kernel, n * n, f, v, n, stream=stream)


@myjit
def apply_v_kernel(xi, f, v, n):
    for d in range(3):
        b1 = l.act(v[xi], f[xi, d])
        b1 = su.ah(b1)
        su.store(f[xi, d], b1)


"""
    Simple integration of forces to obtain 'color momenta'.
"""


def integrate_f(f, fi, n, dt, stream):
    kappa.integrate_f(f, fi, n, dt, stream)


"""
    Computes perpendicular momentum broadening as the trace
    of the square of the integrated color force (i.e. color
    momenta).
"""


def compute_p_perp(fi, p_perp_x, p_perp_y, p_perp_z, n, stream):
    kappa.compute_p_perp(fi, p_perp_x, p_perp_y, p_perp_z, n, stream)


def compute_mean(p_perp_x, p_perp_y, p_perp_z, p_perp_mean, stream):
    kappa.compute_mean(p_perp_x, p_perp_y, p_perp_z, p_perp_mean, stream)
