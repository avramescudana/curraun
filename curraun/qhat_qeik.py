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

        # gauge field components squared
        self.az_sq = np.zeros(self.n ** 2, dtype=np.double)

        self.az_sq_mean = 0.0

        # mean values
        self.p_perp_mean = np.zeros(3, dtype=np.double)
        if use_cuda:
            # use pinned memory for asynchronous data transfer
            self.p_perp_mean = cuda.pinned_array(3, dtype=np.double)
            self.p_perp_mean[0:3] = 0.0

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
        self.d_az_sq = self.az_sq

    def copy_to_device(self):
        self.d_v = cuda.to_device(self.v)
        self.d_f = cuda.to_device(self.f)
        self.d_fi = cuda.to_device(self.fi)
        self.d_p_perp_x = cuda.to_device(self.p_perp_x)
        self.d_p_perp_y = cuda.to_device(self.p_perp_y)
        self.d_p_perp_z = cuda.to_device(self.p_perp_z)
        self.d_p_perp_mean = cuda.to_device(self.p_perp_mean)
        self.d_az_sq = cuda.to_device(self.az_sq)

    def copy_to_host(self):
        self.d_v.copy_to_host(self.v)
        self.d_f.copy_to_host(self.f)
        self.d_fi.copy_to_host(self.fi)
        self.d_p_perp_x.copy_to_host(self.p_perp_x)
        self.d_p_perp_y.copy_to_host(self.p_perp_y)
        self.d_p_perp_z.copy_to_host(self.p_perp_z)
        self.d_p_perp_mean.copy_to_host(self.p_perp_mean)
        self.d_az_sq.copy_to_host(self.az_sq)

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

            # compute asq
            compute_asq(self.s, self.d_az_sq, round(self.s.t - 10E-8), stream)

            self.az_sq_mean = np.mean(self.d_az_sq)

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

    my_parallel_loop(compute_f_kernel, n * n, n, u0, aeta0, aeta1, peta1, peta0, pt1, pt0, f, t, tau, stream=stream)

@myjit
def compute_f_kernel(xi, n, u0, aeta0, aeta1, peta1, peta0, pt1, pt0, f, t, tau):

    xs = l.shift(xi, 0, t, n)

    # f_1 = \partial_y A_x in quantum eikonal approximation

    # Gauge-covariant symmetric derivative
    # xspy1 = l.shift(xs, 1, 1, n)
    # axpy1 = su.mlog(u0[xspy1, 0])
    # axpy1_transp = l.act(u0[xs, 0], axpy1)

    # xsmy1 = l.shift(xs, 1, -1, n)
    # axmy1 = su.mlog(u0[xsmy1, 0])
    # axmy1_transp = l.act(su.dagger(u0[xsmy1, 0]), axmy1)

    # dyax = l.add_mul(axpy1_transp, axmy1_transp, -1.0)

    # Naive symmetric partial derivative
    xspy1 = l.shift(xs, 1, 1, n)
    axpy1 = su.mlog(u0[xspy1, 0])

    xsmx1 = l.shift(xs, 1, -1, n)
    axmy1 = su.mlog(u0[xsmx1, 0])

    dif = l.add_mul(axpy1, axmy1, -1.0)
    dyax = su.mul_s(dif, 0.5)

    # Naive forwaard partial derivative
    # xspy1 = l.shift(xs, 1, 1, n)
    # axpy1 = su.mlog(u0[xspy1, 0])

    # axy = su.mlog(u0[xs, 0])
    # dyax = l.add_mul(axpy1, axy, -1.0)

    su.store(f[xi, 0], dyax)

    # f_2 = \partial_y A_x - D_x A_y in quantum eikonal approximation
    # Gauge-covariant symmetric derivative
    # xspx1 = l.shift(xs, 0, 1, n)
    # aypx1 = su.mlog(u0[xspx1, 1])
    # aypx1_transp = l.act(u0[xs, 1], aypx1)

    # xsmx1 = l.shift(xs, 0, -1, n)
    # aymx1 = su.mlog(u0[xsmx1, 1])
    # aymx1_transp = l.act(su.dagger(u0[xsmx1, 0]), aymx1)

    # dxay = l.add_mul(aypx1_transp, aymx1_transp, -1.0)
    # dxay_dyax = l.add_mul(dyax, dxay, -1.0)

    # # Gauge-covariant forward derivative
    # xspx1 = l.shift(xs, 0, 1, n)
    # aypx1 = su.mlog(u0[xspx1, 1])
    # aypx1_transp = l.act(u0[xs, 1], aypx1)

    # ayx = su.mlog(u0[xs, 1])

    # dxay = l.add_mul(aypx1_transp, ayx, -1.0)
    # dxay_dyax = l.add_mul(dyax, dxay, +1.0)

    # su.store(f[xi, 1], dxay_dyax)

    # f_1 = E_z = 1/\tau^2 A_\eta in canonical momentum
    bf1 = su.mul_s(aeta0[xs],  1.0 / (tau * tau))
    su.store(f[xi, 1],bf1)

    # f_2 = -B_z in classical simulation
    # quadratically accurate -Bz
    bf1 = su.zero()
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

    su.store(f[xi, 2], bf1)



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

@myjit
def transport_act(f, u, x, i, o):
    if o > 0:
        u1 = u[x, i]  # np-array
        result = l.act(u1, f[x])
    else:
        u2 = su.dagger(u[x, i])  # tuple
        result = l.act(u2, f[x])
    return result

def compute_asq(s, az_sq, t, stream):
    aeta0 = s.d_aeta0
    n = s.n

    my_parallel_loop(compute_asq_kernel, n * n, aeta0, az_sq, t, n, stream=stream)

@myjit
def compute_asq_kernel(xi, aeta0, az_sq, t, n):
    xs = l.shift(xi, 0, t, n)
   
    # A_z^2
    az = su.mul_s(aeta0[xs], 1.0 / t)
    # az = su.mul_s(aeta0[xi], 1.0 / t)
    az_sq[xi] = su.sq(az)