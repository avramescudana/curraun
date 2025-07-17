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
def complex_tuple(*t):
    return tuple(map(su.GROUP_TYPE, t))

s1 = complex_tuple(0, 1, 0, 1, 0, 0, 0, 0, 0)
# s1 = complex_tuple(0, 0, -1j, 0, 0, 0, 1j, 0, 0)

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
        self.ftilde = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # integrated force
        self.fi = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.ftildei = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # initial gauge field
        self.ai = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # paralel transported gauge field - initial gage field
        self.delta_ai = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # single components
        self.p_perp_x = np.zeros(self.n ** 2, dtype=np.double)
        self.p_perp_y = np.zeros(self.n ** 2, dtype=np.double)
        self.p_perp_z = np.zeros(self.n ** 2, dtype=np.double)

        # self.p_perp_A_x = np.zeros(self.n ** 2, dtype=np.double)
        # self.p_perp_A_y = np.zeros(self.n ** 2, dtype=np.double)
        # self.p_perp_A_z = np.zeros(self.n ** 2, dtype=np.double)
        self.p_perp_A = np.zeros((self.n ** 2, 3), dtype=np.double)

        # gauge field components squared
        # self.az_sq = np.zeros(self.n ** 2, dtype=np.double)
        # self.ay_sq = np.zeros(self.n ** 2, dtype=np.double)
        # self.ax_sq = np.zeros(self.n ** 2, dtype=np.double) 
        self.ai_sq = np.zeros((self.n ** 2, 3), dtype=np.double)

        # self.az_sq_mean = 0.0
        # self.ay_sq_mean = 0.0
        # self.ax_sq_mean = 0.0
        self.ai_sq_mean = np.zeros(3, dtype=np.double)

        self.deltapi = np.zeros((self.n ** 2, 3), dtype=np.double)
        self.deltap = np.zeros((self.n ** 2, 3), dtype=np.double)
        self.deltaA = np.zeros((self.n ** 2, 3), dtype=np.double)

        self.deltapi_mean = np.zeros(3, dtype=np.double)
        self.deltap_mean = np.zeros(3, dtype=np.double)
        self.deltaA_mean = np.zeros(3, dtype=np.double)

        # mean values
        self.p_perp_mean = np.zeros(3, dtype=np.double)
        self.p_perp_A_mean = np.zeros(3, dtype=np.double)
        if use_cuda:
            # use pinned memory for asynchronous data transfer
            self.p_perp_mean = cuda.pinned_array(3, dtype=np.double)
            self.p_perp_mean[0:3] = 0.0
            self.p_perp_A_mean = cuda.pinned_array(3, dtype=np.double)
            self.p_perp_A_mean[0:3] = 0.0
            self.ai_sq_mean = cuda.pinned_array(3, dtype=np.double)
            self.ai_sq_mean[0:3] = 0.0

            self.deltapi_mean = cuda.pinned_array(3, dtype=np.double)
            self.deltapi_mean[0:3] = 0.0
            self.deltap_mean = cuda.pinned_array(3, dtype=np.double)
            self.deltap_mean[0:3] = 0.0
            self.deltaA_mean = cuda.pinned_array(3, dtype=np.double)
            self.deltaA_mean[0:3] = 0.0

        # time counter
        self.t = 0

        # Memory on the CUDA device:
        self.d_v = self.v
        self.d_f = self.f
        self.d_fi = self.fi
        self.d_ftilde = self.ftilde
        self.d_ftildei = self.ftildei
        self.d_p_perp_x = self.p_perp_x
        self.d_p_perp_y = self.p_perp_y
        self.d_p_perp_z = self.p_perp_z
        self.d_p_perp_mean = self.p_perp_mean
        # self.d_az_sq = self.az_sq
        # self.d_ay_sq = self.ay_sq
        # self.d_ax_sq = self.ax_sq
        self.d_ai_sq = self.ai_sq

        self.d_ai = self.ai
        self.d_delta_ai = self.delta_ai

        # self.d_p_perp_A_x = self.p_perp_A_x
        # self.d_p_perp_A_y = self.p_perp_A_y
        # self.d_p_perp_A_z = self.p_perp_A_z
        self.d_p_perp_A = self.p_perp_A
        self.d_p_perp_A_mean = self.p_perp_A_mean

        self.d_deltapi = self.deltapi
        self.d_deltap = self.deltap
        self.d_deltaA = self.deltaA

        self.d_deltapi_mean = self.deltapi_mean
        self.d_deltap_mean = self.deltap_mean
        self.d_deltaA_mean = self.deltaA_mean



    def copy_to_device(self):
        self.d_v = cuda.to_device(self.v)

        self.d_f = cuda.to_device(self.f)
        self.d_fi = cuda.to_device(self.fi)
        self.d_ftilde = cuda.to_device(self.ftilde)
        self.d_ftildei = cuda.to_device(self.ftildei)

        self.d_p_perp_x = cuda.to_device(self.p_perp_x)
        self.d_p_perp_y = cuda.to_device(self.p_perp_y)
        self.d_p_perp_z = cuda.to_device(self.p_perp_z)
        self.d_p_perp_mean = cuda.to_device(self.p_perp_mean)
        # self.d_az_sq = cuda.to_device(self.az_sq)
        # self.d_ay_sq = cuda.to_device(self.ay_sq)
        # self.d_ax_sq = cuda.to_device(self.ax_sq)
        self.d_ai_sq = cuda.to_device(self.ai_sq)
        self.d_ai_sq_mean = cuda.to_device(self.ai_sq_mean)

        self.d_ai = cuda.to_device(self.ai)
        self.d_delta_ai = cuda.to_device(self.delta_ai)

        # self.d_p_perp_A_x = cuda.to_device(self.p_perp_A_x)
        # self.d_p_perp_A_y = cuda.to_device(self.p_perp_A_y)
        # self.d_p_perp_A_z = cuda.to_device(self.p_perp_A_z)
        self.d_p_perp_A = cuda.to_device(self.p_perp_A)
        self.d_p_perp_A_mean = cuda.to_device(self.p_perp_A_mean)

        self.d_deltapi = cuda.to_device(self.deltapi)
        self.d_deltap = cuda.to_device(self.deltap)
        self.d_deltaA = cuda.to_device(self.deltaA)

        self.d_deltapi_mean = cuda.to_device(self.deltapi_mean)
        self.d_deltap_mean = cuda.to_device(self.deltap_mean)
        self.d_deltaA_mean = cuda.to_device(self.deltaA_mean)

    def copy_to_host(self):
        self.d_v.copy_to_host(self.v)

        self.d_f.copy_to_host(self.f)
        self.d_fi.copy_to_host(self.fi)
        self.d_ftilde.copy_to_host(self.ftilde)
        self.d_ftildei.copy_to_host(self.ftildei)

        self.d_p_perp_x.copy_to_host(self.p_perp_x)
        self.d_p_perp_y.copy_to_host(self.p_perp_y)
        self.d_p_perp_z.copy_to_host(self.p_perp_z)
        self.d_p_perp_mean.copy_to_host(self.p_perp_mean)
        # self.d_az_sq.copy_to_host(self.az_sq)
        # self.d_ay_sq.copy_to_host(self.ay_sq)
        # self.d_ax_sq.copy_to_host(self.ax_sq)
        self.d_ai_sq.copy_to_host(self.ai_sq)
        self.d_ai_sq_mean.copy_to_host(self.ai_sq_mean)

        self.d_ai.copy_to_host(self.ai)
        self.d_delta_ai.copy_to_host(self.delta_ai)
        
        # self.d_p_perp_A_x.copy_to_host(self.p_perp_A_x)
        # self.d_p_perp_A_y.copy_to_host(self.p_perp_A_y)
        # self.d_p_perp_A_z.copy_to_host(self.p_perp_A_z)
        self.d_p_perp_A.copy_to_host(self.p_perp_A)
        self.d_p_perp_A_mean.copy_to_host(self.p_perp_A_mean)

        self.d_deltapi.copy_to_host(self.deltapi)
        self.d_deltap.copy_to_host(self.deltap)
        self.d_deltaA.copy_to_host(self.deltaA)

        self.d_deltapi_mean.copy_to_host(self.deltapi_mean)
        self.d_deltap_mean.copy_to_host(self.deltap_mean)
        self.d_deltaA_mean.copy_to_host(self.deltaA_mean)

    def copy_mean_to_device(self, stream=None):
        self.d_p_perp_mean = cuda.to_device(self.p_perp_mean, stream)
        self.d_p_perp_A_mean = cuda.to_device(self.p_perp_A_mean, stream)

    def copy_mean_to_host(self, stream=None):
        self.d_p_perp_mean.copy_to_host(self.p_perp_mean, stream)
        self.d_p_perp_A_mean.copy_to_host(self.p_perp_A_mean, stream)

    def compute(self,stream=None):
        tint = round(self.s.t / self.s.dt)
        tstart = round(1 / self.s.dt)
        if tint == tstart:
            compute_ai(self.s, self.d_ai, round(self.s.t - 10E-8), stream)

        if tint % self.dtstep == 0 and tint >= tstart:
            # compute un-transported f
            compute_ftilde(self.s, self.d_ftilde, round(self.s.t - 10E-8), stream)
            compute_f(self.s, self.d_f, round(self.s.t - 10E-8), stream)

            # apply parallel transport
            apply_v(self.d_f, self.d_v, self.s.n, stream)
            apply_v(self.d_ftilde, self.d_v, self.s.n, stream)

            # integrate f
            integrate_f(self.d_ftilde, self.d_ftildei, self.s.n, 1.0, stream)
            integrate_f(self.d_f, self.d_fi, self.s.n, 1.0, stream)

            # integrate perpendicular momentum
            compute_p_perp(self.d_ftildei, self.d_p_perp_x, self.d_p_perp_y, self.d_p_perp_z, self.s.n, stream)

            # calculate mean
            compute_mean(self.d_p_perp_x, self.d_p_perp_y, self.d_p_perp_z, self.d_p_perp_mean, stream)

            # calculate parallel transported gauge field minus the initial field
            compute_delta_ai(self.s, self.d_v, self.d_ai, round(self.s.t - 10E-8), self.d_delta_ai, stream)

            # integrate perpendicular momentum
            # compute_p_perp_A(self.s, self.d_fi, self.d_p_perp_A_y, self.d_p_perp_A_z, self.s.n, stream)
            compute_p_perp_A(self.d_ftildei, self.d_delta_ai, self.d_p_perp_A, self.s.n, stream)

            # calculate mean
            # compute_mean(self.d_p_perp_A_x, self.d_p_perp_A_y, self.d_p_perp_A_z, self.d_p_perp_A_mean, stream)
            compute_mean(self.d_p_perp_A[:, 0], self.d_p_perp_A[:, 1], self.d_p_perp_A[:, 2], self.d_p_perp_A_mean, stream)


            # compute asq
            # compute_asq(self.s, self.d_ay_sq, self.d_az_sq, round(self.s.t - 10E-8), stream)
            compute_asq(self.d_delta_ai, self.d_ai_sq, self.s.n, stream)

            # self.az_sq_mean = np.mean(self.d_az_sq)
            # self.ay_sq_mean = np.mean(self.d_ay_sq)

            compute_mean(self.d_ai_sq[:, 0], self.d_ai_sq[:, 1], self.d_ai_sq[:, 2], self.d_ai_sq_mean, stream)

            compute_delta(self.d_fi, self.d_ftildei, self.d_delta_ai, self.d_deltapi, self.d_deltap, self.d_deltaA, self.s.n, stream)

            compute_mean(self.d_deltapi[:, 0], self.d_deltapi[:, 1], self.d_deltapi[:, 2], self.d_deltapi_mean, stream)

            compute_mean(self.d_deltap[:, 0], self.d_deltap[:, 1], self.d_deltap[:, 2], self.d_deltap_mean, stream)

            compute_mean(self.d_deltaA[:, 0], self.d_deltaA[:, 1], self.d_deltaA[:, 2], self.d_deltaA_mean, stream)

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

def compute_delta(fi, ftildei, delta_ai, deltapi, deltap, deltaA, n, stream):

    my_parallel_loop(compute_delta_kernel, n * n, fi, ftildei, delta_ai, deltapi, deltap, deltaA, stream=stream)

@myjit 
def compute_delta_kernel(xi, fi, ftildei, delta_ai, deltapi, deltap, deltaA):
    t1 = su.mul_s(s1, 0.5)
    for i in range(3):
        # deltapi[xi, i] = su.tr(su.mul(fi[xi, i], t1)).real
        # deltap[xi, i] = su.tr(su.mul(ftildei[xi, i], t1)).real
        # deltaA[xi, i] = su.tr(su.mul(delta_ai[xi, i], t1)).real
        deltapi[xi, i] = su.tr(su.mul(fi[xi, i], t1)).imag
        deltap[xi, i] = su.tr(su.mul(ftildei[xi, i], t1)).imag
        deltaA[xi, i] = su.tr(su.mul(delta_ai[xi, i], t1)).imag


def compute_ai(s, ai, t, stream):
    u0 = s.d_u0
    aeta0 = s.d_aeta0

    n = s.n

    my_parallel_loop(compute_ai_kernel, n * n, u0, aeta0, t, ai, stream=stream)

@myjit
def compute_ai_kernel(xi, u0, aeta0, t, ai):
    ax = su.mlog(u0[xi, 0])
    ay = su.mlog(u0[xi, 1])
    az = su.mul_s(aeta0[xi], 1.0 / t)
    # ax = su.mul_s(su.mlog(u0[xi, 0]), 1j)
    # ay = su.mul_s(su.mlog(u0[xi, 1]), 1j)
    # az = su.mul_s(aeta0[xi], 1j / t)

    su.store(ai[xi, 0], ax)
    su.store(ai[xi, 1], ay)
    su.store(ai[xi, 2], az)

def compute_delta_ai(s, v, ai, t, delta_ai, stream):
    u0 = s.d_u0
    aeta0 = s.d_aeta0

    n = s.n

    my_parallel_loop(compute_delta_ai_kernel, n * n, v, u0, aeta0, t, ai, delta_ai, n, stream=stream)  

@myjit
def compute_delta_ai_kernel(xi, v, u0, aeta0, t, ai, delta_ai, n):
    xs = l.shift(xi, 0, t, n)

    ax = su.mlog(u0[xs, 0])
    ay = su.mlog(u0[xs, 1])
    az = su.mul_s(aeta0[xs], 1.0 / t)
    # ax = su.mul_s(su.mlog(u0[xs, 0]), 1j)
    # ay = su.mul_s(su.mlog(u0[xs, 1]), 1j)
    # az = su.mul_s(aeta0[xs], 1j / t)

    # axtransp = su.ah(l.act(v[xs], ax))
    axtransp = l.act(v[xs], ax)
    bufx = l.add_mul(axtransp, ai[xi, 0], -1)
    su.store(delta_ai[xi, 0], bufx)

    # aytransp = su.ah(l.act(v[xs], ay))
    aytransp = l.act(v[xs], ay)
    bufy = l.add_mul(aytransp, ai[xi, 1], -1)
    su.store(delta_ai[xi, 1], bufy)

    # aztransp = su.ah(l.act(v[xs], az))
    aztransp = l.act(v[xs], az)
    bufz = l.add_mul(aztransp, ai[xi, 2], -1)
    su.store(delta_ai[xi, 2], bufz)


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

def compute_ftilde(s, f, t, stream):
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

    my_parallel_loop(compute_ftilde_kernel, n * n, n, u0, aeta0, aeta1, peta1, peta0, pt1, pt0, f, t, tau, stream=stream)

@myjit
def compute_ftilde_kernel(xi, n, u0, aeta0, aeta1, peta1, peta0, pt1, pt0, f, t, tau):

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
    # axpy1 = su.mlog(u0[xspy1, 0])
    axpy1 = su.mul_s(su.mlog(u0[xspy1, 0]), 1j)


    xsmx1 = l.shift(xs, 1, -1, n)
    # axmy1 = su.mlog(u0[xsmx1, 0])
    axmy1 = su.mul_s(su.mlog(u0[xsmx1, 0]), 1j)

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
    bf1 = su.mul_s(aeta0[xs],  - 1.0 / (tau * tau))
    # bf1 = su.mul_s(aeta0[xs],  - 1j / (tau * tau))
    su.store(f[xi, 1], bf1)

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

# def compute_p_perp_A(s, fi, p_perp_A_y, p_perp_A_z, n, stream):
def compute_p_perp_A(fi, delta_ai, p_perp_A, n, stream):

    # my_parallel_loop(compute_p_perp_A_kernel, n * n, u0, aeta0, t, fi, p_perp_A_y, p_perp_A_z, n, stream=stream)
    my_parallel_loop(compute_p_perp_A_kernel, n * n, fi, delta_ai, p_perp_A, n, stream=stream)

@myjit
# def compute_p_perp_A_kernel(xi, u0, aeta0, t, fi, p_perp_A_y, p_perp_A_z, n):
def compute_p_perp_A_kernel(xi, fi, delta_ai, p_perp_A, n):
    # A_y
    # ay = su.mlog(u0[xi, 1])
    
    # A_z
    # az = su.mul_s(aeta0[xi], 1.0 / t)
    
    #p_perp[xi] = 0
    # p_perp_A_y[xi] = su.tr(su.mul(fi[xi, 1], su.dagger(ay))).real
    # p_perp_A_z[xi] = su.tr(su.mul(fi[xi, 2], su.dagger(az))).real
    for i in range(3):
        p_perp_A[xi, i] = su.tr(su.mul(fi[xi, i], su.dagger(delta_ai[xi, i]))).real


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

# def compute_asq(s, ay_sq, az_sq, t, stream):
def compute_asq(delta_ai, ai_sq, n, stream):

    my_parallel_loop(compute_asq_kernel, n * n, delta_ai, ai_sq, stream=stream)

@myjit
# def compute_asq_kernel(xi, u0, aeta0, ay_sq, az_sq, t, n):
def compute_asq_kernel(xi, delta_ai, ai_sq):
    # xs = l.shift(xi, 0, t, n)

    # A_y^2
    # ay = su.mul_s(su.mlog(u0[xi, 1]), 1j)
    # ay = su.mlog(u0[xs, 1])
    # ay = su.mlog(u0[xi, 0])
    # ay_sq[xi] = su.sq(ay)
   
    # A_z^2
    # az = su.mul_s(aeta0[xs], 1.0 / t)
    # az = su.mul_s(aeta0[xi], 1.0 / t)
    # az_sq[xi] = su.sq(az)

    for i in range(3):
        ai_sq[xi, i]= su.sq(delta_ai[xi, i])