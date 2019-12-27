from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda

"""
    A module for computing E_z and B_z correlation functions
"""

@myjit
def compute_Bz(xi, n, u0, u1, aeta0, aeta1, pt0, pt1, peta0, peta1):
    bz = su.zero()

    # quadratically accurate +Bz
    b1 = l.plaq(u0, xi, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    bz = l.add_mul(bz, b2, -0.25)

    b1 = l.plaq(u0, xi, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bz = l.add_mul(bz, b2, +0.25)

    b1 = l.plaq(u0, xi, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bz = l.add_mul(bz, b2, -0.25)

    b1 = l.plaq(u0, xi, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    bz = l.add_mul(bz, b2, +0.25)

    return bz


@myjit
def compute_Ez(xi, n, u0, u1, aeta0, aeta1, pt0, pt1, peta0, peta1):
    ez = su.zero()

    # quadratically accurate +E_z
    ez = l.add_mul(ez, peta1[xi], 0.5)
    ez = l.add_mul(ez, peta0[xi], 0.5)

    return ez


class Correlators:
    def __init__(self, s):
        self.s = s

        self.corr = np.zeros(s.n // 2,  dtype=su.GROUP_TYPE_REAL)
        self.d_corr = self.corr

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_corr = cuda.to_device(self.corr)

    def copy_to_host(self):
        self.d_corr.copy_to_host(self.corr)

    def compute(self, mode):
        s = self.s

        self.corr[:] = 0.0
        if use_cuda:
            self.copy_to_device()

        if mode == 'Ez':
            my_parallel_loop(compute_Ez_correlation_kernel, s.n * s.n, s.n, s.d_u0, s.d_u1, s.d_aeta0, s.d_aeta1, s.d_pt0, s.d_pt1, s.d_peta0, s.d_peta1, self.d_corr)
        elif mode == 'Bz':
            my_parallel_loop(compute_Bz_correlation_kernel, s.n * s.n, s.n, s.d_u0, s.d_u1, s.d_aeta0, s.d_aeta1, s.d_pt0, s.d_pt1, s.d_peta0, s.d_peta1, self.d_corr)
        else:
            print("Correlators: mode '{}' is not implemented.".format(mode))

        if use_cuda:
            self.copy_to_host()

        # normalize, 2 * n ** 2 contributions per distance r
        self.corr /= 2 * s.n ** 2

        return self.corr


@myjit
def compute_Ez_correlation_kernel(xi, n, u0, u1, aeta0, aeta1, pt1, pt0, peta1, peta0, corr):
    Ux = su.unit()
    Uy = su.unit()

    F = compute_Ez(xi, n, u0, u1, aeta0, aeta1, pt1, pt0, peta1, peta0)

    for r in range(n // 2):
        # x shifts
        xs_x = l.shift(xi, 0, r, n)

        Fs_x = compute_Ez(xs_x, n, u0, u1, aeta0, aeta1, pt1, pt0, peta1, peta0)
        Fs_x = l.act(Ux, Fs_x)
        correlation = su.tr(su.mul(F, su.dagger(Fs_x))).real

        cuda.atomic.add(corr, r, correlation)

        # y shifts
        xs_y = l.shift(xi, 1, r, n)

        Fs_y = compute_Ez(xs_y, n, u0, u1, aeta0, aeta1, pt1, pt0, peta1, peta0)

        Fs_y = l.act(Uy, Fs_y)
        correlation = su.tr(su.mul(F, su.dagger(Fs_y))).real

        cuda.atomic.add(corr, r, correlation)

        # update Ux, Uy
        Ux = su.mul(Ux, u0[xs_x, 0])
        Uy = su.mul(Uy, u0[xs_y, 1])


@myjit
def compute_Bz_correlation_kernel(xi, n, u0, u1, aeta0, aeta1, pt1, pt0, peta1, peta0, corr):
    Ux = su.unit()
    Uy = su.unit()

    F = compute_Bz(xi, n, u0, u1, aeta0, aeta1, pt1, pt0, peta1, peta0)

    for r in range(n // 2):
        # x shifts
        xs_x = l.shift(xi, 0, r, n)

        Fs_x = compute_Bz(xs_x, n, u0, u1, aeta0, aeta1, pt1, pt0, peta1, peta0)
        Fs_x = l.act(Ux, Fs_x)
        correlation = su.tr(su.mul(F, su.dagger(Fs_x))).real

        cuda.atomic.add(corr, r, correlation)

        # y shifts
        xs_y = l.shift(xi, 1, r, n)

        Fs_y = compute_Bz(xs_y, n, u0, u1, aeta0, aeta1, pt1, pt0, peta1, peta0)

        Fs_y = l.act(Uy, Fs_y)
        correlation = su.tr(su.mul(F, su.dagger(Fs_y))).real

        cuda.atomic.add(corr, r, correlation)

        # update Ux, Uy
        Ux = su.mul(Ux, u0[xs_x, 0])
        Uy = su.mul(Uy, u0[xs_y, 1])


"""
    Wilson line correlator functions
"""


def wilson_correlator(v, n):
    v_corr = np.zeros(n ** 2, dtype=su.GROUP_TYPE_REAL)

    d_v_corr = v_corr
    d_v = v
    if use_cuda:
        d_v_corr = cuda.to_device(v_corr)
        d_v = cuda.to_device(v)

    if use_cuda:
        my_parallel_loop(wilson_correlator_cuda_kernel, n ** 2, n, d_v, d_v_corr)
    else:
        my_parallel_loop(wilson_correlator_kernel, n ** 2, n, d_v, d_v_corr)

    if use_cuda:
        d_v_corr.copy_to_host(v_corr)

    d_A = su.N_C ** 2 - 1
    v_corr /= n ** 2 * d_A
    return v_corr.reshape(n, n)

@myjit
def wilson_correlator_cuda_kernel(xi, n, v, v_corr):
    vx = v[xi]
    x = l.get_point(xi, n)

    for yi in range(n ** 2):
        vy = su.dagger(v[yi])
        y = l.get_point(yi, n)
        rx, ry = x[0] - y[0], x[1] - y[1]
        ri = l.get_index(rx, ry, n)
        correlation_fund = su.tr(su.mul(vx, vy))
        correlation_adj = correlation_fund.real ** 2 + correlation_fund.imag ** 2 - 1.0
        cuda.atomic.add(v_corr, ri, correlation_adj)

@myjit
def wilson_correlator_kernel(xi, n, v, v_corr):
    vx = v[xi]
    x = l.get_point(xi, n)

    for yi in range(n ** 2):
        vy = su.dagger(v[yi])
        y = l.get_point(yi, n)
        rx, ry = x[0] - y[0], x[1] - y[1]
        ri = l.get_index(rx, ry, n)
        correlation_fund = su.tr(su.mul(vx, vy))
        correlation_adj = correlation_fund.real ** 2 + correlation_fund.imag ** 2 - 1.0
        v_corr[ri] = v_corr[ri] + correlation_adj
