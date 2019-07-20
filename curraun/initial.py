from curraun.numba_target import myjit, my_parallel_loop
import numpy as np
import curraun.lattice as l
import curraun.su as su
from curraun.initial_su3_new import init_kernel_2_su3
from time import time

DEBUG = False


def init(s, w1, w2):
    u0 = s.u0
    u1 = s.u1
    pt1 = s.pt1
    pt0 = s.pt0
    aeta0 = s.aeta0
    aeta1 = s.aeta1
    peta1 = s.peta1
    peta0 = s.peta0
    v1 = w1
    v2 = w2
    n = s.n
    dt = s.dt
    dth = s.dt / 2.0

    # temporary transverse gauge links for each nucleus
    ua = np.zeros_like(u0)
    ub = np.zeros_like(u0)

    en_EL = np.zeros(n ** 2, dtype=np.double)  # TODO: Think about alternative implementation that reduces on GPU?
    en_BL = np.zeros(n ** 2, dtype=np.double)

    # TODO: keep arrays on GPU device during execution of these kernels
    t = time()
    my_parallel_loop(init_kernel_1, n ** 2, v1, v2, n, ua, ub)
    debug_print("Init: temporary transverse gauge links ({:3.2f}s)".format(time() - t))
    t = time()
    if su.N_C == 2:
        my_parallel_loop(init_kernel_2, n ** 2, u0, u1, ua, ub)
    elif su.N_C == 3:
        my_parallel_loop(init_kernel_2_su3, n ** 2, u0, u1, ua, ub)
    else:
        print("initial.py: SU(N) code not implemented")
    debug_print("Init: transverse gauge links ({:3.2f}s)".format(time() - t))
    t = time()
    my_parallel_loop(init_kernel_3, n ** 2, u0, peta1, n, ua, ub)
    debug_print("Init: long. electric field ({:3.2f}s)".format(time() - t))
    t = time()
    my_parallel_loop(init_kernel_4, n ** 2, u0, pt1, n, dt)
    debug_print("Init: trans. electric field corrections ({:3.2f}s)".format(time() - t))
    t = time()
    my_parallel_loop(init_kernel_5, n ** 2, u0, u1, pt1, aeta0, aeta1, peta1, dt, dth)
    debug_print("Init: gauge link corrections field ({:3.2f}s)".format(time() - t))
    t = time()
    my_parallel_loop(init_kernel_6, n ** 2, u0, u1, peta1, n, en_EL, en_BL)
    debug_print("Init: energy density check ({:3.2f}s)".format(time() - t))

    peta0[:,:] = peta1[:,:]
    pt0[:,:] = pt1[:,:]

    en_EL_sum = np.sum(en_EL)
    en_BL_sum = np.sum(en_BL)

    debug_print("Init: e_EL = {}".format(en_EL_sum))
    debug_print("Init: e_BL = {}".format(en_BL_sum))


@myjit
def init_kernel_1(xi, v1, v2, n, ua, ub):
    # temporary transverse gauge fields
    for d in range(2):
        xs = l.shift(xi, d, 1, n)
        buffer1 = su.mul(v1[xi], su.dagger(v1[xs]))
        su.store(ua[xi, d], buffer1)
        buffer2 = su.mul(v2[xi], su.dagger(v2[xs]))
        su.store(ub[xi, d], buffer2)


@myjit
def init_kernel_2(xi, u0, u1, ua, ub):
    # initialize transverse gauge links (longitudinal magnetic field)
    # (see PhD thesis eq.(2.135))  # TODO: add proper link or reference
    # This only works for SU(2).
    for d in range(2):
        b1 = su.load(ua[xi, d])
        b1 = su.add(b1, ub[xi, d])
        b2 = su.dagger(b1)
        b2 = su.inv(b2)
        b3 = su.mul(b1, b2)
        su.store(u0[xi, d], b3)
        su.store(u1[xi, d], b3)

@myjit
def init_kernel_3(xi, u0, peta1, n, ua, ub):
    # initialize pi field (longitudinal electric field)
    # (see PhD thesis eq.(2.136))  # TODO: add proper link or reference
    tmp_peta1 = su.zero()
    for d in range(2):
        xs = l.shift(xi, d, -1, n)

        b1 = su.load(ub[xi, d])
        b1 = l.add_mul(b1, ua[xi, d], -1)
        b1 = su.dagger(b1)

        b2 = su.mul(u0[xi, d], b1)
        b2 = l.add_mul(b2, b1, -1)

        b1 = su.load(ub[xs, d])
        b1 = l.add_mul(b1, ua[xs, d], -1)

        b3 = su.mul(su.dagger(u0[xs, d]), b1)
        b3 = l.add_mul(b3, b1, -1)

        b2 = su.add(b2, b3)
        b3 = su.ah(b2)

        tmp_peta1 = su.add(tmp_peta1, b3)
    tmp_peta1 = su.mul_s(tmp_peta1, 0.5)
    su.store(peta1[xi], tmp_peta1)


@myjit
def init_kernel_4(xi, u0, pt1, n, dt):
    # pt corrections at tau = dt / 2
    for d in range(2):
        # transverse electric field update
        b1 = l.plaquettes(xi, d, u0, n)
        b1 = l.add_mul(pt1[xi, d], b1, - dt ** 2 / 2.0)
        su.store(pt1[xi, d], b1)

@myjit
def init_kernel_5(xi, u0, u1, pt1, aeta0, aeta1, peta1, dt, dth):
    # coordinate update
    for d in range(2):
        # transverse link variables update
        b0 = su.mul_s(pt1[xi, d], dt / dth)
        b1 = su.mexp(b0)
        b2 = su.mul(b1, u0[xi, d])
        su.store(u1[xi, d], b2)

    # longitudinal gauge field update
    b1 = l.add_mul(aeta0[xi], peta1[xi], dth * dt)
    su.store(aeta1[xi], b1)


@myjit
def init_kernel_6(xi, u0, u1, peta1, n, en_EL, en_BL):
    # initial condition check (EL ~ BL?)
    b1 = l.plaq(u0, xi, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    en_BL[xi] += su.sq(b2) / 2

    b1 = l.plaq(u1, xi, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    en_BL[xi] += su.sq(b2) / 2

    # b1 = l.plaq(u0, xi, 0, 1, 1, 1, n)
    # en_BL[0] += 2*(1.0 - b1[0])
    # b1 = l.plaq(u1, xi, 0, 1, 1, 1, n)
    # en_BL[0] += 2*(1.0 - b1[0])

    en_EL[xi] += su.sq(peta1[xi])


def debug_print(s):
    if DEBUG:
        print(s)