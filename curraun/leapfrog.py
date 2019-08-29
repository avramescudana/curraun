from curraun.numba_target import myjit, prange, my_parallel_loop
import curraun.lattice as l
import curraun.su as su
import numpy as np


def evolve(s, stream=None):
    # Standard way to 'cast' numpy arrays from python objects
    # cdef cnp.ndarray[double, ndim=1, mode="c"] array = s.array

    u0 = s.d_u0
    u1 = s.d_u1
    pt1 = s.d_pt1
    pt0 = s.d_pt0
    aeta0 = s.d_aeta0
    aeta1 = s.d_aeta1
    peta1 = s.d_peta1
    peta0 = s.d_peta0

    dt = s.dt
    dth = s.dt * 0.5
    t = s.t
    n = s.n

    my_parallel_loop(evolve_kernel, n * n, u0, u1, pt1, pt0, aeta0, aeta1, peta1, peta0, dt, dth, t, n, stream=stream)


@myjit
def evolve_kernel(xi, u0, u1, pt1, pt0, aeta0, aeta1, peta1, peta0, dt, dth, t, n):
    # Momentum update
    # Input:
    #   t-dt/2: pt0, peta0
    #   t: u0, aeta0
    #
    # Output:
    #   t+dt/2: pt1, peta1
    #   t+dt: u1, aeta1

    peta_local = su.load(peta0[xi])

    for d in range(2):
        # transverse electric field update
        buffer2 = l.plaquettes(xi, d, u0, n)
        b2 = l.add_mul(pt0[xi, d], buffer2, - t * dt)
        buffer1 = l.transport(aeta0, u0, xi, d, 1, n)
        buffer2 = l.comm(buffer1, aeta0[xi])
        b2 = l.add_mul(b2, buffer2, + dt / t)
        su.store(pt1[xi, d], b2)

        # longitudinal electric field update
        buffer1 = l.transport(aeta0, u0, xi, d, 1, n)
        buffer2 = l.transport(aeta0, u0, xi, d, -1, n)
        buffer1 = su.add(buffer1, buffer2)
        buffer1 = l.add_mul(buffer1, aeta0[xi], -2)
        peta_local = l.add_mul(peta_local, buffer1, + dt / t)

    su.store(peta1[xi], peta_local)

    # Coordinate update
    for d in range(2):
        # transverse link variables update
        buffer0 = su.mul_s(pt1[xi, d], dt / (t + dth))
        buffer1 = su.mexp(buffer0)
        buffer2 = su.mul(buffer1, u0[xi, d])
        su.store(u1[xi, d], buffer2)

    # longitudinal gauge field update
    b2 = l.add_mul(aeta0[xi], peta1[xi], (t + dth) * dt)
    su.store(aeta1[xi], b2)

@myjit
def gauss(s):
    u0 = s.u0
    pt1 = s.pt1
    aeta0 = s.aeta0
    peta1 = s.peta1

    n = s.n

    result = 0.0

    buffer1 = l.zero()

    for xi in prange(n * n):
        for d in range(2):
            xs = l.shift(xi, d, -1, n)
            buffer2 = l.act(u0[xs, d], pt1[xs, d], -1)
            buffer2 = l.add_mul(buffer2, pt1[xi, d], -1)
            buffer1 = l.add(buffer1, buffer2)

        buffer2 = l.comm(l.load(aeta0[xi]), peta1[xi])
        buffer1 = l.add_mul(buffer1, buffer2, -1)

        result += l.sq(buffer1)

    return result


def normalize_all(s):
    n = s.n
    u0 = s.d_u0
    u1 = s.d_u1
    pt1 = s.d_pt1
    aeta0 = s.d_aeta0
    aeta1 = s.d_aeta1
    peta1 = s.d_peta1
    my_parallel_loop(normalize_all_kernel, n ** 2, u0, u1, pt1, aeta0, aeta1, peta1)


@myjit
def normalize_all_kernel(xi, u0, u1, pt1, aeta0, aeta1, peta1):
    for d in range(2):
        l.normalize(u0[xi, d])
        l.normalize(u1[xi, d])
        pt1[xi, d, 0] = 0.0
    aeta0[xi, 0] = 0.0
    aeta1[xi, 0] = 0.0
    peta1[xi,0] = 0.0
