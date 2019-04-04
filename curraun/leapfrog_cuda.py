from curraun.numba_target import myjit, prange, my_parallel_loop, mycudajit
import curraun.lattice as l
import numpy as np
import math
import numba.cuda as cuda
import numba

###########################################################################
# CUDA optimized version
#
# This version copies link variables within a block into shared memory
# before doing calculations locally.

BLOCK_X = 8 #16 #8
BLOCK_Y = 8 #16 #8
BLOCK_SHARED_X = 16 # BLOCK_X + 2 # allow for x +/-1
BLOCK_SHARED_Y = 16 # BLOCK_Y + 2 # allow for x +/-1
BLOCK_SHARED = BLOCK_SHARED_X * BLOCK_SHARED_Y

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

    #my_parallel_loop(evolve_kernel, n * n, u0, u1, pt1, pt0, aeta0, aeta1, peta1, peta0, dt, dth, t, n, stream=stream)

    iter_max = n * n
    # threadsperblock = 256
    # blockspergrid = math.ceil(iter_max / threadsperblock)

    threadsperblock = (BLOCK_X, BLOCK_Y)
    blockspergrid = (math.ceil(n / threadsperblock[0]), math.ceil(n / threadsperblock[1]))
    evolve_kernel[blockspergrid, threadsperblock, stream](iter_max, u0, u1, pt1, pt0, aeta0, aeta1, peta1, peta0, dt, dth, t, n)



@mycudajit
def evolve_kernel(iter_max, u0, u1, pt1, pt0, aeta0, aeta1, peta1, peta0, dt, dth, t, n):
    # Momentum update
    # Input:
    #   t-dt/2: pt0, peta0
    #   t: u0, aeta0
    #
    # Output:
    #   t+dt/2: pt1, peta1
    #   t+dt: u1, aeta1

    # Fields used:
    # transverse electric field update
    #   u0: [x,d], [x,i], [x+d,i], [x+i,d], [x+d-i,i], [x-i,d], [x-i,i]    (i = (d+1) % 2)
    #   u1
    #   pt1: --> [x,d]
    #   pt0: [x,d]
    #   aeta0: [x], [x+d]
    #   aeta1
    #   peta1
    #   peta0: [x]
    #
    #   coordinate update
    #   u0: [x,d]
    #   u1: --> [x,d]
    #   pt1: [x,d]
    #   pt0
    #   aeta0
    #   aeta1
    #   peta1
    #   peta0
    #
    #   longidutinal gauge field update
    #   u0
    #   u1
    #   pt1
    #   pt0
    #   aeta0: [x]
    #   aeta1: -->[x]
    #   peta1: [x]
    #   peta0
    #

    # Thread id within the block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Block id within the grid:
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Block size
    wx = cuda.blockDim.x
    wy = cuda.blockDim.y

    # Index position
    ix = tx + bx * wx
    iy = ty + by * wy
    xi = get_index_nm(ix, iy, n)
    xi_tmp = get_index_block(tx, ty)

    # Load u0 into shared memory

    u0_tmp = cuda.shared.array(shape=(BLOCK_SHARED, 2, 4), dtype=numba.float64)

    for d in range(2):
        i = 1 - d

        # [x,d], [x,i]
        xs = get_index_nm(ix, iy, n)
        xs_tmp = get_index_block(tx, ty)
        l.store(u0_tmp[xs_tmp, d], u0[xs, d])

        # [x+d,i], [x+i,d]
        xs = get_index_nm(ix + i, iy + d, n)
        xs_tmp = get_index_block(tx + i, ty + d)
        l.store(u0_tmp[xs_tmp, i], u0[xs, i])

        # [x+d-i,i]
        xs = get_index_nm(ix + i - d, iy + d - i, n)
        xs_tmp = get_index_block(tx + i - d, ty + d - i)
        l.store(u0_tmp[xs_tmp, i], u0[xs, i])

        # [x-i,d]
        xs = get_index_nm(ix - d, iy - i, n)
        xs_tmp = get_index_block(tx - d, ty - i)
        l.store(u0_tmp[xs_tmp, d], u0[xs, d])

        # [x-i,i]
        xs = get_index_nm(ix - d, iy - i, n)
        xs_tmp = get_index_block(tx - d, ty - i)
        l.store(u0_tmp[xs_tmp, i], u0[xs, i])

    cuda.syncthreads()

    # xi = cuda.grid(1)
    if xi < iter_max:

        peta_local = l.load(peta0[xi])

        for d in range(2):
            # transverse electric field update
            #buffer2 = l.plaquettes(xi, d, u0, n)
            buffer2 = plaquettes(xi_tmp, d, u0_tmp, n)
            b2 = l.add_mul(pt0[xi, d], buffer2, - t * dt)
            buffer1 = l.transport(aeta0, u0, xi, d, 1, n)
            buffer2 = l.comm(buffer1, aeta0[xi])
            b2 = l.add_mul(b2, buffer2, + dt / t)
            l.store(pt1[xi, d], b2)

            # longitudinal electric field update
            buffer1 = l.transport(aeta0, u0, xi, d, 1, n)
            buffer2 = l.transport(aeta0, u0, xi, d, -1, n)
            buffer1 = l.add(buffer1, buffer2)
            buffer1 = l.add_mul(buffer1, aeta0[xi], -2)
            peta_local = l.add_mul(peta_local, buffer1, + dt / t)

        l.store(peta1[xi], peta_local)

        # Coordinate update
        for d in range(2):
            # transverse link variables update
            buffer1 = l.mexp(pt1[xi, d], dt / (t + dth))
            buffer2 = l.mul(buffer1, u0[xi, d])
            l.store(u1[xi, d], buffer2)

        # longitudinal gauge field update
        b2 = l.add_mul(aeta0[xi], peta1[xi], (t + dth) * dt)
        l.store(aeta1[xi], b2)



# compute staple sum for optimized eom
@myjit
def plaquettes(x, d, u, n):
    if d == 0:
        dd = BLOCK_SHARED_Y # add to xi to move 1 step in d=x-direction
        di = 1 # add to xi to move 1 step in i=y-direction
    else:
        dd = 1 # add to xi to move 1 step in d=y-direction
        di = BLOCK_SHARED_Y # add to xi to move 1 step in i=x-direction
    ci1 = x + dd # shift(x, d, 1, n)
    i = 1 - d   # (d + 1) % 2
    ci2 = x + di # shift(x, i, 1, n)
    ci3 = ci1 - di # shift(ci1, i, -1, n)
    ci4 = x - di # shift(x, i, -1, n)
    buffer1 = l.mul(u[ci1, i], l.dagger(u[ci2, d]))
    buffer_S = l.mul(buffer1, l.dagger(u[x, i]))
    buffer1 = l.mul(l.dagger(u[ci3, i]), l.dagger(u[ci4, d]))
    buffer2 = l.mul(buffer1, u[ci4, i])
    buffer_S = l.add(buffer_S, buffer2)
    buffer1 =l. mul(u[x, d], buffer_S)
    result = l.ah(buffer1)
    return result


@myjit
def transport(f, u, x, i, o, n):
    xs = shift(x, i, o, n)
    if o > 0:
        u1 = u[x, i]  # np-array
        result = l.act(u1, f[xs])
    else:
        u2 = l.dagger(u[xs, i])  # tuple
        result = l.act(u2, f[xs])
    return result

# index from grid point (no modulo)
@myjit
def get_index_nm(ix, iy, n):
    return n * ix + iy

@myjit
def get_index_block(tx, ty):
    return BLOCK_SHARED_Y * (tx + 1) + (ty + 1)

# compute grid point from index
@myjit
def get_point(x, n):
    r1 = x % n
    r0 = (x - r1) // n
    return r0, r1


# index shifting
#@myjit
#@njit(locals={'result' : nb.int64},
#      parallel=True, nogil=True, fastmath=True)
#@cuda.jit(device=True)
@myjit
def shift(x, i, o, n):
    r0, r1 = get_point(x, n)
    if i == 0:
        r0 = (r0 + o) % n
    else:
        r1 = (r1 + o) % n
    result = get_index_nm(r0, r1, n)
    return result
