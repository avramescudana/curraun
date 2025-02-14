"""
    General group and algebra functions
    Grid functions
"""
from curraun.numba_target import myjit, mynonparjit
import curraun.su as su

import math

"""
    SU(2) group & algebra functions
"""

# product of 4 matrices
# @myjit
@mynonparjit
def mul4(a, b, c, d):
    ab = su.mul(a, b)
    cd = su.mul(c, d)
    abcd = su.mul(ab, cd)
    return abcd

# group add: g0 = g0 + f * g1
# @myjit
@mynonparjit
def add_mul(g0, g1, f):  # TODO: inline explicitly everywhere and remove this function
    return su.add(g0, su.mul_s(g1, f))

# adjoint action a -> u a u^t
# @myjit
@mynonparjit
def act(u, a):
    buffer1 = su.mul(u, a)
    result =  su.mul(buffer1, su.dagger(u))
    return result

# commutator of two su(2) elements
# @myjit
@mynonparjit
def comm(a, b):
    buffer1 = su.mul(a, b)
    buffer2 = su.mul(b, a)
    result = add_mul(buffer1, buffer2, -1)
    return result

"""
    Plaquette functions
"""

# compute 'positive' plaquette U_{x, i, j}
# @myjit
@mynonparjit
def plaq_pos(u, x, i, j, n):
    x1 = shift(x, i, 1, n)
    x2 = shift(x, j, 1, n)

    # U_{x, i} * U_{x+i, j} * U_{x+j, i}^t * U_{x, j}^t
    plaquette = mul4(u[x, i], u[x1, j], su.dagger(u[x2, i]), su.dagger(u[x, j]))
    return plaquette

# compute 'negative' plaquette U_{x, i, -j}
# @myjit
@mynonparjit
def plaq_neg(u, x, i, j, n):
    x0 = x
    x1 = shift(shift(x0, i, 1, n), j, -1, n)
    x2 = shift(x1, i, -1, n)
    x3 = x2

    # U_{x, i} * U_{x+i-j, j}^t * U_{x-j, i}^t * U_{x-j, j}
    return mul4(u[x0, i], su.dagger(u[x1, j]), su.dagger(u[x2, i]), u[x3, j])


# compute general plaquette U_{x, oi*i, oj*j}
# @myjit
@mynonparjit
def plaq(u, x, i, j, oi, oj, n):
    x0 = x
    x1 = shift(x0, i, oi, n)
    x2 = shift(x1, j, oj, n)
    x3 = shift(x2, i, -oi, n)

    u0 = get_link(u, x0, i, oi, n)
    u1 = get_link(u, x1, j, oj, n)
    u2 = get_link(u, x2, i, -oi, n)
    u3 = get_link(u, x3, j, -oj, n)

    # U_{x, i} * U_{x+i, j} * U_{x+i+j, -i} * U_{x+j, -j}
    return mul4(u0, u1, u2, u3)

# @myjit
@mynonparjit
def get_link(u, x, i, oi, n):
    if oi > 0:
        return su.load(u[x, i])
    else:
        xs = shift(x, i, oi, n)
        return su.dagger(u[xs, i])

# compute staple sum for optimized eom
# @myjit
@mynonparjit
def plaquettes(x, d, u, n):
    ci1 = shift(x, d, 1, n)
    i = (d + 1) % 2
    ci2 = shift(x, i, 1, n)
    ci3 = shift(ci1, i, -1, n)
    ci4 = shift(x, i, -1, n)
    buffer1 = su.mul(u[ci1, i], su.dagger(u[ci2, d]))
    buffer_S = su.mul(buffer1, su.dagger(u[x, i]))
    buffer1 = su.mul(su.dagger(u[ci3, i]), su.dagger(u[ci4, d]))
    buffer2 = su.mul(buffer1, u[ci4, i])
    buffer_S = su.add(buffer_S, buffer2)
    buffer1 = su.mul(u[x, d], buffer_S)
    result = su.ah(buffer1)
    return result


"""
    Parallel transport of 'scalar' fields (aeta, peta)
"""

# @myjit
@mynonparjit
def transport(f, u, x, i, o, n):
    xs = shift(x, i, o, n)
    if o > 0:
        u1 = u[x, i]  # np-array
        result = act(u1, f[xs])
    else:
        u2 = su.dagger(u[xs, i])  # tuple
        result = act(u2, f[xs])
    return result

"""
    Handy grid functions (optimized for square grids)
"""

# compute index from grid point
# @myjit
@mynonparjit
def get_index(ix, iy, n):  # TODO: remove
    return n * (ix % n) + (iy % n)

# index from grid point (no modulo)
# @myjit
@mynonparjit
def get_index_nm(ix, iy, n):
    return n * ix + iy

# compute grid point from index
# @myjit
@mynonparjit
def get_point(x, n):
    r1 = x % n
    r0 = (x - r1) // n
    return r0, r1

# index shifting
#@myjit
#@njit(locals={'result' : nb.int64},
#      parallel=True, nogil=True, fastmath=True)
#@cuda.jit(device=True)
# @myjit
@mynonparjit
def shift(x, i, o, n):
    r0, r1 = get_point(x, n)
    if i == 0:
        r0 = (r0 + o) % n
    else:
        r1 = (r1 + o) % n
    result = get_index_nm(r0, r1, n)
    return result
