"""
    SU(2) group and algebra functions
"""
from curraun.numba_target import myjit, mynonparjit
from numba import prange

import os
import math
import numpy as np


"""
    SU(2) group & algebra functions
"""

N_C = 2
ALGEBRA_ELEMENTS = 3
GROUP_ELEMENTS = 4

su_precision = os.environ.get('PRECISION', 'double')

if su_precision == 'single':
    # TODO: convert all input variables to float32 before compiled functions are called
    #       (check this using compiled_function.inspect_types())
    print("Using single precision")
    GROUP_TYPE = np.float32
    GROUP_TYPE_REAL = np.float32
elif su_precision == 'double':
    print("Using double precision")
    GROUP_TYPE = np.float64
    GROUP_TYPE_REAL = np.float64
else:
    print("Unsupported precision: " + su_precision)

# @myjit
@mynonparjit
def get_algebra_element(algebra_factors):
    r0 = GROUP_TYPE(0.)
    r1 = GROUP_TYPE(algebra_factors[0] * 0.5)
    r2 = GROUP_TYPE(algebra_factors[1] * 0.5)
    r3 = GROUP_TYPE(algebra_factors[2] * 0.5)
    return r0, r1, r2, r3

# su2 multiplication
# @myjit
@mynonparjit
def mul(a, b): # TODO: rename to matmul (as in numpy)
    r0 = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    r1 = a[1] * b[0] + a[0] * b[1] + a[3] * b[2] - a[2] * b[3]
    r2 = a[2] * b[0] - a[3] * b[1] + a[0] * b[2] + a[1] * b[3]
    r3 = a[3] * b[0] + a[2] * b[1] - a[1] * b[2] + a[0] * b[3]
    return r0, r1, r2, r3

# exponential map
# @myjit
@mynonparjit
def mexp(a):
    norm = GROUP_TYPE(math.sqrt(a[1] * a[1] + a[2] * a[2] + a[3] * a[3]))

    if norm > 10E-18:
        sin_factor = GROUP_TYPE(math.sin(norm))
        r0 = GROUP_TYPE(math.cos(norm))
        r1 = sin_factor * a[1] / norm
        r2 = sin_factor * a[2] / norm
        r3 = sin_factor * a[3] / norm
    else:
        r0, r1, r2, r3 = unit()
    return r0, r1, r2, r3
    # r[0] = sqrt(1.0 - norm * norm)
    # for i in range(1, 4):
    #     r[i] = a[i]

# inverse
# @myjit
@mynonparjit
def inv(a):
    norm2 = a[0] ** 2 + a[1] ** 2 + a[2] ** 2 + a[3] ** 2
    r0 = a[0] / norm2
    r1 = -a[1]/ norm2
    r2 = -a[2] / norm2
    r3 = -a[3] / norm2
    return r0, r1, r2, r3

# anti-hermitian part
# @myjit
@mynonparjit
def ah(u):
    return GROUP_TYPE(0), u[1], u[2], u[3]

# group add: g0 = g0 + f * g1
# @myjit
@mynonparjit
def add(g0, g1):
    # Unfortunately, tuple creation from list comprehension does not work in numba:
    # see https://github.com/numba/numba/issues/2771
    #
    # result = tuple(g0[i] + g1[i] for i in range(4))
    # return result
    r0 = g0[0] + g1[0]
    r1 = g0[1] + g1[1]
    r2 = g0[2] + g1[2]
    r3 = g0[3] + g1[3]
    return r0, r1, r2, r3

# multiply by scalar
# @myjit
@mynonparjit
def mul_s(g0, f):  # TODO: rename to mul
    # Unfortunately, tuple creation from list comprehension does not work in numba:
    # see https://github.com/numba/numba/issues/2771
    #
    # result = tuple(f * g0[i] for i in range(4))
    # return result
    r0 = GROUP_TYPE(f) * g0[0]
    r1 = GROUP_TYPE(f) * g0[1]
    r2 = GROUP_TYPE(f) * g0[2]
    r3 = GROUP_TYPE(f) * g0[3]
    return r0, r1, r2, r3

# conjugate transpose
# @myjit
@mynonparjit
def dagger(a):  # TODO: rename to 'H'? See Numpy https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.H.html
    return a[0], -a[1], -a[2], -a[3]

"""
    Useful functions for temporary fields (setting to zero, unit and addition, ...)
"""

# get group element zero
# @myjit
@mynonparjit
def zero():
    return GROUP_TYPE(0), GROUP_TYPE(0), GROUP_TYPE(0), GROUP_TYPE(0)

# get group element unit
# @myjit
@mynonparjit
def unit():
    return GROUP_TYPE(1), GROUP_TYPE(0), GROUP_TYPE(0), GROUP_TYPE(0)

# group store: g0 <- g1
# @myjit
@mynonparjit
def store(g_to, g_from):
    g_to[0] = g_from[0] # Type-safe version
    g_to[1] = g_from[1]
    g_to[2] = g_from[2]
    g_to[3] = g_from[3]
    #for i in range(4):
    #    g_to[i] = g_from[i]

# return tuple (local memory)
# @myjit
@mynonparjit
def load(g):
    return g[0], g[1], g[2], g[3]

# trace
# @myjit
@mynonparjit
def tr(a):
    return 2 * a[0]

# trace of square - return real part
# @myjit
@mynonparjit
def sq(a): # tr(mul(a,a)) - valid only for traceless matrices a.
    return 2 * (a[1] ** 2 + a[2] ** 2 + a[3] ** 2)

# algebra dot product
# @myjit
@mynonparjit
def dot(a, b): # TODO: remove
    return a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# normalize su(2) group element
@myjit
def normalize(u):
    norm = GROUP_TYPE(0)
    # for i in range(4):
    for i in prange(4):
        norm += u[i] ** 2

    norm = math.sqrt(norm)
    # for i in range(4):
    for i in prange(4):
        u[i] = u[i] / norm
