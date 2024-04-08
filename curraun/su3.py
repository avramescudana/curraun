"""
    SU(3) group and algebra functions
"""
import os
# os.environ["MY_NUMBA_TARGET"] = "python"  # Pure Python version    # TODO: remove debug code
# from curraun import su as su
from curraun.numba_target import myjit

import math
import numpy as np

# Definition of constants

N_C = 3  # Number of colors
ALGEBRA_ELEMENTS = 8
GROUP_ELEMENTS = 9


zero_algebra = (0,0,0,0,0,0,0,0)

unit_algebra = ((1,0,0,0,0,0,0,0),
                (0,1,0,0,0,0,0,0),
                (0,0,1,0,0,0,0,0),
                (0,0,0,1,0,0,0,0),
                (0,0,0,0,1,0,0,0),
                (0,0,0,0,0,1,0,0),
                (0,0,0,0,0,0,1,0),
                (0,0,0,0,0,0,0,1))


su_precision = os.environ.get('PRECISION', 'double')

if su_precision == 'single':
    # TODO: convert all input variables to float32 before compiled functions are called
    #       (check this using compiled_function.inspect_types())
    print("Using single precision")
    GROUP_TYPE = np.complex64 # two float32
    GROUP_TYPE_REAL = np.float32
elif su_precision == 'double':
    print("Using double precision")
    GROUP_TYPE = np.complex128 # two float64
    GROUP_TYPE_REAL = np.float64
else:
    print("Unsupported precision: " + su_precision)

EXP_MIN_TERMS = -1 # minimum number of terms in Taylor series
EXP_MAX_TERMS = 100 # maximum number of terms in Taylor series
EXP_ACCURACY_SQUARED = 1.e-40 # 1.e-32 # accuracy

def complex_tuple(*t):
    return tuple(map(GROUP_TYPE, t))

# Gell-Mann matrices
id0 = complex_tuple(1, 0, 0, 0, 1, 0, 0, 0, 1)
s1 = complex_tuple(0, 1, 0, 1, 0, 0, 0, 0, 0)
s2 = complex_tuple(0, -1j, 0, 1j, 0, 0, 0, 0, 0)
s3 = complex_tuple(1, 0, 0, 0, -1, 0, 0, 0, 0)
s4 = complex_tuple(0, 0, 1, 0, 0, 0, 1, 0, 0)
s5 = complex_tuple(0, 0, -1j, 0, 0, 0, 1j, 0, 0)
s6 = complex_tuple(0, 0, 0, 0, 0, 1, 0, 1, 0)
s7 = complex_tuple(0, 0, 0, 0, 0, -1j, 0, 1j, 0)
s8 = complex_tuple(1 / math.sqrt(3), 0, 0, 0, 1 / math.sqrt(3), 0, 0, 0, -2 / math.sqrt(3))

slist = (id0, s1, s2, s3, s4, s5, s6, s7, s8)

@myjit
def get_algebra_element(algebra_factors):
    """
    Algebra elements are formed from the algebra factors by:
    U = i t_a A_a = i lambda_a A_a / 2 ... for lambda_a the Gell-Mann matrices.

    The exponential map is given by mexp(get_algebra_element(algebra_factors))

    >>> get_algebra_element((1,0,0,0,0,0,0,0))
    (0j, 0.5j, 0j, 0.5j, 0j, 0j, 0j, 0j, 0j)

    :param a:
    :return:
    """
    ms0 = mul_s(s1, algebra_factors[0])
    ms1 = mul_s(s2, algebra_factors[1])
    ms2 = mul_s(s3, algebra_factors[2])
    ms3 = mul_s(s4, algebra_factors[3])
    ms4 = mul_s(s5, algebra_factors[4])
    ms5 = mul_s(s6, algebra_factors[5])
    ms6 = mul_s(s7, algebra_factors[6])
    ms7 = mul_s(s8, algebra_factors[7])

    # Add all
    b0 = add(ms0, ms1)
    b1 = add(ms2, ms3)
    b2 = add(ms4, ms5)
    b3 = add(ms6, ms7)
    c0 = add(b0, b1)
    c1 = add(b2, b3)
    d = add(c0, c1)

    # Overall factor
    res = mul_s(d, 0.5j)
    return res


"""
    SU(3) group & algebra functions
"""

# SU(3) group elements are given by 3x3 complex matrices:
#   a[0] a[1] a[2]
#   a[3] a[4] a[5]
#   a[6] a[7] a[8]
# (layout corresponds to C-order of numpy array)

# su3 multiplication

# Generate the matrix multiplication code:
# print("\n".join(["r{} = ".format(j) + " + ".join(["a[{}] * b[{}]".format(3 * (j // 3) + i, 3 * i + (j % 3)) for i in range(3)]) for j in range(9)]))

@myjit
def mul(a, b):
    """SU(3) multiplication: 3x3 matrix multiplication

    >>> a=[1,2,3,4,5,6,7,8,9]
    >>> b=[3,6,8,4,3,2,1,3,4]
    >>> c=mul(a,b)
    >>> c
    (14, 21, 24, 38, 57, 66, 62, 93, 108)

    # Check with numpy
    >>> import numpy as np
    >>> ma = np.asarray(a).reshape(3,3)
    >>> mb = np.asarray(b).reshape(3,3)
    >>> mc = np.matmul(ma, mb)
    >>> mc
    array([[ 14,  21,  24],
           [ 38,  57,  66],
           [ 62,  93, 108]])

    >>> tuple(mc.flatten()) == c
    True
    """
    r0 = a[0] * b[0] + a[1] * b[3] + a[2] * b[6]
    r1 = a[0] * b[1] + a[1] * b[4] + a[2] * b[7]
    r2 = a[0] * b[2] + a[1] * b[5] + a[2] * b[8]
    r3 = a[3] * b[0] + a[4] * b[3] + a[5] * b[6]
    r4 = a[3] * b[1] + a[4] * b[4] + a[5] * b[7]
    r5 = a[3] * b[2] + a[4] * b[5] + a[5] * b[8]
    r6 = a[6] * b[0] + a[7] * b[3] + a[8] * b[6]
    r7 = a[6] * b[1] + a[7] * b[4] + a[8] * b[7]
    r8 = a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

# Overload * operator
@myjit
def __mul__(a, b):
    """SU(3) multiplication: 3x3 matrix multiplication

    >>> a=[1,2,3,4,5,6,7,8,9]
    >>> b=[3,6,8,4,3,2,1,3,4]
    >>> c=mul(a,b)
    >>> c
    (14, 21, 24, 38, 57, 66, 62, 93, 108)

    # Check with numpy
    >>> import numpy as np
    >>> ma = np.asarray(a).reshape(3,3)
    >>> mb = np.asarray(b).reshape(3,3)
    >>> mc = np.matmul(ma, mb)
    >>> mc
    array([[ 14,  21,  24],
           [ 38,  57,  66],
           [ 62,  93, 108]])

    >>> tuple(mc.flatten()) == c
    True
    """
    
    if np.shape(a) == np.shape(b) == GROUP_ELEMENTS:
        mul(a,b)
    if np.shape(a) == 1 or np.shape(b) == 1:
        mul_s(a,b)

# exponential map
@myjit
def mexp(a):
    """ Calculate exponential using Taylor series

    mexp(a) = 1 + a + a^2 / 2 + a^3 / 6 + ...

    >>> a = id0
    >>> mexp(a)
    ((2.7182818284590455+0j), 0j, 0j, 0j, (2.7182818284590455+0j), 0j, 0j, 0j, (2.7182818284590455+0j))

    """
    res = id0
    t = id0
    for i in range(1, EXP_MAX_TERMS):
        t = mul(t, a)
        t = mul_s(t, 1/i)
        res = add(res, t)
        n = sq(t)  # TODO: Is it possible to improve performance by checking this not so often?
        if (i > EXP_MIN_TERMS) and (math.fabs(n.real) < EXP_ACCURACY_SQUARED):
            break
    else:
        # print("Exponential did not reach desired accuracy: {}".format(a))   # TODO: remove debugging code
        print("Exponential did not reach desired accuracy")  # TODO: remove debugging code
    return res

LOG_MIN_TERMS = -1 # minimum number of terms in Taylor series
LOG_MAX_TERMS = 100 # maximum number of terms in Taylor series
LOG_ACCURACY_SQUARED = 1.e-32 # 1.e-32 # accuracy

# logarithm map
def mlog(a):
    """
    Computes logarithm of a matrix using Taylor series

    mlog(a) = (a-id0) - (a-id0)^2 / 2 + (a-id0)^3 / 3 - (a-id0)^4 / 4 + ...

    Works for matrices close to identity (for example gauge links)
    """
    res = add(a, mul_s(id0, -1))
    t = add(a, mul_s(id0, -1))
    sign = -1
    for i in range(1, LOG_MAX_TERMS):
        t = mul(t, add(a, mul_s(id0, -1)))
        buff = mul_s(t, sign/(i+1))
        res = add(res, buff)
        sign = sign * (-1)
        n = sq(t) 
        if (i > LOG_MIN_TERMS) and (math.fabs(n.real) < LOG_ACCURACY_SQUARED):
            break
        # else:
            # print("Logarithm did not reach desired accuracy") 

    return res


# derivative of exponential map
@myjit
def dmexp(a, da):
    """ Calculate derivative of exponential using Taylor series

    dmexp(a, da) = 0 + da + (da a + a da) / 2 + (da a a + a da a + a a da) / 6 + ...

    >>> a = id0
    >>> da = s1
    >>> dmexp(a, da)
    (0j, (2.7182818284590455+0j), 0j, (2.7182818284590455+0j), 0j, 0j, 0j, 0j, 0j)

    """
    res = zero()
    t = zero()
    r = da
    s = add(t, r)
    f = 1
    res = add(res, s)   # da
    for i in range(2, EXP_MAX_TERMS):
        t = mul(s, a) # (da a a + a da a + a a da) -> (da a a a + a da a a + a a da a)
        r = mul(a, r) # (a a da) -> (a a a da)
        s = add(t, r) # -> (da a a a + a da a a + a a da a + a a a da)

        f = f / i
        s2 = mul_s(s, f)
        res = add(res, s2)
        n = sq(s2)  # TODO: Is it possible to improve performance by checking this not so often?
        if (i > EXP_MIN_TERMS) and (math.fabs(n.real) < EXP_ACCURACY_SQUARED):
            break
    else:
        # print("Derivative of exponential did not reach desired accuracy: {}".format(a))   # TODO: remove debugging code
        print("Derivative of exponential did not reach desired accuracy")  # TODO: remove debugging code
    return res

# inverse
@myjit
def inv(a):
    """
    >>> u = (1,1,0,1,2,0,0,0,1)
    >>> u
    (1, 1, 0, 1, 2, 0, 0, 0, 1)

    >>> inv(u)
    (2.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    >>> b = np.asarray(u)
    >>> c = b.reshape(3, 3)
    >>> d = np.linalg.inv(c)
    >>> e = d.reshape(9)
    >>> e
    array([ 2., -1.,  0., -1.,  1.,  0.,  0.,  0.,  1.])

    :param a:
    :return:
    """
    # b = np.asarray(a)
    # c = b.reshape(3, 3)
    # d = np.linalg.inv(c)
    # e = d.reshape(9)
    # return e

    d = det(a)

    if abs(d) < 1.e-18:
        print("Determinant too small")  # TODO: remove debugging code

    det_inv = 1 / d  # TODO: What to do if det == 0 ?

    r0 = det_inv * (a[4] * a[8] - a[5] * a[7])
    r1 = det_inv * (a[2] * a[7] - a[1] * a[8])
    r2 = det_inv * (a[1] * a[5] - a[2] * a[4])
    r3 = det_inv * (a[5] * a[6] - a[3] * a[8])
    r4 = det_inv * (a[0] * a[8] - a[2] * a[6])
    r5 = det_inv * (a[2] * a[3] - a[0] * a[5])
    r6 = det_inv * (a[3] * a[7] - a[4] * a[6])
    r7 = det_inv * (a[1] * a[6] - a[0] * a[7])
    r8 = det_inv * (a[0] * a[4] - a[1] * a[3])
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

# determinant
@myjit
def det(a):
    res  = a[0] * (a[4] * a[8] - a[5] * a[7])
    res += a[1] * (a[5] * a[6] - a[3] * a[8])
    res += a[2] * (a[3] * a[7] - a[4] * a[6])
    return res

# anti-hermitian part, eq. (C.25)
@myjit
def ah(u):
    """
    >>> u = get_algebra_element((1,0,0,0,0,0,0,0))
    >>> u
    (0j, 0.5j, 0j, 0.5j, 0j, 0j, 0j, 0j, 0j)

    >>> ah(u)
    (0j, 0.5j, 0j, 0.5j, 0j, 0j, 0j, 0j, 0j)

    Calculate definition of anti-hermitian part:
    # (see PhD thesis eq.(2.119))  # TODO: add proper link or reference
    >>> udiff = add(u, mul_s(dagger(u), -1))
    >>> unit = id0
    >>> tr_udiff = tr(udiff)
    >>> Uah_term1 = mul_s(udiff, 1/2)
    >>> Uah_term2 = mul_s(unit, tr_udiff / (2 * N_C))
    >>> Uah = add(Uah_term1, mul_s(Uah_term2, -1))
    >>> Uah
    (0j, 0.5j, 0j, 0.5j, 0j, 0j, 0j, 0j, 0j)

    >>> ah(u) == Uah
    True

    :param u:
    :return:
    """
    trace = GROUP_TYPE_REAL(u[0].imag + u[4].imag + u[8].imag) / N_C
    r0 = GROUP_TYPE(1j) * GROUP_TYPE_REAL(u[0].imag - trace)
    r1 = GROUP_TYPE(0.5) * (u[1] - u[3].conjugate())
    r2 = GROUP_TYPE(0.5) * (u[2] - u[6].conjugate())
    r3 = GROUP_TYPE(0.5) * (u[3] - u[1].conjugate())
    r4 = GROUP_TYPE(1j) * GROUP_TYPE_REAL(u[4].imag - trace)
    r5 = GROUP_TYPE(0.5) * (u[5] - u[7].conjugate())
    r6 = GROUP_TYPE(0.5) * (u[6] - u[2].conjugate())
    r7 = GROUP_TYPE(0.5) * (u[7] - u[5].conjugate())
    r8 = GROUP_TYPE(1j) * GROUP_TYPE_REAL(u[8].imag - trace)
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

# group add: g0 = g0 + f * g1
@myjit
def add(g0, g1):
    # Unfortunately, tuple creation from list comprehension does not work in numba:
    # see https://github.com/numba/numba/issues/2771
    #
    # result = tuple(g0[i] + g1[i] for i in range(9))
    # return result
    r0 = g0[0] + g1[0]
    r1 = g0[1] + g1[1]
    r2 = g0[2] + g1[2]
    r3 = g0[3] + g1[3]
    r4 = g0[4] + g1[4]
    r5 = g0[5] + g1[5]
    r6 = g0[6] + g1[6]
    r7 = g0[7] + g1[7]
    r8 = g0[8] + g1[8]
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

# overload the + operator
@myjit
def __add__(g0, g1):
    # Unfortunately, tuple creation from list comprehension does not work in numba:
    # see https://github.com/numba/numba/issues/2771
    #
    # result = tuple(g0[i] + g1[i] for i in range(9))
    # return result
    add(g0,g1)

# multiply by scalar
@myjit
def mul_s(g0, f):
    # Unfortunately, tuple creation from list comprehension does not work in numba:
    # see https://github.com/numba/numba/issues/2771
    #
    # result = tuple(f * g0[i] for i in range(9))
    # return result
    r0 = GROUP_TYPE(f) * g0[0]
    r1 = GROUP_TYPE(f) * g0[1]
    r2 = GROUP_TYPE(f) * g0[2]
    r3 = GROUP_TYPE(f) * g0[3]
    r4 = GROUP_TYPE(f) * g0[4]
    r5 = GROUP_TYPE(f) * g0[5]
    r6 = GROUP_TYPE(f) * g0[6]
    r7 = GROUP_TYPE(f) * g0[7]
    r8 = GROUP_TYPE(f) * g0[8]
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

# conjugate transpose
@myjit
def dagger(a):
    r0 = a[0].conjugate()
    r1 = a[3].conjugate()
    r2 = a[6].conjugate()
    r3 = a[1].conjugate()
    r4 = a[4].conjugate()
    r5 = a[7].conjugate()
    r6 = a[2].conjugate()
    r7 = a[5].conjugate()
    r8 = a[8].conjugate()
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

"""
    Useful functions for temporary fields (setting to zero, unit and addition, ...)
"""

# get group element zero
@myjit
def zero():
    return GROUP_TYPE(0), GROUP_TYPE(0), GROUP_TYPE(0), GROUP_TYPE(0),\
           GROUP_TYPE(0), GROUP_TYPE(0), GROUP_TYPE(0), GROUP_TYPE(0), GROUP_TYPE(0)

# get group element unit
@myjit
def unit():
    return id0

# group store: g0 <- g1
@myjit
def store(g_to, g_from):
    for i in range(9):
        g_to[i] = g_from[i]

# return tuple (local memory)
@myjit
def load(g):
    return g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8]

# trace
@myjit
def tr(a):
    return a[0] + a[4] + a[8]

# trace ( a . dagger(a) )
@myjit
def sq(a): # TODO: rename to tr_sq? or tr_abs_sq?
    """
    >>> a = (1,2,3,4,6,8,9,5,4)
    >>> ta = sq(a)
    >>> ta
    252

    >>> tma = tr(mul(a, dagger(a))).real
    >>> tma
    252

    >>> tma == ta
    True

    :param a:
    :return:
    """
    # return tr(mul(a, dagger(a))).real

    s = GROUP_TYPE_REAL(0)
    for i in range(9):
        s += a[i].real * a[i].real + a[i].imag * a[i].imag
    return s

@myjit
def check_unitary(u):  # TODO: remove debugging code
    x = mul(u, dagger(u))
    d = add(x, mul_s(id0, -1))
    s = sq(d)
    # if s > 1e-8:
    #    print("Unitarity violated")  # TODO: remove debugging code
    return s

"""
    Functions for algebra elements
"""

@myjit
def add_algebra(a, b):
    r0 = a[0] + b[0]
    r1 = a[1] + b[1]
    r2 = a[2] + b[2]
    r3 = a[3] + b[3]
    r4 = a[4] + b[4]
    r5 = a[5] + b[5]
    r6 = a[6] + b[6]
    r7 = a[7] + b[7]
    return r0, r1, r2, r3, r4, r5, r6, r7

@myjit
def mul_algebra(a, f):
    r0 = a[0] * f
    r1 = a[1] * f
    r2 = a[2] * f
    r3 = a[3] * f
    r4 = a[4] * f
    r5 = a[5] * f
    r6 = a[6] * f
    r7 = a[7] * f
    return r0, r1, r2, r3, r4, r5, r6, r7

@myjit
def get_algebra_factors_from_group_element_approximate(g):
    r1 = tr(mul(s1, g)).imag
    r2 = tr(mul(s2, g)).imag
    r3 = tr(mul(s3, g)).imag
    r4 = tr(mul(s4, g)).imag
    r5 = tr(mul(s5, g)).imag
    r6 = tr(mul(s6, g)).imag
    r7 = tr(mul(s7, g)).imag
    r8 = tr(mul(s8, g)).imag
    
    return r1, r2, r3, r4, r5, r6, r7, r8

# @myjit
# def get_gauge_links_color_components(g):
    
#     r1 = tr(mul(s1, g)).real
#     r2 = tr(mul(s2, g)).real
#     r3 = tr(mul(s3, g)).real
#     r4 = tr(mul(s4, g)).real
#     r5 = tr(mul(s5, g)).real
#     r6 = tr(mul(s6, g)).real
#     r7 = tr(mul(s7, g)).real
#     r8 = tr(mul(s8, g)).real
    
#     return r1, r2, r3, r4, r5, r6, r7, r8

@myjit
def proj(g, i, j):
    """
    A helper function for initial_su3.py that only seems to work if I put it here.

    Computes the 'matrix components' of g in the sense of
    proj(g, i, j) = 0.5 * Re(tr( s_i g s_j))
    """
    b = mul(slist[i], mul(g, slist[j]))
    return GROUP_TYPE_REAL(0.5 * tr(b).real)


"""
    DocTest
"""

if __name__ == "__main__":
    import doctest
    doctest.testmod()
