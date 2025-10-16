"""
    SU(2) group and algebra functions
"""
import os
from Modules.numba_target import myjit

import math
import numpy as np

# Definition of constants

N_C = 2  # Number of colors
ALGEBRA_ELEMENTS = 3
GROUP_ELEMENTS = 4
CASIMIRS = 1

zero_algebra = (0,0,0)

unit_algebra = ((1,0,0),
                (0,1,0),
                (0,0,1))


su_precision = os.environ.get('PRECISION', 'double')

if su_precision == 'single':
    # TODO: convert all input variables to float32 before compiled functions are called
    #       (check this using compiled_function.inspect_types())
    #print("Using single precision")
    GROUP_TYPE = np.complex64 # two float32
    GROUP_TYPE_REAL = np.float32
elif su_precision == 'double':
    #print("Using double precision")
    GROUP_TYPE = np.complex128 # two float64
    GROUP_TYPE_REAL = np.float64
else:
    print("Unsupported precision: " + su_precision)

EXP_MIN_TERMS = -1 # minimum number of terms in Taylor series
EXP_MAX_TERMS = 100 # maximum number of terms in Taylor series
EXP_ACCURACY_SQUARED = 1.e-40 # 1.e-32 # accuracy

def complex_tuple(*t):
    return tuple(map(GROUP_TYPE, t))

# Pauli matrices
id0 = complex_tuple(1, 0, 0, 1)
s1 = complex_tuple(0, 1, 1, 0)
s2 = complex_tuple(0, -1j, 1j, 0)
s3 = complex_tuple(1, 0, 0, -1)

slist = (id0, s1, s2, s3)

@myjit
def get_algebra_element(algebra_factors):
    """
    Algebra elements are formed from the algebra factors by:
    U = i t_a A_a = i sigma_a A_a / 2 ... for sigma_a the Pauli matrices.

    The exponential map is given by mexp(get_algebra_element(algebra_factors))

    >>> get_algebra_element((1,0,0))
    (0j, 0.5j, 0.5j, 0j)

    :param a:
    :return:
    """
    ms0 = mul_s(s1, algebra_factors[0])
    ms1 = mul_s(s2, algebra_factors[1])
    ms2 = mul_s(s3, algebra_factors[2])

    # Add all
    b0 = add(ms0, ms1)
    b1 = add(b0, ms2)

    # Overall factor
    res = mul_s(b1, 0.5j)
    return res


"""
    SU(2) group & algebra functions
"""

# SU(2) group elements are given by 2x2 complex matrices:
#   a[0] a[1] 
#   a[2] a[3] 
# (layout corresponds to C-order of numpy array)

# su2 multiplication

# Generate the matrix multiplication code:
# print("\n".join(["r{} = ".format(j) + " + ".join(["a[{}] * b[{}]".format(2 * (j // 2) + i, 2 * i + (j % 2)) for i in range(2)]) for j in range(4)]))

@myjit
def mul(a, b):
    """SU(2) multiplication: 2x2 matrix multiplication

    >>> a=[1, 2, 3, 4]
    >>> b=[5, 6, 7, 8]
    >>> c=mul(a,b)
    >>> c
    (19, 22, 43, 50)

    # Check with numpy
    >>> import numpy as np
    >>> ma = np.asarray(a).reshape(2,2)
    >>> mb = np.asarray(b).reshape(2,2)
    >>> mc = np.matmul(ma, mb)
    >>> mc
    array([[19, 22],
           [43, 50]])

    >>> tuple(mc.flatten()) == c
    True
    """
    r0 = a[0] * b[0] + a[1] * b[2] 
    r1 = a[0] * b[1] + a[1] * b[3] 
    r2 = a[2] * b[0] + a[3] * b[2] 
    r3 = a[2] * b[1] + a[3] * b[3] 
    return r0, r1, r2, r3
# exponential map
@myjit
def mexp(a):
    """ Calculate exponential using Taylor series

    mexp(a) = 1 + a + a^2 / 2 + a^3 / 6 + ...

    >>> a = id0
    >>> mexp(a)
    ((2.7182818284590455+0j), 0j, 0j, (2.7182818284590455+0j))
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
@myjit
def mlog(a):
    """
    Computes logarithm of a matrix using Taylor series

    mlog(a) = (a-id0) - (a-id0)^2 / 2 + (a-id0)^3 / 3 - (a-id0)^4 / 4 + ...

    Works for matrices close to identity (for example gauge links)
    Example and comparison with logm() from Scipy:

    >>> Ux = [0.98510928-0.15514118j, 0.06072216+0.04247033j, -0.05816349+0.04591208j, 0.99243458+0.09789123j]
    >>> lnUx_CUDA = mlog(Ux)
    >>> lnUx_CUDA
    ((-5.283685708374216e-09-0.1559693702453399j), (0.05968301710568872+0.04436976704847219j), (-0.059683020380762314+0.044369774147350945j), (-6.818738708510321e-10+0.0980854807916348j))
    >>> from scipy.linalg import logm
    >>> lnUx_Scipy = np.reshape(logm(np.reshape(Ux, (2, 2))), 2*2)
    >>> lnUx_Scipy
    array([-5.28368542e-09-0.15596937j,  5.96830171e-02+0.04436977j,
           -5.96830204e-02+0.04436977j, -6.81873818e-10+0.09808548j])
    >>> lnUx_CUDA - lnUx_Scipy
    array([-2.91666427e-16+0.00000000e+00j, -5.55111512e-17-6.93889390e-17j,
           4.85722573e-17+4.85722573e-17j, -5.30250034e-17-9.71445147e-17j])
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
    (0j, (2.7182818284590455+0j), (2.7182818284590455+0j), 0j)
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
    >>> u = (1, 2, 0, 4)
    >>> u
    (1, 2, 0, 4)

    >>> inv(u)
    (1.0, -0.5, -0.0, 0.25)

    >>> b = np.asarray(u)
    >>> c = b.reshape(2, 2)
    >>> d = np.linalg.inv(c)
    >>> e = d.reshape(4)
    >>> e
    array([ 1.  , -0.5 ,  0.  ,  0.25])

    :param a:
    :return:
    """
    d = det(a)

    if abs(d) < 1.e-18:
        print("Determinant too small")  # TODO: remove debugging code

    det_inv = 1 / d  # TODO: What to do if det == 0 ?

    r0 = det_inv * a[3]
    r1 = - det_inv * a[1]
    r2 = - det_inv * a[2]
    r3 = det_inv * a[0]
    return r0, r1, r2, r3

# determinant
@myjit
def det(a):
    res  = a[0] * a[3] - a[1] * a[2]
    return res

# anti-hermitian part
@myjit
def ah(u):
    """
    >>> u = get_algebra_element((1,0,0))
    >>> u
    (0j, 0.5j, 0.5j, 0j)

    >>> ah(u)
    (0j, 0.5j, 0.5j, 0j)

    Calculate definition of anti-hermitian part:
    # (see PhD thesis eq.(2.119))  # TODO: add proper link or reference
    >>> udiff = add(u, mul_s(dagger(u), -1))
    >>> unit = id0
    >>> tr_udiff = tr(udiff)
    >>> Uah_term1 = mul_s(udiff, 1/2)
    >>> Uah_term2 = mul_s(unit, tr_udiff / (2 * N_C))
    >>> Uah = add(Uah_term1, mul_s(Uah_term2, -1))
    >>> Uah
    (0j, 0.5j, 0.5j, 0j)

    >>> ah(u) == Uah
    True

    :param u:
    :return:
    """
    trace = GROUP_TYPE_REAL(u[0].imag + u[3].imag) / N_C
    r0 = GROUP_TYPE(1j) * GROUP_TYPE_REAL(u[0].imag - trace)
    r1 = GROUP_TYPE(0.5) * (u[1] - u[2].conjugate())
    r2 = GROUP_TYPE(0.5) * (u[2] - u[1].conjugate())
    r3 = GROUP_TYPE(1j) * GROUP_TYPE_REAL(u[3].imag - trace)
    return r0, r1, r2, r3

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
    return r0, r1, r2, r3

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
    return r0, r1, r2, r3

# conjugate transpose
@myjit
def dagger(a):
    r0 = a[0].conjugate()
    r1 = a[2].conjugate()
    r2 = a[1].conjugate()
    r3 = a[3].conjugate()
    return r0, r1, r2, r3

"""
    Useful functions for temporary fields (setting to zero, unit and addition, ...)
"""

# get group element zero
@myjit
def zero():
    return GROUP_TYPE(0), GROUP_TYPE(0), GROUP_TYPE(0), GROUP_TYPE(0)

@myjit
def zero_alg():
    return zero_algebra

# get group element unit
@myjit
def unit():
    return id0

# group store: g0 <- g1
@myjit
def store(g_to, g_from):
    for i in range(4):
        g_to[i] = g_from[i]

# return tuple (local memory)
@myjit
def load(g):
    return g[0], g[1], g[2], g[3]

# trace
@myjit
def tr(a):
    return a[0] + a[3]

# trace ( a . dagger(a) )
@myjit
def sq(a): # TODO: rename to tr_sq? or tr_abs_sq?
    """
    >>> a = (1,2,3,4)
    >>> ta = sq(a)
    >>> ta
    30.0

    >>> tma = tr(mul(a, dagger(a))).real
    >>> tma
    30

    >>> tma == ta
    True

    :param a:
    :return:
    """
    # return tr(mul(a, dagger(a))).real

    s = GROUP_TYPE_REAL(0)
    for i in range(4):
        s += a[i].real * a[i].real + a[i].imag * a[i].imag
    return s


# I ADDED THIS
@myjit
def square(a, b):

    s = GROUP_TYPE_REAL(0)
    for i in range(4):
        s += a[i].real * b[i].real + a[i].imag * b[i].imag
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
    return r0, r1, r2

@myjit
def mul_algebra(a, f):
    r0 = a[0] * f
    r1 = a[1] * f
    r2 = a[2] * f
    return r0, r1, r2

@myjit
def get_algebra_factors_from_group_element_approximate(g):
    r1 = tr(mul(s1, g)).imag
    r2 = tr(mul(s2, g)).imag
    r3 = tr(mul(s3, g)).imag
    return r1, r2, r3

@myjit
def get_algebra_from_group_element(g):
    r1 = tr(mul(s1, g))
    r2 = tr(mul(s2, g))
    r3 = tr(mul(s3, g))
    return r1, r2, r3

@myjit
def proj(g, i, j):
    """
    A helper function for initial_su2.py that only seems to work if I put it here.

    Computes the 'matrix components' of g in the sense of
    proj(g, i, j) = 0.5 * Re(tr( s_i g s_j))
    """
    b = mul(slist[i], mul(g, slist[j]))
    return GROUP_TYPE_REAL(0.5 * tr(b).real)

@myjit
def casimir(Q):
    """
    Computes the quadratic Casimir C_2. 
    Notice that Tr{Q^2}=T(R)C_2 with T(R)=1/2 for R=F.
    """
    C = sq(Q).real * 2 / N_C
    return C


"""
    DocTest
"""

if __name__ == "__main__":
    import doctest
    doctest.testmod()

