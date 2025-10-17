from curraun.numba_target import myjit
import numba
import curraun.su as su
from curraun.su2_complex import proj
from math import sqrt
from numba import cuda
import numpy as np

"""
    A module that solves the initial conditions for the longitudinal magnetic field for SU(3).
    
    This approach could be generalized to SU(N) if we had a way to compute the inverse of a complex
    n x n matrix in CUDA.
"""


# Parameters for the iterative scheme
ACCURACY_GOAL = 1e-8
ITERATION_MAX_ROUND_1 = 100


@myjit
def init_kernel_2_su2_numba(xi, u0, u1, ua, ub):
    # initialize transverse gauge links (longitudinal magnetic field)
    # (see PhD thesis eq.(2.135))  # TODO: add proper link or reference
    for d in range(2):
        u_a = su.load(ua[xi, d])
        u_b = su.load(ub[xi, d])

        b3, check = solve_initial_numba(u_a, u_b)

        if check > ACCURACY_GOAL:
            print("Kernel xi:", xi, "d: ", d, "did not reach goal. check: ", check)

        su.store(u0[xi, d], b3)
        su.store(u1[xi, d], b3)


@myjit
def init_kernel_2_su2_cuda(xi, u0, u1, ua, ub):
    # initialize transverse gauge links (longitudinal magnetic field)
    # (see PhD thesis eq.(2.135))  # TODO: add proper link or reference
    for d in range(2):
        u_a = su.load(ua[xi, d])
        u_b = su.load(ub[xi, d])

        b3, check = solve_initial_cuda(u_a, u_b)

        if check > ACCURACY_GOAL:
            print("Kernel xi:", xi, "d: ", d, "did not reach goal. check: ", check)

        su.store(u0[xi, d], b3)
        su.store(u1[xi, d], b3)


"""
    New approach: fixed point iteration
"""

@myjit
def solve_initial_cuda(u_a, u_b):
    w = su.add(u_a, u_b)

    # initial condition
    u = su.mul(u_a, u_b)

    # temporary arrays
    Y = cuda.local.array(shape=(3, 3), dtype=numba.float64)
    Yinv = cuda.local.array(shape=(3, 3), dtype=numba.float64)

    # iterative fixed point scheme
    for i in range(ITERATION_MAX_ROUND_1):
        b = su.dagger(su.add(su.unit(), u))
        b = su.mul(w, b)

        # check if equation has been solved
        check = su.sq(su.ah(b))
        if check < ACCURACY_GOAL:
            break

        # compute Y and Y_inv
        y = su.mul(w, su.dagger(u))
        for ia in range(su.ALGEBRA_ELEMENTS):
            for ib in range(su.ALGEBRA_ELEMENTS):
                Y[ia, ib] = proj(y, ia, ib)

        invert3x3(Y, Yinv)

        # temporary array for result
        A = cuda.local.array(shape=3, dtype=numba.float64)

        # extract color components of 'b'
        B = su.get_algebra_factors_from_group_element_approximate(b)

        # solve linear system using inverse
        matvmul(Yinv, B, A, 3)

        # reduce 'largeness' of A if needed
        norm = 0.0
        for j in range(3):
            norm += A[j] ** 2
        norm = sqrt(norm)
        if norm > 1.0:
            for j in range(3):
                A[j] /= norm

        # apply change to 'u' using exp(i*A)
        u = su.mul(su.mexp(su.get_algebra_element(A)), u)

    # final check
    b = su.dagger(su.add(su.unit(), u))
    b = su.mul(w, b)
    check = su.sq(su.ah(b))

    return u, check

@myjit
def solve_initial_numba(u_a, u_b):
    w = su.add(u_a, u_b)

    # initial condition
    u = su.mul(u_a, u_b)

    # temporary arrays
    Y = np.zeros(shape=(3, 3), dtype=su.GROUP_TYPE_REAL)

    # iterative fixed point scheme
    for i in range(ITERATION_MAX_ROUND_1):
        b = su.dagger(su.add(su.unit(), u))
        b = su.mul(w, b)

        # check if equation has been solved
        check = su.sq(su.ah(b))
        if check < ACCURACY_GOAL:
            break

        # compute Y and Y_inv
        y = su.mul(w, su.dagger(u))
        for ia in range(su.ALGEBRA_ELEMENTS):
            for ib in range(su.ALGEBRA_ELEMENTS):
                Y[ia, ib] = proj(y, ia, ib)

        # extract color components of 'b'
        B = su.get_algebra_factors_from_group_element_approximate(b)
        A = np.linalg.solve(Y, B)

        # reduce 'largeness' of A if needed
        norm = np.linalg.norm(A)
        if norm > 1.0:
            A /= norm

        # apply change to 'u' using exp(i*A)
        u = su.mul(su.mexp(su.get_algebra_element(A)), u)

    # final check
    b = su.dagger(su.add(su.unit(), u))
    b = su.mul(w, b)
    check = su.sq(su.ah(b))

    return u, check


@myjit
def invert3x3(a, b):
    """
    Inverts a flattened 3x3 matrix 'a' and puts result into 'b' using an explicit formula.

    This code was adapted from https://stackoverflow.com/a/1148405
    """
    b[0, 0] = a[1,1] * a[2,2] - a[1,2] * a[2,0]
    b[0, 1] = a[0,2] * a[2,1] - a[0,1] * a[2,2]
    b[0, 2] = a[0,1] * a[1,2] - a[0,2] * a[1,1]
    b[1, 0] = a[1,2] * a[2,0] - a[1,0] * a[2,2]
    b[1, 1] = a[0,0] * a[2,2] - a[0,2] * a[2,0]
    b[1, 2] = a[0,2] * a[1,0] - a[0,0] * a[1,2]
    b[2, 0] = a[1,0] * a[2,1] - a[1,1] * a[2,0]
    b[2, 1] = a[0,1] * a[2,0] - a[0,0] * a[2,1]
    b[2, 2] = a[0,0] * a[1,1] - a[0,1] * a[1,0]

    det = a[0,0] * b[0, 0] + a[0,1] * b[1, 0] + a[0,2] * b[2, 0] 

    # could check if det == 0 at this point

    det = 1.0 / det

    for i in range(3):
        for j in range(3):
            b[i, j] *= det


@myjit
def extract_flattened_matrix(A, shift_i, shift_j, m, out):
    for i in range(m):
        for j in range(m):
            out[m * i + j] = A[shift_i + i, shift_j + j]

@myjit
def reinsert_flattened_matrix(A, shift_i, shift_j, m, inp):
    for i in range(m):
        for j in range(m):
            A[i + shift_i, j + shift_j] = inp[m * i + j]

@myjit
def matmul(A, B, C, n):
    for i in range(n):
        for j in range(n):
            C[i, j] = 0.0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

@myjit
def matmul_flat(A, B, C, n):
    for i in range(n):
        for j in range(n):
            C[n * i + j] = 0.0
            for k in range(n):
                C[n * i + j] += A[n * i + k] * B[n * k + j]

@myjit
def vec_add(A, B, C, f, n):
    for i in range(n):
        C[i] = A[i] + f * B[i]

@myjit
def vec_mul(A, B, f, n):
    for i in range(n):
        B[i] = f * A[i]

@myjit
def matvmul(A, x, b, n):
    for i in range(n):
        b[i] = 0.0
        for j in range(n):
            b[i] += A[i, j] * x[j]
