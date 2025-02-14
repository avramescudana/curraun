from curraun.numba_target import myjit, mynonparjit
import numba
import curraun.su as su
from curraun.su3 import proj
from math import sqrt
from numba import cuda
import numpy as np

from numba import prange

"""
    A module that solves the initial conditions for the longitudinal magnetic field for SU(3).
    
    This approach could be generalized to SU(N) if we had a way to compute the inverse of a complex
    n x n matrix in CUDA.
"""


# Parameters for the iterative scheme
ACCURACY_GOAL = 1e-16
ITERATION_MAX_ROUND_1 = 100


@myjit
def init_kernel_2_su3_numba(xi, u0, u1, ua, ub):
    # initialize transverse gauge links (longitudinal magnetic field)
    # (see PhD thesis eq.(2.135))  # TODO: add proper link or reference
    # for d in range(2):
    for d in prange(2):
        u_a = su.load(ua[xi, d])
        u_b = su.load(ub[xi, d])

        b3, check = solve_initial_numba(u_a, u_b)

        # if check > ACCURACY_GOAL:
        #     print("Kernel xi:", xi, "d: ", d, "did not reach goal. check: ", check)

        su.store(u0[xi, d], b3)
        su.store(u1[xi, d], b3)


@myjit
def init_kernel_2_su3_cuda(xi, u0, u1, ua, ub):
    # initialize transverse gauge links (longitudinal magnetic field)
    # (see PhD thesis eq.(2.135))  # TODO: add proper link or reference
    for d in range(2):
        u_a = su.load(ua[xi, d])
        u_b = su.load(ub[xi, d])

        b3, check = solve_initial_cuda(u_a, u_b)

        # if check > ACCURACY_GOAL:
        #     print("Kernel xi:", xi, "d: ", d, "did not reach goal. check: ", check)

        su.store(u0[xi, d], b3)
        su.store(u1[xi, d], b3)


"""
    New approach: fixed point iteration
"""

# @myjit
@mynonparjit
def solve_initial_cuda(u_a, u_b):
    w = su.add(u_a, u_b)

    # initial condition
    u = su.mul(u_a, u_b)

    # temporary arrays
    Y = cuda.local.array(shape=(8, 8), dtype=numba.float64)
    Yinv = cuda.local.array(shape=(8, 8), dtype=numba.float64)

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

        invert8x8(Y, Yinv)

        # temporary array for result
        A = cuda.local.array(shape=8, dtype=numba.float64)

        # extract color components of 'b'
        B = su.get_algebra_factors_from_group_element_approximate(b)

        # solve linear system using inverse
        matvmul(Yinv, B, A, 8)

        # reduce 'largeness' of A if needed
        norm = 0.0
        for j in range(8):
            norm += A[j] ** 2
        norm = sqrt(norm)
        if norm > 1.0:
            for j in range(8):
                A[j] /= norm

        # apply change to 'u' using exp(i*A)
        u = su.mul(su.mexp(su.get_algebra_element(A)), u)

    # final check
    b = su.dagger(su.add(su.unit(), u))
    b = su.mul(w, b)
    check = su.sq(su.ah(b))

    return u, check

# @myjit
@mynonparjit
def solve_initial_numba(u_a, u_b):
    w = su.add(u_a, u_b)

    # initial condition
    u = su.mul(u_a, u_b)

    # temporary arrays
    Y = np.zeros(shape=(8, 8), dtype=su.GROUP_TYPE_REAL)

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

        # fix error from numba
        # A = np.linalg.solve(Y, B)
        B_array = np.array(B)
        A = np.linalg.solve(Y, B_array)

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
def invert4x4(a, b):
    """
    Inverts a flattened 4x4 matrix 'a' and puts result into 'b' using an explicit formula.

    This code was adapted from https://stackoverflow.com/a/1148405
    """
    b[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[
        11] - a[13] * a[7] * a[10]

    b[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * \
           a[11] + a[12] * a[7] * a[10]

    b[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[
        11] - a[12] * a[7] * a[9]

    b[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * \
            a[10] + a[12] * a[6] * a[9]

    b[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * \
           a[11] + a[13] * a[3] * a[10]

    b[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[
        11] - a[12] * a[3] * a[10]

    b[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[
        11] + a[12] * a[3] * a[9]

    b[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[
        10] - a[12] * a[2] * a[9]

    b[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[
        7] - a[13] * a[3] * a[6]

    b[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[
        7] + a[12] * a[3] * a[6]

    b[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[
        7] - a[12] * a[3] * a[5]

    b[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[
        6] + a[12] * a[2] * a[5]

    b[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[
        7] + a[9] * a[3] * a[6]

    b[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[
        7] - a[8] * a[3] * a[6]

    b[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[
        7] + a[8] * a[3] * a[5]

    b[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - \
            a[8] * a[2] * a[5]

    det = a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12]

    # could check if det == 0 at this point

    det = 1.0 / det

    for i in range(16):
        b[i] *= det

@myjit
def invert8x8(M, Inv):
    """
    Inverts a 8x8 matrix 'M' and puts result into 'Inv' using the block matrix formula.
    """
    # allocate temporary flattned 4x4 matrices
    A = cuda.local.array(shape=4*4, dtype=numba.float64)
    B = cuda.local.array(shape=4*4, dtype=numba.float64)
    C = cuda.local.array(shape=4*4, dtype=numba.float64)
    D = cuda.local.array(shape=4*4, dtype=numba.float64)

    Ai = cuda.local.array(shape=4*4, dtype=numba.float64)
    Bi = cuda.local.array(shape=4*4, dtype=numba.float64)
    Ci = cuda.local.array(shape=4*4, dtype=numba.float64)
    Di = cuda.local.array(shape=4*4, dtype=numba.float64)

    # extract block matrices
    extract_flattened_matrix(M, 0, 0, 4, A)
    extract_flattened_matrix(M, 0, 4, 4, B)
    extract_flattened_matrix(M, 4, 0, 4, C)
    extract_flattened_matrix(M, 4, 4, 4, D)

    # allocate buffers
    buff1 = cuda.local.array(shape=4*4, dtype=numba.float64)
    buff2 = cuda.local.array(shape=4*4, dtype=numba.float64)

    # block 'A'
    invert4x4(D, buff1)
    matmul_flat(B, buff1, buff2, 4)
    matmul_flat(buff2, C, buff1, 4)
    vec_add(A, buff1, buff2, -1, 16)
    invert4x4(buff2, Ai)
    reinsert_flattened_matrix(Inv, 0, 0, 4, Ai)

    # block 'B'
    invert4x4(D, buff1)
    matmul_flat(B, buff1, buff2, 4)
    matmul_flat(Ai, buff2, buff1, 4)
    vec_mul(buff1, Bi, -1, 16)
    reinsert_flattened_matrix(Inv, 0, 4, 4, Bi)

    # block 'C'
    invert4x4(D, buff1)
    matmul_flat(buff1, C, buff2, 4)
    matmul_flat(buff2, Ai, buff1, 4)
    vec_mul(buff1, Ci, -1, 16)
    reinsert_flattened_matrix(Inv, 4, 0, 4, Ci)

    # block 'D'
    invert4x4(D, Bi)
    matmul_flat(Bi, C, buff1, 4)
    matmul_flat(buff1, Ai, buff2, 4)
    matmul_flat(buff2, B, buff1, 4)
    matmul_flat(buff1, Bi, buff2, 4)
    vec_add(Bi, buff2, Di, 1, 16)
    reinsert_flattened_matrix(Inv, 4, 4, 4, Di)


"""
    A few matrix and vector operations
"""

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
