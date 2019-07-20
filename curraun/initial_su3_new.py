from curraun.numba_target import myjit
import numba
import curraun.su as su
from curraun.su3 import proj
from math import sqrt
from numba import cuda

"""
    A module that solves the initial conditions for the longitudinal magnetic field for SU(3).
    
    This approach could be generalized to SU(N) if we had a way to compute the inverse of a complex
    n x n matrix in CUDA.
    
    Also, this code only works for CUDA right now.
"""


# Parameters for the iterative scheme
ACCURACY_GOAL = 1e-16
ITERATION_MAX_ROUND_1 = 100


@myjit
def init_kernel_2_su3(xi, u0, u1, ua, ub):
    # initialize transverse gauge links (longitudinal magnetic field)
    # (see PhD thesis eq.(2.135))  # TODO: add proper link or reference
    for d in range(2):
        u_a = su.load(ua[xi, d])
        u_b = su.load(ub[xi, d])

        b3, check = solve_initial(u_a, u_b)

        if check > ACCURACY_GOAL:
            print("Kernel xi:", xi, "d: ", d, "did not reach goal. check: ", check)
            pass

        su.store(u0[xi, d], b3)
        su.store(u1[xi, d], b3)


"""
    New approach: fixed point iteration
"""
@myjit
def solve_initial(u_a, u_b):
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


@myjit
def invert4x4(m, inv):
    """
    Inverts a flattened 4x4 matrix 'm' and puts result into 'inv' using an explicit formula.
    """
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[
        11] - m[13] * m[7] * m[10]

    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * \
             m[11] + m[12] * m[7] * m[10]

    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[
        11] - m[12] * m[7] * m[9]

    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * \
              m[10] + m[12] * m[6] * m[9]

    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * \
             m[11] + m[13] * m[3] * m[10]

    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[
        11] - m[12] * m[3] * m[10]

    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[
        11] + m[12] * m[3] * m[9]

    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[
        10] - m[12] * m[2] * m[9]

    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[
        7] - m[13] * m[3] * m[6]

    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[
        7] + m[12] * m[3] * m[6]

    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[
        7] - m[12] * m[3] * m[5]

    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[
        6] + m[12] * m[2] * m[5]

    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[
        7] + m[9] * m[3] * m[6]

    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[
        7] - m[8] * m[3] * m[6]

    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[
        7] + m[8] * m[3] * m[5]

    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - \
              m[8] * m[2] * m[5]

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12]

    det = 1.0 / det

    for i in range(16):
        inv[i] *= det

@myjit
def invert8x8(m, inv):
    """
    Inverts a 8x8 matrix 'm' and puts result into 'inv' using the block matrix formula.
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
    extract_flattened_matrix(m, 0, 0, 4, A)
    extract_flattened_matrix(m, 0, 4, 4, B)
    extract_flattened_matrix(m, 4, 0, 4, C)
    extract_flattened_matrix(m, 4, 4, 4, D)

    # allocate buffers
    buff1 = cuda.local.array(shape=4*4, dtype=numba.float64)
    buff2 = cuda.local.array(shape=4*4, dtype=numba.float64)

    # block 'A'
    invert4x4(D, buff1)
    matmul_flat(B, buff1, buff2, 4)
    matmul_flat(buff2, C, buff1, 4)
    vec_add(A, buff1, buff2, -1, 16)
    invert4x4(buff2, Ai)
    reinsert_flattened_matrix(inv, 0, 0, 4, Ai)

    # block 'B'
    invert4x4(D, buff1)
    matmul_flat(B, buff1, buff2, 4)
    matmul_flat(Ai, buff2, buff1, 4)
    vec_mul(buff1, Bi, -1, 16)
    reinsert_flattened_matrix(inv, 0, 4, 4, Bi)

    # block 'C'
    invert4x4(D, buff1)
    matmul_flat(buff1, C, buff2, 4)
    matmul_flat(buff2, Ai, buff1, 4)
    vec_mul(buff1, Ci, -1, 16)
    reinsert_flattened_matrix(inv, 4, 0, 4, Ci)

    # block 'D'
    invert4x4(D, Bi)
    matmul_flat(Bi, C, buff1, 4)
    matmul_flat(buff1, Ai, buff2, 4)
    matmul_flat(buff2, B, buff1, 4)
    matmul_flat(buff1, Bi, buff2, 4)
    vec_add(Bi, buff2, Di, 1, 16)
    reinsert_flattened_matrix(inv, 4, 4, 4, Di)


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
