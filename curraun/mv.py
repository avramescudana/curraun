from curraun.numba_target import myjit, my_parallel_loop
import curraun.su as su
import curraun.lattice as l
import numpy as np
from numpy.fft import rfft2, irfft2
from numpy import newaxis as na
import math

PI = np.pi

def wilson(s, mu, m, uv, num_sheets, shape_func=None):
    n = s.n
    g = s.g

    # compute poisson kernel
    kernel = np.zeros((n, n // 2 + 1), dtype=np.double)
    my_parallel_loop(wilson_compute_poisson_kernel, n, m, n, uv, kernel)

    # iterate over number of sheets
    # field = np.zeros((n ** 2, 3), dtype=np.double)

    # TODO: Keep arrays on GPU device
    wilsonfield = np.empty((n ** 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
    my_parallel_loop(reset_wilsonfield, n ** 2, wilsonfield)

    for sheet in range(num_sheets):
        # solve poisson equation
        field = np.random.normal(loc=0.0, scale=g ** 2 * mu / math.sqrt(num_sheets), size=(n ** 2, su.ALGEBRA_ELEMENTS))

        # Apply shape function to charge density
        if shape_func is not None:
            for ix in range(n):
                for iy in range(n):
                    index = l.get_index_nm(ix, iy)
                    field[index, :] *= shape_func(ix - n // 2, iy - n // 2)

        if su.N_C > 3:
            print("mv.py: SU(N) code not implemented")
            exit()

        # TODO: Use FFT functions from Cupy: https://docs-cupy.chainer.org/en/stable/reference/fft.html
        field = irfft2(
                rfft2(field.reshape((n, n, su.ALGEBRA_ELEMENTS)), s=(n, n), axes=(0, 1)) * kernel[:, :, na],
                s=(n, n), axes=(0, 1)
            ).reshape((n ** 2, su.ALGEBRA_ELEMENTS))

        my_parallel_loop(wilson_exponentiation_kernel, n ** 2, field, wilsonfield)

    return wilsonfield

@myjit
def reset_wilsonfield(x, wilsonfield):
    su.store(wilsonfield[x], su.unit())

@myjit
def wilson_compute_poisson_kernel(x, m, n, uv, kernel):
    # compute poisson kernel
    for y in range(n // 2 + 1):
        k2 = k2_latt(x, y, n)
        if (x > 0 or y > 0) and k2 <= uv ** 2:
            kernel[x, y] = 1.0 / (k2 + m ** 2)

@myjit
def wilson_exponentiation_kernel(x, field, wilsonfield):
    a = su.get_algebra_element(field[x])
    buffer1 = su.mexp(a)
    buffer2 = su.mul(buffer1, wilsonfield[x])
    su.store(wilsonfield[x], buffer2)

@myjit
def k2_latt(x, y, nt):
    result = 4.0 * (math.sin((PI * x) / nt) ** 2 + math.sin((PI * y) / nt) ** 2)
    return result