from curraun.numba_target import myjit, my_parallel_loop, use_cuda, mynonparjit
import curraun.su as su
import numpy as np
from numpy.fft import rfft2, irfft2
from numpy import newaxis as na
import math

# Use cupy only if cuda is available
# cupy can be turned off by changing 'use_cupy'
if use_cuda:
    use_cupy = True
    import numba.cuda as cuda
    import cupy
else:
    use_cupy = False

PI = np.pi

from numba import prange

if use_cupy:
    random_cupy = cupy.random.RandomState()
random_np = np.random.RandomState()

# This function can be used to fix seeds. Note that cupy and numpy give different results
# with the same seed.
def set_seed(seed):
    if use_cupy:
        global random_cupy
        random_cupy = cupy.random.RandomState(seed)
    else:
        global random_np
        random_np = np.random.RandomState(seed)


def wilson(s, mu, m, uv, num_sheets, shape_func=None):
    n = s.n
    g = s.g

    # compute poisson kernel
    new_n = (n // 2 + 1) if n % 2 == 0 else (n + 1) // 2
    kernel = np.zeros((n, new_n), dtype=su.GROUP_TYPE_REAL)
    d_kernel = kernel
    if use_cupy:
        d_kernel = cupy.array(kernel)

    my_parallel_loop(wilson_compute_poisson_kernel, n, m, n, new_n, uv, d_kernel)

    # create shape 'mask' array for charge density (this is pretty slow..)
    if shape_func is not None:
        shape_mask = np.zeros((n, n), dtype=su.GROUP_TYPE_REAL)
        for ix in range(n):
            for iy in range(n):
                shape_mask[ix, iy] = shape_func(ix - n // 2, iy - n // 2)

        d_shape_mask = shape_mask
        if use_cupy:
            d_shape_mask = cupy.array(shape_mask)
            d_shape_mask = d_shape_mask.reshape(n * n)

    # initialize wilson lines
    wilsonfield = np.zeros((n ** 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
    d_wilsonfield = wilsonfield
    if use_cupy:
        d_wilsonfield = cuda.to_device(wilsonfield)

    my_parallel_loop(reset_wilsonfield, n ** 2, d_wilsonfield)

    # create color sheets and multiply them
    for sheet in range(num_sheets):
        if use_cupy:
            # generate random color charges
            d_charge = random_cupy.randn(n * n * su.ALGEBRA_ELEMENTS, dtype=np.float32) \
                       * (g ** 2 * mu / math.sqrt(num_sheets))
            d_charge = cupy.reshape(d_charge, (n * n, su.ALGEBRA_ELEMENTS))
            # apply shape mask
            if shape_func is not None:
                my_parallel_loop(modulate_kernel, n ** 2, d_charge, su.ALGEBRA_ELEMENTS, d_shape_mask)

            # fourier transform charge density
            d_charge = cupy.reshape(d_charge, (n, n, su.ALGEBRA_ELEMENTS))
            d_field_fft = cupy.fft.rfft2(d_charge, axes=(0, 1))

            # apply poisson kernel
            d_kernel = cupy.reshape(d_kernel, n * new_n)
            d_field_fft = cupy.reshape(d_field_fft, (n * new_n, su.ALGEBRA_ELEMENTS))
            my_parallel_loop(modulate_kernel, n * new_n, d_field_fft, su.ALGEBRA_ELEMENTS, d_kernel)
            d_field_fft = cupy.reshape(d_field_fft, (n, new_n, su.ALGEBRA_ELEMENTS))

            # fourier transform back
            d_field = cupy.fft.irfft2(d_field_fft, axes=(0, 1), s=(n, n))
            d_field = cupy.reshape(d_field, (n * n, su.ALGEBRA_ELEMENTS))

            # exponentiate and multiply with previous sheets
            my_parallel_loop(wilson_exponentiation_kernel, n ** 2, d_field, d_wilsonfield)

        else:
            # generate random color charges
            field = random_np.normal(loc=0.0, scale=g ** 2 * mu / math.sqrt(num_sheets),
                                     size=(n ** 2 * su.ALGEBRA_ELEMENTS))
            field = field.reshape((n * n, su.ALGEBRA_ELEMENTS))

            # apply shape mask
            if shape_func is not None:
                field = field.reshape((n, n, su.ALGEBRA_ELEMENTS))
                field = field[:, :, :] * d_shape_mask[:, :, None]
                field = field.reshape((n * n, su.ALGEBRA_ELEMENTS))

            # fourier transform charge density
            # apply poisson kernel
            # fourier transform back
            field = irfft2(
                rfft2(field.reshape((n, n, su.ALGEBRA_ELEMENTS)), s=(n, n), axes=(0, 1)) * kernel[:, :, na],
                s=(n, n), axes=(0, 1)
            ).reshape((n ** 2, su.ALGEBRA_ELEMENTS))

            # exponentiate and multiply with previous sheets
            my_parallel_loop(wilson_exponentiation_kernel, n ** 2, field, d_wilsonfield)

    if use_cupy:
        d_wilsonfield.copy_to_host(wilsonfield)

    return wilsonfield


"""
    Kernels
"""


@myjit
def modulate_kernel(xi, field, num, kernel):
    for i in range(num):
        field[xi, i] = field[xi, i] * kernel[xi]

# @myjit
@mynonparjit
def reset_wilsonfield(x, wilsonfield):
    su.store(wilsonfield[x], su.unit())

@myjit
def wilson_compute_poisson_kernel(x, mass, n, new_n, uv, kernel):
    # for y in range(new_n):
    for y in prange(new_n):
        k2 = k2_latt(x, y, n)
        if (x > 0 or y > 0) and k2 <= uv ** 2:
            kernel[x, y] = 1.0 / (k2 + mass ** 2)

# @myjit
@mynonparjit
def wilson_exponentiation_kernel(x, field, wilsonfield):
    a = su.get_algebra_element(field[x])
    buffer1 = su.mexp(a)
    buffer2 = su.mul(buffer1, wilsonfield[x])
    su.store(wilsonfield[x], buffer2)

# @myjit
@mynonparjit
def k2_latt(x, y, nt):
    result = 4.0 * (math.sin((PI * x) / nt) ** 2 + math.sin((PI * y) / nt) ** 2)
    return result
