from curraun.numba_target import myjit, my_parallel_loop, use_cuda, mynonparjit
import numpy as np
import curraun.lattice as l
import curraun.su as su
import os
import math

from numpy.fft import rfft2, irfft2
from numpy import newaxis as na

from numba import prange

PI = np.pi

if use_cuda:
    print("\nuse_cuda =", use_cuda, "\nGo ahead and import CuPy.")
    use_cupy = True
    import numba.cuda as cuda
    import cupy

    random_cupy = cupy.random.RandomState()

    print("\nuse_cupy =", use_cupy)
    print("\nCuPy has been imported successfully.")
    print("\nCuPy random number generator initialized.\n")

else:
    print("\nuse_cuda =", use_cuda, "\nCuPy will not be imported.")
    use_cupy = False
    print("use_cupy =", use_cupy, "\nCuPy is not imported.\n")
    random_np = np.random.RandomState()
    print("\nNumPy random number generator initialized.\n")

# Set precision
su_precision = os.environ.get('PRECISION', 'double')
if use_cupy:
    DTYPE = cupy.float32 if su_precision == 'single' else cupy.float64
else:
    DTYPE = np.float32 if su_precision == 'single' else np.float64


class Lyapunov():
    def __init__(self, s, sprime):
        self.s = s
        self.sprime = sprime
        n = self.s.n

        self.tr_sq_el = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.tr_sq_dif = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.d_tr_sq_el = self.tr_sq_el
        self.d_tr_sq_dif = self.tr_sq_dif
        self.ratio_dif = 0.0

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_tr_sq_el = cuda.to_device(self.tr_sq_el)
        self.d_tr_sq_dif = cuda.to_device(self.tr_sq_dif)

    def copy_to_host(self):
        self.d_tr_sq_el.copy_to_host(self.tr_sq_el)
        self.d_tr_sq_dif.copy_to_host(self.tr_sq_dif)

    def change_EL(self, alpha, m_noise):
        peta1 = self.sprime.d_peta1
        n = self.sprime.n
        noise_n = (n // 2 + 1) if n % 2 == 0 else (n + 1) // 2

        if use_cupy:
            eta = random_cupy.normal(loc=0.0, scale=alpha, size=(n, n, su.GROUP_ELEMENTS)).astype(DTYPE)
            noise_kernel = cupy.zeros((n, noise_n), dtype=su.GROUP_TYPE_REAL)
            my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)

            for i in range(su.GROUP_ELEMENTS):
                fft_eta = cupy.fft.rfft2(eta[:, :, i])
                fft_eta *= noise_kernel
                eta[:, :, i] = cupy.fft.irfft2(fft_eta, s=(n, n))

            eta = cupy.reshape(eta, (n ** 2, su.GROUP_ELEMENTS))
            eta = cupy.asnumpy(eta)  # convert back to host for CPU kernel

        else:
            eta = random_np.normal(loc=0.0, scale=alpha, size=(n, n, su.GROUP_ELEMENTS)).astype(DTYPE)
            noise_kernel = np.zeros((n, noise_n), dtype=su.GROUP_TYPE_REAL)
            my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)

            for i in range(su.GROUP_ELEMENTS):
                fft_eta = rfft2(eta[:, :, i])
                fft_eta *= noise_kernel
                eta[:, :, i] = irfft2(fft_eta, s=(n, n))

            eta = eta.reshape((n ** 2, su.GROUP_ELEMENTS))

        my_parallel_loop(change_EL_kernel, n ** 2, peta1, eta)

    def compute_change_EL(self):
        peta1s = self.s.d_peta1
        peta1sprime = self.sprime.d_peta1
        n = self.s.n

        my_parallel_loop(compute_change_EL_kernel, n ** 2, peta1s, peta1sprime, self.d_tr_sq_el, self.d_tr_sq_dif)

        if use_cuda:
            self.copy_to_host()

        dif_avg = np.mean(self.d_tr_sq_dif)
        el_avg = np.mean(self.d_tr_sq_el)

        self.ratio_dif = dif_avg



@mynonparjit
def compute_change_EL_kernel(xi, peta1s, peta1sprime, tr_sq_el, tr_sq_dif):
    buf1 = l.add_mul(peta1sprime[xi], peta1s[xi], -1)
    tr_sq_dif[xi] = su.sq(buf1)
    tr_sq_el[xi] = su.sq(peta1s[xi])


@myjit
def compute_noise_kernel(x, mass, n, new_n, kernel):
    for y in prange(new_n):
        k2 = k2_latt(x, y, n)
        if x > 0 or y > 0:
            kernel[x, y] = mass ** 2 / (k2 + mass ** 2)


@mynonparjit
def k2_latt(x, y, nt):
    return 4.0 * (math.sin((PI * x) / nt) ** 2 + math.sin((PI * y) / nt) ** 2)




@mynonparjit
def change_EL_kernel(xi, peta1, eta):
    buf1 = su.add(peta1[xi], eta[xi])
    peta1[xi] = buf1
