"""
    Module for evaluating correlators of the Lorentz force along the trajectory of the particle
"""

from curraun.numba_target import use_cuda, myjit, my_parallel_loop
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda
import numpy as np

class ForceCorrelators:
    def __init__(self, s, wong, n_particles):
        self.s = s
        self.wong = wong
        self.n_particles = n_particles

        # color lorentz force for each particle
        # force layouy (f): x,y,eta
        self.f = np.zeros((n_particles, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # lorentz force correlator
        self.corr = np.zeros((n_particles, 3), dtype=su.GROUP_TYPE)

        # lorentz force autocorrelator
        # self.auto_corr = np.zeros((n_particles, 3), dtype=su.GROUP_TYPE)

        # set-up device pointers
        self.d_f = self.f
        self.d_corr = self.corr
        # self.d_auto_corr = self.auto_corr

        # move data to GPU
        if use_cuda:
            self.copy_to_device()


    def copy_to_device(self):
        self.d_f = cuda.to_device(self.f)
        self.d_corr = cuda.to_device(self.corr)
        # self.d_auto_corr = cuda.to_device(self.auto_corr)

    def copy_to_host(self):
        self.d_f.copy_to_host(self.f)
        self.d_corr.copy_to_host(self.corr)
        # self.d_auto_corr.copy_to_host(self.auto_corr)


    def compute_lorentz_force(self):

        my_parallel_loop(compute_lorentz_force_kernel, self.n_particles, self.d_f, self.wong.x0, self.wong.p, self.s.d_pt0, 
        self.s.d_pt1, self.s.d_u0, self.s.peta0, self.s.peta1, self.s.aeta0, self.s.n, self.s.t)

        if use_cuda:
            self.copy_to_host()

    def compute_lorentz_force_correlator(self, f0, f, w):
         my_parallel_loop(compute_force_correlator_kernel, self.n_particles, f0, f, w, self.d_corr)

         if use_cuda:
            self.copy_to_host()
    
    # def compute_lorentz_force_autocorrelator(self, f0, f1, w0, w1):
    #      my_parallel_loop(compute_force_autocorrelator_kernel, self.n_particles, f0, f1, w0, w1, self.d_auto_corr)

    #      if use_cuda:
    #         self.copy_to_host()


@myjit
def ngp(x):
    """
        Computes the nearest grid point of a particle position
    """
    return int(round(x[0])), int(round(x[1]))

@myjit
def compute_electric_field(ngp_index, pt0, pt1, u0, peta0, peta1, n, t):
    # Electric fields
    Ex = su.zero()
    Ex = su.add(Ex, pt1[ngp_index, 0, :])
    Ex = su.add(Ex, pt0[ngp_index, 0, :])
    xs = l.shift(ngp_index, 0, -1, n)
    b1 = l.act(su.dagger(u0[xs, 0, :]), pt1[xs, 0, :])
    Ex = su.add(Ex, b1)
    b1 = l.act(su.dagger(u0[xs, 0]), pt0[xs, 0, :])
    Ex = su.add(Ex, b1)
    Ex = su.mul_s(Ex, 0.25 / t)

    Ey = su.zero()
    Ey = su.add(Ey, pt1[ngp_index, 1, :])
    Ey = su.add(Ey, pt0[ngp_index, 1, :])
    xs = l.shift(ngp_index, 1, -1, n)
    b1 = l.act(su.dagger(u0[xs, 1, :]), pt1[xs, 1, :])
    Ey = su.add(Ey, b1)
    b1 = l.act(su.dagger(u0[xs, 1, :]), pt0[xs, 1, :])
    Ey = su.add(Ey, b1)
    Ey = su.mul_s(Ey, 0.25 / t)

    Eeta = su.zero()
    Eeta = l.add_mul(Eeta, peta1[ngp_index, :], 0.5)
    Eeta = l.add_mul(Eeta, peta0[ngp_index, :], 0.5)

    return Ex, Ey, Eeta

@myjit
def compute_magnetic_field(ngp_index, aeta0, u0, n, t):
    # Magnetic fields
    b1 = l.transport(aeta0, u0, ngp_index, 1, +1, n)
    b2 = l.transport(aeta0, u0, ngp_index, 1, -1, n)
    b2 = l.add_mul(b1, b2, -1.0)
    Bx = su.mul_s(b2, -0.5 / t)

    b1 = l.transport(aeta0, u0, ngp_index, 0, +1, n)
    b2 = l.transport(aeta0, u0, ngp_index, 0, -1, n)
    b2 = l.add_mul(b1, b2, -1.0)
    By = su.mul_s(b2, +0.5 / t)

    bf1 = su.zero()
    b1 = l.plaq(u0, ngp_index, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, ngp_index, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, ngp_index, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, ngp_index, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    Beta = l.add_mul(bf1, b2, +0.25)

    return Bx, By, Beta

@myjit
def compute_lorentz_force_kernel(index, f, x0, p, pt0, pt1, u0, peta0, peta1, aeta0, n, t):
    ngp_pos = ngp(x0[index, :2])
    ngp_index = l.get_index(ngp_pos[0], ngp_pos[1], n)
    Ex, Ey, Eeta = compute_electric_field(ngp_index, pt0, pt1, u0, peta0, peta1, n, t)
    Bx, By, Beta = compute_magnetic_field(ngp_index, aeta0, u0, n, t)

    b1 = su.mul_s(Beta, p[index, 2]/p[index, 0])
    b2 = su.add(Ex, b1)
    b3 = su.mul_s(By, -t*p[index, 3]/p[index, 0])
    f[index, 0, :] = su.add(b2, b3)

    b1 = su.mul_s(Beta, -p[index, 1]/p[index, 0])
    b2 = su.add(Ey, b1)
    b3 = su.mul_s(Bx, t*p[index, 3]/p[index, 0])
    f[index, 1, :] = su.add(b2, b3)

    b1 = su.mul_s(By, p[index, 1]/p[index, 0])
    b2 = su.add(Eeta, b1)
    b3 = su.mul_s(Bx, -p[index, 2]/p[index, 0])
    f[index, 2, :] = su.add(b2, b3)

@myjit
def compute_force_correlator_kernel(index, f0, f, w, corr):
    for d in range(3):
        buf1 = l.act(w[index, :], f0[index, d, :])
        buf2 = su.mul(f[index, d, :], su.dagger(buf1))
        corr[index, d] = su.tr(buf2).real

# @myjit
# def compute_force_autocorrelator_kernel(index, f0, f1, w0, w1, auto_corr):
#     for d in range(3):
#         w01 = su.mul(w1[index, :], su.dagger(w0[index, :]))
#         buf1 = l.act(w01, f0[index, d, :])
#         buf2 = su.mul(f1[index, d, :], su.dagger(buf1))
#         auto_corr[index, d] = su.tr(buf2).real
