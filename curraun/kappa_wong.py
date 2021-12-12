from curraun.numba_target import myjit, use_cuda, my_parallel_loop, my_cuda_sum, mycudajit
import numpy as np
import curraun.lattice as l
import curraun.su as su
import curraun.kappa as kappa
if use_cuda:
    import numba.cuda as cuda

class TransportedForce:
    def __init__(self, s, wong, n_particles):
        self.s = s
        self.n = s.n
        self.dtstep = round(1.0 / s.dt)
        self.wong = wong
        self.n_particles = n_particles

        # transported color force
        self.fc_transp = np.zeros((n_particles, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # integrated color force
        self.fc_int = np.zeros((n_particles, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # integrated invariant force
        self.finv_int = np.zeros((n_particles, 3), dtype=su.GROUP_TYPE)

        # single components
        self.p_perp_x_fc = np.zeros(n_particles, dtype=np.double)
        self.p_perp_y_fc = np.zeros(n_particles, dtype=np.double)
        self.p_perp_eta_fc = np.zeros(n_particles, dtype=np.double)

        self.p_perp_x_finv = np.zeros(n_particles, dtype=np.double)
        self.p_perp_y_finv = np.zeros(n_particles, dtype=np.double)
        self.p_perp_eta_finv = np.zeros(n_particles, dtype=np.double)

        # mean values
        self.p_perp_mean_fc = np.zeros(3, dtype=np.double)
        if use_cuda:
            # use pinned memory for asynchronous data transfer
            self.p_perp_mean_fc = cuda.pinned_array(3, dtype=np.double)
            self.p_perp_mean_fc[0:3] = 0.0

        self.p_perp_mean_finv = np.zeros(3, dtype=np.double)
        if use_cuda:
            # use pinned memory for asynchronous data transfer
            self.p_perp_mean_finv = cuda.pinned_array(3, dtype=np.double)
            self.p_perp_mean_finv[0:3] = 0.0

        # time counter
        self.t = 0

        # Memory on the CUDA device:
        self.d_fc_transp = self.fc_transp
        self.d_fc_int = self.fc_int
        self.d_finv_int = self.finv_int
        self.d_p_perp_x_fc = self.p_perp_x_fc
        self.d_p_perp_y_fc = self.p_perp_y_fc
        self.d_p_perp_eta_fc = self.p_perp_eta_fc
        self.d_p_perp_mean_fc = self.p_perp_mean_fc

        self.d_p_perp_x_finv = self.p_perp_x_finv
        self.d_p_perp_y_finv = self.p_perp_y_finv
        self.d_p_perp_eta_finv = self.p_perp_eta_finv
        self.d_p_perp_mean_finv = self.p_perp_mean_finv

    def copy_to_device(self):
        self.d_fc_transp = cuda.to_device(self.fc_transp)
        self.d_fc_int = cuda.to_device(self.fc_int)
        self.d_finv_int = cuda.to_device(self.finv_int)
        self.d_p_perp_x_fc = cuda.to_device(self.p_perp_x_fc)
        self.d_p_perp_y_fc = cuda.to_device(self.p_perp_y_fc)
        self.d_p_perp_eta_fc = cuda.to_device(self.p_perp_eta_fc)
        self.d_p_perp_mean_fc = cuda.to_device(self.p_perp_mean_fc)

        self.d_p_perp_x_finv = cuda.to_device(self.p_perp_x_finv)
        self.d_p_perp_y_finv = cuda.to_device(self.p_perp_y_finv)
        self.d_p_perp_eta_finv = cuda.to_device(self.p_perp_eta_finv)
        self.d_p_perp_mean_finv = cuda.to_device(self.p_perp_mean_finv)

    def copy_to_host(self):
        self.d_fc_transp.copy_to_host(self.fc_transp)
        self.d_fc_int.copy_to_host(self.fc_int)
        self.d_finv_int.copy_to_host(self.finv_int)
        self.d_p_perp_x_fc.copy_to_host(self.p_perp_x_fc)
        self.d_p_perp_y_fc.copy_to_host(self.p_perp_y_fc)
        self.d_p_perp_eta_fc.copy_to_host(self.p_perp_eta_fc)
        self.d_p_perp_mean_fc.copy_to_host(self.p_perp_mean_fc)

        self.d_p_perp_x_finv.copy_to_host(self.p_perp_x_finv)
        self.d_p_perp_y_finv.copy_to_host(self.p_perp_y_finv)
        self.d_p_perp_eta_finv.copy_to_host(self.p_perp_eta_finv)
        self.d_p_perp_mean_finv.copy_to_host(self.p_perp_mean_finv)

    def copy_mean_to_device(self, stream=None):
        self.d_p_perp_mean_fc = cuda.to_device(self.p_perp_mean_fc, stream)
        self.d_p_perp_mean_finv = cuda.to_device(self.p_perp_mean_finv, stream)

    def copy_mean_to_host(self, stream=None):
        self.d_p_perp_mean_fc.copy_to_host(self.p_perp_mean_fc, stream)
        self.d_p_perp_mean_finv.copy_to_host(self.p_perp_mean_finv, stream)

    def compute(self, wong, stream=None):
        tint = round(self.s.t / self.s.dt)
        if tint % self.dtstep == 0 and tint >= 1:
            """
                Momentum broadenings from color Lorentz forces
            """
            # apply Wilson lines to color Lorentz force
            apply_w(wong.d_w, wong.d_fc, self.d_fc_transp, self.n_particles, stream)

            # integrate fc
            integrate_fc(self.d_fc_transp, self.d_fc_int, self.n_particles, 1.0, self.s.t, stream)

            # integrate perpendicular momentum
            compute_p_perp_fc(self.d_fc_int, self.d_p_perp_x_fc, self.d_p_perp_y_fc, self.d_p_perp_eta_fc, self.n_particles, self.s.t, stream)

            # calculate mean
            kappa.compute_mean(self.d_p_perp_x_fc, self.d_p_perp_y_fc, self.d_p_perp_eta_fc, self.d_p_perp_mean_fc, stream)

            """
                Momentum broadenings from invariant Lorentz forces
            """
            # integrate finv 
            integrate_finv(wong.d_finv,  self.d_finv_int, self.n_particles, 1.0, self.s.t, stream)

            # compute squared momentum
            compute_p_perp_finv(self.d_finv_int, self.d_p_perp_x_finv, self.d_p_perp_y_finv, self.d_p_perp_eta_finv, self.n_particles, self.s.t, stream)

            # calculate mean
            kappa.compute_mean(self.d_p_perp_x_finv, self.d_p_perp_y_finv, self.d_p_perp_eta_finv, self.d_p_perp_mean_finv, stream)

def apply_w(w, fc, fc_transp, n_particles, stream):
    my_parallel_loop(apply_w_kernel, n_particles, fc, w, fc_transp, stream=stream)

@myjit
def apply_w_kernel(index, fc, w, fc_transp):
    for d in range(3):
        b1 = l.act(w[index], fc[index, d])
        su.store(fc_transp[index, d], b1)

def integrate_fc(fc_transp, fc_int, n_particles, dt, t, stream):
    my_parallel_loop(integrate_fc_kernel, n_particles, fc_transp, fc_int, dt, t, stream=stream)

@myjit
def integrate_fc_kernel(index, fc_transp, fc_int, dt, t):
    for d in range(2):
        bfi = l.add_mul(fc_int[index, d], fc_transp[index, d], dt)
        su.store(fc_int[index, d], bfi)
    bfi = l.add_mul(fc_int[index, 2], fc_transp[index, 2], dt * t ** 2)
    su.store(fc_int[index, 2], bfi)

def integrate_finv(finv, finv_int, n_particles, dt, t, stream):
    my_parallel_loop(integrate_finv_kernel, n_particles, finv, finv_int, dt, t, stream=stream)

@myjit
def integrate_finv_kernel(index, finv, finv_int, dt, t):
    for d in range(2):
        finv_int[index, d] += finv[index, d] * dt
    finv_int[index, 2] += finv[index, 2] * dt * t ** 2


def compute_p_perp_fc(fc_int, p_perp_x, p_perp_y, p_perp_eta, n_particles, t, stream):
    my_parallel_loop(compute_p_perp_fc_kernel, n_particles, fc_int, p_perp_x, p_perp_y, p_perp_eta, t, stream=stream)

@myjit
def compute_p_perp_fc_kernel(index, fc_int, p_perp_x, p_perp_y, p_perp_eta, t):
    p_perp_x[index] = su.sq(fc_int[index, 0])
    p_perp_y[index] = su.sq(fc_int[index, 1])
    p_perp_eta[index] = su.sq(fc_int[index, 2]) / t ** 2

def compute_p_perp_finv(finv_int, p_perp_x, p_perp_y, p_perp_eta, n_particles, t, stream):
    my_parallel_loop(compute_p_perp_finv_kernel, n_particles, finv_int, p_perp_x, p_perp_y, p_perp_eta, t, stream=stream)

@myjit
def compute_p_perp_finv_kernel(index, finv_int, p_perp_x, p_perp_y, p_perp_eta, t):
    p_perp_x[index] = finv_int[index, 0] ** 2
    p_perp_y[index] = finv_int[index, 1] ** 2
    p_perp_eta[index] = finv_int[index, 2] ** 2 / t ** 2
