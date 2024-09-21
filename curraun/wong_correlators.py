"""
    Module for evaluating correlators of the Lorentz force along the trajectory of the particle
"""

from curraun.numba_target import use_cuda, myjit, my_parallel_loop, my_cuda_sum, mycudajit
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda
import numpy as np

class ForceCorrelators:
    def __init__(self, wong, n_particles):
        self.wong = wong
        self.n_particles = n_particles

        # lorentz force correlator
        self.corr = np.zeros((n_particles, 3), dtype=su.GROUP_TYPE_REAL)
        # correlator averaged over particles
        self.corr_mean = np.zeros(3, dtype=np.double)
        if use_cuda:
            # use pinned memory for asynchronous data transfer
            self.corr_mean = cuda.pinned_array(3, dtype=np.double)
            self.corr_mean[0:3] = 0.0

        # set-up device pointers
        self.d_corr = self.corr
        self.d_corr_mean = self.corr_mean

        # move data to GPU
        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_corr = cuda.to_device(self.corr)
        self.d_corr_mean = cuda.to_device(self.corr_mean)
        
    def copy_to_host(self):
        self.d_corr.copy_to_host(self.corr)
        self.d_corr_mean.copy_to_host(self.corr_mean)
        
    def copy_mean_to_device(self, stream=None):
        self.d_corr_mean = cuda.to_device(self.corr_mean, stream)
    
    def copy_mean_to_host(self, stream=None):
        self.d_corr_mean.copy_to_host(self.corr_mean, stream)

    def compute(self, f0, tag, stream=None):

        if tag=='transported':
            my_parallel_loop(compute_force_correlator_transported_kernel, self.n_particles, f0, self.wong.d_fc, self.wong.d_w, self.d_corr)
        elif tag=='naive':
            my_parallel_loop(compute_force_correlator_naive_kernel, self.n_particles, f0, self.wong.d_fc, self.d_corr)

        compute_mean(self.d_corr, self.d_corr_mean, stream)

        if use_cuda:
            self.copy_mean_to_host()

@myjit
def compute_force_correlator_transported_kernel(index, f0, f, w, corr):
    for d in range(2):
        buf1 = l.act(w[index, :], f0[index, d, :])
        buf2 = su.mul(f[index, d, :], su.dagger(buf1))
        corr[index, d] = su.tr(buf2).real
    # fz, not feta
    d = 3
    buf1 = l.act(w[index, :], f0[index, d, :])
    buf2 = su.mul(f[index, d, :], su.dagger(buf1))
    corr[index, 2] = su.tr(buf2).real


@myjit
def compute_force_correlator_naive_kernel(index, f0, f, corr):
    for d in range(2):
        buf1 = su.dagger(f0[index, d, :])
        buf2 = su.mul(f[index, d, :], buf1)
        corr[index, d] = su.tr(buf2).real
    # fz, not feta
    d = 3
    buf1 = su.dagger(f0[index, d, :])
    buf2 = su.mul(f[index, d, :], buf1)
    corr[index, 2] = su.tr(buf2).real

def compute_mean(corr, corr_mean, stream):
    if use_cuda:
        for i in range(3):
            my_cuda_sum(corr[:, i], stream)
        collect_results[1, 1, stream](corr_mean, corr)
    else:
        for i in range(3):
            corr_mean[i] = np.mean(corr[:, i])

@mycudajit  
def collect_results(corr_mean, corr):
    for i in range(3):
        corr_mean[i] = corr[0, i] / corr[:, i].size