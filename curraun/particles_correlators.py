"""
    Module for evaluating the angle between quark antiquark pairs in a toy model computation
"""
import math
from curraun.numba_target import use_cuda, myjit, my_parallel_loop
if use_cuda:
    import numba.cuda as cuda
import numpy as np

class Angles:
    def __init__(self, wong_q, wong_aq, n_particles):
        self.wong_q = wong_q
        self.wong_aq = wong_aq
        self.n_particles = n_particles

        # angles
        self.deta = np.zeros(n_particles, dtype=np.double)
        self.dphi = np.zeros(n_particles, dtype=np.double)
        # pTs
        self.pT = np.zeros((n_particles,2), dtype=np.double)

        # set-up device pointers
        self.d_deta = self.deta
        self.d_dphi = self.dphi
        self.d_pT = self.pT

        # move data to GPU
        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_deta = cuda.to_device(self.deta)
        self.d_dphi = cuda.to_device(self.dphi)
        self.d_pT = cuda.to_device(self.pT)

    def copy_to_host(self):
        self.d_deta.copy_to_host(self.deta)
        self.d_dphi.copy_to_host(self.dphi)
        self.d_pT.copy_to_host(self.pT)

    def compute(self):

        my_parallel_loop(compute_angles_kernel, self.n_particles, self.wong_q.d_p, self.wong_aq.d_p, self.d_pT, self.d_deta, self.d_dphi)

        if use_cuda:
            self.copy_to_host()


@myjit
def compute_angles_kernel(index, pmuq, pmuaq, pT, deta, dphi):
    # |\vec{p}|^2=p^2=p_T^2+p_L^2=p_x^2+p_y^2+p_z^2, where pmu=(p^\tau, p^x, p^y, p^\eta, p^z)
    pq = math.sqrt(pmuq[index, 1]**2 + pmuq[index, 2] **2 + pmuq[index, 4] **2)
    etaq = math.atan(pmuq[index, 4]/pq)

    paq = math.sqrt(pmuaq[index, 1]**2 + pmuaq[index, 2] **2 + pmuaq[index, 4] **2)
    etaaq = math.atan(pmuaq[index, 4]/paq)

    deta[index] = math.fabs(etaq-etaaq)

    phiq = math.atan(pmuq[index, 2]/pmuq[index, 1])
    phiaq = math.atan(pmuaq[index, 2]/pmuaq[index, 1])

    dphi[index] = math.fabs(phiq-phiaq)

    # pT for quark and antiquark
    pT[index, 0] = math.sqrt(pmuq[index, 1]**2 + pmuq[index, 2] **2)
    pT[index, 1] = math.sqrt(pmuaq[index, 1]**2 + pmuaq[index, 2] **2)

class TransMom:
    def __init__(self, wong, n_particles):
        self.wong = wong
        self.n_particles = n_particles

        # pTs
        self.pT = np.zeros(n_particles, dtype=np.double)

        # set-up device pointers
        self.d_pT = self.pT

        # move data to GPU
        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_pT = cuda.to_device(self.pT)

    def copy_to_host(self):
        self.d_pT.copy_to_host(self.pT)

    def compute(self):

        my_parallel_loop(compute_trans_mom_kernel, self.n_particles, self.d_pT, self.wong.d_p)

        if use_cuda:
            self.copy_to_host()

@myjit
def compute_trans_mom_kernel(index, pT, p):
    pT[index] = math.sqrt(p[index, 1]**2 + p[index, 2] **2)