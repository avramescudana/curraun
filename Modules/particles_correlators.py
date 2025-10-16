"""
    Module for evaluating the angle between quark antiquark pairs in a toy model computation
"""
import math
from Modules.numba_target import use_cuda, myjit, my_parallel_loop
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

        # my_parallel_loop(compute_angles_kernel, self.n_particles, self.wong_q.d_p, self.wong_aq.d_p, self.d_pT, self.d_deta, self.d_dphi)
        my_parallel_loop(compute_angles_dotproduct_kernel, self.n_particles, self.wong_q.d_p, self.wong_aq.d_p, self.d_pT, self.d_deta, self.d_dphi)

        if use_cuda:
            self.copy_to_host()


@myjit
def angle_dot_product(a0, a1, b0, b1):

    """
        Computes the dot product between 2 vectors 2-dimensional
        Divides by the norm of the vectors
    """
    dot_product = a0*b0+a1*b1
    cross_product = a0*b1-a1*b0
    # norm = math.sqrt(a0**2+a1**2)/math.sqrt(b0**2+b1**2)
    # angle = math.degress(math.acos(c))
    angle = math.atan2(cross_product, dot_product)
    if angle<0:
        angle += 2*math.pi
    return angle


@myjit
def compute_angles_kernel(index, pmuq, pmuaq, pT, deta, dphi):
    # |\vec{p}|^2=p^2=p_T^2+p_L^2=p_x^2+p_y^2+p_z^2, where pmu=(p^\tau, p^x, p^y, p^\eta, p^z)
    pq = math.sqrt(pmuq[index, 1]**2 + pmuq[index, 2] **2 + pmuq[index, 4] **2)
    # etaq = math.atanh(pmuq[index, 4]/pq)
    etaq = math.log((pq+pmuq[index, 4])/(pq-pmuq[index, 4]))/2

    paq = math.sqrt(pmuaq[index, 1]**2 + pmuaq[index, 2] **2 + pmuaq[index, 4] **2)
    # etaaq = math.atanh(pmuaq[index, 4]/paq)
    etaaq = math.log((paq+pmuaq[index, 4])/(paq-pmuaq[index, 4]))/2

    # deta[index] = math.fabs(etaq-etaaq)
    deta[index] = etaq-etaaq

    phiq = math.atan(pmuq[index, 2]/pmuq[index, 1]) + math.pi/2
    phiaq = math.atan(pmuaq[index, 2]/pmuaq[index, 1]) + math.pi/2

    # dphi[index] = math.fabs(phiq-phiaq)
    dphi[index] = phiq-phiaq + math.pi

    # phiq = math.acos(pmuq[index, 1]/math.sqrt(pmuq[index, 1]**2 + pmuq[index, 2]**2))
    # phiaq = math.acos(pmuaq[index, 1]/math.sqrt(pmuaq[index, 1]**2 + pmuaq[index, 2]**2))

    # dphi[index] = phiq-phiaq + math.pi/2

    # pT for quark and antiquark
    pT[index, 0] = math.sqrt(pmuq[index, 1]**2 + pmuq[index, 2] **2)
    pT[index, 1] = math.sqrt(pmuaq[index, 1]**2 + pmuaq[index, 2] **2)

@myjit
def compute_angles_dotproduct_kernel(index, pmuq, pmuaq, pT, deta, dphi):
    pq = math.sqrt(pmuq[index, 1]**2 + pmuq[index, 2] **2 + pmuq[index, 4] **2)
    etaq = math.log((pq+pmuq[index, 4])/(pq-pmuq[index, 4]))/2

    paq = math.sqrt(pmuaq[index, 1]**2 + pmuaq[index, 2] **2 + pmuaq[index, 4] **2)
    etaaq = math.log((paq+pmuaq[index, 4])/(paq-pmuaq[index, 4]))/2

    deta[index] = etaq-etaaq

    dphi[index] = angle_dot_product(pmuq[index, 1], pmuq[index, 2], pmuaq[index, 1], pmuaq[index, 2])

    pT[index, 0] = math.sqrt(pmuq[index, 1]**2 + pmuq[index, 2] **2)
    pT[index, 1] = math.sqrt(pmuaq[index, 1]**2 + pmuaq[index, 2] **2)

class BorderAngles:
    def __init__(self, wong_q, n_particles, pmuaq):
        self.wong_q = wong_q
        self.n_particles = n_particles
        self.pmuaq = pmuaq

        # angles
        self.deta = np.zeros(n_particles, dtype=np.double)
        self.dphi = np.zeros(n_particles, dtype=np.double)
        # pTs
        self.pT = np.zeros(n_particles, dtype=np.double)

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

        my_parallel_loop(compute_borderangles_kernel, self.n_particles, self.wong_q.d_p, self.pmuaq, self.d_pT, self.d_deta, self.d_dphi)

        if use_cuda:
            self.copy_to_host()

@myjit
def compute_borderangles_kernel(index, pmuq, pmuaq, pT, deta, dphi):
    # |\vec{p}|^2=p^2=p_T^2+p_L^2=p_x^2+p_y^2+p_z^2, where pmu=(p^\tau, p^x, p^y, p^\eta, p^z)
    pq = math.sqrt(pmuq[index, 1]**2 + pmuq[index, 2] **2 + pmuq[index, 4] **2)
    etaq = math.atanh(pmuq[index, 4]/pq)
    # etaq = math.log((pq+pmuq[index, 4])/(pq-pmuq[index, 4]))/2

    paq = math.sqrt(pmuaq[index, 1]**2 + pmuaq[index, 2] **2 + pmuaq[index, 4] **2)
    etaaq = math.atanh(pmuaq[index, 4]/paq)
    # etaaq = math.log((paq+pmuaq[index, 4])/(paq-pmuaq[index, 4]))/2

    # deta[index] = math.fabs(etaq-etaaq)
    deta[index] = etaq-etaaq

    phiq = math.atan(pmuq[index, 2]/pmuq[index, 1])
    phiaq = math.atan(pmuaq[index, 2]/pmuaq[index, 1])

    # dphi[index] = math.fabs(phiq-phiaq)
    dphi[index] = phiq-phiaq

    # pT for quark
    pT[index] = math.sqrt(pmuq[index, 1]**2 + pmuq[index, 2] **2)


class TransMom:
    def __init__(self, wong, n_particles):
        self.wong = wong
        self.n_particles = n_particles

        # pTs
        self.pT = np.zeros(n_particles, dtype=np.double)
        self.p2 = np.zeros((n_particles, 3), dtype=np.double)

        # set-up device pointers
        self.d_pT = self.pT
        self.d_p2 = self.p2

        # move data to GPU
        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_pT = cuda.to_device(self.pT)
        self.d_p2 = cuda.to_device(self.p2)

    def copy_to_host(self):
        self.d_pT.copy_to_host(self.pT)
        self.d_p2.copy_to_host(self.p2)

    def compute(self):

        my_parallel_loop(compute_trans_mom_kernel, self.n_particles, self.d_pT, self.d_p2, self.wong.d_p)

        if use_cuda:
            self.copy_to_host()

@myjit
def compute_trans_mom_kernel(index, pT, p2, p):
    pT[index] = math.sqrt(p[index, 1]**2 + p[index, 2] **2)
    p2[index, 0] = p[index, 1]**2 
    p2[index, 1] = p[index, 2]**2 
    p2[index, 2] = p[index, 1]**2 + p[index, 2] **2
