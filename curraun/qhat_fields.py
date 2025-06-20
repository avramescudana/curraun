from curraun.numba_target import myjit, prange, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda

class QhatFields:
    def __init__(self, s):
        self.s = s
        self.n = s.n

        # E fields (Ey, Ez, tildeEz)
        # self.E = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.Esq = np.zeros((self.n ** 2, 3), dtype=su.GROUP_TYPE)

        # B fields (By, Bz, tildeBz)
        # self.B = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.Bsq = np.zeros((self.n ** 2, 3), dtype=su.GROUP_TYPE)

        # A fields (Ay, Az, Aeta)
        # self.A = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.Asq = np.zeros((self.n ** 2, 3), dtype=su.GROUP_TYPE)

        self.Esq_mean = np.zeros(3, dtype=np.double)
        self.Bsq_mean = np.zeros(3, dtype=np.double)
        self.Asq_mean = np.zeros(3, dtype=np.double)

        # Memory on the CUDA device:
        # self.d_E = self.E
        # self.d_B = self.B
        # self.d_A = self.A
        self.d_Esq = self.Esq
        self.d_Bsq = self.Bsq
        self.d_Asq = self.Asq
        # self.d_Esq_mean = self.Esq_mean
        # self.d_Bsq_mean = self.Bsq_mean
        # self.d_Asq_mean = self.Asq_mean

    def copy_to_device(self):
        # self.d_E = cuda.to_device(self.E)
        # self.d_B = cuda.to_device(self.B)
        # self.d_A = cuda.to_device(self.A)
        self.d_Esq = cuda.to_device(self.Esq)
        self.d_Bsq = cuda.to_device(self.Bsq)
        self.d_Asq = cuda.to_device(self.Asq)
        # self.d_Esq_mean = cuda.to_device(self.Esq_mean)
        # self.d_Bsq_mean = cuda.to_device(self.Bsq_mean)
        # self.d_Asq_mean = cuda.to_device(self.Asq_mean)

    def copy_to_host(self):
        # self.d_E.copy_to_host(self.E)
        # self.d_B.copy_to_host(self.B)
        # self.d_A.copy_to_host(self.A)       
        self.d_Esq.copy_to_host(self.Esq)
        self.d_Bsq.copy_to_host(self.Bsq)
        self.d_Asq.copy_to_host(self.Asq)
        # self.d_Esq_mean.copy_to_host(self.Esq_mean)
        # self.d_Bsq_mean.copy_to_host(self.Bsq_mean)
        # self.d_Asq_mean.copy_to_host(self.Asq_mean)
        
    def compute(self):
        
        # tint = round(self.s.t / self.s.dt)
        # if tint % self.dtstep == 0 and tint >= 1:
        # compute_E(self.s, self.d_E)
        # compute_B(self.s, self.d_B)
        # compute_A(self.s, self.d_A)

        # compute_sq(self.s, self.d_E, self.d_Esq)
        # compute_sq(self.s, self.d_B, self.d_Bsq)
        # compute_sq(self.s, self.d_A, self.d_Asq)

        compute_Esq(self.s, self.d_Esq)
        compute_Bsq(self.s, self.d_Bsq)
        compute_Asq(self.s, self.d_Asq)

        self.Esq_mean = np.mean(self.d_Esq, axis=0)
        self.Bsq_mean = np.mean(self.d_Bsq, axis=0)
        self.Asq_mean = np.mean(self.d_Asq, axis=0)

        if use_cuda:
            self.copy_to_host()

def compute_E(s, E):
    u0 = s.d_u0
    pt1 = s.d_pt1
    aeta0 = s.d_aeta0
    peta1 = s.d_peta1
    pt0 = s.d_pt0
    peta0 = s.d_peta0

    n = s.n
    tau = s.t 

    my_parallel_loop(compute_E_kernel, n * n, n, u0, aeta0, peta1, peta0, pt1, pt0, E, tau)

@myjit
def compute_E_kernel(xi, n, u0, aeta0, peta1, peta0, pt1, pt0, E, tau):
    # Ey
    bf1 = su.zero()
    bf1 = l.add_mul(bf1, pt1[xi, 1], 0.25 / tau)
    bf1 = l.add_mul(bf1, pt0[xi, 1], 0.25 / tau)
    xs2 = l.shift(xi, 0, -1, n)
    xs3 = l.shift(xi, 1, -1, n)
    b1 = l.act(su.dagger(u0[xs3, 1]), pt1[xs2, 1])
    bf1 = l.add_mul(bf1, b1, 0.25 / tau)
    b1 = l.act(su.dagger(u0[xs3, 1]), pt0[xs2, 1])
    bf1 = l.add_mul(bf1, b1, 0.25 / tau)
    # bf1 = su.mul_s(pt0[xi, 1], 1 / tau)
    su.store(E[xi, 0], bf1)

    # Ez
    bf2 = su.zero()
    bf2 = l.add_mul(bf2, peta1[xi], 0.5)
    bf2 = l.add_mul(bf2, peta0[xi], 0.5)
    # bf2 = peta0[xi]
    su.store(E[xi, 1], bf2)

    # tildeEz
    bf3 = su.mul_s(aeta0[xi],  - 1.0 / (tau * tau))
    su.store(E[xi, 2], bf3)

def compute_B(s, B):
    u0 = s.d_u0
    aeta0 = s.d_aeta0

    n = s.n
    tau = s.t 

    my_parallel_loop(compute_B_kernel, n * n, n, u0, aeta0, B, tau) 

@myjit
def compute_B_kernel(xi, n, u0, aeta0, B, tau): 
    # By
    b1 = l.transport(aeta0, u0, xi, 0, +1, n)
    b2 = l.transport(aeta0, u0, xi, 0, -1, n)
    b3 = l.add_mul(b1, b2, -1.0)
    bf1 = l.add_mul(b3, b1, 0.5 / tau)
    # b1 = l.transport(aeta0, u0, xi, 0, 1, n)
    # bf1 = l.add_mul(b1, aeta0[xi], -1)
    su.store(B[xi, 0], bf1)

    # Bz
    b1 = l.plaq(u0, xi, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xi, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, xi, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xi, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)
    su.store(B[xi, 1], bf1)

    # tildeBz
    xspy1 = l.shift(xi, 1, 1, n)
    axpy1 = su.mlog(u0[xspy1, 0])

    xsmx1 = l.shift(xi, 1, -1, n)
    axmy1 = su.mlog(u0[xsmx1, 0])

    dif = l.add_mul(axpy1, axmy1, -1.0)
    dyax = su.mul_s(dif, 0.5)
    su.store(B[xi, 2], dyax)

def compute_A(s, A):    
    u0 = s.d_u0
    aeta0 = s.d_aeta0

    n = s.n
    tau =round(s.t) 

    my_parallel_loop(compute_A_kernel, n * n, u0, aeta0, A, tau)

@myjit
def compute_A_kernel(xi, u0, aeta0, A, tau):
    # Ay
    bf1 = su.mlog(u0[xi, 1])
    su.store(A[xi, 0], bf1)

    # Az
    bf2 = su.mul_s(aeta0[xi],  1.0 / tau)
    su.store(A[xi, 1], bf2)

    # Aeta
    su.store(A[xi, 2], aeta0[xi])

def compute_sq(s, fields, fields_sq):
    n = s.n
    my_parallel_loop(compute_sq_kernel, n * n, fields, fields_sq)

@myjit
def compute_sq_kernel(xi, fields, fields_sq):
    for i in range(3):
        fields_sq[i] = su.sq(fields[xi, i])

def compute_Esq(s, Esq):
    u0 = s.d_u0
    aeta0 = s.d_aeta0
    pt0 = s.d_pt0
    pt1 = s.d_pt1
    peta0 = s.d_peta0
    peta1 = s.d_peta1

    n = s.n
    tau = s.t 

    my_parallel_loop(compute_Esq_kernel, n * n, n, u0, aeta0, peta0, peta1, pt0, pt1, Esq, tau)

@myjit
def compute_Esq_kernel(xi, n, u0, aeta0, peta0, peta1, pt0, pt1, Esq, tau):
    # Ey
    # bf1 = su.mul_s(pt0[xi, 1], 1 / tau)
    bf1 = su.zero()
    bf1 = l.add_mul(bf1, pt1[xi, 1], 0.25 / tau)
    bf1 = l.add_mul(bf1, pt0[xi, 1], 0.25 / tau)
    xs2 = l.shift(xi, 0, -1, n)
    xs3 = l.shift(xi, 1, -1, n)
    b1 = l.act(su.dagger(u0[xs3, 1]), pt1[xs2, 1])
    bf1 = l.add_mul(bf1, b1, 0.25 / tau)
    b1 = l.act(su.dagger(u0[xs3, 1]), pt0[xs2, 1])
    bf1 = l.add_mul(bf1, b1, 0.25 / tau)
    Esq[xi, 0] = su.sq(bf1)

    # Ez
    # bf2 = peta0[xi]
    bf2 = su.zero()
    bf2 = l.add_mul(bf2, peta1[xi], 0.5)
    bf2 = l.add_mul(bf2, peta0[xi], 0.5)
    Esq[xi, 1] = su.sq(bf2)

    # tildeEz
    bf3 = su.mul_s(aeta0[xi], -1.0 / (tau * tau))
    Esq[xi, 2] = su.sq(bf3)

def compute_Bsq(s, Bsq):
    aeta0 = s.d_aeta0
    u0 = s.d_u0

    n = s.n
    tau = s.t 

    my_parallel_loop(compute_Bsq_kernel, n * n, n, aeta0, u0, Bsq, tau)    

@myjit
def compute_Bsq_kernel(xi, n, aeta0, u0, Bsq, tau):    
    # By
    b1 = l.transport(aeta0, u0, xi, 0, 1, n)
    b2 = l.add_mul(b1, aeta0[xi], -1)
    bf1 = su.mul_s(b2, 1 / tau)
    Bsq[xi, 0] = su.sq(bf1)

    # Bz
    b1 = l.plaq(u0, xi, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xi, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, xi, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xi, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)
    Bsq[xi, 1] = su.sq(bf1)

    # tildeBz
    xspy1 = l.shift(xi, 1, 1, n)
    axpy1 = su.mlog(u0[xspy1, 0])

    xsmx1 = l.shift(xi, 1, -1, n)
    axmy1 = su.mlog(u0[xsmx1, 0])

    dif = l.add_mul(axpy1, axmy1, -1.0)
    dyax = su.mul_s(dif, 0.5)
    Bsq[xi, 2] = su.sq(dyax)

def compute_Asq(s, Asq):
    aeta0 = s.d_aeta0
    u0 = s.d_u0

    n = s.n
    tau = round(s.t) 

    my_parallel_loop(compute_Asq_kernel, n * n, aeta0, u0, Asq, tau)    

@myjit
def compute_Asq_kernel(xi, aeta0, u0, Asq, tau):
    # Ay
    bf1 = su.mlog(u0[xi, 1])
    Asq[xi, 0] = su.sq(bf1)

    # Az
    bf2 = su.mul_s(aeta0[xi], 1.0 / tau)
    Asq[xi, 1] = su.sq(bf2)

    # Aeta
    Asq[xi, 2] = su.sq(aeta0[xi])