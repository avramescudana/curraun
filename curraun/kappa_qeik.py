from curraun.numba_target import myjit, prange, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
import curraun.kappa as kappa
if use_cuda:
    import numba.cuda as cuda

class KineticCanonicCheck:
    def __init__(self, s):
        self.s = s
        self.n = s.n
        self.dtstep = round(1.0 / s.dt)

        # initial gauge field
        self.a0 = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        # gauge field
        self.a = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # force
        self.fcan = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.fkin = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.fa = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # integrated force
        self.intfcan = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.intfkin = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.intfa = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # paralel transported gauge field - initial gage field
        self.da = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        self.dpcan_sq = np.zeros((self.n ** 2, 3), dtype=np.double)
        self.dpkin_sq = np.zeros((self.n ** 2, 3), dtype=np.double)
        self.da_sq = np.zeros((self.n ** 2, 3), dtype=np.double)
        self.dpcanda = np.zeros((self.n ** 2, 3), dtype=np.double)
        self.dpcanda_transp = np.zeros((self.n ** 2, 3), dtype=np.double)
        self.da_transp_sq = np.zeros((self.n ** 2, 3), dtype=np.double)

        # mean values
        self.dpcan_sq_mean = np.zeros(3, dtype=np.double)
        self.dpkin_sq_mean = np.zeros(3, dtype=np.double)
        self.da_sq_mean = np.zeros(3, dtype=np.double)
        self.dpcanda_mean = np.zeros(3, dtype=np.double)
        self.dpcanda_transp_mean = np.zeros(3, dtype=np.double)
        self.da_transp_sq_mean = np.zeros(3, dtype=np.double)

        if use_cuda:
            # use pinned memory for asynchronous data transfer
            self.dpcan_sq_mean = cuda.pinned_array(3, dtype=np.double)
            self.dpcan_sq_mean[0:3] = 0.0
            self.dpkin_sq_mean = cuda.pinned_array(3, dtype=np.double)
            self.dpkin_sq_mean[0:3] = 0.0
            self.da_sq_mean = cuda.pinned_array(3, dtype=np.double)
            self.da_sq_mean[0:3] = 0.0
            self.dpcanda_mean = cuda.pinned_array(3, dtype=np.double)
            self.dpcanda_mean[0:3] = 0.0
            self.dpcanda_transp_mean = cuda.pinned_array(3, dtype=np.double)
            self.dpcanda_transp_mean[0:3] = 0.0
            self.da_transp_sq_mean = cuda.pinned_array(3, dtype=np.double)
            self.da_transp_sq_mean[0:3] = 0.0

        # time counter
        self.t = 0

        # Memory on the CUDA device:
        self.d_a = self.a
        self.d_fcan = self.fcan
        self.d_fkin = self.fkin
        self.d_fa = self.fa
        self.d_intfcan = self.intfcan
        self.d_intfkin = self.intfkin
        self.d_intfa = self.intfa
        self.d_a0 = self.a0
        self.d_da = self.da
        self.d_dpcan_sq = self.dpcan_sq
        self.d_dpkin_sq = self.dpkin_sq
        self.d_da_sq = self.da_sq
        self.d_da_transp_sq = self.da_transp_sq
        self.d_dpcanda = self.dpcanda
        self.d_dpcanda_transp = self.dpcanda_transp
        self.d_dpcan_sq_mean = self.dpcan_sq_mean
        self.d_dpkin_sq_mean = self.dpkin_sq_mean
        self.d_da_sq_mean = self.da_sq_mean
        self.d_da_transp_sq_mean = self.da_transp_sq_mean
        self.d_dpcanda_mean = self.dpcanda_mean
        self.d_dpcanda_transp_mean = self.dpcanda_transp_mean


    def copy_to_device(self):
        self.d_a = cuda.to_device(self.a)
        self.d_fcan = cuda.to_device(self.fcan)
        self.d_fkin = cuda.to_device(self.fkin)
        self.d_fa = cuda.to_device(self.fa)
        self.d_intfcan = cuda.to_device(self.intfcan)
        self.d_intfkin = cuda.to_device(self.intfkin)
        self.d_intfa = cuda.to_device(self.intfa)
        self.d_a0 = cuda.to_device(self.a0)
        self.d_da = cuda.to_device(self.da)
        self.d_dpcan_sq = cuda.to_device(self.dpcan_sq)
        self.d_dpkin_sq = cuda.to_device(self.dpkin_sq)
        self.d_da_sq = cuda.to_device(self.da_sq)
        self.d_da_transp_sq = cuda.to_device(self.da_transp_sq)
        self.d_dpcanda = cuda.to_device(self.dpcanda)
        self.d_dpcanda_transp = cuda.to_device(self.dpcanda_transp)
        self.d_dpcan_sq_mean = cuda.to_device(self.dpcan_sq_mean)
        self.d_dpkin_sq_mean = cuda.to_device(self.dpkin_sq_mean)
        self.d_da_sq_mean = cuda.to_device(self.da_sq_mean)
        self.d_da_transp_sq_mean = cuda.to_device(self.da_transp_sq_mean)
        self.d_dpcanda_mean = cuda.to_device(self.dpcanda_mean)
        self.d_dpcanda_transp_mean = cuda.to_device(self.dpcanda_transp_mean)
        
    def copy_to_host(self):
        self.d_a.copy_to_host(self.a)
        self.d_fcan.copy_to_host(self.fcan)
        self.d_fkin.copy_to_host(self.fkin)
        self.d_fa.copy_to_host(self.fa)
        self.d_intfcan.copy_to_host(self.intfcan)
        self.d_intfkin.copy_to_host(self.intfkin)
        self.d_intfa.copy_to_host(self.intfa)
        self.d_a0.copy_to_host(self.a0)
        self.d_da.copy_to_host(self.da)
        self.d_dpcan_sq.copy_to_host(self.dpcan_sq)
        self.d_dpkin_sq.copy_to_host(self.dpkin_sq)
        self.d_da_sq.copy_to_host(self.da_sq)
        self.d_da_transp_sq.copy_to_host(self.da_transp_sq)
        self.d_dpcanda.copy_to_host(self.dpcanda)
        self.d_dpcanda_transp.copy_to_host(self.dpcanda_transp)
        self.d_dpcan_sq_mean.copy_to_host(self.dpcan_sq_mean)
        self.d_dpkin_sq_mean.copy_to_host(self.dpkin_sq_mean)
        self.d_da_sq_mean.copy_to_host(self.da_sq_mean)
        self.d_da_transp_sq_mean.copy_to_host(self.da_transp_sq_mean)
        self.d_dpcanda_mean.copy_to_host(self.dpcanda_mean)
        self.d_dpcanda_transp_mean.copy_to_host(self.dpcanda_transp_mean)

    def compute(self):
        tint = round(self.s.t / self.s.dt)
        tstart = round(1 / self.s.dt)
        t = round(self.s.t)
        n = self.s.n

        fcan = self.d_fcan
        fkin = self.d_fkin
        fa = self.d_fa

        a0 = self.d_a0
        a = self.d_a

        if tint == tstart:
            compute_ai(self.s, a0, t)

        if tint % self.dtstep == 0 and tint >= tstart:
            compute_fcan(self.s, fcan)
            compute_fkin(self.s, fkin)

            # compute gauge field, d gauge field
            compute_ai(self.s, a, t)
            compute_dai(a, a0, self.d_da, n)

            compute_p_perp(self.d_da, self.d_da_transp_sq[:, 0], self.d_da_transp_sq[:, 1], self.d_da_transp_sq[:, 2], n)
            compute_mean(self.d_da_transp_sq[:, 0], self.d_da_transp_sq[:, 1], self.d_da_transp_sq[:, 2], self.d_da_transp_sq_mean)

            compute_fa(fcan, fkin, fa, n)

            # integrate f
            integrate_f(fcan, self.d_intfcan, n, 1.0)
            integrate_f(fkin, self.d_intfkin, n, 1.0)
            integrate_f(fa, self.d_intfa, n, 1.0)

            # integrate perpendicular momentum
            compute_p_perp(self.d_intfcan, self.d_dpcan_sq[:, 0], self.d_dpcan_sq[:, 1], self.d_dpcan_sq[:, 2], n)
            compute_p_perp(self.d_intfkin, self.d_dpkin_sq[:, 0], self.d_dpkin_sq[:, 1], self.d_dpkin_sq[:, 2], n)
            compute_p_perp(self.d_intfa, self.d_da_sq[:, 0], self.d_da_sq[:, 1], self.d_da_sq[:, 2], n)
            compute_p_perp_A(self.d_intfcan, self.d_da, self.d_dpcanda_transp, n)
            compute_p_perp_A(self.d_intfcan, self.d_intfa, self.d_dpcanda, n)

            # calculate mean
            compute_mean(self.d_dpcan_sq[:, 0], self.d_dpcan_sq[:, 1], self.d_dpcan_sq[:, 2], self.d_dpcan_sq_mean)
            compute_mean(self.d_dpkin_sq[:, 0], self.d_dpkin_sq[:, 1], self.d_dpkin_sq[:, 2], self.d_dpkin_sq_mean)
            compute_mean(self.d_da_sq[:, 0], self.d_da_sq[:, 1], self.d_da_sq[:, 2], self.d_da_sq_mean)
            compute_mean(self.d_dpcanda[:, 0], self.d_dpcanda[:, 1], self.d_dpcanda[:, 2], self.d_dpcanda_mean)
            compute_mean(self.d_dpcanda_transp[:, 0], self.d_dpcanda_transp[:, 1], self.d_dpcanda_transp[:, 2], self.d_dpcanda_transp_mean)


def compute_ai(s, ai, t):
    u0 = s.d_u0
    aeta0 = s.d_aeta0

    n = s.n

    my_parallel_loop(compute_ai_kernel, n * n, u0, aeta0, t, ai)

@myjit
def compute_ai_kernel(xi, u0, aeta0, t, ai): 
    ax = su.mlog(u0[xi, 0])
    ay = su.mlog(u0[xi, 1])
    az = su.mul_s(aeta0[xi], 1.0 / t)

    su.store(ai[xi, 0], ax)
    su.store(ai[xi, 1], ay)
    su.store(ai[xi, 2], az)

def compute_dai(a, a0, dai, n):
    my_parallel_loop(compute_dai_kernel, n * n, a, a0, dai)  

@myjit
def compute_dai_kernel(xi, a, a0, dai):
    for i in range(3):
        su.store(dai[xi, i], l.add_mul(a[xi, i], a0[xi, i], -1))



"""
    Computes the correctly aligned (in the sense of lattice
    sites) force acting on a light-like trajectory particle.
"""

def compute_fkin(s, f):
    u0 = s.d_u0
    pt1 = s.d_pt1
    peta1 = s.d_peta1
    pt0 = s.d_pt0
    peta0 = s.d_peta0

    n = s.n
    tau = s.t 

    my_parallel_loop(compute_fkin_kernel, n * n, n, u0, peta1, peta0, pt1, pt0, f, tau)

@myjit
def compute_fkin_kernel(xi, n, u0, peta1, peta0, pt1, pt0, f, tau):
    xs = xi
    # f_1 = E_1 (index 0)
    bf0 = su.zero()

    bf0 = su.add(bf0, pt1[xs, 0])
    bf0 = su.add(bf0, pt0[xs, 0])

    xs2 = l.shift(xs, 0, -1, n)
    b1 = l.act(su.dagger(u0[xs2, 0]), pt1[xs2, 0])
    bf0 = su.add(bf0, b1)
    b1 = l.act(su.dagger(u0[xs2, 0]), pt0[xs2, 0])
    bf0 = su.add(bf0, b1)
    bf0 = su.mul_s(bf0, 0.25 / tau)
    su.store(f[xi, 0], bf0)

    # f_2 = E_2 (index 1)
    xs2 = l.shift(xs, 1, -1, n)

    bf1 = su.zero()

    # quadratically accurate +Ey
    bf1 = l.add_mul(bf1, pt1[xs, 1], 0.25 / tau)
    bf1 = l.add_mul(bf1, pt0[xs, 1], 0.25 / tau)
    xs3 = l.shift(xs, 1, -1, n)
    b1 = l.act(su.dagger(u0[xs3, 1]), pt1[xs2, 1])
    bf1 = l.add_mul(bf1, b1, 0.25 / tau)
    b1 = l.act(su.dagger(u0[xs3, 1]), pt0[xs2, 1])
    bf1 = l.add_mul(bf1, b1, 0.25 / tau)

    su.store(f[xi, 1], bf1)

    # f_3 = E_3 (index 2)
    bf2 = su.zero()

    # Accurate +E_z
    bf2 = l.add_mul(bf2, peta1[xs], 0.5)
    bf2 = l.add_mul(bf2, peta0[xs], 0.5)

    su.store(f[xi, 2], bf2)

def compute_fcan(s, f):
    aeta0 = s.d_aeta0

    n = s.n
    tau = s.t 

    my_parallel_loop(compute_fcan_kernel, n * n, aeta0, f, tau)

@myjit
def compute_fcan_kernel(xi, aeta0, f, tau):
    bf0 = su.zero()
    su.store(f[xi, 1], bf0)

    bf1 = su.mul_s(aeta0[xi], 1.0 / (tau * tau))
    su.store(f[xi, 2], bf1)

def compute_fa(fcan, fkin, fa, n):
    my_parallel_loop(compute_fa_kernel, n * n, fcan, fkin, fa)

@myjit
def compute_fa_kernel(xi, fcan, fkin, fa):
    
    for i in range(3):
        # d A = fkin - fcan convention Alatt = -igaA
        buf = l.add_mul(fkin[xi, i], fcan[xi, i], -1)
        su.store(fa[xi, i], buf)

"""
    Simple integration of forces to obtain 'color momenta'.
"""


def integrate_f(f, fi, n, dt):
    kappa.integrate_f(f, fi, n, dt, stream=None)


"""
    Computes perpendicular momentum broadening as the trace
    of the square of the integrated color force (i.e. color
    momenta).
"""


def compute_p_perp(fi, p_perp_x, p_perp_y, p_perp_z, n):
    kappa.compute_p_perp(fi, p_perp_x, p_perp_y, p_perp_z, n, stream=None)

def compute_p_perp_A(fi, d_ai, p_perp_A, n):
    my_parallel_loop(compute_p_perp_A_kernel, n * n, fi, d_ai, p_perp_A, n)

@myjit
def compute_p_perp_A_kernel(xi, fi, d_ai, p_perp_A, n):
    for i in range(3):
        p_perp_A[xi, i] = su.tr(su.mul(fi[xi, i], su.dagger(d_ai[xi, i]))).real

def compute_mean(p_perp_x, p_perp_y, p_perp_z, p_perp_mean):
    kappa.compute_mean(p_perp_x, p_perp_y, p_perp_z, p_perp_mean, stream=None)