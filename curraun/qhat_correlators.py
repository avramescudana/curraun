"""
    Computation of the Glasma electric and magnetic fields correlators at different times, averaged over the transverse plane.
"""

from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
import curraun.kappa as kappa
if use_cuda:
    import numba.cuda as cuda


class JetFieldsCorrelators:
    def __init__(self, s):
        self.s = s
        self.n = s.n

        self.Eform = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_Eform = self.Eform

        self.E = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_E = self.E

        self.EformE = np.zeros((self.n ** 2, 3), dtype=su.GROUP_TYPE_REAL)
        self.d_EformE = self.EformE

        self.Ux = np.zeros((self.n ** 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        my_parallel_loop(reset_wilsonfield, self.n ** 2, self.Ux)
        self.d_Ux = self.Ux

        self.Bform = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_Bform = self.Bform

        self.B = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_B = self.B

        self.BformB = np.zeros((self.n ** 2, 3), dtype=su.GROUP_TYPE_REAL)
        self.d_BformB = self.BformB

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_EformE = cuda.to_device(self.EformE)
        self.d_BformB = cuda.to_device(self.BformB)

    def copy_to_host(self):
        self.d_EformE.copy_to_host(self.EformE)
        self.EformE /= self.s.g ** 2

        self.d_BformB.copy_to_host(self.BformB)
        self.BformB /= self.s.g ** 2

    def compute_elfield(self):
        u0 = self.s.d_u0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_parallel_loop(compute_E, n ** 2, n, u0, peta1, peta0, pt1, pt0, t, self.d_E)

        if use_cuda:
            self.copy_to_host()

        return self.d_E

    def compute_magfield(self):
        u0 = self.s.d_u0
        aeta0 = self.s.d_aeta0

        t = self.s.t
        n = self.n

        my_parallel_loop(compute_B, n ** 2, n, u0, aeta0, t, self.d_B)

        if use_cuda:
            self.copy_to_host()

        return self.d_B
    
    def compute_corr(self):
        t = self.s.t
        dt = self.s.dt
        n = self.n
        # dtstep = round(1.0 / dt)

        tint = round(t/dt)
    
        if tint==1:
            compute_E(self.s, self.d_Eform)
            compute_FformF(self.s, self.d_Eform, self.d_Eform, self.d_EformE)

            compute_B(self.s, self.d_Bform) 
            compute_FformF(self.s, self.d_Bform, self.d_Bform, self.d_BformB)

        else:
            compute_E(self.s, self.d_E)
            compute_B(self.s, self.d_B)

            update_Ux(self.s, self.d_Ux, t)
            apply_Ux(self.d_E, self.d_Ux, n)
            apply_Ux(self.d_B, self.d_Ux, n)

            compute_FformF(self.s, self.d_Eform, self.d_E, self.d_EformE)
            compute_FformF(self.s, self.d_Bform, self.d_B, self.d_BformB)
        
        if use_cuda:
            self.copy_to_host()
    
        return self.d_EformE, self.d_BformB

    # def compute_elcorr(self):
    #     t = self.s.t
    #     dt = self.s.dt
    #     n = self.n
    #     # dtstep = round(1.0 / dt)

    #     tint = round(t/dt)
    
    #     if tint==1:
    #         compute_E(self.s, self.d_Eform)
    #         compute_FformF(self.s, self.d_Eform, self.d_Eform, self.d_EformE)

    #     else:
    #         compute_E(self.s, self.d_E)

    #         update_Ux(self.s, self.d_Ux, t)
    #         apply_Ux(self.d_E, self.d_Ux, n)

    #         compute_FformF(self.s, self.d_Eform, self.d_E, self.d_EformE)
        
    #     if use_cuda:
    #         self.copy_to_host()
    
    #     return self.d_EformE

    # def compute_magcorr(self):
    #     t = self.s.t
    #     dt = self.s.dt
    #     n = self.n
    #     # dtstep = round(1.0 / dt)

    #     tint = round(t/dt)
    
    #     if tint==1:
    #         compute_B(self.s, self.d_Bform) 
    #         compute_FformF(self.s, self.d_Bform, self.d_Bform, self.d_BformB)
    #     else:
    #         compute_B(self.s, self.d_B)

    #         update_Ux(self.s, self.d_Ux, t)
    #         apply_Ux(self.d_B, self.d_Ux, n)

    #         compute_FformF(self.s, self.d_Bform, self.d_B, self.d_BformB)
        
    #     if use_cuda:
    #         self.copy_to_host()
    
    #     return self.d_BformB
    
def compute_E(s, E):
    u0 = s.d_u0

    pt0 = s.d_pt0
    pt1 = s.d_pt1

    peta0 = s.d_peta0
    peta1 = s.d_peta1

    t = s.t
    n = s.n

    my_parallel_loop(compute_E_kernel, n**2, n, u0, peta1, peta0, pt1, pt0, t, E)

@myjit
def compute_E_kernel(xi, n, u0, peta1, peta0, pt1, pt0, tau, E):

    for i in range(2):
        Ei_temp1 = su.zero()
        Ei_temp2 = su.add(Ei_temp1, pt1[xi, i])
        Ei_temp3 = su.add(Ei_temp2, pt0[xi, i])
        xs = l.shift(xi, i, -1, n)
        b1 = l.act(su.dagger(u0[xs, i]), pt1[xs, i])
        Ei_temp4 = su.add(Ei_temp3, b1)
        b1 = l.act(su.dagger(u0[xs, i]), pt0[xs, i])
        Ei_temp5 = su.add(Ei_temp4, b1)
        Ei_temp6 = su.mul_s(Ei_temp5, 0.25 / tau)

        su.store(E[xi, i], Ei_temp6)
    
    Ez = su.zero()
    Ez = l.add_mul(Ez, peta1[xi], 0.5)
    Ez = l.add_mul(Ez, peta0[xi], 0.5)

    su.store(E[xi, 2], Ez)

def compute_B(s, B):
    u0 = s.d_u0
    aeta0 = s.d_aeta0

    t = s.t
    n = s.n

    my_parallel_loop(compute_B_kernel, n**2, u0, n, aeta0, t, B)

@myjit
def compute_B_kernel(xi, u0, n, aeta0, tau, B):

    for i in range(2):
        b1 = l.transport(aeta0, u0, xi, (i+1)%2, +1, n)
        b2 = l.transport(aeta0, u0, xi, (i+1)%2, -1, n)
        b2 = l.add_mul(b1, b2, -1.0)
        Bi = su.mul_s(b2, -0.5 / tau)

        su.store(B[xi, i], Bi)

    bf1 = su.zero()
    b1 = l.plaq(u0, xi, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, xi, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xi, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, xi, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    Bz = l.add_mul(bf1, b2, +0.25)

    su.store(B[xi, 2], Bz)


def apply_Ux(E, Ux, n):
    my_parallel_loop(apply_Ux_kernel, n * n, E, Ux)


@myjit
def apply_Ux_kernel(xi, E, Ux):
    for d in range(3):
        b1 = l.act(Ux[xi], E[xi, d])
        su.store(E[xi, d], b1)
        
@myjit
def reset_wilsonfield(x, wilsonfield):
    su.store(wilsonfield[x], su.unit())



def update_Ux(s, Ux, t):
    u = s.d_u0
    n = s.n

    my_parallel_loop(update_Ux_kernel, n * n, u, Ux, t, n)

@myjit
def update_Ux_kernel(xi, u, Ux, t, n):
    xs = l.shift(xi, 0, int(t), n)

    b1 = su.mul(Ux[xi], u[xs, 0])
    su.store(Ux[xi], b1)



def compute_FformF(s, Fform, F, FformF):    
    n = s.n
    t = s.t

    my_parallel_loop(compute_FformF_kernel, n**2, F, Fform, FformF, t, n)

@myjit
def compute_FformF_kernel(xi, Fform, F, FformF, t, n):
    xs = l.shift(xi, 0, int(t), n) 

    for d in range(3):
        FformF[xi, d] = su.tr(su.mul(Fform[xi, d], su.dagger(F[xs, d])))



class LongTransportedForce:
    def __init__(self, s):
        self.s = s
        self.n = s.n
        self.dtstep = round(1.0 / s.dt)

        # light-like wilson lines
        self.v = np.zeros((self.n ** 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        my_parallel_loop(reset_wilsonfield, self.n ** 2, self.v)

        # transported force
        self.f = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # integrated force
        self.fi = np.zeros((self.n ** 2, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # single components
        self.p_perp_x = np.zeros(self.n ** 2, dtype=np.double)
        self.p_perp_y = np.zeros(self.n ** 2, dtype=np.double)
        self.p_perp_z = np.zeros(self.n ** 2, dtype=np.double)

        # mean values
        self.p_perp_mean = np.zeros(3, dtype=np.double)
        if use_cuda:
            # use pinned memory for asynchronous data transfer
            self.p_perp_mean = cuda.pinned_array(3, dtype=np.double)
            self.p_perp_mean[0:3] = 0.0

        # time counter
        self.t = 0

        # Memory on the CUDA device:
        self.d_v = self.v
        self.d_f = self.f
        self.d_fi = self.fi
        self.d_p_perp_x = self.p_perp_x
        self.d_p_perp_y = self.p_perp_y
        self.d_p_perp_z = self.p_perp_z
        self.d_p_perp_mean = self.p_perp_mean

    def copy_to_device(self):
        self.d_v = cuda.to_device(self.v)
        self.d_f = cuda.to_device(self.f)
        self.d_fi = cuda.to_device(self.fi)
        self.d_p_perp_x = cuda.to_device(self.p_perp_x)
        self.d_p_perp_y = cuda.to_device(self.p_perp_y)
        self.d_p_perp_z = cuda.to_device(self.p_perp_z)
        self.d_p_perp_mean = cuda.to_device(self.p_perp_mean)

    def copy_to_host(self):
        self.d_v.copy_to_host(self.v)
        self.d_f.copy_to_host(self.f)
        self.d_fi.copy_to_host(self.fi)
        self.d_p_perp_x.copy_to_host(self.p_perp_x)
        self.d_p_perp_y.copy_to_host(self.p_perp_y)
        self.d_p_perp_z.copy_to_host(self.p_perp_z)
        self.d_p_perp_mean.copy_to_host(self.p_perp_mean)

    def copy_mean_to_device(self, stream=None):
        self.d_p_perp_mean = cuda.to_device(self.p_perp_mean, stream)

    def copy_mean_to_host(self, stream=None):
        self.d_p_perp_mean.copy_to_host(self.p_perp_mean, stream)

    def compute(self,stream=None):
        tint = round(self.s.t / self.s.dt)
        if tint % self.dtstep == 0 and tint >= 1:
            # compute un-transported f
            compute_f(self.s, self.d_f, round(self.s.t - 10E-8), stream)

            # apply parallel transport
            apply_v(self.d_f, self.d_v, self.s.n, stream)

            # integrate f
            integrate_f(self.d_f, self.d_fi, self.s.n, 1.0, stream)

            # integrate perpendicular momentum
            compute_p_perp(self.d_fi, self.d_p_perp_x, self.d_p_perp_y, self.d_p_perp_z, self.s.n, stream)

            # calculate mean
            compute_mean(self.d_p_perp_x, self.d_p_perp_y, self.d_p_perp_z, self.d_p_perp_mean, stream)

        if tint % self.dtstep == self.dtstep / 2:
            # update v
            update_v(self.s, self.d_v, round(self.s.t - 10E-8), stream)



@myjit
def reset_wilsonfield(x, wilsonfield):
    su.store(wilsonfield[x], su.unit())


"""
    "Update" the light-like Wilson line.
    Adds a single link to the Wilson line.
"""


def update_v(s, v, t, stream):
    u = s.d_u0
    n = s.n

    my_parallel_loop(update_v_kernel, n * n, u, v, t, n, stream=stream)

@myjit
def update_v_kernel(xi, u, v, t, n):
    xs = l.shift(xi, 0, t, n)

    b1 = su.mul(v[xi], u[xs, 0])
    su.store(v[xi], b1)
    #l.normalize(v[xi])


"""
    Computes the correctly aligned (in the sense of lattice
    sites) force acting on a light-like trajectory particle.
"""


def compute_f(s, f, t, stream):
    u0 = s.d_u0
    u1 = s.d_u1
    pt1 = s.d_pt1
    aeta0 = s.d_aeta0
    aeta1 = s.d_aeta1
    peta1 = s.d_peta1
    pt0 = s.d_pt0
    peta0 = s.d_peta0

    n = s.n
    dt = s.dt
    dth = s.dt / 2.0
    tau = s.t # TODO: use tau_inverse = 1/s.t to avoid division in kernel? (measurable effect?)
    sign = +1.0 # TODO: can this constant be removed?

    my_parallel_loop(compute_f_kernel, n * n, n, u0, aeta0, aeta1, peta1, peta0, pt1, pt0, f, t, tau,
                     stream=stream)

@myjit
def compute_f_kernel(xi, n, u0, aeta0, aeta1, peta1, peta0, pt1, pt0, f, t, tau):

    # f_1 = E_1 (index 0)

    xs = l.shift(xi, 0, t, n)

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

    # f_2 = E_2 - B_3 (index 1)

    xs = l.shift(xi, 0, t, n)
    xs2 = l.shift(xs, 1, -1, n)

    bf1 = su.zero()

    # # quadratically accurate +Ey
    # bf1 = l.add_mul(bf1, pt1[xs, 1], 0.25 / tau)
    # bf1 = l.add_mul(bf1, pt0[xs, 1], 0.25 / tau)
    # xs3 = l.shift(xs, 1, -1, n)
    # b1 = l.act(su.dagger(u0[xs3, 1]), pt1[xs2, 1])
    # bf1 = l.add_mul(bf1, b1, 0.25 / tau)
    # b1 = l.act(su.dagger(u0[xs3, 1]), pt0[xs2, 1])
    # bf1 = l.add_mul(bf1, b1, 0.25 / tau)

    # quadratically accurate -Bz
    b1 = l.plaq(u0, xs, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xs, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, xs, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xs, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    su.store(f[xi, 1], bf1)

    # f_3 = E_3 + B_2 (index 2)
    bf2 = su.zero()

    # Accurate +E_z
    bf2 = l.add_mul(bf2, peta1[xs], 0.5)
    bf2 = l.add_mul(bf2, peta0[xs], 0.5)

    # # Quadratically accurate +B_y
    # b1 = l.transport(aeta0, u0, xs, 0, +1, n)
    # b2 = l.transport(aeta0, u0, xs, 0, -1, n)
    # b1 = l.add_mul(b1, b2, -1.0)
    # bf2 = l.add_mul(bf2, b1, 0.5 / tau)

    su.store(f[xi, 2], bf2)


"""
    Applies the Wilson line to the untransported force.
    This is important for gauge covariance.
"""


def apply_v(f, v, n, stream):
    my_parallel_loop(apply_v_kernel, n * n, f, v, n, stream=stream)


@myjit
def apply_v_kernel(xi, f, v, n):
    for d in range(3):
        b1 = l.act(v[xi], f[xi, d])
        b1 = su.ah(b1)
        su.store(f[xi, d], b1)


"""
    Simple integration of forces to obtain 'color momenta'.
"""


def integrate_f(f, fi, n, dt, stream):
    kappa.integrate_f(f, fi, n, dt, stream)


"""
    Computes perpendicular momentum broadening as the trace
    of the square of the integrated color force (i.e. color
    momenta).
"""


def compute_p_perp(fi, p_perp_x, p_perp_y, p_perp_z, n, stream):
    kappa.compute_p_perp(fi, p_perp_x, p_perp_y, p_perp_z, n, stream)


def compute_mean(p_perp_x, p_perp_y, p_perp_z, p_perp_mean, stream):
    kappa.compute_mean(p_perp_x, p_perp_y, p_perp_z, p_perp_mean, stream)
