from curraun.numba_target import myjit, use_cuda, my_parallel_loop, my_cuda_sum, mycudajit
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda

class TransportedForce:
    def __init__(self, s, wong, n_particles):
        self.s = s
        self.n = s.n
        self.dtstep = round(1.0 / s.dt)
        self.wong = wong
        self.n_particles = n_particles

        # untransported force
        self.f = np.zeros((n_particles, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # integrated force
        self.fi = np.zeros((n_particles, 3, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # single components
        self.p_perp_x = np.zeros(n_particles, dtype=np.double)
        self.p_perp_y = np.zeros(n_particles, dtype=np.double)
        self.p_perp_z = np.zeros(n_particles, dtype=np.double)

        # mean values
        self.p_perp_mean = np.zeros(3, dtype=np.double)
        if use_cuda:
            # use pinned memory for asynchronous data transfer
            self.p_perp_mean = cuda.pinned_array(3, dtype=np.double)
            self.p_perp_mean[0:3] = 0.0

        # time counter
        self.t = 0

        # Memory on the CUDA device:
        self.d_f = self.f
        self.d_fi = self.fi
        self.d_p_perp_x = self.p_perp_x
        self.d_p_perp_y = self.p_perp_y
        self.d_p_perp_z = self.p_perp_z
        self.d_p_perp_mean = self.p_perp_mean

    def copy_to_device(self):
        self.d_f = cuda.to_device(self.f)
        self.d_fi = cuda.to_device(self.fi)
        self.d_p_perp_x = cuda.to_device(self.p_perp_x)
        self.d_p_perp_y = cuda.to_device(self.p_perp_y)
        self.d_p_perp_z = cuda.to_device(self.p_perp_z)
        self.d_p_perp_mean = cuda.to_device(self.p_perp_mean)

    def copy_to_host(self):
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

    def compute(self, wong, stream=None):
        tint = round(self.s.t / self.s.dt)
        if tint % self.dtstep == 0 and tint >= 1:
            # compute un-transported f (temporal gauge does not need any transport)
            compute_f(self.s, wong, self.d_f, self.n_particles, stream)

            # apply Wilson lines
            apply_w(wong, self.d_f, self.n_particles, stream)

            # integrate f
            integrate_f(self.d_f, self.d_fi, self.n_particles, 1.0, stream)

            # integrate perpendicular momentum
            compute_p_perp(self.d_fi, self.d_p_perp_x, self.d_p_perp_y, self.d_p_perp_z, self.n_particles, stream)

            # calculate mean
            compute_mean(self.d_p_perp_x, self.d_p_perp_y, self.d_p_perp_z, self.d_p_perp_mean, stream)

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


def compute_f(s, wong, f, n_particles, stream):
    u0 = s.d_u0
    pt1 = s.d_pt1
    aeta0 = s.d_aeta0
    peta1 = s.d_peta1
    pt0 = s.d_pt0
    peta0 = s.d_peta0

    n = s.n
    t = s.t

    x0 = wong.x0
    p = wong.p

    my_parallel_loop(compute_f_kernel, n_particles, x0, p, pt0, pt1, u0, peta0, peta1, aeta0, n, t, f, stream=stream)
    

@myjit
def compute_f_kernel(index, x0, p, pt0, pt1, u0, peta0, peta1, aeta0, n, t, f):
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

def apply_w(wong, f, n_particles, stream):
    w = wong.w
    my_parallel_loop(apply_w_kernel, n_particles, f, w, stream=stream)


@myjit
def apply_w_kernel(index, f, w):
    for d in range(3):
        b1 = l.act(w[index], f[index, d])
        su.store(f[index, d], b1)

def integrate_f(f, fi, n_particles, dt, stream):
    my_parallel_loop(integrate_f_kernel, n_particles, f, fi, dt, stream=stream)

@myjit
def integrate_f_kernel(index, f, fi, dt):
    for d in range(3):
        bfi = l.add_mul(fi[index, d], f[index, d], dt)
        su.store(fi[index, d], bfi)


def compute_p_perp(fi, p_perp_x, p_perp_y, p_perp_z, n_particles, stream):
    my_parallel_loop(compute_p_perp_kernel, n_particles, fi, p_perp_x, p_perp_y, p_perp_z, stream=stream)


@myjit
def compute_p_perp_kernel(index, fi, p_perp_x, p_perp_y, p_perp_z):
    p_perp_x[index] = su.sq(fi[index, 0])
    p_perp_y[index] = su.sq(fi[index, 1])
    p_perp_z[index] = su.sq(fi[index, 2])


def compute_mean(p_perp_x, p_perp_y, p_perp_z, p_perp_mean, stream):
    if use_cuda:
        my_cuda_sum(p_perp_x, stream)
        my_cuda_sum(p_perp_y, stream)
        my_cuda_sum(p_perp_z, stream)
        collect_results[1, 1, stream](p_perp_mean, p_perp_x, p_perp_y, p_perp_z)
    else:
        p_perp_mean[0] = np.mean(p_perp_x)
        p_perp_mean[1] = np.mean(p_perp_y)
        p_perp_mean[2] = np.mean(p_perp_z)

@mycudajit 
def collect_results(p_perp_mean, p_perp_x, p_perp_y, p_perp_z):
    p_perp_mean[0] = p_perp_x[0] / p_perp_x.size
    p_perp_mean[1] = p_perp_y[0] / p_perp_y.size
    p_perp_mean[2] = p_perp_z[0] / p_perp_z.size
