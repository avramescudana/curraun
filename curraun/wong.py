"""
    A general solver for Wong's equations without back-reaction.
"""

from curraun.numba_target import use_cuda, myjit, my_parallel_loop
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda
import numpy as np
from scipy.stats import unitary_group
from math import sqrt
DEBUG = True

import os
su_group = os.environ["GAUGE_GROUP"]
boundary = os.environ["BOUNDARY"]

class WongSolver:
    def __init__(self, s, n_particles):
        """
        A note regarding units:
        Mass and momenta are given in units of lattice spacing
        Positions are within the interval [0, n), where n is the number of lattice cells along one direction
        """
        self.s = s
        self.n_particles = n_particles
        self.allocated = 0

        # particle degrees of freedom (positions, momenta, charges)
        # position layout (x0, x1): x,y,eta
        # momentum layout (p): tau,x,y,eta
        self.x0 = np.zeros((n_particles, 3), dtype=su.GROUP_TYPE_REAL)
        self.x1 = np.zeros((n_particles, 3), dtype=su.GROUP_TYPE_REAL)
        self.p = np.zeros((n_particles, 4), dtype=su.GROUP_TYPE_REAL)
        self.q = np.zeros((n_particles, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # particle properties (mass)
        self.m = np.zeros(n_particles, dtype=su.GROUP_TYPE_REAL)

        # active or deactivated particles
        self.active = np.ones(n_particles, dtype=np.int64)

        # wilson line for each charge
        self.w = np.zeros((n_particles, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # casimirs
        if DEBUG:
            self.c = np.zeros((n_particles, su.CASIMIRS), dtype=su.GROUP_TYPE_REAL)

        # set-up device pointers
        self.d_x0 = self.x0
        self.d_x1 = self.x1
        self.d_p = self.p
        self.d_q = self.q
        self.d_m = self.m
        self.d_active = self.active
        self.d_w = self.w

        if DEBUG:
            self.d_c = self.c

        # move data to GPU
        if use_cuda:
            self.copy_to_device()

        self.initialized = False

    def copy_to_device(self):
        self.d_x0 = cuda.to_device(self.x0)
        self.d_x1 = cuda.to_device(self.x1)
        self.d_p = cuda.to_device(self.p)
        self.d_q = cuda.to_device(self.q)
        self.d_m = cuda.to_device(self.m)
        self.d_active = cuda.to_device(self.active)
        self.d_w = cuda.to_device(self.w)

        if DEBUG:
            self.d_c = cuda.to_device(self.c)

    def copy_to_host(self):
        self.d_x0.copy_to_host(self.x0)
        self.d_x1.copy_to_host(self.x1)
        self.d_p.copy_to_host(self.p)
        self.d_q.copy_to_host(self.q)
        self.d_m.copy_to_host(self.m)
        self.d_active.copy_to_host(self.active)
        self.d_w.copy_to_host(self.w)

        if DEBUG:
            self.d_c.copy_to_host(self.c)

    def init(self):
        if use_cuda:
            self.copy_to_device()

        my_parallel_loop(init_momenta_kernel, self.n_particles,
                         self.d_p, self.d_m, self.s.t)

        self.initialized = True

    # def add_particle(self, x0, p0, q0, m):
    #     i = self.allocated
    #     if i < self.n_particles:
    #         self.x0[i, :] = x0[:]
    #         self.x1[i, :] = x0[:]
    #         self.p[i, :] = p0[:]
    #         q0_algebra = su.get_algebra_element(q0)
    #         self.q[i, :] = q0_algebra[:]

    #         self.m[i] = m
    #         self.allocated += 1
    #     else:
    #         raise Exception("Maximum number of particles reached.")

    def initialize(self, x0s, p0s, q0s, masses):

        self.x0[:, :] = x0s[:, :]
        self.x1[:, :] = x0s[:, :]
        self.p[:, :] = p0s[:, :]
        self.m[:] = masses[:]

        my_parallel_loop(init_charge_kernel, self.n_particles, self.q, q0s)
        my_parallel_loop(init_wilson_lines_kernel, self.n_particles, self.w)


    def evolve(self):
        if not self.initialized:
            self.init()

        # update positions and perform parallel transport

        my_parallel_loop(update_coordinates_kernel, self.n_particles,
                         self.d_x0, self.d_x1, self.d_p, self.d_q,
                         self.s.dt, self.s.d_u0, self.s.d_aeta0, self.s.n, self.d_w, self.d_active)

        # update momenta

        my_parallel_loop(update_momenta_kernel, self.n_particles,
                         self.d_x1, self.d_p, self.d_q, self.d_m,
                         self.s.t, self.s.dt, self.s.d_u0, self.s.d_pt0, self.s.d_pt1,
                         self.s.aeta0, self.s.peta0, self.s.peta1,
                         self.s.n, self.d_active)


        # swap variables
        my_parallel_loop(swap_coordinates_kernel, self.n_particles, self.d_x0, self.d_x1, self.d_active)
        # self.d_x0, self.d_x1 = self.d_x1, self.d_x0

        if use_cuda:
            self.copy_to_host()

    def compute_casimirs(self):
        my_parallel_loop(compute_casimirs_kernel, self.n_particles, self.d_q, self.d_c)

        if use_cuda:
            self.copy_to_host()

  
@myjit
def init_charge_kernel(index, q, q0s):
    q0_algebra = su.get_algebra_element(q0s[index, :])
    q[index, :] = q0_algebra[:]

@myjit
def init_wilson_lines_kernel(index, w):
    w[index, :] = su.unit()

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
def ngp(x):
    """
        Computes the nearest grid point of a particle position
    """
    return int(round(x[0])), int(round(x[1]))


@myjit
def init_momenta_kernel(index, p, m, tau):
    """
        Initializes the tau component of the momentum vectors
    """
    # TODO: check tau factor
    p[index, 0] = sqrt(p[index, 1] ** 2 + p[index, 2] ** 2 + (tau * p[index, 3]) ** 2 + m[index] ** 2)


@myjit
def update_coordinates_kernel(index, x0, x1, p, q, dt, u0, aeta0, n, w, active):
    if boundary=='frozen':
        for d in range(1):
            if x0[index, d] < 0 or x0[index, d]>n:
                active[index] = 0

    if active[index]==1:
        """
            Solve coordinate and charge equations
        """
        # x, y update
        ptau = p[index, 0]
        for d in range(2):
            x1[index, d] = x0[index, d] + p[index, d+1] / ptau * dt
        # eta update
        x1[index, 2] = x0[index, 2] + p[index, 3] / ptau * dt

        # check for nearest-grid-point changes and update charge accordingly
        ngp0 = ngp(x0[index, :2])
        ngp1 = ngp(x1[index, :2])
        ngp0_index = l.get_index(ngp0[0], ngp0[1], n)  # this takes care of bounds
        ngp1_index = l.get_index(ngp1[0], ngp0[1], n)  # ...

        for d in range(2):
            delta_ngp = ngp1[d] - ngp0[d]
            if delta_ngp == 1:
                # U = u0[ngp0_index, d]
                # q1 = su.mul(su.mul(U, q[index]), su.dagger(U))
                w[index, :] = su.mul(w[index, :], u0[ngp0_index, d])

            if delta_ngp == -1:
                # U = su.dagger(u0[ngp1_index, d])
                # q1 = su.mul(su.mul(U, q[index]), su.dagger(U))
                w[index, :] = su.mul(w[index, :], su.dagger(u0[ngp1_index, d]))

        # parallel transport charge along eta
        delta_eta = x1[index, 2] - x0[index, 2]
        U = su.mexp(su.mul_s(aeta0[ngp0_index], delta_eta))
        w[index, :] = su.mul(w[index, :], U)

        # apply the wilson lines to the color charge
        q1 = su.mul(su.mul(w[index, :], q[index]), su.dagger(w[index, :]))
        q[index, :] = q1[:]


@myjit
def update_momenta_kernel(index, x1, p, q, m, t, dt, u0, pt0, pt1, aeta0, peta0, peta1, n, active):

    if active[index]==1:
        """
            Computes force acting on particle and updates momenta accordingly
        """
        # TODO: What is the right position here?
        ngp_pos = ngp(x1[index, :2])
        ngp_index = l.get_index(ngp_pos[0], ngp_pos[1], n)

        """
            Compute tr(QE) and tr(QB)
        """
        # TODO: Check these formulas

        Ex, Ey, Eeta = compute_electric_field(ngp_index, pt0, pt1, u0, peta0, peta1, n, t)
        Bx, By, Beta = compute_magnetic_field(ngp_index, aeta0, u0, n, t)

        Q0 = q[index]

        trQE = su.tr(su.mul(Q0, Ex)).real, su.tr(su.mul(Q0, Ey)).real, su.tr(su.mul(Q0, Eeta)).real
        trQB = su.tr(su.mul(Q0, Bx)).real, su.tr(su.mul(Q0, By)).real, su.tr(su.mul(Q0, Beta)).real

        """
            Force computation and momentum update
        """
        # TODO: Check these formulas

        tr = -1/2

        ptau0, px0, py0, peeta0 = p[index, 0], p[index, 1], p[index, 2], p[index, 3]
        mass = m[index]
        px1 = px0 + dt / tr * (trQE[0] + trQB[2] * py0 / ptau0 - trQB[1] * peeta0 * t / ptau0)
        py1 = py0 + dt / tr * (trQE[1] - trQB[2] * px0 / ptau0 + trQB[0] * peeta0 * t / ptau0)
        peeta1 = peeta0 + dt * ((trQE[2] * ptau0 - trQB[0] * py0 + trQB[1] * px0) / tr - 2 * peeta0 * ptau0) / (t * ptau0)
        # peeta1 = peeta0 + dt * ((trQE[2] * ptau0 - trQB[0] * py0 + trQB[1] * px0) / tr - peeta0 * ptau0) / (t * ptau0)
        # pz0 = peeta0 * t
        # pz1 = pz0 + dt / tr * (trQE[2] - trQB[0] * py0 / ptau0 + trQB[1] * px0 * t / ptau0)
        # peeta1 = pz1 / t

        ptau1 = sqrt(px1 ** 2 + py1 ** 2 + (t * peeta1) ** 2 + mass ** 2)

        p[index, 0] = ptau1
        p[index, 1] = px1
        p[index, 2] = py1
        p[index, 3] = peeta1

@myjit
def swap_coordinates_kernel(index, x0, x1, active):
    if active[index]==1:
        for d in range(3):
            x0[index, d], x1[index, d] = x1[index, d], x0[index, d]

@myjit
def compute_casimirs_kernel(index, q, c):
    c[index, :] = su.casimir(q[index, :])

# gell-mann matrices

gm = [
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1/ np.sqrt(3), 0, 0], [0, 1/ np.sqrt(3), 0], [0, 0, -2/ np.sqrt(3)]]
]

T = np.array(gm) / 2.0

def init_charge():
    if su_group=='su2':

        """
            Random color charges uniformly distributed on sphere of fixed radius
        """

        J = np.sqrt(1.5)
        #TODO fix J for adjoint representation
        phi, pi = np.random.uniform(0, 2*np.pi), np.random.uniform(-J, J)
        Q1 = np.cos(phi) * np.sqrt(J**2 - pi**2)
        Q2 = np.sin(phi) * np.sqrt(J**2 - pi**2)
        Q3 = pi
        q0 = np.array([Q1, Q2, Q3])
        return q0
    elif su_group=='su3':
        
        """
            New step 1: specific random color vector
        """
        #TODO find q0 for adjoint representation
        q0 = [0., 0., 0., 0., -1.69469, 0., 0., -1.06209]
        Q0 = np.einsum('ijk,i', T, q0)
        
        
        
        """
            Step 2: create a random SU(3) matrix to rotate Q.
        """
        V = unitary_group.rvs(3)
        detV = np.linalg.det(V)
        U = V / detV ** (1/3)
        Ud = np.conj(U).T
        
        Q = np.einsum('ab,bc,cd', U, Q0, Ud)
        
        """
            Step 3: Project onto color components
        """
        
        q = 2*np.einsum('ijk,kj', T, Q)
        return np.real(q)