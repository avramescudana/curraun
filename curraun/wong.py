"""
    A general solver for Wong's equations without back-reaction.
"""

from curraun.numba_target import use_cuda, myjit, my_parallel_loop
import curraun.lattice as l
import curraun.su as su
import curraun.kappa as kappa

if use_cuda:
    import numba.cuda as cuda
import numpy as np
from scipy.stats import unitary_group
from math import sqrt

import os
su_group = os.environ["GAUGE_GROUP"]

import math

WONG_TO_HOST = False       # copy_to_host() all variables from the WongSolver
CASIMIRS = False        # compute Casimir invariants
BOUNDARY = 'periodic'           # 'periodic' or 'frozen'
NUM_CHECK = False            # checks the p^\tau constaint

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
        self.p = np.zeros((n_particles, 5), dtype=su.GROUP_TYPE_REAL)
        self.q0 = np.zeros((n_particles, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)  # initial color charge
        self.q = np.zeros((n_particles, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # store initial momenta, used in computing momentum broadenings
        self.p0 = np.zeros((n_particles, 5), dtype=su.GROUP_TYPE_REAL)

        # particle properties (mass)
        self.m = np.zeros(n_particles, dtype=su.GROUP_TYPE_REAL)

        # active or deactivated particles
        self.active = np.ones(n_particles, dtype=np.int64)

        # wilson line for each charge
        self.w = np.zeros((n_particles, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # casimirs
        if CASIMIRS:
            self.c = np.zeros((n_particles, su.CASIMIRS), dtype=su.GROUP_TYPE_REAL)

        # p^\tau constaint
        if NUM_CHECK:
            self.ptau_constraint = np.zeros((n_particles, 2), dtype=su.GROUP_TYPE_REAL)

        # px^2, py^2 and pz^2 for each particle
        self.p_sq_x = np.zeros(n_particles, dtype=np.double)
        self.p_sq_y = np.zeros(n_particles, dtype=np.double)
        self.p_sq_z = np.zeros(n_particles, dtype=np.double)

        # pi^2 averaged over all particles, with i=x,y,z
        self.p_sq_mean = np.zeros(3, dtype=np.double)
        if use_cuda:
            # use pinned memory for asynchronous data transfer
            self.p_sq_mean = cuda.pinned_array(3, dtype=np.double)
            self.p_sq_mean[0:3] = 0.0

        # set-up device pointers
        self.d_x0 = self.x0
        self.d_x1 = self.x1
        self.d_p = self.p
        self.d_q = self.q
        self.d_q0 = self.q0
        self.d_m = self.m
        self.d_active = self.active
        self.d_w = self.w

        self.d_p0 = self.p0
        self.d_p_sq_x = self.p_sq_x
        self.d_p_sq_y = self.p_sq_y
        self.d_p_sq_z = self.p_sq_z
        self.d_p_sq_mean = self.p_sq_mean

        if CASIMIRS:
            self.d_c = self.c

        if NUM_CHECK:
            self.d_ptau_constraint = self.ptau_constraint

        # move data to GPU
        if use_cuda:
            self.copy_to_device()

        self.initialized = False

    def copy_to_device(self):
        self.d_x0 = cuda.to_device(self.x0)
        self.d_x1 = cuda.to_device(self.x1)
        self.d_p = cuda.to_device(self.p)
        self.d_q = cuda.to_device(self.q)
        self.d_q0 = cuda.to_device(self.q0)
        self.d_m = cuda.to_device(self.m)
        self.d_active = cuda.to_device(self.active)
        self.d_w = cuda.to_device(self.w)

        if CASIMIRS:
            self.d_c = cuda.to_device(self.c)

        if NUM_CHECK:
            self.d_ptau_constraint = cuda.to_device(self.ptau_constraint)

        self.d_p_sq_x = cuda.to_device(self.p_sq_x)
        self.d_p_sq_y = cuda.to_device(self.p_sq_y)
        self.d_p_sq_z = cuda.to_device(self.p_sq_z)
        self.d_p_sq_mean = cuda.to_device(self.p_sq_mean)

    def copy_to_host(self):
        self.d_x0.copy_to_host(self.x0)
        self.d_x1.copy_to_host(self.x1)
        self.d_p.copy_to_host(self.p)
        self.d_q.copy_to_host(self.q)
        self.d_q0.copy_to_host(self.q0)
        self.d_m.copy_to_host(self.m)
        self.d_active.copy_to_host(self.active)
        self.d_w.copy_to_host(self.w)

        if CASIMIRS:
            self.d_c.copy_to_host(self.c)

        if NUM_CHECK:
            self.d_ptau_constraint.copy_to_host(self.ptau_constraint)

        self.d_p_sq_x.copy_to_host(self.p_sq_x)
        self.d_p_sq_y.copy_to_host(self.p_sq_y)
        self.d_p_sq_z.copy_to_host(self.p_sq_z)
        self.d_p_sq_mean.copy_to_host(self.p_sq_mean)

    def copy_mom_broad_to_device(self, stream=None):
        self.d_p_sq_mean = cuda.to_device(self.p_sq_mean, stream)

    def copy_mom_broad_to_host(self, stream=None):
        self.d_p_sq_mean.copy_to_host(self.p_sq_mean, stream)

    def init(self):
        if use_cuda:
            self.copy_to_device()

        my_parallel_loop(init_momenta_kernel, self.n_particles,
                         self.d_p, self.d_p0, self.d_m, self.s.t, self.d_x0)

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

        my_parallel_loop(init_charge_kernel, self.n_particles, self.q0, q0s)
        my_parallel_loop(init_wilson_lines_kernel, self.n_particles, self.w)

    def evolve(self):
        if not self.initialized:
            self.init()

        # update positions and perform parallel transport
        my_parallel_loop(update_coordinates_kernel, self.n_particles,
                         self.d_x0, self.d_x1, self.d_p, self.d_q, self.d_q0,
                         self.s.dt, self.s.d_u0, self.s.d_aeta0, self.s.n, self.d_w, self.d_active)

        # update momenta
        my_parallel_loop(update_momenta_kernel, self.n_particles,
                         self.d_x1, self.d_p, self.d_q, self.d_m,
                         self.s.t, self.s.dt, self.s.d_u0, self.s.d_pt0, self.s.d_pt1,
                         self.s.d_aeta0, self.s.d_peta0, self.s.d_peta1,
                         self.s.n, self.d_active)

        # swap variables
        self.d_x0, self.d_x1 = self.d_x1, self.d_x0
        self.x0, self.x1 = self.x1, self.x0

        if WONG_TO_HOST:
            if use_cuda:
                self.copy_to_host()

    def compute_mom_broad(self, stream=None):
        # compute momenta components squared
        compute_p_sq(self.n_particles, self.d_p0, self.d_p, self.d_p_sq_x, self.d_p_sq_y, self.d_p_sq_z, stream)

        # compute mean
        kappa.compute_mean(self.d_p_sq_x, self.d_p_sq_y, self.d_p_sq_z,  self.d_p_sq_mean, stream)

        if use_cuda:
            self.copy_mom_broad_to_host()

    def compute_casimirs(self, repr):
        if repr=='fundamental':
            my_parallel_loop(compute_casimirs_fundamental_kernel, self.n_particles, self.d_q, self.d_c)
        elif repr=='adjoint':
            my_parallel_loop(compute_casimirs_adjoint_kernel, self.n_particles, self.d_q, self.d_c)

        if use_cuda:
            self.copy_to_host()
@myjit
def init_charge_kernel(index, q0, q0s):
    q0_algebra = su.get_algebra_element(q0s[index, :])
    q0[index, :] = q0_algebra[:]


@myjit
def init_wilson_lines_kernel(index, w):
    w[index, :] = su.unit()


@myjit
def compute_electric_field(ngp_index, pt0, pt1, u0, peta0, peta1, n, t):
    # Electric fields
    # Ex
    Ex = su.zero()
    Ex = su.add(Ex, pt1[ngp_index, 0, :])
    Ex = su.add(Ex, pt0[ngp_index, 0, :])
    xs = l.shift(ngp_index, 0, -1, n)
    b1 = l.act(su.dagger(u0[xs, 0, :]), pt1[xs, 0, :])
    Ex = su.add(Ex, b1)
    b1 = l.act(su.dagger(u0[xs, 0]), pt0[xs, 0, :])
    Ex = su.add(Ex, b1)
    Ex = su.mul_s(Ex, 0.25 / t)

    # Ey
    Ey = su.zero()
    Ey = su.add(Ey, pt1[ngp_index, 1, :])
    Ey = su.add(Ey, pt0[ngp_index, 1, :])
    xs = l.shift(ngp_index, 1, -1, n)
    b1 = l.act(su.dagger(u0[xs, 1, :]), pt1[xs, 1, :])
    Ey = su.add(Ey, b1)
    b1 = l.act(su.dagger(u0[xs, 1, :]), pt0[xs, 1, :])
    Ey = su.add(Ey, b1)
    Ey = su.mul_s(Ey, 0.25 / t)

    # Eeta
    Eeta = su.zero()
    Eeta = l.add_mul(Eeta, peta1[ngp_index, :], 0.5)
    Eeta = l.add_mul(Eeta, peta0[ngp_index, :], 0.5)

    return Ex, Ey, Eeta


@myjit
def compute_magnetic_field(ngp_index, aeta0, u0, n, t):
    # Magnetic fields
    # Bx
    b1 = l.transport(aeta0, u0, ngp_index, 1, +1, n)
    b2 = l.transport(aeta0, u0, ngp_index, 1, -1, n)
    b2 = l.add_mul(b1, b2, -1.0)
    Bx = su.mul_s(b2, -0.5 / t)

    # By
    b1 = l.transport(aeta0, u0, ngp_index, 0, +1, n)
    b2 = l.transport(aeta0, u0, ngp_index, 0, -1, n)
    b2 = l.add_mul(b1, b2, -1.0)
    By = su.mul_s(b2, +0.5 / t)

    # Bz (Beta)
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
def init_momenta_kernel(index, p, p0, m, tau, x0):
    """
        Initializes the tau and z component of the momentum vectors
    """
    p[index, 0] = sqrt(p[index, 1] ** 2 + p[index, 2] ** 2 + (tau * p[index, 3]) ** 2 + m[index] ** 2)

    eta = x0[index, 2]
    p[index, 4] = math.sinh(eta) * p[index, 0] + math.cosh(eta) * tau * p[index, 3]

    for i in range(5):
        p0[index, i] = p[index, i]


@myjit
def update_coordinates_kernel(index, x0, x1, p, q, q0, dt, u0, aeta0, n, w, active):
    if BOUNDARY == 'frozen':
        for d in range(2):
            if x0[index, d] < 0 or x0[index, d] > n:
                active[index] = 0
                # set x1 to x0 such that they are not affected by swapping x0 with x1 later on
                x1[index, d] = x0[index, d]

    if active[index] == 1:
        """
            Solve coordinate and charge equations
        """
        # x, y update
        ptau = p[index, 0]
        for d in range(2):
            x1[index, d] = x0[index, d] + p[index, d + 1] / ptau * dt
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
                # w[index, :] = su.mul(w[index, :], u0[ngp0_index, d])
                w[index, :] = su.mul(w[index, :], u0[ngp1_index, d])

            if delta_ngp == -1:
                # U = su.dagger(u0[ngp1_index, d])
                # q1 = su.mul(su.mul(U, q[index]), su.dagger(U))
                w[index, :] = su.mul(w[index, :], su.dagger(u0[ngp1_index, d]))

        # parallel transport charge along eta
        delta_eta = x1[index, 2] - x0[index, 2]
        U = su.mexp(su.mul_s(aeta0[ngp0_index], delta_eta))
        w[index, :] = su.mul(w[index, :], U)

        # apply the wilson lines to the color charge
        q1 = su.ah(l.act(su.dagger(w[index, :]), q0[index]))
        q[index, :] = q1[:]


@myjit
def update_momenta_kernel(index, x1, p, q, m, t, dt, u0, pt0, pt1, aeta0, peta0, peta1, n, active):
    if active[index] == 1:
        """
            Computes force acting on particle and updates momenta accordingly
        """
        # TODO: What is the right position here?
        ngp_pos = ngp(x1[index, :2])
        ngp_index = l.get_index(ngp_pos[0], ngp_pos[1], n)

        """
            Compute tr(QE) and tr(QB)
        """
        Ex, Ey, Eeta = compute_electric_field(ngp_index, pt0, pt1, u0, peta0, peta1, n, t)
        Bx, By, Beta = compute_magnetic_field(ngp_index, aeta0, u0, n, t)

        Q0 = q[index]

        trQE = su.tr(su.mul(Q0, Ex)).real, su.tr(su.mul(Q0, Ey)).real, su.tr(su.mul(Q0, Eeta)).real
        trQB = su.tr(su.mul(Q0, Bx)).real, su.tr(su.mul(Q0, By)).real, su.tr(su.mul(Q0, Beta)).real

        """
            Force computation and momentum update
        """
        tr = -1 / 2
        mass = m[index]

        ptau0, px0, py0, peeta0 = p[index, 0], p[index, 1], p[index, 2], p[index, 3]
        vx, vy, veta = px0 / ptau0, py0 / ptau0, peeta0 / ptau0
        fx = 1.0 / tr * (trQE[0] + trQB[2] * vy - trQB[1] * veta * t)
        fy = 1.0 / tr * (trQE[1] - trQB[2] * vx + trQB[0] * veta * t)
        feta = (trQE[2] - trQB[0] * vy + trQB[1] * vx) / tr
        px1 = px0 + dt * fx
        py1 = py0 + dt * fy
        peeta1 = peeta0 + dt * (feta - 2.0 * peeta0) / t
        ptau1 = sqrt(px1 ** 2 + py1 ** 2 + (t * peeta1) ** 2 + mass ** 2)

        p[index, 0] = ptau1
        p[index, 1] = px1
        p[index, 2] = py1
        p[index, 3] = peeta1

        # compute pz from peta or from dpz/dt=feta for limiting cases (qhat.py or kappa.py)
        # if LIMITING_CASE:
        #     pz0 = p[index, 4]
        #     pz1 = pz0 + dt * feta
        # else:
        #     eta0 = x1[index, 2]
        #     pz1 = math.sinh(eta0) * ptau0 + math.cosh(eta0) * t * peeta0

        eta1 = x1[index, 2]
        pz1 = math.sinh(eta1) * ptau1 + math.cosh(eta1) * t * peeta1
        p[index, 4] = pz1

def compute_p_sq(ntp, p0, p, p_sq_x, p_sq_y,p_sq_z, stream):
    my_parallel_loop(compute_p_sq_kernel, ntp, p0, p, p_sq_x, p_sq_y, p_sq_z, stream=stream)

@myjit
def compute_p_sq_kernel(index, p0, p, p_sq_x, p_sq_y, p_sq_z):
    p_sq_x[index] = (p[index, 1] - p0[index, 1]) ** 2 
    p_sq_y[index] = (p[index, 2] - p0[index, 2]) ** 2 
    p_sq_z[index] = (p[index, 4] - p0[index, 4]) ** 2

# @myjit
# def swap_coordinates_kernel(index, x0, x1, active):
#     if active[index] == 1:
#         for d in range(3):
#             x0[index, d], x1[index, d] = x1[index, d], x0[index, d]

@myjit
def compute_casimirs_fundamental_kernel(index, q, c):
    c[index, :] = su.casimir_fundamental(q[index, :])

@myjit
def compute_casimirs_adjoint_kernel(index, q, c):
    c[index, :] = su.casimir_adjoint(q[index, :])

# gell-mann matrices

gm = [
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(3), 0], [0, 0, -2 / np.sqrt(3)]]
]

T = np.array(gm) / 2.0

def init_pos(n):
    """
        Particles randomly distributed in the transverse plane
        TODO: Initialize them according to local fluctuations in energy density at formation time
    """

    xT = np.random.rand(2) * n
    x0 = [xT[0], xT[1], 0.0]

    return x0

def init_mom(type_init, p):
    """
        Initialize all particles with the same initial transverse momentum
        TODO: Add other types of initializations (FONLL for HQs, for example)
    """

    if type_init=='pT':
        angle = 2*np.pi*np.random.rand(1)
        p0 = [0.0, p * np.cos(angle), p * np.sin(angle), 0.0, 0.0]
    elif type_init=='px':
        p0 = [0.0, p, 0.0, 0.0, 0.0]

    return p0

def init_charge(representation):
    if su_group == 'su2':

        """
            Random color charges uniformly distributed on sphere of fixed radius
        """

        if representation=='fundamental':
            q2 = 3/2
        elif representation=='adjoint':
            q2 = 6
    
        J = np.sqrt(q2)
        phi, pi = np.random.uniform(0, 2 * np.pi), np.random.uniform(-J, J)
        Q1 = np.cos(phi) * np.sqrt(J ** 2 - pi ** 2)
        Q2 = np.sin(phi) * np.sqrt(J ** 2 - pi ** 2)
        Q3 = pi
        q0 = np.array([Q1, Q2, Q3])

        return q0

    elif su_group == 'su3':

        """
            Step 1: specific random color vector
        """

        if representation=='fundamental':
            q0 = [0., 0., 0., 0., -1.69469, 0., 0., -1.06209]
        elif representation=='adjoint':
            q0 = [np.sqrt(24), 0., 0., 0., 0., 0., 0., 0.]
        Q0 = np.einsum('ijk,i', T, q0)

        """
            Step 2: create a random SU(3) matrix to rotate Q.
        """
        
        V = unitary_group.rvs(3)
        detV = np.linalg.det(V)
        U = V / detV ** (1 / 3)
        Ud = np.conj(U).T

        Q = np.einsum('ab,bc,cd', U, Q0, Ud)

        """
            Step 3: Project onto color components
        """

        q = 2 * np.einsum('ijk,kj', T, Q)
        return np.real(q)