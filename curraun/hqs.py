"""
    Module for evaluating the angle between quark antiquark pairs in a toy model computation
"""
import math
from curraun.numba_target import use_cuda, myjit, my_parallel_loop
if use_cuda:
    import numba.cuda as cuda
import numpy as np

class Angles:
    def __init__(self, wong, n_particles):
        self.wong = wong
        self.n_particles = n_particles

        # angles
        self.angle = np.zeros(n_particles, dtype=np.double)
        # pTs
        self.pT = np.zeros(n_particles, dtype=np.double)

        # set-up device pointers
        self.d_angle = self.angle
        self.d_pT = self.pT

        # move data to GPU
        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_angle = cuda.to_device(self.angle)
        self.d_pT = cuda.to_device(self.pT)

    def copy_to_host(self):
        self.d_angle.copy_to_host(self.angle)
        self.d_pT.copy_to_host(self.pT)

    def compute(self):

        my_parallel_loop(compute_angles_kernel, self.n_particles, self.d_angle, self.d_pT, self.wong.d_p0, self.wong.d_p)

        if use_cuda:
            self.copy_to_host()

@myjit
def dot_product(a, b):

    """
        Computes the dot product between 2 vectors
    """
    c = a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
    return c

@myjit
def cross_product(a, b):

    """
        Computes the cross product between 2 vectors
    """

    c1 = a[1]*b[2]-a[2]*b[1]
    c2 = a[2]*b[0]-a[0]*b[2]
    c3 = a[0]*b[1]-a[1]*b[0]

    return c1, c2, c3

@myjit
def compute_angles_kernel(index, angle, pT, p0, p):
    # pq = [-p0[index, 1], -p0[index, 2], -p0[index, 4]]
    # paq = [p[index, 1], p[index, 2], p[index, 4]]
    # n = [0, 0, 1]
    pq = (-p0[index, 1], -p0[index, 2], -p0[index, 4])
    paq = (p[index, 1], p[index, 2], p[index, 4])
    n = (0, 0, 1)
    cos_theta = dot_product(pq, paq)
    sin_theta = dot_product(cross_product(pq, paq), n)
    angle_rad = math.atan2(sin_theta, cos_theta)
    angle[index] = 180 * angle_rad / math.pi
    if angle[index]<0:
        angle[index] += 360
    # angle[index] = math.atan2(sin_theta, cos_theta)
    # if angle[index]<0:
    #     angle[index] += 2*math.pi

    pT[index] = math.sqrt(p[index, 1]**2 + p[index, 2] **2)

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