import numpy as np
from curraun.numba_target import use_cuda
if use_cuda:
    import numba.cuda as cuda
import curraun.leapfrog as leapfrog
import curraun.leapfrog_cuda as leapfrog_cuda
import curraun.su as su

class Simulation:
    def __init__(self, n, dt, g):
        # basic parameters
        self.n = n
        self.dt = dt
        self.g = g
        nn = self.n ** 2

        # fields (times after evolve())
        self.u0 = np.empty((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) # U_{x,i}(tau_n)
        self.u1 = np.empty((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) # U_{x,i}(tau_(n+1))
        self.pt1 = np.empty((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) # P^i(tau_(n+1/2))
        self.pt0 = np.empty((nn, 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) # P^i(tau_(n-1/2))

        self.aeta0 = np.empty((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) # A_eta(tau_n)
        self.aeta1 = np.empty((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) # A_eta(tau_(n+1))
        self.peta1 = np.empty((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) # P^{eta}(tau_(n+1/2))
        self.peta0 = np.empty((nn, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE) # P^{eta}(tau_(n-1/2))

        self.data = [self.u0, self.u1, self.pt1, self.pt0, self.aeta0, self.aeta1, self.peta1, self.peta0]

        self.t = 0.0

        # Memory on the device:
        # - on CPU: contains pointer to Numpy array
        # - for CUDA: contains pointer to device (GPU) memory
        self.d_u0 = self.u0
        self.d_u1 = self.u1
        self.d_pt1 = self.pt1
        self.d_pt0 = self.pt0
        self.d_aeta0 = self.aeta0
        self.d_aeta1 = self.aeta1
        self.d_peta1 = self.peta1
        self.d_peta0 = self.peta0

        self.reset()

    def reset(self):
        # time variable
        self.t = 0.0

        self.u0[:,:,:] = 0.0
        self.u1[:,:,:] = 0.0
        self.pt1[:, :, :] = 0.0
        self.pt0[:, :, :] = 0.0

        self.aeta0[:, :] = 0.0
        self.aeta1[:, :] = 0.0
        self.peta1[:, :] = 0.0
        self.peta0[:, :] = 0.0

        self.u0[:,:,0] = 1.0
        self.u1[:,:,0] = 1.0

    def swap(self):
        self.peta1, self.peta0 = self.peta0, self.peta1
        self.pt1, self.pt0 = self.pt0, self.pt1
        self.u1, self.u0 = self.u0, self.u1
        self.aeta1, self.aeta0 = self.aeta0, self.aeta1

        # Also swap pointers to CUDA device memory
        self.d_peta1, self.d_peta0 = self.d_peta0, self.d_peta1
        self.d_pt1, self.d_pt0 = self.d_pt0, self.d_pt1
        self.d_u1, self.d_u0 = self.d_u0, self.d_u1
        self.d_aeta1, self.d_aeta0 = self.d_aeta0, self.d_aeta1

    def get_ngb(self):
        nbytes = 0
        for d in self.data:
            nbytes += d.nbytes
        return nbytes / 1024.0 ** 3

    # def get_ngb(self):
    #     return self.data.nbytes / 1024.0 ** 3

    def copy_to_device(self):
        self.d_u0 = cuda.to_device(self.u0)
        self.d_u1 = cuda.to_device(self.u1)
        self.d_pt1 = cuda.to_device(self.pt1)
        self.d_pt0 = cuda.to_device(self.pt0)
        self.d_aeta0 = cuda.to_device(self.aeta0)
        self.d_aeta1 = cuda.to_device(self.aeta1)
        self.d_peta1 = cuda.to_device(self.peta1)
        self.d_peta0 = cuda.to_device(self.peta0)

    def copy_to_host(self):
        self.d_u0.copy_to_host(self.u0)
        self.d_u1.copy_to_host(self.u1)
        self.d_pt1.copy_to_host(self.pt1)
        self.d_pt0.copy_to_host(self.pt0)
        self.d_aeta0.copy_to_host(self.aeta0)
        self.d_aeta1.copy_to_host(self.aeta1)
        self.d_peta1.copy_to_host(self.peta1)
        self.d_peta0.copy_to_host(self.peta0)


def evolve_leapfrog(s, stream=None):
    s.swap()
    s.t += s.dt
    leapfrog.evolve(s, stream)
    # leapfrog.normalize_all(s)
