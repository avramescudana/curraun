from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda

"""
    A module to perform the LC gauge transformation of the Glasma fields at each x^+ slice
"""

class LCGaugeTransf:
    def __init__(self, s, nplus):
        self.s = s
        self.n = s.n
        self.t = s.t
        self.dts = round(1.0 / s.dt)
        self.nplus = nplus
        
        # We create an object to store the Ux at every time step
        self.ux = np.zeros((self.n**2 * nplus, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create an object to store the Aeta at every time step
        self.aeta = np.zeros((self.n**2 * nplus, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # We create the pointers to the GPU
        self.d_ux = self.ux
        self.d_aeta = self.aeta
        
        self.initialized = False

    # Copies the CPU objects to the GPU
    def copy_to_device(self):
        self.d_ux = cuda.to_device(self.ux)
        self.d_aeta = cuda.to_device(self.aeta)

    # Copies back the transformed field to the CPU
    def copy_to_host(self):
        self.d_ux.copy_to_host(self.ux)
        self.d_aeta.copy_to_host(self.aeta)

    # We copy the fields to the GPU
    def init(self):
        if use_cuda:
            self.copy_to_device()

        self.initialized = True
    
    # We initialize the simulation    
    def initialize_lc(self):
        
        # We copy the objects to the GPU if they have not been copied yet
        if not self.initialized:
            self.init()
        

    # We evolve the gauge transformation
    def evolve_lc(self, xplus):
        
        # We copy the fields at this tau to the objects ux and aeta
        store_timestep(self.d_ux, self.d_aeta, self.s.d_u1, self.s.d_aeta1, xplus, self.s.n)

        if use_cuda:
            self.copy_to_host()


# We define the function to store the Glasma fields at  a given tau step
def store_timestep(ux, aeta, u1, aeta1, n, tau):
    my_parallel_loop(store_timestep_kernel, n*n, tau, n, u1, aeta1, ux, aeta)

@myjit
def store_timestep_kernel(yi, tau, n, u1, aeta1, ux, aeta):
    txy_latt = l.get_index_nm(tau, yi, n*n)
    ux_latt = u1[yi, 0, :]
    aeta_latt = aeta[yi, :]
    
    su.store(ux[txy_latt], ux_latt)
    su.store(aeta[txy_latt], aeta_latt)