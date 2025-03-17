from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import curraun.su as su
import numpy as np
import curraun.lattice as l

if use_cuda:
    import numba.cuda as cuda



"""
    A module to get the U_+ links in the temporal gauge at mid-rapidity
"""

class PlusLink:
    
    def __init__(self, s):
        self.s = s
        self.n = s.n
    
        # We create object to store the links
        self.ux = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.up = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create the pointer to the GPU
        self.d_ux = self.ux
        self.d_up = self.up
        
    # Copies from the host to the device
    def copy_to_device(self):
        self.d_ux = cuda.to_device(self.ux)
        self.d_up = cuda.to_device(self.up)
        
    # Copies from the device to the host
    def copy_to_host(self):
        self.d_up.copy_to_host(self.up)
    
    # We copy the objects to the device
    def init(self):
        if use_cuda:
            self.copy_to_device()
        
    
    # We compute the plus links
    def compute_plus_link(self, xplus):
        compute_ux(self.s.d_u1, self.d_ux, self.n, xplus)
        conjugate_ux(self.d_ux, self.d_up, self.n)
        
        # We copy back the results to the host
        if use_cuda:
            self.copy_to_host()


""" 
    Computes the Ux links along x^+ axis at every time step
"""
def compute_ux(u1, ux, n, xplus):
    my_parallel_loop(compute_ux_kernel, n**2, u1, ux, xplus, n)

@myjit
def compute_ux_kernel(yi, u1, ux, xplus, n):
    
    # We get the transverse indices
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    # Construct the (x, y) index
    xy = l.get_index(xplus, y, n)
    
    # Compute the corresponding U_x link
    ux_latt = u1[xy, 0, :]
    
    su.store(ux[yi], ux_latt)


""" 
    Takes the complex conjugate of the U_1x links
"""
def conjugate_ux(ux, up, n):
    my_parallel_loop(conjugate_ux_kernel, n**2, ux, up)

@myjit
def conjugate_ux_kernel(yi, ux, up):
    res = su.dagger(ux[yi])
    su.store(up[yi], res)