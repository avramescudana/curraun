from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import curraun.su as su
import numpy as np
import curraun.lattice as l

if use_cuda:
    import numba.cuda as cuda



"""
    A module to get the Glamsa fields in the temporal gauge at mid-rapidity
"""

class GlasmaFields:
    
    def __init__(self, s):
        self.s = s
        self.n = s.n
    
        # We create object to store the links
        self.up = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.ay = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.az = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create the pointer to the GPU
        self.d_up = self.up
        self.d_ay = self.ay
        self.d_az = self.az
        
    # Copies from the host to the device
    def copy_to_device(self):
        self.d_up = cuda.to_device(self.up)
        self.d_ay = cuda.to_device(self.ay)
        self.d_az = cuda.to_device(self.az)
        
    # Copies from the device to the host
    def copy_to_host(self):
        self.d_up.copy_to_host(self.up)
        self.d_ay.copy_to_host(self.ay)
        self.d_az.copy_to_host(self.az)
    
    # We copy the objects to the device
    def init(self):
        if use_cuda:
            self.copy_to_device()
        
    
    # We compute the Glasma fields at the given time step
    def compute_fields(self, xplus, ap):
        compute_up(self.s.d_u1, self.d_up, self.n, xplus)
        compute_ay(self.s.d_u1, self.d_ay, self.n, xplus, ap)
        compute_az(self.s.d_aeta1, self.d_az, self.n, xplus, ap)
        
        # We copy back the results to the host
        if use_cuda:
            self.copy_to_host()


""" 
    Computes the Ux links along x^+ axis at every time step
"""
def compute_up(u1, up, n, xplus):
    my_parallel_loop(compute_up_kernel, n**2, u1, up, xplus, n)

@myjit
def compute_up_kernel(yi, u1, up, xplus, n):
    
    su.store(up[yi], su.unit())

    # if xplus == 0:
    #     su.store(up[yi], su.unit())
    
    # else:
    #     # We get the transverse indices 
    #     yz = l.get_point(yi, n)
    #     y, z = yz[0], yz[1]
    
    #     # Construct the (x, y) index
    #     xy = l.get_index(xplus, y, n)
    
    #     # Compute the corresponding U_x link
    #     ux_latt = u1[xy, 0, :]
    
    #     # We take complex conjugation
    #     res = su.dagger(ux_latt)
    
    #     su.store(up[yi], res)
    

"""
    Computes the g*Ay fields along x^+ axis at every time step
"""
def compute_ay(u1, ay, n, xplus, ap):
    my_parallel_loop(compute_ay_kernel, n**2, u1, ay, xplus, n, ap)

@myjit
def compute_ay_kernel(yi, u1, ay, xplus, n, ap):
        
    # We get the transverse indices
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    # Construct the (x, y) index
    xy = l.get_index(xplus, y, n)
    
    # Compute the corresponding U_x link
    uy = u1[xy, 1, :]
    
    # Take the logarithm
    luy = su.mlog(uy)
    
    # We extract the field
    res  = su.mul_s(luy, 1/(ap*1j))
    
    su.store(ay[yi], res)


"""
    Computes the g*Az fields along x^+ axis at every time step
"""
def compute_az(aeta1, az, n, xplus, ap):
    my_parallel_loop(compute_az_kernel, n**2, aeta1, az, xplus, n, ap)

@myjit
def compute_az_kernel(yi, aeta1, az, xplus, n, ap):
    
    if xplus == 0:
        return
    
    else:
    
        # We get the transverse indices
        yz = l.get_point(yi, n)
        y, z = yz[0], yz[1]
    
        # Construct the (x, y) index
        xy = l.get_index(xplus, y, n)
    
        # Compute the corresponding A_eta field
        aeta_latt = aeta1[xy, :]
    
        # We get the Az field
        res  = su.mul_s(aeta_latt, 1/(1j*xplus*ap))
    
        su.store(az[yi], res)


