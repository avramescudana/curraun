from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import curraun.su as su
import numpy as np
import curraun.lattice as l
from math import sqrt

if use_cuda:
    import numba.cuda as cuda



"""
    A module to get the Glamsa fields in the temporal gauge at mid-rapidity
"""

class GlasmaFields:
    
    def __init__(self, s, ux, uy, Aeta, dts):
        self.s = s
        self.n = s.n
        self.ux = ux
        self.uy = uy
        self.Aeta = Aeta
        self.dts = dts
    
        # We create object to store the links
        self.up = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.ay = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.az = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create the pointer to the GPU
        self.d_up = self.up
        self.d_ay = self.ay
        self.d_az = self.az
        
        self.d_ux = self.ux
        self.d_uy = self.uy
        self.d_Aeta = self.Aeta
        
    # Copies from the host to the device
    def copy_to_device(self):
        self.d_up = cuda.to_device(self.up)
        self.d_ay = cuda.to_device(self.ay)
        self.d_az = cuda.to_device(self.az)
        
        self.d_ux = cuda.to_device(self.ux)
        self.d_uy = cuda.to_device(self.uy)
        self.d_Aeta = cuda.to_device(self.Aeta)
        
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
        compute_up(self.d_ux, self.d_Aeta, self.d_up, self.n, self.dts, xplus)
        compute_ay(self.d_uy, self.d_ay, self.n, self.dts, xplus, ap)
        compute_az(self.d_Aeta, self.d_az, self.n, self.dts, xplus, ap)
        
        # We copy back the results to the host
        if use_cuda:
            self.copy_to_host()


"""
    Sets the up links to the unit matrix
"""
def set_to_unit(up, n):
    my_parallel_loop(set_to_unit_kernel, n**2, up)

@myjit
def set_to_unit_kernel(yi, up):
    su.store(up[yi], su.unit())

""" 
    Computes the Ux links along x^+ axis at every time step
"""
def compute_up(ux, aeta, up, n, dts, xplus):
    my_parallel_loop(compute_up_kernel, n**2, ux, aeta, up, dts, xplus, n)

@myjit
def compute_up_kernel(yi, ux, aeta, up, dts, xplus, n):
    
    # We get the transverse indices 
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    # Rearrange the indices
    if z>= n/2:
        z = z - n
    
    if xplus > abs(z):
        # Obtain the (tau, x, y) index
        tau = round(dts*sqrt(xplus**2 - z**2))
        tauxy = l.get_index_n2xm(tau, xplus, y, n)
    
        # Compute the corresponding U_x link
        ux_latt = ux[tauxy, :]
        ux_dag = su.dagger(ux_latt)
        
        # Compute the U_t link
        Aeta_latt = aeta[tauxy, :]
        Az = su.mul_s(Aeta_latt, -z/(xplus**2 - z**2))
        uz = su.mexp(Az)
    
    else:
        # Approximate by tau=0
        tauxy = l.get_index_n2xm(0, xplus, y, n)
    
        # Compute the corresponding U_x link
        ux_latt = ux[tauxy, :]
        ux_dag = su.dagger(ux_latt)
        
        # The Az field is 0 at tau=0
        uz = su.unit()
    
    # Compute the plus link
    res = su.mul(ux_dag, uz)

    # We store the result
    su.store(up[yi], res)
    

"""
    Computes the g*Ay fields along x^+ axis at every time step
"""
def compute_ay(uy, ay, n, dts, xplus, ap):
    my_parallel_loop(compute_ay_kernel, n**2, uy, ay, dts, xplus, n, ap)

@myjit
def compute_ay_kernel(yi, uy, ay, dts, xplus, n, ap):
        
    # We get the transverse indices
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    # Rearrange the indices
    if z>= n/2:
        z = z - n
    
    if xplus > abs(z):
        # Obtain the (tau, x, y) index
        tau = round(dts*sqrt(xplus**2 - z**2))
        tauxy = l.get_index_n2xm(tau, xplus, y, n)
    
        # Compute the corresponding U_y link
        uy = uy[tauxy, :]
    
    else:
        # Approximate by tau=0
        tauxy = l.get_index_n2xm(0, xplus, y, n)
    
        # Compute the corresponding U_y link
        uy = uy[tauxy, :]
    
    # Take the logarithm
    luy = su.mlog(uy)
    
    # We extract the field (extra -1 because we want -gA_y)
    res  = su.mul_s(luy, - 1/(ap*1j))
    
    su.store(ay[yi], res)
    

"""
    Computes the g*Az fields along x^+ axis at every time step
"""
def compute_az(aeta, az, n, dts, xplus, ap):
    my_parallel_loop(compute_az_kernel, n**2, aeta, az, dts, xplus, n, ap)

@myjit
def compute_az_kernel(yi, aeta, az, dts, xplus, n, ap):
    
    # We get the transverse indices
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    # Rearrange the indices
    if z>= n/2:
        z = z - n
    
    if xplus > abs(z):
        # Obtain the (tau, x, y) index
        tau = round(dts*sqrt(xplus**2 - z**2))
        tauxy = l.get_index_n2xm(tau, xplus, y, n)
        Aeta_latt = aeta[tauxy, :]
        res = su.mul_s(Aeta_latt, -1j*xplus/(ap*(xplus**2 - z**2))) # Extra -1 because we want -gA_z
        
    else:
        res = su.mul_s(su.unit(), 0)
        
    su.store(az[yi], res)

