from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
import math
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
        
        # We create the objects to store the fields
        self.ux = np.zeros((nplus*self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.aeta = np.zeros((nplus*self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # We create U_+ before the gauge transformation
        self.up_temp = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create U_+ after the gauge transformation
        self.up_lc = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create the LC gauge transformation operators
        self.vlc0 = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.vlc1 = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.vlc = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.vlcprev = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        self.set_to_one()
        
        # To reorder the U_+ temp fields and input in Meijian's code
        self.up_temp_reorder = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.up_lc_reorder = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # We create the pointers to the GPU
        self.d_up_temp = self.up_temp
        self.d_up_temp_reorder = self.up_temp_reorder
        self.d_up_lc = self.up_lc 
        self.d_up_lc_reorder = self.up_lc_reorder
        self.d_ux = self.ux
        self.d_aeta = self.aeta
        self.d_vlc0 = self.vlc0 
        self.d_vlc1 = self.vlc1
        self.d_vlcprev = self.vlcprev
        self.d_vlc = self.vlc
        
        
        self.initialized = False

    # Copies the CPU objects to the GPU
    def copy_to_device(self):
        self.d_up_temp = cuda.to_device(self.up_temp)
        self.d_up_temp_reorder = cuda.to_device(self.up_temp_reorder)
        self.d_up_lc = cuda.to_device(self.up_lc)
        self.d_up_lc_reorder = cuda.to_device(self.up_lc_reorder)
        self.d_ux = cuda.to_device(self.ux)
        self.d_aeta = cuda.to_device(self.aeta)
        self.d_vlc0 = cuda.to_device(self.vlc0)
        self.d_vlc1 = cuda.to_device(self.vlc1)
        self.d_vlcprev = cuda.to_device(self.vlcprev)
        self.d_vlc = cuda.to_device(self.vlc)

    # Copies back the transformed field to the CPU
    def copy_to_host(self):
        # self.d_up_temp_reorder.copy_to_host(self.up_temp_reorder)
        # self.d_up_temp.copy_to_host(self.up_temp)
        self.d_up_lc_reorder.copy_to_host(self.up_lc_reorder)
        self.d_up_lc.copy_to_host(self.up_lc)
    
    # Set the gauge links to one
    def set_to_one(self):
        n = self.s.n
        my_parallel_loop(set_to_one_kernel, n**2, self.up_temp, self.up_lc, self.vlc0, self.vlc1, self.vlcprev, self.vlc)
        
    # Set the gauge transformation operator to one
    def set_to_one_vlc(self):
        n = self.s.n
        nplus = self.nplus
        my_parallel_loop(set_to_one_kernel, n**2, self.vlc0, self.vlc1)

    # Copies the fields to the GPU
    def init(self):
        if use_cuda:
            self.copy_to_device()

        self.initialized = True
    
    # Stores the fields in the designed objects
    def store_fields(self, xplus):
        n = self.s.n
        my_parallel_loop(store_fields_kernel, n**2, n, xplus, self.d_ux, self.d_aeta, self.s.d_u1, self.s.d_aeta1)

    # We evolve the gauge transformation
    def evolve_lc(self, xplus):
        
        # We construct the gauge transformation operator for the given x^+ value
        for xminus in range (-xplus, 0):
            compute_vlc(self.d_vlc0, self.d_vlc1, xplus, xminus, self.s.n, self.d_ux, self.d_aeta)
            update_vlc(self.d_vlc0, self.d_vlc1, self.s.n)
        set_vlc(self.d_vlc, self.d_vlc1, self.s.n)
        
        if xplus != 0:
            # We gauge transform the U_+(x^+-1)
            act_vlc_uplus(self.d_up_lc, self.d_up_temp, self.s.n, self.d_vlc, self.d_vlcprev)
        
        # We construct the U_+ in temporal gauge
        compute_uplus_temp(self.d_up_temp, self.d_ux, self.d_aeta, self.s.n, xplus)
        
        # We store the gauge transformation operator at time x^+ in an object to use latter
        update_vlc(self.d_vlcprev, self.d_vlc, self.s.n)
        
        # We reorder the transformed fields
        reorder(self.d_up_lc_reorder, self.d_up_lc, self.s.n)
            
        if use_cuda:
            self.copy_to_host()


"""
    Initialize the objects as unity.
"""
@myjit
def set_to_one_kernel(yi, a, b, c, d, e, f):
    su.store(a[yi], su.unit())
    su.store(b[yi], su.unit())
    su.store(c[yi], su.unit())
    su.store(d[yi], su.unit())
    su.store(e[yi], su.unit())
    su.store(f[yi], su.unit())

"""
    Store the fields in the memory objects.
"""
@myjit
def store_fields_kernel(yi, n, xplus, ux, aeta, u1, aeta1):
    ux_step = u1[yi, 0, :]
    aeta_step = aeta1[yi, :]
    
    xplusxy_latt = l.get_index_nm(xplus, yi, n*n)
    
    su.store(ux[xplusxy_latt], ux_step)
    su.store(aeta[xplusxy_latt], aeta_step)

"""
    Extracts the value of U_+ along the x^+ axis.
"""

def compute_uplus_temp(up_temp, ux, aeta, n, xplus):
    my_parallel_loop(compute_uplus_temp_kernel, n*n, xplus, n, ux, aeta, up_temp)  

@myjit
def compute_uplus_temp_kernel(yi, xplus, n, ux, aeta, up_temp):
    
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    if -xplus<(z-n/2)<xplus:
        tau_latt = round(math.sqrt(xplus**2-(z-n/2)**2))
        tauxy = l.get_index_n2xm(tau_latt, xplus, y, n)

        ux_latt = ux[tauxy]
        ux_dag = su.dagger(ux_latt)
    
        aeta_latt = aeta[tauxy]
        aeta_fact = su.mul_s(aeta_latt, (z-n/2)/(xplus**2-(z-n/2)**2))
        ueta = su.mexp(aeta_fact)
    
        res = su.mul(ueta, ux_dag)
    
        su.store(up_temp[yi], res)
        

"""
    Constructs the gauge transformation operator for a given xplus.
"""

def compute_vlc(vlc0, vlc1, xplus, xminus, n, ux, aeta):
    my_parallel_loop(compute_vlc_kernel, n*n, xplus, xminus, n, ux, aeta, vlc0, vlc1)  

@myjit
def compute_vlc_kernel(yi, xplus, xminus, n, ux, aeta, vlc0, vlc1):
    
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    if -(xplus+xminus)<(z-n/2)<(xplus+xminus):
        tau_latt = round(math.sqrt((xplus+xminus)**2-(z-n/2)**2))
        tauxy = l.get_index_n2xm(tau_latt, xplus-xminus, y, n)

        ux_latt = ux[tauxy]
    
        aeta_latt = aeta[tauxy]
        aeta_fact = su.mul_s(aeta_latt, (z-n/2)/((xplus+xminus)**2-(z-n/2)**2))
        ueta = su.mexp(aeta_fact)
        
        umin = su.mul(ueta, ux_latt)
    
        res = su.mul(umin, vlc0[yi])
    
        su.store(vlc1[yi], res)

"""
    The previous vlc0 becomes the current vlc1
"""

def update_vlc(vlc0, vlc1, n):
    my_parallel_loop(update_vlc_kernel, n*n, vlc0, vlc1)  

@myjit
def update_vlc_kernel(yi, vlc0, vlc1):
    su.store(vlc0[yi], vlc1[yi])

"""
    We fix the gauge transformation operator
"""

def set_vlc(vlc, vlc1, n):
    my_parallel_loop(set_vlc_kernel, n*n, vlc, vlc1)  

@myjit
def set_vlc_kernel(yi, vlc, vlc1):
    vlc_dag = vlc1[yi]
    res = su.dagger(vlc_dag)
    su.store(vlc[yi], res)


"""
    Applies the gauge transformation V_LC on the U_+ gauge link.
"""

def act_vlc_uplus(up_lc, up_temp, n, vlc, vlcprev):
    my_parallel_loop(act_vlc_uplus_kernel, n*n, up_lc, up_temp, vlc, vlcprev)  

@myjit
def act_vlc_uplus_kernel(yi, up_lc, up_temp, vlc, vlcprev):
    
    vlcprev_latt = vlcprev[yi]
    
    vlc_latt = vlc[yi]
    vlc_dag = su.dagger(vlc_latt)
    
    buff = su.mul(up_temp[yi], vlc_dag)
    res = su.mul(vlcprev_latt, buff)

    su.store(up_lc[yi], res)


"""
    We reorder the fields in the correct way for Meijian
"""

def reorder(up_lc_reorder, up_lc, n):
    my_parallel_loop(reorder_kernel, n**2, n, up_lc, up_lc_reorder)  

@myjit
def reorder_kernel(yi, n, up_lc, up_lc_reorder):
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    if z < n/2:
        ind = l.get_index(y, z+n//2, n)
        aux = up_lc[ind]
                
    elif z >= n//2:
        ind = l.get_index(y, z-n//2, n)
        aux = up_lc[ind]
    
    su.store(up_lc_reorder[yi], aux)