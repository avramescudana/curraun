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
    def __init__(self, s, nplus, ux, aeta):
        self.s = s
        self.n = s.n 
        self.t = s.t
        self.dts = round(1.0 / s.dt)
        self.nplus = nplus
        self.ux = ux
        self.aeta = aeta
        
        # We create object to store the links
        self.up_temp = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.up_lc = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.up_lc_reorder = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create an object to store the gauge transformation operator
        self.vlc_dag = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.vlc_prev = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create the pointers to the GPU
        self.d_ux = self.ux
        self.d_aeta = self.aeta
        self.d_up_temp = self.up_temp
        self.d_up_lc = self.up_lc
        self.d_up_lc_reorder = self.up_lc_reorder
        self.d_vlc_dag = self.vlc_dag
        self.d_vlc_prev = self.vlc_prev
        
    # Copies from the host to the device
    def copy_to_device(self):
        self.d_ux = cuda.to_device(self.ux)
        self.d_aeta = cuda.to_device(self.aeta)
        self.d_up_temp = cuda.to_device(self.up_temp)
        self.d_up_lc = cuda.to_device(self.up_lc)
        self.d_up_lc_reorder = cuda.to_device(self.up_lc_reorder)
        self.d_vlc_dag = cuda.to_device(self.vlc_dag)
        self.d_vlc_prev = cuda.to_device(self.vlc_prev)
    
    # Copies from the device to the host
    def copy_to_host(self):
        self.d_up_lc_reorder.copy_to_host(self.up_lc_reorder)
    
    # We copy the object to the device
    def init(self):
        if use_cuda:
            self.copy_to_device()
    
    # We evolve the gauge transformation
    def evolve_lc(self, xplus):
        
        # Update the gauge transformation operator
        update_vlc(self.d_vlc_dag, self.d_vlc_prev, self.n)
        
        # Compute the gauge transformation operator
        for xminus in range(0, -xplus, -1):
            compute_vlc(self.d_ux, self.d_aeta, self.n, self.dts, xplus, xminus, self.d_vlc_dag)
        
        # We gauge transform the field at the previous step
        if xplus != 0:
            act_vlc(self.d_up_temp, self.d_vlc_dag, self.d_vlc_prev, self.n, self.d_up_lc)
        
        # Compute the plus links in temp gauge
        compute_uplus_temp(self.d_ux, self.d_aeta, self.n, self.dts, xplus, self.d_up_temp)
        
        # We reorder the transformed fields
        reorder(self.d_up_lc_reorder, self.d_up_lc, self.s.n)
        
        # Copy the results back to the host
        if use_cuda:
            self.copy_to_host()


"""
    Compute the plus links in the temp gauge
"""
def compute_uplus_temp(ux, aeta, n, DTS, xplus, up_temp):
    my_parallel_loop(compute_uplus_temp_kernel, n**2, ux, aeta, n, DTS, xplus, up_temp)
    
@myjit
def compute_uplus_temp_kernel(yi, ux, aeta, n, DTS, xplus, up_temp):
    
    # We get the transverse indices
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    # We restrict ourselves to the light-cone of the Glasma
    if -xplus<(z-n//2)<xplus:
        
        # We get the proper time and the GPU index
        tau = round(DTS*math.sqrt(xplus**2-(z-n//2)**2))
        tauxy_x = l.get_index_n2xm(tau, xplus, y, n)
        tauxy_t = l.get_index_n2xm(tau, xplus+1, y, n)
        
        # We compute the x and t links
        ux_latt = ux[tauxy_x]
        
        aeta_latt = aeta[tauxy_t]
        aeta_fact = su.mul_s(aeta_latt, -(z-n//2)/tau**2)
        ut = su.mexp(aeta_fact)
        
        # We compute the plus links
        res = su.mul(ux_latt, ut)
        su.store(up_temp[yi], res)
    
    else:
        su.store(up_temp[yi], su.unit())


"""
    Compute the gauge transformation operator
"""
def compute_vlc(ux, aeta, n, DTS, xplus, xminus, vlc_dag):
    my_parallel_loop(compute_vlc_kernel, n**2, ux, aeta, n, DTS, xplus, xminus, vlc_dag)
    
@myjit
def compute_vlc_kernel(yi, ux, aeta, n, DTS, xplus, xminus, vlc_dag):
    
    # We get the transverse indices
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    # We compute the gauge transformation operator
    if -(xplus+xminus-1)<(z-n//2)<(xplus+xminus-1):
            
        # Compute the proper time and time index
        tau = round(DTS*math.sqrt((xplus+xminus-1)**2-(z-n//2)**2))
        tauxy = l.get_index_n2xm(tau, xplus-xminus, y, n)
            
        # Compute the x and t links
        ux_latt = ux[tauxy]
            
        aeta_latt = aeta[tauxy]
        aeta_fact = su.mul_s(aeta_latt, -(z-n//2)/tau**2)
        ut = su.mexp(aeta_fact)
            
        # Compute the gauge transformation operator at the given x^-
        aux = su.mul(su.dagger(ut), ux_latt)
    
    else:
        aux = su.unit()
        
    res = su.mul(vlc_dag[yi], aux)
    su.store(vlc_dag[yi], res)


"""
    Apply the gauge transformation V_LC on the U_+ gauge link.
"""

def act_vlc(up_temp, vlc_dag, vlcprev_dag, n, up_lc):
    my_parallel_loop(act_vlc_kernel, n*n, up_temp, vlc_dag, vlcprev_dag, up_lc)  

@myjit
def act_vlc_kernel(yi, up_temp, vlc_dag, vlcprev_dag, up_lc):
    
    aux = su.mul(up_temp[yi], vlc_dag[yi])
    res_dag = su.mul(su.dagger(vlcprev_dag[yi]), aux)
    res = su.dagger(res_dag)

    su.store(up_lc[yi], res)
    

"""
    Update the gauge transformation operator
"""
def update_vlc(vlc_dag, vlcprev_dag, n):
    my_parallel_loop(update_vlc_kernel, n*n, vlc_dag, vlcprev_dag)

@myjit
def update_vlc_kernel(yi, vlc_dag, vlcprev_dag):
    su.store(vlcprev_dag[yi], vlc_dag[yi])
    su.store(vlc_dag[yi], su.unit())


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