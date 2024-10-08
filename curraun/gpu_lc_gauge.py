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

        # We create U_+ before the gauge transformation
        self.up_temp = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # We create U_+ after the gauge transformation
        self.up_lc = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        self.set_to_one_links()
        
        # We create the LC gauge transformation operator at tau_n
        self.vlc0 = np.zeros((self.n**2 * nplus, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # We create the LC gauge transformation operator at tau_{n+1}
        self.vlc1 = np.zeros((self.n**2 * nplus, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        self.set_to_one_vlc()
        
        # We create an auxiliar object to reorder the fields
        self.up_lc_reorder = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        

        # We create the pointers to the GPU
        self.d_up_temp = self.up_temp
        self.d_up_lc = self.up_lc
        self.d_vlc0 = self.vlc0
        self.d_vlc1 = self.vlc1
        self.d_up_lc_reorder = self.up_lc_reorder

        
        self.initialized = False

    # Copies the CPU objects to the GPU
    def copy_to_device(self):
        self.d_up_temp = cuda.to_device(self.up_temp)
        self.d_up_lc = cuda.to_device(self.up_lc)
        self.d_vlc0 = cuda.to_device(self.vlc0)
        self.d_vlc1 = cuda.to_device(self.vlc1)
        self.d_up_lc_reorder = cuda.to_device(self.up_lc_reorder)

    # Copies back the transformed field to the CPU
    def copy_to_host(self):
        self.d_up_lc_reorder.copy_to_host(self.up_lc_reorder)
        self.d_up_lc.copy_to_host(self.up_lc)
        
    # Set the gauge transformation operator to one
    def set_to_one_links(self):
        n = self.s.n
        my_parallel_loop(set_to_one_kernel, n**2, self.up_temp, self.up_lc)
    
    # Set the gauge links to one
    def set_to_one_vlc(self):
        n = self.s.n
        nplus = self.nplus
        my_parallel_loop(set_to_one_kernel, n**2*nplus, self.vlc0, self.vlc1)

    # We copy the fields to the GPU
    def init(self):
        if use_cuda:
            self.copy_to_device()

        self.initialized = True
        
    def initialize_lc(self):
        
        # We copy the objects to the GPU if they have not been copied yet
        if not self.initialized:
            self.init()
            
        init_uplus_temp(self.d_up_temp, self.t, self.s.n, self.s.d_u1)
        init_vlc(self.d_vlc1, self.t, self.s.n, self.nplus, self.s.d_u1)
        update_vlc(self.d_vlc0, self.d_vlc1, self.s.n, self.nplus)
        

    # We evolve the gauge transformation
    def evolve_lc(self, xplus):

        # We restrict to the points where the two lattices are the same
        if self.s.t % self.s.dt == 0:

            # At each time step we compute the gauge transformation operator
            compute_vlc(self.d_vlc0, self.d_vlc1, xplus, self.s.n, self.nplus, self.s.d_u1, self.s.d_aeta1)
            
            # We construct the U_+ in temporal gauge and transform them
            # TODO: Merge the two kernels into one
            act_vlc_uplus(self.s.n, xplus, self.d_up_lc, self.d_up_temp, self.d_vlc1)
            compute_uplus_temp(self.d_up_temp, xplus, self.s.n, self.s.d_u1, self.s.d_aeta1)
            
            # We save vlc1 into vlc0
            update_vlc(self.d_vlc0, self.d_vlc1, self.s.n, self.nplus)
            
            # We reorder the fields in the correct way for inputting in Meijian's code
            reorder(self.d_up_lc_reorder, self.d_up_lc, self.s.n)

        if use_cuda:
            self.copy_to_host()


"""
    Initialize the objects as unity.
"""
@myjit
def set_to_one_kernel(yi, a, b):
    su.store(a[yi], su.unit())
    su.store(b[yi], su.unit())

"""
    Initialize the plus links with the value at the first time step.
"""

def init_uplus_temp(up_temp, t, n, u1):
    my_parallel_loop(init_uplus_temp_kernel, n*n, t, n, u1, up_temp)  

@myjit
def init_uplus_temp_kernel(yi, t, n, u1, up_temp):
    
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    if (z-n//2) <= t and (z-n//2) >= -t:
        
        ty_latt = l.get_index_nm(0, y, n)

        ux_latt = u1[ty_latt, 0, :]
        ux_dag = su.dagger(ux_latt)
    
        su.store(up_temp[yi], ux_dag)
    
    
"""
    Initialize the gauge transformation operator the value at the first time step.
"""

def init_vlc(vlc1, t, n, nplus, u1):
    my_parallel_loop(init_vlc_kernel, n*n*nplus, n, t, u1, vlc1)  

@myjit
def init_vlc_kernel(yi, n, t, u1, vlc1):

    indices = l.get_point_n2xm(yi, n)
    xp_slice, y, z = indices[0], indices[1], indices[2]
    
    if (z-n//2) <= t and (z-n//2) >= -t:
        if xp_slice!=0:

            xy_latt = l.get_index_nm(xp_slice+xp_slice, y, n)
        
            ux_latt = u1[xy_latt, 0, :]

            umin_dag = su.dagger(ux_latt)

            su.store(vlc1[yi], umin_dag)


"""
    Computes the infinitesimal gauge transformation V_LC. 
"""

def compute_vlc(vlc0, vlc1, t, n, nplus, u1, aeta1):
    my_parallel_loop(compute_vlc_kernel, n*n*nplus, n, t, u1, aeta1, vlc0, vlc1)  

@myjit
def compute_vlc_kernel(yi, n, t, u1, aeta1, vlc0, vlc1):

    indices = l.get_point_n2xm(yi, n)
    xp_slice, y, z = indices[0], indices[1], indices[2]
    
    if (z-n//2) <= t and (z-n//2) >= -t:
        if xp_slice > t and xp_slice!=0:

            xy_latt = l.get_index_nm(xp_slice+xp_slice-t, y, n)
        
            ux_latt = u1[xy_latt, 0, :]

            aeta_latt = aeta1[xy_latt, :]
            aeta_fact = su.mul_s(aeta_latt, (z-n//2)/t**2)
            ueta = su.mexp(aeta_fact)

            umin = su.mul(ueta, ux_latt)
            umin_dag = su.dagger(umin)
        
            res = su.mul(umin_dag, vlc0[yi])
            su.store(vlc1[yi], res)


"""
    Extracts the value of U_+ along the x^+ axis.
"""

def compute_uplus_temp(up_temp, t, n, u1, aeta1):
    my_parallel_loop(compute_uplus_temp_kernel, n*n, t, n, u1, aeta1, up_temp)  

@myjit
def compute_uplus_temp_kernel(yi, t, n, u1, aeta1, up_temp):
    
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    if (z-n//2) <= t and (z-n//2) >= -t:
        ty_latt = l.get_index_nm(t, y, n)

        ux_latt = u1[ty_latt, 0, :]
        ux_dag = su.dagger(ux_latt)
    
        aeta_latt = aeta1[ty_latt, :]
        aeta_fact = su.mul_s(aeta_latt, (z-n//2)/t**2)
        ueta = su.mexp(aeta_fact)
    
        res = su.mul(ueta, ux_dag)
    
    
        su.store(up_temp[yi], res)


"""
    Applies the gauge transformation V_LC on the U_+ gauge link.
"""

def act_vlc_uplus(n, xplus, up_lc, up_temp, vlc1):
    my_parallel_loop(act_vlc_uplus_kernel, n*n, xplus, n, up_lc, up_temp, vlc1)  

@myjit
def act_vlc_uplus_kernel(yi, xplus, n, up_lc, up_temp, vlc1):
    
    xplusy_latt = l.get_index_nm(xplus, yi, n*n)
    buff0 = su.dagger(vlc1[xplusy_latt])
    
    buff1 = su.mul(buff0, up_temp[yi])
    
    xplusy_prev = l.get_index_nm(xplus-1, yi, n*n)
    buff2 = su.mul(buff1, vlc1[xplusy_prev])

    su.store(up_lc[yi], buff2)
    

"""
    The previous vlc0 becomes the current vlc1
"""

def update_vlc(vlc0, vlc1, n, nplus):
    my_parallel_loop(update_vlc_kernel, n**2*nplus, vlc0, vlc1)  

@myjit
def update_vlc_kernel(yi, vlc0, vlc1):
    su.store(vlc0[yi], vlc1[yi])
    

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
                
    if z >= n//2:
        ind = l.get_index(y, z-n//2, n)
        aux = up_lc[ind]
    
    su.store(up_lc_reorder[yi], aux)