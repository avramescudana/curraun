from curraun.numba_target import myjit, my_parallel_loop, use_cuda
import curraun.su as su
import numpy as np
import curraun.lattice as l

if use_cuda:
    import numba.cuda as cuda



"""
    A module to get the U_+ links in the LC gauge at mid-rapidity
"""

class PlusLink:
    
    def __init__(self, s):
        self.s = s
        self.n = s.n
    
        # We create object to store the links
        self.up = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create objects to store the gauge transformation operator
        self.vlc_dag = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.vlc_prev = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create the pointer to the GPU
        self.d_up = self.up
        self.d_vlc_dag = self.vlc_dag
        self.d_vlc_prev = self.vlc_prev
        
    # Copies from the host to the device
    def copy_to_device(self):
        self.d_up = cuda.to_device(self.up)
        self.d_vlc_dag = cuda.to_device(self.vlc_dag)
        self.d_vlc_prev = cuda.to_device(self.vlc_prev)
    
    # Copies from the device to the host
    def copy_to_host(self):
        self.d_up.copy_to_host(self.up)
    
    # We copy the objects to the device
    def init(self):
        if use_cuda:
            self.copy_to_device()
    
    # We compute the plus links in the LC gauge
    def compute_plus_link(self, xplus, ux):
        
        # We update the gauge transformation operator
        update_vlc(self.d_vlc_dag, self.d_vlc_prev, self.n)
        
        # Compute the gauge transformation operator
        for xminus in range(0, -xplus+1, -1):
            compute_vlc(ux, self.n, xplus, xminus, self.d_vlc_dag)
        
        # We gauge transform the U_+ links at the previous step
        if xplus != 0:
            act_vlc(ux, self.d_vlc_dag, self.d_vlc_prev, self.n, self.d_up, xplus)
            conjugate_up(self.d_up, self.n)
        
        
        # We copy back the results to the host
        if use_cuda:
            self.copy_to_host()
            


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
    Compute the gauge transformation operator
"""
def compute_vlc(ux, n, xplus, xminus, vlc_dag):
    my_parallel_loop(compute_vlc_kernel, n**2, ux, n, xplus, xminus, vlc_dag)
    
@myjit
def compute_vlc_kernel(yi, ux, n, xplus, xminus, vlc_dag):
    
    # We get the transverse indices
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
            
    # Compute the xy index
    xy = l.get_index(xplus-xminus, y, n)
            
    # Compute the x and t links
    ux_latt = ux[xplus+xminus, xy]
        
    res = su.mul(vlc_dag[yi], ux_latt)
    
    su.store(vlc_dag[yi], res)


"""
    Apply the gauge transformation V_LC on the U_+ gauge link.
"""

def act_vlc(up_temp, vlc_dag, vlcprev_dag, n, up_lc, xplus):
    my_parallel_loop(act_vlc_kernel, n*n, up_temp, vlc_dag, vlcprev_dag, up_lc, xplus)  

@myjit
def act_vlc_kernel(yi, up_temp, vlc_dag, vlcprev_dag, up_lc, xplus):
    
    # We compute U_+^{LC}
    aux = su.mul(up_temp[xplus-1, yi], vlc_dag[yi])
    res = su.mul(su.dagger(vlcprev_dag[yi]), aux)

    su.store(up_lc[yi], res)
    

""" 
    Takes the complex conjugate of the U_p links
"""
def conjugate_up(up, n):
    my_parallel_loop(conjugate_up_kernel, n**2, up)

@myjit
def conjugate_up_kernel(yi, up):
    res = su.dagger(up[yi])
    su.store(up[yi], res)