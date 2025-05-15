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
    
    def __init__(self, s, nplus, ux, uy, Aeta):
        self.s = s
        self.n = s.n
        self.nplus = nplus
        self.ux = ux
        self.uy = uy
        self.Aeta = Aeta
    
        # We create object to store the links
        self.up = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.ay = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.az = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create objects to store the gauge transformation operator
        self.vlc_act = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.vlc_prev = np.zeros((self.n**2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # We create the pointer to the GPU
        self.d_up = self.up
        self.d_ay = self.ay
        self.d_az = self.az
        
        self.d_vlc_act = self.vlc_act
        self.d_vlc_prev = self.vlc_prev
        
        self.d_ux = self.ux
        self.d_uy = self.uy
        self.d_Aeta = self.Aeta
        
    # Copies from the host to the device
    def copy_to_device(self):
        self.d_up = cuda.to_device(self.up)
        self.d_ay = cuda.to_device(self.ay)
        self.d_az = cuda.to_device(self.az)
        
        self.d_vlc_act = cuda.to_device(self.vlc_act)
        self.d_vlc_prev = cuda.to_device(self.vlc_prev)
        
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
        
        # We update the gauge transformation operator
        update_vlc(self.d_vlc_act, self.d_vlc_prev, self.n)
        
        # Compute the gauge transformation operator
        for xminus in range(0, -xplus+1, -1):
            compute_vlc(self.d_ux, self.n, xplus, xminus, self.d_vlc_act)
            
        # We transform the fields at the previous step
        if xplus != 0:
            act_up(self.d_ux, self.d_vlc_act, self.d_vlc_prev, self.n, xplus, self.d_up)
            act_ay(self.d_uy, self.d_vlc_prev, self.n, xplus, ap, self.d_ay)
            act_az(self.d_Aeta, self.d_vlc_prev, self.n, xplus, ap, self.d_az)
        
        
        # We copy back the results to the host
        if use_cuda:
            self.copy_to_host()



"""
    Update the gauge transformation operator
"""
def update_vlc(vlc_act, vlc_prev, n):
    my_parallel_loop(update_vlc_kernel, n*n, vlc_act, vlc_prev)

@myjit
def update_vlc_kernel(yi, vlc_act, vlc_prev):
    su.store(vlc_prev[yi], vlc_act[yi])
    su.store(vlc_act[yi], su.unit())


"""
    Compute the gauge transformation operator
"""
def compute_vlc(ux, n, xplus, xminus, vlc_act):
    my_parallel_loop(compute_vlc_kernel, n**2, ux, n, xplus, xminus, vlc_act)
    
@myjit
def compute_vlc_kernel(yi, ux, n, xplus, xminus, vlc_act):
    
    # We get the transverse indices
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
            
    # Compute the proper time and time index
    tauxy = l.get_index_n2xm(xplus+xminus, xplus-xminus, y, n)
            
    # Compute the link in this step
    aux = ux[tauxy]
        
    # Multiply the link by the previous gauge transformation operator
    res = su.mul(vlc_act[yi], aux)
    
    su.store(vlc_act[yi], res)
    

"""
    Apply the gauge transformation V_LC on the U_+ gauge link.
"""

def act_up(ux_temp, vlc_act, vlc_prev, n, xplus, up_lc):
    my_parallel_loop(act_up_kernel, n*n, xplus, n, ux_temp, vlc_act, vlc_prev, up_lc)  

@myjit
def act_up_kernel(yi, xplus, n, ux_temp, vlc_act, vlc_prev, up_lc):
    
    # We get the transverse indices
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    # Compute the index corresponding to U_+(x^+-1)
    tauxy = l.get_index_n2xm(xplus-1, xplus-1, y, n)
    
    # Construct the plus link+
    up = ux_temp[tauxy]
    
    # We perform the gauge transformation
    aux = su.mul(up, vlc_act[yi])
    res = su.mul(su.dagger(vlc_prev[yi]), aux)

    su.store(up_lc[yi], res)


"""
    Apply the gauge transformation V_LC on the U_y gauge link and extract the Ay fields
"""

def act_ay(uy_temp, vlc_prev, n, ap, xplus, ay_lc):
    my_parallel_loop(act_ay_kernel, n*n, xplus, n, ap, uy_temp, vlc_prev, ay_lc)  

@myjit
def act_ay_kernel(yi, xplus, n, ap, uy_temp, vlc_prev, ay_lc):
    
    # We get the transverse indices
    yz = l.get_point(yi, n)
    y, z = yz[0], yz[1]
    
    # Compute the index corresponding to U(y)
    tauxy = int(l.get_index_n2xm(xplus-1, xplus-1, y, n))
    
    # Compute the index associated Vlc(y+1)
    y1z = int(l.get_index((y+1)%n, z, n))
    
    # Construct the plus link+
    uy = uy_temp[tauxy]
    
    # We perform the gauge transformation
    aux = su.mul(uy, vlc_prev[y1z])
    uy_lc = su.mul(su.dagger(vlc_prev[yi]), aux)
    
    # We extract the Ay fields from the gauge links
    luy = su.mlog(uy_lc)
    gAy = su.mul_s(luy, 1/(1j*ap))
    
    res = su.dagger(gAy)

    su.store(ay_lc[yi], res)
    

"""
    Apply the gauge transformation V_LC on the A_z gauge field and extract the Az fields
"""

def act_az(aeta_temp, vlc_prev, n, ap, xplus, az_lc):
    my_parallel_loop(act_az_kernel, n*n, xplus, n, ap, aeta_temp, vlc_prev, az_lc)  

@myjit
def act_az_kernel(yi, xplus, n, ap, aeta_temp, vlc_prev, az_lc):
    
    if xplus == 1:
        return
    
    else:
    
        # We get the transverse indices
        yz = l.get_point(yi, n)
        y, z = yz[0], yz[1]
    
        # Compute the index corresponding to U(y)
        tauxy = int(l.get_index_n2xm(xplus-1, xplus-1, y, n))
    
        # Compute the index associated Vlc(z+1)
        yz1 = int(l.get_index(y, (z+1)%n, n))
    
        # Construct the plus link+
        aeta = aeta_temp[tauxy]
    
        # We perform the gauge transformation
        aux = su.mul(aeta, vlc_prev[yz1])
        aeta_lc = su.mul(su.dagger(vlc_prev[yi]), aux)

        # We extract the Az fields from the A_eta fields
        gAz = su.mul_s(aeta_lc, 1/(1j*ap*(xplus-1)))
    
        res = su.dagger(gAz)

        su.store(az_lc[yi], res)