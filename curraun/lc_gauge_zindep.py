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

        # Number of lattice points in the xplus direction
        self.nplus = nplus

        # The arrays store the y dependence in n lattice points

        # U_+ gauge before the gauge transformation, in the temporal gauge
        self.up_temp = np.zeros((self.n, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # U_+ gauge after the gauge transformation, in the LC gauge
        self.up_lc = np.zeros((self.n, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        self.LCDEBUG = True
        if self.LCDEBUG:
            # U_- gauge before the gauge transformation, in the temporal gauge
            self.um_temp = np.zeros((self.n, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

            # U_- gauge after the gauge transformation, in the LC gauge
            self.um_lc = np.zeros((self.n, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # The intermediate gauge transformation operators have dimension (y, xplus)
        
        # LC gauge transformation operator at tau_n
        self.vlc0 = np.zeros((self.n * nplus, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        # my_parallel_loop(init_vlc_kernel, self.n ** 2, self.vlc0)

        # LC gauge transformation operator at tau_{n+1}
        self.vlc1 = np.zeros((self.n * nplus, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.initialize_vlc()

        # Memory on the CUDA device:
        self.d_up_temp = self.up_temp
        self.d_up_lc = self.up_lc
        if self.LCDEBUG:
            self.d_um_temp = self.um_temp
            self.d_um_lc = self.um_lc
        self.d_vlc0 = self.vlc0
        self.d_vlc1 = self.vlc1
        

        # Move data to GPU
        if use_cuda:
            self.copy_to_device()

        self.initialized = False

    def copy_to_device(self):
        self.d_up_temp = cuda.to_device(self.up_temp)
        self.d_up_lc = cuda.to_device(self.up_lc)
        if self.LCDEBUG:
            self.d_um_temp = cuda.to_device(self.um_temp)
            self.d_um_lc = cuda.to_device(self.um_lc)
        self.d_vlc0 = cuda.to_device(self.vlc0)
        self.d_vlc1 = cuda.to_device(self.vlc1)

    def copy_to_host(self):
        self.d_up_temp.copy_to_host(self.up_temp)
        self.d_up_lc.copy_to_host(self.up_lc)
        if self.LCDEBUG:
            self.d_um_temp.copy_to_host(self.um_temp)
            self.d_um_lc.copy_to_host(self.um_lc)
        self.d_vlc0.copy_to_host(self.vlc0)
        self.d_vlc1.copy_to_host(self.vlc1)

    def initialize_vlc(self):
        n = self.s.n
        nplus = self.nplus
        my_parallel_loop(init_vlc_kernel, n*nplus, self.vlc0, self.vlc1)

    def init(self):
        if use_cuda:
            self.copy_to_device()

        self.initialized = True

    def evolve_lc(self, xplus):
        if not self.initialized:
            self.init()

        tint = round(self.s.t / self.s.dt)
        tmoddts = tint//self.dts

        # Or equivalently if tint % self.dts == 0
        if self.s.t % self.s.dt == 0:

            # if xplus > tint: 
            if xplus > tmoddts:
                # compute_vlc(self.d_vlc1, xplus, tint, self.s.n, self.nplus, self.s.d_u1, self.s.d_aeta1)
                compute_vlc(self.d_vlc1, xplus, tint, self.s.n, self.nplus, self.s.d_u1)

                # the previous vlc0 becomes the current vlc1, for the next xplus time step
                self.d_vlc0 = self.d_vlc1
                self.vlc0 = self.vlc1

            # elif xplus == tint and xplus != 0:
            elif xplus == tmoddts and xplus != 0:
                # compute_uplus(self.d_up_temp, tint, self.s.n, self.s.d_u0, self.s.d_aeta0)
                compute_uplus_temp(self.d_up_temp, xplus, self.s.n, self.s.d_u0)
                act_vlc_uplus(self.s.n, xplus, self.nplus, self.d_up_lc, self.d_up_temp, self.d_vlc0, self.d_vlc1)

                if self.LCDEBUG:
                    compute_uminus_temp(self.d_um_temp, xplus, self.s.n, self.s.d_u0)
                    act_vlc_uminus(self.s.n, xplus, self.nplus, self.d_um_lc, self.d_um_temp, self.d_vlc1)

        if use_cuda:
            self.copy_to_host()


"""
    Initialize the LC gauge transformation as unity.
"""
@myjit
def init_vlc_kernel(yi, vlc0, vlc1):
    su.store(vlc0[yi], su.unit())
    su.store(vlc1[yi], su.unit())


"""
    Computes the infinitesimal gauge transformation V_LC. 
"""

# def compute_vlc(vlc1, xplus, t, n, nplus, u1, aeta1):
#     my_parallel_loop(compute_vlc_kernel, n*nplus, n, nplus, t, u1, aeta1, vlc1, xplus)  

def compute_vlc(vlc1, xplus, t, n, nplus, u1):
    my_parallel_loop(compute_vlc_kernel, n*nplus, n, nplus, t, u1,  vlc1, xplus)  

@myjit
# def compute_vlc_kernel(yi, n, nplus, t, u1, aeta1, vlc1, xplus):
def compute_vlc_kernel(yi, n, nplus, t, u1, vlc1, xplus):
    xplusy = l.get_point_nxm(yi, nplus, n)
    xplus, y = xplusy[0], xplusy[1]

    xy_latt = l.get_index_nm(xplus+xplus-t, y, n)
    ux_latt = u1[xy_latt, 0, :]
    ux_dag = su.dagger(ux_latt)

    # aeta_latt = aeta1[xy_latt, :]
    # aez = su.mul_s(aeta_latt, yz[1]-n//2)
    # ae_exp = su.mexp(aez)
    # umin = su.mul(ae_exp, ux_dag)
    # res = su.mul(umin, vlc1[zi])

    res = su.mul(ux_dag, vlc1[yi])
    
    su.store(vlc1[yi], res)


"""
    Extracts the value of U_+ along the x^+ axis.
"""

# def compute_uplus(up_temp, t, n, u0, aeta0):
#     my_parallel_loop(compute_uplus_kernel, n, t, n, u0, aeta0, up_temp)  

def compute_uplus_temp(up_temp, t, n, u0):
    my_parallel_loop(compute_uplus_temp_kernel, n, t, n, u0, up_temp)  

@myjit
# def compute_uplus_kernel(yi, t, n, u0, aeta0, up_temp):
def compute_uplus_temp_kernel(yi, t, n, u0, up_temp):
    ty_latt = l.get_index_nm(t, yi, n)

    ux_latt = u0[ty_latt, 0, :]
    ux_dag_latt = su.dagger(ux_latt)
    # aeta_latt = aeta0[ty_latt, :]

    # aez = su.mul_s(aeta_latt, yz[1]-n//2)
    # ae_exp = su.mexp(aez)
    # res = su.mul(ae_exp, ux_latt)
    
    su.store(up_temp[yi], ux_dag_latt)


"""
    Aplies the gauge transformation V_LC on the U_+ gauge link.
"""

def act_vlc_uplus(n, xplus, nplus, up_lc, up_temp, vlc0, vlc1):
    my_parallel_loop(act_vlc_uplus_kernel, n, xplus, nplus, up_lc, up_temp, vlc0, vlc1)  

@myjit
def act_vlc_uplus_kernel(yi, xplus, nplus, up_lc, up_temp, vlc0, vlc1):
    xplusy_latt = l.get_index_nm(xplus, yi, nplus)

    #TODO: Why this order?
    buff0 = su.dagger(vlc1[xplusy_latt])
    buff1 = su.mul(buff0, up_temp[yi])
    buff2 = su.mul(buff1, vlc0[xplusy_latt])
    
    su.store(up_lc[yi], buff2)



"""
    Extracts the value of U_- along the x^+ axis.
"""

def compute_uminus_temp(um_temp, t, n, u0):
    my_parallel_loop(compute_uminus_temp_kernel, n, t, n, u0, um_temp)  

@myjit
def compute_uminus_temp_kernel(yi, t, n, u0, um_temp):
    ty_latt = l.get_index_nm(t, yi, n)
    ux_latt = u0[ty_latt, 0, :]   

    su.store(um_temp[yi], ux_latt)


"""
    Aplies the gauge transformation V_LC on the U_+ gauge link.
"""

def act_vlc_uminus(n, xplus, nplus, um_lc, um_temp, vlc1):
    my_parallel_loop(act_vlc_uminus_kernel, n, xplus, nplus, um_lc, um_temp, vlc1)  

@myjit
def act_vlc_uminus_kernel(yi, xplus, nplus, um_lc, um_temp, vlc1):
    xplusy_latt = l.get_index_nm(xplus, yi, nplus)

    # Gauge operator at x^- + delta x^-
    vlc2 = su.mul(um_temp[yi], vlc1[xplusy_latt])

    #TODO: Why this order?
    buff0 = su.dagger(vlc2)
    buff1 = su.mul(buff0, um_temp[yi])
    buff2 = su.mul(buff1, vlc1[xplusy_latt])
    
    su.store(um_lc[yi], buff2)


"""
Carlos' functions for the z-independent gauge transformation
CPU version only, for numerical cheks
"""

# Constructs the plus links in temporal gauge over the x^+ axis
def get_plus_links(ux):
    r = su.dagger(ux)
    return r

# Defines the gauge operator for a given step given its value at the previous one
def gauge_transformation_operator(ux, v):
    umin = su.dagger(ux)
    r = su.mul(umin, v)
    return r

# Defines the gauge operator for a given step given its value at the previous one
def act_on_links(u, v1, v2):
    r = su.mul(su.mul(su.dagger(v1), u), v2)
    return r