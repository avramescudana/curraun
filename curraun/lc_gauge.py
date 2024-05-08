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
        
        # We create the LC gauge transformation operator at tau_n
        self.vlc0 = np.zeros((self.n**2 * nplus, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        # my_parallel_loop(init_vlc_kernel, self.n ** 2, self.vlc0)

        # We create the LC gauge transformation operator at tau_{n+1}
        self.vlc1 = np.zeros((self.n**2 * nplus, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # We create the pointers to the GPU
        self.d_up_temp = self.up_temp
        self.d_up_lc = self.up_lc
        self.d_vlc0 = self.vlc0
        self.d_vlc1 = self.vlc1

        self.initialized = False

    # Copies the CPU objects to the GPU
    def copy_to_device(self):
        self.d_up_temp = cuda.to_device(self.up_temp)
        self.d_up_lc = cuda.to_device(self.up_lc)
        self.d_vlc0 = cuda.to_device(self.vlc0)
        self.d_vlc1 = cuda.to_device(self.vlc1)

    # Copies back the transformed field to the CPU
    def copy_to_host(self):
        self.d_up_lc.copy_to_host(self.up_lc)

    # We initialize the gauge transformation operator as unity
    # TODO: Initialize using the fields at tau=1
    def initialize_vlc(self):
        n = self.s.n
        nplus = self.nplus
        my_parallel_loop(init_vlc_kernel, n**2 * nplus, self.vlc0, self.vlc1)

    # We copy the fields to the GPU
    def init(self):
        if use_cuda:
            self.copy_to_device()

        self.initialized = True

    # The current value of xplus is given as input, where xplus \in [0, maxt//DTS]
    def evolve_lc(self, xplus):
        if not self.initialized:
            self.init()

        tint = round(self.s.t / self.s.dt)

        # print('t=', self.s.t)
        # print('dt=', self.s.dt)

        # if tint % self.dts == 0:
        if self.s.t % self.s.dt == 0:

            # print('tint=', tint)
            # print('xplus=', xplus)

            #TODO: Should be xplus > tint//self.dts, needs to be corrected
            if xplus > tint:
            # if xplus > self.t:
                # Use self.s.t, self.s.n, self.s.d_u1, self.s.d_aeta1 from the Glasma simulation
                # compute_vlc(self.s, self.d_vlc0, self.d_vlc1, xplus)
                # compute_vlc(self.d_vlc0, self.d_vlc1, xplus, tint, self.s.n, self.s.d_u1, self.s.d_aeta1)
                compute_vlc(self.d_vlc1, xplus, tint, self.s.n, self.s.d_u1, self.s.d_aeta1)

                # swap variables
                # self.d_vlc0, self.d_vlc1 = self.d_vlc1, self.d_vlc0 
                # self.vlc0, self.vlc1 = self.vlc1, self.vlc0


            elif xplus == tint and xplus != 0:
            # elif xplus == (self.t//self.dts) and xplus != 0:
                # Use self.s.t, self.s.n, self.s.d_u0, self.s.d_aeta0 from the Glasma simulation
                # This doesn't need xplus as input, it is evaluated at xplus==t//DTS
                # compute_uplus(self.s, self.d_up_temp)
                compute_uplus(self.d_up_temp, tint, self.s.n, self.s.d_u0, self.s.d_aeta0)

                # act_vlc_uplus(self.s, self.d_up_lc, self.d_up_temp, self.d_vlc0, self.d_vlc1)

                # act_vlc_uplus(self.s.n, self.d_up_lc, self.d_up_temp, self.d_vlc0, self.d_vlc1)

                # When xplus==(t//DTS), the xplus loop reached its end and vlc needs to be reinitialized to su.unit()
                # self.initialized = False

                # the previous vlc0 becomes the current vlc1, for the next xplus step
                # self.d_vlc0 = self.d_vlc1
                # self.vlc0 = self.vlc1

        if use_cuda:
            self.copy_to_host()


"""
    Initialize the LC gauge transformation as unity.
"""
@myjit
def init_vlc_kernel(zi, vlc0, vlc1):
    su.store(vlc0[zi], su.unit())
    su.store(vlc1[zi], su.unit())


"""
    Computes the infinitesimal gauge transformation V_LC. 
"""

# def compute_vlc(s, vlc0, vlc1, xplus):
def compute_vlc(vlc1, xplus, t, n, u1, aeta1):
    # t = s.t
    # n = s.n
    # u1 = s.d_u1
    # aeta1 = s.d_aeta1

    my_parallel_loop(compute_vlc_kernel, n*n, n, t, u1, aeta1, vlc1, xplus)  

@myjit
def compute_vlc_kernel(zi, n, t, u1, aeta1, vlc1, xplus):
    yz = l.get_point(zi, n)
    #TODO: Should be get_index() instead? Probably no but check to be sure.
    xy_latt = l.get_index_nm(xplus+xplus-t, yz[0], n)
    # xy_latt = l.get_index(xplus+xplus-t, yz[0], n)

    ux_latt = u1[xy_latt, 0, :]
    aeta_latt = aeta1[xy_latt, :]

    # res = gauge_transformation_operator(ux_latt, aeta_latt, vlc0, yz[1], n)

    aez = su.mul_s(aeta_latt, yz[1]-n//2)
    ae_exp = su.mexp(aez)
    ux_dag = su.dagger(ux_latt)
    umin = su.mul(ae_exp, ux_dag)
    res = su.mul(umin, vlc1[zi])
    
    # su.store(vlc1[zi], res)
    su.store(vlc1[zi], res)

# Defines the gauge operator for a given step given its value at the previous one
@myjit
def gauge_transformation_operator(ux, ae, v, z, n):
    umin = su.mul(su.mexp(su.mul_s(ae, z-n//2)), su.dagger(ux))
    r = su.mul(umin, v)
    return r


"""
    Extracts the value of U_+ along the x^+ axis.
"""

# def compute_uplus(s, up_temp):
def compute_uplus(up_temp, t, n, u0, aeta0):
    # t = s.t
    # n = s.n
    # u0 = s.d_u0
    # aeta0 = s.d_aeta0

    my_parallel_loop(compute_uplus_kernel, n*n, t, n, u0, aeta0, up_temp)  

@myjit
def compute_uplus_kernel(zi, t, n, u0, aeta0, up_temp):
    yz = l.get_point(zi, n)
    #TODO: Should be get_index() instead? Probably no but check to be sure.
    ty_latt = l.get_index_nm(t, yz[0], n)
    # ty_latt = l.get_index(t, yz[0], n)

    ux_latt = u0[ty_latt, 0, :]
    aeta_latt = aeta0[ty_latt, :]

    # res = get_plus_links(ux_latt, aeta_latt, yz[1], n)

    aez = su.mul_s(aeta_latt, yz[1]-n//2)
    ae_exp = su.mexp(aez)
    res = su.mul(ae_exp, ux_latt)
    
    su.store(up_temp[zi], res)


# Constructs the plus links in temporal gauge over the x^+ axis
@myjit
def get_plus_links(ux, ae, z, n):
    r = su.mul(su.mexp(su.mul_s(ae, z-n//2)), ux)
    return r



"""
    Aplies the gauge transformation V_LC on the U_+ gauge link.
"""

# def act_vlc_uplus(s, up_lc, up_temp, vlc0, vlc1):
def act_vlc_uplus(n, up_lc, up_temp, vlc0, vlc1):
    # n = s.n

    my_parallel_loop(act_vlc_uplus_kernel, n*n, up_lc, up_temp, vlc0, vlc1)  

@myjit
def act_vlc_uplus_kernel(zi, up_lc, up_temp, vlc0, vlc1):

    res = act_on_links(up_temp, vlc1, vlc0)
    
    su.store(up_lc[zi], res)


# Defines the gauge operator for a given step given its value at the previous one
@myjit
def act_on_links(u, v1, v2):
    r = su.mul(su.mul(su.dagger(v1), u), v2)
    return r