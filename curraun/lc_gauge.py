from curraun.numba_target import myjit, prange, my_parallel_loop, use_cuda
import numpy as np
import curraun.lattice as l
import curraun.su as su
if use_cuda:
    import numba.cuda as cuda

"""
    A module to perform the LC gauge transformation of the Glasma fields at each x^+ slice
"""

class LCGaugeTransf:
    def __init__(self, s):
        self.s = s
        self.n = s.n
        self.t = s.t
        self.dts = round(1.0 / s.dt)

        # The arrays store the (y,z) dependence in n ** 2 lattice points

        # U_+ gauge before the gauge transformation, in the temporal gauge
        self.up_temp = np.zeros((self.n ** 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # U_+ gauge after the gauge transformation, in the LC gauge
        self.up_lc = np.zeros((self.n ** 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        
        # LC gauge transformation operator at tau_n
        self.vlc0 = np.zeros((self.n ** 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # LC gauge transformation operator at tau_{n+1}
        self.vlc1 = np.zeros((self.n ** 2, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)

        # Memory on the CUDA device:
        self.d_up_temp = self.up_temp
        self.d_up_lc = self.up_lc
        self.d_vlc0 = self.vlc0
        self.d_vlc1 = self.vlc1

    def copy_to_device(self):
        self.d_up_temp = cuda.to_device(self.up_temp)
        self.d_up_lc = cuda.to_device(self.up_lc)
        self.d_vlc0 = cuda.to_device(self.vlc0)
        self.d_vlc1 = cuda.to_device(self.vlc1)

    def copy_to_host(self):
        self.d_up_temp.copy_to_host(self.up_temp)
        self.d_up_lc.copy_to_host(self.up_lc)
        self.d_vlc0.copy_to_host(self.vlc0)
        self.d_vlc1.copy_to_host(self.vlc1)


    # The current value of xplus is given as input, where xplus \in [0, maxt//DTS]
    def compute(self, xplus, stream=None):
        tint = round(self.s.t / self.s.dt)

        # if tint % self.dtstep == 0 and tint >= 1:
        if tint % self.dtstep == 0:

            if xplus > self.s.t:
                # Use self.s.t, self.s.n, self.s.d_u1, self.s.d_aeta1 from the Glasma simulation
                compute_vlc(self.s, self.d_vlc0, self.d_vlc1, xplus)
            elif xplus == (self.s.t//self.s.dt) and xplus != 0:
                # Use self.s.t, self.s.n, self.s.d_u0, self.s.d_aeta0 from the Glasma simulation
                # This doesn't need xplus as input, it is evaluated at xplus==t//DTS
                compute_uplus(self.s, self.d_up_temp)
                act_vlc(self.d_up_lc, self.d_up_temp, self.d_vlc0, self.d_vlc1, xplus)


def compute_vlc(s, vlc0, vlc1, xplus):
    t = s.t
    n = s.n
    u1 = s.d_u1
    aeta1 = s.d_aeta1

    my_parallel_loop(compute_vlc_kernel, n*n, t, u1, aeta1, vlc0, vlc1, xplus)  

@myjit
def compute_vlc_kernel(zi, t, u1, aeta1, vlc0, vlc1, xplus, n):
    y, z = l.get_point(zi)
    xy_latt = l.get_index(2*xplus-t, y, n)

    ux_latt = u1[xy_latt, 0, :]
    aeta_latt = aeta1[xy_latt, :]

    res = gauge_transformation_operator(ux_latt, aeta_latt, vlc0, z, n)
    
    su.store(vlc1[zi], res)

# Defines the gauge operator for a given step given its value at the previous one
@myjit
def gauge_transformation_operator(ux, ae, v, z, n):
    umin = su.mul(su.mexp(su.mul_s(ae, z-n//2)), su.dagger(ux))
    r = su.mul(umin, v)
    return r






# Constructs the plus links in temporal gauge over the x^+ axis
@myjit
def get_plus_links(ux, ae, z, n):
    r = su.mul(su.mexp(su.mul_s(ae, z-n//2)), ux)
    return r



# Defines the gauge operator for a given step given its value at the previous one
@myjit
def act_on_links(u, v1, v2):
    r = su.mul(su.mul(su.dagger(v1), u), v2)
    return r