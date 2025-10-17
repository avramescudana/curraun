from curraun.numba_target import myjit, my_parallel_loop, use_cuda, mynonparjit
import numpy as np
import curraun.lattice as l
import curraun.su as su
import os
import math

from numpy.fft import rfft2, irfft2
from numpy import newaxis as na

from numba import prange

PI = np.pi
random_np = np.random.RandomState()

if use_cuda:
    import numba.cuda as cuda


# set precision of variable
su_precision = os.environ.get('PRECISION', 'double')
if su_precision == 'single':
    DTYPE = np.float32
elif su_precision == 'double':
    DTYPE = np.float64


class Lyapunov():
    def __init__(self, s, sprime):
        self.s = s
        self.sprime = sprime

        n = self.s.n
        N2 = n * n


        """ Start: For E_eta"""
        self.tr_sq_EL = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.tr_sq_EL_dif = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)        
        self.d_tr_sq_EL = self.tr_sq_EL
        self.d_tr_sq_EL_dif = self.tr_sq_EL_dif

        self.ratio_EL_dif = 0.0
        """ End: For E_eta"""



        """ Start: For B_eta"""
        self.tr_sq_BL = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.tr_sq_BL_dif = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.d_tr_sq_BL = self.tr_sq_BL
        self.d_tr_sq_BL_dif = self.tr_sq_BL_dif

        self.ratio_BL_dif = 0.0
        """ End: For B_eta"""




    """ Start: For E_eta"""
    def change_EL(self, Option_noise_type, alpha, m_noise, K, dk):                                  # Added 11.09.2025
    #def change_EL(self, alpha, m_noise):                                                           # Commented 10.09.2025

        """
        print("\nchange_EL called with parameters:")
        print("Option_noise_type =", Option_noise_type)        
        print("m_noise             =", m_noise)                  
        print("alpha               =", alpha)             
        print("K                   =", K)             
        print("dk                  =", dk)          
        """

        peta1 = self.sprime.d_peta1
        n = self.sprime.n

        # Add Gaussian noise with parameter alpha
        eta = random_np.normal(loc=0.0, scale=alpha, size=(n ** 2 * su.GROUP_ELEMENTS))  
        eta = eta.reshape((n * n, su.GROUP_ELEMENTS))
        
        noise_n = (n // 2 + 1) if n % 2 == 0 else (n + 1) // 2             
        noise_kernel = np.zeros((n, noise_n), dtype = su.GROUP_TYPE_REAL)  
        

        my_parallel_loop(compute_noise_kernel, n, Option_noise_type, n, noise_n, noise_kernel, m_noise, K, dk)      # Just for reference: def compute_noise_kernel(x, Option_noise_type, n, new_n, kernel, m_noise, K, dk): # Added 11.09.2025
        #my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)                               # Just for reference: def compute_noise_kernel(x, m_noise, n, new_n, kernel):                           # Commented 10.09.2025

        """
        Normally, my_parallel_loop calls the kernel like kernel(xi, *args...), where xi is the loop index (here x).
        
        def kernel(xi, *args):
        where
           xi is the loop index (0, 1, 2, ... N-1)
           *args are the extra arguments you pass in my_parallel_loop(kernel, N, arg1, arg2, ...)
        """
        
        eta = irfft2(  rfft2( eta.reshape((n, n, su.GROUP_ELEMENTS)), s=(n, n), axes=(0, 1) ) * noise_kernel[:, :, na],  s=(n, n),  axes=(0, 1)  ).reshape((n ** 2, su.GROUP_ELEMENTS))
        #eta = irfft2(  rfft2(eta.reshape((n, n, su.GROUP_ELEMENTS)), s=(n, n), axes=(0, 1)) ,  s=(n, n),  axes=(0, 1)  ).reshape((n ** 2, su.GROUP_ELEMENTS))  # Eliminating the noise kernel for now

        my_parallel_loop(change_EL_kernel, n ** 2, peta1, eta)



    def compute_change_EL(self):
        peta1s = self.s.d_peta1
        peta1sprime = self.sprime.d_peta1

        n = self.s.n

        my_parallel_loop(compute_change_EL_kernel, n ** 2, peta1s, peta1sprime, self.d_tr_sq_EL, self.d_tr_sq_EL_dif)

        dif_avg = np.mean(self.d_tr_sq_EL_dif)
        EL_avg = np.mean(self.d_tr_sq_EL)

        self.ratio_EL_dif = dif_avg
    """ End: For E_eta"""




    """ Start: For B_eta"""
    def change_Ui(self, Option_noise_type, alpha_lattice, m_noise, K, dk):                                      
        # alpha = alpha_lattice
        u1 = self.sprime.u1
        n = self.sprime.n

        N  = n
        N2 = N * N

        # Add Gaussian noise with parameter alpha
        # eta = random_np.normal(loc=0.0, scale=alpha_lattice, size=(n ** 2 * su.GROUP_ELEMENTS))  
        # eta = eta.reshape((n * n, su.GROUP_ELEMENTS))

        eta = random_np.normal(loc=0.0, scale=alpha_lattice, size=(N2 * 2 * su.GROUP_ELEMENTS))  
        eta = eta.reshape((N2, 2, su.GROUP_ELEMENTS))
        
        
        noise_n = (N // 2 + 1) if N % 2 == 0 else (N + 1) // 2             
        noise_kernel = np.zeros((N, noise_n), dtype = su.GROUP_TYPE_REAL)  
	
        my_parallel_loop(compute_noise_kernel, N, Option_noise_type, N, noise_n, noise_kernel, m_noise, K, dk)     
        
        eta = irfft2(  rfft2( eta.reshape((n, n, 2, su.GROUP_ELEMENTS)), s=(n, n), axes=(0, 1) ) * noise_kernel[:, :, na, na],  s=(n, n),  axes=(0, 1)  ).reshape((n ** 2, 2, su.GROUP_ELEMENTS))
        
        my_parallel_loop(change_Ui_kernel, N2,  u1, eta)


    def compute_change_BL(self):
        u1_s = self.s.d_u1
        u1_sprime = self.sprime.d_u1

        n = self.s.n

        my_parallel_loop(compute_change_BL_kernel, n ** 2, n, u1_s, u1_sprime, self.d_tr_sq_BL, self.d_tr_sq_BL_dif)

        dif_BL_avg = np.mean(self.d_tr_sq_BL_dif)
        BL_avg = np.mean(self.d_tr_sq_BL)

        self.ratio_BL_dif = dif_BL_avg

    """ End: For B_eta"""




@mynonparjit
def change_EL_kernel(xi, peta1, eta):
    buf1 = su.add(peta1[xi], eta[xi])
    peta1[xi] = buf1



@mynonparjit
def compute_change_EL_kernel(xi, peta1s, peta1sprime, tr_sq_EL, tr_sq_EL_dif):
    buf1 = l.add_mul(peta1sprime[xi], peta1s[xi], -1)
    tr_sq_EL_dif[xi] = su.sq(buf1)

    buf2 = peta1s[xi]
    tr_sq_EL[xi] = su.sq(buf2)




@mynonparjit
def change_Ui_kernel(xi, u1, eta):

    Exp_Noise_x = su.mexp(eta[xi,0]) 
    New_Link_x  = su.mul(u1[xi,0], Exp_Noise_x)
    su.store(u1[xi,0], New_Link_x)

    Exp_Noise_y = su.mexp(eta[xi,1]) 
    New_Link_y  = su.mul(u1[xi,1], Exp_Noise_y)
    su.store(u1[xi,1], New_Link_y)

    # print("u1 before", u1[xi])
    # print("u1[xi, 1]", u1[xi, 0])
    # print("u1[xi, 2]", u1[xi, 1])

    # print("eta ", eta[xi])

    # print("eta[xi, 1]", eta[xi, 0])
    # print("eta[xi, 2]", eta[xi, 1])

    # print("buf1", buf1)
    # print("buf2 ", buf2)


    # u1[xi,0] = buf1b
    # u1[xi,1] = buf2b
    #u1[xi] = [buf1b, buf2b]
    #u1[xi] = buf2

    # print("buf1b", buf1b)
    # print("buf2b", buf2b)

    # print("u1 after", u1[xi])
    # print("u1[xi, 1] after", u1[xi, 0])
    # print("u1[xi, 2] after", u1[xi, 1])







@mynonparjit
def compute_change_BL_kernel(xi, n, u1_s, u1_sprime, tr_sq_BL, tr_sq_BL_dif):

    #    BL[xi] = 0.5 * (su.sq(su.ah(l.plaq_pos(u0, xi, 0, 1, n))) * t + su.sq(su.ah(l.plaq_pos(u1, xi, 0, 1, n))) * (t+dt))  : IN FILE energy.py
    #    BL[xi] = (NC - su.tr(l.plaq_pos(u0, xi, 0, 1, n)).real) * t + (NC - su.tr(l.plaq_pos(u1, xi, 0, 1, n)).real) * (t + dt)


    BL       = su.ah( l.plaq_pos(u1_s,      xi, 0, 1, n) )
    BL_prime = su.ah( l.plaq_pos(u1_sprime, xi, 0, 1, n) )

    """
    # Bz (Beta)
    bf1 = su.zero()
    b1 = l.plaq(u1_s, xi, 0, 1, 1, 1, n)		    #  plaq(u, x, i, j, oi, oj, n): Just for reference
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, ngp_index, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, ngp_index, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, ngp_index, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    Beta = l.add_mul(bf1, b2, +0.25)
    """


    #buf1 = BL[xi]
    tr_sq_BL[xi] = su.sq(BL)

    buf2 = l.add_mul(BL_prime, BL, -1)
    tr_sq_BL_dif[xi] = su.sq(buf2)




# @mynonparjit
# def compute_change_EL_kernel(xi, peta1s, peta1sprime, tr_sq_EL, tr_sq_EL_dif):

#     buf1 = l.add_mul(peta1sprime[xi], peta1s[xi], -1)

#     tr_sq_EL_dif[xi] = su.sq(buf1)

#     buf2 = peta1s[xi]
#     tr_sq_EL[xi] = su.sq(buf2)






#@mynonparjit                                                          
@myjit             
def compute_noise_kernel(x, Option_noise_type, n, new_n, kernel, m_noise, K, dk):       # Just for reference: my_parallel_loop(compute_noise_kernel, n, Option_noise_type,  n, noise_n, noise_kernel, m_noise, K, dk)   # Added 11.09.2025
#def compute_noise_kernel(x, m_noise, n, new_n, kernel):                                # Just for reference: my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)                              # Commented 10.09.2025                                                                     
    
    for y in prange(new_n):                                                     
                                                      
        k2 = k2_latt(x, y, n)   

        #if (x > 0 or y > 0):                                           # Comment this statement as our kernel does not blow up at (0,0)

        if Option_noise_type == 0:                                          # No noise     
            #print("Option_noise_type = 0: No noise")
            kernel[x, y] = 1.0

        elif Option_noise_type == 1:                                        # Exponential noise
            #print("Option_noise_type = 1: Exponential noise")
            kernel[x, y] = np.exp(-k2/m_noise**2)

        elif Option_noise_type == 2:                                        # Power-law noise
            #print("Option_noise_type = 2: Power-law noise")
            kernel[x, y] = m_noise ** 2 / (k2 + m_noise ** 2)

        elif Option_noise_type == 3:                                        # Independent noise, Theta function

            #print("Option_noise_type = 3: Independent noise, Theta function")
            k  = np.sqrt(k2)
            k_lower_limit = K - dk/2
            k_upper_limit = K + dk/2

            if (k_lower_limit <= k <= k_upper_limit):
                kernel[x, y] = 1.0
            else:
                kernel[x, y] = 0.0      

        else: 
            print("Warning: Invalid Option_noise_type. Setting kernel to ZERO !!!")
            kernel[x, y] = 0.0

        """

        k_lower_limit = K - dk/2
        k_upper_limit = K + dk/2

        if (k_lower_limit <= k <= k_upper_limit):
            kernel[x, y] = 1.0
        else:
            kernel[x, y] = 0.0
        """



@mynonparjit
#@myjit                                                                                     
def k2_latt(nx, ny, N_T):
    #kx_latt_sq = (4.0) * (math.sin((PI * nx) / N_T) ** 2)         # In lattice units (Dimensionless)
    #ky_latt_sq = (4.0) * (math.sin((PI * ny) / N_T) ** 2)         # In lattice units (Dimensionless)
    #result = kx_latt_sq + ky_latt_sq                            # This is Discrete Lattice Momentum squared, k_latt² : in lattice units (Dimensionless)

    result = 4.0 * (math.sin((PI * nx) / N_T) ** 2 + math.sin((PI * ny) / N_T) ** 2)            # In lattice units (Dimensionless)
    return result


