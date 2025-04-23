from curraun.numba_target import myjit, my_parallel_loop, use_cuda, mynonparjit
import numpy as np
import curraun.lattice as l
import curraun.su as su
import os
import math
from numpy.fft import rfft2, irfft2
from numpy import newaxis as na



PI = np.pi


random_np = np.random.RandomState()

if use_cuda:
    import numba.cuda as cuda

"""
    A module for computing energy density components
"""

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

        #print("n = = self.s.n", n)


        self.tr_sq_el = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.tr_sq_dif = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.d_tr_sq_el = self.tr_sq_el
        self.d_tr_sq_dif = self.tr_sq_dif

        self.ratio_dif = 0.0


    #def change_el(self, alpha):                                            # Commented 23.04.2025
    def change_el(self, alpha, m_noise):                                    # Added 23.04.2025


        peta1 = self.sprime.d_peta1

        n = self.sprime.n

        #TODO: Add Gaussian noise with parameter alpha

    #   field = random_np.normal(loc=0.0, scale=g ** 2 * mu / math.sqrt(num_sheets), size=(n ** 2 * su.ALGEBRA_ELEMENTS))           # In mv.py module
    #   field = field.reshape((n * n, su.ALGEBRA_ELEMENTS))                                                                         # In mv.py module
        eta = random_np.normal(loc=0.0, scale=alpha, size=(n ** 2 * su.GROUP_ELEMENTS))     
        eta = eta.reshape((n * n, su.GROUP_ELEMENTS))



        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Add Fourier transform   23.04.2025

        noise_n = (n // 2 + 1) if n % 2 == 0 else (n + 1) // 2            # If n is even (n % 2 == 0): new_n = n // 2 + 1
                                                                        # If n is odd (n % 2 NE 0):  new_n = (n + 1) // 2 
                                                                        # // is not the regular division operator, but a floor division operator
                                                                        # a // b : Divides a by b and returns the largest whole number less than or equal to the result.

        noise_kernel = np.zeros((n, noise_n), dtype=su.GROUP_TYPE_REAL)
        
        

        my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)                                            # Just for reference: def compute_noise_kernel(x, mass, n, new_n, kernel): 
        # my_parallel_loop(wilson_compute_poisson_kernel, n, m, n, new_n, uv, d_kernel)                                         # In mv.py module. Just for reference: def wilson_compute_poisson_kernel(x, mass, n, new_n, uv, kernel):


        print("\nsu_group, SU(N)                = ", su.su_group)                                                                                         
        print("su.GROUP_ELEMENTS, N x N       = ", su.GROUP_ELEMENTS)
        print("su.ALGEBRA_ELEMENTS, a = N² -1 = ", su.ALGEBRA_ELEMENTS)

        print("\nmnoise =", m_noise)
        print("n =", n)
        print("new_n = noise_n =", noise_n, "\n")




        #def compute_noise_kernel(x, mass, n, new_n, kernel):

        #eta = irfft2(
         #     rfft2(eta.reshape((n, n, su.ALGEBRA_ELEMENTS)), s=(n, n), axes=(0, 1)) * noise_kernel[:, :, na],
          #    s=(n, n), axes=(0, 1)).reshape((n ** 2, su.ALGEBRA_ELEMENTS))

       # eta = eta.reshape((n, n, su.ALGEBRA_ELEMENTS))
       # eta = rfft2(eta, s=(n, n), axes=(0, 1))
        #eta = eta* 0        #noise_kernel[:, :, na]  # * 0
      #  eta = irfft2(eta,  s=(n, n), axes=(0, 1)) 
      #  eta = eta.reshape((n ** 2, su.ALGEBRA_ELEMENTS))

 # fourier transform charge density
            # apply poisson kernel
            # fourier transform back
        eta = irfft2(  rfft2(eta.reshape((n, n, su.GROUP_ELEMENTS)), s=(n, n), axes=(0, 1)) * noise_kernel[:, :, na],  s=(n, n),  axes=(0, 1)  ).reshape((n ** 2, su.GROUP_ELEMENTS))
        #eta = irfft2(
         #       rfft2(eta.reshape((n, n, su.GROUP_ELEMENTS)), s=(n, n), axes=(0, 1)) * noise_kernel[:, :, na],
           #     s=(n, n), axes=(0, 1)
          # ).reshape((n ** 2, su.GROUP_ELEMENTS))




        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




        my_parallel_loop(change_el_kernel, n ** 2, peta1, eta)



    def compute_change_el(self):
        peta1s = self.s.d_peta1
        peta1sprime = self.sprime.d_peta1

        n = self.s.n

        my_parallel_loop(compute_change_el_kernel, n ** 2, peta1s, peta1sprime, self.d_tr_sq_el, self.d_tr_sq_dif)

        dif_avg = np.mean(self.d_tr_sq_dif)
        el_avg = np.mean(self.d_tr_sq_el)

        self.ratio_dif = dif_avg



@mynonparjit
def change_el_kernel(xi, peta1, eta):
    buf1 = su.add(peta1[xi], eta[xi])
    peta1[xi] = buf1

@mynonparjit
def compute_change_el_kernel(xi, peta1s, peta1sprime, tr_sq_el, tr_sq_dif):
    buf1 = l.add_mul(peta1sprime[xi], peta1s[xi], -1)
    tr_sq_dif[xi] = su.sq(buf1) 

    buf2 = peta1s[xi]
    tr_sq_el[xi] = su.sq(buf2)






# ************************** Copied from mv.py module **************************
# ******************************************************************************
#@myjit
#def wilson_compute_poisson_kernel(x, mass, n, new_n, uv, kernel):          # Just for reference: my_parallel_loop(wilson_compute_poisson_kernel, n, m, n, new_n, uv, d_kernel)     # In mv.py module
#    # for y in range(new_n):                                               # Commented Already, use this with @mynonparjit
#   for y in prange(new_n):
#       k2 = k2_latt(x, y, n)
#       if (x > 0 or y > 0) and k2 <= uv ** 2:
#           kernel[x, y] = 1.0 / (k2 + mass ** 2)


@mynonparjit                                                # Works             # @mynonparjit for SERIAL Computations  
#@myjit                                                     # Gives warnings    # @myjit for PARALLEL Computations 
def compute_noise_kernel(x, mass, n, new_n, kernel):                            # Just for reference: my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)                                                                       
    
    for y in range(new_n):                                                      # Use range with mynonparjit
                                                                                # The range() function returns a sequence of numbers, starting from 0 by default, and increments by 1 (by default), and ends at a given specified number.
                                                                                # Use prange with myjit (p stands for PARALLEL)    # for y in prange(new_n): 
        k2 = k2_latt(x, y, n)
        
        if (x > 0 or y > 0):                                                    # if (x > 0 or y > 0) and k2 <= uv ** 2:    # Used in mv.py module
            
            kernel[x, y] = 1.0 / (k2 + mass ** 2)                               # np.exp(-k2/mass**2)
                                                                                # kernel[x, y] = 1.0 / (k2 + mass ** 2)     # Used in mv.py module           




@mynonparjit                                                                                # Works
#@myjit                                                                                     # Gives warnings
def k2_latt(x, y, nt):
    result = 4.0 * (math.sin((PI * x) / nt) ** 2 + math.sin((PI * y) / nt) ** 2)
    return result
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
