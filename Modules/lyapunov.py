from Modules.numba_target import myjit, my_parallel_loop, use_cuda, mynonparjit
import Modules.lattice as l
import Modules.su as su


import os
import math
import numpy as np


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


class Lyapunov_CLASS():
    def __init__(self, s, sprime):
        self.s = s
        self.sprime = sprime

        n = self.s.n

        self.tr_sq_el = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.tr_sq_dif = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.d_tr_sq_el = self.tr_sq_el
        self.d_tr_sq_dif = self.tr_sq_dif

        self.EL_Ratio_Diff        = 0.0
        self.EL_Ratio_Diff_alpha2 = 0.0


    def change_EL(self, alpha, m_noise):                                    # Added 23.04.2025        
        
        peta1 = self.sprime.d_peta1
        n = self.sprime.n

        # Add Gaussian noise with parameter alpha
        eta = random_np.normal(loc=0.0, scale=alpha, size=(n ** 2 * su.GROUP_ELEMENTS))  
        eta = eta.reshape((n * n, su.GROUP_ELEMENTS))
        
        noise_n = (n // 2 + 1) if n % 2 == 0 else (n + 1) // 2             
        noise_kernel = np.zeros((n, noise_n), dtype = su.GROUP_TYPE_REAL)  
        
        
        my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)                                           # Just for reference: def compute_noise_kernel(x, mass, n, new_n, kernel): 
        
        eta = irfft2(  rfft2( eta.reshape((n, n, su.GROUP_ELEMENTS)), s=(n, n), axes=(0, 1) ) * noise_kernel[:, :, na],  s=(n, n),  axes=(0, 1)  ).reshape((n ** 2, su.GROUP_ELEMENTS))
        #eta = irfft2(  rfft2(eta.reshape((n, n, su.GROUP_ELEMENTS)), s=(n, n), axes=(0, 1)) ,  s=(n, n),  axes=(0, 1)  ).reshape((n ** 2, su.GROUP_ELEMENTS))      # Eliminating the noise kernel for now
       
        my_parallel_loop(change_EL_kernel, n ** 2, peta1, eta)



    def compute_change_EL(self, alpha):
        peta1s = self.s.d_peta1
        peta1sprime = self.sprime.d_peta1

        n = self.s.n

        my_parallel_loop(compute_change_EL_kernel, n ** 2, peta1s, peta1sprime, self.d_tr_sq_el, self.d_tr_sq_dif)

        dif_avg = np.mean(self.d_tr_sq_dif)
        el_avg = np.mean(self.d_tr_sq_el)

        self.EL_Ratio_Diff        = dif_avg
        self.EL_Ratio_Diff_alpha2 = dif_avg/(alpha**2.0)


@mynonparjit
def change_EL_kernel(xi, peta1, eta):
    buf1 = su.add(peta1[xi], eta[xi])
    peta1[xi] = buf1



@mynonparjit
def compute_change_EL_kernel(xi, peta1s, peta1sprime, tr_sq_el, tr_sq_dif):
    buf1 = l.add_mul(peta1sprime[xi], peta1s[xi], -1)
    tr_sq_dif[xi] = su.sq(buf1) 

    buf2 = peta1s[xi]
    tr_sq_el[xi] = su.sq(buf2)



#@mynonparjit                                                          
@myjit                                                                         
def compute_noise_kernel(x, mass, n, new_n, kernel):    # Just for reference: my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)                                                                       
    
    for y in prange(new_n):                                                     
        k2 = k2_latt(x, y, n)                                                  
        if (x > 0 or y > 0):                                                   
            
            kernel[x, y] = mass ** 2 / (k2 + mass ** 2) 
            #kernel[x, y] = np.exp(-k2/mass**2)
                                                                                             
    

@mynonparjit                                                                                
#@myjit                                                                                     # Gives warnings
def k2_latt(x, y, nt):
    result = 4.0 * (math.sin((PI * x) / nt) ** 2 + math.sin((PI * y) / nt) ** 2)
    return result