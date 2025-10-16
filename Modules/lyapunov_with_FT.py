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

        #print("n = = self.s.n", n)

        self.tr_sq_el = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.tr_sq_dif = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.d_tr_sq_el = self.tr_sq_el
        self.d_tr_sq_dif = self.tr_sq_dif

        self.ratio_dif = 0.0


    def change_el(self, alpha, m_noise):                                    # Added 23.04.2025

        peta1 = self.sprime.d_peta1
        n = self.sprime.n



        """
        print(" ++++++++++++++++ In function change_el ++++++++++++++++ ")                  
        print("m_noise        =", m_noise)                  
        print("alpha          =", alpha)

        print("GROUP_TYPE =", su.GROUP_TYPE)
        print("GROUP_TYPE_REAL =", su.GROUP_TYPE_REAL)       

        print("\nsu_group, SU(N)              = ", su.su_group)                                                                                         
        print("su.GROUP_ELEMENTS, N x N       = ", su.GROUP_ELEMENTS)
        print("su.ALGEBRA_ELEMENTS, a = N² -1 = ", su.ALGEBRA_ELEMENTS)
        print("\n")
        """


        # Add Gaussian noise with parameter alpha

        eta = random_np.normal(loc=0.0, scale=alpha, size=(n ** 2 * su.GROUP_ELEMENTS))  
        """"
        print("\n\n\n     eta = random_np.normal(loc=0.0, scale=alpha, size=(n ** 2 * su.GROUP_ELEMENTS)) ")
        print("\neta shape before reshaping:", eta.shape)
        print("eta: \n", eta)
        """

        eta = eta.reshape((n * n, su.GROUP_ELEMENTS))
        """       
        print("\n\n     eta = eta.reshape((n * n, su.GROUP_ELEMENTS)) ")
        print("\neta shape after reshaping:", eta.shape)
        print("eta: \n", eta)
        """

 
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Add Fourier transform !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   
        """
                PARAMETERS USED TO CHECK FAST FOURIER TRANSFORM (FFT):

        Length of the simulation box, L    =  23.678158175122032 fm
        Number of lattice points, N        =  4
        Lattice spacing, delta_x           =  5.919539543780508 fm 

        taumax                       =  30.0
        Maximum proper time, tau_max =  5.91978 fm/c or fm

        'DTS':  8,
        """


        noise_n = (n // 2 + 1) if n % 2 == 0 else (n + 1) // 2      
        """       
        print("\n\n     noise_n = (n // 2 + 1) if n % 2 == 0 else (n + 1) // 2  ")
        print("\nn =", n)
        print("new_n = noise_n =", noise_n)
        """       
  

        noise_kernel = np.zeros((n, noise_n), dtype = su.GROUP_TYPE_REAL)      
        """
        print("\n\n     noise_kernel = np.zeros((n, noise_n), dtype = su.GROUP_TYPE_REAL)  ")
        print("\nnoise_kernel shape after generating it with all zeroes:", noise_kernel.shape)
        print("noise_kernel: \n", noise_kernel)
        """
        
        my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)                                           # Just for reference: def compute_noise_kernel(x, mass, n, new_n, kernel): 
        """
        print("\n\n     my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)  ")
        print("\nnoise_kernel shape after my_parallel_loop(compute_noise_kernel):", noise_kernel.shape)
        print("noise_kernel: \n", noise_kernel)
        """


        """
        eta = eta.reshape((n, n, su.GROUP_ELEMENTS))
        print("\n\n     eta = eta.reshape((n, n, su.GROUP_ELEMENTS)) ")
        print("\neta shape (For FT):", eta.shape)
        print("eta before FT: \n", eta)


        eta = rfft2(eta, s=(n, n), axes=(0, 1))                                                              # Performing Fourier Transform on eta for x and y axis
        print("\n\n     eta = rfft2(eta, s=(n, n), axes=(0, 1)) ")
        print("\neta shape after FT:", eta.shape)
        print("eta after FT: \n", eta)
        
        NK = noise_kernel[:, :, na]                                                                                                                 # noise_kernel[:, :, na]  # * 0
        print("\n\n     NK = noise_kernel[:, :, na]  ")
        print("\nnoise_kernel shape after reshaping (For FT), NK:", NK.shape)
        print("NK: \n", NK)

        print("\n\neta shape before multiplying with noise kernel :", eta.shape)

        eta = eta * NK  
        print("\n\n     eta = eta * NK  ") 
        print("\neta shape after multiplying with noise kernel, eta * NK  :", eta.shape)
        print("eta: \n", eta)
            
        eta = irfft2( eta,  s=(n, n), axes=(0, 1) )  # Performing Inverse Fourier Transform on eta for x and y axis
        print("\n\n     eta = irfft2( eta,  s=(n, n), axes=(0, 1) ) ")
        print("\neta shape after Inverse FT:", eta.shape)
        print("eta after Inverse FT: \n", eta)
       
        eta = eta.reshape((n ** 2, su.GROUP_ELEMENTS))
        print("\n\n     eta = eta.reshape((n ** 2, su.GROUP_ELEMENTS)) ")
        print("\neta shape after reshaping:", eta.shape)
        print("eta after reshaping: \n", eta)
        """



        """
        print("\n\n\n\n ============== Checking FT using a single line now ==============")
        print("\neta shape:", eta.shape)
        print("eta:\n", eta)
        """

        eta = irfft2(  rfft2( eta.reshape((n, n, su.GROUP_ELEMENTS)), s=(n, n), axes=(0, 1) ) * noise_kernel[:, :, na],  s=(n, n),  axes=(0, 1)  ).reshape((n ** 2, su.GROUP_ELEMENTS))
        #eta = irfft2(  rfft2(eta.reshape((n, n, su.GROUP_ELEMENTS)), s=(n, n), axes=(0, 1)) ,  s=(n, n),  axes=(0, 1)  ).reshape((n ** 2, su.GROUP_ELEMENTS))      # Eliminating the noise kernel for now (just to check the FT)
       
        """
        print("\neta shape:", eta.shape)
        print("eta:\n", eta)
        """
       
       
        """
        Input:        
        M = 2
        N = 3
        eta = np.zeros((M,N), dtype = su.GROUP_TYPE_REAL)  
        eta[1][2] = 100.0
        print("\n\n\neta shape:", eta.shape)
        print("eta:\n", eta)

        eta = rfft2( eta, s=(M,N), axes=(0, 1) )
        print("\neta shape:", eta.shape)
        print("eta:\n", eta)
       
        eta = irfft2( eta, s=(M,N), axes=(0, 1) )
        print("\neta shape:", eta.shape)
        print("eta:\n", eta)
        quit()

        Output:
        eta shape: (2, 3)
        eta:
        [[  0.   0.   0.]
        [  0.   0. 100.]]

        eta shape: (2, 2)
        eta:
        [[ 100. +0.j          -50.+86.60254038j]
        [-100. +0.j           50.-86.60254038j]]

        eta shape: (2, 3)
        eta:
        [[  0.   0.   0.]
        [  0.   0. 100.]]
        """






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






#@mynonparjit                                                          
@myjit                                                                         
def compute_noise_kernel(x, mass, n, new_n, kernel):    # Just for reference: my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_n, noise_kernel)                                                                       
    
    for y in prange(new_n):                                                     
        k2 = k2_latt(x, y, n)                                                  
        if (x > 0 or y > 0):                                                   
            
            kernel[x, y] = mass ** 2 / (k2 + mass ** 2)                             # kernel[x, y] = np.exp(-k2/mass**2)
                                                                                             
    #kernel[:,:] = 1.0                                  # Remove this later on, just to check Fourier Transform
                                                        # Set all elements of the array kernel to 1.0. 


@mynonparjit                                                                                
#@myjit                                                                                     # Gives warnings
def k2_latt(x, y, nt):
    result = 4.0 * (math.sin((PI * x) / nt) ** 2 + math.sin((PI * y) / nt) ** 2)
    return result