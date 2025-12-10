import os
import math


from numpy import newaxis as na

from numba import prange

from curraun.numba_target import myjit, my_parallel_loop, use_cuda, mynonparjit

import curraun.lattice as l
import curraun.su as su
from curraun.su import NC





# ============================= CPU / GPU ============================= #
# ===================================================================== #
# Use cupy only if cuda is available
# cupy can be turned off by changing 'use_cupy'

if use_cuda:
    import cupy as xp
    from cupy.fft import rfft2, irfft2
    from numba import cuda

    print("\n\nuse_cuda =", use_cuda, ":     Go ahead and import CuPy.")

    use_cupy = True
    # cuda.detect()

    print("use_cupy =", use_cupy, ":     CuPy has been imported successfully.")
    print("🟢 CuPy random number generator will be initialized.\n\n")

else:
    import numpy as xp
    from numpy.fft import rfft2, irfft2

    print("\n\nuse_cuda =", use_cuda, ":     CuPy will not be imported.")

    use_cupy = False

    print("use_cupy =", use_cupy, ":     CuPy is not imported.")
    print("🟢 NumPy random number generator will be initialized.\n\n")

random_state = xp.random.RandomState()
PI = xp.pi
# ===================================================================== #
# ===================================================================== #





# ??????????????????????????????????????????????????????????????????????????????
# set precision of variable
su_precision = os.environ.get('PRECISION', 'double')

if su_precision == 'single':
        DTYPE = xp.float32            
elif su_precision == 'double':
        DTYPE = xp.float64

# print(xp.float64)
# print("Data type:", DTYPE, "\n")
# ??????????????????????????????????????????????????????????????????????????????







class Lyapunov():
    def __init__(self, s, sprime):
        self.s = s
        self.sprime = sprime

        self.NC = NC
        # NC = self.s.NC

        n = self.s.n
        N2 = n * n




        # Create all arrays using xp (works for both CPU and GPU)        
        # Create arrays on CPU (host) using numpy OR on GPU (device) using cupy automatically based on use_cuda flag
        self.Trace_EL2     =  xp.zeros(N2, dtype=su.GROUP_TYPE_REAL)
        self.Trace_ELdif2  =  xp.zeros(N2, dtype=su.GROUP_TYPE_REAL)
        self.Trace_BL2     =  xp.zeros(N2, dtype=su.GROUP_TYPE_REAL)
        self.Trace_BLdif2  =  xp.zeros(N2, dtype=su.GROUP_TYPE_REAL)

        """
        This part is not required anymore because we are using xp arrays directly
        # For GPU compatibility, point d_ arrays to the same xp arrays
        # When xp = cupy, these are GPU arrays; when xp = numpy, these are CPU arrays
        # self.d_Trace_EL2     =  self.Trace_EL2
        # self.d_Trace_ELdif2  =  self.Trace_ELdif2
        # self.d_Trace_BL2     =  self.Trace_BL2
        # self.d_Trace_BLdif2  =  self.Trace_BLdif2
        """

        # Scalar values (these are automatically handled correctly)
        self.Trace_ELdiff2                  =  0.0
        self.Ratio_Trace_ELdiff2_Trace_EL2  =  0.0
        self.Trace_BLdiff2                  =  0.0
        self.Ratio_Trace_BLdiff2_Trace_BL2  =  0.0


        # NOTE: copy_to_device not required when everything allocated with xp (cupy) directly.
        # if use_cuda:
        #     self.copy_to_device()


    """
    # Optional function: Keep for debugging or if needed later
    def copy_to_host(self):
        
        # Copy GPU (device) arrays (like d_Trace_EL2) to CPU (host) arrays (like Trace_EL2) without destroying GPU data.
        # Only needed if you want to access the full arrays on CPU for analysis.
        
        if use_cuda:
            self.d_Trace_EL2.copy_to_host(self.Trace_EL2)
            self.d_Trace_ELdif2.copy_to_host(self.Trace_ELdif2)
            self.d_Trace_BL2.copy_to_host(self.Trace_BL2)
            self.d_Trace_BLdif2.copy_to_host(self.Trace_BLdif2)
    """

    """
    #  THIS DESTROYS GPU ARRAYS! DO NOT USE UNLESS ABSOLUTELY NECESSARY!
    def copy_to_host(self):
            # Only needed if we want to convert cupy arrays to numpy arrays explicitly.
            # In most cases, we won't need this when using xp approach.
            if use_cuda:
                # Convert cupy arrays to numpy arrays
                self.Trace_EL2 = self.d_Trace_EL2.get()         #  THIS DESTROYS GPU ARRAYS!
                self.Trace_ELdif2 = self.d_Trace_ELdif2.get()
                self.Trace_BL2 = self.d_Trace_BL2.get()
                self.Trace_BLdif2 = self.d_Trace_BLdif2.get()
            # For numpy, arrays are already on host, so no action needed
    """




    """
        # Create arrays on CPU (host)       #  Following are the CPU arrays (NumPy)
        self.Trace_EL2     =  np.zeros(N2, dtype=su.GROUP_TYPE_REAL)
        self.Trace_ELdif2  =  np.zeros(N2, dtype=su.GROUP_TYPE_REAL)
        self.Trace_BL2     =  np.zeros(N2, dtype=su.GROUP_TYPE_REAL)
        self.Trace_BLdif2  =  np.zeros(N2, dtype=su.GROUP_TYPE_REAL)

        # Point GPU references to same CPU arrays initially  # Initially pointers to the same CPU arrays
        # This creates two names for the same memory on CPU
        self.d_Trace_EL2     =  self.Trace_EL2
        self.d_Trace_ELdif2  =  self.Trace_ELdif2
        self.d_Trace_BL2     =  self.Trace_BL2
        self.d_Trace_BLdif2  =  self.Trace_BLdif2


        self.Trace_ELdiff2                  =  0.0
        self.Ratio_Trace_ELdiff2_Trace_EL2  =  0.0
        self.Trace_BLdiff2                  =  0.0
        self.Ratio_Trace_BLdiff2_Trace_BL2  =  0.0

        if use_cuda:
            self.copy_to_device()

        
    def copy_to_device(self):                   # Actually copies the CPU arrays to GPU arrays
        # cuda.to_device() takes the CPU array and creates a copy on GPU
        # Now d_Trace_EL2 points to GPU memory, Trace_EL2 still points to CPU memory
        # These become two separate arrays - changes on one don't affect the other
        self.d_Trace_EL2     =  cuda.to_device(self.d_Trace_EL2)
        self.d_Trace_ELdif2  =  cuda.to_device(self.d_Trace_ELdif2)
        self.d_Trace_BL2     =  cuda.to_device(self.d_Trace_BL2)
        self.d_Trace_BLdif2  =  cuda.to_device(self.d_Trace_BLdif2)
        



    def copy_to_host(self):
        self.d_Trace_EL2.copy_to_host(self.Trace_EL2)
        self.d_Trace_ELdif2.copy_to_host(self.Trace_ELdif2)
        self.d_Trace_BL2.copy_to_host(self.Trace_BL2)
        self.d_Trace_BLdif2.copy_to_host(self.Trace_BLdif2)
    """

    

    # Apply noise perturbation to E_eta fields

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

        N     = self.sprime.n
        N2 = N * N
        
        noise_N = (N // 2 + 1) if N % 2 == 0 else (N + 1) // 2

        eta = random_state.normal(loc=0.0, scale=alpha, size=(N2 * su.GROUP_ELEMENTS))                  # Add Gaussian noise with parameter alpha 
        # eta = xp.reshape(eta, (N2, su.GROUP_ELEMENTS))                                                # Reshaping eta required for FT # (128,128,4)
        #                                                                                              # Commented 29.10.2025 intentioanlly because I think it is unnecessary here


        noise_kernel = xp.zeros((N, noise_N), dtype = su.GROUP_TYPE_REAL)                               # GPU (CuPy) OR CPU (NumPy) based on use_cuda flag

        my_parallel_loop(compute_noise_kernel, N, Option_noise_type, N, noise_N, noise_kernel, m_noise, K, dk)


        # Apply noise kernel in Fourier space    
        eta_for_FFT        =  xp.reshape(eta, (N, N, su.GROUP_ELEMENTS))                        # Reshape eta required for FFT    
        eta_FFT            =  xp.fft.rfft2(eta_for_FFT, s=(N, N), axes=(0, 1))                  # Performing FFT on eta_for_FFT for x and y axis only   
        NK                 =  noise_kernel[:, :, na]
        modulated_eta_FFT  =  eta_FFT * NK
        eta_after_IFFT     =  xp.fft.irfft2(modulated_eta_FFT, s=(N, N), axes=(0, 1))           # Performing Inverse FFT on eta_after_IFFT for x and y axis only       
        eta                =  xp.reshape(eta_after_IFFT, (N2, su.GROUP_ELEMENTS))               # Reshape eta_after_IFFT to original shape  

        #eta = irfft2(  rfft2( eta.reshape((N, N, su.GROUP_ELEMENTS)), s=(N, N), axes=(0, 1) ) * noise_kernel[:, :, na],  s=(N, N),  axes=(0, 1)  ).reshape((N2, su.GROUP_ELEMENTS))
        #eta = irfft2(  rfft2(eta.reshape((n, n, su.GROUP_ELEMENTS)), s=(n, n), axes=(0, 1)) ,  s=(n, n),  axes=(0, 1)  ).reshape((n ** 2, su.GROUP_ELEMENTS))  # Eliminating the noise kernel for now

        my_parallel_loop(change_EL_kernel, N2, peta1, eta)                  # Apply the perturbation using the kernel


        """
        Normally, my_parallel_loop calls the kernel like kernel(xi, *args...), where xi is the loop index (here x).
        
        def kernel(xi, *args):
        where
           xi is the loop index (0, 1, 2, ... N-1)
           *args are the extra arguments you pass in my_parallel_loop(kernel, N, arg1, arg2, ...)
        """




    # Apply noise perturbation to gauge links.
    def change_Ui(self, Option_noise_type, alpha_lattice, m_noise, K, dk):             

        u1 = self.sprime.u1
        N = self.sprime.n

        N2 = N * N
        noise_N = (N // 2 + 1) if N % 2 == 0 else (N + 1) // 2             


        eta = random_state.normal(loc=0.0, scale=alpha_lattice, size=(N2 * 2 * su.GROUP_ELEMENTS))          
        # eta  = xp.reshape(eta, (N2, 2, su.GROUP_ELEMENTS)) 

        noise_kernel = xp.zeros((N, noise_N), dtype = su.GROUP_TYPE_REAL)  
	
        my_parallel_loop(compute_noise_kernel, N, Option_noise_type, N, noise_N, noise_kernel, m_noise, K, dk)     


        # Apply noise kernel in Fourier space    
        eta_for_FFT        =  xp.reshape(eta, (N, N, 2, su.GROUP_ELEMENTS))                         # Reshape eta required for FFT    # 4D: spatial × link directions × group
        eta_FFT            =  xp.fft.rfft2(eta_for_FFT, s=(N, N), axes=(0, 1))                      # Performing FFT on eta_for_FFT for x and y axis (spatial dimensions only)   
        NK                 =  noise_kernel[:, :, na, na]
        modulated_eta_FFT  =  eta_FFT * NK
        eta_after_IFFT     =  xp.fft.irfft2(modulated_eta_FFT, s=(N, N), axes=(0, 1))               # Performing Inverse FFT on eta_after_IFFT for x and y axis only       
        eta                =  xp.reshape(eta_after_IFFT, (N2, 2, su.GROUP_ELEMENTS))                # Reshape eta_after_IFFT to original shape  

       # eta = irfft2(  rfft2( eta.reshape((N, N, 2, su.GROUP_ELEMENTS)), s=(N, N), axes=(0, 1) ) * noise_kernel[:, :, na, na],  s=(N, N),  axes=(0, 1)  ).reshape((N2, 2, su.GROUP_ELEMENTS))

        my_parallel_loop(change_Ui_kernel, N2,  u1, eta)





    def Measure_EL(self):
        peta1_s = self.s.d_peta1
        peta1_sprime = self.sprime.d_peta1

        N = self.s.n
        N2 = N * N

        my_parallel_loop(Measure_EL_kernel, N2, peta1_s, peta1_sprime, self.Trace_EL2, self.Trace_ELdif2)

        EL2_avg      =  xp.mean(self.Trace_EL2)
        ELdif2_avg   =  xp.mean(self.Trace_ELdif2)

        self.Trace_ELdiff2                  =  ELdif2_avg
        self.Ratio_Trace_ELdiff2_Trace_EL2  =  ELdif2_avg / EL2_avg                     



    def Measure_BL(self):
        u1_s = self.s.d_u1
        u1_sprime = self.sprime.d_u1

        N = self.s.n
        #NC= self.s.NC
        N2 = N * N

        my_parallel_loop(Measure_BL_kernel, N2, N, self.NC, u1_s, u1_sprime, self.Trace_BL2, self.Trace_BLdif2)

        BL2_avg    = xp.mean(self.Trace_BL2)
        BLdif2_avg = xp.mean(self.Trace_BLdif2)

        self.Trace_BLdiff2                  =  BLdif2_avg
        self.Ratio_Trace_BLdiff2_Trace_BL2  =  BLdif2_avg / BL2_avg







@mynonparjit
def change_EL_kernel(xi, peta1, eta):
    #buf1  =  su.add(peta1[xi], eta[xi])					# Commented on 7 December 2025


    #eta_algebra_element[xi] = su.get_algebra_element(eta[xi])			# To check if eta[xi] is a valid algebra element    # 7 December 2025
    #buf1 = su.add(peta1[xi], eta_algebra_element[xi])
    eta_algebra_element = su.get_algebra_element(eta[xi])			# To check if eta[xi] is a valid algebra element    # 7 December 2025
    buf1 = su.add(peta1[xi], eta_algebra_element)
    
    #peta1[xi] = buf1								# Works in CPU only
    su.store(peta1[xi], buf1)    












@mynonparjit
def Measure_EL_kernel(xi, peta1_s, peta1_sprime, Trace_EL2, Trace_ELdif2):
    buf1  =  peta1_s[xi]
    buf2  =  l.add_mul(peta1_sprime[xi], peta1_s[xi], -1)

    Trace_EL2[xi]     =  su.sq(buf1)
    Trace_ELdif2[xi]  =  su.sq(buf2)




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
def Measure_BL_kernel(xi, n, Nc, u1_s, u1_sprime, Trace_BL2, Trace_BLdif2):

    Option_BL_Measure = 1                                                   # Choose : 1 OR 2 only

    if(Option_BL_Measure == 1):
        BL       = su.ah( l.plaq_pos(u1_s,      xi, 0, 1, n) )
        BL_prime = su.ah( l.plaq_pos(u1_sprime, xi, 0, 1, n) )

        Trace_BL2[xi] = su.sq(BL)

        buf1 = l.add_mul(BL_prime, BL, -1)
        Trace_BLdif2[xi] = su.sq(buf1)


    elif(Option_BL_Measure == 2):

        Trace_BL2[xi]    = Nc - su.tr(l.plaq_pos(u1_s, xi, 0, 1, n)).real        
        Trace_BLdif2[xi] = Nc - su.tr(     su.mul(  l.plaq_pos(u1_s, xi, 0, 1, n) , su.dagger(l.plaq_pos(u1_sprime, xi, 0, 1, n))  )    ).real


        # print(xi, "ALL ", l.plaq_pos(u1_s, xi, 0, 1, n)  )
        # print(xi, "REAL", l.plaq_pos(u1_s, xi, 0, 1, n).real  )

        # print(xi,  "ALL ", su.tr(l.plaq_pos(u1_s, xi, 0, 1, n)))
        # print(xi,  "REAL", su.tr(l.plaq_pos(u1_s, xi, 0, 1, n)).real  , "\n")

        # print(xi,  "ALL", su.tr(   su.dagger(l.plaq_pos(u1_s, xi, 0, 1, n))  ) )
        # print(xi,  "REAL", su.tr(  su.dagger(l.plaq_pos(u1_s, xi, 0, 1, n))  ).real  )

        #Trace_BL2[xi]    =  su.tr( su.unit() - l.plaq_pos(u1_s, xi, 0, 1, n)).real
        # Trace_BL2[xi]    =  BL
        # Trace_BLdif2[xi] =  BLprime_BL

        #print("NC" , Nc)

    # print("type(l.plaq_pos(u1_s, xi, 0, 1, n))", l.plaq_pos(u1_s, xi, 0, 1, n))  :  plaq_pos(u1_s, xi, 0, 1, n) : tuple of 4 objects
    

    """
    BL[xi] = (NC - su.tr(l.plaq_pos(U0, xi, 0, 1, n)).real) * t + (NC - su.tr(l.plaq_pos(u1, xi, 0, 1, n)).real) * (t + dt)
    BL[xi] = 0.5 * (su.sq(su.ah(l.plaq_pos(U0, xi, 0, 1, n))) * t + su.sq(su.ah(l.plaq_pos(u1, xi, 0, 1, n))) * (t+dt))


    # Bz (Beta)
    bf1 = su.zero()
    b1 = l.plaq(u1_s, xi, 0, 1, 1, 1, n)		    #  plaq(u, x, i, j, oi, oj, n): Just for reference
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(U0, ngp_index, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(U0, ngp_index, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(U0, ngp_index, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    Beta = l.add_mul(bf1, b2, +0.25)
    """




# @mynonparjit
# def Measure_EL_kernel(xi, peta1_s, peta1_sprime, Trace_EL2, Trace_ELdif2):

#     buf1 = l.add_mul(peta1_sprime[xi], peta1_s[xi], -1)

#     Trace_ELdif2[xi] = su.sq(buf1)

#     buf2 = peta1_s[xi]
#     Trace_EL2[xi] = su.sq(buf2)












#@mynonparjit                                                          
@myjit             
def compute_noise_kernel(x, Option_noise_type, n, new_n, kernel, m_noise, K, dk):       # Just for reference: my_parallel_loop(compute_noise_kernel, n, Option_noise_type,  n, noise_N, noise_kernel, m_noise, K, dk)   # Added 11.09.2025
#def compute_noise_kernel(x, m_noise, n, new_n, kernel):                                # Just for reference: my_parallel_loop(compute_noise_kernel, n, m_noise, n, noise_N, noise_kernel)                              # Commented 10.09.2025                                                                     
    
    for y in prange(new_n):                                                     
                                                      
        k2 = k2_latt(x, y, n)   

        #if (x > 0 or y > 0):                                           # Comment this statement as our kernel does not blow up at (0,0)

        if Option_noise_type == 0:                                          # No noise     
            #print("Option_noise_type = 0: No noise")
            kernel[x, y] = 1.0

        elif Option_noise_type == 1:                                        # Exponential noise
            #print("Option_noise_type = 1: Exponential noise")
            #kernel[x, y] = np.exp(-k2/m_noise**2)
            kernel[x, y] = math.exp(-k2/m_noise**2)

        elif Option_noise_type == 2:                                        # Power-law noise
            #print("Option_noise_type = 2: Power-law noise")
            kernel[x, y] = (m_noise ** 2) / (k2 + m_noise ** 2)

        elif Option_noise_type == 3:                                        # Independent noise, Theta function

            #print("Option_noise_type = 3: Independent noise, Theta function")
            #k  = np.sqrt(k2)
            k  = math.sqrt(k2)
           
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
    kx2_latt = (4.0) * (math.sin((PI * nx) / N_T) ** 2)         # In lattice units (Dimensionless)
    ky2_latt = (4.0) * (math.sin((PI * ny) / N_T) ** 2)         # In lattice units (Dimensionless)

    result = kx2_latt + ky2_latt                                # This is Discrete Lattice Momentum squared, k_latt² : in lattice units (Dimensionless)
    #result = 4.0 * (math.sin((PI * nx) / N_T) ** 2 + math.sin((PI * ny) / N_T) ** 2)            # In lattice units (Dimensionless)
    return result
