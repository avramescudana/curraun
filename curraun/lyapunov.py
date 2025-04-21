from curraun.numba_target import myjit, my_parallel_loop, use_cuda, mynonparjit
import numpy as np
import curraun.lattice as l
import curraun.su as su
import os
import math

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
        self.tr_sq_el = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.tr_sq_dif = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.d_tr_sq_el = self.tr_sq_el
        self.d_tr_sq_dif = self.tr_sq_dif

        self.ratio_dif = 0.0

        # if use_cuda:
        #     self.copy_to_device()

    # def copy_to_device(self):

    # def copy_to_host(self):


    def change_el(self, alpha):

    #!def change_el(self, alpha, m_noise):
        peta1 = self.sprime.d_peta1

        n = self.sprime.n

        eta = random_np.normal(loc=0.0, scale=alpha, size=(n ** 2 * su.GROUP_ELEMENTS))
        eta = eta.reshape((n * n, su.GROUP_ELEMENTS))


            # Learn about the Fourier transforms 
            #Add Fourier transform

       #! new_n = (n // 2 + 1) if n % 2 == 0 else (n + 1) // 2
       #! noise_kernel = np.zeros((n, new_n), dtype=su.GROUP_TYPE_REAL)
        
        #print("mnoise =", m_noise)
        # print("n =", n)
        #print("new_n =", new_n)

        #! my_parallel_loop(compute_noise_kernel, n, m_noise, n, new_n,  noise_kernel)

        #def compute_noise_kernel(x, mass, n, new_n, kernel):

        #eta = irfft2(
         #     rfft2(eta.reshape((n, n, su.ALGEBRA_ELEMENTS)), s=(n, n), axes=(0, 1)) * noise_kernel[:, :, na],
          #    s=(n, n), axes=(0, 1)).reshape((n ** 2, su.ALGEBRA_ELEMENTS))

       # eta = eta.reshape((n, n, su.ALGEBRA_ELEMENTS))
       # eta = rfft2(eta, s=(n, n), axes=(0, 1))
        #eta = eta* 0        #noise_kernel[:, :, na]  # * 0
      #  eta = irfft2(eta,  s=(n, n), axes=(0, 1)) 
      #  eta = eta.reshape((n ** 2, su.ALGEBRA_ELEMENTS))

        

        my_parallel_loop(change_el_kernel, n ** 2, peta1, eta)





    def compute_change_el(self):
        peta1s = self.s.d_peta1
        peta1sprime = self.sprime.d_peta1

        n = self.s.n

        my_parallel_loop(compute_change_el_kernel, n ** 2, peta1s, peta1sprime, self.d_tr_sq_el, self.d_tr_sq_dif)

        dif_avg = np.mean(self.d_tr_sq_dif)
        el_avg = np.mean(self.d_tr_sq_el)

        # self.ratio_dif = dif_avg / el_avg
        self.ratio_dif = dif_avg


        # if use_cuda:
        #     self.copy_to_host()





#TODO: Add Gaussian noise with parameter alpha
@mynonparjit
def change_el_kernel(xi, peta1, eta):
    # buf1 = su.mul_s(peta1[xi], 1.1)
    buf1 = su.add(peta1[xi], eta[xi])
    # eta = random_np.normal(loc=0.0, scale=alpha, size=(su.GROUP_ELEMENTS))
    # buf1 = su.add(peta1[xi], eta)tau_max = 10.0 / g2mu * hbarc  # Maximum proper time

    peta1[xi] = buf1

    #field = random_np.normal(loc=0.0, scale=g ** 2 * mu / math.sqrt(num_sheets), size=(n ** 2 * su.ALGEBRA_ELEMENTS))
     #       field = field.reshape((n * n, su.ALGEBRA_ELEMENTS))

    # alpha = 0.0001
    # pert =  random_np.normal(loc=0.0, scale=alpha,  size=())

@mynonparjit
def compute_change_el_kernel(xi, peta1s, peta1sprime, tr_sq_el, tr_sq_dif):
    buf1 = l.add_mul(peta1sprime[xi], peta1s[xi], -1)
    tr_sq_dif[xi] = su.sq(buf1) 

    buf2 = peta1s[xi]
    tr_sq_el[xi] = su.sq(buf2)




       # my_parallel_loop(compute_noise_kernel, n, m, n, new_n,  noise_kernel)


#@mynonparjit
#def compute_noise_kernel(x, mass, n, new_n, kernel):
    # for y in range(new_n): ....commented already
    #for y in prange(new_n):
       #k2 = k2_latt(x, y, n)
       #if (x > 0 or y > 0) :             #        if (x > 0 or y > 0) and k2 <= uv ** 2:
          #  kernel[x, y] = np.exp(-k2/mass**2)          #/ (k2 + mass ** 2)



# @myjit
#@mynonparjit
#def k2_latt(x, y, nt):
 #  result = 4.0 * (math.sin((PI * x) / nt) ** 2 + math.sin((PI * y) / nt) ** 2)
 #  return result
