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


class Lyapunov():           # All classes have a function called __init__(), which is always executed when the class is being initiated.
                            # Use the __init__() function to assign values to object properties, or other operations that are necessary to do when the object is being created:

    def __init__(self, s, sprime):              # This is the constructor method, called __init__. This is the class constructor. 
                                                # It gets automatically called when you create a new instance of the Lyapunov class. 
                                                # The self keyword refers to the instance being created.
                                                # It takes in two parameters, s and sprime, in addition to the required self.


        # Define two fields configuration: s and sprime
        # s: Original field configuration.      
        # sprime: Perturbed/nearby configuration of s                                   
        self.s = s
        self.sprime = sprime                    # They store the input arguments (s and sprime) as attributes of the class instance. 
                                                # They store the passed-in values (s and sprime) internally inside the Lyapunov object. 
                                                # They are now attributes of the object and can be accessed (with self.s and self.sprime).




        n = self.s.n                 
        # This pulls the attribute n from the object self.s, which was set in the constructor.
        # n = lattice size

        print("n = = self.s.n", n)

        self.tr_sq_el = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.tr_sq_dif = np.zeros(n*n, dtype=su.GROUP_TYPE_REAL)
        self.d_tr_sq_el = self.tr_sq_el
        self.d_tr_sq_dif = self.tr_sq_dif

        self.ratio_dif = 0.0


    # Objects can also contain methods. 
    # Methods in objects are functions that belong to the object.
    def change_el(self, alpha):                             # change_el: It is a method of class Lyapunov (It is a Function)
                                                            # change_el: This function changes (adds a random perturbation) to the sprime configuration. 
                                                            # alpha: This parameter controls the magnitude or strength of the perturbation



        peta1 = self.sprime.d_peta1  

        n = self.sprime.n                                   # Gets the lattice size n from the perturbed config

        eta = random_np.normal(loc=0.0, scale=alpha, size=(n ** 2 * su.GROUP_ELEMENTS))     
        # This generates a random array eta with values drawn from a normal (Gaussian) distribution centered at 0.0, with standard deviation alpha.
        # The total number of elements is n² × N²
        # su.GROUP_ELEMENTS: Represents the number of matrix elements, i.e., N*N for SU(N) 
        # 4 for SU(2) and 9 for SU(3)  
        # So, it is creating a random SU(N)-like perturbation field, where each lattice site (n*n) gets a set of random numbers to perturb the field.
        #print("su.GROUP_ELEMENTS", nu.GROUP_ELEMENTS)

        
        eta = eta.reshape((n * n, su.GROUP_ELEMENTS))
        # Reshapes the flat array into a 2D shape so that each lattice site has its own su.GROUP_ELEMENTS (N²)-sized vector of perturbation data.


        my_parallel_loop(change_el_kernel, n ** 2, peta1, eta)
        # This is the key part where actual computation or field update happens. It directs to the function change_el_kernel.
        # my_parallel_loop(...) is likely a wrapper that distributes the computation across multiple CPU cores or GPU threads.
        # change_el_kernel is a function (or kernel) that defines what to do at each site of the lattice.
        # The loop runs n² times — once per lattice site — applying the kernel.
        # peta1 is the buffer to store the updated/perturbed field.
        # eta is the random perturbation data passed to the kernel.







    def compute_change_el(self):                                    # compute_change_el: It is a method of class Lyapunov (It is a Function)
        peta1s = self.s.d_peta1
        peta1sprime = self.sprime.d_peta1

        n = self.s.n

        my_parallel_loop(compute_change_el_kernel, n ** 2, peta1s, peta1sprime, self.d_tr_sq_el, self.d_tr_sq_dif)

        dif_avg = np.mean(self.d_tr_sq_dif)
        el_avg = np.mean(self.d_tr_sq_el)

        self.ratio_dif = dif_avg



#TODO: Add Gaussian noise with parameter alpha
@mynonparjit
def change_el_kernel(xi, peta1, eta):           # This adds Gaussian noise eta[xi] to peta1[xi] and updates it in place
                                                # xi: lattice site index
                                                # peta1[xi]: the original electric field at site xi
                                                # eta[xi]: the random perturbation generated earlier

    buf1 = su.add(peta1[xi], eta[xi])           # buf1 = peta1[xi] + eta[xi]
    peta1[xi] = buf1                            # peta1[xi] = buf1 = peta1[xi] + eta[xi] 


@mynonparjit
def compute_change_el_kernel(xi, peta1s, peta1sprime, tr_sq_el, tr_sq_dif):
    buf1 = l.add_mul(peta1sprime[xi], peta1s[xi], -1)       # buf1 = peta1sprime[xi]+ (-1)*peta1s[xi]
                                                            # This function add_mul is defined in file lattice.py
    tr_sq_dif[xi] = su.sq(buf1)                             # At site xi, computes the trace of the squared of buf1 and store it as a scalar in tr_sq_dif.                                         

    buf2 = peta1s[xi]                                       # buf2; Stores original field configuration at site xi
    tr_sq_el[xi] = su.sq(buf2)                              # At site xi, computes the trace of the squared of buf2 and store it as a scalar in tr_sq_el. 