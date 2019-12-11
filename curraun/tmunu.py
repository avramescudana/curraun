"""
    A model for computing all components of the energy momentum tensor as a function of the transverse
    coordinate at some fixed proper time. This can be used to perform Landau matching to hydrodynamics
    e.g. MUSIC.
"""

from curraun.numba_target import myjit, prange, my_parallel_loop, use_cuda
from numba import jit
import numpy as np
import curraun.lattice as l
import curraun.su as su
import curraun.kappa as kappa
if use_cuda:
    import numba.cuda as cuda


class EnergyMomentumTensor:
    def __init__(self, s):
        self.s = s
        self.n = s.n

        # 10 independent components of T_\mu\nu in the following order
        # diagonal, energy flux, shear
        # T00, T11, T22, T33, T01, T02, T03, T12, T13, T23
        # where 0 corresponds to \tau, 1,2 correspond to transverse coordinates and 3 is \eta.
        self.t_munu = np.zeros((self.n ** 2, 10), dtype=su.GROUP_TYPE_REAL)

        self.d_t_munu = self.t_munu

        if use_cuda:
            self.copy_to_device()

    def copy_to_device(self):
        self.d_t_munu = cuda.to_device(self.t_munu)

    def copy_to_host(self):
        self.d_t_munu.copy_to_host(self.t_munu)
        self.t_munu /= self.s.g ** 2

    def compute(self):
        # compute contributions in 2d
        u0 = self.s.d_u0

        aeta0 = self.s.d_aeta0

        pt0 = self.s.d_pt0
        pt1 = self.s.d_pt1

        peta0 = self.s.d_peta0
        peta1 = self.s.d_peta1

        t = self.s.t
        n = self.n

        my_parallel_loop(tmunu_kernel, n ** 2, n, u0, aeta0, peta1, peta0, pt1, pt0, t, self.d_t_munu)


# kernels
@myjit
def tmunu_kernel(xi, n, u0, aeta0, peta1, peta0, pt1, pt0, tau, t_munu):
    # Compute correctly averaged field strength components
    # Electric components: spatial and temporal (for Ex, Ey) and temporal (Ez)
    # Magnetic components: only spatial averaging (one direction for Bx, By, two for Bz)
    i = 0
    Ex = su.zero()
    Ex = su.add(Ex, pt1[xi, i])
    Ex = su.add(Ex, pt0[xi, i])
    xs = l.shift(xi, i, -1, n)
    b1 = l.act(su.dagger(u0[xs, i]), pt1[xs, i])
    Ex = su.add(Ex, b1)
    b1 = l.act(su.dagger(u0[xs, i]), pt0[xs, i])
    Ex = su.add(Ex, b1)
    Ex = su.mul_s(Ex, 0.25 / tau)
    
    i = 1
    Ey = su.zero()
    Ey = su.add(Ey, pt1[xi, i])
    Ey = su.add(Ey, pt0[xi, i])
    xs = l.shift(xi, i, -1, n)
    b1 = l.act(su.dagger(u0[xs, i]), pt1[xs, i])
    Ey = su.add(Ey, b1)
    b1 = l.act(su.dagger(u0[xs, i]), pt0[xs, i])
    Ey = su.add(Ey, b1)
    Ey = su.mul_s(Ey, 0.25 / tau)

    Ez = su.zero()
    Ez = l.add_mul(Ez, peta1[xi], 0.5)
    Ez = l.add_mul(Ez, peta0[xi], 0.5)
    # Ez = su.mul_s(Ez, tau)

    b1 = l.transport(aeta0, u0, xi, 1, +1, n)
    b2 = l.transport(aeta0, u0, xi, 1, -1, n)
    b2 = l.add_mul(b1, b2, -1.0)
    Bx = su.mul_s(b2, -0.5 / tau) # check these signs

    b1 = l.transport(aeta0, u0, xi, 0, +1, n)
    b2 = l.transport(aeta0, u0, xi, 0, -1, n)
    b2 = l.add_mul(b1, b2, -1.0)
    By = su.mul_s(b2, +0.5 / tau) # check these signs

    bf1 = su.zero()
    b1 = l.plaq(u0, xi, 0, 1, 1, 1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, xi, 0, 1, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, +0.25)

    b1 = l.plaq(u0, xi, 1, 0, 1, -1, n)
    b2 = su.ah(b1)
    bf1 = l.add_mul(bf1, b2, -0.25)

    b1 = l.plaq(u0, xi, 1, 0, -1, -1, n)
    b2 = su.ah(b1)
    Bz = l.add_mul(bf1, b2, +0.25)

    # 0-3: Diagonal components
    eEx = dot(Ex, Ex)
    eEy = dot(Ey, Ey)
    eEz = dot(Ez, Ez)

    eBx = dot(Bx, Bx)
    eBy = dot(By, By)
    eBz = dot(Bz, Bz)

    # 0: Energy density
    t_munu[xi, 0] = eEx + eEy + eEz + eBx + eBy + eBz

    # 1: Transverse pressure x
    t_munu[xi, 1] = (- eEx + eEy + eEz - eBx + eBy + eBz)

    # 2: Transverse pressure y
    t_munu[xi, 2] = (+ eEx - eEy + eEz + eBx - eBy + eBz)

    # 3: Longitudinal pressure
    t_munu[xi, 3] = (eEx + eEy - eEz + eBx + eBy - eBz) * tau ** 2

    # 4-6: Energy flux
    # 4: S_x
    t_munu[xi, 4] = 2 * ( - dot(Ey, Bz) + dot(Ez, By))

    # 5: S_y
    t_munu[xi, 5] = 2 * ( - dot(Ez, Bx) + dot(Ex, Bz))

    # 6: S_\eta
    t_munu[xi, 6] = 2 * ( - dot(Ex, By) + dot(Ey, Bx)) * tau

    # 6-10: Shear stress
    # 7: s_xy
    t_munu[xi, 7] = 2 * ( - dot(Ex, Ey) - dot(Bx, By))

    # 8: s_xz
    t_munu[xi, 8] = 2 * ( - dot(Ex, Ez) - dot(Bx, Bz)) * tau

    # 9: s_yz
    t_munu[xi, 9] = 2 * ( - dot(Ey, Ez) - dot(By, Bz)) * tau


@myjit
def dot(a0, a1):
    return su.tr(su.mul(a0, su.dagger(a1))).real


def convert_to_matrix(t_munu):
    n = int(np.sqrt(t_munu.shape[0]))
    t_munu = t_munu.reshape(n, n, 10)
    t_matrix = np.zeros((n, n, 4, 4))

    # fill non-diagonal elements
    t_matrix[:, :, 0, 1] = t_munu[:, :, 4]
    t_matrix[:, :, 0, 2] = t_munu[:, :, 5]
    t_matrix[:, :, 0, 3] = t_munu[:, :, 6]

    t_matrix[:, :, 1, 0] = t_munu[:, :, 4]
    t_matrix[:, :, 2, 0] = t_munu[:, :, 5]
    t_matrix[:, :, 3, 0] = t_munu[:, :, 6]

    t_matrix[:, :, 1, 2] = t_munu[:, :, 7]
    t_matrix[:, :, 1, 3] = t_munu[:, :, 8]
    t_matrix[:, :, 2, 3] = t_munu[:, :, 9]

    t_matrix[:, :, 2, 1] = t_munu[:, :, 7]
    t_matrix[:, :, 3, 1] = t_munu[:, :, 8]
    t_matrix[:, :, 3, 2] = t_munu[:, :, 9]

    # fill in diagonal
    t_matrix[:, :, 0, 0] = t_munu[:, :, 0]
    t_matrix[:, :, 1, 1] = t_munu[:, :, 1]
    t_matrix[:, :, 2, 2] = t_munu[:, :, 2]
    t_matrix[:, :, 3, 3] = t_munu[:, :, 3]

    return t_matrix


def landau(s, E0, downsample_step = 0, use_reduced_T=True):
    """
    This function computes the energy momentum tensor and diagonalizes it to obtain the local rest frame (LRF) energy
    density and the flow velocity components u^\tau, u^x, u^y. The fourth component u^\eta can be computed from
    u_\mu u^\mu = 1.

    The first argument `s` is the Simulation object, which has been evolved up to the proper time \tau where one
    intends to do the Landau matching.

    The second argument `E0` is the energy scale used in the simulation, i.e. the inverse lattice spacing in units of
    GeV.

    The third argument `downsample_step` is an integer, which specifies how much the simulation data should be
    downsampled. Yang-Mills simulations typically need a finer lattice than hydrodynamical simulations. If one has a
    lattice of 1024 x 1024 and chooses downsample_step = 4, the returned data will be on a 256 x 256 lattice.

    If one chooses use_reduced_T = True, then the eigenvalue problem will be solved using only the 3x3 submatrix of
    T^{\mu\nu}. This is equivalent to neglecting all \eta components of T^{\mu\nu}.

    This functions returns the LRF energy density `epsilon`, the flow velocity components `u` and the downsampled
    energy momentum tensor `T` with lowered indicies, i.e. T_{\mu\nu}.
    """

    # Define hbar * c in units of GeV * fm
    hbarc = 0.197326

    # Lattice spacing in fm
    aT = hbarc / E0

    # The proper time at which the simulation stopped (in fm):
    tau = s.t * aT

    # These lines of code return `T`, which is T_{\mu\nu}
    t_munu = EnergyMomentumTensor(s)
    t_munu.compute()
    t_munu.copy_to_host()
    T = convert_to_matrix(t_munu.t_munu)

    # Restore physical units [GeV / fm^3]
    T *= E0 ** 4 / s.g ** 2 / hbarc ** 3

    # Downsampling for smaller simulation boxes in hydrodynamics
    if downsample_step > 0:
        step = int(downsample_step)
        n = T.shape[0]
        T2 = np.zeros((n // step, n // step, 3, 3))

        for x in range(n // step):
            for y in range(n // step):
                reduced_T = T[step * x:step * (x + 1), step * y:step * (y + 1), :, :]
                T2[x, y, :, :] = np.mean(np.mean(reduced_T, axis=0), axis=0)

        # Overwrite
        T = T2

    T_orig = T.copy()

    # We reduce the full energy momentum tensor to the 3x3 sub-matrix if option is selected.
    if use_reduced_T is True:
        T = T[:, :, 0:3, 0:3]

    n = T.shape[0]
    d = T.shape[2]

    # We pull up the first Lorentz index
    T[:, :, 0, :] *= +1.0
    T[:, :, 1, :] *= -1.0
    T[:, :, 2, :] *= -1.0

    if use_reduced_T is not True:
        T[:, :, 3, :] *= - tau ** 2

    # Now we solve the Eigenvalue problem:
    # W contains all the eigenvalues
    # V contains all the eigenvectors
    W, V = np.linalg.eig(T)

    # We have to select the velocity vector, i.e. the timelike eigen vector
    if use_reduced_T is True:
        def m_norm(v, tau):
            return v[0] ** 2 - v[1] ** 2 - v[2] ** 2
    else:
        def m_norm(v, tau):
            return v[0] ** 2 - v[1] ** 2 - v[2] ** 2 - tau ** 2 * v[3] ** 2

    # We take count of the lattice cells, where the diagonalization process has failed.
    number_of_problematic_cells_a = 0       # Type A: no timelike vector was found
    number_of_problematic_cells_b = 0       # Type B: no positive LRF energy was found

    # This is where we put the energy density and the flow velocity
    epsilon = np.zeros((n, n))
    u = np.zeros((n, n, 3))

    for x in range(n):
        for y in range(n):
            w = W[x, y].real
            v = V[x, y].real

            # Select the largest norm, which is hopefully positive
            max_index = -1
            max_norm = -np.inf

            d = w.shape[0]
            for i in range(d):
                i_u = v[:, i]
                u_norm = m_norm(i_u, tau)

                if u_norm > max_norm:
                    max_norm = u_norm
                    max_index = i

            # If no timelike vector can be found, simply set the LRF energy density to T^{\tau\tau}
            # and set the flow velocity to zero.
            if max_norm < 0:
                epsilon[x, y] = T[x, y, 0, 0]
                u[x, y, 0] = 1.0
                u[x, y, 1:3] = 0.0
                number_of_problematic_cells_a += 1
            else:
                # If a positive norm was found, normalize the eigenvector to get the flow velocity.
                i_u = v[:, max_index]
                u_norm = m_norm(i_u, tau)
                i_u /= np.sqrt(u_norm)

                # Flip signs if necessary (timelike four velocity should point forward in time)
                if i_u[0] < 0:
                    i_u *= (-1)

                # Set the correct flow velocity (just the first three components)
                u[x, y, :] = i_u[0:3]

                # Set the energy density. If the associated energy density turns out to be negative
                # (which can happen in empty regions or due to numerical issues), set to T^{\tau\tau}
                # and reset the flow velocity.
                if w[max_index] > 0:
                    epsilon[x, y] = w[max_index]
                else:
                    number_of_problematic_cells_b += 1
                    epsilon[x, y] = T[x, y, 0, 0]
                    u[x, y, 0] = 1.0
                    u[x, y, 1:3] = 0.0

    if number_of_problematic_cells_a > 0:
        print("While solving the eigenvalue problem {} of {} cells did not have a timelike velocity.".format(
            number_of_problematic_cells_a, n ** 2
        ))

    if number_of_problematic_cells_a > 0:
        print("While solving the eigenvalue problem {} of {} cells did not have a positive LRF energy density.".format(
            number_of_problematic_cells_b, n ** 2
        ))

    return epsilon, u, T_orig
