# Curraun

A 2+1D boost-invariant CGC/Glasma code for simulating the earliest stages
of heavy-ion collisions.

This code is a Numba-based version of the original code
at [curraun_cy](https://gitlab.com/dmueller/curraun_cy)
which was based on Cython and NumPy with OpenMP support.

Features:
* Standard leapfrog solver in terms of link variables and color-electric fields for SU(2) and SU(3)
* McLerran-Venugopalan model initial conditions with settings for multiple
color sheets ("rapidity slices") and transverse shapes (finite radii)
* Scripts for scanning through parameters, basic real-time visualization and plotting
* Calculation of kappa and q-hat
* Landau matching for hydrodynamical simulations

# Papers

This code has been used in the following works:

- "Jet momentum broadening in the pre-equilibrium Glasma", Andreas Ipp, David I. Müller, Daniel Schuh, [Phys.Lett.B 810 (2020) 135810](https://doi.org/10.1016/j.physletb.2020.135810), [arXiv:2009.14206](https://arxiv.org/abs/2009.14206) [hep-ph]
- "Anisotropic momentum broadening in the 2+1D Glasma: analytic weak field approximation and lattice simulations", Andreas Ipp, David I. Müller, Daniel Schuh, [Phys.Rev.D 102 (2020) 7, 074001](https://doi.org/10.1103/PhysRevD.102.074001), [arXiv:2001.10001](https://arxiv.org/abs/2001.10001) [hep-ph] 
- "Simulations of the Glasma in 3+1D", David I. Müller, PhD thesis (2019), [arXiv:1904.04267](https://arxiv.org/abs/1904.04267) [hep-ph]

# Installation
You can install all dependencies with [Anaconda](https://www.anaconda.com/distribution/):

```
conda install python=3.6 numba=0.43 six jupyter matplotlib tqdm
```

For the CUDA version of this application, one additionally requires:

```
conda install cudatoolkit=9.0 cupy
```

The CUDA toolkit version should match the graphics card driver version - see nvidia-smi.

# Example Jupyter notebooks
Jupyter notebooks are provided in the [`notebooks`](notebooks) folder.

# Launching the application
To launch a script module from the command line, go to the root directory
of this git repository and do the following:
```
python3 -m scripts.transport_cmd
```

You can change environment variables before the launch and pass additional
parameters as command line arguments:
```
export MY_NUMBA_TARGET=cuda     # use 'python', 'numba' (default), or 'cuda'
export GAUGE_GROUP=su3          # use 'su2' (default) or 'su3'
export PRECISION=single         # use 'single' or 'double' (default)
python3 -m scripts.transport_cmd -N 512 -DTS 8
```
The CUDA-version only works with a CUDA-capable graphics card.
The Python-version is slowest, but great for debugging.

To list all available parameters, do the following:
```
python3 -m scripts.transport_cmd --help
```
