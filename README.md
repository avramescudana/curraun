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

# Installation
# Installation
You can install all dependencies with [Anaconda](https://www.anaconda.com/distribution/):

```
conda install python=3.6 numba=0.43 six jupyter matplotlib tqdm
```

For the CUDA version of this application, one additionally requires:

```
conda install cudatoolkit=9.0
```

The CUDA toolkit version should match the graphics card driver version - see nvidia-smi.

# Example Jupyter notebooks
Jupyter notebooks are provided in the `notebooks` folder.

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
