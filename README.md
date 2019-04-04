# Curraun

A 2+1D boost-invariant CGC/Glasma code for simulating the earliest stages
of heavy-ion collisions.

This code is a Numba-based version of the original code
at [curraun_cy](https://gitlab.com/dmueller/curraun_cy)
which was based on Cython and NumPy with OpenMP support.

Features:
* Standard leapfrog solver in terms of link variables and color-electric fields
* McLerran-Venugopalan model initial conditions with settings for multiple
color sheets ("rapidity slices")
* Scripts for scanning through parameters, basic real-time visualization and plotting
* Calculation of kappa and q-hat

# Installation
This application requires [Numba](http://numba.pydata.org/). You can install Numba with PIP
```
pip install numba==0.43
```

or with [Anaconda](https://www.anaconda.com/distribution/)
```
conda install numba=0.43
```

Please install the most recent version of Numba, at least version 0.43 or higher.
The CUDA version of this application has been tested
using the following package versions:
```
conda install python=3.6 numba=0.43 cudatoolkit=9.0 six
```

The CUDA toolkit version should match the graphics card driver version - see nvidia-smi.

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
