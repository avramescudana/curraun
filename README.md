# Curraun, a PyGlasma GPU solver

A 2+1D boost-invariant CGC/Glasma code for simulating the earliest stages
of heavy-ion collisions, forked from [gitlab.com/curraun](https://gitlab.com/openpixi/curraun).

This code is a Numba-based version of the original code
at [curraun_cy](https://gitlab.com/dmueller/curraun_cy)
which was based on Cython and NumPy with OpenMP support.

Features:
* Standard leapfrog solver in terms of link variables and color-electric fields for $\mathrm{SU}(2)$ and $\mathrm{SU}(3)$
* McLerran-Venugopalan model initial conditions with settings for multiple
color sheets ("rapidity slices") and transverse shapes (finite radii)
* Scripts for scanning through parameters, basic real-time visualization and plotting
* Calculation of $\kappa$ and $\hat{q}$ transport coefficient
* Landau matching for hydrodynamical simulations
- [ ] Update list of features

## Installation
#### Check CUDA version
If you intend to use the GPU version of the code, you need a [CUDA](https://www.nvidia.com/en-gb/geforce/technologies/cuda/)-capable NVIDIA graphics card. Additionally, GPU drivers have to be installed. Assuming they are, check the CUDA version using

```
nvidia-smi
```

and the CUDA drivers version (they should in principle match the CUDA version) with

```
nvcc --version
```

Remember the CUDA version `$CUDA_VERSION`, this will be important in the next step, especially for the installation of `CuPy`. 

All the Python packages used by `curraun` can be intalled with your preffer Python package manager. Here we present a brief tutorials using `conda` and `pip`.

#### Using `conda`
You can install all dependencies with [Anaconda](https://www.anaconda.com/distribution/). If Anaconda is not installed, follow the tutorial [install Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html). 

Equipped with a functional Anaconda, proceed to install the necessary packages. First, create a custom Python environment

```
mkdir $LOCATION/condacurraun
conda create --prefix $LOCATION/condacurraun
conda activate $LOCATION/condacurraun
```

where `$LOCATION` is your desired location. Then, install the Python version you prefer. Here we install the latest Python version compatible with `CuPy` according to the [`CuPy` documentation](https://docs.cupy.dev/en/stable/install.html#installation).

```
conda install python=3.12
```

Now install `CuPy` and the `CUDA Toolkit` compatible with your `$CUDA_VERSION`

```
conda install -c conda-forge cupy cudatoolkit=$CUDA_VERSION
```

It turns out that at this date *(Dec 23)* the latest version of the `CUDA Toolkit` available through the Anaconda channel `conda-forge` is 11.8, [link here](https://anaconda.org/conda-forge/cudatoolkit) so if your `$CUDA_VERSION`>=11.8, install this one. The installation with `pip` has newer versions of `CUDA Toolkit` but this should't be a problem. 

Finally, install the remianing necessary packages. Conda should take care of the package version but lately it has been slow so prepare to wait for a while... A possible sollution would be [Libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community).
- [ ] Look into `Libmamba`

```
conda install numba tqdm matplotlib jupyter
```

- [ ] Check if these are all the necessary packages
- [ ] Split the packages into mandatory (like `numba`) and auxiliary (like `matplotlib`)
- [ ] Create a [Conda environment YAML file](https://saturncloud.io/blog/how-to-create-a-conda-environment-based-on-a-yaml-file-a-guide-for-data-scientists/) to install all these packages in one go

#### Using pip

For this installation, you may use your local Python>=3.6, otherwise `Numba` won't be compatible, [link here](https://numba.pydata.org/numba-doc/dev/user/installing.html). In case you run on a remote server where you don't have acces to installing a specific custom version `$PYTHON_VERSION` of Python, you may install one using Conda

```
mkdir condapython$PYTHON_VERSION
conda create --prefix $LOCATION/condapython$PYTHON_VERSION/
conda activate $LOCATION/condapython$PYTHON_VERSION
conda install python=$PYTHON_VERSION
```

Then, use `pip` to install both `CuPy` and `CUDA Toolkit`. Carefully check the [documentation](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi) and choose the version of `cupy-cudaXYZ` compatible with your `$CUDA_VERSION`

```
pip install cupy-cudaXYZ
```

Finally, install the remaning packages

```
pip install numba tqdm matplotlib jupyter
```

- [ ] Check if these are all the necessary packages

## Apptainer

- [ ] Add the image and running commands for the `curraun` container

## Branches
The bare Glasma solver is located in the main branch `master`. The Wong solver may be found in the branch `wong`. Additionaly functionalities will be added later on in new brances.

- [ ] Document the `wong` branch


## Example Jupyter notebooks
Jupyter notebooks are provided in the [`notebooks`](notebooks) folder.

- [ ] Add more details

## Launching the application
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

## Papers

This code has been used in the following works:

- "Jet momentum broadening in the pre-equilibrium Glasma", Andreas Ipp, David I. Müller, Daniel Schuh, [Phys.Lett.B 810 (2020) 135810](https://doi.org/10.1016/j.physletb.2020.135810), [arXiv:2009.14206](https://arxiv.org/abs/2009.14206) [hep-ph]
- "Anisotropic momentum broadening in the 2+1D Glasma: analytic weak field approximation and lattice simulations", Andreas Ipp, David I. Müller, Daniel Schuh, [Phys.Rev.D 102 (2020) 7, 074001](https://doi.org/10.1103/PhysRevD.102.074001), [arXiv:2001.10001](https://arxiv.org/abs/2001.10001) [hep-ph] 
- "Simulations of the Glasma in 3+1D", David I. Müller, PhD thesis (2019), [arXiv:1904.04267](https://arxiv.org/abs/1904.04267) [hep-ph]
- [ ] Update list of papers