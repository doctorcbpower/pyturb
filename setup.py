from setuptools import setup, find_packages

setup(
    name="pyturb",
    version="0.1.0",
    packages=find_packages(),
    license="GPL-3.0",
    description="Scripts to generate and measure turbulent spectra in hydrodynamical simulations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Chris Power (chris.power@uwa.edu.au)",
    install_requires=["h5py",        # HDF5 support
                      "numpy",       # numerical routines
		      "numba",	     # accelerated numerical routines
		      "scipy",	     # useful algorithms for e.g. FFTs
		      "matplotlib",  # plotting 
		      "pykdtree",    # fast KD tree construction
		      "tqdm",	     # progress tracking
		      "PyWavelets",  # wavelet transform support
                     ],  

)
