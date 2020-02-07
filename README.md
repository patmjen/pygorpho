# pygorpho

Python bindings for [gorpho](https://github.com/patmjen/gorpho).

This is a Python library for fast 3D mathematical morphology using CUDA. Currently, the library provides:
* Dilation and erosion for grayscale 3D images.
* Support for flat or grayscale structuring elements.
* A van Herk/Gil-Werman implementation for fast dilation/erosion with flat line segments in 3D.
* Automatic block processing for 3D images which can't fit in GPU memory.

**Documentation** can be found on [https://pygorpho.readthedocs.io](https://pygorpho.readthedocs.io)

## Installation
First, make sure you have [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 9.2 or later installed. Then, install with pip:
```
pip install pygorpho
```

## Installing from source
Again, make sure you have [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 9.2 or later installed. Also, you need a compatible C++ compiler, which supports C++14. Then, following these instructions should allow you to build and install the package:

1. Clone the repo: `git clone https://github.com/patmjen/pygorpho.git`
2. Change directory: `cd pygorpho`
3. Install the required Python packages: `pip install numpy scikit-build cmake ninja`
4. Build and install: `python setup.py install`

That should be it! To test, navigate to another directory (e.g. `cd ..`), run `python`, and try to `import pygorpho as pg`.

**Note**: the reason you have to navigate to another directory before importing the package is because the build step makes a shared library. This will be installed with pygorpho, but if you stay in the folder where you cloned the git repo., `import pygorpho` will import from the local version, and not the installed version.
