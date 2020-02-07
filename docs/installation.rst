============
Installation
============
This page contains instructions on how to install pygorpho.
To use the library you must have an NVIDIA GPU and install `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ 9.2 or later.

Installing with pip
===================
Install with pip: ::

    pip install pygorpho


Installing from source
======================
First, you need a compatible C++ compiler, which supports C++14.
Then, following these instructions should allow you to build and install the package:

1. Clone the repo: ``git clone https://github.com/patmjen/pygorpho.git``
2. Change directory: ``cd pygorpho``
3. Install the required Python packages: ``pip install numpy scikit-build cmake ninja``
4. Build and install: ``python setup.py install``

That should be it! To test, run ``python``, and try to ``import pygorpho as pg``.
