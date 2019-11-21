.. pygorpho documentation master file, created by
   sphinx-quickstart on Mon Oct 28 00:44:00 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================
Welcome to pygorpho
===================
Welcome to the documentation for pygorpho. This is a library for fast 3D mathematical morphology using CUDA.

Features
========

* Dilation and erosion for grayscale 3D images.
* Support for flat or grayscale structuring elements.
* A van Herk/Gil-Werman implementation for fast dilation/erosion with flat line segments in 3D.
* Automatic block processing for 3D images which can't fit in GPU memory.

.. toctree::
   :maxdepth: 4
   :caption: Contents

   installation
   api-doc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`


Resources
=========
* Free software: MIT license
* Source code: https://github.com/patmjen/pygorpho