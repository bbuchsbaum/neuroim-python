Installation
============

Requirements
------------

neuroimpy requires Python 3.8 or later. The main dependencies are:

* numpy >= 1.20
* scipy >= 1.7
* nibabel >= 3.0
* scikit-learn >= 0.24
* h5py >= 3.0

Installation from PyPI
----------------------

The easiest way to install neuroimpy is using pip::

    pip install neuroimpy

Installation from Source
------------------------

To install from source, clone the repository and install using pip::

    git clone https://github.com/yourusername/neuroimpy.git
    cd neuroimpy
    pip install -e .

Development Installation
------------------------

For development, install with the development dependencies::

    git clone https://github.com/yourusername/neuroimpy.git
    cd neuroimpy
    pip install -e ".[dev]"

This will install additional packages needed for testing and documentation:

* pytest
* pytest-cov
* sphinx
* sphinx-rtd-theme
* nbsphinx

Verifying Installation
----------------------

To verify that neuroimpy is installed correctly, you can run::

    python -c "import neuroimpy; print(neuroimpy.__version__)"

Or run the test suite::

    pytest tests/