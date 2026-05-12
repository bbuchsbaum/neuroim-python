:orphan:

Tutorial Notebooks
==================

This section contains interactive Jupyter notebooks that demonstrate key functionality
of the neuroim library. All notebooks are self-contained and generate their own
example data.

.. toctree::
   :maxdepth: 1
   :caption: Available Notebooks:

   notebooks/image_volumes
   notebooks/neuro_vectors
   notebooks/regions_of_interest
   notebooks/pipelines

Getting Started
---------------

All notebooks can be run directly without any external data files. To get started:

1. Install neuroim and Jupyter::

    pip install neuroim jupyter

2. Clone the repository and navigate to the notebooks directory::

    git clone https://github.com/yourusername/neuroim-python.git
    cd neuroim-python/docs/source/tutorials/notebooks

3. Start Jupyter Notebook::

    jupyter notebook

4. Open any notebook and run all cells

Environment Validation
----------------------

Before running the notebooks, you can validate your environment::

    python validate_environment.py

This will check that all required packages are installed and functioning correctly.

Notebook Descriptions
---------------------

**image_volumes.ipynb**
    Introduction to 3D neuroimaging volumes, including creation, manipulation,
    and conversion between different representations (dense, sparse, logical).

**neuro_vectors.ipynb**
    Working with 4D neuroimaging data such as time series. Covers vector
    creation, time series extraction, concatenation, and memory-mapped arrays.

**regions_of_interest.ipynb**
    Creating and manipulating regions of interest (ROIs). Includes spherical,
    square, and cuboid ROIs, as well as searchlight analysis techniques.

**pipelines.ipynb**
    Building complete analysis pipelines. Demonstrates connected components
    analysis, cluster-based operations, and parallel processing with searchlight.

Testing the Notebooks
---------------------

Two testing methods are available:

1. **Quick functional test** (no Jupyter required)::

    python test_notebooks_simple.py

2. **Full notebook execution** (requires nbformat and nbconvert)::

    python notebook_execution_test.py --full

Additional Resources
--------------------

- `GitHub Repository <https://github.com/bbuchsbaum/neuroim-python>`_ - Source code
- :doc:`API Reference <../api/index>` - Detailed API documentation