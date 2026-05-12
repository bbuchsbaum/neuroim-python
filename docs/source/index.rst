.. neuroim documentation master file

neuroim
=========

**neuroim** is a comprehensive Python library for neuroimaging data analysis, providing a Python-native equivalent of the R neuroim2 package. It offers efficient, memory-aware data structures for working with 3D, 4D, and 5D+ neuroimaging datasets, along with a rich ecosystem of analysis tools for fMRI, structural imaging, and multimodal neuroimaging research.

The package combines nibabel's robust file I/O capabilities with optimized numpy-based data structures, enabling seamless integration into modern Python scientific workflows while maintaining performance and memory efficiency for large-scale neuroimaging analyses.

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: 🧠 Flexible Data Structures
        :text-align: center

        Work with 3D volumes (NeuroVol), 4D time series (NeuroVec), and 5D+ hyperdimensional data (NeuroHyperVec). Dense, sparse, and memory-mapped representations for datasets of any size.

    .. grid-item-card:: 📂 File I/O
        :text-align: center

        Read and write NIfTI and AFNI formats via nibabel. Automatic format detection, metadata preservation, and support for compressed files.

    .. grid-item-card:: 🎯 ROI Analysis
        :text-align: center

        Define regions of interest using spheres, cuboids, parcellations, or coordinate lists. Extract time series, compute statistics, and analyze local patterns.

    .. grid-item-card:: 🔍 Searchlight Analysis
        :text-align: center

        Flexible searchlight framework for local multivariate analysis. Standard, random, bootstrap, and clustered searchlight implementations.

    .. grid-item-card:: 🎨 Spatial Filtering
        :text-align: center

        Gaussian blur, bilateral filtering, guided filtering, graph-based smoothing. Create custom kernels for specialized filtering operations.

    .. grid-item-card:: 🔬 Simulation Tools
        :text-align: center

        Generate synthetic fMRI data with configurable noise models. Create confound regressors and temporal weighting schemes for method development and testing.

    .. grid-item-card:: 📊 Visualization
        :text-align: center

        Plot single slices, orthogonal views, slice montages, and statistical overlays. Flexible colormapping and integration with matplotlib.

    .. grid-item-card:: 🧮 Connected Components
        :text-align: center

        Find connected regions in binary masks using 6-, 18-, or 26-connectivity. Extract cluster sizes, locations, and spatial properties.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Migration from R

   migration/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
