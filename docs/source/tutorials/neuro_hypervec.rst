NeuroHyperVec: 5D+ Neuroimaging Data
====================================

NeuroHyperVec extends neuroimpy's capabilities to handle neuroimaging data with more than 4 dimensions. This is essential for modern neuroimaging applications that require additional feature dimensions beyond the standard spatial and temporal dimensions.

Overview
--------

While traditional neuroimaging data is 4D (3 spatial dimensions + time), many applications require additional dimensions:

- **Multi-echo fMRI**: Multiple echo times at each timepoint
- **Spectral analysis**: Power in different frequency bands
- **Model parameters**: Multiple parameter estimates per voxel
- **Connectivity matrices**: Time-varying connections between regions

NeuroHyperVec provides efficient data structures and operations for these use cases.

Basic Usage
-----------

Creating a NeuroHyperVec
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import neuroimpy as pn
    import numpy as np

    # Create 5D space: 64x64x40 voxels, 200 timepoints, 3 features
    space = pn.NeuroSpace(dim=(64, 64, 40, 200, 3))
    data = np.random.randn(64, 64, 40, 200, 3)
    
    # Create dense hypervector
    hvec = pn.DenseNeuroHyperVec(data, space, label="my_data")
    
    # Access properties
    print(hvec.shape)           # (64, 64, 40, 200, 3)
    print(hvec.spatial_shape)   # (64, 64, 40)
    print(hvec.n_timepoints)    # 200
    print(hvec.n_features)      # 3

Data Structures
---------------

DenseNeuroHyperVec
~~~~~~~~~~~~~~~~~~

Standard dense array representation for full datasets:

.. code-block:: python

    # Dense representation stores all values
    dense_hvec = pn.DenseNeuroHyperVec(data, space)
    
    # Supports all standard operations
    mean_vec = dense_hvec.mean_features()
    subset = dense_hvec.select_features([0, 2])

SparseNeuroHyperVec
~~~~~~~~~~~~~~~~~~~

Memory-efficient representation for masked data:

.. code-block:: python

    # Create brain mask
    mask = pn.LogicalNeuroVol(brain_mask, mask_space)
    
    # Sparse data: [features x time x voxels]
    n_voxels = mask.sum
    sparse_data = np.random.randn(5, 100, n_voxels)
    
    # Create sparse hypervector
    sparse_hvec = pn.SparseNeuroHyperVec(sparse_data, mask, space)

MappedNeuroHyperVec
~~~~~~~~~~~~~~~~~~~

Memory-mapped arrays for very large datasets:

.. code-block:: python

    # Create memory-mapped hypervector
    mapped_hvec = pn.MappedNeuroHyperVec(
        filename="large_data.dat",
        space=space,
        dtype=np.float32
    )
    
    # Work with slices without loading full data
    slice_data = mapped_hvec[32, 32, 20, :, :]

Common Operations
-----------------

Feature Operations
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Average across features
    mean_vec = hvec.mean_features()  # Returns NeuroVec
    
    # Weighted average
    weights = np.array([0.5, 0.3, 0.2])
    weighted = hvec.weighted_mean_features(weights)
    
    # Select specific features
    subset = hvec.select_features([0, 2, 4])
    
    # Apply custom function
    def compute_range(features):
        return np.max(features) - np.min(features)
    
    ranges = hvec.apply_feature_func(compute_range)

Indexing and Slicing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Extract single feature as NeuroVec
    feature_0 = hvec[:, :, :, :, 0]
    
    # Extract time series for voxel
    voxel_series = hvec[32, 32, 20, :, :]  # shape: (time, features)
    
    # Extract spatial ROI
    roi_data = hvec[30:40, 30:40, 15:25, :, :]

Time Series Extraction
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Single voxel, all features
    series = hvec.series([32, 32, 20])
    # Returns: (n_timepoints, n_features)
    
    # Multiple voxels
    coords = np.array([[30, 30, 20], [35, 35, 20]])
    multi_series = hvec.series(coords)
    # Returns: (n_timepoints, n_features, n_voxels)
    
    # Specific feature only
    feature_series = hvec.series([32, 32, 20], feature=1)
    # Returns: (n_timepoints,)

Use Cases
---------

Multi-Echo fMRI
~~~~~~~~~~~~~~~

.. code-block:: python

    # Load 3-echo fMRI data
    echo_times = np.array([15.0, 30.0, 45.0])  # milliseconds
    space = pn.NeuroSpace(dim=(64, 64, 40, 300, 3))
    multi_echo = pn.DenseNeuroHyperVec(echo_data, space)
    
    # Optimal echo combination
    weights = echo_times / echo_times.sum()
    combined = multi_echo.weighted_mean_features(weights)
    
    # Extract individual echoes
    echo1 = multi_echo[:, :, :, :, 0]
    echo2 = multi_echo[:, :, :, :, 1]
    echo3 = multi_echo[:, :, :, :, 2]

Frequency Band Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Store power in 5 frequency bands
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    space = pn.NeuroSpace(dim=(64, 64, 40, 100, len(bands)))
    spectral = pn.DenseNeuroHyperVec(power_data, space)
    
    # Extract alpha band
    alpha_idx = bands.index('alpha')
    alpha_power = spectral[:, :, :, :, alpha_idx]
    
    # Find dominant frequency at each voxel/time
    dominant = spectral.apply_feature_func(np.argmax)

Parameter Maps
~~~~~~~~~~~~~~

.. code-block:: python

    # Store model parameters
    # e.g., amplitude, delay, width, baseline
    param_space = pn.NeuroSpace(dim=(64, 64, 40, 1, 4))
    params = pn.DenseNeuroHyperVec(param_data, param_space)
    
    # Extract specific parameter
    amplitudes = params[:, :, :, 0, 0]  # First parameter
    
    # Compute derived measure
    def compute_snr(params):
        amplitude = params[0]
        baseline = params[3]
        return amplitude / baseline
    
    snr_map = params.apply_feature_func(compute_snr)

I/O Operations
--------------

NeuroHyperVec uses HDF5 format for efficient storage:

.. code-block:: python

    # Save to HDF5
    pn.write_neurohypervec(hvec, "my_data.h5")
    
    # Load from HDF5
    loaded = pn.read_neurohypervec("my_data.h5")
    
    # Preserves type (dense/sparse) and metadata

Best Practices
--------------

1. **Choose the right representation**:

   - Use ``DenseNeuroHyperVec`` for full brain data
   - Use ``SparseNeuroHyperVec`` for masked/ROI data  
   - Use ``MappedNeuroHyperVec`` for datasets larger than RAM

2. **Memory management**:

   - 5D data can be very large (e.g., 64×64×40×500×10 = 800MB)
   - Consider sparse or mapped representations
   - Process in chunks when possible

3. **Feature semantics**:

   - Keep features conceptually related
   - Document what each feature represents
   - Use meaningful labels

See Also
--------

- :doc:`neuro_vectors` - 4D neuroimaging data
- :doc:`image_volumes` - 3D neuroimaging data
- :doc:`../api/core` - Core API reference