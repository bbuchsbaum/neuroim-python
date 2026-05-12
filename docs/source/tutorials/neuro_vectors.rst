Four-dimensional Neuroimaging Data
==================================

The neuroim package contains data structures and functions for reading, accessing, and processing 4-dimensional neuroimaging data.

Reading a four-dimensional NIFTI image
--------------------------------------

To read a 4D image, use the :func:`read_vec` function::

    import neuroim
    import numpy as np
    
    # Read a single 4D NIFTI file
    vec = neuroim.read_vec("path/to/4d_image.nii")
    print(vec.shape)
    # (64, 64, 25, 100)  # x, y, z, time
    
    print(vec)
    # DenseNeuroVec
    #   Type      : DenseNeuroVec
    #   Dimension : 64 X 64 X 25 X 100
    #   Spacing   : 3.5 X 3.5 X 5.0 X 2.0
    #   Origin    : -110.5, -88.9342, -42.75, 0.0

Reading multiple 4D images
--------------------------

You can also read multiple 4D images and concatenate them::

    # Read multiple files (here using the same file 3 times as example)
    file_list = ["image1.nii", "image2.nii", "image3.nii"]
    vec = neuroim.read_vec(file_list)
    print(vec.shape)
    # Will concatenate along the time dimension

Extracting subsets of volumes
-----------------------------

To extract a subset of volumes, use the :meth:`sub_vector` method::

    # Extract volumes 0-5 (first 6 volumes)
    vec_subset = vec.sub_vector(slice(0, 6))
    print(vec_subset.shape)
    # (64, 64, 25, 6)
    
    # Extract specific volumes
    vec_subset = vec.sub_vector([0, 2, 4, 6])
    print(vec_subset.shape)
    # (64, 64, 25, 4)

Extracting time series data
---------------------------

Single voxel time series
~~~~~~~~~~~~~~~~~~~~~~~~

To get the time series at a specific voxel, use the :meth:`series` method::

    # Get time series at voxel (10, 10, 10)
    ts = vec.series(10, 10, 10)
    print(ts.shape)
    # (100,)  # Length equals number of time points
    
    # Plot the time series
    import matplotlib.pyplot as plt
    plt.plot(ts)
    plt.xlabel('Time point')
    plt.ylabel('Signal')
    plt.title('Time series at voxel (10, 10, 10)')

Multiple voxel time series
~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract time series for multiple voxels at once::

    # Using coordinate matrix (N x 3)
    coords = np.array([[10, 10, 10],
                       [20, 20, 10],
                       [30, 30, 10]])
    
    ts_matrix = vec.series(coords)
    print(ts_matrix.shape)
    # (100, 3)  # time x voxels

ROI-based time series extraction
--------------------------------

Extract time series for all voxels in a region of interest (ROI)::

    # First, create an ROI (e.g., a sphere)
    center = [32, 32, 12]
    roi = neuroim.spherical_roi(vec, center, radius=8)
    
    # Extract time series for all voxels in the ROI
    roi_series = vec.series_roi(roi)
    print(roi_series.shape)
    # (100, n_voxels)  # where n_voxels is the number of voxels in the sphere

Using linear indices
~~~~~~~~~~~~~~~~~~~~

You can also extract ROIs using 1D indices::

    # Extract first 100 voxels (in linear index order)
    roi_series = vec.series(np.arange(100))
    print(roi_series.shape)
    # (100, 100)  # time x voxels

Working with masks
------------------

Often we want to extract data only from voxels within a brain mask::

    # Load a binary mask
    mask = neuroim.read_vol("brain_mask.nii")
    
    # Get indices of non-zero voxels
    mask_indices = np.where(mask.data.ravel(order='F'))[0]
    
    # Extract time series for masked voxels
    masked_series = vec.series(mask_indices)
    print(f"Extracted {masked_series.shape[1]} voxels from mask")

Converting between formats
--------------------------

Dense to Sparse
~~~~~~~~~~~~~~~

For memory efficiency with masked data::

    # Convert to sparse representation using a mask
    mask_vol = neuroim.LogicalNeuroVol(mask.data > 0, mask.space)
    sparse_vec = vec.as_sparse(mask_vol)
    
    print(type(sparse_vec))
    # <class 'neuroim.neuro_vec.SparseNeuroVec'>
    
    # Sparse vectors work the same way
    ts = sparse_vec.series(10, 10, 10)

Converting to matrix format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For machine learning applications, you might want data as a 2D matrix::

    # Get as voxels x time matrix
    matrix = vec.as_matrix()
    print(matrix.shape)
    # (n_voxels, n_timepoints)

Concatenating NeuroVecs
-----------------------

Combine multiple 4D datasets along the time dimension::

    # Create some example vectors
    vec1 = neuroim.neurovec(np.random.randn(10, 10, 10, 50))
    vec2 = neuroim.neurovec(np.random.randn(10, 10, 10, 30))
    
    # Concatenate
    combined = vec1.concat(vec2)
    print(combined.shape)
    # (10, 10, 10, 80)

Extracting individual volumes
-----------------------------

Get individual 3D volumes from the 4D data::

    # Get the first volume
    vol0 = vec[..., 0]
    print(type(vol0))
    # <class 'neuroim.neuro_vol.DenseNeuroVol'>
    
    # Get multiple volumes as a list
    vols = vec.vols([0, 10, 20, 30])
    print(len(vols))
    # 4

Time series preprocessing
-------------------------

Scale each voxel's time series::

    # Center and scale each time series
    vec_scaled = vec.scale_series(center=True, scale=True)
    
    # Verify: each voxel should have mean≈0, std≈1
    ts = vec_scaled.series(10, 10, 10)
    print(f"Mean: {np.mean(ts):.6f}, Std: {np.std(ts):.6f}")

Memory-mapped vectors
---------------------

For very large datasets that don't fit in memory::

    # Create a memory-mapped vector
    big_data = np.random.randn(100, 100, 50, 1000)  # Large dataset
    space = neuroim.NeuroSpace(dim=[100, 100, 50, 1000])
    
    # This keeps data on disk
    big_vec = neuroim.BigNeuroVec(big_data, space)
    
    # Access works the same way
    ts = big_vec.series(50, 50, 25)
    print(ts.shape)
    # (1000,)
    
    # Data stays on disk until accessed
    print(big_vec)
    # BigNeuroVec (memory-mapped)
    #   ...