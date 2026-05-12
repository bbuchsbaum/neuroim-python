Working with Image Volumes
==========================

This tutorial covers the basics of working with 3D neuroimaging volumes in neuroim.

Reading a NIFTI formatted image volume
--------------------------------------

The primary way to read a volumetric image file is to use :func:`read_vol`::

    import neuroim
    import numpy as np
    
    # Read a NIFTI file
    vol = neuroim.read_vol("path/to/image.nii.gz")

Working with image volumes
--------------------------

Information about the geometry of the image volume can be displayed::

    print(vol)
    # Output:
    # DenseNeuroVol
    #   Type      : DenseNeuroVol
    #   Dimension : 64 X 64 X 25
    #   Spacing   : 3.5 X 3.5 X 5.0
    #   Origin    : -110.5, -88.9342, -42.75
    #   Range     : [0.000, 1.000]

:func:`read_vol` returns a :class:`NeuroVol` object which extends a numpy array and has 3 dimensions (x, y, z)::

    print(type(vol))
    # <class 'neuroim.neuro_vol.DenseNeuroVol'>
    
    print(isinstance(vol.data, np.ndarray))
    # True
    
    print(vol.shape)
    # (64, 64, 25)
    
    # Access individual voxels
    print(vol[0, 0, 0])
    # 0.0
    
    print(vol[63, 63, 24])
    # 0.0

Arithmetic operations
---------------------

Arithmetic can be performed on images as if they were ordinary arrays::

    # Add two volumes
    vol2 = vol + vol
    assert np.sum(vol2.data) == 2 * np.sum(vol.data)
    
    # Subtract and multiply
    vol3 = vol2 - 2 * vol
    assert np.all(vol3.data == 0)
    
    # Element-wise operations
    vol4 = vol * 2.5
    vol5 = vol + 10
    vol6 = vol ** 2

Converting to logical (binary) volumes
--------------------------------------

A numeric image volume can be converted to a binary image::

    # Convert to logical volume
    vol_binary = vol.as_logical()
    print(type(vol_binary))
    # <class 'neuroim.neuro_vol.LogicalNeuroVol'>
    
    print(vol_binary[0, 0, 0])
    # False

Creating volumes from arrays
----------------------------

We can create a :class:`NeuroVol` instance from an array or numeric vector. First we construct a standard numpy array::

    x = np.zeros((64, 64, 64))

Now we create a :class:`NeuroSpace` instance that describes the geometry of the image including its dimensions and voxel spacing::

    bspace = neuroim.NeuroSpace(dim=[64, 64, 64], spacing=[1, 1, 1])
    vol = neuroim.DenseNeuroVol(x, bspace)
    print(vol)

Working with existing spaces
----------------------------

We don't usually have to create :class:`NeuroSpace` objects manually, because geometric information is automatically determined from the image file header. We can copy spatial information from existing images using the :attr:`space` property::

    # Create a new volume with the same space as an existing one
    vol2 = neuroim.DenseNeuroVol((vol.data + 1) * 25, vol.space)
    print(np.max(vol2.data))
    # 25.0
    
    print(vol2.space)
    # NeuroSpace(
    #   dim     : (64, 64, 64)
    #   origin  : (0.0, 0.0, 0.0)
    #   spacing : (1.0, 1.0, 1.0)
    # )

Writing a NIFTI formatted image volume
--------------------------------------

When we're ready to write an image volume to disk, we use :func:`write_vol`::

    # Write uncompressed NIFTI
    neuroim.write_vol(vol2, "output.nii")
    
    # Write compressed NIFTI (adding .gz extension)
    neuroim.write_vol(vol2, "output.nii.gz")

Sparse volumes
--------------

For volumes with many zero values, we can use :class:`SparseNeuroVol` to save memory::

    # Create a sparse volume with only a few non-zero values
    data = np.zeros((64, 64, 64))
    data[10:20, 10:20, 10:20] = 1.0
    
    # Convert to sparse
    dense_vol = neuroim.DenseNeuroVol(data, bspace)
    sparse_vol = dense_vol.as_sparse()
    
    print(type(sparse_vol))
    # <class 'neuroim.neuro_vol.SparseNeuroVol'>
    
    # Sparse volumes behave like dense volumes
    print(sparse_vol[15, 15, 15])
    # 1.0
    
    print(sparse_vol[0, 0, 0])
    # 0.0

Volume statistics
-----------------

Common statistical operations are available as methods::

    # Create a volume with random data
    data = np.random.randn(64, 64, 64)
    vol = neuroim.DenseNeuroVol(data, bspace)
    
    # Get statistics
    print(f"Mean: {vol.vol_mean():.3f}")
    print(f"SD: {vol.vol_sd():.3f}")
    print(f"Min: {vol.min():.3f}")
    print(f"Max: {vol.max():.3f}")
    
    # Find location of minimum and maximum
    min_idx = vol.which_min()
    max_idx = vol.which_max()
    
    # Convert indices to coordinates
    min_coords = vol.space.index_to_grid(np.array([min_idx]))
    max_coords = vol.space.index_to_grid(np.array([max_idx]))
    
    print(f"Min at: {min_coords[0]}")
    print(f"Max at: {max_coords[0]}")