Quickstart Guide
================

This quickstart guide will get you up and running with neuroim in just a few minutes.

Installation
------------

Install neuroim using pip:

.. code-block:: bash

    pip install neuroim

Basic Usage
-----------

Creating and Working with 3D Volumes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import neuroim
    import numpy as np

    # Create a NeuroSpace defining the image geometry
    space = neuroim.NeuroSpace(
        dim=[64, 64, 32],           # dimensions (x, y, z)
        spacing=[2.0, 2.0, 3.0],    # voxel sizes in mm
        origin=[0, 0, 0]            # origin coordinates
    )

    # Create a volume from a numpy array
    data = np.random.randn(64, 64, 32)
    vol = neuroim.DenseNeuroVol(data, space)

    # Display information about the volume
    print(vol)

    # Basic indexing
    print(vol[32, 32, 16])  # Access a single voxel
    print(vol[30:35, 30:35, 15:20])  # Slice the volume

    # Arithmetic operations
    vol2 = vol * 2.0
    vol3 = vol + vol2
    vol_squared = vol ** 2

Reading and Writing Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Read a NIfTI file
    vol = neuroim.read_vol("path/to/structural.nii.gz")

    # Perform some operation
    smoothed = neuroim.gaussian_blur(vol, sigma=2.0)

    # Write the result
    neuroim.write_vol(smoothed, "smoothed_structural.nii.gz")

Working with 4D Time Series Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Read a 4D fMRI dataset
    vec = neuroim.read_vec("path/to/fmri.nii.gz")
    print(vec.shape)  # (x, y, z, time)

    # Extract a time series from a single voxel
    ts = vec.series(32, 32, 16)
    print(ts.shape)  # (n_timepoints,)

    # Extract time series from multiple voxels
    coords = np.array([[32, 32, 16],
                       [33, 32, 16],
                       [32, 33, 16]])
    ts_matrix = vec.series(coords)
    print(ts_matrix.shape)  # (n_timepoints, 3)

    # Get a single volume at time point 10
    vol = vec[..., 10]

Creating an ROI and Extracting Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create a spherical ROI centered at (32, 32, 16) with radius 8mm
    center = [32, 32, 16]
    roi = neuroim.spherical_roi(vol, center, radius=8.0)

    # Get ROI properties
    print(f"ROI volume: {roi.size} voxels")
    print(f"ROI center of mass: {roi.centroid}")

    # Extract time series from ROI voxels (for 4D data)
    vec = neuroim.read_vec("fmri.nii.gz")
    roi_series = vec.series_roi(roi)
    print(roi_series.shape)  # (n_timepoints, n_roi_voxels)

    # Average time series across ROI
    mean_ts = np.mean(roi_series, axis=1)

Saving Results
~~~~~~~~~~~~~~

.. code-block:: python

    # Save a 3D volume
    neuroim.write_vol(vol, "output.nii.gz")

    # Save a 4D dataset
    neuroim.write_vec(vec, "output_4d.nii.gz")

Next Steps
----------

* Learn more about :doc:`tutorials/image_volumes`
* Explore :doc:`tutorials/neuro_vectors` for 4D data
* Try :doc:`tutorials/searchlight` analysis
* Apply :doc:`tutorials/filtering` techniques
* Check out the complete :doc:`api/index`
