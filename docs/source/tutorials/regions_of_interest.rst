Regions of Interest
===================

In neuroim there is comprehensive support for creating and working with regions of interest (ROI).

Creating a spherical ROI
------------------------

To create a spherical ROI around a central point, we need an existing :class:`NeuroVol` or :class:`NeuroSpace` object.

Basic spherical ROI
~~~~~~~~~~~~~~~~~~~

To create a spherical region of interest with a 5mm radius around a central voxel at (20, 20, 20)::

    import neuroim
    import numpy as np

    # First read in an image to define the space
    vol = neuroim.read_vol("brain_image.nii.gz")

    # Create a spherical ROI centered at voxel [20, 20, 20] with 5mm radius
    sphere = neuroim.spherical_roi(vol, [20, 20, 20], radius=5, fill=100)

    print(sphere)
    # ROIVol Object
    # Properties:
    #   Dimensions:  49 x 3
    #   ROI Points:  49
    #   Value Range: [100.00, 100.00]

The ``fill`` parameter sets all values in the ROI to 100. If omitted, the ROI will contain the original values from the volume.

Creating a spherical ROI around real-world coordinates
------------------------------------------------------

To create a spherical ROI centered around a real coordinate in millimeters, we need to first convert the real-valued coordinates to voxel coordinates.

Suppose our real-world coordinate is at (-34, -28, 10) in MNI space::

    # Real-world coordinate in mm
    rpoint = np.array([-34, -28, 10])

    # Convert to voxel coordinates
    vox = vol.space.world_to_grid(rpoint.reshape(1, -1))[0]

    # Create spherical ROI with 10mm radius
    sphere = neuroim.spherical_roi(vol, vox, radius=10, fill=1)
    print(f"ROI contains {len(sphere)} voxels")

Now we can verify the center by converting back to real-world coordinates::

    # Get all coordinates in the ROI
    roi_world_coords = sphere.get_coords(real=True)

    # Calculate center of mass
    center_of_mass = np.mean(roi_world_coords, axis=0)
    print(f"Center of mass: {center_of_mass}")

Converting an ROI to a SparseNeuroVol
-------------------------------------

We may want to convert a region of interest to a NeuroVol instance without storing every voxel value::

    # Create a spherical ROI
    sphere = neuroim.spherical_roi(vol, [50, 10, 10], radius=10, fill=1)
    print(sphere)

    # Convert to SparseNeuroVol
    sparsevol = sphere.as_sparse()
    print(f"Sum of ROI values: {np.sum(sparsevol.data)}")
    print(f"Sum matches: {np.sum(sparsevol.data) == np.sum(sphere.data)}")
    print(f"Dimensions match: {sparsevol.shape == vol.shape}")

Creating other ROI shapes
-------------------------

Square/rectangular ROIs
~~~~~~~~~~~~~~~~~~~~~~~

Create a square ROI in a specific plane::

    # Create a 7x7 square ROI in the z=10 plane
    square = neuroim.square_roi(vol, centroid=[30, 30, 10],
                                  surround=3, fixdim=2, fill=1)
    print(f"Square ROI has {len(square)} voxels")

Cuboid ROIs
~~~~~~~~~~~

Create a 3D box ROI::

    # Create a cuboid with equal dimensions
    cube = neuroim.cuboid_roi(vol, centroid=[30, 30, 15],
                               surround=5, fill=1)

    # Create an asymmetric cuboid
    cuboid = neuroim.cuboid_roi(vol, centroid=[30, 30, 15],
                                 surround=[3, 4, 5], fill=1)
    print(f"Cuboid ROI has {len(cuboid)} voxels")

Working with multiple ROIs
--------------------------

Create multiple spherical ROIs efficiently::

    # Define multiple ROI centers
    centers = np.array([[20, 20, 10],
                       [40, 40, 15],
                       [30, 50, 12]])

    # Create ROIs with different values
    roi_list = neuroim.spherical_roi_set(vol, centers, radius=6,
                                          fill=[100, 200, 300])

    print(f"Created {len(roi_list)} ROIs")
    for i, roi in enumerate(roi_list):
        print(f"ROI {i}: {len(roi)} voxels, value={roi.data[0]}")

Simple searchlight analysis
---------------------------

The "searchlight" approach creates ROIs around each voxel for local analyses::

    # Generate searchlight ROIs around each voxel
    from neuroim import searchlight

    # Create searchlights with 8mm radius
    slist = searchlight(vol, radius=8, mask=vol.data > 0)

    # Compute mean value in each searchlight
    means = []
    for roi in slist:
        # Get values from volume at ROI coordinates
        values = vol[roi.coords[:, 0], roi.coords[:, 1], roi.coords[:, 2]]
        means.append(np.mean(values))

    print(f"Computed means for {len(means)} searchlights")

Random searchlight
~~~~~~~~~~~~~~~~~~

A randomized searchlight samples voxels without replacement::

    from neuroim import random_searchlight

    # Create random searchlights
    rois = random_searchlight(vol, radius=8, mask=vol.data > 0)

    # Process each ROI
    results = []
    for roi in rois:
        values = vol[roi.coords[:, 0], roi.coords[:, 1], roi.coords[:, 2]]
        results.append(np.mean(values))

Clustered searchlight
~~~~~~~~~~~~~~~~~~~~~

Use a parcellation to define ROIs::

    # First create a clustering
    mask_indices = np.where(vol.data > 0)
    coords = np.column_stack(mask_indices)

    # Simple k-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=50, random_state=42)
    labels = kmeans.fit_predict(coords)

    # Create ClusteredNeuroVol
    cluster_data = np.zeros_like(vol.data)
    cluster_data[mask_indices] = labels + 1  # Labels start at 1
    kvol = neuroim.ClusteredNeuroVol(cluster_data, vol.space)

    # Create ROIs from clusters
    from neuroim import clustered_searchlight
    cluster_rois = clustered_searchlight(vol, kvol)

    # Analyze each cluster
    cluster_means = []
    for roi in cluster_rois:
        values = vol[roi.coords[:, 0], roi.coords[:, 1], roi.coords[:, 2]]
        cluster_means.append(np.mean(values))

ROI-based time series extraction
--------------------------------

Extract time series from 4D data using ROIs::

    # Load 4D data
    vec = neuroim.read_vec("functional_data.nii")

    # Create ROI
    roi = neuroim.spherical_roi(vec, [32, 32, 16], radius=10)

    # Extract time series for all voxels in ROI
    roi_timeseries = vec.series_roi(roi)
    print(f"Time series shape: {roi_timeseries.shape}")
    # (n_timepoints, n_voxels_in_roi)

    # Compute mean time series
    mean_ts = np.mean(roi_timeseries, axis=1)

    # Plot
    import matplotlib.pyplot as plt
    plt.plot(mean_ts)
    plt.xlabel('Time point')
    plt.ylabel('Mean signal')
    plt.title('Mean ROI time series')

Saving and loading ROIs
-----------------------

ROIs can be saved as sparse volumes::

    # Create ROI
    roi = neuroim.spherical_roi(vol, [30, 30, 15], radius=8)

    # Convert to sparse volume and save
    sparse_roi = roi.as_sparse()
    neuroim.write_vol(sparse_roi, "my_roi.nii.gz")

    # Load it back
    loaded_roi = neuroim.read_vol("my_roi.nii.gz")

    # Convert back to ROI if needed
    roi_coords = np.column_stack(np.where(loaded_roi.data > 0))
    roi_values = loaded_roi.data[loaded_roi.data > 0]
    reconstructed_roi = neuroim.ROIVol(roi_values, loaded_roi.space, roi_coords)