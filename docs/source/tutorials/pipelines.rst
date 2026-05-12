Pipelines and Functional Operations
====================================

The neuroim package provides a set of functions that allow you to split image data in various ways for processing. By breaking a dataset into pieces, operations can be parallelized and complex analyses can be composed from simple building blocks.

Splitting an image into connected components
--------------------------------------------

First, let's find connected components in a thresholded volume::

    import neuroim
    import numpy as np
    import matplotlib.pyplot as plt

    # Load an example volume
    vol = neuroim.read_vol("brain_mask.nii")

    # Create random values in the mask
    mask_idx = np.where(vol.data > 0)
    vol2 = vol.copy()
    vol2.data[mask_idx] = np.random.rand(len(mask_idx[0]))

    # Find connected components with threshold of 0.8
    comp = neuroim.conn_comp(vol2, threshold=0.8)

    # comp is a ConnCompResult with:
    # - comp.index: ClusteredNeuroVol with component labels
    # - comp.size: component sizes
    # - comp.maxima: local maxima information

    print(f"Found {len(comp.size)} connected components")
    print(f"Largest component has {max(comp.size)} voxels")

    # Visualize the components
    # (plotting code would go here)

Computing statistics on connected components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`split_clusters` to compute statistics for each component::

    # Calculate mean value in each connected component
    from neuroim import split_clusters

    cluster_rois = split_clusters(vol2, comp.index)

    # Compute mean for each cluster
    means = [np.mean(roi.data) for roi in cluster_rois]

    print(f"Component means: {means[:5]}...")  # Show first 5

Searchlight analysis
--------------------

Local standard deviation
~~~~~~~~~~~~~~~~~~~~~~~~

Compute the local standard deviation within a spherical neighborhood::

    # Create searchlights with 5mm radius
    from neuroim import searchlight

    # Get all searchlight ROIs
    rois = searchlight(vol, radius=5, mask=vol.data > 0)

    # Compute standard deviation in each
    sd_values = []
    indices = []

    for roi in rois:
        # Extract values from the volume
        values = vol2.data[roi.coords[:, 0], roi.coords[:, 1], roi.coords[:, 2]]
        sd_values.append(np.std(values))
        # Get center voxel index
        indices.append(roi.indices()[roi.center_index] if hasattr(roi, 'center_index') else roi.indices()[0])

    # Create output volume
    sd_vol = neuroim.SparseNeuroVol(
        data=np.array(sd_values),
        space=vol.space,
        indices=np.array(indices)
    )

    print(f"Computed local SD for {len(sd_values)} voxels")

K-nearest neighbors smoothing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace each voxel with the average of its k-nearest neighbors::

    k = 12
    smoothed_values = []
    indices = []

    for roi in searchlight(vol2, radius=6, mask=vol2.data > 0):
        # Get values in the ROI
        roi_coords = roi.coords
        roi_values = vol2.data[roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]]

        if hasattr(roi, 'center_index'):
            center_val = roi_values[roi.center_index]
            # Find k nearest neighbors by value
            distances = np.abs(roi_values - center_val)
            k_nearest_idx = np.argsort(distances)[1:k+1]  # Exclude center itself
            smoothed_values.append(np.mean(roi_values[k_nearest_idx]))
        else:
            # If no center index, just use mean
            smoothed_values.append(np.mean(roi_values))

        indices.append(roi.indices()[0])

    # Create smoothed volume
    smoothed_vol = neuroim.SparseNeuroVol(
        data=np.array(smoothed_values),
        space=vol2.space,
        indices=np.array(indices)
    )

Using searchlight coordinates only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For efficiency, sometimes we only need coordinates without creating full ROI objects::

    from neuroim import searchlight_coords

    # Get coordinates for each searchlight
    avg_values = []
    center_indices = []

    for coords in searchlight_coords(vol, radius=12, mask=vol.data > 0):
        # coords is an Nx3 array of voxel coordinates
        vals = vol.data[coords[:, 0], coords[:, 1], coords[:, 2]]
        # Average only non-zero values
        nonzero_vals = vals[vals != 0]
        if len(nonzero_vals) > 0:
            avg_values.append(np.mean(nonzero_vals))
            # Assume first coordinate is center
            center_indices.append(vol.space.grid_to_index(coords[0:1])[0])

    # Create averaged volume
    avg_vol = neuroim.SparseNeuroVol(
        data=np.array(avg_values),
        space=vol.space,
        indices=np.array(center_indices)
    )

Processing slices
-----------------

Apply operations to each 2D slice of a volume::

    # Process each axial slice
    slice_means = []

    for z in range(vol.shape[2]):
        slice_data = vol.data[:, :, z]
        slice_means.append(np.mean(slice_data))

    # Plot slice means
    plt.plot(slice_means)
    plt.xlabel('Slice number')
    plt.ylabel('Mean intensity')
    plt.title('Mean intensity by slice')
    plt.show()

Processing volumes in a 4D dataset
----------------------------------

Apply operations to each volume in a NeuroVec::

    # Create a 4D dataset by concatenating volumes
    vec = neuroim.neurovec([vol, vol, vol, vol, vol])
    print(f"4D data shape: {vec.shape}")

    # Calculate statistics for each volume
    volumes = vec.vols()  # Get list of all volumes

    mean_vec = [np.mean(v.data) for v in volumes]
    sd_vec = [np.std(v.data) for v in volumes]

    print(f"Number of volumes: {len(mean_vec)}")
    print(f"Mean values: {mean_vec}")
    print(f"SD values: {sd_vec}")

Processing voxel time series
----------------------------

Apply operations to each voxel's time series::

    # Calculate mean across time for each voxel
    vec = neuroim.neurovec(np.random.randn(10, 10, 10, 50))

    # Method 1: Using numpy operations
    mean_vol_data = np.mean(vec.data, axis=3)
    mean_vol = neuroim.DenseNeuroVol(mean_vol_data,
                                       neuroim.NeuroSpace(dim=mean_vol_data.shape))

    # Method 2: Iterating over voxels (more flexible but slower)
    mean_values = []
    for i in range(vec.shape[0]):
        for j in range(vec.shape[1]):
            for k in range(vec.shape[2]):
                ts = vec.series(i, j, k)
                mean_values.append(np.mean(ts))

    # Reshape to volume
    mean_vol2_data = np.array(mean_values).reshape(vec.shape[:3], order='F')
    mean_vol2 = neuroim.DenseNeuroVol(mean_vol2_data, mean_vol.space)

Parallel processing
-------------------

Many operations can be parallelized using Python's multiprocessing or joblib::

    from joblib import Parallel, delayed

    def process_roi(roi, volume):
        """Process a single ROI"""
        values = volume.data[roi.coords[:, 0], roi.coords[:, 1], roi.coords[:, 2]]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'max': np.max(values),
            'size': len(values)
        }

    # Create ROIs
    rois = list(searchlight(vol2, radius=8, mask=vol2.data > 0))

    # Process in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_roi)(roi, vol2) for roi in rois
    )

    # Extract results
    roi_means = [r['mean'] for r in results]
    roi_stds = [r['std'] for r in results]

    print(f"Processed {len(results)} ROIs in parallel")

Building analysis pipelines
---------------------------

Combine operations into reusable pipelines::

    def denoise_pipeline(vol, smooth_radius=5, threshold=0.1):
        """
        Example denoising pipeline:
        1. Identify noise components
        2. Smooth within components
        3. Threshold final result
        """
        # Find connected components above threshold
        comp = neuroim.conn_comp(vol, threshold=threshold)

        # Process each component
        processed_rois = []
        for roi in split_clusters(vol, comp.index):
            if len(roi) < 10:  # Skip small components
                continue

            # Smooth within component
            roi_coords = roi.coords
            values = vol.data[roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]]

            # Simple Gaussian smoothing would go here
            smoothed_values = values * 0.8  # Placeholder

            # Update ROI with smoothed values
            roi.data = smoothed_values
            processed_rois.append(roi)

        # Combine back into volume
        result = neuroim.DenseNeuroVol(np.zeros_like(vol.data), vol.space)
        for roi in processed_rois:
            result.data[roi.coords[:, 0], roi.coords[:, 1], roi.coords[:, 2]] = roi.data

        return result

    # Apply pipeline
    denoised = denoise_pipeline(vol2, smooth_radius=5, threshold=0.2)

Custom iterators
----------------

Create custom iterators for specific processing patterns::

    def sliding_window_iterator(vec, window_size=10, step=5):
        """
        Iterate over overlapping time windows
        """
        n_timepoints = vec.shape[3]
        for start in range(0, n_timepoints - window_size + 1, step):
            end = start + window_size
            yield vec.sub_vector(slice(start, end))

    # Use the iterator
    vec = neuroim.neurovec(np.random.randn(10, 10, 10, 100))

    window_means = []
    for window_vec in sliding_window_iterator(vec, window_size=20, step=10):
        # Calculate mean volume for this time window
        mean_vol = np.mean(window_vec.data, axis=3)
        window_means.append(mean_vol)

    print(f"Computed {len(window_means)} windowed mean volumes")