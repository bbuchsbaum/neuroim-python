Searchlight Analysis
====================

Searchlight analysis is a technique for performing local multivariate pattern analysis across the brain. Instead of analyzing each voxel independently, searchlight creates small neighborhoods around each voxel and applies a machine learning or statistical analysis within each neighborhood.

Basic Searchlight Iterator
---------------------------

The simplest way to create a searchlight is with :func:`searchlight_iterator`:

.. code-block:: python

    import neuroimpy
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score

    # Load fMRI data
    vec = neuroimpy.read_vec("fmri_data.nii.gz")
    mask = neuroimpy.read_vol("brain_mask.nii.gz")

    # Create searchlight iterator with 8mm radius
    searchlight = neuroimpy.searchlight_iterator(
        vec,
        mask=mask,
        radius=8.0
    )

    # Iterate over searchlights
    for sl in searchlight:
        center = sl.center  # Center voxel coordinates
        indices = sl.indices  # Voxel indices in this searchlight
        data = vec.series(indices)  # Extract time series
        # data.shape = (n_timepoints, n_voxels_in_searchlight)

Applying a Function to Each Searchlight
----------------------------------------

Process each searchlight with a custom function:

.. code-block:: python

    # Define an analysis function
    def classify(data, labels):
        """
        Perform cross-validated classification.

        Parameters
        ----------
        data : ndarray, shape (n_samples, n_features)
            Time series data from searchlight voxels
        labels : ndarray, shape (n_samples,)
            Class labels

        Returns
        -------
        accuracy : float
            Cross-validated classification accuracy
        """
        clf = SVC(kernel='linear')
        scores = cross_val_score(clf, data, labels, cv=5)
        return np.mean(scores)

    # Create labels (e.g., experimental conditions)
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1] * 10)  # Example labels

    # Run searchlight analysis
    results = []
    centers = []

    searchlight = neuroimpy.searchlight_iterator(vec, mask=mask, radius=8.0)

    for sl in searchlight:
        data = vec.series(sl.indices).T  # Transpose to (samples, features)
        accuracy = classify(data, labels)
        results.append(accuracy)
        centers.append(sl.center)

    # Create a volume with the results
    result_vol = neuroimpy.DenseNeuroVol(
        np.zeros(vec.shape[:3]),
        vec.space
    )

    for center, accuracy in zip(centers, results):
        result_vol[center[0], center[1], center[2]] = accuracy

    # Save the searchlight map
    neuroimpy.write_vol(result_vol, "searchlight_accuracy.nii.gz")

Random Searchlight Sampling
----------------------------

For large datasets, sample random searchlights instead of exhaustively searching:

.. code-block:: python

    # Create random searchlight with 1000 samples
    random_searchlight = neuroimpy.random_searchlight(
        vec,
        mask=mask,
        radius=8.0,
        n_samples=1000,
        seed=42
    )

    results = []
    centers = []

    for sl in random_searchlight:
        data = vec.series(sl.indices).T
        accuracy = classify(data, labels)
        results.append(accuracy)
        centers.append(sl.center)

    print(f"Sampled {len(results)} searchlights randomly")

Bootstrap Searchlight for Statistical Testing
----------------------------------------------

Use bootstrap resampling to estimate confidence intervals:

.. code-block:: python

    # Bootstrap searchlight with 100 iterations
    bootstrap_searchlight = neuroimpy.bootstrap_searchlight(
        vec,
        mask=mask,
        radius=8.0,
        n_bootstraps=100,
        seed=42
    )

    # Store all bootstrap results
    all_results = []

    for sl in bootstrap_searchlight:
        data = vec.series(sl.indices).T

        # Resample data with replacement
        boot_idx = np.random.choice(len(data), size=len(data), replace=True)
        boot_data = data[boot_idx]
        boot_labels = labels[boot_idx]

        accuracy = classify(boot_data, boot_labels)
        all_results.append((sl.center, accuracy))

    # Compute statistics across bootstraps
    center_accuracies = {}
    for center, acc in all_results:
        key = tuple(center)
        if key not in center_accuracies:
            center_accuracies[key] = []
        center_accuracies[key].append(acc)

    # Create mean and CI volumes
    mean_vol = neuroimpy.DenseNeuroVol(np.zeros(vec.shape[:3]), vec.space)
    ci_lower_vol = neuroimpy.DenseNeuroVol(np.zeros(vec.shape[:3]), vec.space)
    ci_upper_vol = neuroimpy.DenseNeuroVol(np.zeros(vec.shape[:3]), vec.space)

    for center, accs in center_accuracies.items():
        mean_acc = np.mean(accs)
        ci_lower = np.percentile(accs, 2.5)
        ci_upper = np.percentile(accs, 97.5)

        mean_vol[center] = mean_acc
        ci_lower_vol[center] = ci_lower
        ci_upper_vol[center] = ci_upper

    neuroimpy.write_vol(mean_vol, "searchlight_mean.nii.gz")
    neuroimpy.write_vol(ci_lower_vol, "searchlight_ci_lower.nii.gz")
    neuroimpy.write_vol(ci_upper_vol, "searchlight_ci_upper.nii.gz")

Clustered Searchlight with Parcellations
-----------------------------------------

Constrain searchlights to anatomical or functional parcellations:

.. code-block:: python

    # Load a parcellation atlas
    atlas = neuroimpy.read_vol("atlas_parcellation.nii.gz")

    # Create clustered searchlight
    clustered_searchlight = neuroimpy.clustered_searchlight(
        vec,
        atlas=atlas,
        mask=mask,
        radius=8.0
    )

    results = []
    centers = []

    for sl in clustered_searchlight:
        # Searchlight voxels are constrained to the same parcel
        data = vec.series(sl.indices).T
        accuracy = classify(data, labels)
        results.append(accuracy)
        centers.append(sl.center)

    print(f"Analyzed {len(results)} searchlights constrained to parcels")

Advanced: Custom Searchlight Shapes
------------------------------------

Create searchlights with custom shapes:

.. code-block:: python

    # Create cuboid searchlights instead of spherical
    for x in range(10, 50, 5):
        for y in range(10, 50, 5):
            for z in range(5, 25, 3):
                if not mask[x, y, z]:
                    continue

                # Define a cuboid ROI
                roi = neuroimpy.cuboid_roi(
                    vec,
                    center=[x, y, z],
                    extents=[6, 6, 4]  # Half-widths in each dimension
                )

                data = vec.series(roi.indices).T
                accuracy = classify(data, labels)
                # Store result...

Performance Tips
----------------

For large-scale searchlight analyses:

1. **Use sparse data structures** for masked data to reduce memory usage
2. **Process in chunks** when results don't fit in memory
3. **Parallelize** across searchlights using multiprocessing
4. **Use random sampling** for initial exploratory analyses

.. code-block:: python

    from concurrent.futures import ProcessPoolExecutor

    def process_searchlight(sl_data):
        """Helper function for parallel processing."""
        indices, center = sl_data
        data = vec.series(indices).T
        return center, classify(data, labels)

    # Prepare searchlight data
    searchlight = neuroimpy.searchlight_iterator(vec, mask=mask, radius=8.0)
    sl_data_list = [(sl.indices, sl.center) for sl in searchlight]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_searchlight, sl_data_list))

    print(f"Processed {len(results)} searchlights in parallel")
