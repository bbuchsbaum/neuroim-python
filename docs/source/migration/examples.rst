Side-by-Side Examples
=====================

This page shows common neuroimaging tasks implemented in both R (neuroim2) and Python (neuroim).

Basic Volume Operations
-----------------------

Creating and Manipulating Volumes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**R (neuroim2):**

.. code-block:: r

   library(neuroim2)

   # Create a volume with random data
   space <- NeuroSpace(dim = c(64, 64, 32),
                      spacing = c(3, 3, 4),
                      origin = c(0, 0, 0))

   data <- array(rnorm(64 * 64 * 32), dim = c(64, 64, 32))
   vol <- NeuroVol(data, space)

   # Basic operations
   mean_vol <- mean(vol)
   sd_vol <- sd(vol)

   # Thresholding
   mask <- vol > 2

   # Arithmetic
   vol2 <- vol * 2 + 1

**Python (neuroim):**

.. code-block:: python

   import neuroim as pn
   import numpy as np

   # Create a volume with random data
   space = pn.NeuroSpace(dim=(64, 64, 32),
                         spacing=(3, 3, 4),
                         origin=(0, 0, 0))

   data = np.random.randn(64, 64, 32)
   vol = pn.NeuroVol(data, space)

   # Basic operations
   mean_vol = np.mean(vol.data)
   sd_vol = np.std(vol.data)

   # Thresholding
   mask = vol > 2

   # Arithmetic
   vol2 = vol * 2 + 1

Loading and Saving Data
-----------------------

Working with NIfTI Files
~~~~~~~~~~~~~~~~~~~~~~~~

**R (neuroim2):**

.. code-block:: r

   # Load a NIfTI file
   brain <- read_vol("anatomical.nii.gz")

   # Get information
   print(dim(brain))
   print(spacing(brain))

   # Save the result
   write_vol(brain, "output.nii.gz")

**Python (neuroim):**

.. code-block:: python

   # Load a NIfTI file
   brain = pn.read_vol("anatomical.nii.gz")

   # Get information
   print(brain.shape)
   print(brain.spacing)

   # Save the result
   pn.write_vol(brain, "output.nii.gz")

Time Series Analysis
--------------------

Extracting and Analyzing Time Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**R (neuroim2):**

.. code-block:: r

   # Load 4D fMRI data
   fmri <- read_vec("functional.nii.gz")

   # Extract time series from a voxel
   ts <- series(fmri, c(32, 32, 16))

   # Extract from multiple voxels
   coords <- rbind(c(32, 32, 16),
                  c(40, 40, 20),
                  c(25, 25, 10))
   multi_ts <- series(fmri, coords)

   # Compute correlation
   cor_ts <- cor(t(multi_ts))

**Python (neuroim):**

.. code-block:: python

   # Load 4D fMRI data
   fmri = pn.read_vec("functional.nii.gz")

   # Extract time series from a voxel (0-based)
   ts = fmri.series(31, 31, 15)

   # Extract from multiple voxels
   coords = np.array([[31, 31, 15],
                      [39, 39, 19],
                      [24, 24, 9]])
   multi_ts = fmri.series(coords)

   # Compute correlation
   cor_ts = np.corrcoef(multi_ts)

ROI Analysis
------------

Creating and Using ROIs
~~~~~~~~~~~~~~~~~~~~~~~

**R (neuroim2):**

.. code-block:: r

   # Create a spherical ROI
   center <- c(32, 32, 16)
   roi <- spherical_roi(center, radius = 10,
                       space = space(fmri))

   # Extract data from ROI
   roi_data <- series_roi(fmri, roi)

   # Average time series in ROI
   mean_ts <- colMeans(roi_data)

**Python (neuroim):**

.. code-block:: python

   # Create a spherical ROI (0-based)
   center = [31, 31, 15]
   roi = pn.spherical_roi(fmri, center, radius=10)

   # Extract data from ROI
   roi_data = fmri.series_roi(roi)

   # Average time series in ROI
   mean_ts = np.mean(roi_data, axis=1)

Searchlight Analysis
--------------------

Running Searchlight
~~~~~~~~~~~~~~~~~~~

**R (neuroim2):**

.. code-block:: r

   # Define analysis function
   analyze_sphere <- function(data) {
     if (ncol(data) < 2) return(NA)
     cor_mat <- cor(t(data))
     mean(cor_mat[upper.tri(cor_mat)])
   }

   # Create mask
   mask <- read_vol("brain_mask.nii")

   # Run searchlight
   result <- searchlight(mask,
                        radius = 5,
                        method = analyze_sphere,
                        combiner = "mean")

   # Save results
   write_vol(result, "searchlight_results.nii")

**Python (neuroim):**

.. code-block:: python

   # Define analysis function
   def analyze_sphere(data):
       if data.shape[1] < 2:
           return np.nan
       cor_mat = np.corrcoef(data.T)
       upper_tri = np.triu_indices_from(cor_mat, k=1)
       return np.mean(cor_mat[upper_tri])

   # Create mask
   mask = pn.read_vol("brain_mask.nii")

   # Run searchlight
   for roi in pn.searchlight(mask, radius=5):
       result = analyze_sphere(roi.data)

Statistical Analysis
--------------------

Clustering and Partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**R (neuroim2):**

.. code-block:: r

   # Load statistical map
   stat_map <- read_vol("tstat_map.nii")

   # Threshold and find clusters
   clusters <- conn_comp(stat_map,
                        threshold = 3.1,
                        connect = "26-connect")

   # Partition brain into regions
   brain_mask <- read_vol("brain_mask.nii")
   partitioned <- partition(stat_map,
                           k = 20,
                           mask = brain_mask)

**Python (neuroim):**

.. code-block:: python

   # Load statistical map
   stat_map = pn.read_vol("tstat_map.nii")

   # Threshold and find clusters
   clusters = pn.conn_comp(stat_map,
                          threshold=3.1,
                          connect="26-connect")

   # Partition brain into regions
   brain_mask = pn.read_vol("brain_mask.nii")
   partitioned = pn.partition(stat_map,
                             k=20,
                             mask=brain_mask)

Working with Sparse Data
------------------------

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**R (neuroim2):**

.. code-block:: r

   # Create sparse representation
   mask <- read_vol("gm_mask.nii")
   fmri <- read_vec("functional.nii.gz")

   # Convert to sparse
   sparse_fmri <- SparseNeuroVec(fmri, mask = mask)

   # Work with sparse data
   n_voxels <- sum(mask)
   print(paste("Active voxels:", n_voxels))

   # Extract data matrix
   data_matrix <- as.matrix(sparse_fmri)

**Python (neuroim):**

.. code-block:: python

   # Create sparse representation
   mask = pn.read_vol("gm_mask.nii")
   fmri = pn.read_vec("functional.nii.gz")

   # Convert to sparse using mask
   mask_vol = pn.LogicalNeuroVol(mask.data > 0, mask.space)
   sparse_fmri = fmri.as_sparse(mask_vol)

   # Work with sparse data
   n_voxels = mask.data.sum()
   print(f"Active voxels: {n_voxels}")

   # Extract data matrix
   data_matrix = sparse_fmri.data

Visualization
-------------

Plotting Brain Data
~~~~~~~~~~~~~~~~~~~

**R (neuroim2):**

.. code-block:: r

   # Orthogonal views
   ortho_plot(brain,
             coords = c(32, 32, 16),
             crosshairs = TRUE)

   # Overlay statistical map
   anat <- read_vol("anatomical.nii")
   overlay(anat, stat_map,
          zlim = c(2, 5),
          col = heat.colors(100))

**Python (neuroim):**

.. code-block:: python

   # Orthogonal views
   pn.plot_ortho(brain, coords=[31, 31, 15])

   # Overlay statistical map
   anat = pn.read_vol("anatomical.nii")
   pn.plot_overlay(anat, stat_map, vmin=2, vmax=5, cmap='hot')

Memory-Mapped Large Datasets
-----------------------------

**R (neuroim2):**

.. code-block:: r

   # Create memory-mapped 4D data
   big_data <- BigNeuroVec("large_fmri.dat",
                          dim = c(128, 128, 64, 1000),
                          type = "float")

   # Process in chunks
   chunk_size <- 100
   for (i in seq(1, 1000, by = chunk_size)) {
     end_idx <- min(i + chunk_size - 1, 1000)
     chunk <- big_data[,,,i:end_idx]
   }

**Python (neuroim):**

.. code-block:: python

   # Create memory-mapped 4D data
   big_data = pn.BigNeuroVec("large_fmri.dat",
                             shape=(128, 128, 64, 1000),
                             dtype=np.float32)

   # Process in chunks
   chunk_size = 100
   for i in range(0, 1000, chunk_size):
       end_idx = min(i + chunk_size, 1000)
       chunk = big_data[:, :, :, i:end_idx]

Key Takeaways
-------------

1. **Indexing**: Always subtract 1 from R indices when using Python
2. **Methods vs Functions**: Python uses object methods, R uses generic functions
3. **Properties**: Python accesses attributes directly (e.g., ``vol.shape``)
4. **Broadcasting**: Python requires explicit broadcasting for array operations
5. **Memory**: Both languages support memory-mapped operations for large datasets
