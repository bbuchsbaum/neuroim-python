Migrating from R neuroim2 to Python neuroim
==============================================

This guide helps R users transition to the Python neuroim package, highlighting key differences and providing direct mappings.

Installation
------------

**R (neuroim2):**

.. code-block:: r

   install.packages("devtools")
   devtools::install_github("bbuchsbaum/neuroim2")
   library(neuroim2)

**Python (neuroim):**

.. code-block:: python

   pip install neuroim
   import neuroim as pn

Key Differences
---------------

Indexing
~~~~~~~~

The most critical difference is 0-based vs 1-based indexing:

**R (1-based):**

.. code-block:: r

   # Access voxel at position (10, 20, 30)
   vol[10, 20, 30]
   
   # First slice
   vol[,,1]

**Python (0-based):**

.. code-block:: python

   # Access voxel at position (10, 20, 30) in R = (9, 19, 29) in Python
   vol[9, 19, 29]
   
   # First slice
   vol[:, :, 0]

NA/NaN Handling
~~~~~~~~~~~~~~~

**R:**

.. code-block:: r

   # R uses NA for missing values
   vol[is.na(vol)] <- 0

**Python:**

.. code-block:: python

   # Python uses numpy.nan
   import numpy as np
   vol.data[np.isnan(vol.data)] = 0

Method Syntax
~~~~~~~~~~~~~

**R (S4 methods):**

.. code-block:: r

   space(vol)
   dim(vol)
   series(vec, coords)

**Python (object methods):**

.. code-block:: python

   vol.space
   vol.shape  # or vol.dim for exact R compatibility
   vec.series(coords)

Core Classes Mapping
--------------------

NeuroVol
~~~~~~~~

**R:**

.. code-block:: r

   # Create a NeuroVol
   vol <- NeuroVol(data = array(rnorm(64*64*32), dim = c(64, 64, 32)),
                   space = NeuroSpace(dim = c(64, 64, 32)))
   
   # Access data
   vol[10, 20, 15]
   
   # Get dimensions
   dim(vol)

**Python:**

.. code-block:: python

   # Create a NeuroVol
   import numpy as np
   vol = pn.NeuroVol(data=np.random.randn(64, 64, 32),
                     space=pn.NeuroSpace(dim=(64, 64, 32)))
   
   # Access data (note: 0-based indexing)
   vol[9, 19, 14]
   
   # Get dimensions
   vol.shape

NeuroVec (4D data)
~~~~~~~~~~~~~~~~~~

**R:**

.. code-block:: r

   # Create a 4D NeuroVec
   vec <- NeuroVec(data = array(rnorm(64*64*32*100), dim = c(64, 64, 32, 100)),
                   space = NeuroSpace(dim = c(64, 64, 32, 100)))
   
   # Extract time series
   ts <- series(vec, c(10, 20, 15))

**Python:**

.. code-block:: python

   # Create a 4D NeuroVec
   vec = pn.NeuroVec(data=np.random.randn(64, 64, 32, 100),
                     space=pn.NeuroSpace(dim=(64, 64, 32, 100)))
   
   # Extract time series (note: 0-based indexing)
   ts = vec.series_3d(9, 19, 14)

SparseNeuroVec
~~~~~~~~~~~~~~

**R:**

.. code-block:: r

   # Create mask
   mask <- LogicalNeuroVol(data = array(runif(64*64*32) > 0.5, 
                                       dim = c(64, 64, 32)),
                          space = NeuroSpace(dim = c(64, 64, 32)))
   
   # Create sparse vector
   sparse_vec <- SparseNeuroVec(data = matrix(rnorm(sum(mask) * 100), 
                                             nrow = sum(mask), 
                                             ncol = 100),
                               mask = mask)

**Python:**

.. code-block:: python

   # Create mask
   mask = pn.LogicalNeuroVol(data=(np.random.rand(64, 64, 32) > 0.5),
                             space=pn.NeuroSpace(dim=(64, 64, 32)))
   
   # Create sparse vector
   sparse_vec = pn.SparseNeuroVec(data=np.random.randn(mask.data.sum(), 100),
                                  mask=mask,
                                  space=pn.NeuroSpace(dim=(64, 64, 32, 100)))

I/O Operations
--------------

Reading Files
~~~~~~~~~~~~~

**R:**

.. code-block:: r

   # Read NIfTI file
   vol <- read_vol("brain.nii")
   
   # Read 4D file
   vec <- read_vec("fmri_data.nii")

**Python:**

.. code-block:: python

   # Read NIfTI file
   vol = pn.read_vol("brain.nii")
   
   # Read 4D file
   vec = pn.read_vec("fmri_data.nii")

Writing Files
~~~~~~~~~~~~~

**R:**

.. code-block:: r

   # Write volume
   write_vol(vol, "output.nii")
   
   # Write 4D data
   write_vec(vec, "output_4d.nii")

**Python:**

.. code-block:: python

   # Write volume
   pn.write_vol(vol, "output.nii")
   
   # Write 4D data
   pn.write_vec(vec, "output_4d.nii")

ROI Operations
--------------

Creating ROIs
~~~~~~~~~~~~~

**R:**

.. code-block:: r

   # Spherical ROI
   roi <- spherical_roi(c(32, 32, 16), radius = 5, 
                       space = space(vol))
   
   # Cubic ROI
   roi <- cubic_roi(c(32, 32, 16), surround = 3,
                   space = space(vol))

**Python:**

.. code-block:: python

   # Spherical ROI (note: 0-based center)
   roi = pn.spherical_roi([31, 31, 15], radius=5, 
                          space=vol.space)
   
   # Cubic ROI
   roi = pn.cubic_roi([31, 31, 15], surround=3,
                      space=vol.space)

Searchlight Analysis
~~~~~~~~~~~~~~~~~~~~

**R:**

.. code-block:: r

   # Define processing function
   process_fun <- function(x) {
     cor(x[,1], x[,2])
   }
   
   # Run searchlight
   result <- searchlight(mask = mask,
                        radius = 5,
                        method = process_fun,
                        combiner = "list")

**Python:**

.. code-block:: python

   # Define processing function
   def process_fun(x):
       return np.corrcoef(x[:, 0], x[:, 1])[0, 1]
   
   # Run searchlight
   result = pn.searchlight(mask=mask,
                          radius=5,
                          method=process_fun,
                          combiner="list")

Statistical Operations
----------------------

Partition & Clustering
~~~~~~~~~~~~~~~~~~~~~~

**R:**

.. code-block:: r

   # Partition volume into k clusters
   clustered <- partition(vol, k = 10)
   
   # Split by clusters
   cluster_list <- split_clusters(vol, clustered)

**Python:**

.. code-block:: python

   # Partition volume into k clusters
   clustered = pn.partition(vol, k=10)
   
   # Split by clusters
   cluster_list = pn.split_clusters(vol, clustered)

Connected Components
~~~~~~~~~~~~~~~~~~~~

**R:**

.. code-block:: r

   # Find connected components
   cc_result <- conn_comp(vol, threshold = 2.5,
                         connect = "26-connect")

**Python:**

.. code-block:: python

   # Find connected components
   cc_result = pn.conn_comp(vol, threshold=2.5,
                           connect="26-connect")

Common Gotchas
--------------

1. **Indexing Errors**: Always remember Python uses 0-based indexing
2. **Dimension Ordering**: Both use (X, Y, Z, Time) but be careful with nibabel which uses RAS+ ordering
3. **Memory Layout**: R uses column-major (Fortran) order, Python uses row-major (C) order by default
4. **Method vs Function**: Python uses object methods where R uses generic functions

Features Not Needed in Python
-----------------------------

Several R features are not implemented in Python due to language differences:

- **IndexLookupVol**: Python's numpy provides efficient indexing natively
- **linear_access**: Use numpy's `.ravel()` and `.flat` instead
- **ArrayLike interface**: Python's duck typing makes this unnecessary

Performance Tips
----------------

1. **Use NumPy operations** instead of loops whenever possible
2. **Memory-mapped files** work similarly in both languages
3. **Parallel processing**: Use `joblib` in Python instead of R's `parallel` package

Getting Help
------------

- **API Reference**: See :doc:`/api/index` for detailed function documentation
- **Examples**: Check :doc:`examples` for side-by-side R/Python comparisons
- **Issues**: Report bugs at https://github.com/bbuchsbaum/neuroim-python/issues