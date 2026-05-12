API Function Mappings
=====================

This page provides a comprehensive mapping between R neuroim2 functions and their Python neuroim equivalents.

Core Classes
------------

.. list-table:: Class Mappings
   :widths: 40 40 20
   :header-rows: 1

   * - R (neuroim2)
     - Python (neuroim)
     - Notes
   * - ``NeuroSpace()``
     - ``pn.NeuroSpace()``
     - Direct mapping
   * - ``NeuroVol()``
     - ``pn.NeuroVol()``
     - Direct mapping
   * - ``DenseNeuroVol()``
     - ``pn.DenseNeuroVol()``
     - Direct mapping
   * - ``SparseNeuroVol()``
     - ``pn.SparseNeuroVol()``
     - Direct mapping
   * - ``LogicalNeuroVol()``
     - ``pn.LogicalNeuroVol()``
     - Direct mapping
   * - ``ClusteredNeuroVol()``
     - ``pn.ClusteredNeuroVol()``
     - Direct mapping
   * - ``NeuroVec()``
     - ``pn.NeuroVec()``
     - Direct mapping
   * - ``DenseNeuroVec()``
     - ``pn.DenseNeuroVec()``
     - Direct mapping
   * - ``SparseNeuroVec()``
     - ``pn.SparseNeuroVec()``
     - Direct mapping
   * - ``BigNeuroVec()``
     - ``pn.BigNeuroVec()``
     - Direct mapping
   * - ``MappedNeuroVec()``
     - ``pn.MappedNeuroVec()``
     - Direct mapping
   * - ``NeuroSlice()``
     - ``pn.NeuroSlice()``
     - Direct mapping
   * - ``IndexLookupVol()``
     - Not needed
     - Use numpy indexing

I/O Functions
-------------

.. list-table:: I/O Function Mappings
   :widths: 40 40 20
   :header-rows: 1

   * - R (neuroim2)
     - Python (neuroim)
     - Notes
   * - ``read_vol(filename)``
     - ``pn.read_vol(filename)``
     - Direct mapping
   * - ``write_vol(vol, filename)``
     - ``pn.write_vol(vol, filename)``
     - Direct mapping
   * - ``read_vec(filename)``
     - ``pn.read_vec(filename)``
     - Direct mapping
   * - ``write_vec(vec, filename)``
     - ``pn.write_vec(vec, filename)``
     - Direct mapping
   * - ``read_header(filename)``
     - ``pn.read_header(filename)``
     - Direct mapping

ROI Functions
-------------

.. list-table:: ROI Function Mappings
   :widths: 40 40 20
   :header-rows: 1

   * - R (neuroim2)
     - Python (neuroim)
     - Notes
   * - ``spherical_roi(center, radius, space)``
     - ``pn.spherical_roi(center, radius, space)``
     - 0-based center
   * - ``cubic_roi(center, surround, space)``
     - ``pn.cubic_roi(center, surround, space)``
     - 0-based center
   * - ``regionSphere(centroid, radius, voxels)``
     - ``pn.region_sphere(centroid, radius, voxels)``
     - 0-based indices
   * - ``regionCube(centroid, surround, voxels)``
     - ``pn.region_cube(centroid, surround, voxels)``
     - 0-based indices

Searchlight Functions
---------------------

.. list-table:: Searchlight Function Mappings
   :widths: 40 40 20
   :header-rows: 1

   * - R (neuroim2)
     - Python (neuroim)
     - Notes
   * - ``searchlight(mask, radius, method)``
     - ``pn.searchlight(mask, radius, method)``
     - Direct mapping
   * - ``random_searchlight(...)``
     - ``pn.random_searchlight(...)``
     - Direct mapping
   * - ``clustered_searchlight(...)``
     - ``pn.clustered_searchlight(...)``
     - Direct mapping
   * - ``bootstrap_searchlight(...)``
     - ``pn.bootstrap_searchlight(...)``
     - Direct mapping

Statistical Functions
---------------------

.. list-table:: Statistical Function Mappings
   :widths: 40 40 20
   :header-rows: 1

   * - R (neuroim2)
     - Python (neuroim)
     - Notes
   * - ``partition(x, k)``
     - ``pn.partition(x, k)``
     - Direct mapping
   * - ``split_clusters(x, clusters)``
     - ``pn.split_clusters(x, clusters)``
     - Direct mapping
   * - ``split_blocks(x, indices, ids)``
     - ``pn.split_blocks(x, indices, ids)``
     - Direct mapping
   * - ``conn_comp(x, threshold)``
     - ``pn.conn_comp(x, threshold)``
     - Direct mapping

Data Access Methods
-------------------

.. list-table:: Method Mappings
   :widths: 40 40 20
   :header-rows: 1

   * - R (neuroim2)
     - Python (neuroim)
     - Notes
   * - ``dim(x)``
     - ``x.shape`` or ``x.dim``
     - Property access
   * - ``space(x)``
     - ``x.space``
     - Property access
   * - ``spacing(x)``
     - ``x.spacing``
     - Property access
   * - ``origin(x)``
     - ``x.origin``
     - Property access
   * - ``trans(x)``
     - ``x.trans``
     - Property access
   * - ``inverse_trans(x)``
     - ``x.inverse_trans``
     - Property access
   * - ``coords(x)``
     - ``x.coords()``
     - Method call
   * - ``series(vec, i)``
     - ``vec.series(i)``
     - 0-based indices
   * - ``values(x)``
     - ``x.data``
     - Direct data access
   * - ``mask(x)``
     - ``x.mask``
     - Property access

Arithmetic Operations
---------------------

Both R and Python support standard arithmetic operations on volumes and vectors:

.. list-table:: Arithmetic Operations
   :widths: 40 40 20
   :header-rows: 1

   * - Operation
     - R (neuroim2)
     - Python (neuroim)
   * - Addition
     - ``vol1 + vol2``
     - ``vol1 + vol2``
   * - Subtraction
     - ``vol1 - vol2``
     - ``vol1 - vol2``
   * - Multiplication
     - ``vol1 * 2``
     - ``vol1 * 2``
   * - Division
     - ``vol1 / 2``
     - ``vol1 / 2``

Indexing Operations
-------------------

.. list-table:: Indexing Mappings
   :widths: 40 40 20
   :header-rows: 1

   * - R (neuroim2)
     - Python (neuroim)
     - Notes
   * - ``vol[i, j, k]``
     - ``vol[i-1, j-1, k-1]``
     - 0-based indexing
   * - ``vec[i, j, k, t]``
     - ``vec[i-1, j-1, k-1, t-1]``
     - 0-based indexing
   * - ``vol[, , k]``
     - ``vol[:, :, k-1]``
     - Slice selection
   * - ``vec[, , , t]``
     - ``vec[:, :, :, t-1]``
     - Volume at time t

Utility Functions
-----------------

.. list-table:: Utility Function Mappings
   :widths: 40 40 20
   :header-rows: 1

   * - R (neuroim2)
     - Python (neuroim)
     - Notes
   * - ``grid_to_index(space, coords)``
     - ``space.grid_to_index(coords)``
     - Method of space
   * - ``index_to_grid(space, idx)``
     - ``space.index_to_grid(idx)``
     - Method of space
   * - ``index_to_coord(space, idx)``
     - ``space.index_to_coord(idx)``
     - Method of space
   * - ``coord_to_index(space, coords)``
     - ``space.coord_to_index(coords)``
     - Method of space
   * - ``coord_to_grid(space, coords)``
     - ``space.coord_to_grid(coords)``
     - Method of space

Memory-Mapped Operations
------------------------

.. list-table:: Memory-Mapped Mappings
   :widths: 40 40 20
   :header-rows: 1

   * - R (neuroim2)
     - Python (neuroim)
     - Notes
   * - ``BigNeuroVec(...)``
     - ``pn.BigNeuroVec(...)``
     - File-backed storage
   * - ``MappedNeuroVec(...)``
     - ``pn.MappedNeuroVec(...)``
     - Memory-mapped file
   * - ``FileBackedNeuroVec(...)``
     - ``pn.FileBackedNeuroVec(...)``
     - Lazy loading

Missing Functions
-----------------

The following R functions are not yet implemented in Python:

- ``drop()`` - Use standard numpy indexing instead
- ``lookup()`` - Use numpy's advanced indexing
- ``voxels()`` - Access `.data` property directly
- ``mapf()`` - Use numpy's ``apply_along_axis`` or list comprehensions
- ``perm_mat()`` - Use ``np.random.permutation``
- ``patch_set()`` - Custom implementation needed

Notes on Differences
--------------------

1. **Function vs Method**: R uses generic functions while Python uses object methods
2. **Indexing**: Always subtract 1 from R indices when using Python
3. **NA vs NaN**: R's NA becomes numpy's NaN in Python
4. **Recycling**: R's automatic vector recycling requires explicit broadcasting in numpy