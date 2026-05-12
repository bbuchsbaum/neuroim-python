# Neuroim Vignette Functionality Test Summary

This document summarizes the results of testing the functionality described in the R neuroim2 vignettes against the Python neuroim implementation.

## Test Results Overview

All four test scripts were created to verify the Python implementation matches the R vignette functionality:

1. **test_vignette_image_volumes.py** - Tests 3D volume operations
2. **test_vignette_neuro_vectors.py** - Tests 4D time series operations  
3. **test_vignette_roi.py** - Tests region of interest functionality
4. **test_vignette_pipelines.py** - Tests pipeline and functional operations

## ImageVolumes Functionality (test_vignette_image_volumes.py)

**Working:**
- ✓ Basic imports (read_vol, write_vol, DenseNeuroVol, NeuroSpace)
- ✓ Read/write NIFTI volumes
- ✓ Volume arithmetic operations (+, -, *, scalar operations)
- ✓ Volume indexing
- ✓ Logical (binary) volume conversion
- ✓ Sparse volume conversion

**Missing/Different:**
- ✗ Volume statistics methods (vol_mean, vol_sd) - not implemented
  - Workaround: Use numpy (np.mean, np.std)
- Note: Arithmetic order matters (vol * 2 works, but 2 * vol doesn't)

## NeuroVector Functionality (test_vignette_neuro_vectors.py) 

**Working:**
- ✓ Create 4D NeuroVec from array or list of volumes
- ✓ Extract subsets with sub_vector (slice and list indexing)
- ✓ Single and multiple voxel time series extraction
- ✓ Linear index series extraction
- ✓ ROI-based series extraction (series_roi)
- ✓ Concatenation of NeuroVecs
- ✓ Extract individual volumes
- ✓ vols() method to get all volumes
- ✓ Sparse conversion with mask
- ✓ scale_series for time series normalization

**API Differences:**
- Need to create 3D space separately (no space.to_3d() method)
- ROIs must be created from 3D volumes, not 4D vectors

## ROI Functionality (test_vignette_roi.py)

**Working:**
- ✓ Spherical ROI creation with and without fill values
- ✓ World coordinate to voxel conversion (coord_to_grid)
- ✓ ROI to sparse volume conversion
- ✓ Square and cuboid ROI creation
- ✓ Multiple ROI creation (spherical_roi_set)
- ✓ Basic searchlight functionality
- ✓ Searchlight coordinates iteration
- ✓ Random searchlight

**Missing/Different:**
- ✗ Clustered searchlight - requires cvol parameter
- API: searchlight takes mask as first parameter, not volume
- coord_to_grid instead of world_to_grid

## Pipelines Functionality (test_vignette_pipelines.py)

**Working:**
- ✓ Connected components analysis (conn_comp)
- ✓ Split clusters functionality
- ✓ Searchlight with statistics computation
- ✓ Slice-wise processing
- ✓ 4D volume processing and statistics
- ✓ Voxel time series processing
- ✓ K-nearest neighbor smoothing concepts
- ✓ Custom pipeline building

**Missing/Different:**
- ✗ partition() doesn't accept mask parameter
- ClusteredNeuroVol has different structure (no .data attribute)
- Access components via comp.voxels list instead

## Key API Differences from R

1. **Import structure**: Functions are in submodules, not all at package level
2. **Space conversion**: No automatic 3D/4D space conversion methods
3. **Method names**: Some differences (coord_to_grid vs world_to_grid)
4. **Parameter order**: searchlight takes mask first, not volume
5. **Missing methods**: Some statistical methods not implemented on volumes

## Recommendations

1. **Documentation**: Update RST documentation to reflect actual API
2. **Examples**: Create tested example scripts instead of static RST code
3. **Missing features**: Consider implementing:
   - Volume statistics methods (vol_mean, vol_sd, which_min, which_max)
   - Improved partition function with mask support
   - Clustered searchlight with simpler API
4. **Testing**: Add these vignette tests to CI to ensure compatibility

## Conclusion

The core functionality from the R vignettes is largely implemented in Python, with most operations working as expected. The main differences are in API details and some missing convenience methods. The Python implementation successfully provides:

- Full 3D and 4D neuroimaging data support
- ROI creation and manipulation
- Searchlight analysis capabilities
- Connected components and clustering
- Pipeline composition for complex analyses

With minor adjustments to documentation and a few missing method implementations, the Python package provides equivalent functionality to the R version.