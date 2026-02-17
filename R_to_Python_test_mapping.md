# R to Python Test Mapping

This document shows which R tests from neuroim2 have corresponding Python tests in neuroimpy.

## Test Coverage Summary

### ✅ Tests with Python Equivalents

1. **test-neurovol.R** → Multiple Python test files:
   - `test_neurovol.py`
   - `test_neurovol_comprehensive.py`
   - `test_neurovol_edge_cases.py`
   - `test_phase2_neurovol.py`
   - `test_phase2_neurovol_core.py`
   - `test_dense_neuro_vol.py`

2. **test-neurovec.R** → Multiple Python test files:
   - `test_neurovec_comprehensive.py`
   - `test_neurovec_edge_cases.py`
   - `test_phase3_neurovec.py`
   - `test_phase3_neurovec_core.py`

3. **test-neuroslice.R** → Python test files:
   - `test_phase4_neuroslice.py`

4. **test-conn_comp.R** → Python test files:
   - `test_connected_components.py`

5. **test-roivol.R** → Multiple Python test files:
   - `test_roi_comprehensive.py`
   - `test_phase5_roi.py`
   - `test_series_roi.py`
   - `test_series_roi_memmapped.py`

6. **test-resample.R** → Python test files:
   - `test_phase8_resample.py`

7. **test-searchlight.R** → Multiple Python test files:
   - `test_searchlight.py`
   - `test_searchlight_parallel.py`

8. **test-spatfilter.R** → Python test files:
   - `test_phase7_spatial_filters.py` (includes gaussian_blur, guided_filter, bilateral_filter)

9. **test-hypervec.R** → Multiple Python test files:
   - `test_neurohypervec.py`
   - `test_neuro_hypervec.py`

10. **test-filebacked.R** → Python test files:
    - `test_file_backed_neuro_vec.py`

11. **test-mapped.R** → Python test files:
    - `test_memory_mapped.py`

12. **test-latentvec.R** → Python test files:
    - `test_latent_neuro_vec.py`

13. **test-vignette-neuro-vectors.R** → Python test files:
    - `test_vignette_neuro_vectors.py`

14. **test-vignette-pipelines.R** → Python test files:
    - `test_vignette_pipelines.py`

15. **test-vignette-roi.R** → Python test files:
    - `test_vignette_roi.py`

16. **test-splitscale.R** → Python test files:
    - `test_stats.py` (includes split_reduce, split_scale)
    - `test_stats_extra.py`

### ⚠️ Tests with Partial or Different Implementation

1. **test-clusteredneurovol.R** → Partial coverage:
   - ClusteredNeuroVol functionality appears in `test_stats.py` and `test_connected_components.py`
   - Not a dedicated test file but functionality is tested

2. **test-vecseq.R** → Partial coverage:
   - NeuroVecSeq functionality tested within `test_neurovec_comprehensive.py` (using `neurovecseq` function)
   - Also referenced in `test_memory_mapped.py`, `test_phase3_neurovec.py`
   - Not a dedicated test file but functionality is tested

### ❌ Tests Without Clear Python Equivalents

1. **test-imageproc.R** - Image processing tests
   - While `test_phase7_spatial_filters.py` covers some filtering (gaussian_blur, guided_filter)
   - The specific image processing operations from test-imageproc.R may not be fully covered

2. **test-h5neurovec.R** - HDF5 format tests
   - Although commented out in R, no clear H5NeuroVec tests found in Python
   - Some HDF5 functionality might be in `test_phase9_file_formats.py` but needs verification

## Additional Python Tests Not in R

The Python implementation includes several test files that don't have direct R equivalents:

1. **test_coordinate_systems.py** - Coordinate system tests
2. **test_coordinate_transformations.py** - Coordinate transformation tests
3. **test_io_comprehensive.py** - Comprehensive I/O tests
4. **test_memory_efficiency.py** - Memory efficiency tests
5. **test_performance_benchmarks.py** - Performance benchmarking
6. **test_edge_cases.py** - General edge case tests
7. **test_r_equivalence.py** - Tests for R equivalence
8. **test_r_equivalence_numpy.py** - NumPy-based R equivalence tests
9. **test_plotting.py** - Plotting functionality tests
10. **test_vignette_image_volumes.py** - Image volume vignette tests
11. **test_phase1_core.py** - Core functionality tests
12. **test_phase6_io.py** - I/O phase tests
13. **test_phase9_file_formats.py** - File format tests

## Recommendations

1. **test-imageproc.R**: While some functionality exists in spatial filters, specific image processing operations may need additional tests
2. **test-h5neurovec.R**: If HDF5 support is needed, dedicated tests should be added
3. **test-clusteredneurovol.R** and **test-vecseq.R**: Consider creating dedicated test files for clearer organization

## Summary

- **20 R test files** total
- **15 have clear Python equivalents** (75%)
- **2 have partial coverage** (10%)
- **2 are missing** (10%)
- **1 is commented out in R** (5%)

The Python test suite is more comprehensive overall, with additional tests for coordinate systems, I/O, performance, and edge cases that aren't present in the R test suite.