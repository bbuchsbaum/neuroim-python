# Migrating from R neuroim2 to Python neuroimpy

This guide helps R users transition to the Python neuroimpy package, providing quick references and key differences.

## Quick Start

### Installation

```r
# R
install.packages("devtools")
devtools::install_github("bbuchsbaum/neuroim2")
library(neuroim2)
```

```python
# Python
pip install neuroimpy
import neuroimpy as pn
```

## Critical Differences

### 🔴 Indexing: 0-based vs 1-based

This is the most important difference that will cause bugs if not handled properly:

```r
# R: 1-based indexing
vol[10, 20, 30]     # 10th row, 20th column, 30th slice
vol[,,1]            # First slice
```

```python
# Python: 0-based indexing  
vol[9, 19, 29]      # 10th row, 20th column, 30th slice (same as R's [10,20,30])
vol[:, :, 0]        # First slice
```

### 📦 Function vs Method Calls

```r
# R: Generic functions
dim(vol)
space(vol)
series(vec, coords)
```

```python
# Python: Object methods
vol.shape  # or vol.dim
vol.space
vec.series(coords)
```

## Common Operations Mapping

### Creating Volumes

```r
# R
space <- NeuroSpace(dim = c(64, 64, 32))
data <- array(rnorm(64*64*32), dim = c(64, 64, 32))
vol <- NeuroVol(data, space)
```

```python
# Python
import numpy as np
space = pn.NeuroSpace(dim=(64, 64, 32))
data = np.random.randn(64, 64, 32)
vol = pn.NeuroVol(data, space)
```

### Reading/Writing Files

```r
# R
brain <- read_vol("brain.nii")
write_vol(brain, "output.nii")
```

```python
# Python
brain = pn.read_vol("brain.nii")
pn.write_vol(brain, "output.nii")
```

### Time Series Extraction

```r
# R
ts <- series(vec, c(32, 32, 16))          # Single voxel
multi_ts <- series(vec, coords_matrix)     # Multiple voxels
```

```python
# Python (remember 0-based indexing!)
ts = vec.series_3d(31, 31, 15)           # Single voxel
multi_ts = vec.series(coords_matrix - 1)  # Multiple voxels
```

### ROI Operations

```r
# R
roi <- spherical_roi(c(32, 32, 16), radius = 5, space = space(vol))
roi_data <- series_roi(vec, roi)
```

```python
# Python
roi = pn.spherical_roi([31, 31, 15], radius=5, space=vol.space)
roi_data = vec.series_roi(roi)
```

### Searchlight Analysis

```r
# R
result <- searchlight(mask, radius = 5, method = my_function)
```

```python
# Python
result = pn.searchlight(mask, radius=5, method=my_function)
```

## Function Name Mappings

| R Function | Python Equivalent | Notes |
|------------|------------------|-------|
| `read_vol()` | `pn.read_vol()` | Direct mapping |
| `write_vol()` | `pn.write_vol()` | Direct mapping |
| `read_vec()` | `pn.read_vec()` | Direct mapping |
| `write_vec()` | `pn.write_vec()` | Direct mapping |
| `dim()` | `.shape` | Property access |
| `space()` | `.space` | Property access |
| `series()` | `.series()` | Method call, 0-based |
| `series_roi()` | `.series_roi()` | Method call |
| `conn_comp()` | `pn.conn_comp()` | Direct mapping |
| `partition()` | `pn.partition()` | Direct mapping |
| `searchlight()` | `pn.searchlight()` | Direct mapping |

## Class Name Mappings

| R Class | Python Class | Usage |
|---------|--------------|-------|
| `NeuroSpace` | `pn.NeuroSpace` | Spatial metadata |
| `NeuroVol` | `pn.NeuroVol` | 3D volumes |
| `NeuroVec` | `pn.NeuroVec` | 4D time series |
| `SparseNeuroVol` | `pn.SparseNeuroVol` | Sparse 3D data |
| `SparseNeuroVec` | `pn.SparseNeuroVec` | Sparse 4D data |
| `LogicalNeuroVol` | `pn.LogicalNeuroVol` | Binary masks |
| `ClusteredNeuroVol` | `pn.ClusteredNeuroVol` | Clustered data |

## Features Not Needed in Python

Several R features are not implemented because Python/NumPy provides better alternatives:

- **`IndexLookupVol`**: NumPy's indexing is already efficient
- **`linear_access`**: Use `array.ravel()` or `array.flat`
- **`ArrayLike` interface**: Python's duck typing makes this unnecessary
- **`NeuroBucket`**: Not needed (per design decision)
- **`NIML` format**: Not needed (per design decision)

## Common Gotchas

### 1. Index Conversion
When converting R code, always subtract 1 from indices:
```python
# R: vol[10, 20, 30] 
# Python: vol[9, 19, 29]
```

### 2. NA vs NaN
```r
# R
vol[is.na(vol)] <- 0
```
```python
# Python
vol.data[np.isnan(vol.data)] = 0
```

### 3. Matrix Operations
R is column-major, Python is row-major by default:
```python
# Ensure Fortran (column-major) order when needed
data = np.asarray(data, order='F')
```

### 4. Vector Recycling
R automatically recycles vectors, Python requires explicit broadcasting:
```python
# Use numpy broadcasting rules
result = array[:, np.newaxis] + vector
```

## Example: Complete fMRI Analysis Pipeline

### R Version
```r
# Load data
mask <- read_vol("brain_mask.nii")
fmri <- read_vec("functional.nii")

# Create ROI
roi <- spherical_roi(c(40, 40, 20), radius = 10, space = space(mask))

# Extract time series
roi_ts <- series_roi(fmri, roi)
mean_ts <- colMeans(roi_ts)

# Run searchlight
result <- searchlight(mask, radius = 5, 
                     method = function(x) cor(x[,1], x[,2]))

# Save results
write_vol(result, "searchlight_results.nii")
```

### Python Version
```python
# Load data
mask = pn.read_vol("brain_mask.nii")
fmri = pn.read_vec("functional.nii")

# Create ROI (note 0-based indexing)
roi = pn.spherical_roi([39, 39, 19], radius=10, space=mask.space)

# Extract time series
roi_ts = fmri.series_roi(roi)
mean_ts = np.mean(roi_ts, axis=1)

# Run searchlight
def correlation(x):
    return np.corrcoef(x[:, 0], x[:, 1])[0, 1]

result = pn.searchlight(mask, radius=5, method=correlation)

# Save results
pn.write_vol(result, "searchlight_results.nii")
```

## Performance Tips

1. **Use NumPy operations** instead of loops
2. **Preallocate arrays** when possible
3. **Use sparse representations** for masked data
4. **Consider memory-mapped arrays** for very large datasets

## Getting Help

- **Documentation**: See the full docs at `/docs/build/html/index.html`
- **API Reference**: Detailed function documentation in `/docs/source/api/`
- **Examples**: Check `/docs/source/migration/examples.rst` for more comparisons
- **Issues**: Report bugs or request features on GitHub

## Checklist for Migration

- [ ] Replace all 1-based indices with 0-based
- [ ] Convert function calls to method calls
- [ ] Replace `NA` checks with `np.isnan()`
- [ ] Update array creation to use NumPy
- [ ] Test numerical outputs match R version (within tolerance)
- [ ] Update any R-specific idioms to Python equivalents