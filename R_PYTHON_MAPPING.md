# R to Python API Mapping Guide

## Quick Reference: Common Translations

### Basic Operations
| R Code | Python Code | Notes |
|--------|-------------|-------|
| `vol <- read_vol("brain.nii")` | `vol = read_vol("brain.nii")` | |
| `vol[10, 20, 30]` | `vol[9, 19, 29]` | 0-indexed |
| `dim(vol)` | `vol.shape` | Property |
| `space(vol)` | `vol.space` | Property |
| `values(vol)` | `vol.values()` | Method |
| `as.mask(vol)` | `vol.as_mask()` | Method |
| `vol > 100` | `vol > 100` | Returns mask |

### Creating Objects
| R Code | Python Code |
|--------|-------------|
| `NeuroSpace(dim=c(64,64,32), spacing=c(3,3,4))` | `NeuroSpace(dim=(64,64,32), spacing=(3,3,4))` |
| `DenseNeuroVol(data, space)` | `DenseNeuroVol(data, space)` |
| `SparseNeuroVol(data, space, mask)` | `SparseNeuroVol(data, space, mask)` |

### Coordinate Operations
| R Code | Python Code | Notes |
|--------|-------------|-------|
| `coord_to_grid(space, c(30, 40, 20))` | `coord_to_grid(space, [30, 40, 20])` | |
| `grid_to_coord(space, c(10, 15, 8))` | `grid_to_coord(space, [10, 15, 8])` | |
| `index_to_grid(space, 1000)` | `index_to_grid(space, 999)` | 0-indexed |

### 4D Data Access
| R Code | Python Code | Notes |
|--------|-------------|-------|
| `vec[10, 20, 30, ]` | `vec[9, 19, 29, :]` | Time series |
| `series(vec, 10, 20, 30)` | `vec.series(9, 19, 29)` | |
| `vec[,,, 5]` | `vec[:, :, :, 4]` | 5th volume |
| `sub_vector(vec, 1:10)` | `vec.sub_vector(range(10))` | First 10 timepoints |

### ROI Operations
| R Code | Python Code |
|--------|-------------|
| `roi <- spherical_roi(center, radius, space)` | `roi = spherical_roi(center, radius, space)` |
| `extract_roi(vol, roi)` | `vol.extract_roi(roi)` |
| `coords(roi)` | `roi.coords` |

### Searchlight
| R Code | Python Code |
|--------|-------------|
| `searchlight(vol, radius=3)` | `searchlight(vol, radius=3)` |
| `searchlight_coords(mask, radius=3)` | `searchlight_coords(mask, radius=3)` |

### File I/O
| R Code | Python Code |
|--------|-------------|
| `write_vol(vol, "output.nii")` | `write_vol(vol, "output.nii")` |
| `hdr <- read_header("brain.nii")` | `hdr = read_header("brain.nii")` |

## Detailed Class Method Mappings

### NeuroVol Methods
| R Generic/Method | Python Method | Description |
|-----------------|---------------|-------------|
| `dim(vol)` | `vol.shape` | Get dimensions |
| `spacing(vol)` | `vol.spacing` | Get voxel size |
| `origin(vol)` | `vol.origin` | Get origin |
| `bounds(vol)` | `vol.bounds()` | Get bounding box |
| `as.matrix(vol)` | `vol.as_matrix()` | Convert to 2D |
| `as.sparse(vol)` | `vol.as_sparse()` | Convert to sparse |
| `as.dense(vol)` | `vol.as_dense()` | Convert to dense |
| `as.mask(vol)` | `vol.as_mask()` | Convert to binary |
| `values(vol)` | `vol.values()` | Extract all values |

### NeuroVec Methods
| R Generic/Method | Python Method | Description |
|-----------------|---------------|-------------|
| `series(vec, i, j, k)` | `vec.series(i, j, k)` | Extract time series |
| `sub_vector(vec, indices)` | `vec.sub_vector(indices)` | Temporal subset |
| `scale_series(vec, ...)` | `vec.scale_series(...)` | Z-score series |
| `vols(vec, indices)` | `vec.vols(indices)` | Extract volumes |

### Indexing Differences

#### 3D Indexing
```r
# R (1-indexed)
vol[10, 20, 30]         # Single voxel
vol[1:10, , ]           # Slab
vol[c(5,10,15), 20, ]   # Multiple slices
```

```python
# Python (0-indexed)
vol[9, 19, 29]          # Single voxel
vol[0:10, :, :]         # Slab
vol[[4,9,14], 19, :]    # Multiple slices
```

#### 4D Indexing
```r
# R
vec[10, 20, 30, ]       # Time series at voxel
vec[, , , 1:10]         # First 10 volumes
vec[10:20, 10:20, 5, ]  # ROI time series
```

```python
# Python
vec[9, 19, 29, :]       # Time series at voxel
vec[:, :, :, 0:10]      # First 10 volumes
vec[9:20, 9:20, 4, :]   # ROI time series
```

## Function Argument Mappings

### read_vol / read_vec
| R Argument | Python Argument | Default |
|-----------|-----------------|---------|
| `file_name` | `file_name` | Required |
| `mode` | `mode` | "normal" |
| `data_type` | `dtype` | None (auto) |
| `mask` | `mask` | None |

### searchlight
| R Argument | Python Argument | Type |
|-----------|-----------------|------|
| `mask` | `mask` | NeuroVol |
| `radius` | `radius` | float |
| `eager` | `eager` | bool |
| `nonzero` | `nonzero` | bool |

### gaussian_blur
| R Argument | Python Argument | Type |
|-----------|-----------------|------|
| `vol` | `vol` | NeuroVol |
| `sigma` | `sigma` | float or tuple |
| `mask` | `mask` | NeuroVol |

## Common Patterns

### Creating Masks
```r
# R
mask <- vol > threshold
masked_vol <- vol[mask]
```

```python
# Python
mask = vol > threshold
masked_vol = vol[mask]
```

### Applying Functions
```r
# R
result <- map_fun(vol, function(x) x^2)
```

```python
# Python
result = vol.map_fun(lambda x: x**2)
# or
result = vol.apply(np.square)
```

### Working with ROIs
```r
# R
rois <- spherical_roi_set(centers, radius, space)
values <- extract_roi_values(vol, rois)
```

```python
# Python
rois = spherical_roi_set(centers, radius, space)
values = vol.extract_roi_values(rois)
```

## Key Differences to Remember

1. **Indexing**: Python uses 0-based indexing
2. **Slicing**: Python excludes the end index (e.g., `0:10` gives indices 0-9)
3. **Properties vs Functions**: Many R functions become Python properties
4. **Method Chaining**: Python supports method chaining more naturally
5. **Missing Values**: R's `NA` → Python's `np.nan`
6. **Vectors**: R's `c()` → Python's lists `[]` or tuples `()`
7. **Ranges**: R's `1:10` → Python's `range(10)` or `np.arange(10)`

## Error Messages

Common errors when transitioning from R:

1. **IndexError**: Remember 0-indexing
2. **AttributeError**: Check if it's a property (no parentheses)
3. **TypeError**: Tuples need parentheses, not `c()`
4. **ValueError**: Check array shapes match