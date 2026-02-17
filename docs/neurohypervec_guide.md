# NeuroHyperVec: Working with 5D+ Neuroimaging Data

## Overview

NeuroHyperVec is a powerful extension to the neuroimpy library that enables working with neuroimaging data that has more than 4 dimensions. While traditional neuroimaging data is 4D (3 spatial dimensions + time), many modern applications require additional dimensions to represent features, parameters, or other varying quantities.

## Use Cases

### 1. Multi-Echo fMRI
Multi-echo fMRI collects data at multiple echo times (TEs) to improve signal quality and enable advanced denoising:

```python
import neuroimpy as pn
import numpy as np

# Create 5D space: 64x64x40 voxels, 200 timepoints, 3 echoes
space = pn.NeuroSpace(dim=(64, 64, 40, 200, 3))
echo_data = np.random.randn(64, 64, 40, 200, 3)

# Create multi-echo data
multi_echo = pn.DenseNeuroHyperVec(echo_data, space)

# Combine echoes using optimal weighting
te_weights = np.array([0.5, 0.3, 0.2])  # TE-dependent weights
combined = multi_echo.weighted_mean_features(te_weights)
# Returns a NeuroVec with combined echo data
```

### 2. Frequency Band Analysis
Store power in multiple frequency bands from spectral analysis:

```python
# 5 frequency bands: delta, theta, alpha, beta, gamma
freq_space = pn.NeuroSpace(dim=(64, 64, 40, 100, 5))
spectral_data = np.random.randn(64, 64, 40, 100, 5)
spectral = pn.DenseNeuroHyperVec(spectral_data, freq_space)

# Extract alpha band power (index 2)
alpha_power = spectral[:, :, :, :, 2]  # Returns NeuroVec

# Get mean power across all bands
mean_power = spectral.mean_features()  # Returns NeuroVec
```

### 3. Parameter Maps
Store multiple parameter estimates from model fitting:

```python
# 4 parameters from a model fit
param_space = pn.NeuroSpace(dim=(64, 64, 40, 50, 4))
param_data = np.random.randn(64, 64, 40, 50, 4)
params = pn.DenseNeuroHyperVec(param_data, param_space)

# Apply function to compute R² from parameters
def compute_r_squared(param_vec):
    # Example: compute R² from model parameters
    return 1 - param_vec[3] / np.var(param_vec)

r_squared = params.apply_feature_func(compute_r_squared)
# Returns NeuroVec with R² values
```

## Data Structures

### DenseNeuroHyperVec
Standard dense array representation for 5D+ data:

```python
# Create from numpy array
data = np.random.randn(10, 10, 10, 20, 5)
space = pn.NeuroSpace(dim=(10, 10, 10, 20, 5))
hvec = pn.DenseNeuroHyperVec(data, space, label="my_data")

# Access properties
print(hvec.shape)           # (10, 10, 10, 20, 5)
print(hvec.spatial_shape)   # (10, 10, 10)
print(hvec.n_timepoints)    # 20
print(hvec.n_features)      # 5
```

### SparseNeuroHyperVec
Memory-efficient representation for sparse 5D+ data:

```python
# Create mask for sparse data
mask = pn.LogicalNeuroVol(
    np.random.rand(64, 64, 32) > 0.7,
    pn.NeuroSpace(dim=(64, 64, 32))
)

# Sparse data: [features x time x voxels]
n_voxels = mask.data.sum()
sparse_data = np.random.randn(8, 100, n_voxels)

# Create sparse hypervector
space = pn.NeuroSpace(dim=(64, 64, 32, 100, 8))
sparse_hvec = pn.SparseNeuroHyperVec(sparse_data, mask, space)

# Convert to dense if needed
dense_hvec = sparse_hvec.as_dense()
```

### MappedNeuroHyperVec
Memory-mapped arrays for very large datasets:

```python
# Create memory-mapped hypervector
space = pn.NeuroSpace(dim=(128, 128, 64, 500, 10))
mapped_hvec = pn.MappedNeuroHyperVec(
    filename="large_data.dat",
    space=space,
    dtype=np.float32,
    mode='w+'  # read-write mode
)

# Work with slices without loading full data
slice_data = mapped_hvec[64, 64, 32, :, :]  # shape: (500, 10)

# Compute mean across time and features (chunked processing)
mean_vol = mapped_hvec.mean_time_features()
```

## Operations

### Feature Operations
```python
# Average across features
mean_vec = hvec.mean_features()  # Returns NeuroVec

# Standard deviation across features at specific timepoint
std_vol = hvec.std_features(time_idx=0)  # Returns NeuroVol

# Select subset of features
subset = hvec.select_features([0, 2, 4])  # Select features 0, 2, 4

# Weighted combination
weights = np.array([0.3, 0.3, 0.4])
weighted = hvec.weighted_mean_features(weights)
```

### Time Series Extraction
```python
# Single voxel, all features
series = hvec.series([30, 30, 20])
# Returns: (n_timepoints, n_features)

# Multiple voxels, all features
coords = np.array([[30, 30, 20], [40, 40, 25]])
multi_series = hvec.series(coords)
# Returns: (n_timepoints, n_features, n_voxels)

# Single voxel, specific feature
feature_series = hvec.series([30, 30, 20], feature=2)
# Returns: (n_timepoints,)
```

### Concatenation
```python
# Concatenate along feature dimension
hvec1 = pn.DenseNeuroHyperVec(data1, space1)  # 3 features
hvec2 = pn.DenseNeuroHyperVec(data2, space2)  # 2 features

combined = pn.concat_features([hvec1, hvec2])  # 5 features total
```

## I/O Operations

NeuroHyperVec uses HDF5 format for efficient storage:

```python
# Save to HDF5
pn.write_neurohypervec(hvec, "my_data.h5")

# Load from HDF5
loaded = pn.read_neurohypervec("my_data.h5")

# HDF5 preserves sparse/dense type and all metadata
```

## Best Practices

1. **Choose the right representation**:
   - Use `DenseNeuroHyperVec` for full brain data
   - Use `SparseNeuroHyperVec` for masked/ROI data
   - Use `MappedNeuroHyperVec` for very large datasets

2. **Memory considerations**:
   - 5D+ data can be memory-intensive
   - Consider sparse or mapped representations
   - Process in chunks when possible

3. **Feature dimension semantics**:
   - Keep features conceptually related
   - Document what each feature represents
   - Use meaningful labels

## Examples

### Complete Multi-Echo fMRI Pipeline
```python
# Load multi-echo data
space = pn.NeuroSpace(dim=(64, 64, 40, 300, 3))
echo_data = load_multiecho_data()  # Your loading function
multi_echo = pn.DenseNeuroHyperVec(echo_data, space, label="ME-fMRI")

# Compute temporal SNR for each echo
def compute_tsnr(time_series):
    return np.mean(time_series) / np.std(time_series)

tsnr_map = multi_echo.apply_feature_func(compute_tsnr)

# Optimal combination based on TE-dependent weighting
echo_times = np.array([14.5, 28.0, 42.5])  # milliseconds
weights = echo_times / np.sum(echo_times)
combined = multi_echo.weighted_mean_features(weights)

# Save results
pn.write_neurohypervec(multi_echo, "multiecho_data.h5")
pn.write_vec(combined, "combined_timeseries.nii.gz")
```

### Spectral Analysis Results
```python
# Store power spectral density in frequency bands
freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
space = pn.NeuroSpace(dim=(64, 64, 40, 200, len(freq_bands)))

# Compute spectral power (placeholder)
spectral_power = compute_spectral_power(eeg_data)  # Your function
spectral_hvec = pn.DenseNeuroHyperVec(
    spectral_power, 
    space, 
    label="EEG Power Spectrum"
)

# Extract specific band
alpha_idx = freq_bands.index('alpha')
alpha_power = spectral_hvec[:, :, :, :, alpha_idx]

# Find peak frequency band at each voxel/time
def find_peak_band(power_vec):
    return np.argmax(power_vec)

peak_bands = spectral_hvec.apply_feature_func(find_peak_band)

# Average power across time and bands
mean_power_map = spectral_hvec.mean_time_features()
```

## Advanced Usage

### Custom Feature Functions
```python
# Apply PCA across features
from sklearn.decomposition import PCA

def feature_pca_score(feature_vec, n_components=1):
    """Extract first PC score from features."""
    if len(feature_vec) < 2:
        return feature_vec[0]
    
    pca = PCA(n_components=n_components)
    # Reshape for PCA (need 2D input)
    scores = pca.fit_transform(feature_vec.reshape(-1, 1))
    return scores[0, 0]

pc1_scores = hvec.apply_feature_func(feature_pca_score)
```

### ROI Analysis with Features
```python
# Extract hypervector data for ROI
roi = pn.spherical_roi(center=[32, 32, 20], radius=5)
roi_coords = roi.coords

# Get all features for ROI voxels
roi_data = hvec.series(roi_coords)  # (time, features, voxels)

# Compute mean across ROI for each feature
roi_mean = np.mean(roi_data, axis=2)  # (time, features)

# Correlate features within ROI
feature_corr = np.corrcoef(roi_mean.T)  # (features, features)
```

## Limitations

1. **Dimension limit**: Currently supports up to 5D (3 spatial + time + 1 feature dimension)
2. **Memory usage**: 5D data can quickly become very large
3. **Visualization**: Limited visualization options for 5D+ data

## Future Directions

- Support for 6D+ data (multiple feature dimensions)
- Lazy evaluation for large datasets
- Integration with deep learning frameworks
- Advanced visualization tools for high-dimensional data