# neuroimpy

A modern Python library for neuroimaging data analysis, providing efficient data structures and algorithms for working with 3D, 4D, and 5D+ brain imaging data.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-543%20passing-green)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-71%25-yellowgreen)](tests/)

## Features

- 📊 **Multi-dimensional support**: Work with 3D (NeuroVol), 4D (NeuroVec), and 5D+ (NeuroHyperVec) neuroimaging data
- 🚀 **Memory efficient**: Sparse and memory-mapped representations for large datasets
- 🔍 **Advanced analysis**: ROI extraction, searchlight analysis, spatial filtering, and connectivity
- 📁 **File format support**: Read/write NIFTI, AFNI, and HDF5 formats
- 🔄 **R compatibility**: Designed to match R's neuroim2 package for cross-language workflows
- ⚡ **Performance**: Optimized operations using NumPy and SciPy

## Installation

### From PyPI (coming soon)

```bash
pip install neuroimpy
```

### From Source

```bash
git clone https://github.com/bbuchsbaum/neuroimpy.git
cd neuroimpy
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/bbuchsbaum/neuroimpy.git
cd neuroimpy
pip install -e ".[dev]"
```

## Quick Start

```python
import neuroimpy as pn
import numpy as np

# Load a 3D brain volume
vol = pn.read_vol("brain.nii.gz")
print(f"Volume shape: {vol.shape}")
print(f"Voxel size: {vol.spacing} mm")

# Create a 4D time series (fMRI data)
space_4d = pn.NeuroSpace(dim=(64, 64, 40, 200))
data = np.random.randn(64, 64, 40, 200)
fmri = pn.NeuroVec(data, space_4d)

# Extract time series from a voxel
ts = fmri.series(30, 30, 20)  # Returns time series at voxel (30,30,20)

# Create an ROI and extract data
roi = pn.spherical_roi(vol, center=[32, 32, 20], radius=5)
roi_data = fmri.series_roi(roi)  # Returns time x voxels matrix

# Searchlight analysis
def correlation_searchlight(data):
    """Example searchlight function - compute mean correlation"""
    if data.shape[1] < 2:
        return 0
    corr = np.corrcoef(data.T)
    return np.mean(corr[np.triu_indices_from(corr, k=1)])

result = pn.searchlight(fmri, radius=3, method=correlation_searchlight)
```

## Core Data Structures

### NeuroVol - 3D Brain Volumes

```python
# Create from data
space = pn.NeuroSpace(dim=(91, 109, 91), spacing=(2, 2, 2))
vol = pn.NeuroVol(data_3d, space)

# Operations
smoothed = pn.gaussian_blur(vol, sigma=2)
mask = vol > vol.mean()
masked_vol = vol[mask]
```

### NeuroVec - 4D Time Series

```python
# Create 4D fMRI data
space_4d = pn.NeuroSpace(dim=(64, 64, 40, 200))
fmri = pn.NeuroVec(data_4d, space_4d)

# Extract volumes at specific timepoints
vol_t0 = fmri[..., 0]  # First volume
vol_t10 = fmri[..., 10]  # 11th volume

# Time series operations
mean_vol = fmri.mean(axis=3)  # Mean across time
```

### NeuroHyperVec - 5D+ Multi-Feature Data

```python
# Create 5D data (e.g., multi-echo fMRI with 3 echoes)
space_5d = pn.NeuroSpace(dim=(64, 64, 40, 200, 3))
multi_echo = pn.NeuroHyperVec(data_5d, space_5d)

# Combine echoes with weighted average
weights = np.array([0.5, 0.3, 0.2])
combined = multi_echo.weighted_mean_features(weights)

# Extract specific feature
echo1 = multi_echo[..., 0]  # First echo as NeuroVec
```

## Advanced Features

### Region of Interest (ROI) Analysis

```python
# Spherical ROI
roi_sphere = pn.spherical_roi(vol, center=[45, 54, 45], radius=10)

# Cuboid ROI  
roi_cube = pn.cuboid_roi(vol, center=[45, 54, 45], width=[20, 20, 20])

# Extract ROI data
roi_values = vol[roi_sphere]
roi_coords = roi_sphere.coords
```

### Searchlight Analysis

```python
# Define searchlight function
def pattern_similarity(data):
    """Compute pattern similarity within searchlight"""
    return np.mean(np.corrcoef(data))

# Run searchlight
results = pn.searchlight(fmri, radius=3, method=pattern_similarity)

# Run searchlight with parallel processing for faster computation
results_parallel = pn.searchlight(
    fmri, 
    radius=3, 
    method=pattern_similarity,
    eager=True,  # Required for parallel processing
    cores=4      # Use 4 CPU cores
)
```

### Spatial Filtering

```python
# Gaussian smoothing
smoothed = pn.gaussian_blur(vol, sigma=2)

# Bilateral filter (edge-preserving)
bilateral = pn.bilateral_filter(vol, spatial_sigma=2, range_sigma=0.5)

# Guided filter
guided = pn.guided_filter(vol, guide=mask, radius=3, epsilon=0.1)
```

### Memory-Efficient Operations

```python
# Sparse representation for masked data
mask = pn.LogicalNeuroVol(brain_mask, space)
sparse_fmri = fmri.as_sparse(mask)

# Memory-mapped for huge datasets
big_data = pn.MappedNeuroVec("huge_fmri.dat", space_4d)
chunk = big_data[30:40, 30:40, 20:30, :]  # Load only what's needed
```

## Documentation

Full documentation is available at [https://neuroimpy.readthedocs.io](https://neuroimpy.readthedocs.io) (coming soon).

- [Installation Guide](docs/source/installation.rst)
- [Tutorials](docs/source/tutorials/)
- [API Reference](docs/source/api/)
- [Migration from R](docs/source/migration/)

## Requirements

- Python 3.8+
- NumPy >= 1.20
- SciPy >= 1.7
- NiBabel >= 3.0 (for NIFTI I/O)
- scikit-learn >= 0.24
- h5py >= 3.0 (for HDF5 support)

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/source/contributing.rst) for details.

```bash
# Run tests
pytest

# Check code style
black --check src/
flake8 src/

# Type checking
mypy src/
```

## Citation

If you use neuroimpy in your research, please cite:

```bibtex
@software{neuroimpy,
  author = {Buchsbaum, Brad},
  title = {neuroimpy: Python Neuroimaging Library},
  year = {2024},
  url = {https://github.com/bbuchsbaum/neuroimpy}
}
```

## Related Projects

- [neuroim2](https://github.com/bbuchsbaum/neuroim2) - R package (neuroimpy is designed for compatibility)
- [NiBabel](https://nipy.org/nibabel/) - Neuroimaging file I/O
- [Nilearn](https://nilearn.github.io/) - Machine learning for neuroimaging
- [MNE-Python](https://mne.tools/) - MEG/EEG analysis

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

neuroimpy is inspired by and designed to be compatible with the R neuroim2 package. Special thanks to the neuroimaging community for feedback and contributions.
