# neuroim

A modern Python library for neuroimaging data analysis, providing efficient data structures and algorithms for working with 3D, 4D, and 5D+ brain imaging data.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/bbuchsbaum/neuroim-python/actions/workflows/tests.yml/badge.svg)](https://github.com/bbuchsbaum/neuroim-python/actions/workflows/tests.yml)
[![Scenario evidence](https://img.shields.io/badge/scenario%20evidence-make%20verify--evidence-green)](docs/scenarios.md)

> ⚠️ **Experimental — release candidate (0.3.0rc1).** Usable,
> documented, and tested, but the public API is **not frozen until
> 1.0** and may change between minors. Binding terms:
> [STABILITY.md](STABILITY.md). Known gaps: [CHANGELOG.md](CHANGELOG.md).
> Do not pin a production pipeline to the native API yet.

**On-ramp:** [docs/quickstart.md](docs/quickstart.md) — four canonical patterns shown side-by-side against raw `nibabel`+`numpy`, each verdict pinned to a runnable scenario.

## Features

- **Spatial contracts**: Keep image data tied to shape, affine, voxel spacing, and coordinate transforms.
- **Analysis-ready containers**: Work with 3D volumes, 4D time series, and 5D+ feature stacks.
- **ROI and searchlight workflows**: Extract validated time-by-voxel matrices and neighborhood summaries.
- **Memory-aware representations**: Use dense, sparse, mapped, and file-backed data structures.
- **Interoperable I/O**: Read and write NIfTI/AFNI data while preserving a typed neuroim surface.
- **Public provenance contract**: Receipts written by `to_nibabel` survive the file boundary in a documented NIfTI extension — readable by [10 lines of `nibabel` alone](docs/spec/receipt-nifti-extension.md), no `neuroim` import required.
- **NumPy/SciPy performance**: Build on the scientific Python stack without losing spatial metadata.

## Installation

### From PyPI (coming soon)

```bash
pip install neuroim
```

### From Source

```bash
git clone https://github.com/bbuchsbaum/neuroim-python.git
cd neuroim-python
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/bbuchsbaum/neuroim-python.git
cd neuroim-python
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import neuroim as ni

bold = ni.read_image("golden_tests/fixtures/tiny_bold.nii.gz")
mask = ni.read_image("golden_tests/fixtures/tiny_mask.nii.gz", type="vol")
bold.space.compatible_with(mask.space)

roi = ni.spherical_roi(mask, centroid=(4, 4, 2), radius=2)
extraction = bold.series_roi(roi)
# extraction is an ROIExtractionResult: values, coords, space, and a content-addressable Receipt.

mean_data = np.zeros(mask.shape, dtype=np.float32)
mean_data[tuple(roi.coords.T)] = extraction.values.mean(axis=0)
mean_map = ni.NeuroVol.from_array(mean_data, space=mask.space)

out = mean_map.to_nibabel()
out.shape, out.affine.tolist()
```

## Core Data Structures

### NeuroVol - 3D Brain Volumes

```python
space = ni.NeuroSpace(dim=(91, 109, 91), spacing=(2, 2, 2))
vol = ni.NeuroVol.from_array(data_3d, space)

# Operations
smoothed = ni.gaussian_blur(vol, sigma=2)
mask = vol > vol.mean()
masked_vol = vol[mask]
```

### NeuroVec - 4D Time Series

```python
space_4d = ni.NeuroSpace(dim=(64, 64, 40, 200))
fmri = ni.NeuroVec.from_array(data_4d, space_4d)

# Extract volumes at specific timepoints
vol_t0 = fmri[..., 0]  # First volume
vol_t10 = fmri[..., 10]  # 11th volume

# Time series operations
mean_vol = fmri.mean(axis=3)  # Mean across time
```

### NeuroHyperVec - 5D+ Multi-Feature Data

```python
# Create 5D data (e.g., multi-echo fMRI with 3 echoes)
space_5d = ni.NeuroSpace(dim=(64, 64, 40, 200, 3))
multi_echo = ni.NeuroHyperVec(data_5d, space_5d)

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
roi_sphere = ni.spherical_roi(vol, centroid=[45, 54, 45], radius=10)

# Cuboid ROI  
roi_cube = ni.cuboid_roi(vol, centroid=[45, 54, 45], surround=[10, 10, 10])

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
results = ni.searchlight(fmri, radius=3, method=pattern_similarity)

# Run searchlight with parallel processing for faster computation
results_parallel = ni.searchlight(
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
smoothed = ni.gaussian_blur(vol, sigma=2)

# Bilateral filter (edge-preserving)
bilateral = ni.bilateral_filter(vol, spatial_sigma=2, range_sigma=0.5)

# Guided filter
guided = ni.guided_filter(vol, guide=mask, radius=3, epsilon=0.1)
```

### Memory-Efficient Operations

```python
# Sparse representation for masked data
mask = ni.LogicalNeuroVol(brain_mask, space)
sparse_fmri = fmri.to_sparse(mask)

# Memory-mapped for huge datasets
big_data = ni.MappedNeuroVec("huge_fmri.dat", space_4d)
chunk = big_data[30:40, 30:40, 20:30, :]  # Load only what's needed
```

## Documentation

The documentation site is built with Quarto from `docs/`.

- [Documentation source](docs/)
- [Tutorials](docs/tutorials/)
- [API reference](docs/reference/)
- [Porting and migration notes](docs/porting/)

## Requirements

- Python 3.8+
- NumPy >= 1.20
- SciPy >= 1.7
- NiBabel >= 3.0 (for NIFTI I/O)
- scikit-learn >= 0.24
- h5py >= 3.0 (for HDF5 support)

## Contributing

We welcome contributions. For local development:

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

If you use neuroim in your research, please cite:

```bibtex
@software{neuroim,
  author = {Buchsbaum, Brad},
  title = {neuroim: Python Neuroimaging Library},
  year = {2024},
  url = {https://github.com/bbuchsbaum/neuroim-python}
}
```

## Related Projects

- [neuroim2](https://github.com/bbuchsbaum/neuroim2) - related R package and migration reference
- [NiBabel](https://nipy.org/nibabel/) - Neuroimaging file I/O
- [Nilearn](https://nilearn.github.io/) - Machine learning for neuroimaging
- [MNE-Python](https://mne.tools/) - MEG/EEG analysis

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

neuroim builds on ideas from neuroim2 and the broader neuroimaging community. Special thanks to users and contributors who test these workflows across real analyses.
