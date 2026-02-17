# PyNeuroim Tutorial Notebooks

This directory contains Jupyter notebooks that demonstrate key functionality of the neuroimpy library. All notebooks have been tested and verified to execute correctly.

## Available Notebooks

### 1. image_volumes.ipynb
**Purpose**: Introduction to working with 3D neuroimaging volumes
- Creating and manipulating `NeuroVol` objects
- Arithmetic operations on volumes
- Converting between dense, sparse, and logical representations
- Basic volume visualization

### 2. neuro_vectors.ipynb
**Purpose**: Working with 4D neuroimaging data (e.g., time series)
- Creating and manipulating `NeuroVec` objects
- Extracting time series from specific voxels
- Concatenating multiple vectors
- Memory-mapped vectors for large datasets

### 3. regions_of_interest.ipynb
**Purpose**: Creating and using regions of interest (ROIs)
- Creating spherical, square, and cuboid ROIs
- Converting ROIs to sparse representations
- Using searchlight analysis
- Clustered searchlight approaches

### 4. pipelines.ipynb
**Purpose**: Building analysis pipelines
- Connected components analysis
- Splitting data by clusters
- Parallel processing with searchlight
- Combining multiple analysis steps

## Running the Notebooks

All notebooks are self-contained and generate their own example data. No external data files are required.

### Prerequisites
```bash
pip install neuroimpy numpy scipy
```

### Quick Start
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to this directory and open any notebook

3. Run all cells in order (Cell → Run All)

## Notebook Features

### Self-Contained Examples
Each notebook generates its own synthetic data, making them immediately runnable without downloading external datasets.

### Consistent Structure
All notebooks follow a similar structure:
1. Imports and setup
2. Data generation
3. Core functionality demonstration
4. Practical examples

### Key Patterns

**Import Convention**: All notebooks use the standard import:
```python
import neuroimpy as pn
```

**Data Generation**: Example volumes are created with:
```python
space_3d = pn.NeuroSpace(dim=(64, 64, 25), spacing=(3.5, 3.5, 5.0))
data = np.random.randn(64, 64, 25)
vol = pn.DenseNeuroVol(data, space_3d)
```

## Testing the Notebooks

A test script is provided to verify all notebooks execute correctly:

```bash
python test_notebooks_simple.py
```

This script tests key functionality from each notebook and reports any issues.

## Common Operations

### Creating a Volume
```python
import neuroimpy as pn
import numpy as np

# Define spatial dimensions
space = pn.NeuroSpace(
    dim=(64, 64, 25),  # dimensions in voxels
    spacing=(3.5, 3.5, 5.0),  # voxel size in mm
    origin=(-110.5, -88.9342, -42.75)  # origin in world coordinates
)

# Create data and volume
data = np.random.randn(64, 64, 25)
vol = pn.DenseNeuroVol(data, space)
```

### Extracting Time Series
```python
# Create 4D data
space_4d = pn.NeuroSpace(dim=(64, 64, 25, 100))
data_4d = np.random.randn(64, 64, 25, 100)
vec = pn.DenseNeuroVec(data_4d, space_4d)

# Extract time series at voxel (10, 10, 10)
ts = vec.series(10, 10, 10)
```

### Creating ROIs
```python
# Spherical ROI
sphere = pn.spherical_roi(vol, center=[20, 20, 10], radius=5)

# Searchlight iterator
from neuroimpy import searchlight
mask = vol > 0.2
mask_vol = pn.LogicalNeuroVol(mask, vol.space)
rois = searchlight(mask_vol, radius=5)
```

## Troubleshooting

### Import Errors
Ensure neuroimpy is installed:
```bash
pip install -e /path/to/neuroimpy
```

### Memory Issues
For large datasets, use memory-mapped arrays:
```python
vec = pn.file_backed_neurovec("data.npy", space_4d, mode="r")
```

### Performance
Use sparse representations when working with ROIs:
```python
sparse_roi = roi.as_sparse()
```

## Next Steps

After exploring these notebooks:
1. Check the API documentation for detailed function references
2. Explore the test suite for more usage examples
3. Try applying these techniques to your own neuroimaging data

## Contributing

If you find issues or have suggestions for improving these notebooks:
1. Open an issue on the project repository
2. Submit a pull request with improvements
3. Share your use cases and examples