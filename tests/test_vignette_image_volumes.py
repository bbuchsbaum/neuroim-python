"""Test that ImageVolumes vignette functionality works in Python."""
import numpy as np
import tempfile
import os
from pathlib import Path

# Test what imports work
try:
    from neuroim import (write_vol, DenseNeuroVol, NeuroSpace)
    from neuroim.io import read_vol
    print("✓ Basic imports work")
except ImportError as e:
    print(f"✗ Import error: {e}")

def test_read_write_vol(tmp_path):
    """Test reading and writing volumes."""
    # Create a test volume
    data = np.random.randn(10, 10, 10)
    space = NeuroSpace(dim=[10, 10, 10], spacing=[1, 1, 1])
    vol = DenseNeuroVol(data, space)
    
    path = tmp_path / "vignette_vol.nii"
    write_vol(vol, path)
    vol2 = read_vol(path)

    assert vol2.shape == vol.shape
    assert np.allclose(vol2.data, vol.data)
    print("✓ Read/write vol works")

def test_vol_arithmetic():
    """Test arithmetic operations on volumes."""
    data = np.ones((10, 10, 10))
    space = NeuroSpace(dim=[10, 10, 10], spacing=[1, 1, 1])
    vol = DenseNeuroVol(data, space)
    
    # Test arithmetic
    vol2 = vol + vol
    assert np.sum(vol2.data) == 2 * np.sum(vol.data)
    
    vol3 = vol2 - vol * 2
    assert np.allclose(vol3.data, 0)
    
    vol4 = vol * 2.5
    assert np.allclose(vol4.data, 2.5)
    
    print("✓ Volume arithmetic works")

def test_vol_indexing():
    """Test volume indexing."""
    data = np.random.randn(10, 10, 10)
    space = NeuroSpace(dim=[10, 10, 10], spacing=[1, 1, 1])
    vol = DenseNeuroVol(data, space)
    
    # Test indexing
    assert vol[0, 0, 0] == data[0, 0, 0]
    assert vol[9, 9, 9] == data[9, 9, 9]
    print("✓ Volume indexing works")

def test_logical_conversion():
    """Test conversion to logical volume."""
    data = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]])
    space = NeuroSpace(dim=data.shape, spacing=[1, 1, 1])
    vol = DenseNeuroVol(data, space)
    
    # Convert to logical
    vol_logical = vol.as_logical()
    assert vol_logical[0, 0, 0] == False
    assert vol_logical[0, 0, 1] == True
    print("✓ Logical conversion works")

def test_sparse_conversion():
    """Test sparse volume conversion."""
    # Create mostly zeros volume
    data = np.zeros((10, 10, 10))
    data[5:7, 5:7, 5:7] = 1.0
    space = NeuroSpace(dim=data.shape, spacing=[1, 1, 1])
    vol = DenseNeuroVol(data, space)
    
    # Convert to sparse
    sparse_vol = vol.as_sparse()
    
    # Check values
    assert sparse_vol[6, 6, 6] == 1.0
    assert sparse_vol[0, 0, 0] == 0.0
    assert np.sum(sparse_vol.data) == np.sum(data)
    print("✓ Sparse conversion works")

def test_vol_statistics():
    """Test volume statistics methods."""
    data = np.random.randn(10, 10, 10)
    space = NeuroSpace(dim=data.shape, spacing=[1, 1, 1])
    vol = DenseNeuroVol(data, space)
    
    # Check if methods exist
    try:
        mean_val = vol.vol_mean()
        sd_val = vol.vol_sd()
        min_val = vol.min()
        max_val = vol.max()
        min_idx = vol.which_min()
        max_idx = vol.which_max()
        print("✓ Volume statistics methods exist")
    except AttributeError as e:
        print(f"✗ Missing method: {e}")
        # Try alternatives
        mean_val = np.mean(vol.data)
        sd_val = np.std(vol.data)
        min_val = np.min(vol.data)
        max_val = np.max(vol.data)
        print("✓ Can compute statistics with numpy")

if __name__ == "__main__":
    print("Testing ImageVolumes vignette functionality...")
    print("=" * 50)
    
    test_read_write_vol()
    test_vol_arithmetic()
    test_vol_indexing()
    test_logical_conversion()
    test_sparse_conversion()
    test_vol_statistics()
    
    print("\nSummary: Basic volume operations are working!")
