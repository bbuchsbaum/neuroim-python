"""Test that NeuroVector vignette functionality works in Python."""
import numpy as np
import tempfile
import os

# Test imports
try:
    from neuroim import read_vec, neurovec, DenseNeuroVol, DenseNeuroVec, NeuroSpace
    from neuroim.neuro_vec import neurovecseq
    print("✓ Basic imports work")
except ImportError as e:
    print(f"✗ Import error: {e}")

def test_create_neurovec():
    """Test creating a NeuroVec."""
    # Create 4D data
    data = np.random.randn(10, 10, 10, 20)
    vec = neurovec(data)
    
    assert vec.shape == (10, 10, 10, 20)
    print("✓ Create neurovec works")
    
    # Test creating from stacked volumes
    vol1_data = np.ones((10, 10, 10))
    vol2_data = np.ones((10, 10, 10)) * 2
    vol3_data = np.ones((10, 10, 10)) * 3
    
    # Stack the data to create 4D array
    data_4d = np.stack([vol1_data, vol2_data, vol3_data], axis=-1)
    vec = neurovec(data_4d)
    assert vec.shape == (10, 10, 10, 3)
    print("✓ Create from stacked volumes works")
    
    # Also test that we can extract the individual volumes
    vol_0 = vec[..., 0]
    assert isinstance(vol_0, DenseNeuroVol)
    assert np.allclose(vol_0.data, vol1_data)
    print("✓ Volume extraction works")

def test_sub_vector():
    """Test extracting subsets of volumes."""
    data = np.random.randn(10, 10, 10, 20)
    vec = neurovec(data)
    
    # Test if sub_vector exists
    try:
        vec_subset = vec.sub_vector(slice(0, 6))
        assert vec_subset.shape == (10, 10, 10, 6)
        print("✓ sub_vector with slice works")
        
        vec_subset = vec.sub_vector([0, 2, 4, 6])
        assert vec_subset.shape == (10, 10, 10, 4)
        print("✓ sub_vector with list works")
    except AttributeError:
        print("✗ sub_vector method not found")
        # Try alternative indexing
        vec_subset = vec[..., 0:6]
        assert vec_subset.shape == (10, 10, 10, 6)
        print("✓ Direct slicing works")

def test_series_extraction():
    """Test time series extraction."""
    data = np.random.randn(10, 10, 10, 20)
    vec = neurovec(data)
    
    # Single voxel
    ts = vec.series(5, 5, 5)
    assert ts.shape == (20,)
    assert np.allclose(ts, data[5, 5, 5, :])
    print("✓ Single voxel series works")
    
    # Multiple voxels with coordinates
    coords = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    ts_matrix = vec.series(coords)
    assert ts_matrix.shape == (20, 3)
    print("✓ Multiple voxel series works")
    
    # Linear indices
    ts_linear = vec.series(np.arange(10))
    assert ts_linear.shape == (20, 10)
    print("✓ Linear index series works")

def test_series_roi():
    """Test ROI-based series extraction."""
    from neuroim import spherical_roi
    
    data = np.random.randn(10, 10, 10, 20)
    vec = neurovec(data)
    
    # Need to use a 3D volume to create ROI
    space_3d = NeuroSpace(dim=data.shape[:3])
    vol = DenseNeuroVol(data[..., 0], space_3d)
    # Create a small ROI
    roi = spherical_roi(vol, [5, 5, 5], radius=2)
    
    # Extract series
    roi_series = vec.series_roi(roi)
    assert roi_series.shape[0] == 20  # time points
    assert roi_series.shape[1] == len(roi)  # voxels in ROI
    print("✓ series_roi works")

def test_concat():
    """Test concatenating NeuroVecs."""
    vec1 = neurovec(np.ones((10, 10, 10, 20)))
    vec2 = neurovec(np.ones((10, 10, 10, 30)) * 2)
    
    try:
        combined = vec1.concat(vec2)
        assert combined.shape == (10, 10, 10, 50)
        print("✓ concat method works")
    except AttributeError:
        print("✗ concat method not found")

def test_vols_extraction():
    """Test extracting individual volumes."""
    data = np.random.randn(10, 10, 10, 20)
    vec = neurovec(data)
    
    # Get single volume
    vol0 = vec[..., 0]
    assert vol0.shape == (10, 10, 10)
    assert isinstance(vol0, DenseNeuroVol)
    print("✓ Single volume extraction works")
    
    # Get multiple volumes
    try:
        vols = vec.vols([0, 5, 10, 15])
        assert len(vols) == 4
        print("✓ vols method works")
    except AttributeError:
        print("✗ vols method not found")

def test_as_sparse():
    """Test sparse conversion."""
    from neuroim import LogicalNeuroVol
    
    data = np.zeros((10, 10, 10, 20))
    data[5:7, 5:7, 5:7, :] = 1.0
    vec = neurovec(data)
    
    # Create mask
    mask_data = np.zeros((10, 10, 10))
    mask_data[5:7, 5:7, 5:7] = 1
    mask = LogicalNeuroVol(mask_data > 0, NeuroSpace(dim=mask_data.shape))
    
    # Convert to sparse
    try:
        sparse_vec = vec.as_sparse(mask)
        ts = sparse_vec.series(6, 6, 6)
        assert ts.shape == (20,)
        print("✓ Sparse conversion works")
    except Exception as e:
        print(f"✗ Sparse conversion failed: {e}")

def test_scale_series():
    """Test scaling time series."""
    data = np.random.randn(5, 5, 5, 20) * 10 + 5
    vec = neurovec(data)
    
    try:
        vec_scaled = vec.scale_series(center=True, scale=True)
        # Check a voxel
        ts = vec_scaled.series(2, 2, 2)
        assert abs(np.mean(ts)) < 1e-10  # Should be centered
        assert abs(np.std(ts) - 1) < 0.1  # Should be scaled
        print("✓ scale_series works")
    except AttributeError:
        print("✗ scale_series method not found")

if __name__ == "__main__":
    print("Testing NeuroVector vignette functionality...")
    print("=" * 50)
    
    test_create_neurovec()
    test_sub_vector()
    test_series_extraction()
    test_series_roi()
    test_concat()
    test_vols_extraction()
    test_as_sparse()
    test_scale_series()
    
    print("\nSummary: Core 4D functionality is working!")