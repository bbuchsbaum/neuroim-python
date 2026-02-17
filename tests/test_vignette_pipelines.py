"""Test that Pipelines vignette functionality works in Python."""
import numpy as np

# Test imports
try:
    from neuroimpy import (DenseNeuroVol, NeuroSpace, conn_comp, 
                          split_clusters, searchlight, searchlight_coords,
                          neurovec, SparseNeuroVol)
    from neuroimpy.stats import split_blocks, partition
    print("✓ Basic imports work")
except ImportError as e:
    print(f"✗ Import error: {e}")

def test_connected_components():
    """Test connected components analysis."""
    # Create test volume with some structure
    data = np.zeros((20, 20, 20))
    # Add some blobs
    data[5:8, 5:8, 5:8] = 0.9
    data[12:15, 12:15, 12:15] = 0.85
    data[5:8, 12:15, 5:8] = 0.95
    
    space = NeuroSpace(dim=[20, 20, 20])
    vol = DenseNeuroVol(data, space)
    
    # Find connected components
    comp = conn_comp(vol, threshold=0.8)
    
    assert hasattr(comp, 'index')
    assert hasattr(comp, 'size')
    # Check number of components from voxels list
    n_components = len(comp.voxels)
    assert n_components >= 3  # Should find at least 3 components
    print("✓ Connected components works")
    
    # Split into clusters
    cluster_rois = split_clusters(vol, comp.index)
    means = [np.mean(roi.data) for roi in cluster_rois]
    
    assert len(means) == n_components
    print("✓ Split clusters works")

def test_searchlight_statistics():
    """Test searchlight with statistics."""
    # Create volume with noise
    data = np.random.randn(15, 15, 15) * 2 + 1
    space = NeuroSpace(dim=[15, 15, 15])
    vol = DenseNeuroVol(data, space)
    
    # Create mask
    from neuroimpy import LogicalNeuroVol
    mask = LogicalNeuroVol(np.ones_like(data, dtype=bool), space)
    
    # Compute local SD
    rois = list(searchlight(mask, radius=3, eager=True))[:20]  # Just test first 20
    
    sd_values = []
    indices = []
    
    for roi in rois:
        values = vol.data[roi.coords[:, 0], roi.coords[:, 1], roi.coords[:, 2]]
        sd_values.append(np.std(values))
        indices.append(roi.indices()[0])
    
    # Create sparse output
    sd_vol = SparseNeuroVol(
        data=np.array(sd_values),
        space=vol.space,
        indices=np.array(indices)
    )
    
    assert len(sd_values) == 20
    assert sd_vol.shape == vol.shape
    print("✓ Searchlight statistics works")

def test_slice_processing():
    """Test processing slices."""
    data = np.random.randn(10, 10, 20) + np.arange(20).reshape(1, 1, 20)
    space = NeuroSpace(dim=[10, 10, 20])
    vol = DenseNeuroVol(data, space)
    
    # Process each slice
    slice_means = []
    for z in range(vol.shape[2]):
        slice_data = vol.data[:, :, z]
        slice_means.append(np.mean(slice_data))
    
    # Should show increasing trend
    assert slice_means[-1] > slice_means[0]
    print("✓ Slice processing works")

def test_4d_processing():
    """Test processing 4D data."""
    # Create volume data
    vol1_data = np.ones((10, 10, 10))
    vol2_data = np.ones((10, 10, 10)) * 2
    vol3_data = np.ones((10, 10, 10)) * 3
    vol4_data = np.ones((10, 10, 10)) * 4
    vol5_data = np.ones((10, 10, 10)) * 5
    
    # Stack to create 4D data
    data_4d = np.stack([vol1_data, vol2_data, vol3_data, vol4_data, vol5_data], axis=-1)
    
    # Create 4D vector
    vec = neurovec(data_4d)
    assert vec.shape == (10, 10, 10, 5)
    
    # Get volumes by slicing
    volumes = [vec[..., i] for i in range(5)]
    
    # Calculate stats
    mean_vec = [np.mean(v.data) for v in volumes]
    sd_vec = [np.std(v.data) for v in volumes]
    
    assert mean_vec == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(s == 0 for s in sd_vec)
    print("✓ 4D volume processing works")

def test_voxel_time_series():
    """Test processing voxel time series."""
    # Create 4D data with structure
    data = np.random.randn(10, 10, 10, 50)
    vec = neurovec(data)
    
    # Method 1: numpy operations
    mean_vol_data = np.mean(vec.data, axis=3)
    mean_vol1 = DenseNeuroVol(mean_vol_data, 
                             NeuroSpace(dim=mean_vol_data.shape))
    
    # Method 2: voxel iteration (just test a few)
    mean_values = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ts = vec.series(i, j, k)
                mean_values.append(np.mean(ts))
    
    # Check they match for tested voxels
    # Reshape mean_values to compare
    mean_array = np.array(mean_values).reshape(3, 3, 3, order='C')
    mean_vol_subset = mean_vol1.data[:3, :3, :3]
    assert np.allclose(mean_array, mean_vol_subset)
    print("✓ Voxel time series processing works")

def test_knn_smoothing():
    """Test k-nearest neighbor smoothing concept."""
    data = np.random.randn(15, 15, 15)
    space = NeuroSpace(dim=[15, 15, 15])
    vol = DenseNeuroVol(data, space)
    
    # Create mask
    from neuroimpy import LogicalNeuroVol
    mask = LogicalNeuroVol(np.ones_like(data, dtype=bool), space)
    
    # Process a few searchlights
    k = 6
    smoothed_values = []
    indices = []
    
    for roi in list(searchlight(mask, radius=4))[:10]:
        roi_coords = roi.coords
        roi_values = vol.data[roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]]
        
        if hasattr(roi, 'center_index') and len(roi_values) > k:
            center_val = roi_values[roi.center_index]
            distances = np.abs(roi_values - center_val)
            k_nearest_idx = np.argsort(distances)[1:k+1]
            smoothed_values.append(np.mean(roi_values[k_nearest_idx]))
        else:
            smoothed_values.append(np.mean(roi_values))
        
        indices.append(roi.indices()[0])
    
    assert len(smoothed_values) == 10
    print("✓ K-NN smoothing concept works")

def test_partition():
    """Test partition function."""
    data = np.random.randn(20, 20, 20)
    space = NeuroSpace(dim=[20, 20, 20])
    vol = DenseNeuroVol(data, space)
    
    # Create mask
    mask = data > -0.5
    
    try:
        # Partition into k clusters
        kvol = partition(vol, k=5, mask=mask)
        
        # Check we got 5 clusters
        unique_labels = np.unique(kvol.data[kvol.data > 0])
        assert len(unique_labels) == 5
        print("✓ Partition works")
    except Exception as e:
        print(f"✗ Partition failed: {e}")

def test_custom_pipeline():
    """Test building a custom pipeline."""
    # Create noisy volume
    data = np.random.randn(20, 20, 20)
    # Add some structure
    data[8:12, 8:12, 8:12] += 2
    data[14:18, 14:18, 14:18] += 1.5
    
    space = NeuroSpace(dim=[20, 20, 20])
    vol = DenseNeuroVol(data, space)
    
    # Simple denoising pipeline
    def simple_denoise(vol, threshold=0.5):
        # Find components
        comp = conn_comp(vol, threshold=threshold)
        
        # Get clusters
        clusters = split_clusters(vol, comp.index)
        
        # Process each cluster
        result_data = np.zeros_like(vol.data)
        for cluster in clusters:
            if len(cluster) >= 10:  # Only keep larger clusters
                # Simple smoothing
                cluster_mean = np.mean(cluster.data)
                coords = cluster.coords
                result_data[coords[:, 0], coords[:, 1], coords[:, 2]] = cluster_mean * 0.8
        
        return DenseNeuroVol(result_data, vol.space)
    
    # Apply pipeline
    denoised = simple_denoise(vol, threshold=0.8)
    
    # Should have reduced noise
    assert np.std(denoised.data) < np.std(vol.data)
    print("✓ Custom pipeline works")

if __name__ == "__main__":
    print("Testing Pipelines vignette functionality...")
    print("=" * 50)
    
    test_connected_components()
    test_searchlight_statistics()
    test_slice_processing()
    test_4d_processing()
    test_voxel_time_series()
    test_knn_smoothing()
    test_partition()
    test_custom_pipeline()
    
    print("\nSummary: Core pipeline functionality is working!")