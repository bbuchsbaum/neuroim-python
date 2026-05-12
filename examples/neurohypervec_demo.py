#!/usr/bin/env python
"""
NeuroHyperVec Demo: Working with 5D+ Neuroimaging Data

This script demonstrates the key features of NeuroHyperVec for handling
neuroimaging data with additional feature dimensions beyond the standard
4D (x, y, z, time) structure.
"""

import numpy as np
import neuroim as pn
import tempfile
import os


def demo_basic_construction():
    """Demonstrate basic NeuroHyperVec construction."""
    print("=== Basic NeuroHyperVec Construction ===")
    
    # Create 5D data: 10x10x10 spatial, 20 timepoints, 5 features
    data = np.random.randn(10, 10, 10, 20, 5)
    space = pn.NeuroSpace(dim=(10, 10, 10, 20, 5))
    
    hvec = pn.DenseNeuroHyperVec(data, space, label="demo_data")
    
    print(f"Shape: {hvec.shape}")
    print(f"Spatial shape: {hvec.spatial_shape}")
    print(f"Number of timepoints: {hvec.n_timepoints}")
    print(f"Number of features: {hvec.n_features}")
    print()


def demo_multi_echo_fmri():
    """Demonstrate multi-echo fMRI use case."""
    print("=== Multi-Echo fMRI Example ===")
    
    # Simulate 3-echo fMRI data
    n_echoes = 3
    echo_times = np.array([15.0, 30.0, 45.0])  # milliseconds
    
    # Create synthetic multi-echo data
    space = pn.NeuroSpace(dim=(32, 32, 20, 100, n_echoes))
    
    # Simulate echo data with decreasing SNR
    base_signal = np.random.randn(32, 32, 20, 100, 1)
    echo_data = np.zeros((32, 32, 20, 100, n_echoes))
    
    for i in range(n_echoes):
        # Add echo-dependent noise
        noise_level = 0.1 * (i + 1)
        echo_data[..., i] = base_signal[..., 0] + noise_level * np.random.randn(32, 32, 20, 100)
    
    multi_echo = pn.DenseNeuroHyperVec(echo_data, space, label="Multi-Echo fMRI")
    
    # Optimal echo combination
    weights = echo_times / echo_times.sum()
    print(f"Echo weights: {weights}")
    
    combined = multi_echo.weighted_mean_features(weights)
    print(f"Combined data shape: {combined.shape}")
    print(f"Combined data type: {type(combined).__name__}")
    print()


def demo_spectral_analysis():
    """Demonstrate spectral analysis use case."""
    print("=== Spectral Analysis Example ===")
    
    # Define frequency bands
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    # Create data for frequency band analysis
    n_bands = len(freq_bands)
    space = pn.NeuroSpace(dim=(20, 20, 15, 50, n_bands))
    
    # Simulate band power data
    band_power = np.random.exponential(scale=2.0, size=(20, 20, 15, 50, n_bands))
    
    spectral = pn.DenseNeuroHyperVec(band_power, space, label="Frequency Bands")
    
    # Extract alpha band (index 2)
    alpha_power = spectral[:, :, :, :, 2]
    print(f"Alpha power shape: {alpha_power.shape}")
    print(f"Alpha power type: {type(alpha_power).__name__}")
    
    # Find dominant frequency band at each voxel/time
    def find_dominant_band(power_vec):
        return np.argmax(power_vec)
    
    dominant_bands = spectral.apply_feature_func(find_dominant_band)
    print(f"Dominant bands shape: {dominant_bands.shape}")
    
    # Average power across all bands
    mean_power = spectral.mean_features()
    print(f"Mean power shape: {mean_power.shape}")
    print()


def demo_sparse_hypervec():
    """Demonstrate sparse NeuroHyperVec."""
    print("=== Sparse NeuroHyperVec Example ===")
    
    # Create a brain mask
    mask_data = np.zeros((30, 30, 20), dtype=bool)
    # Simple sphere mask
    center = np.array([15, 15, 10])
    for i in range(30):
        for j in range(30):
            for k in range(20):
                if np.linalg.norm([i, j, k] - center) < 8:
                    mask_data[i, j, k] = True
    
    mask = pn.LogicalNeuroVol(mask_data, pn.NeuroSpace(dim=(30, 30, 20)))
    n_voxels = mask.sum
    print(f"Number of voxels in mask: {n_voxels}")
    
    # Create sparse data: 4 features, 60 timepoints
    sparse_data = np.random.randn(4, 60, n_voxels)
    space = pn.NeuroSpace(dim=(30, 30, 20, 60, 4))
    
    sparse_hvec = pn.SparseNeuroHyperVec(sparse_data, mask, space)
    print(f"Sparse data shape: {sparse_hvec.data.shape}")
    
    # Extract series for a voxel in the mask
    series = sparse_hvec.series([15, 15, 10])
    print(f"Series shape: {series.shape}")
    
    # Convert to dense if needed
    dense_hvec = sparse_hvec.as_dense()
    print(f"Dense shape: {dense_hvec.shape}")
    print()


def demo_io_operations():
    """Demonstrate I/O operations."""
    print("=== I/O Operations Example ===")
    
    # Create test data
    space = pn.NeuroSpace(dim=(10, 10, 10, 20, 3))
    data = np.random.randn(10, 10, 10, 20, 3)
    hvec = pn.DenseNeuroHyperVec(data, space, label="test_io")
    
    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_name = tmp.name
        
    try:
        # Save
        pn.write_neurohypervec(hvec, tmp_name)
        print(f"Saved to: {tmp_name}")
        
        # Load
        loaded = pn.read_neurohypervec(tmp_name)
        print(f"Loaded shape: {loaded.shape}")
        print(f"Data preserved: {np.allclose(hvec.data, loaded.data)}")
    finally:
        # Clean up
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    print()


def demo_feature_operations():
    """Demonstrate various feature operations."""
    print("=== Feature Operations Example ===")
    
    # Create data with 6 features
    space = pn.NeuroSpace(dim=(15, 15, 10, 30, 6))
    data = np.random.randn(15, 15, 10, 30, 6)
    
    # Make features have different scales
    for i in range(6):
        data[..., i] *= (i + 1)
    
    hvec = pn.DenseNeuroHyperVec(data, space)
    
    # Select subset of features
    subset = hvec.select_features([0, 2, 4])
    print(f"Subset shape: {subset.shape}")
    
    # Apply custom function across features
    def compute_range(feature_vec):
        return np.max(feature_vec) - np.min(feature_vec)
    
    feature_range = hvec.apply_feature_func(compute_range)
    print(f"Feature range shape: {feature_range.shape}")
    
    # Standard deviation across features at first timepoint
    std_vol = hvec.std_features(time_idx=0)
    print(f"Std volume shape: {std_vol.shape}")
    
    # Mean across both time and features
    mean_vol = hvec.mean_time_features()
    print(f"Mean volume shape: {mean_vol.shape}")
    print()


def demo_concatenation():
    """Demonstrate feature concatenation."""
    print("=== Feature Concatenation Example ===")
    
    # Create two hypervectors with different features
    space1 = pn.NeuroSpace(dim=(10, 10, 10, 25, 3))
    space2 = pn.NeuroSpace(dim=(10, 10, 10, 25, 2))
    
    data1 = np.random.randn(10, 10, 10, 25, 3)
    data2 = np.random.randn(10, 10, 10, 25, 2)
    
    hvec1 = pn.DenseNeuroHyperVec(data1, space1, label="features_1-3")
    hvec2 = pn.DenseNeuroHyperVec(data2, space2, label="features_4-5")
    
    # Concatenate along feature dimension
    combined = pn.concat_features([hvec1, hvec2])
    print(f"Combined shape: {combined.shape}")
    print(f"Total features: {combined.n_features}")
    print()


def main():
    """Run all demonstrations."""
    print("NeuroHyperVec Demonstration")
    print("=" * 50)
    print()
    
    demo_basic_construction()
    demo_multi_echo_fmri()
    demo_spectral_analysis()
    demo_sparse_hypervec()
    demo_io_operations()
    demo_feature_operations()
    demo_concatenation()
    
    print("Demonstration complete!")


if __name__ == "__main__":
    main()