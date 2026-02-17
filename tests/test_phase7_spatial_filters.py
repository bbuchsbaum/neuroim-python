"""Tests for Phase 7: Spatial Filtering.

Direct translation of R's neuroim2 tests for spatial filtering functionality.
"""

import pytest
import numpy as np
from scipy import ndimage

from neuroimpy import (
    # Core classes
    DenseNeuroVol, LogicalNeuroVol, DenseNeuroVec, NeuroSpace,
    # Kernel classes
    Kernel, gaussian_kernel, spherical_kernel, box_kernel, embed_kernel,
    # Filtering functions
    gaussian_blur, guided_filter, bilateral_filter, bilateral_filter_vec
)


class TestKernel:
    """Test Kernel class functionality."""
    
    def test_kernel_creation(self):
        """Test basic kernel creation."""
        width = (3, 3, 3)
        weights = np.ones(27) / 27  # Uniform weights
        voxels = np.array([[i, j, k] for i in [-1, 0, 1] 
                          for j in [-1, 0, 1] 
                          for k in [-1, 0, 1]])
        coords = voxels.astype(float)
        
        kernel = Kernel(width, weights, voxels, coords)
        
        assert kernel.width == (3, 3, 3)
        assert len(kernel.weights) == 27
        assert kernel.voxels.shape == (27, 3)
        assert kernel.coords.shape == (27, 3)
    
    def test_kernel_get_voxels(self):
        """Test getting voxel locations with offset."""
        width = (3, 3, 3)
        weights = np.ones(27) / 27
        voxels = np.array([[i, j, k] for i in [-1, 0, 1] 
                          for j in [-1, 0, 1] 
                          for k in [-1, 0, 1]])
        coords = voxels.astype(float)
        
        kernel = Kernel(width, weights, voxels, coords)
        
        # Without center
        vox1 = kernel.get_voxels()
        np.testing.assert_array_equal(vox1, voxels)
        
        # With center
        center = np.array([10, 20, 30])
        vox2 = kernel.get_voxels(center)
        expected = voxels + center
        np.testing.assert_array_equal(vox2, expected)
    
    def test_gaussian_kernel(self):
        """Test Gaussian kernel creation."""
        vdim = (1.0, 1.0, 1.0)  # Unit spacing
        kdim = (3, 3, 3)
        sigma = 1.0
        
        kernel = gaussian_kernel(vdim, kdim, sigma, normalize=True)
        
        assert kernel.width == (3, 3, 3)
        assert len(kernel.weights) == 27
        # Check weights sum to 1
        assert np.abs(np.sum(kernel.weights) - 1.0) < 1e-6
        
        # Center voxel should have highest weight
        center_idx = 13  # Middle of 27 elements
        assert kernel.weights[center_idx] == np.max(kernel.weights)
    
    def test_spherical_kernel(self):
        """Test spherical kernel creation."""
        vdim = (1.0, 1.0, 1.0)
        radius = 2.0
        
        kernel = spherical_kernel(vdim, radius)
        
        # Check that all voxels are within radius
        distances = np.sqrt(np.sum(kernel.coords**2, axis=1))
        assert np.all(distances <= radius)
        
        # Check weights sum to 1
        assert np.abs(np.sum(kernel.weights) - 1.0) < 1e-6
    
    def test_box_kernel(self):
        """Test box kernel creation."""
        kdim = (3, 3, 3)
        
        kernel = box_kernel(kdim)
        
        assert kernel.width == (3, 3, 3)
        assert len(kernel.weights) == 27
        # All weights should be equal
        assert np.allclose(kernel.weights, 1.0/27)
    
    def test_embed_kernel(self):
        """Test embedding kernel in space."""
        # Create simple space
        space = NeuroSpace((10, 10, 10), spacing=(1, 1, 1))
        
        # Create small kernel
        kernel = box_kernel((3, 3, 3))
        
        # Embed at center
        center_voxel = np.array([5, 5, 5])
        sparse_vol = embed_kernel(kernel, space, center_voxel)
        
        assert sparse_vol.shape == (10, 10, 10)
        assert len(sparse_vol.indices) == 27
        assert np.sum(sparse_vol.data) == pytest.approx(1.0)


class TestSpatialFilters:
    """Test spatial filtering functions."""
    
    def setup_method(self):
        """Create test data."""
        # Create test volume
        self.space = NeuroSpace((20, 20, 20), spacing=(1, 1, 1))
        
        # Create test data with a sphere
        data = np.zeros((20, 20, 20))
        center = np.array([10, 10, 10])
        radius = 5
        
        for i in range(20):
            for j in range(20):
                for k in range(20):
                    dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                    if dist <= radius:
                        data[i, j, k] = 1.0
        
        self.vol = DenseNeuroVol(data, self.space)
        self.mask = LogicalNeuroVol(data > 0, self.space)
    
    def test_gaussian_blur_basic(self):
        """Test basic Gaussian blur."""
        blurred = gaussian_blur(self.vol, sigma=2, window=1)
        
        assert isinstance(blurred, DenseNeuroVol)
        assert blurred.shape == self.vol.shape
        
        # Check that values are smoothed 
        # For a binary sphere, edges should be smoothed
        edge_mask = (self.vol.data > 0) & (self.vol.data < 1)
        if np.any(edge_mask):
            # If there are edge values, they should change
            assert not np.allclose(blurred.data[edge_mask], self.vol.data[edge_mask])
        else:
            # For pure binary, check that some smoothing occurred at boundaries
            assert np.any(blurred.data != self.vol.data)
        
        # Check that total energy is roughly preserved
        assert np.sum(blurred.data) == pytest.approx(np.sum(self.vol.data), rel=0.1)
    
    def test_gaussian_blur_with_mask(self):
        """Test Gaussian blur with mask."""
        blurred = gaussian_blur(self.vol, mask=self.mask, sigma=2, window=1)
        
        # Values outside mask should be unchanged
        outside_mask = ~self.mask.data
        np.testing.assert_array_equal(
            blurred.data[outside_mask],
            self.vol.data[outside_mask]
        )
    
    def test_gaussian_blur_parameters(self):
        """Test Gaussian blur parameter validation."""
        # Invalid window
        with pytest.raises(ValueError):
            gaussian_blur(self.vol, window=0)
        
        # Invalid sigma
        with pytest.raises(ValueError):
            gaussian_blur(self.vol, sigma=-1)
    
    def test_guided_filter_basic(self):
        """Test basic guided filter."""
        filtered = guided_filter(self.vol, radius=2, epsilon=0.1)
        
        assert isinstance(filtered, DenseNeuroVol)
        assert filtered.shape == self.vol.shape
        
        # Guided filter should preserve edges better than Gaussian
        # Create edge detector
        edges_orig = np.abs(ndimage.sobel(self.vol.data, axis=0))
        edges_filt = np.abs(ndimage.sobel(filtered.data, axis=0))
        
        # Some edges should be preserved
        assert np.max(edges_filt) > 0
    
    def test_guided_filter_parameters(self):
        """Test guided filter parameter validation."""
        # Invalid radius
        with pytest.raises(ValueError):
            guided_filter(self.vol, radius=0)
        
        # Invalid epsilon
        with pytest.raises(ValueError):
            guided_filter(self.vol, epsilon=-1)
    
    def test_bilateral_filter_basic(self):
        """Test basic bilateral filter."""
        filtered = bilateral_filter(
            self.vol, 
            spatial_sigma=2, 
            intensity_sigma=0.5,
            window=1
        )
        
        assert isinstance(filtered, DenseNeuroVol)
        assert filtered.shape == self.vol.shape
        
        # Should smooth while preserving edges
        assert np.max(filtered.data) <= np.max(self.vol.data)
    
    def test_bilateral_filter_with_mask(self):
        """Test bilateral filter with mask."""
        filtered = bilateral_filter(
            self.vol,
            mask=self.mask,
            spatial_sigma=2,
            intensity_sigma=0.5,
            window=1
        )
        
        # Values outside mask should be unchanged
        outside_mask = ~self.mask.data
        np.testing.assert_array_equal(
            filtered.data[outside_mask],
            self.vol.data[outside_mask]
        )
    
    def test_bilateral_filter_parameters(self):
        """Test bilateral filter parameter validation."""
        # Invalid window
        with pytest.raises(ValueError):
            bilateral_filter(self.vol, window=0)
        
        # Invalid spatial sigma
        with pytest.raises(ValueError):
            bilateral_filter(self.vol, spatial_sigma=-1)
        
        # Invalid intensity sigma
        with pytest.raises(ValueError):
            bilateral_filter(self.vol, intensity_sigma=-1)
    
    def test_bilateral_filter_vec(self):
        """Test bilateral filter on NeuroVec."""
        # Create 4D data
        data_4d = np.stack([self.vol.data] * 3, axis=-1)
        vec = DenseNeuroVec(data_4d, self.space.add_dim(1, size=3))
        
        filtered_vec = bilateral_filter_vec(
            vec,
            mask=self.mask,
            spatial_sigma=2,
            intensity_sigma=0.5,
            window=1
        )
        
        assert isinstance(filtered_vec, DenseNeuroVec)
        assert filtered_vec.shape == vec.shape
        
        # Each volume should be filtered
        for i in range(3):
            vol_i = filtered_vec.vols(i)
            assert np.max(vol_i.data) <= np.max(self.vol.data)


class TestFilterComparison:
    """Test comparing different filters."""
    
    def setup_method(self):
        """Create noisy test data."""
        self.space = NeuroSpace((30, 30, 30), spacing=(1, 1, 1))
        
        # Create smooth signal
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, 30),
            np.linspace(-1, 1, 30),
            np.linspace(-1, 1, 30),
            indexing='ij'
        )
        signal = np.exp(-(x**2 + y**2 + z**2) / 0.2)
        
        # Add noise
        noise = np.random.randn(30, 30, 30) * 0.1
        data = signal + noise
        
        self.vol = DenseNeuroVol(data, self.space)
        self.signal = signal
    
    def test_filter_comparison(self):
        """Compare different filters on noisy data."""
        # Apply different filters
        gauss_filtered = gaussian_blur(self.vol, sigma=1.5, window=2)
        guided_filtered = guided_filter(self.vol, radius=3, epsilon=0.01)
        bilateral_filtered = bilateral_filter(
            self.vol, 
            spatial_sigma=1.5, 
            intensity_sigma=0.2,
            window=2
        )
        
        # Calculate errors vs true signal
        gauss_error = np.mean((gauss_filtered.data - self.signal)**2)
        guided_error = np.mean((guided_filtered.data - self.signal)**2)
        bilateral_error = np.mean((bilateral_filtered.data - self.signal)**2)
        
        # All should reduce noise
        orig_error = np.mean((self.vol.data - self.signal)**2)
        assert gauss_error < orig_error
        assert guided_error < orig_error
        assert bilateral_error < orig_error


# Run tests
if __name__ == "__main__":
    pytest.main([__file__])