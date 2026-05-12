"""
Test suite for image processing functions.

This module tests the image processing functionality in neuroim,
corresponding to the R neuroim2 test-imageproc.R tests.
"""

import pytest
import numpy as np
from neuroim import (
    NeuroSpace, DenseNeuroVol, LogicalNeuroVol,
    gaussian_blur, guided_filter, bilateral_filter
)
import tempfile
import nibabel as nib


class TestImageProcessing:
    """Test cases for image processing functionality."""
    
    @pytest.fixture
    def create_test_volume(self):
        """Create a test volume for image processing."""
        # Create a 32x32x32 volume with a sphere in the center
        space = NeuroSpace((32, 32, 32), spacing=(1, 1, 1))
        data = np.zeros((32, 32, 32))
        
        # Create a sphere of high intensity in the center
        center = np.array([16, 16, 16])
        radius = 8
        
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    if dist <= radius:
                        data[i, j, k] = 100
        
        # Add some noise
        np.random.seed(42)
        noise = np.random.normal(0, 5, data.shape)
        data += noise
        
        vol = DenseNeuroVol(data, space)
        return vol, space
    
    @pytest.fixture
    def create_test_mask(self):
        """Create a test mask."""
        space = NeuroSpace((32, 32, 32), spacing=(1, 1, 1))
        
        # Create mask that excludes edges
        mask_data = np.ones((32, 32, 32), dtype=bool)
        mask_data[:2, :, :] = False
        mask_data[-2:, :, :] = False
        mask_data[:, :2, :] = False
        mask_data[:, -2:, :] = False
        mask_data[:, :, :2] = False
        mask_data[:, :, -2:] = False
        
        mask = LogicalNeuroVol(mask_data, space)
        return mask
    
    def test_gaussian_blur_basic(self, create_test_volume):
        """Test basic Gaussian blur functionality."""
        vol, space = create_test_volume
        
        # Apply Gaussian blur with default parameters
        blurred = gaussian_blur(vol, sigma=2)
        
        assert isinstance(blurred, DenseNeuroVol)
        assert blurred.shape == vol.shape
        assert blurred.space == vol.space
        
        # Blurred image should have lower maximum intensity
        assert np.max(blurred.data) < np.max(vol.data)
        
        # Blurred image should have less variance
        assert np.var(blurred.data) < np.var(vol.data)
    
    def test_gaussian_blur_with_mask(self, create_test_volume, create_test_mask):
        """Test Gaussian blur with mask."""
        vol, space = create_test_volume
        mask = create_test_mask
        
        # Apply Gaussian blur with mask
        blurred = gaussian_blur(vol, mask=mask, sigma=2)
        
        assert isinstance(blurred, DenseNeuroVol)
        assert blurred.shape == vol.shape
        
        # Values outside mask should remain unchanged
        mask_indices = np.where(~mask.data)
        np.testing.assert_array_equal(blurred.data[mask_indices], vol.data[mask_indices])
    
    def test_gaussian_blur_different_sigmas(self, create_test_volume, create_test_mask):
        """Test Gaussian blur with different sigma values."""
        vol, space = create_test_volume
        mask = create_test_mask
        
        # Test with sigma=2
        g1 = gaussian_blur(vol, mask, sigma=2)
        
        # Test with sigma=8
        g2 = gaussian_blur(vol, mask, sigma=8)
        
        # Test with sigma=8 and window=3
        g3 = gaussian_blur(vol, mask, sigma=8, window=3)
        
        # All should return valid results
        assert g1 is not None
        assert g2 is not None
        assert g3 is not None
        
        # Higher sigma should produce more smoothing
        assert np.var(g2.data[mask.data]) < np.var(g1.data[mask.data])
        
        # Window parameter should affect the result
        assert not np.allclose(g2.data, g3.data)
    
    def test_gaussian_blur_edge_cases(self, create_test_volume):
        """Test Gaussian blur edge cases."""
        vol, space = create_test_volume
        
        # Test with minimum window size
        g1 = gaussian_blur(vol, sigma=1, window=1)
        assert g1 is not None
        
        # Test with large sigma
        g2 = gaussian_blur(vol, sigma=20, window=5)
        assert g2 is not None
        
        # Should produce very smooth result
        assert np.var(g2.data) < np.var(vol.data) / 10
    
    def test_gaussian_blur_invalid_params(self, create_test_volume):
        """Test Gaussian blur with invalid parameters."""
        vol, space = create_test_volume
        
        # Invalid window size
        with pytest.raises(ValueError, match="Window size must be at least 1"):
            gaussian_blur(vol, window=0)
        
        # Invalid sigma
        with pytest.raises(ValueError, match="Sigma must be positive"):
            gaussian_blur(vol, sigma=0)
        
        with pytest.raises(ValueError, match="Sigma must be positive"):
            gaussian_blur(vol, sigma=-1)
    
    def test_guided_filter_basic(self, create_test_volume):
        """Test basic guided filter functionality."""
        vol, space = create_test_volume
        
        # Apply guided filter
        filtered = guided_filter(vol, radius=4)
        
        assert isinstance(filtered, DenseNeuroVol)
        assert filtered.shape == vol.shape
        assert filtered.space == vol.space
        
        # Guided filter should preserve edges better than Gaussian blur
        # but still reduce noise
        assert np.var(filtered.data) < np.var(vol.data)
    
    def test_guided_filter_different_params(self, create_test_volume):
        """Test guided filter with different parameters."""
        vol, space = create_test_volume
        
        # Test with different radius values
        g1 = guided_filter(vol, radius=2)
        g2 = guided_filter(vol, radius=8)
        
        # Test with different epsilon values
        g3 = guided_filter(vol, radius=4, epsilon=0.1)
        g4 = guided_filter(vol, radius=4, epsilon=1.0)
        
        # All should return valid results
        assert g1 is not None
        assert g2 is not None
        assert g3 is not None
        assert g4 is not None
        
        # Larger radius should produce more smoothing
        assert np.var(g2.data) < np.var(g1.data)
        
        # Smaller epsilon should preserve more edges
        assert np.var(g3.data) > np.var(g4.data)
    
    def test_guided_filter_invalid_params(self, create_test_volume):
        """Test guided filter with invalid parameters."""
        vol, space = create_test_volume
        
        # Invalid radius
        with pytest.raises(ValueError, match="Radius must be at least 1"):
            guided_filter(vol, radius=0)
        
        # Invalid epsilon
        with pytest.raises(ValueError, match="Epsilon must be non-negative"):
            guided_filter(vol, epsilon=-1)
    
    def test_bilateral_filter_if_exists(self, create_test_volume):
        """Test bilateral filter if it exists."""
        vol, space = create_test_volume
        
        # Test bilateral filter (additional functionality in Python)
        try:
            filtered = bilateral_filter(vol, spatial_sigma=2, intensity_sigma=10)
            
            assert isinstance(filtered, DenseNeuroVol)
            assert filtered.shape == vol.shape
            assert filtered.space == vol.space
            
            # Should reduce noise while preserving edges
            assert np.var(filtered.data) < np.var(vol.data)
        except AttributeError:
            # bilateral_filter might not be imported if not implemented
            pytest.skip("bilateral_filter not available")
    
    def test_processing_preserves_data_type(self, create_test_volume):
        """Test that processing functions preserve data type properties."""
        vol, space = create_test_volume
        
        # Test Gaussian blur
        blurred = gaussian_blur(vol, sigma=2)
        assert blurred.data.dtype == np.float64 or blurred.data.dtype == np.float32
        
        # Test guided filter
        filtered = guided_filter(vol, radius=4)
        assert filtered.data.dtype == np.float64 or filtered.data.dtype == np.float32
    
    def test_processing_with_edge_volume(self):
        """Test processing with volumes containing edge values."""
        space = NeuroSpace((16, 16, 16), spacing=(1, 1, 1))
        
        # Create volume with extreme values
        data = np.zeros((16, 16, 16))
        data[7:9, 7:9, 7:9] = 1000  # Very high values in center
        data[0, 0, 0] = -1000  # Very low value at corner
        
        vol = DenseNeuroVol(data, space)
        
        # Both filters should handle extreme values
        blurred = gaussian_blur(vol, sigma=2)
        filtered = guided_filter(vol, radius=2)
        
        assert not np.any(np.isnan(blurred.data))
        assert not np.any(np.isinf(blurred.data))
        assert not np.any(np.isnan(filtered.data))
        assert not np.any(np.isinf(filtered.data))
    
    def test_compare_smoothing_methods(self, create_test_volume):
        """Compare different smoothing methods."""
        vol, space = create_test_volume
        
        # Apply different smoothing methods
        gauss_light = gaussian_blur(vol, sigma=2)
        gauss_heavy = gaussian_blur(vol, sigma=8)
        guided = guided_filter(vol, radius=4)
        
        # Calculate edge strength (gradient magnitude)
        def edge_strength(vol_data):
            grad_x = np.gradient(vol_data, axis=0)
            grad_y = np.gradient(vol_data, axis=1)
            grad_z = np.gradient(vol_data, axis=2)
            return np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        edges_original = edge_strength(vol.data)
        edges_gauss_light = edge_strength(gauss_light.data)
        edges_gauss_heavy = edge_strength(gauss_heavy.data)
        edges_guided = edge_strength(guided.data)
        
        # Guided filter should preserve edges better than heavy Gaussian blur
        edge_preservation_guided = np.sum(edges_guided) / np.sum(edges_original)
        edge_preservation_gauss = np.sum(edges_gauss_heavy) / np.sum(edges_original)
        
        assert edge_preservation_guided > edge_preservation_gauss
    
    def test_real_world_example(self):
        """Test with a more realistic brain-like structure."""
        # Create a simple brain-like phantom
        space = NeuroSpace((64, 64, 64), spacing=(1, 1, 1))
        data = np.zeros((64, 64, 64))
        
        # Create concentric spheres with different intensities (CSF, GM, WM)
        center = np.array([32, 32, 32])
        
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    if dist <= 20:  # White matter
                        data[i, j, k] = 100
                    elif dist <= 25:  # Gray matter
                        data[i, j, k] = 60
                    elif dist <= 30:  # CSF
                        data[i, j, k] = 20
        
        # Add noise
        np.random.seed(42)
        data += np.random.normal(0, 5, data.shape)
        
        vol = DenseNeuroVol(data, space)
        
        # Test different processing methods
        smoothed = gaussian_blur(vol, sigma=1.5)
        edge_preserved = guided_filter(vol, radius=3, epsilon=0.3)
        
        # Smoothing should produce reasonable output
        # The smoothed volume should have similar mean but reduced dynamic range
        assert smoothed.data.shape == vol.data.shape
        # Smoothing should reduce the overall range of values
        assert np.ptp(smoothed.data) <= np.ptp(vol.data)