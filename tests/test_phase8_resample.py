"""Tests for Phase 8: Resampling & Interpolation."""

import pytest
import numpy as np
from neuroimpy import (
    NeuroSpace, NeuroVol, DenseNeuroVol, DenseNeuroVec,
    resample, resample_vec, reorient,
    LEFT_RIGHT, RIGHT_LEFT, ANT_POST, POST_ANT, SUP_INF, INF_SUP
)


class TestResample:
    """Test resample function for NeuroVol."""
    
    def setup_method(self):
        """Set up test data."""
        # Create source volume with simple pattern
        self.source_space = NeuroSpace(
            dim=[10, 10, 10],
            spacing=[2, 2, 2],
            origin=[0, 0, 0]
        )
        
        # Create data with a simple pattern for testing
        data = np.zeros((10, 10, 10))
        data[4:7, 4:7, 4:7] = 1.0  # 3x3x3 cube
        self.source_vol = DenseNeuroVol(data, self.source_space)
        
        # Create target spaces
        self.target_space_same = NeuroSpace(
            dim=[10, 10, 10],
            spacing=[2, 2, 2],
            origin=[0, 0, 0]
        )
        
        self.target_space_double = NeuroSpace(
            dim=[20, 20, 20],
            spacing=[1, 1, 1],
            origin=[0, 0, 0]
        )
        
        self.target_space_half = NeuroSpace(
            dim=[5, 5, 5],
            spacing=[4, 4, 4],
            origin=[0, 0, 0]
        )
    
    def test_resample_to_same_space(self):
        """Test resampling to same space."""
        try:
            resampled = resample(self.source_vol, self.target_space_same)
            
            # Should be identical (within floating point precision)
            assert resampled.space.dim[0] == 10
            assert resampled.space.dim[1] == 10
            assert resampled.space.dim[2] == 10
            assert np.allclose(resampled.data, self.source_vol.data, rtol=1e-5)
        except ImportError:
            pytest.skip("nibabel not installed")
    
    def test_resample_to_higher_resolution(self):
        """Test resampling to higher resolution."""
        try:
            resampled = resample(self.source_vol, self.target_space_double)
            
            assert resampled.space.dim[0] == 20
            assert resampled.space.dim[1] == 20
            assert resampled.space.dim[2] == 20
            
            # Should have more voxels but preserve the structure
            assert resampled.data.sum() > 0
        except ImportError:
            pytest.skip("nibabel not installed")
    
    def test_resample_to_lower_resolution(self):
        """Test resampling to lower resolution."""
        try:
            resampled = resample(self.source_vol, self.target_space_half)
            
            assert resampled.space.dim[0] == 5
            assert resampled.space.dim[1] == 5
            assert resampled.space.dim[2] == 5
            
            # Should have fewer voxels
            assert resampled.data.shape == (5, 5, 5)
        except ImportError:
            pytest.skip("nibabel not installed")
    
    def test_resample_interpolation_methods(self):
        """Test different interpolation methods."""
        try:
            # Nearest neighbor
            resampled_nn = resample(self.source_vol, self.target_space_double, 
                                  interpolation=0)
            
            # Linear
            resampled_linear = resample(self.source_vol, self.target_space_double,
                                      interpolation=1)
            
            # Cubic
            resampled_cubic = resample(self.source_vol, self.target_space_double,
                                     interpolation=3)
            
            # All should produce valid results
            assert resampled_nn.data.shape == (20, 20, 20)
            assert resampled_linear.data.shape == (20, 20, 20)
            assert resampled_cubic.data.shape == (20, 20, 20)
            
            # All interpolation methods should produce non-zero results
            assert resampled_nn.data.sum() > 0
            assert resampled_linear.data.sum() > 0
            assert resampled_cubic.data.sum() > 0
            
            # Nearest neighbor should preserve the exact values (0 or 1)
            # Due to the simple binary cube pattern
            assert np.allclose(np.unique(resampled_nn.data), [0, 1], atol=1e-5)
        except ImportError:
            pytest.skip("nibabel not installed")
    
    def test_resample_to_target_vol(self):
        """Test resampling to match another volume."""
        try:
            target_vol = DenseNeuroVol(
                np.zeros((15, 15, 15)),
                NeuroSpace(dim=[15, 15, 15], spacing=[1.5, 1.5, 1.5])
            )
            
            resampled = resample(self.source_vol, target_vol)
            
            assert resampled.space.dim[0] == 15
            assert resampled.space.dim[1] == 15
            assert resampled.space.dim[2] == 15
            assert np.array_equal(resampled.space.spacing, target_vol.space.spacing)
        except ImportError:
            pytest.skip("nibabel not installed")
    
    def test_resample_invalid_interpolation(self):
        """Test invalid interpolation parameter."""
        try:
            with pytest.raises(ValueError):
                resample(self.source_vol, self.target_space_same, interpolation=2)
            
            with pytest.raises(ValueError):
                resample(self.source_vol, self.target_space_same, interpolation=4)
        except ImportError:
            pytest.skip("nibabel not installed")


class TestResampleVec:
    """Test resample_vec function for NeuroVec."""
    
    def setup_method(self):
        """Set up test data."""
        # Create 4D source
        self.source_space_4d = NeuroSpace(
            dim=[10, 10, 10, 3],
            spacing=[2, 2, 2, 1],
            origin=[0, 0, 0, 0]
        )
        
        # Create 4D data
        data_4d = np.zeros((10, 10, 10, 3))
        data_4d[4:7, 4:7, 4:7, :] = np.array([1.0, 2.0, 3.0])
        self.source_vec = DenseNeuroVec(data_4d, self.source_space_4d)
        
        # Target spaces
        self.target_space_4d = NeuroSpace(
            dim=[20, 20, 20, 3],
            spacing=[1, 1, 1, 1],
            origin=[0, 0, 0, 0]
        )
    
    def test_resample_vec_basic(self):
        """Test basic NeuroVec resampling."""
        try:
            resampled = resample_vec(self.source_vec, self.target_space_4d)
            
            assert resampled.shape == (20, 20, 20, 3)
            assert isinstance(resampled, DenseNeuroVec)
            
            # Check that time dimension is preserved
            assert resampled.shape[3] == self.source_vec.shape[3]
        except ImportError:
            pytest.skip("nibabel not installed")
    
    def test_resample_vec_to_target_vec(self):
        """Test resampling to match another NeuroVec."""
        try:
            target_vec = DenseNeuroVec(
                np.zeros((15, 15, 15, 3)),
                NeuroSpace(dim=[15, 15, 15, 3], spacing=[1.5, 1.5, 1.5, 1])
            )
            
            resampled = resample_vec(self.source_vec, target_vec)
            
            assert resampled.shape == target_vec.shape
            assert np.array_equal(resampled.space.dim, target_vec.space.dim)
        except ImportError:
            pytest.skip("nibabel not installed")
    
    def test_resample_vec_interpolation(self):
        """Test different interpolation methods for NeuroVec."""
        try:
            # Test all interpolation methods
            for interp in [0, 1, 3]:
                resampled = resample_vec(self.source_vec, self.target_space_4d,
                                       interpolation=interp)
                assert resampled.shape == (20, 20, 20, 3)
        except ImportError:
            pytest.skip("nibabel not installed")
    
    def test_resample_vec_invalid_target(self):
        """Test error with non-4D target space."""
        try:
            target_3d = NeuroSpace(dim=[20, 20, 20], spacing=[1, 1, 1])
            
            with pytest.raises(ValueError):
                resample_vec(self.source_vec, target_3d)
        except ImportError:
            pytest.skip("nibabel not installed")


class TestReorient:
    """Test reorient function."""
    
    def setup_method(self):
        """Set up test data."""
        # Create space without specifying axes (will use defaults)
        self.space = NeuroSpace(
            dim=[64, 64, 40],
            spacing=[2, 2, 2]
        )
        
        data = np.random.randn(64, 64, 40)
        self.vol = DenseNeuroVol(data, self.space)
    
    def test_reorient_space_string(self):
        """Test reorienting NeuroSpace with string."""
        reoriented = reorient(self.space, "RAS")
        
        assert isinstance(reoriented, NeuroSpace)
        assert reoriented.axes.i == RIGHT_LEFT
        assert reoriented.axes.j == ANT_POST
        assert reoriented.axes.k == SUP_INF
    
    def test_reorient_space_list(self):
        """Test reorienting NeuroSpace with list."""
        reoriented = reorient(self.space, ["R", "A", "S"])
        
        assert isinstance(reoriented, NeuroSpace)
        assert reoriented.axes.i == RIGHT_LEFT
        assert reoriented.axes.j == ANT_POST
        assert reoriented.axes.k == SUP_INF
    
    def test_reorient_vol(self):
        """Test reorienting NeuroVol."""
        reoriented = reorient(self.vol, "RAS")
        
        assert isinstance(reoriented, DenseNeuroVol)
        assert reoriented.space.axes.i == RIGHT_LEFT
        assert reoriented.space.axes.j == ANT_POST
        assert reoriented.space.axes.k == SUP_INF
        
        # Data shape should be preserved (for now)
        assert reoriented.data.shape == self.vol.data.shape
    
    def test_reorient_lowercase(self):
        """Test that lowercase orientation codes work."""
        reoriented = reorient(self.space, "ras")
        
        assert reoriented.axes.i == RIGHT_LEFT
        assert reoriented.axes.j == ANT_POST
        assert reoriented.axes.k == SUP_INF
    
    def test_reorient_invalid_string_length(self):
        """Test error with invalid orientation string length."""
        with pytest.raises(ValueError):
            reorient(self.space, "RA")  # Too short
        
        with pytest.raises(ValueError):
            reorient(self.space, "RASP")  # Too long
    
    def test_reorient_invalid_codes(self):
        """Test error with invalid orientation codes."""
        with pytest.raises(ValueError):
            reorient(self.space, "XAS")  # Invalid first code
        
        with pytest.raises(ValueError):
            reorient(self.space, "RXS")  # Invalid second code
        
        with pytest.raises(ValueError):
            reorient(self.space, "RAX")  # Invalid third code
    
    def test_reorient_invalid_type(self):
        """Test error with invalid orient type."""
        with pytest.raises(TypeError):
            reorient(self.space, 123)  # Not string or list
    
    def test_reorient_all_orientations(self):
        """Test all valid orientation combinations."""
        valid_orientations = [
            "RAS", "RAI", "RPS", "RPI",
            "LAS", "LAI", "LPS", "LPI"
        ]
        
        for orient in valid_orientations:
            reoriented = reorient(self.space, orient)
            assert isinstance(reoriented, NeuroSpace)


class TestResampleIntegration:
    """Integration tests combining multiple operations."""
    
    def test_resample_and_reorient(self):
        """Test combining resampling and reorientation."""
        # Create source volume
        space = NeuroSpace(
            dim=[32, 32, 20],
            spacing=[3, 3, 3]
        )
        vol = DenseNeuroVol(np.random.randn(32, 32, 20), space)
        
        # First reorient
        reoriented = reorient(vol, "RAS")
        
        # Then resample
        target_space = NeuroSpace(
            dim=[64, 64, 40],
            spacing=[1.5, 1.5, 1.5]
        )
        
        try:
            resampled = resample(reoriented, target_space)
            
            assert resampled.shape == (64, 64, 40)
            assert resampled.space.axes.i == RIGHT_LEFT
            assert resampled.space.axes.j == ANT_POST
            assert resampled.space.axes.k == SUP_INF
        except ImportError:
            pytest.skip("nibabel not installed")
    
    def test_nibabel_not_installed(self):
        """Test graceful failure when nibabel is not installed."""
        # This test simulates nibabel not being available
        import sys
        # Get the actual module (not the function)
        resample_module = sys.modules['neuroimpy.resample']
        original_has_nibabel = resample_module.HAS_NIBABEL
        
        try:
            # Temporarily set HAS_NIBABEL to False
            resample_module.HAS_NIBABEL = False
            
            space = NeuroSpace(dim=[10, 10, 10], spacing=[1, 1, 1])
            vol = DenseNeuroVol(np.zeros((10, 10, 10)), space)
            
            with pytest.raises(ImportError, match="nibabel is required"):
                resample_module.resample(vol, space)
            
            with pytest.raises(ImportError, match="nibabel is required"):
                vec = DenseNeuroVec(np.zeros((10, 10, 10, 3)), 
                                  NeuroSpace(dim=[10, 10, 10, 3]))
                resample_module.resample_vec(vec, vec.space)
        finally:
            # Restore original value
            resample_module.HAS_NIBABEL = original_has_nibabel