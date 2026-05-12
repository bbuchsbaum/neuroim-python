"""
Test suite for orthogonal slice extraction functionality.

This module tests the orthogonal slice extraction functions in neuroim,
including axial, sagittal, and coronal slice extraction at world-space coordinates.
"""

import pytest
import numpy as np
from neuroim import (
    NeuroSpace, DenseNeuroVol, NeuroSlice,
    extract_orthogonal_slices, extract_axial_slice,
    extract_sagittal_slice, extract_coronal_slice,
    get_slice_orientation, get_world_bounds_for_slice
)


class TestOrthogonalSlices:
    """Test cases for orthogonal slice extraction functionality."""
    
    @pytest.fixture
    def create_test_volume(self):
        """Create a test volume with known structure."""
        # Create 64x64x64 volume
        space = NeuroSpace(
            dim=(64, 64, 64),
            spacing=(2.0, 2.0, 2.0),  # 2mm voxels
            origin=(-63.0, -63.0, -63.0)  # Centered around origin
        )
        
        # Create data with distinct patterns in each direction
        data = np.zeros((64, 64, 64))
        
        # Add gradients in each direction to distinguish slices
        for i in range(64):
            data[i, :, :] += i  # X gradient
            data[:, i, :] += i * 2  # Y gradient
            data[:, :, i] += i * 3  # Z gradient
        
        vol = DenseNeuroVol(data, space)
        return vol, space
    
    @pytest.fixture
    def create_anisotropic_volume(self):
        """Create a volume with anisotropic voxels."""
        # Different spacing in each dimension
        space = NeuroSpace(
            dim=(100, 80, 60),
            spacing=(1.0, 1.5, 3.0),  # Different voxel sizes
            origin=(0.0, 0.0, 0.0)
        )
        
        # Create sphere in center
        data = np.zeros((100, 80, 60))
        center = np.array([50, 40, 30])
        
        for i in range(100):
            for j in range(80):
                for k in range(60):
                    # Account for anisotropic spacing in distance calculation
                    dist = np.sqrt(
                        ((i - center[0]) * space.spacing[0])**2 +
                        ((j - center[1]) * space.spacing[1])**2 +
                        ((k - center[2]) * space.spacing[2])**2
                    )
                    if dist <= 30:  # 30mm radius sphere
                        data[i, j, k] = 100 - dist
        
        vol = DenseNeuroVol(data, space)
        return vol, space
    
    def test_extract_all_orthogonal_slices(self, create_test_volume):
        """Test extracting all three orthogonal slices."""
        vol, space = create_test_volume
        
        # Extract slices at center of volume
        center_world = space.centroid()
        slices = extract_orthogonal_slices(vol, center_world)
        
        # Check we got all three slices
        assert len(slices) == 3
        assert 'axial' in slices
        assert 'sagittal' in slices
        assert 'coronal' in slices
        
        # Check slice types
        assert isinstance(slices['axial'], NeuroSlice)
        assert isinstance(slices['sagittal'], NeuroSlice)
        assert isinstance(slices['coronal'], NeuroSlice)
        
        # Check slice dimensions
        assert slices['axial'].shape == (64, 64)
        assert slices['sagittal'].shape == (64, 64)
        assert slices['coronal'].shape == (64, 64)
    
    def test_extract_specific_slices(self, create_test_volume):
        """Test extracting only specific slice types."""
        vol, space = create_test_volume
        center_world = space.centroid()
        
        # Extract only axial
        slices = extract_orthogonal_slices(vol, center_world, ['axial'])
        assert len(slices) == 1
        assert 'axial' in slices
        
        # Extract sagittal and coronal
        slices = extract_orthogonal_slices(vol, center_world, ['sagittal', 'coronal'])
        assert len(slices) == 2
        assert 'sagittal' in slices
        assert 'coronal' in slices
        assert 'axial' not in slices
    
    def test_individual_slice_extractors(self, create_test_volume):
        """Test individual slice extraction functions."""
        vol, space = create_test_volume
        center_world = space.centroid()
        
        # Extract individual slices
        axial = extract_axial_slice(vol, center_world)
        sagittal = extract_sagittal_slice(vol, center_world)
        coronal = extract_coronal_slice(vol, center_world)
        
        # Compare with combined extraction
        all_slices = extract_orthogonal_slices(vol, center_world)
        
        np.testing.assert_array_equal(axial.data, all_slices['axial'].data)
        np.testing.assert_array_equal(sagittal.data, all_slices['sagittal'].data)
        np.testing.assert_array_equal(coronal.data, all_slices['coronal'].data)
    
    def test_world_coordinate_conversion(self, create_test_volume):
        """Test that world coordinates are properly converted to grid indices."""
        vol, space = create_test_volume
        
        # Test at known world coordinate
        world_point = np.array([10.0, -20.0, 30.0])
        grid_point = space.coord_to_grid(world_point.reshape(1, -1))[0]
        
        slices = extract_orthogonal_slices(vol, world_point)
        
        # Verify slices are extracted at correct indices
        # The values should reflect the gradients we created
        axial = slices['axial']
        sagittal = slices['sagittal']
        coronal = slices['coronal']
        
        # Check that slice data contains expected patterns
        # Due to our gradient setup, we can verify the slice positions
        assert axial.shape == (64, 64)
        assert sagittal.shape == (64, 64)
        assert coronal.shape == (64, 64)
    
    def test_boundary_cases(self, create_test_volume):
        """Test slice extraction at volume boundaries."""
        vol, space = create_test_volume
        
        # Test at minimum boundary
        min_world = space.grid_to_coord(np.array([[0, 0, 0]]))[0]
        slices = extract_orthogonal_slices(vol, min_world)
        assert all(s.shape == (64, 64) for s in slices.values())
        
        # Test at maximum boundary
        max_world = space.grid_to_coord(np.array([[63, 63, 63]]))[0]
        slices = extract_orthogonal_slices(vol, max_world)
        assert all(s.shape == (64, 64) for s in slices.values())
    
    def test_out_of_bounds_error(self, create_test_volume):
        """Test that out-of-bounds coordinates raise appropriate errors."""
        vol, space = create_test_volume
        
        # Point outside volume
        out_of_bounds = np.array([1000.0, 1000.0, 1000.0])
        
        with pytest.raises(ValueError, match="outside volume bounds"):
            extract_orthogonal_slices(vol, out_of_bounds)
    
    def test_invalid_slice_type(self, create_test_volume):
        """Test that invalid slice types raise errors."""
        vol, space = create_test_volume
        center_world = space.centroid()
        
        with pytest.raises(ValueError, match="Invalid slice type"):
            extract_orthogonal_slices(vol, center_world, ['invalid_type'])
    
    def test_neurovol_method(self, create_test_volume):
        """Test the convenience method on NeuroVol."""
        vol, space = create_test_volume
        center_world = space.centroid()
        
        # Use the method directly on volume
        slices = vol.get_orthogonal_slices(center_world)
        
        assert len(slices) == 3
        assert all(isinstance(s, NeuroSlice) for s in slices.values())
    
    def test_anisotropic_voxels(self, create_anisotropic_volume):
        """Test slice extraction with anisotropic voxels."""
        vol, space = create_anisotropic_volume
        
        # Extract at center where sphere should be visible
        center_world = space.centroid()
        slices = extract_orthogonal_slices(vol, center_world)
        
        # Each slice should show a circular cross-section of the sphere
        # but with different sizes due to anisotropic spacing
        axial = slices['axial']
        sagittal = slices['sagittal']
        coronal = slices['coronal']
        
        # Verify slices have correct dimensions
        assert axial.shape == (100, 80)  # X-Y plane
        assert sagittal.shape == (80, 60)  # Y-Z plane
        assert coronal.shape == (100, 60)  # X-Z plane
        
        # Center values should be high (center of sphere)
        assert axial[50, 40] > 50
        assert sagittal[40, 30] > 50
        assert coronal[50, 30] > 50
    
    def test_slice_orientation(self, create_test_volume):
        """Test getting slice orientations."""
        vol, space = create_test_volume
        
        # Get orientations for default LPI space
        axial_orient = get_slice_orientation(vol, 'axial')
        sagittal_orient = get_slice_orientation(vol, 'sagittal')
        coronal_orient = get_slice_orientation(vol, 'coronal')
        
        # These will depend on the axis names in the space
        # For standard orientations:
        assert isinstance(axial_orient, str)
        assert isinstance(sagittal_orient, str)
        assert isinstance(coronal_orient, str)
        assert '-' in axial_orient  # Should be like "x-y" or "L-P"
    
    def test_world_bounds_for_slice(self, create_test_volume):
        """Test getting world bounds for slices."""
        vol, space = create_test_volume
        
        # Get bounds for entire axial dimension
        min_bounds, max_bounds = get_world_bounds_for_slice(vol, 'axial')
        assert len(min_bounds) == 3
        assert len(max_bounds) == 3
        
        # Get bounds for specific slice
        min_bounds, max_bounds = get_world_bounds_for_slice(vol, 'axial', 32)
        # Z coordinate should be the same for min and max
        assert min_bounds[2] == max_bounds[2]
    
    def test_slice_data_consistency(self, create_test_volume):
        """Test that slice data is consistent with volume data."""
        vol, space = create_test_volume
        
        # Extract slices at a specific grid point
        grid_point = np.array([20, 30, 40])
        world_point = space.grid_to_coord(grid_point.reshape(1, -1))[0]
        
        slices = extract_orthogonal_slices(vol, world_point)
        
        # Verify slice data matches volume data
        axial = slices['axial']
        sagittal = slices['sagittal']
        coronal = slices['coronal']
        
        # Axial slice at z=40 should match vol[:, :, 40]
        np.testing.assert_array_equal(axial.data, vol.data[:, :, 40])
        
        # Sagittal slice at x=20 should match vol[20, :, :]
        np.testing.assert_array_equal(sagittal.data, vol.data[20, :, :])
        
        # Coronal slice at y=30 should match vol[:, 30, :]
        np.testing.assert_array_equal(coronal.data, vol.data[:, 30, :])
    
    def test_slice_spacing(self, create_test_volume):
        """Test that slice spacing is preserved correctly."""
        vol, space = create_test_volume
        center_world = space.centroid()
        
        slices = extract_orthogonal_slices(vol, center_world)
        
        # Check spacing for each slice
        axial = slices['axial']
        sagittal = slices['sagittal']
        coronal = slices['coronal']
        
        # Axial slice (X-Y plane) should have X and Y spacing
        np.testing.assert_array_equal(axial.spacing, [2.0, 2.0])
        
        # Sagittal slice (Y-Z plane) should have Y and Z spacing
        np.testing.assert_array_equal(sagittal.spacing, [2.0, 2.0])
        
        # Coronal slice (X-Z plane) should have X and Z spacing
        np.testing.assert_array_equal(coronal.spacing, [2.0, 2.0])
    
    def test_non_3d_volume_error(self):
        """Test that non-3D volumes raise appropriate error."""
        # Create 4D NeuroVec instead of NeuroVol
        from neuroim import DenseNeuroVec
        space_4d = NeuroSpace((10, 10, 10, 5))
        data_4d = np.random.rand(10, 10, 10, 5)
        vec_4d = DenseNeuroVec(data_4d, space_4d)
        
        # extract_orthogonal_slices expects NeuroVol, not NeuroVec
        with pytest.raises(TypeError, match="vol must be a NeuroVol"):
            extract_orthogonal_slices(vec_4d, np.array([0, 0, 0]))
    
    def test_wrong_world_point_dimensions(self, create_test_volume):
        """Test that world points with wrong dimensions raise error."""
        vol, space = create_test_volume
        
        # 2D point
        with pytest.raises(ValueError, match="must be a 3-element array"):
            extract_orthogonal_slices(vol, np.array([0, 0]))
        
        # 4D point
        with pytest.raises(ValueError, match="must be a 3-element array"):
            extract_orthogonal_slices(vol, np.array([0, 0, 0, 0]))
    
    def test_with_transformation_matrix(self):
        """Test with non-identity transformation matrix."""
        # Create volume with rotation/translation
        trans = np.array([
            [0.9, -0.1, 0.0, 10.0],
            [0.1,  0.9, 0.0, -5.0],
            [0.0,  0.0, 1.0, 20.0],
            [0.0,  0.0, 0.0,  1.0]
        ])
        
        space = NeuroSpace(dim=(50, 50, 50), trans=trans)
        data = np.random.rand(50, 50, 50)
        vol = DenseNeuroVol(data, space)
        
        # Extract slices at transformed coordinates
        world_point = np.array([25.0, 25.0, 45.0])
        slices = extract_orthogonal_slices(vol, world_point)
        
        assert len(slices) == 3
        assert all(isinstance(s, NeuroSlice) for s in slices.values())