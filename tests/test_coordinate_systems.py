"""
Comprehensive tests for coordinate system transformations.

This module tests the conversion between different coordinate systems:
- Grid coordinates (i, j, k indices)
- Linear indices (single index into flattened array)
- World coordinates (x, y, z in mm or other units)
"""

import pytest
import numpy as np
import neuroim as pn
from neuroim.neuro_space import NeuroSpace
from neuroim.neuro_vol import DenseNeuroVol


class TestCoordinateTransformations:
    """Test coordinate system transformations matching R's behavior."""
    
    @pytest.fixture
    def basic_space(self):
        """Create a basic 3D space for testing."""
        return NeuroSpace(
            dim=(10, 10, 10),
            spacing=(1, 1, 1),
            origin=(0, 0, 0)
        )
    
    @pytest.fixture  
    def anisotropic_space(self):
        """Create space with non-uniform voxel sizes."""
        return NeuroSpace(
            dim=(64, 64, 32),
            spacing=(3.5, 3.5, 5.0),
            origin=(-112, -112, -80)
        )
    
    @pytest.fixture
    def affine_space(self):
        """Create space with custom affine transformation."""
        # Rotation + translation affine matrix
        affine = np.array([
            [0.9, -0.1, 0.0, 10.0],
            [0.1,  0.9, 0.0, 20.0],
            [0.0,  0.0, 1.0, 30.0],
            [0.0,  0.0, 0.0,  1.0]
        ])
        return NeuroSpace(
            dim=(20, 20, 20),
            spacing=(2, 2, 2),
            origin=(0, 0, 0),
            trans=affine
        )
    
    def test_grid_to_index_basic(self, basic_space):
        """Test grid to linear index conversion."""
        # Test single coordinate
        assert basic_space.grid_to_index(np.array([[0, 0, 0]])) == [0]
        assert basic_space.grid_to_index(np.array([[9, 9, 9]])) == [999]
        assert basic_space.grid_to_index(np.array([[5, 5, 5]])) == [555]
        
        # Test multiple coordinates
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        indices = basic_space.grid_to_index(coords)
        expected = [0, 1, 10, 100]  # Column-major order like R
        np.testing.assert_array_equal(indices, expected)
    
    def test_index_to_grid_basic(self, basic_space):
        """Test linear index to grid conversion."""
        # Test single index
        np.testing.assert_array_equal(
            basic_space.index_to_grid(np.array([0])),
            [[0, 0, 0]]
        )
        np.testing.assert_array_equal(
            basic_space.index_to_grid(np.array([999])),
            [[9, 9, 9]]
        )
        
        # Test round-trip conversion
        for idx in [0, 100, 500, 999]:
            grid = basic_space.index_to_grid(np.array([idx]))
            idx_back = basic_space.grid_to_index(grid)
            assert idx_back[0] == idx
    
    def test_grid_to_coord_world(self, anisotropic_space):
        """Test grid to world coordinate conversion."""
        # Origin voxel should map to origin coordinates
        origin_world = anisotropic_space.grid_to_coord(np.array([[0, 0, 0]]))
        np.testing.assert_allclose(origin_world[0], [-112, -112, -80])
        
        # Test voxel in center
        center_grid = np.array([[32, 32, 16]])
        center_world = anisotropic_space.grid_to_coord(center_grid)
        expected = [-112 + 32*3.5, -112 + 32*3.5, -80 + 16*5.0]
        np.testing.assert_allclose(center_world[0], expected)
    
    def test_coord_to_grid_world(self, anisotropic_space):
        """Test world coordinate to grid conversion."""
        # Test origin
        origin_grid = anisotropic_space.coord_to_grid(np.array([[-112, -112, -80]]))
        np.testing.assert_allclose(origin_grid[0], [0, 0, 0])
        
        # Test round-trip
        test_grids = np.array([[10, 20, 15], [32, 32, 16], [50, 50, 25]])
        for grid in test_grids:
            world = anisotropic_space.grid_to_coord(np.array([grid]))
            grid_back = anisotropic_space.coord_to_grid(world)
            np.testing.assert_allclose(grid_back[0], grid, atol=1e-10)
    
    def test_affine_transformations(self, affine_space):
        """Test coordinate transforms with affine matrix."""
        # Grid [0,0,0] should map to translated origin
        origin_world = affine_space.grid_to_coord(np.array([[0, 0, 0]]))
        np.testing.assert_allclose(origin_world[0], [10, 20, 30])
        
        # Test that affine rotation is applied
        grid_point = np.array([[10, 0, 0]])
        world_point = affine_space.grid_to_coord(grid_point)
        # With rotation, x-direction grid movement affects both x and y world
        assert world_point[0, 0] != 10 * 2 + 10  # Not just translation + scale
    
    def test_index_coord_conversion(self, basic_space):
        """Test direct index to coordinate conversion."""
        # Test index_to_coord
        coords = basic_space.index_to_coord(np.array([0, 100, 999]))
        expected_grids = basic_space.index_to_grid(np.array([0, 100, 999]))
        expected_coords = basic_space.grid_to_coord(expected_grids)
        np.testing.assert_allclose(coords, expected_coords)
        
        # Test coord_to_index  
        test_coords = np.array([[5, 5, 5], [2.5, 7.5, 3.0]])
        indices = basic_space.coord_to_index(test_coords)
        # Verify by round-trip
        coords_back = basic_space.index_to_coord(indices)
        grids = basic_space.coord_to_grid(test_coords)
        grids_int = np.round(grids).astype(int)
        expected_coords = basic_space.grid_to_coord(grids_int)
        np.testing.assert_allclose(coords_back, expected_coords)
    
    def test_boundary_conditions(self, basic_space):
        """Test coordinate conversions at boundaries."""
        dim = basic_space.dim[:3]
        
        # Test all corners of the volume
        corners = [
            [0, 0, 0],
            [dim[0]-1, 0, 0],
            [0, dim[1]-1, 0],
            [0, 0, dim[2]-1],
            [dim[0]-1, dim[1]-1, 0],
            [dim[0]-1, 0, dim[2]-1],
            [0, dim[1]-1, dim[2]-1],
            [dim[0]-1, dim[1]-1, dim[2]-1]
        ]
        
        for corner in corners:
            # Test grid->index->grid round trip
            idx = basic_space.grid_to_index(np.array([corner]))[0]
            assert 0 <= idx < np.prod(dim)
            grid_back = basic_space.index_to_grid(np.array([idx]))[0]
            np.testing.assert_array_equal(grid_back, corner)
    
    def test_out_of_bounds_handling(self, basic_space):
        """Test handling of out-of-bounds coordinates."""
        # Test negative indices
        with pytest.raises((IndexError, ValueError)):
            basic_space.grid_to_index(np.array([[-1, 0, 0]]))
        
        # Test indices beyond dimensions
        with pytest.raises((IndexError, ValueError)):
            basic_space.grid_to_index(np.array([[10, 5, 5]]))  # dim is (10,10,10)
        
        # Test invalid linear indices
        with pytest.raises((IndexError, ValueError)):
            basic_space.index_to_grid(np.array([1000]))  # Max valid is 999
    
    def test_coordinate_types(self, basic_space):
        """Test that different numeric types work correctly."""
        # Test with integers
        coords_int = np.array([[1, 2, 3]], dtype=np.int32)
        idx_int = basic_space.grid_to_index(coords_int)
        
        # Test with floats (should work for coord_to_grid)
        coords_float = np.array([[1.5, 2.5, 3.5]], dtype=np.float32)
        grid_float = basic_space.coord_to_grid(coords_float)
        # coord_to_grid returns continuous (float) grid coordinates
        np.testing.assert_allclose(grid_float[0], [1.5, 2.5, 3.5])
        
        # Test with different int types
        for dtype in [np.int16, np.int32, np.int64]:
            coords = np.array([[5, 5, 5]], dtype=dtype)
            idx = basic_space.grid_to_index(coords)
            assert isinstance(idx[0], (int, np.integer))
    
    def test_large_coordinate_arrays(self, basic_space):
        """Test performance with large coordinate arrays."""
        # Generate 10000 random valid coordinates
        n_coords = 10000
        coords = np.random.randint(0, 10, size=(n_coords, 3))
        
        # Test grid to index
        indices = basic_space.grid_to_index(coords)
        assert len(indices) == n_coords
        assert all(0 <= idx < 1000 for idx in indices)
        
        # Test round-trip
        coords_back = basic_space.index_to_grid(indices)
        np.testing.assert_array_equal(coords_back, coords)
    
    def test_neurospace_with_time(self):
        """Test coordinate handling for 4D spaces."""
        space_4d = NeuroSpace(dim=(10, 10, 10, 20))
        
        # For 4D space, grid_to_index requires all dimensions
        coords_4d = np.array([[5, 5, 5, 0]])  # Include time dimension
        idx = space_4d.grid_to_index(coords_4d)
        assert idx[0] == 555  # First 3D volume
        
        # Verify coordinate transformations work with all dimensions
        coords_4d_full = np.array([[5, 5, 5, 10]])
        world_coords = space_4d.grid_to_coord(coords_4d_full)
        assert world_coords.shape == (1, 4)  # All 4 dimensions
        
        # For 4D+ spaces, transformation is simple scaling
        expected = coords_4d_full * space_4d.spacing + space_4d.origin
        np.testing.assert_array_almost_equal(world_coords, expected)
    
    def test_column_vs_row_major(self, basic_space):
        """Test that we correctly handle column-major ordering like R."""
        # In column-major (Fortran/R style):
        # [0,0,0]=0, [1,0,0]=1, [2,0,0]=2, ..., [9,0,0]=9,
        # [0,1,0]=10, [1,1,0]=11, ...
        
        # Test specific indices
        test_cases = [
            ([0, 0, 0], 0),
            ([1, 0, 0], 1),
            ([9, 0, 0], 9),
            ([0, 1, 0], 10),
            ([1, 1, 0], 11),
            ([0, 0, 1], 100),
            ([5, 5, 5], 555),
        ]
        
        for grid, expected_idx in test_cases:
            idx = basic_space.grid_to_index(np.array([grid]))[0]
            assert idx == expected_idx, f"Grid {grid} should map to index {expected_idx}, got {idx}"
    
    def test_real_world_example(self):
        """Test with real-world neuroimaging parameters."""
        # Standard MNI space parameters
        mni_space = NeuroSpace(
            dim=(91, 109, 91),
            spacing=(2.0, 2.0, 2.0),
            origin=(-90, -126, -72)
        )
        
        # Test anatomical landmarks
        # Anterior commissure in MNI is approximately at (0, 0, 0)
        ac_world = np.array([[0, 0, 0]])
        ac_grid = mni_space.coord_to_grid(ac_world)
        expected_grid = [45, 63, 36]  # Approximate AC location in grid
        np.testing.assert_allclose(ac_grid[0], expected_grid, atol=1)
        
        # Verify round-trip
        ac_world_back = mni_space.grid_to_coord(np.round(ac_grid).astype(int))
        np.testing.assert_allclose(ac_world_back, ac_world, atol=2.0)  # Within 1 voxel


class TestCoordinateEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_coordinate_arrays(self):
        """Test handling of empty coordinate arrays."""
        space = NeuroSpace(dim=(10, 10, 10))
        
        # Empty array should return empty result
        empty_coords = np.array([]).reshape(0, 3)
        indices = space.grid_to_index(empty_coords)
        assert len(indices) == 0
        
        empty_indices = np.array([], dtype=int)
        coords = space.index_to_grid(empty_indices)
        assert coords.shape == (0, 3)
    
    def test_singleton_dimensions(self):
        """Test spaces with singleton dimensions."""
        # 2D slice encoded as 3D with singleton dimension
        slice_space = NeuroSpace(dim=(64, 64, 1))
        
        coords = np.array([[32, 32, 0]])
        idx = slice_space.grid_to_index(coords)
        assert idx[0] == 32 + 32 * 64
        
        # Round trip
        coords_back = slice_space.index_to_grid(idx)
        np.testing.assert_array_equal(coords_back, coords)
    
    def test_extreme_dimensions(self):
        """Test with very large or small dimensions."""
        # Very small volume
        tiny_space = NeuroSpace(dim=(1, 1, 1))
        assert tiny_space.grid_to_index(np.array([[0, 0, 0]]))[0] == 0
        
        # Large volume (but not too large to cause memory issues)
        large_space = NeuroSpace(dim=(256, 256, 256))
        max_idx = 256**3 - 1
        max_coords = large_space.index_to_grid(np.array([max_idx]))
        np.testing.assert_array_equal(max_coords[0], [255, 255, 255])
    
    def test_fractional_coordinates(self):
        """Test handling of fractional coordinates."""
        space = NeuroSpace(dim=(10, 10, 10), spacing=(2, 2, 2))

        # coord_to_grid returns continuous (float) grid coordinates
        frac_world = np.array([[3.7, 5.3, 7.8]])  # Between grid points
        frac_grid = space.coord_to_grid(frac_world)

        # Should return continuous grid coordinates
        np.testing.assert_allclose(frac_grid[0], [1.85, 2.65, 3.9])

        # coord_to_index handles rounding internally
        idx = space.coord_to_index(frac_world)
        assert isinstance(idx[0], (int, np.integer))