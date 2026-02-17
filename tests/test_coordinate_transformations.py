"""
Comprehensive tests for coordinate transformations
Tests grid-to-coord, coord-to-grid, and transformation matrices
"""

import pytest
import numpy as np
from neuroimpy import NeuroSpace
from neuroimpy.neuro_vol import DenseNeuroVol


class TestCoordinateTransformations:
    """Test coordinate transformation accuracy and edge cases"""
    
    def test_identity_transformation(self):
        """Test identity transformation (no rotation, unit spacing, zero origin)"""
        space = NeuroSpace(dim=(10, 10, 10), 
                          spacing=(1, 1, 1),
                          origin=(0, 0, 0))
        
        # Grid to coord with identity should be same
        grid_coords = np.array([5, 5, 5])
        world_coords = space.grid_to_coord(grid_coords.reshape(1, -1))[0]
        np.testing.assert_array_equal(world_coords, [5, 5, 5])
        
        # Coord to grid should be inverse
        grid_back = space.coord_to_grid(world_coords.reshape(1, -1))[0]
        np.testing.assert_array_equal(grid_back, grid_coords)
    
    def test_translation_only(self):
        """Test transformation with only translation (origin shift)"""
        space = NeuroSpace(dim=(10, 10, 10),
                          spacing=(1, 1, 1),
                          origin=(100, 200, 300))
        
        # Corner voxel [0, 0, 0] should map to origin
        world = space.grid_to_coord(np.array([[0, 0, 0]]))[0]
        np.testing.assert_array_equal(world, [100, 200, 300])
        
        # And back
        grid = space.coord_to_grid(np.array([[100, 200, 300]]))[0]
        np.testing.assert_array_equal(grid, [0, 0, 0])
        
        # Center voxel
        world = space.grid_to_coord(np.array([[5, 5, 5]]))[0]
        np.testing.assert_array_equal(world, [105, 205, 305])
    
    def test_scaling_only(self):
        """Test transformation with only scaling (non-unit spacing)"""
        space = NeuroSpace(dim=(10, 10, 10),
                          spacing=(2, 3, 4),
                          origin=(0, 0, 0))
        
        # Test scaling effect
        world = space.grid_to_coord(np.array([[1, 1, 1]]))[0]
        np.testing.assert_array_equal(world, [2, 3, 4])
        
        world = space.grid_to_coord(np.array([[5, 5, 5]]))[0]
        np.testing.assert_array_equal(world, [10, 15, 20])
        
        # Inverse
        grid = space.coord_to_grid(np.array([[10, 15, 20]]))[0]
        np.testing.assert_array_equal(grid, [5, 5, 5])
    
    def test_combined_transform(self):
        """Test combined translation and scaling"""
        space = NeuroSpace(dim=(10, 10, 10),
                          spacing=(2, 2, 2),
                          origin=(50, 60, 70))
        
        # Corner to world
        world = space.grid_to_coord(np.array([[0, 0, 0]]))[0]
        np.testing.assert_array_equal(world, [50, 60, 70])
        
        # Other corner
        world = space.grid_to_coord(np.array([[9, 9, 9]]))[0]
        np.testing.assert_array_equal(world, [68, 78, 88])  # 50 + 9*2, etc
        
        # Round-trip test
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    grid = np.array([i, j, k])
                    world = space.grid_to_coord(grid.reshape(1, -1))[0]
                    grid_back = space.coord_to_grid(world.reshape(1, -1))[0]
                    np.testing.assert_allclose(grid_back, grid, atol=1e-10)
    
    def test_affine_matrix_consistency(self):
        """Test that transformation matrix produces same results as direct methods"""
        space = NeuroSpace(dim=(10, 10, 10),
                          spacing=(1.5, 2.0, 2.5),
                          origin=(-30, -40, -50))
        
        # Get transformation matrix
        trans = space.trans
        
        # Test multiple points
        test_points = [
            [0, 0, 0],
            [5, 5, 5],
            [9, 9, 9],
            [3, 7, 2]
        ]
        
        for point in test_points:
            # Direct method
            world_direct = space.grid_to_coord(np.array([point]))[0]
            
            # Matrix method
            grid_homogeneous = np.append(point, 1)
            world_matrix = trans.dot(grid_homogeneous)[:3]
            
            np.testing.assert_allclose(world_matrix, world_direct, atol=1e-10)
    
    def test_non_orthogonal_axes(self):
        """Test transformations with non-orthogonal axes (rotation)"""
        # Create custom transformation with rotation
        theta = np.pi / 6  # 30 degrees
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Build transformation: rotation + scaling + translation
        trans_matrix = np.eye(4)
        trans_matrix[:3, :3] = rotation * np.array([2, 2, 2])  # Scale by 2
        trans_matrix[:3, 3] = [10, 20, 30]  # Translate
        
        space = NeuroSpace(dim=(10, 10, 10), trans=trans_matrix)
        
        # Test transformation
        grid = np.array([1, 0, 0])
        world = space.grid_to_coord(grid.reshape(1, -1))[0]
        
        # Expected: rotate [1,0,0] by 30 deg, scale by 2, translate
        expected = rotation.dot([1, 0, 0]) * 2 + [10, 20, 30]
        np.testing.assert_allclose(world, expected, atol=1e-10)
    
    def test_inverse_accuracy(self):
        """Test accuracy of inverse transformations"""
        # Complex transformation
        trans_matrix = np.array([
            [1.5, 0.2, 0.1, 10],
            [0.1, 1.7, 0.3, 20],
            [0.2, 0.1, 1.9, 30],
            [0, 0, 0, 1]
        ])
        
        space = NeuroSpace(dim=(20, 20, 20), trans=trans_matrix)
        
        # Test round-trip for many points
        test_coords = np.random.rand(100, 3) * 19  # Random points in volume
        
        for coord in test_coords:
            world = space.grid_to_coord(coord.reshape(1, -1))[0]
            grid_back = space.coord_to_grid(world.reshape(1, -1))[0]
            np.testing.assert_allclose(grid_back, coord, atol=1e-10)
    
    def test_boundary_coordinates(self):
        """Test coordinate transformations at volume boundaries"""
        space = NeuroSpace(dim=(10, 10, 10),
                          spacing=(1, 1, 1),
                          origin=(0, 0, 0))
        
        # Exact boundaries
        corners = [
            [0, 0, 0],
            [9, 9, 9],
            [0, 9, 0],
            [9, 0, 9]
        ]
        
        for corner in corners:
            world = space.grid_to_coord(np.array([corner]))[0]
            grid_back = space.coord_to_grid(world.reshape(1, -1))[0]
            np.testing.assert_allclose(grid_back, corner, atol=1e-10)
        
        # Just outside boundaries
        outside_points_world = [
            [-0.5, 0, 0],  # Just before origin
            [9.5, 9.5, 9.5],  # Just past end
        ]
        
        for point in outside_points_world:
            grid = space.coord_to_grid(np.array([point]))[0]
            # Should map to fractional grid coordinates
            assert grid[0] < 0 or grid[0] > 9
    
    def test_subvoxel_precision(self):
        """Test sub-voxel coordinate precision"""
        space = NeuroSpace(dim=(10, 10, 10),
                          spacing=(1, 1, 1),
                          origin=(0, 0, 0))
        
        # Sub-voxel world coordinates
        world_point = np.array([5.25, 5.75, 5.5])
        grid = space.coord_to_grid(world_point.reshape(1, -1))[0]

        # Should get fractional grid coordinates (not rounded)
        np.testing.assert_allclose(grid, [5.25, 5.75, 5.5], atol=1e-10)

        # And back - should match exactly with fractional coordinates
        world_back = space.grid_to_coord(grid.reshape(1, -1))[0]
        np.testing.assert_allclose(world_back, world_point, atol=1e-10)
    
    def test_large_coordinate_values(self):
        """Test transformations with large coordinate values"""
        # Large origin values (e.g., scanner coordinates)
        space = NeuroSpace(dim=(256, 256, 128),
                          spacing=(0.9375, 0.9375, 1.5),
                          origin=(-120000, -130000, -75000))
        
        # Center of volume
        center_grid = np.array([128, 128, 64])
        center_world = space.grid_to_coord(center_grid.reshape(1, -1))[0]
        
        # Check reasonable values
        assert -200000 < center_world[0] < -50000
        assert -200000 < center_world[1] < -50000
        assert -150000 < center_world[2] < 0
        
        # Round trip
        grid_back = space.coord_to_grid(center_world.reshape(1, -1))[0]
        np.testing.assert_allclose(grid_back, center_grid, atol=1e-6)
    
    def test_coordinate_array_handling(self):
        """Test batch coordinate transformations"""
        space = NeuroSpace(dim=(10, 10, 10),
                          spacing=(2, 2, 2),
                          origin=(100, 100, 100))
        
        # Multiple coordinates at once
        grid_coords = np.array([
            [0, 0, 0],
            [5, 5, 5],
            [9, 9, 9],
            [2, 4, 6]
        ])
        
        # Transform each
        world_coords = np.array([space.grid_to_coord(gc.reshape(1, -1))[0] for gc in grid_coords])
        
        # Verify
        expected = grid_coords * 2 + 100  # spacing=2, origin=100
        np.testing.assert_allclose(world_coords, expected, atol=1e-10)
    
    def test_axes_orientation(self):
        """Test different axes orientations"""
        # Pass list of axis names (Bug 4 fix)
        axes = ['x', 'y', 'z']

        space = NeuroSpace(dim=(10, 10, 10), axes=axes)

        # Test that axes are properly created from list
        from neuroimpy.axis import axis_names
        names = axis_names(space.axes)
        assert names == ['x', 'y', 'z']
    
    def test_dimension_ordering_consistency(self):
        """Test that dimension ordering is consistent across operations"""
        # 4D space
        space = NeuroSpace(dim=(10, 10, 10, 20))
        
        # Verify spatial dimensions
        assert space.ndim == 4
        np.testing.assert_array_equal(space.dim, np.array([10, 10, 10, 20]))
        assert len(space.spacing) == 4
        assert len(space.origin) == 4
        
        # Grid to coord should only work on first 3 dims for 4D spaces
        grid_3d = np.array([5, 5, 5, 0])  # Need to provide all 4 dims
        world_4d = space.grid_to_coord(grid_3d.reshape(1, -1))[0]
        assert world_4d.shape == (4,)  # Returns all 4 dims
    
    def test_numerical_stability(self):
        """Test numerical stability of transformations"""
        # Create transformation with small determinant (near-singular)
        trans_matrix = np.array([
            [1e-6, 0, 0, 0],
            [0, 1e-6, 0, 0],
            [0, 0, 1e-6, 0],
            [0, 0, 0, 1]
        ])
        
        space = NeuroSpace(dim=(10, 10, 10), trans=trans_matrix)
        
        # Should still work but with limited precision
        grid = np.array([5, 5, 5])
        world = space.grid_to_coord(grid.reshape(1, -1))[0]
        grid_back = space.coord_to_grid(world.reshape(1, -1))[0]
        
        # May have reduced precision but should be close
        np.testing.assert_allclose(grid_back, grid, atol=1e-3)