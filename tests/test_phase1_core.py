"""Phase 1 Tests: Core Infrastructure (NeuroSpace, Axes, Coordinate Transformations)

These tests are direct translations from R's neuroim2 tests to ensure
complete compatibility between the R and Python implementations.

Note: R uses 1-based indexing while Python uses 0-based indexing.
All grid coordinates and indices in these tests have been adjusted accordingly.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from neuroimpy.neuro_space import NeuroSpace, neurospace
from neuroimpy.axis import (
    NamedAxis, AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D,
    LEFT_RIGHT, RIGHT_LEFT, ANT_POST, POST_ANT, INF_SUP, SUP_INF,
    TIME, NullAxis, TimeAxis,
    axis_set, axis_names, axis_directions, 
    find_anatomy_3d, match_axis, flip_axis, permute_axes, drop_axis, add_axis,
    OrientationList3D, OrientationList2D
)


class TestNeuroSpaceConstruction:
    """Test NeuroSpace construction with various parameters."""
    
    def test_basic_construction(self):
        """Test basic NeuroSpace construction."""
        # R: spc <- NeuroSpace(c(64,64,64))
        spc = NeuroSpace((64, 64, 64))
        
        assert spc is not None
        assert_array_equal(spc.dim, [64, 64, 64])
        assert_array_equal(spc.spacing, [1.0, 1.0, 1.0])
        assert_array_equal(spc.origin, [0.0, 0.0, 0.0])
        assert spc.ndim == 3
        assert spc.nvoxels == 64 * 64 * 64
    
    def test_construction_with_spacing_origin(self):
        """Test NeuroSpace with custom spacing and origin."""
        # R: spc <- NeuroSpace(c(64, 64, 40), spacing = c(2, 2, 2), origin = c(-90, -126, -72))
        spc = NeuroSpace((64, 64, 40), spacing=(2, 2, 2), origin=(-90, -126, -72))
        
        assert_array_equal(spc.dim, [64, 64, 40])
        assert_array_equal(spc.spacing, [2.0, 2.0, 2.0])
        assert_array_equal(spc.origin, [-90.0, -126.0, -72.0])
    
    def test_factory_function(self):
        """Test neurospace factory function matches NeuroSpace constructor."""
        # R style: spc <- NeuroSpace(c(10, 10, 10))
        spc1 = NeuroSpace((10, 10, 10))
        spc2 = neurospace((10, 10, 10))
        
        assert spc1 == spc2
    
    def test_transformation_matrix_creation(self):
        """Test that transformation matrix is created correctly."""
        spc = NeuroSpace((64, 64, 40), spacing=(2, 2, 2), origin=(-90, -126, -72))
        trans = spc.trans
        
        # Check matrix dimensions
        assert trans.shape == (4, 4)
        
        # Check diagonal elements (spacing)
        assert trans[0, 0] == 2.0
        assert trans[1, 1] == 2.0
        assert trans[2, 2] == 2.0
        assert trans[3, 3] == 1.0
        
        # Check origin in last column
        assert trans[0, 3] == -90.0
        assert trans[1, 3] == -126.0
        assert trans[2, 3] == -72.0
        
        # Check off-diagonal elements are zero
        assert trans[0, 1] == 0.0
        assert trans[0, 2] == 0.0
        assert trans[1, 0] == 0.0
    
    def test_inverse_transformation(self):
        """Test that inverse transformation is correct."""
        spc = NeuroSpace((64, 64, 40), spacing=(2, 2, 2), origin=(-90, -126, -72))
        trans = spc.trans
        inv = spc.inverse
        
        # Check that trans * inv = identity
        result = trans @ inv
        identity = np.eye(4)
        assert_array_almost_equal(result, identity, decimal=10)
    
    def test_2d_space(self):
        """Test 2D NeuroSpace construction."""
        # R: NeuroSpace(c(10,10))
        spc = NeuroSpace((10, 10))
        
        assert spc.ndim == 2
        assert_array_equal(spc.dim, [10, 10])
        assert spc.nvoxels == 100


class TestCoordinateTransformations:
    """Test coordinate transformation functions."""
    
    def test_grid_to_coord_basic(self):
        """Test grid to world coordinate conversion."""
        spc = NeuroSpace((64, 64, 40), spacing=(2, 2, 2), origin=(-90, -126, -72))
        
        # Test corner voxel (R: c(1,1,1) -> Python: [0,0,0])
        grid_coords = np.array([0, 0, 0])
        world_coords = spc.grid_to_coord(grid_coords)
        assert_array_equal(world_coords[0], [-90, -126, -72])
        
        # Test another point (R: c(32,32,20) -> Python: [31,31,19])
        grid_coords = np.array([31, 31, 19])
        world_coords = spc.grid_to_coord(grid_coords)
        expected = np.array([-90 + 31*2, -126 + 31*2, -72 + 19*2])
        assert_array_equal(world_coords[0], expected)
    
    def test_coord_to_grid_basic(self):
        """Test world to grid coordinate conversion."""
        spc = NeuroSpace((64, 64, 40), spacing=(2, 2, 2), origin=(-90, -126, -72))
        
        # Test origin
        world_coords = np.array([-90, -126, -72])
        grid_coords = spc.coord_to_grid(world_coords)
        assert_array_equal(grid_coords[0], [0, 0, 0])  # Python uses 0-based indexing
        
        # Test another point
        world_coords = np.array([-90 + 10*2, -126 + 15*2, -72 + 5*2])
        grid_coords = spc.coord_to_grid(world_coords)
        assert_array_equal(grid_coords[0], [10, 15, 5])
    
    def test_round_trip_grid_coord(self):
        """Test round-trip conversions between grid and world coordinates."""
        spc = NeuroSpace((64, 64, 40), spacing=(2, 2, 2), origin=(-90, -126, -72))
        
        # Test multiple points
        original_grid = np.array([
            [0, 0, 0],      # R: [1, 1, 1]
            [31, 31, 19],   # R: [32, 32, 20]
            [63, 63, 39]    # R: [64, 64, 40]
        ])
        
        # Grid -> World -> Grid
        world = spc.grid_to_coord(original_grid)
        back_to_grid = spc.coord_to_grid(world)
        assert_array_equal(back_to_grid, original_grid)
    
    def test_index_to_grid_3d(self):
        """Test 1D index to grid conversion for 3D space."""
        spc = NeuroSpace((64, 64, 25))
        
        # R: index_to_grid(vol1, 65) -> matrix(c(1,2,1), nrow=1)
        # Python: index 64 -> [0, 1, 0] (0-based)
        grid = spc.index_to_grid(64)
        assert_array_equal(grid[0], [0, 1, 0])
        
        # R: index_to_grid(vol1, 1) -> matrix(c(1,1,1), nrow=1)
        # Python: index 0 -> [0, 0, 0]
        grid = spc.index_to_grid(0)
        assert_array_equal(grid[0], [0, 0, 0])
    
    def test_grid_to_index_3d(self):
        """Test grid to 1D index conversion for 3D space."""
        spc = NeuroSpace((64, 64, 25))
        
        # Python: [0, 0, 0] -> index 0
        idx = spc.grid_to_index(np.array([[0, 0, 0]]))
        assert idx[0] == 0
        
        # Python: [0, 1, 0] -> index 64
        idx = spc.grid_to_index(np.array([[0, 1, 0]]))
        assert idx[0] == 64
    
    def test_round_trip_index_coord(self):
        """Test round-trip conversions between index and coordinates."""
        spc = NeuroSpace((64, 64, 40), spacing=(2, 2, 2), origin=(-90, -126, -72))
        
        # Test multiple indices
        indices = np.array([0, 99, 999])  # Python 0-based
        
        # Index -> Coord -> Index
        coords = spc.index_to_coord(indices)
        back_to_index = spc.coord_to_index(coords)
        assert_array_equal(back_to_index, indices)
    
    def test_grid_to_index_2d(self):
        """Test grid to index for 2D space (matching R NeuroSlice tests)."""
        spc = NeuroSpace((64, 64), spacing=(1, 1))
        
        # R: coords <- matrix(c(1, 1, 64, 64), nrow=2, byrow=TRUE)
        # Python: [[0, 0], [63, 63]]
        coords = np.array([[0, 0], [63, 63]])
        idx = spc.grid_to_index(coords)
        
        # R: expect idx = c(1, 4096)
        # Python: expect idx = [0, 4095]
        assert_array_equal(idx, [0, 4095])
        
        # Single coordinate
        coords = np.array([0, 0])
        idx = spc.grid_to_index(coords)
        assert idx[0] == 0
    
    def test_index_to_grid_2d(self):
        """Test index to grid for 2D space."""
        spc = NeuroSpace((64, 64), spacing=(1, 1))
        
        # R: idx <- c(1, 4096)
        # Python: idx = [0, 4095]
        idx = np.array([0, 4095])
        coords = spc.index_to_grid(idx)
        
        # R: expect coords = matrix(c(1, 1, 64, 64), nrow = 2, byrow = TRUE)
        # Python: expect coords = [[0, 0], [63, 63]]
        expected = np.array([[0, 0], [63, 63]])
        assert_array_equal(coords, expected)
    
    def test_grid_to_grid_transformation(self):
        """Test grid to grid transformation between spaces."""
        # Source space
        spc1 = NeuroSpace((64, 64, 40), spacing=(2, 2, 2), origin=(-90, -126, -72))
        
        # Target space with different parameters
        spc2 = NeuroSpace((128, 128, 80), spacing=(1, 1, 1), origin=(-90, -126, -72))
        
        # Test grid point
        source_grid = np.array([[10, 10, 10]])
        
        # Transform to target space
        target_grid = spc1.grid_to_grid(source_grid, spc2)
        
        # Verify by converting to world coordinates and back
        world_coords = spc1.grid_to_coord(source_grid)
        expected_target = spc2.coord_to_grid(world_coords)
        
        assert_array_equal(target_grid, expected_target)


class TestAxisAndOrientation:
    """Test AxisSet and NamedAxis functionality."""
    
    def test_named_axis_creation(self):
        """Test NamedAxis creation and properties."""
        axis = NamedAxis("Left-to-Right", [1, 0, 0])
        
        assert axis.axis == "Left-to-Right"
        assert_array_equal(axis.direction, [1, 0, 0])
        
        # Test equality
        axis2 = NamedAxis("Left-to-Right", [1, 0, 0])
        assert axis == axis2
        
        # Test inequality
        axis3 = NamedAxis("Right-to-Left", [-1, 0, 0])
        assert axis != axis3
    
    def test_anatomical_constants(self):
        """Test predefined anatomical axis constants."""
        assert LEFT_RIGHT.axis == "Left-to-Right"
        assert_array_equal(LEFT_RIGHT.direction, [1, 0, 0])
        
        assert RIGHT_LEFT.axis == "Right-to-Left"
        assert_array_equal(RIGHT_LEFT.direction, [-1, 0, 0])
        
        assert INF_SUP.axis == "Inferior-to-Superior"
        assert_array_equal(INF_SUP.direction, [0, 0, 1])
    
    def test_axis_set_creation(self):
        """Test different AxisSet types."""
        # 1D
        axis1d = AxisSet1D(TIME)
        assert axis1d.ndim == 1
        assert axis1d.i == TIME
        
        # 2D
        axis2d = AxisSet2D(LEFT_RIGHT, POST_ANT)
        assert axis2d.ndim == 2
        assert axis2d.i == LEFT_RIGHT
        assert axis2d.j == POST_ANT
        
        # 3D
        axis3d = AxisSet3D(LEFT_RIGHT, POST_ANT, INF_SUP)
        assert axis3d.ndim == 3
        assert axis3d.i == LEFT_RIGHT
        assert axis3d.j == POST_ANT
        assert axis3d.k == INF_SUP
    
    def test_find_anatomy_3d(self):
        """Test finding anatomical orientation from string."""
        # R: orient <- findAnatomy3D("L", "P", "I")
        orient = find_anatomy_3d("LPI")
        
        assert isinstance(orient, AxisSet3D)
        assert orient.i == LEFT_RIGHT
        assert orient.j == POST_ANT
        assert orient.k == INF_SUP
        
        # Test another orientation
        orient2 = find_anatomy_3d("RAS")
        assert orient2.i == RIGHT_LEFT
        assert orient2.j == ANT_POST
        assert orient2.k == SUP_INF
    
    def test_match_axis(self):
        """Test matching axis character to NamedAxis."""
        assert match_axis('L') == LEFT_RIGHT
        assert match_axis('R') == RIGHT_LEFT
        assert match_axis('A') == ANT_POST
        assert match_axis('P') == POST_ANT
        assert match_axis('I') == INF_SUP
        assert match_axis('S') == SUP_INF
        
        # Test lowercase
        assert match_axis('l') == LEFT_RIGHT
        
        # Full-name aliases (R-compatible)
        assert match_axis("LEFT") == LEFT_RIGHT
        assert match_axis("Right") == RIGHT_LEFT
        assert match_axis("ANTERIOR") == ANT_POST
        assert match_axis("Posterior") == POST_ANT
        assert match_axis("INFERIOR") == INF_SUP
        assert match_axis("superior") == SUP_INF
        assert match_axis("left_to_right") == LEFT_RIGHT
        assert match_axis("Right to Left") == RIGHT_LEFT

        # Test invalid
        with pytest.raises(ValueError):
            match_axis('X')
    
    def test_axis_names(self):
        """Test extracting axis names from AxisSet."""
        axis3d = AxisSet3D(
            NamedAxis("x", [1, 0, 0]),
            NamedAxis("y", [0, 1, 0]),
            NamedAxis("z", [0, 0, 1])
        )
        
        names = axis_names(axis3d)
        assert names == ["x", "y", "z"]
    
    def test_predefined_orientations(self):
        """Test predefined orientation lists."""
        # Test 3D orientations
        axial_lpi = OrientationList3D["AXIAL_LPI"]
        assert isinstance(axial_lpi, AxisSet3D)
        assert axial_lpi.i == LEFT_RIGHT
        assert axial_lpi.j == POST_ANT
        assert axial_lpi.k == INF_SUP
        
        # Test 2D orientations
        axial_lp = OrientationList2D["AXIAL_LP"]
        assert isinstance(axial_lp, AxisSet2D)
        assert axial_lp.i == LEFT_RIGHT
        assert axial_lp.j == POST_ANT
    
    def test_axis_set_drop_dim(self):
        """Test dropping dimension from AxisSet."""
        axis3d = AxisSet3D(LEFT_RIGHT, POST_ANT, INF_SUP)
        
        # Drop first dimension
        axis2d = axis3d.drop_dim(0)
        assert isinstance(axis2d, AxisSet2D)
        assert axis2d.i == POST_ANT
        assert axis2d.j == INF_SUP
        
        # Drop last dimension
        axis2d = axis3d.drop_dim(2)
        assert isinstance(axis2d, AxisSet2D)
        assert axis2d.i == LEFT_RIGHT
        assert axis2d.j == POST_ANT

    def test_flip_axis_preserves_direction(self):
        """Test flip_axis keeps axis identity and flips the direction vector."""
        axis3d = find_anatomy_3d("LPI")
        flipped = flip_axis(axis3d, 0)

        assert flipped.i == RIGHT_LEFT
        assert_array_equal(flipped.i.direction, [-1, 0, 0])
        assert_array_equal(flipped.j.direction, [0, 1, 0])
        assert_array_equal(flipped.k.direction, [0, 0, 1])

    def test_permute_axes_preserves_direction(self):
        """Test permute_axes preserves per-axis direction vectors."""
        axis3d = find_anatomy_3d("LPI")
        permuted = permute_axes(axis3d, [2, 0, 1])

        assert permuted.i == INF_SUP
        assert permuted.j == LEFT_RIGHT
        assert permuted.k == POST_ANT
        assert_array_equal(axis_directions(permuted), [np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])])

    def test_drop_axis_preserves_direction(self):
        """Test drop_axis keeps all remaining axis directions."""
        axis3d = find_anatomy_3d("LPI")
        dropped = drop_axis(axis3d, 1)

        assert dropped.i == LEFT_RIGHT
        assert dropped.j == INF_SUP
        assert_array_equal(axis_directions(dropped), [np.array([1, 0, 0]), np.array([0, 0, 1])])

    def test_add_axis_preserves_existing_direction(self):
        """Test add_axis appends a new axis while preserving existing directions."""
        axis3d = find_anatomy_3d("LPI")
        added = add_axis(axis3d, "time")

        assert isinstance(added, AxisSet4D)
        assert added.i == LEFT_RIGHT
        assert added.j == POST_ANT
        assert added.k == INF_SUP
        assert added.l.axis == "time"
        assert_array_equal(axis_directions(added)[0], [1, 0, 0])
        assert_array_equal(axis_directions(added)[1], [0, 1, 0])
        assert_array_equal(axis_directions(added)[2], [0, 0, 1])


class TestNeuroSpaceDimensionManipulation:
    """Test dimension manipulation methods."""
    
    def test_dim_of(self):
        """Test getting dimension length."""
        spc = NeuroSpace((64, 32, 16))
        
        # By index
        assert spc.dim_of(0) == 64
        assert spc.dim_of(1) == 32
        assert spc.dim_of(2) == 16
        
        # By axis name (assuming default x,y,z axes)
        assert spc.dim_of("x") == 64
        assert spc.dim_of("y") == 32
        assert spc.dim_of("z") == 16
        
        # Out of range
        with pytest.raises(ValueError):
            spc.dim_of(3)
    
    def test_which_dim(self):
        """Test finding dimension index by name."""
        spc = NeuroSpace((64, 32, 16))
        
        # Assuming default x,y,z axes
        assert spc.which_dim("x") == 0
        assert spc.which_dim("y") == 1
        assert spc.which_dim("z") == 2
        
        # Non-existent axis
        with pytest.raises(ValueError):
            spc.which_dim("t")
    
    def test_drop_dim(self):
        """Test dropping a dimension."""
        spc3d = NeuroSpace((64, 32, 16), spacing=(2, 2, 2), origin=(-90, -60, -30))
        
        # Drop middle dimension
        spc2d = spc3d.drop_dim(1)
        
        assert spc2d.ndim == 2
        assert_array_equal(spc2d.dim, [64, 16])
        assert_array_equal(spc2d.spacing, [2, 2])
        assert_array_equal(spc2d.origin, [-90, -30])
        assert axis_names(spc2d.axes) == ["x", "z"]
        
        # Cannot drop from 1D
        spc1d = NeuroSpace((64,))
        with pytest.raises(ValueError):
            spc1d.drop_dim(0)
    
    def test_add_dim(self):
        """Test adding dimensions."""
        spc2d = NeuroSpace((64, 32), spacing=(2, 2), origin=(-90, -60))
        
        # Add one dimension
        spc3d = spc2d.add_dim(1, 16)
        
        assert spc3d.ndim == 3
        assert_array_equal(spc3d.dim, [64, 32, 16])
        assert_array_equal(spc3d.spacing, [2, 2, 1])  # New dim gets spacing=1
        assert_array_equal(spc3d.origin, [-90, -60, 0])  # New dim gets origin=0
        assert axis_names(spc3d.axes) == ["x", "y", "v3"]
        
        # Add multiple dimensions
        spc4d = spc2d.add_dim(2, 10)
        assert spc4d.ndim == 4
        assert_array_equal(spc4d.dim, [64, 32, 10, 10])

    def test_add_dim_preserves_transformation(self):
        """add_dim should preserve existing transformation and not reinitialize it."""
        trans = np.array([
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        spc3d = NeuroSpace((10, 20, 30), trans=trans)

        spc4d = spc3d.add_dim(1, size=5)

        expected_trans = np.eye(5)
        expected_trans[:3, :3] = trans[:3, :3]
        expected_trans[:3, 4] = trans[:3, 3]

        assert_array_equal(spc4d.trans, expected_trans)

    def test_drop_dim_preserves_affine_submatrix(self):
        """drop_dim should keep the affine rows/cols for retained dimensions."""
        trans = np.array([
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        spc3d = NeuroSpace((10, 20, 30), trans=trans)

        spc2d = spc3d.drop_dim(1)
        keep = [0, 2]
        expected_trans = np.eye(4)
        expected_trans[:2, :2] = trans[np.ix_(keep, keep)]
        expected_trans[:2, 3] = trans[keep, 3]

        assert_array_equal(spc2d.trans, expected_trans)

    def test_drop_dim_2d_preserves_affine(self):
        trans = np.array([
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        spc2d = NeuroSpace((10, 20), trans=trans)

        spc1d = spc2d.drop_dim(1)

        expected_trans = np.eye(4)
        expected_trans[0, 0] = trans[0, 0]
        expected_trans[0, 3] = trans[0, 3]

        assert spc1d.ndim == 1
        assert_array_equal(spc1d.trans, expected_trans)

    def test_get_subspace_preserves_affine_and_shape(self):
        """get_subspace should preserve affine sub-block and use valid 4x4 transform."""
        trans = np.array([
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        spc3d = NeuroSpace((10, 20, 30), trans=trans)

        spc2d = spc3d.get_subspace([0, 2])
        expected_trans = np.eye(4)
        expected_trans[:2, :2] = trans[[0, 2], :][:, [0, 2]]
        expected_trans[:2, 3] = trans[[0, 2], 3]

        assert_array_equal(spc2d.dim, [10, 30])
        assert_array_equal(spc2d.trans, expected_trans)

    def test_get_subspace_2d_preserves_affine(self):
        trans = np.array([
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        spc2d = NeuroSpace((10, 20), trans=trans)

        sub = spc2d.get_subspace([0])

        expected_trans = np.eye(4)
        expected_trans[0, 0] = trans[0, 0]
        expected_trans[0, 3] = trans[0, 3]

        assert sub.ndim == 1
        assert_array_equal(sub.trans, expected_trans)


class TestNeuroSpaceProperties:
    """Test NeuroSpace property methods."""
    
    def test_bounds(self):
        """Test spatial bounds calculation."""
        spc = NeuroSpace((64, 32, 16), spacing=(2, 3, 4), origin=(10, 20, 30))
        
        bounds = spc.bounds()
        
        # Min bounds should be origin
        assert_array_equal(bounds[0], [10, 20, 30])
        
        # Max bounds should be origin + (dim-1) * spacing
        expected_max = [10 + 63*2, 20 + 31*3, 30 + 15*4]
        assert_array_equal(bounds[1], expected_max)
    
    def test_centroid(self):
        """Test centroid calculation."""
        spc = NeuroSpace((64, 32, 16), spacing=(2, 2, 2), origin=(0, 0, 0))
        
        centroid = spc.centroid()
        
        # Centroid should be at center of grid
        expected = [(64-1)/2 * 2, (32-1)/2 * 2, (16-1)/2 * 2]
        assert_array_almost_equal(centroid, expected)
    
    def test_repr(self):
        """Test string representation."""
        spc = NeuroSpace((64, 32, 16), spacing=(2, 2, 2), origin=(-90, -60, -30))
        
        repr_str = repr(spc)
        assert "NeuroSpace" in repr_str
        assert "dim     : (64, 32, 16)" in repr_str
        assert "spacing : (2.0, 2.0, 2.0)" in repr_str
        assert "origin  : (-90.0, -60.0, -30.0)" in repr_str
        assert "nvoxels : 32768" in repr_str


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_dimensions(self):
        """Test errors for invalid dimensions."""
        # Empty dimensions
        with pytest.raises(Exception):
            NeuroSpace(())
        
        # Negative dimensions
        with pytest.raises(Exception):
            NeuroSpace((-10, 10, 10))
    
    def test_coordinate_dimension_mismatch(self):
        """Test errors for coordinate dimension mismatches."""
        spc = NeuroSpace((64, 64, 64))
        
        # Wrong number of dimensions in coordinates
        with pytest.raises(ValueError):
            spc.coord_to_grid(np.array([10, 20]))  # 2D coords for 3D space
        
        with pytest.raises(ValueError):
            spc.grid_to_coord(np.array([10, 20, 30, 40]))  # 4D grid for 3D space
    
    def test_invalid_transformation_matrix(self):
        """Test error for non-invertible transformation matrix."""
        # Create singular matrix
        trans = np.zeros((4, 4))
        
        with pytest.raises(ValueError):
            NeuroSpace((64, 64, 64), trans=trans)
    
    def test_index_out_of_bounds(self):
        """Test behavior with out-of-bounds indices."""
        spc = NeuroSpace((64, 64, 25))
        
        # Python's unravel_index raises ValueError for out-of-bounds indices
        # This differs from R behavior but is more consistent with Python conventions
        with pytest.raises(ValueError):
            spc.index_to_grid(64*64*25 + 1)  # Beyond max index


# R-Python compatibility notes to include in test output
def test_compatibility_notes():
    """Document key differences between R and Python implementations."""
    notes = """
    R-Python Compatibility Notes:
    1. Indexing: R uses 1-based indexing, Python uses 0-based
       - R grid [1,1,1] → Python grid [0,0,0]
       - R index 1 → Python index 0
    
    2. Array ordering: R uses column-major (Fortran), Python uses row-major (C)
       - This is handled in grid_to_index and index_to_grid conversions
    
    3. Function names: R uses dots, Python uses underscores
       - R: coord.to.grid() → Python: coord_to_grid()
    
    4. Method access: R uses functions, Python uses methods/properties
       - R: dim(space) → Python: space.dim or space.shape
       - R: ndim(space) → Python: space.ndim
    """
    print(notes)


if __name__ == "__main__":
    # Run specific test or all tests
    pytest.main([__file__, "-v"])
