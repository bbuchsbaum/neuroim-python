import numpy as np
import pytest

from neuroim import NeuroSpace
from neuroim.axis import AxisSet3D, NamedAxis
from neuroim.protocols import voxel_coord, world_coord


def test_neurospace_hash_and_equality_for_identical_fields():
    left = NeuroSpace((4, 5, 6), spacing=(2, 2, 3), origin=(10, 20, 30))
    right = NeuroSpace((4, 5, 6), spacing=(2, 2, 3), origin=(10, 20, 30))

    assert left == right
    assert hash(left) == hash(right)


def test_neurospace_defensively_copies_constructor_arrays():
    dim = np.array([4, 5, 6])
    spacing = np.array([2.0, 2.0, 3.0])
    origin = np.array([10.0, 20.0, 30.0])
    trans = np.eye(4)

    space = NeuroSpace(dim, spacing=spacing, origin=origin, trans=trans)

    dim[:] = 99
    spacing[:] = 99
    origin[:] = 99
    trans[:, :] = 99

    np.testing.assert_array_equal(space.dim, [4, 5, 6])
    np.testing.assert_array_equal(space.spacing, [2.0, 2.0, 3.0])
    np.testing.assert_array_equal(space.origin, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(space.trans, np.eye(4))


def test_neurospace_attributes_and_arrays_are_read_only():
    space = NeuroSpace((4, 5, 6))

    with pytest.raises(AttributeError):
        space.dim = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="read-only"):
        space.dim[0] = 10

    with pytest.raises(ValueError, match="read-only"):
        space.trans[0, 0] = 10


def test_typed_coordinate_helpers_preserve_dimensionality():
    space = NeuroSpace((4, 5, 6), spacing=(2, 2, 3), origin=(10, 20, 30))

    world = space.grid_to_world(voxel_coord((1, 2, 3)))
    np.testing.assert_array_equal(world, [12, 24, 39])

    voxel = space.world_to_grid(world_coord(world))
    np.testing.assert_array_equal(voxel, [1, 2, 3])
    assert voxel.dtype.kind == "i"


def test_world_to_grid_accepts_3d_coords_on_4d_space():
    """PAIN-3: world-mm seeding on a 4-D NeuroVec.space must not require drop_dim(3)."""
    affine = np.diag([3.0, 3.0, 3.5, 1.0])
    bold_space = NeuroSpace.from_affine(affine, (32, 32, 24, 40))
    assert bold_space.ndim == 4

    voxel = bold_space.world_to_grid(np.array([15.0, 30.0, 21.0]))
    np.testing.assert_array_equal(voxel, [5, 10, 6])
    assert voxel.ndim == 1
    assert voxel.shape == (3,)


def test_grid_to_world_accepts_3d_grid_on_4d_space():
    """PAIN-3: 3-D voxel input should produce 3-D world output on N-D spaces."""
    affine = np.diag([3.0, 3.0, 3.5, 1.0])
    bold_space = NeuroSpace.from_affine(affine, (32, 32, 24, 40))

    world = bold_space.grid_to_world(np.array([5, 10, 6]))
    np.testing.assert_array_equal(world, [15.0, 30.0, 21.0])
    assert world.shape == (3,)


def test_world_grid_round_trip_on_4d_space_is_identity_at_grid():
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    bold_space = NeuroSpace.from_affine(affine, (16, 16, 12, 24))

    grid = np.array([[3, 4, 5], [0, 0, 0], [15, 15, 11]])
    world = bold_space.grid_to_world(grid)
    grid_back = bold_space.world_to_grid(world)
    np.testing.assert_array_equal(grid_back, grid)


def test_world_to_grid_still_rejects_truly_wrong_shape_on_4d_space():
    affine = np.diag([3.0, 3.0, 3.5, 1.0])
    bold_space = NeuroSpace.from_affine(affine, (32, 32, 24, 40))

    with pytest.raises(ValueError, match="3 for spatial-only on a 4-D space"):
        bold_space.world_to_grid(np.array([1.0, 2.0]))


def test_full_dim_coord_still_works_on_4d_space():
    affine = np.diag([3.0, 3.0, 3.5, 1.0])
    bold_space = NeuroSpace.from_affine(affine, (32, 32, 24, 40))

    voxel_4d = bold_space.coord_to_grid(np.array([[6.0, 9.0, 7.0, 4.0]]))
    assert voxel_4d.shape == (1, 4)


def test_spatial_3d_coord_uses_full_spatial_affine_not_just_diag():
    """The spatial sub-affine should honor a tilted spatial 3x3, not only scaling."""
    # Spatial rotation by 90 deg around z + diag scale, embedded in a 4-D affine
    R = np.array(
        [
            [0.0, -2.0, 0.0, 10.0],
            [2.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    bold_space = NeuroSpace.from_affine(R, (8, 8, 8, 6))
    # Grid [1, 0, 0] -> spatial 4x4 says R @ [1,0,0,1] = [10, 22, 5]
    np.testing.assert_allclose(
        bold_space.grid_to_world(np.array([1, 0, 0])),
        np.array([10.0, 22.0, 5.0]),
    )
    np.testing.assert_array_equal(
        bold_space.world_to_grid(np.array([10.0, 22.0, 5.0])),
        np.array([1, 0, 0]),
    )


def test_axis_value_objects_are_immutable():
    axis = NamedAxis("x", [1, 0, 0])
    axes = AxisSet3D(axis, NamedAxis("y", [0, 1, 0]), NamedAxis("z", [0, 0, 1]))

    with pytest.raises(AttributeError):
        axis.axis = "changed"

    with pytest.raises(ValueError, match="read-only"):
        axis.direction[0] = -1

    with pytest.raises(AttributeError):
        axes.i = NamedAxis("other", [1, 0, 0])
