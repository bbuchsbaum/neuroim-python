import numpy as np
import pytest

from neuroim import NeuroSpace
from neuroim.axis import AxisSet3D, NamedAxis
from neuroim.typing import voxel_coord, world_coord


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


def test_axis_value_objects_are_immutable():
    axis = NamedAxis("x", [1, 0, 0])
    axes = AxisSet3D(axis, NamedAxis("y", [0, 1, 0]), NamedAxis("z", [0, 0, 1]))

    with pytest.raises(AttributeError):
        axis.axis = "changed"

    with pytest.raises(ValueError, match="read-only"):
        axis.direction[0] = -1

    with pytest.raises(AttributeError):
        axes.i = NamedAxis("other", [1, 0, 0])
