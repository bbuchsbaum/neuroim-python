"""World-coordinate regression tests for connected-component peak pruning."""

import numpy as np
import pytest

from neuroim import DenseNeuroVol, NeuroSpace
from neuroim.connected_components import conn_comp


def _peak_values(data: np.ndarray, space: NeuroSpace, distance: float) -> np.ndarray:
    result = conn_comp(
        DenseNeuroVol(data, space),
        threshold=0.5,
        cluster_table=False,
        local_maxima=True,
        local_maxima_dist=distance,
    )
    assert result.local_maxima is not None
    return result.local_maxima[:, 4]


def test_anisotropic_pruning_removes_peaks_two_mm_apart() -> None:
    data = np.zeros((7, 7, 7), dtype=float)
    data[2, 3, 3] = 10.0
    data[3, 3, 3] = 1.0
    data[4, 3, 3] = 9.0
    space = NeuroSpace(data.shape, spacing=(1.0, 1.0, 4.0))

    values = _peak_values(data, space, distance=3.0)

    np.testing.assert_array_equal(values, [10.0])


def test_anisotropic_pruning_keeps_peaks_four_mm_apart() -> None:
    data = np.zeros((7, 7, 7), dtype=float)
    data[3, 3, 2] = 10.0
    data[3, 3, 3] = 10.0
    space = NeuroSpace(data.shape, spacing=(1.0, 1.0, 4.0))

    values = _peak_values(data, space, distance=3.0)

    np.testing.assert_array_equal(values, [10.0, 10.0])


def test_sheared_affine_pruning_uses_full_world_transform() -> None:
    data = np.zeros((7, 7, 7), dtype=float)
    data[2, 4, 3] = 10.0
    data[3, 3, 3] = 1.0
    data[4, 2, 3] = 9.0
    affine = np.array(
        [
            [1.0, 0.9, 0.0, 13.0],
            [0.0, np.sqrt(0.19), 0.0, -7.0],
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    space = NeuroSpace(data.shape, trans=affine)
    peak_world = space.grid_to_coord(np.array([[2, 4, 3], [4, 2, 3]]))
    assert np.linalg.norm(peak_world[0] - peak_world[1]) < 1.0

    values = _peak_values(data, space, distance=1.0)

    np.testing.assert_array_equal(values, [10.0])


@pytest.mark.parametrize("distance", [np.nan, np.inf, -np.inf, -1.0, [1.0]])
def test_local_maxima_distance_must_be_finite_nonnegative(distance: object) -> None:
    data = np.zeros((3, 3, 3), dtype=float)
    data[1, 1, 1] = 2.0
    vol = DenseNeuroVol(data, NeuroSpace(data.shape))

    with pytest.raises(ValueError, match="local_maxima_dist"):
        conn_comp(
            vol,
            threshold=0.5,
            cluster_table=False,
            local_maxima_dist=distance,
        )
