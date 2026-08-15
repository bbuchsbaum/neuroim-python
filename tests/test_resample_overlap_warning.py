"""Regression tests for misplaced resampling-target diagnostics."""

import warnings

import numpy as np
import pytest

from neuroim import DenseNeuroVol, NeuroSpace
from neuroim.resample import _target_overlap_fraction, resample


def _volume(space: NeuroSpace) -> DenseNeuroVol:
    return DenseNeuroVol(np.ones(tuple(space.dim[:3]), dtype=float), space)


def _assert_no_overlap_warning(source: DenseNeuroVol, target: NeuroSpace) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resample(source, target, interpolation=0)
    assert not [
        warning for warning in caught if "covers little" in str(warning.message)
    ]


def test_clear_world_overlap_does_not_warn() -> None:
    space = NeuroSpace((10, 10, 10), spacing=(2.0, 2.0, 2.0))
    source = _volume(space)

    _assert_no_overlap_warning(source, space)


def test_low_world_overlap_warns_but_resampling_still_completes() -> None:
    source_space = NeuroSpace((10, 10, 10))
    source = _volume(source_space)
    target = NeuroSpace((10, 10, 10), origin=(9.6, 0.0, 0.0))
    fraction, _, _ = _target_overlap_fraction(source_space, target)
    assert fraction == pytest.approx(0.04)

    with pytest.warns(UserWarning, match="covers little.*4.0%"):
        result = resample(source, target, interpolation=0)

    assert tuple(result.shape) == (10, 10, 10)
    assert result.space.trans.tolist() == target.trans.tolist()


def test_disjoint_world_boxes_warn_with_zero_overlap() -> None:
    source_space = NeuroSpace((10, 10, 10))
    source = _volume(source_space)
    target = NeuroSpace((10, 10, 10), origin=(20.0, 0.0, 0.0))

    with pytest.warns(UserWarning, match="covers little.*0.0%"):
        resample(source, target, interpolation=1)


@pytest.mark.parametrize("kind", ["flip", "permutation"])
def test_equivalent_flipped_or_permuted_world_coverage_does_not_warn(
    kind: str,
) -> None:
    source_space = NeuroSpace((4, 5, 6))
    source = _volume(source_space)
    if kind == "flip":
        affine = np.array(
            [
                [-1.0, 0.0, 0.0, 3.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        target = NeuroSpace((4, 5, 6), trans=affine)
    else:
        affine = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        target = NeuroSpace((5, 4, 6), trans=affine)

    fraction, _, _ = _target_overlap_fraction(source_space, target)
    assert fraction == pytest.approx(1.0)
    _assert_no_overlap_warning(source, target)
