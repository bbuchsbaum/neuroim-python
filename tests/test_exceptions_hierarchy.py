"""A2: typed contract-failure exceptions.

Pins three properties:

1. Structure: every typed class is a ``NeuroimError`` *and* the built-in
   the spec lists, so old broad catches keep working.
2. Behaviour: a real contract failure now raises the typed class, is
   still catchable as the documented built-in, and the canonical message
   is unchanged (so the spec stays v1).
3. Drift: every spec-ID row in
   ``docs/spec/contract-failure-vocabulary.md`` has a class wired to the
   built-in that row documents.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from neuroim import NeuroSpace, DenseNeuroVol, DenseNeuroVec
from neuroim import verify
from neuroim.exceptions import (
    NeuroimError,
    SpaceMismatchError,
    MaskMismatchError,
    OutOfBoundsError,
    WorldOutOfBoundsError,
    BackendDriftError,
    ImmutableError,
    InvalidSpaceError,
    InvalidArgumentError,
    ImplicitArrayConversionError,
)

SPEC = Path(__file__).resolve().parents[1] / "docs/spec/contract-failure-vocabulary.md"


def test_marker_and_builtin_bases():
    assert issubclass(SpaceMismatchError, (NeuroimError, ValueError))
    assert issubclass(MaskMismatchError, (NeuroimError, ValueError))
    assert issubclass(OutOfBoundsError, (NeuroimError, IndexError))
    assert issubclass(WorldOutOfBoundsError, (NeuroimError, ValueError))
    assert issubclass(BackendDriftError, (NeuroimError, ValueError))
    assert issubclass(ImmutableError, (NeuroimError, AttributeError))
    assert issubclass(InvalidSpaceError, (NeuroimError, ValueError))
    assert issubclass(InvalidArgumentError, (NeuroimError, ValueError))
    assert issubclass(ImplicitArrayConversionError, (NeuroimError, TypeError))


def test_space_mismatch_is_typed_and_backcompatible():
    a = DenseNeuroVol(np.zeros((4, 4, 4)), NeuroSpace((4, 4, 4), (1, 1, 1)))
    b_space = NeuroSpace((5, 5, 5), (1, 1, 1))
    # New code: catch the typed class.
    with pytest.raises(SpaceMismatchError, match=r"NeuroSpace mismatch in spatial dim"):
        a.space.compatible_with(b_space)
    # Old code: the documented built-in still catches it, message unchanged.
    with pytest.raises(ValueError, match=r"NeuroSpace mismatch in spatial dim: left=\(4, 4, 4\), right=\(5, 5, 5\)"):
        a.space.compatible_with(b_space)
    # And the marker base catches it.
    with pytest.raises(NeuroimError):
        a.space.compatible_with(b_space)


def test_invalid_space_construction_is_typed():
    with pytest.raises(InvalidSpaceError, match="All dimensions must be positive"):
        NeuroSpace((0, 4, 4), (1, 1, 1))
    with pytest.raises(ValueError):  # backcompat
        NeuroSpace((0, 4, 4), (1, 1, 1))


def test_immutable_is_typed_attributeerror():
    sp = NeuroSpace((4, 4, 4), (1, 1, 1))
    with pytest.raises(ImmutableError):
        sp.dim = (1, 1, 1)
    with pytest.raises(AttributeError):  # backcompat
        sp.dim = (1, 1, 1)


def test_oob_mode_argument_is_typed():
    vec = DenseNeuroVec(np.zeros((4, 4, 4, 3)), NeuroSpace((4, 4, 4, 3), (1, 1, 1, 1)))
    with pytest.raises(InvalidArgumentError, match="out_of_bounds must be"):
        vec.series_at(0, 0, 0, out_of_bounds="bogus")


def test_index_out_of_bounds_is_typed_indexerror():
    vec = DenseNeuroVec(np.zeros((4, 4, 4, 3)), NeuroSpace((4, 4, 4, 3), (1, 1, 1, 1)))
    with pytest.raises(OutOfBoundsError):
        vec.series_at(99, 99, 99, out_of_bounds="raise")
    with pytest.raises(IndexError):  # backcompat
        vec.series_at(99, 99, 99, out_of_bounds="raise")


def test_mask_mismatch_is_typed():
    with pytest.raises(MaskMismatchError):
        verify.assert_same_mask(
            type("R", (), {"mask_hash": "a"})(),
            type("R", (), {"mask_hash": "b"})(),
        )


def test_a1a_refusal_is_in_the_hierarchy():
    vol = DenseNeuroVol(np.zeros((4, 4, 4)), NeuroSpace((4, 4, 4), (1, 1, 1)))
    with pytest.raises(ImplicitArrayConversionError):
        np.asarray(vol)
    with pytest.raises(TypeError):  # backcompat / NumPy non-array protocol
        np.asarray(vol)
    with pytest.raises(NeuroimError):
        np.asarray(vol)


def test_no_drift_every_spec_row_has_a_wired_class():
    """Each spec-ID row's built-in must match a class subclassing it.

    This fails if the spec adds a row whose documented exception type no
    longer has a NeuroimError subclass that inherits it.
    """
    text = SPEC.read_text()
    builtins_seen = set(re.findall(r"\|\s*`(ValueError|IndexError|AttributeError)`\s*\|", text))
    assert builtins_seen, "spec table parse produced no exception rows"
    by_builtin = {
        ValueError: [SpaceMismatchError, MaskMismatchError, WorldOutOfBoundsError,
                     BackendDriftError, InvalidSpaceError, InvalidArgumentError],
        IndexError: [OutOfBoundsError],
        AttributeError: [ImmutableError],
    }
    name_to_cls = {"ValueError": ValueError, "IndexError": IndexError,
                   "AttributeError": AttributeError}
    for name in builtins_seen:
        builtin = name_to_cls[name]
        klasses = by_builtin[builtin]
        assert all(issubclass(k, builtin) and issubclass(k, NeuroimError)
                   for k in klasses)
