"""A1a: implicit NumPy conversion of a typed image must refuse explicitly.

Before this change ``np.asarray(vol)`` returned a 0-d ``object`` array of
shape ``()`` — silent, wrong-shaped garbage. The typed-image contract
(docs/concepts/images.qmd) says implicit array conversion is refused so
the spatial frame is never dropped. These tests pin the explicit refusal
and that the documented explicit accessor still works.
"""

import numpy as np
import pytest

from neuroim import NeuroSpace, DenseNeuroVol, DenseNeuroVec
from neuroim.neuro_hypervec import DenseNeuroHyperVec


def _vol():
    return DenseNeuroVol(np.arange(64, dtype=float).reshape(4, 4, 4),
                         NeuroSpace((4, 4, 4), (1, 1, 1)))


def _vec():
    return DenseNeuroVec(np.zeros((4, 4, 4, 3), dtype=float),
                         NeuroSpace((4, 4, 4, 3), (1, 1, 1, 1)))


def _hypervec():
    return DenseNeuroHyperVec(np.zeros((4, 4, 4, 3, 2), dtype=float),
                              NeuroSpace((4, 4, 4, 3, 2), (1, 1, 1, 1, 1)))


@pytest.mark.parametrize("factory,accessor", [
    (_vol, ".as_array()"),
    (_vec, ".as_matrix()"),
    (_hypervec, ".data"),
])
def test_np_asarray_refuses_explicitly(factory, accessor):
    obj = factory()
    with pytest.raises(TypeError, match=r"does not implicitly convert to a NumPy array"):
        np.asarray(obj)
    with pytest.raises(TypeError, match=re_escape(accessor)):
        np.array(obj)


def re_escape(s: str) -> str:
    import re
    return re.escape(s)


def test_no_silent_zero_d_object_array():
    """The exact pre-fix footgun: np.asarray must NOT yield shape ()."""
    vol = _vol()
    try:
        arr = np.asarray(vol)
    except TypeError:
        return  # refused — correct
    pytest.fail(f"np.asarray returned {arr!r} (shape {arr.shape}) instead of refusing")


def test_explicit_accessors_still_return_full_array():
    vol = _vol()
    a = vol.as_array()
    assert isinstance(a, np.ndarray) and a.shape == (4, 4, 4)
    vec = _vec()
    m = vec.as_matrix()
    assert isinstance(m, np.ndarray) and m.ndim == 2
    hv = _hypervec()
    assert isinstance(hv.data, np.ndarray) and hv.data.shape == (4, 4, 4, 3, 2)
