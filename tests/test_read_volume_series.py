"""B1: read_volume() / read_series() intent-revealing readers.

These are additive wrappers over read_image's type= dispatch. The point
of the mote is that they ENCODE the documented 4-D rank contract
(docs/concepts/images.qmd), not that they are a cosmetic rename:

- read_volume on a 4-D file extracts a single 3-D volume (index 0) —
  it does NOT refuse, and it is NOT a guess.
- read_series on a 3-D file promotes to singleton-time 4-D, shape
  (..., 1).
"""

import numpy as np
import nibabel as nib
import pytest

import neuroim as ni
from neuroim import read_volume, read_series
from neuroim.neuro_vol import DenseNeuroVol
from neuroim.neuro_vec import DenseNeuroVec


@pytest.fixture
def vol3d(tmp_path):
    data = np.random.rand(5, 6, 7).astype(np.float32)
    path = tmp_path / "vol3d.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path, data


@pytest.fixture
def vec4d(tmp_path):
    data = np.random.rand(5, 6, 7, 10).astype(np.float32)
    path = tmp_path / "vec4d.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path, data


def test_public_top_level_names():
    """B1's value is intent-revealing PUBLIC names, not submodule-only."""
    assert "read_volume" in ni.__all__
    assert "read_series" in ni.__all__
    assert ni.read_volume is read_volume and ni.read_series is read_series


def test_read_volume_3d_returns_vol(vol3d):
    path, data = vol3d
    vol = read_volume(path)
    assert isinstance(vol, DenseNeuroVol)
    assert vol.shape == (5, 6, 7)
    np.testing.assert_allclose(vol.as_array(), data, rtol=1e-5)


def test_read_volume_on_4d_extracts_index0_not_refuses(vec4d):
    """The rank contract: 4-D is NOT refused; volume reader extracts idx0."""
    path, data = vec4d
    vol = read_volume(path)  # no exception
    assert isinstance(vol, DenseNeuroVol)
    assert vol.shape == (5, 6, 7)
    np.testing.assert_allclose(vol.as_array(), data[..., 0], rtol=1e-5)


def test_read_volume_index_selects_volume(vec4d):
    path, data = vec4d
    vol = read_volume(path, index=3)
    np.testing.assert_allclose(vol.as_array(), data[..., 3], rtol=1e-5)


def test_read_series_4d_returns_vec(vec4d):
    path, data = vec4d
    vec = read_series(path)
    assert isinstance(vec, DenseNeuroVec)
    assert vec.shape == (5, 6, 7, 10)


def test_read_series_on_3d_promotes_to_singleton_time(vol3d):
    """The rank contract: 3-D is promoted to (..., 1), not refused."""
    path, data = vol3d
    vec = read_series(path)
    assert isinstance(vec, DenseNeuroVec)
    assert vec.shape == (5, 6, 7, 1)


def test_not_cosmetic_no_stringly_type_arg():
    """The wrappers must not re-expose the stringly-typed type= smell."""
    import inspect

    for fn in (read_volume, read_series):
        params = inspect.signature(fn).parameters
        assert "type" not in params, f"{fn.__name__} re-exposes type="
