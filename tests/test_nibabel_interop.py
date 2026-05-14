"""Tests for explicit nibabel interoperability adapters."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim import NeuroSpace, NeuroVec, NeuroVol


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "golden_tests" / "fixtures"
TINY_BOLD = FIXTURE_DIR / "tiny_bold.nii.gz"
TINY_MASK = FIXTURE_DIR / "tiny_mask.nii.gz"


def _xform_codes(img):
    _, qform_code = img.get_qform(coded=True)
    _, sform_code = img.get_sform(coded=True)
    return int(qform_code), int(sform_code)


def test_neurovol_round_trips_nifti1_fixture():
    img = nib.load(str(TINY_MASK))

    vol = NeuroVol.from_nibabel(img)
    out = vol.to_nibabel()

    assert isinstance(out, nib.Nifti1Image)
    assert out.shape == img.shape
    assert out.get_data_dtype() == img.get_data_dtype()
    assert np.allclose(out.affine, img.affine)
    assert _xform_codes(out) == _xform_codes(img)
    np.testing.assert_array_equal(
        np.asanyarray(out.dataobj), np.asanyarray(img.dataobj)
    )


def test_neurovec_round_trips_nifti1_fixture():
    img = nib.load(str(TINY_BOLD))

    vec = NeuroVec.from_nibabel(img)
    out = vec.to_nibabel()

    assert isinstance(out, nib.Nifti1Image)
    assert out.shape == img.shape
    assert out.get_data_dtype() == img.get_data_dtype()
    assert np.allclose(out.affine, img.affine)
    assert _xform_codes(out) == _xform_codes(img)
    np.testing.assert_array_equal(
        np.asanyarray(out.dataobj), np.asanyarray(img.dataobj)
    )


def test_nifti2_interop_with_cls_override():
    data = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    img = nib.Nifti2Image(data, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=2)

    vol = NeuroVol.from_nibabel(img)
    out = vol.to_nibabel(cls=nib.Nifti2Image)

    assert isinstance(out, nib.Nifti2Image)
    assert out.shape == img.shape
    assert np.allclose(out.affine, img.affine)
    assert _xform_codes(out) == _xform_codes(img)


def test_from_nibabel_dispatches_by_dimensionality():
    vol_img = nib.load(str(TINY_MASK))
    vec_img = nib.load(str(TINY_BOLD))

    assert ni.from_nibabel(vol_img).shape == vol_img.shape
    assert ni.from_nibabel(vec_img).shape == vec_img.shape


def test_space_mismatch_error_names_field_and_values():
    img = nib.load(str(TINY_MASK))
    space = NeuroSpace.from_nibabel(img)
    shifted_affine = img.affine.copy()
    shifted_affine[:3, 3] += np.array([2.0, 0.0, 0.0])
    shifted = NeuroSpace.from_affine(
        shifted_affine,
        img.shape,
        header=img.header,
    )

    with pytest.raises(ValueError, match="affine.*left=.*right="):
        space.compatible_with(shifted)


def test_neurovec_from_nibabel_uses_dataobj_without_get_fdata():
    class ProxyImage:
        def __init__(self):
            self.dataobj = np.zeros((3, 4, 2, 5), dtype=np.float32)
            self.shape = self.dataobj.shape
            self.affine = np.eye(4)
            self.header = None

        def get_fdata(self):
            raise AssertionError("from_nibabel should not call get_fdata eagerly")

    vec = NeuroVec.from_nibabel(ProxyImage(), lazy=True)

    assert vec.shape == (3, 4, 2, 5)
    assert vec.data.dtype == np.float32


def test_lazy_adapter_does_not_call_get_fdata():
    class Header:
        def get_zooms(self):
            return (1.0, 1.0, 1.0, 1.0)

        def copy(self):
            return self

    class Image:
        shape = (3, 4, 2, 5)
        affine = np.eye(4)
        header = Header()
        dataobj = np.zeros(shape, dtype=np.float32)

        def get_fdata(self):
            raise AssertionError("get_fdata should not be called for lazy=True")

    vec = NeuroVec.from_nibabel(Image(), lazy=True)

    assert vec.shape == Image.shape
    assert vec.data.dtype == np.float32
