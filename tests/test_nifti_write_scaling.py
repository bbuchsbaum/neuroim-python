"""NIfTI writer contracts for dtype preservation and scaled integers."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni


AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, -8.0],
        [0.0, 3.0, 0.0, 12.0],
        [0.0, 0.0, 4.0, 20.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@pytest.mark.parametrize(
    "dtype", [np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
)
def test_default_write_preserves_representable_source_integer_dtype(tmp_path, dtype):
    data = np.arange(60, dtype=dtype).reshape((3, 4, 5), order="F")
    vol = ni.DenseNeuroVol(data, ni.NeuroSpace(data.shape, trans=AFFINE))
    path = tmp_path / f"default-{np.dtype(dtype).name}.nii"

    ni.write_vol(vol, path)
    image = nib.load(path)

    assert image.header.get_data_dtype() == np.dtype(dtype)
    np.testing.assert_array_equal(np.asanyarray(image.dataobj), data)


def test_default_write_preserves_source_header_metadata_and_xform_codes(tmp_path):
    data = np.arange(60, dtype=np.int16).reshape((3, 4, 5), order="F")
    header = nib.Nifti1Header()
    header.set_data_dtype(np.int16)
    header["descrip"] = b"source metadata"
    header.set_xyzt_units("mm", "sec")
    source = nib.Nifti1Image(data, AFFINE, header=header)
    source.set_qform(AFFINE, code=2)
    source.set_sform(AFFINE, code=4)
    vol = ni.DenseNeuroVol.from_nibabel(source)
    path = tmp_path / "metadata.nii.gz"

    ni.write_vol(vol, path)
    image = nib.load(path)

    assert image.header.get_data_dtype() == np.dtype(np.int16)
    assert bytes(image.header["descrip"]).rstrip(b"\x00") == b"source metadata"
    assert image.header.get_xyzt_units() == ("mm", "sec")
    assert int(image.header["qform_code"]) == 2
    assert int(image.header["sform_code"]) == 4
    np.testing.assert_allclose(image.affine, AFFINE)
    np.testing.assert_array_equal(np.asanyarray(image.dataobj), data)


@pytest.mark.parametrize(
    "data_type,dtype", [("INT16", np.int16), ("UINT16", np.uint16)]
)
def test_fractional_data_uses_scaled_integer_quantization(tmp_path, data_type, dtype):
    lo = -3.7 if np.issubdtype(dtype, np.signedinteger) else 0.2
    data = np.linspace(lo, 3.7, 60, dtype=np.float64).reshape((3, 4, 5), order="F")
    vol = ni.DenseNeuroVol(data, ni.NeuroSpace(data.shape, trans=AFFINE))
    path = tmp_path / f"scaled-{data_type.lower()}.nii"

    ni.write_vol(vol, path, data_type=data_type)
    image = nib.load(path)
    restored = np.asanyarray(image.dataobj)

    assert image.header.get_data_dtype() == np.dtype(dtype)
    assert image.dataobj.slope is not None
    assert np.isfinite(image.dataobj.slope)
    assert image.dataobj.slope > 0
    max_error = float(np.max(np.abs(restored - data)))
    assert max_error <= abs(float(image.dataobj.slope)) / 2 + 1e-7
    assert max_error < 1e-3


def test_integral_float_data_is_stored_exactly_without_scaling_error(tmp_path):
    data = np.arange(-30, 30, dtype=np.float64).reshape((3, 4, 5), order="F")
    vol = ni.DenseNeuroVol(data, ni.NeuroSpace(data.shape, trans=AFFINE))
    path = tmp_path / "exact-integral-int16.nii"

    ni.write_vol(vol, path, data_type="INT16")
    image = nib.load(path)

    assert image.header.get_data_dtype() == np.dtype(np.int16)
    np.testing.assert_array_equal(np.asanyarray(image.dataobj), data)


def test_write_vec_uses_same_scaled_integer_contract(tmp_path):
    data = np.linspace(-2.25, 5.75, 120, dtype=np.float64).reshape(
        (3, 4, 5, 2), order="F"
    )
    trans = np.eye(5)
    trans[:3, :3] = AFFINE[:3, :3]
    trans[:3, 4] = AFFINE[:3, 3]
    trans[3, 3] = 2.0
    space = ni.NeuroSpace(data.shape, trans=trans)
    vec = ni.DenseNeuroVec(data, space)
    path = tmp_path / "scaled-vec.nii.gz"

    ni.write_vec(vec, path, data_type="INT16")
    image = nib.load(path)
    restored = np.asanyarray(image.dataobj)

    assert image.shape == data.shape
    assert image.header.get_data_dtype() == np.dtype(np.int16)
    assert image.header.get_zooms() == pytest.approx((2.0, 3.0, 4.0, 2.0))
    assert np.max(np.abs(restored - data)) <= image.dataobj.slope / 2 + 1e-7


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_integer_output_rejects_nonfinite_values(tmp_path, bad):
    data = np.arange(60, dtype=float).reshape((3, 4, 5), order="F")
    data[1, 2, 3] = bad
    vol = ni.DenseNeuroVol(data, ni.NeuroSpace(data.shape, trans=AFFINE))

    with pytest.raises(ValueError, match="finite"):
        ni.write_vol(vol, tmp_path / "nonfinite.nii", data_type="INT16")
