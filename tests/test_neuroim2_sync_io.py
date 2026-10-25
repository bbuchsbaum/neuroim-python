"""NIfTI I/O contracts carried over from neuroim2's I/O rebuild (ca0045a).

Header metadata survives partial reads, unsupported payloads are refused by
name, scaling is reported and re-used, 5-D singleton series fold to 4-D, and
NIfTI-2 is chosen when NIfTI-1 cannot hold the image.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.io import read_header, read_meta_info, read_vec, read_vol
from neuroim.io import write_vec, write_vol
from neuroim.nifti_utils import as_nifti_header

MNI = np.array(
    [
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _mni_4d(tmp_path):
    data = np.arange(4 * 5 * 6 * 3, dtype=np.int16).reshape(4, 5, 6, 3)
    img = nib.Nifti1Image(data, MNI)
    img.set_sform(MNI, code=4)
    img.set_qform(MNI, code=4)
    img.header.set_xyzt_units("mm", "msec")
    img.header["descrip"] = b"mni run"
    path = tmp_path / "src4d.nii"
    nib.save(img, str(path))
    return str(path)


def _codes(path):
    img = nib.load(path)
    return (
        int(img.header["sform_code"]),
        int(img.header["qform_code"]),
        img.header.get_xyzt_units(),
        bytes(img.header["descrip"]).rstrip(b"\0"),
        img.get_data_dtype(),
    )


@pytest.mark.parametrize(
    "reader",
    [
        lambda p: read_vec(p, indices=[0, 2]),
        lambda p: read_vol(p, index=1),
    ],
    ids=["read_vec_indices", "read_vol_index"],
)
def test_partial_reads_keep_header(tmp_path, reader):
    src = _mni_4d(tmp_path)
    obj = reader(src)
    out = str(tmp_path / "out.nii")
    (write_vec if obj.space.ndim == 4 else write_vol)(obj, out)
    sform, qform, units, descrip, dtype = _codes(out)
    assert (sform, qform) == (4, 4)
    assert units == ("mm", "msec")
    assert descrip == b"mni run"
    assert dtype == np.int16


def test_read_vol_index_values(tmp_path):
    src = _mni_4d(tmp_path)
    np.testing.assert_array_equal(
        read_vol(src, index=2).data, np.asanyarray(nib.load(src).dataobj)[..., 2]
    )


def test_complex_and_rgb_refused(tmp_path):
    cplx = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.complex64), np.eye(4))
    nib.save(cplx, str(tmp_path / "c.nii"))
    with pytest.raises(ValueError, match="Complex"):
        ni.read_image(str(tmp_path / "c.nii"))

    rgb = np.zeros((2, 2, 2), dtype=[("R", "u1"), ("G", "u1"), ("B", "u1")])
    nib.save(nib.Nifti1Image(rgb, np.eye(4)), str(tmp_path / "rgb.nii"))
    with pytest.raises(ValueError, match="RGB"):
        ni.read_image(str(tmp_path / "rgb.nii"))


def test_read_header_reports_scaling(tmp_path):
    img = nib.Nifti1Image(np.arange(8, dtype=np.int16).reshape(2, 2, 2), np.eye(4))
    img.header.set_slope_inter(2.0, 5.0)
    nib.save(img, str(tmp_path / "s.nii"))
    hdr = read_header(str(tmp_path / "s.nii"))
    assert (hdr["scl_slope"], hdr["scl_inter"]) == (2.0, 5.0)


def test_scaled_integer_source_writes_back_in_source_storage(tmp_path):
    img = nib.Nifti1Image(np.arange(27, dtype=np.int16).reshape(3, 3, 3), np.eye(4))
    img.header.set_slope_inter(2.0, -4.0)
    nib.save(img, str(tmp_path / "s.nii"))
    vol = read_vol(str(tmp_path / "s.nii"))
    write_vol(vol, str(tmp_path / "o.nii"))
    out = nib.load(str(tmp_path / "o.nii"))
    assert out.get_data_dtype() == np.int16
    assert (out.dataobj.slope, out.dataobj.inter) == (2.0, -4.0)
    np.testing.assert_array_equal(np.asanyarray(out.dataobj), vol.data)

    # Values that no longer fit the source scaling fall back losslessly.
    write_vol(vol * 0.5, str(tmp_path / "half.nii"))
    half = nib.load(str(tmp_path / "half.nii"))
    np.testing.assert_allclose(np.asanyarray(half.dataobj), vol.data * 0.5)


def test_five_d_singleton_series_folds_to_4d(tmp_path):
    data = np.random.default_rng(0).normal(size=(3, 4, 5, 1, 6)).astype(np.float32)
    img = nib.Nifti1Image(data, MNI)
    img.header.set_zooms((2.0, 2.0, 2.0, 1.0, 1.5))
    nib.save(img, str(tmp_path / "five.nii"))
    vec = ni.read_image(str(tmp_path / "five.nii"))
    assert vec.shape == (3, 4, 5, 6)
    np.testing.assert_allclose(vec.space.spacing, [2.0, 2.0, 2.0, 1.5])
    np.testing.assert_allclose(vec.data, data[:, :, :, 0, :])


def test_five_d_hypervec_refused_clearly(tmp_path):
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 2, 3), dtype=np.float32), np.eye(4))
    nib.save(img, str(tmp_path / "hyper.nii"))
    with pytest.raises(ValueError, match="hyper"):
        read_vec(str(tmp_path / "hyper.nii"))


def test_large_dimension_writes_nifti2(tmp_path):
    vol = ni.DenseNeuroVol(np.zeros((40000, 2, 1)), ni.NeuroSpace((40000, 2, 1)))
    write_vol(vol, str(tmp_path / "big.nii"))
    assert isinstance(nib.load(str(tmp_path / "big.nii")), nib.Nifti2Image)


def test_nifti2_source_round_trips_as_nifti2(tmp_path):
    img = nib.Nifti2Image(np.ones((2, 3, 4), dtype=np.float32), MNI)
    nib.save(img, str(tmp_path / "n2.nii"))
    write_vol(read_vol(str(tmp_path / "n2.nii")), str(tmp_path / "o.nii"))
    out = nib.load(str(tmp_path / "o.nii"))
    assert isinstance(out, nib.Nifti2Image)
    np.testing.assert_allclose(out.affine, MNI)


def test_meta_info_byte_order(tmp_path):
    data = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
    for order in ("<", ">"):
        img = nib.Nifti1Image(data.astype(data.dtype.newbyteorder(order)), np.eye(4))
        path = str(tmp_path / f"e{order == '>'}.nii")
        nib.save(img, path)
        meta = read_meta_info(path)
        assert meta.get_data_dtype() == nib.load(path).get_data_dtype()


def test_as_nifti_header_uses_spatial_affine_and_codes(tmp_path):
    vec = read_vec(_mni_4d(tmp_path))
    hdr = as_nifti_header(vec, file_name="x.nii", data_type="INT64")
    assert (hdr["qform_code"], hdr["sform_code"]) == (4, 4)
    assert (hdr["qoffset_x"], hdr["qoffset_y"], hdr["qoffset_z"]) == (90, -126, -72)
    assert list(hdr["srow_x"]) == list(MNI[0])
    assert hdr["bitpix"] == 64
