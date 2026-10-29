"""AFNI / binary I/O contracts from the second correctness audit.

nibabel's own AFNI reader and its bundled real AFNI datasets are the oracle.
"""

from __future__ import annotations

import os
import shutil
import struct

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.afni_io import read_niml_data, write_afni_pair
from neuroim.binary_io import BinaryReader

try:
    from nibabel.testing import data_path as NIB_DATA
except ImportError:  # pragma: no cover
    NIB_DATA = None


def _nib_file(tmp_path, name):
    if NIB_DATA is None or not os.path.exists(os.path.join(NIB_DATA, name + ".HEAD")):
        pytest.skip("nibabel AFNI test data not available")
    for f in os.listdir(NIB_DATA):
        if f.startswith(name + "."):
            shutil.copy(os.path.join(NIB_DATA, f), tmp_path / f)
    return str(tmp_path / (name + ".HEAD"))


@pytest.mark.parametrize("name", ["scaled+tlrc", "example4d+orig"])
def test_real_afni_matches_nibabel(tmp_path, name):
    path = _nib_file(tmp_path, name)
    ref = nib.load(path)
    obj = ni.read_image(path)
    np.testing.assert_allclose(obj.space.affine, ref.affine, atol=1e-4)
    np.testing.assert_allclose(obj.data, np.squeeze(ref.get_fdata()), rtol=1e-6)
    if obj.space.ndim == 4:
        assert obj.space.spacing[3] == pytest.approx(ref.header.get_zooms()[3])


def test_written_afni_loads_in_nibabel_with_same_geometry(tmp_path):
    affine = np.array(
        [[-2.0, 0, 0, 90], [0, 3.0, 0, -20], [0, 0, 4.0, -30], [0, 0, 0, 1]]
    )
    data = np.random.default_rng(0).random((3, 4, 5, 2)).astype(np.float32)
    write_afni_pair(str(tmp_path / "t+orig"), data, affine=affine, tr=2.5)
    ref = nib.load(str(tmp_path / "t+orig.HEAD"))
    np.testing.assert_allclose(ref.affine, affine)
    assert ref.header.get_zooms()[3] == pytest.approx(2.5)
    np.testing.assert_allclose(ref.get_fdata(), data)
    back = ni.read_vec(str(tmp_path / "t+orig.HEAD"))
    np.testing.assert_allclose(back.space.affine, affine)


def test_write_vol_afni_round_trip_keeps_world_position(tmp_path):
    sp = ni.NeuroSpace((5, 6, 7), spacing=(2, 3, 4), origin=(-10, -20, -30))
    vol = ni.DenseNeuroVol(np.random.default_rng(1).random((5, 6, 7)), sp)
    ni.write_vol(vol, str(tmp_path / "w+orig.HEAD"), format="AFNI")
    ref = nib.load(str(tmp_path / "w+orig.HEAD"))
    np.testing.assert_allclose(ref.affine, sp.affine)


def test_integer_afni_output_is_scaled_not_truncated(tmp_path):
    data = np.random.default_rng(2).normal(0, 0.4, (4, 4, 4))
    vol = ni.DenseNeuroVol(data, ni.NeuroSpace((4, 4, 4)))
    ni.write_vol(vol, str(tmp_path / "s+orig.HEAD"), format="AFNI", data_type="INT16")
    back = ni.read_vol(str(tmp_path / "s+orig.HEAD"))
    assert np.abs(back.data - data).max() < 1e-4


def test_quoted_msb_first_and_mixed_brick_types(tmp_path):
    head = tmp_path / "m+orig.HEAD"
    head.write_text(
        "\ntype = integer-attribute\nname = DATASET_DIMENSIONS\ncount = 5\n2 2 2 0 0\n"
        "\ntype = integer-attribute\nname = DATASET_RANK\ncount = 2\n3 2\n"
        "\ntype = float-attribute\nname = IJK_TO_DICOM\ncount = 12\n"
        "1 0 0 0 0 1 0 0 0 0 1 0\n"
        "\ntype = integer-attribute\nname = BRICK_TYPES\ncount = 2\n1 3\n"
        "\ntype = string-attribute\nname = BYTEORDER_STRING\ncount = 10\n'MSB_FIRST~\n"
    )
    b0 = np.arange(8, dtype=">i2")
    b1 = (np.arange(8) / 2 + 0.5).astype(">f4")
    (tmp_path / "m+orig.BRIK").write_bytes(b0.tobytes() + b1.tobytes())
    vec = ni.read_vec(str(head))
    np.testing.assert_allclose(vec.data[..., 0].ravel(order="F"), np.arange(8))
    np.testing.assert_allclose(vec.data[..., 1].ravel(order="F"), b1.astype(float))


def test_niml_double_payload():
    import io

    values = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    payload = io.BytesIO(struct.pack("<6d", *values))
    meta = {"ni_type": "2*double", "ni_dimen": "3", "ni_form": "binary.lsbfirst"}
    out = read_niml_data(payload, meta)
    np.testing.assert_allclose(np.sort(out.ravel()), values)


@pytest.mark.parametrize(
    "dtype_name,bpe,expected",
    [("double", 8, np.float64), ("integer", 4, np.int32), ("short", 2, np.int16)],
)
def test_binary_reader_r_type_names(tmp_path, dtype_name, bpe, expected):
    arr = np.arange(3).astype(expected)
    path = tmp_path / "b.bin"
    path.write_bytes(arr.astype("<" + np.dtype(expected).str[1:]).tobytes())
    with BinaryReader(str(path), 0, dtype_name, bpe) as reader:
        np.testing.assert_array_equal(reader.read_elements(3), arr)
    with pytest.raises(ValueError):
        BinaryReader(str(path), 0, "nonsense", 4).read_elements(1)
