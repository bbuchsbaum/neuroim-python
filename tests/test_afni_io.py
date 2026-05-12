"""Tests for AFNI HEAD/BRIK support in io.py and file_format.py."""

import gzip
import struct
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytest

from neuroimpy import read_meta_info, read_header, read_vol, read_vec, write_vol, write_vec
from neuroimpy import DenseNeuroVol, DenseNeuroVec, NeuroSpace
from neuroimpy.afni_io import parse_niml_element, parse_niml_file
from neuroimpy.meta_info import AFNIMetaInfo


def _write_afni_pair(
    out_dir: Path,
    stem: str,
    data: np.ndarray,
    *,
    float_facs=None,
    gzip_data: bool = False,
    ijk_to_dicom: Optional[Sequence[float]] = None,
    include_ijk_to_dicom: bool = True,
    include_brick_types: bool = True,
) -> Path:
    if ijk_to_dicom is None:
        ijk_to_dicom = [2.0, 0.0, 0.0, 10.0, 0.0, -2.0, 0.0, 20.0, 0.0, 0.0, 3.0, 30.0]

    if data.ndim == 3:
        dims = data.shape
        nvols = 1
    else:
        dims = data.shape[:3]
        nvols = data.shape[3]

    if float_facs is None:
        float_facs = [1.0] * nvols

    brick_types = " ".join(["3"] * nvols)
    facs = " ".join(str(x) for x in float_facs)
    labels = "~" + "~".join([f"vol{i}" for i in range(nvols)]) + "~"

    head = out_dir / f"{stem}.HEAD"
    brik = out_dir / f"{stem}.BRIK"
    if gzip_data:
        brik = out_dir / f"{stem}.BRIK.gz"

    head_txt = (
        "type = integer-attribute\n"
        "name = DATASET_DIMENSIONS\n"
        "count = 5\n"
        f"{dims[0]} {dims[1]} {dims[2]} 1 0\n\n"
        "type = integer-attribute\n"
        "name = DATASET_RANK\n"
        "count = 2\n"
        f"3 {nvols}\n\n"
        "type = float-attribute\n"
        "name = DELTA\n"
        "count = 3\n"
        "2.0 -2.0 3.0\n\n"
        "type = float-attribute\n"
        "name = ORIGIN\n"
        "count = 3\n"
        "10.0 20.0 30.0\n\n"
    )
    if include_brick_types:
        head_txt += (
            "type = integer-attribute\n"
            "name = BRICK_TYPES\n"
            f"count = {nvols}\n"
            f"{brick_types}\n\n"
        )
    head_txt += (
        "type = float-attribute\n"
        "name = BRICK_FLOAT_FACS\n"
        f"count = {nvols}\n"
        f"{facs}\n\n"
        "type = string-attribute\n"
        "name = BYTEORDER_STRING\n"
        "count = 10\n"
        "~LSB_FIRST~\n\n"
    )
    if include_ijk_to_dicom and ijk_to_dicom is not None:
        ijk_count = len(ijk_to_dicom)
        head_txt += (
            "type = float-attribute\n"
            "name = IJK_TO_DICOM\n"
            f"count = {ijk_count}\n"
            + " ".join(f"{x}" for x in ijk_to_dicom)
            + "\n\n"
        )
    head_txt += (
        "type = string-attribute\n"
        "name = BRICK_LABS\n"
        f"count = {len(labels)}\n"
        f"{labels}\n"
    )

    head.write_text(head_txt, encoding="utf-8")
    raw = np.asarray(data, dtype=np.float32).ravel(order="F").tobytes()
    if gzip_data:
        with gzip.open(brik, "wb") as f:
            f.write(raw)
    else:
        brik.write_bytes(raw)
    return head


def test_read_meta_info_afni(tmp_path):
    data = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
    head = _write_afni_pair(tmp_path, "mini+orig", data, float_facs=[1.0, 2.0])

    meta = read_meta_info(head)
    assert isinstance(meta, AFNIMetaInfo)
    assert meta.dims == (4, 3, 2, 2)
    assert meta.spacing == (2.0, 2.0, 3.0)
    assert meta.origin == (10.0, 20.0, 30.0)
    np.testing.assert_array_equal(np.asarray(meta.slope), np.asarray([1.0, 2.0]))


def test_read_meta_info_afni_msb_first_without_tildes(tmp_path):
    data = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
    head = _write_afni_pair(tmp_path, "mini_no_tilde+orig", data, float_facs=[1.0, 2.0])

    text = head.read_text(encoding="utf-8")
    head.write_text(text.replace("~LSB_FIRST~", "MSB_FIRST"), encoding="utf-8")

    meta = read_meta_info(head)
    assert meta.endian == "big"


def test_read_meta_info_afni_missing_ijk_to_dicom(tmp_path):
    data = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2), order="F")
    head = _write_afni_pair(
        tmp_path,
        "no_ijk+orig",
        data,
        include_ijk_to_dicom=False,
    )

    with pytest.raises(ValueError, match="Invalid IJK_TO_DICOM transformation in AFNI header"):
        read_meta_info(head)


def test_read_meta_info_afni_short_ijk_to_dicom(tmp_path):
    data = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2), order="F")
    head = _write_afni_pair(
        tmp_path,
        "short_ijk+orig",
        data,
        ijk_to_dicom=[2.0, 0.0, 0.0, 10.0, 0.0, -2.0],
    )

    with pytest.raises(ValueError, match="Invalid IJK_TO_DICOM transformation in AFNI header"):
        read_meta_info(head)


def test_read_meta_info_afni_nan_ijk_to_dicom(tmp_path):
    data = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2), order="F")
    head = _write_afni_pair(
        tmp_path,
        "nan_ijk+orig",
        data,
        ijk_to_dicom=[2.0, 0.0, 0.0, 10.0, 0.0, float("nan"), 0.0, 20.0, 0.0, 0.0, 3.0, 30.0],
    )

    with pytest.raises(ValueError, match="Invalid IJK_TO_DICOM transformation in AFNI header"):
        read_meta_info(head)


def test_read_meta_info_afni_zero_volume_dimension(tmp_path):
    data = np.zeros((4, 3, 0), dtype=np.float32)
    head = _write_afni_pair(
        tmp_path,
        "bad_dims+orig",
        data,
    )

    with pytest.raises(ValueError, match="AFNI dataset must have at least 3 dimensions"):
        read_meta_info(head)


def test_read_meta_info_afni_missing_brick_types(tmp_path):
    data = np.zeros((4, 3, 2), dtype=np.float32)
    head = _write_afni_pair(
        tmp_path,
        "no_brick_types+orig",
        data,
        include_brick_types=False,
    )

    with pytest.raises(ValueError, match="Missing BRICK_TYPES in AFNI header"):
        read_meta_info(head)


def test_read_header_afni(tmp_path):
    data = np.ones((4, 3, 2), dtype=np.float32)
    head = _write_afni_pair(tmp_path, "header_only+orig", data)

    hdr = read_header(head)
    assert hdr["dim"] == (4, 3, 2)
    assert tuple(hdr["spacing"]) == (2.0, 2.0, 3.0)
    assert tuple(hdr["origin"]) == (10.0, 20.0, 30.0)
    assert "afni_header" in hdr


def test_read_vol_afni_4d_index_and_scaling(tmp_path):
    base = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
    head = _write_afni_pair(tmp_path, "vol+orig", base, float_facs=[1.0, 2.0])

    vol0 = read_vol(head, index=0)
    vol1 = read_vol(head, index=1)

    np.testing.assert_allclose(vol0.data, base[..., 0] * 1.0)
    np.testing.assert_allclose(vol1.data, base[..., 1] * 2.0)


def test_read_vec_afni_indices_and_gzip(tmp_path):
    base = np.arange(4 * 3 * 2 * 3, dtype=np.float32).reshape((4, 3, 2, 3), order="F")
    head = _write_afni_pair(tmp_path, "vec+orig", base, float_facs=[1.0, 2.0, 3.0], gzip_data=True)

    vec = read_vec(head, indices=[0, 2])
    expected = np.stack([base[..., 0] * 1.0, base[..., 2] * 3.0], axis=3)
    np.testing.assert_allclose(vec.data, expected)
    assert vec.shape == expected.shape


def test_read_vol_afni_respects_ijk_to_dicom_orientation(tmp_path):
    data = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2), order="F")
    head = _write_afni_pair(
        tmp_path,
        "ijk_to_dicom+orig",
        data,
        ijk_to_dicom=[-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )

    vol = read_vol(head)
    assert vol.space.axes.i.axis == "Left-to-Right"
    assert vol.space.axes.j.axis == "Posterior-to-Anterior"
    assert vol.space.axes.k.axis == "Inferior-to-Superior"


def test_write_read_vol_afni_roundtrip(tmp_path):
    data = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2), order="F")
    space = NeuroSpace(dim=data.shape, spacing=(2.0, 2.0, 3.0), origin=(10.0, 20.0, 30.0))
    vol = DenseNeuroVol(data, space)
    head = tmp_path / "wr_vol+orig.HEAD"

    write_vol(vol, head, format="AFNI")
    loaded = read_vol(head)
    hdr = read_header(head)

    np.testing.assert_allclose(loaded.data, data)
    assert tuple(hdr["dim"][:3]) == data.shape
    np.testing.assert_allclose(np.asarray(hdr["spacing"])[:3], np.asarray([2.0, 2.0, 3.0]))


def test_write_read_vol_afni_defaults_to_compatible_dtype(tmp_path):
    data = np.arange(4 * 3 * 2, dtype=np.int16).reshape((4, 3, 2), order="F")
    space = NeuroSpace(dim=data.shape, spacing=(2.0, 2.0, 3.0), origin=(10.0, 20.0, 30.0))
    vol = DenseNeuroVol(data, space)
    head = tmp_path / "wr_vol_default_afni+orig.HEAD"

    write_vol(vol, head, format="AFNI")
    loaded = read_vol(head)
    hdr = read_header(head)

    assert loaded.shape == data.shape
    np.testing.assert_allclose(loaded.data, data.astype(np.float32))
    assert tuple(hdr["dim"][:3]) == data.shape


def test_write_read_vec_afni_gz_roundtrip(tmp_path):
    data = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
    space = NeuroSpace(dim=data.shape, spacing=(2.0, 2.0, 3.0, 1.0), origin=(10.0, 20.0, 30.0, 0.0))
    vec = DenseNeuroVec(data, space)
    head = tmp_path / "wr_vec+orig.HEAD"

    write_vec(vec, head, format="AFNI_GZ")
    loaded = read_vec(head)
    vol1 = read_vol(head, index=1)

    np.testing.assert_allclose(loaded.data, data)
    np.testing.assert_allclose(vol1.data, data[..., 1])


def test_parse_niml_element_attributes():
    parsed = parse_niml_element('SPARSE_DATA ni_type="2*float" ni_dimen="3" ni_form="text"')

    assert parsed["label"] == "SPARSE_DATA"
    assert parsed["attr"] == {
        "ni_type": "2*float",
        "ni_dimen": "3",
        "ni_form": "text",
    }


def test_parse_niml_file_ascii_sparse_data(tmp_path):
    niml = tmp_path / "sparse.niml"
    niml.write_text(
        "<AFNI_dataset ni_form=\"text\">\n"
        "<SPARSE_DATA ni_type=\"2*float\" ni_dimen=\"3\" ni_form=\"text\">\n"
        "1\n2\n3\n4\n5\n6\n"
        "</SPARSE_DATA>\n"
        "</AFNI_dataset>\n",
        encoding="utf-8",
    )

    parsed = parse_niml_file(niml)

    assert parsed[0]["label"] == "AFNI_dataset"
    assert parsed[1]["label"] == "SPARSE_DATA"
    np.testing.assert_allclose(parsed[1]["data"], np.array([[1, 3, 5], [2, 4, 6]], dtype=float))


def test_parse_niml_file_binary_lsbfirst_index_list(tmp_path):
    niml = tmp_path / "index.niml"
    payload = struct.pack("<4i", 1, 2, 3, 4)
    niml.write_bytes(
        b'<AFNI_dataset ni_form="binary.lsbfirst">\n'
        b'<INDEX_LIST ni_type="2*int" ni_dimen="2" ni_form="binary.lsbfirst">'
        + payload
        + b"</INDEX_LIST>\n</AFNI_dataset>\n"
    )

    parsed = parse_niml_file(niml)

    assert parsed[1]["label"] == "INDEX_LIST"
    np.testing.assert_array_equal(parsed[1]["data"], np.array([[1, 3], [2, 4]], dtype=np.int32))
