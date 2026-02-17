"""AFNI header parsing helpers.

This module provides a lightweight AFNI ``.HEAD`` parser compatible with the
attribute block format used by AFNI BRIK/HEAD datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union, Any
import gzip
import numpy as np


def _parse_int_attribute(lines: List[str]) -> List[int]:
    toks = " ".join(lines).strip().split()
    return [int(tok) for tok in toks]


def _parse_float_attribute(lines: List[str]) -> List[float]:
    toks = " ".join(lines).strip().split()
    return [float(tok) for tok in toks]


def _parse_string_attribute(lines: List[str]) -> List[str]:
    # AFNI string attributes are tilde-delimited: ~val1~val2~...
    text = "\n".join(lines)
    parts = text.split("~")
    values: List[str] = []
    for part in parts[1:]:
        if part == "":
            continue
        values.append(part)
    return values


def _parse_element(block: List[str]) -> Dict[str, Any]:
    if len(block) < 4:
        raise ValueError("Invalid AFNI attribute block: expected at least 4 lines")

    atype = block[0].split("=", 1)[1].strip()
    name = block[1].split("=", 1)[1].strip()
    count = int(block[2].split("=", 1)[1].strip())
    content_lines = block[3:]

    if atype == "integer-attribute":
        content = _parse_int_attribute(content_lines)
    elif atype == "float-attribute":
        content = _parse_float_attribute(content_lines)
    elif atype == "string-attribute":
        content = _parse_string_attribute(content_lines)
    else:
        raise ValueError(f"Unrecognized AFNI attribute type: {atype}")

    return {"type": atype, "name": name, "count": count, "content": content}


def read_afni_header(file_name: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Read and parse an AFNI ``.HEAD`` file.

    Parameters
    ----------
    file_name : str or Path
        Path to an AFNI header file.

    Returns
    -------
    dict
        Mapping from attribute name to parsed attribute dictionary containing
        ``type``, ``name``, ``count``, and ``content``.
    """
    pth = Path(file_name)
    lines = pth.read_text(encoding="utf-8").splitlines()

    header: Dict[str, Dict[str, Any]] = {}
    block: List[str] = []
    for line in lines:
        if line.strip() == "":
            if block:
                parsed = _parse_element(block)
                header[parsed["name"]] = parsed
                block = []
            continue
        block.append(line)

    if block:
        parsed = _parse_element(block)
        header[parsed["name"]] = parsed

    return header


def _afni_dtype_info(dtype: np.dtype) -> tuple[int, np.dtype]:
    dt = np.dtype(dtype)
    if dt == np.dtype(np.uint8):
        return 0, dt
    if dt == np.dtype(np.int16):
        return 1, dt
    # AFNI reader in neuroimpy currently supports FLOAT32 as general fallback.
    return 3, np.dtype(np.float32)


def write_afni_pair(
    file_name: Union[str, Path],
    data: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    data_encoding: str = "raw",
    data_type: str | None = None,
) -> tuple[str, str]:
    """Write AFNI HEAD/BRIK pair.

    Parameters
    ----------
    file_name : str or Path
        Output path hint. May be ``.HEAD``, ``.BRIK``, ``.BRIK.gz``, or stem.
    data : ndarray
        3D or 4D array to write.
    spacing : tuple
        Spatial spacing for x/y/z.
    origin : tuple
        Spatial origin for x/y/z.
    data_encoding : {"raw","gzip"}
        Data payload encoding. ``gzip`` writes ``.BRIK.gz``.
    data_type : str, optional
        Requested storage type. Supports ``FLOAT32``, ``INT16``, ``UINT8``.

    Returns
    -------
    (header_file, data_file) : tuple[str, str]
    """
    arr = np.asarray(data)
    if arr.ndim not in (3, 4):
        raise ValueError(f"AFNI write supports 3D/4D arrays, got {arr.ndim}D")

    out = str(file_name)
    if out.endswith(".HEAD"):
        stem = out[:-5]
    elif out.endswith(".BRIK.gz"):
        stem = out[:-8]
    elif out.endswith(".BRIK"):
        stem = out[:-5]
    else:
        stem = out

    head_file = f"{stem}.HEAD"
    brik_file = f"{stem}.BRIK.gz" if data_encoding == "gzip" else f"{stem}.BRIK"

    if arr.ndim == 3:
        dims = arr.shape
        nvols = 1
    else:
        dims = arr.shape[:3]
        nvols = arr.shape[3]

    if data_type is not None:
        dt_req = str(data_type).upper()
        dtype_lookup = {
            "FLOAT32": np.float32,
            "INT16": np.int16,
            "UINT8": np.uint8,
        }
        if dt_req not in dtype_lookup:
            raise ValueError(f"Unsupported AFNI data_type: {data_type}")
        arr_dtype = np.dtype(dtype_lookup[dt_req])
    else:
        arr_dtype = arr.dtype

    brick_type, storage_dtype = _afni_dtype_info(arr_dtype)
    arr_write = np.asarray(arr, dtype=storage_dtype)

    facs = " ".join(["1.0"] * nvols)
    brick_types = " ".join([str(brick_type)] * nvols)
    labels = "~" + "~".join([f"vol{i}" for i in range(nvols)]) + "~"
    byte_order = "LSB_FIRST" if np.little_endian else "MSB_FIRST"

    sx, sy, sz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    ox, oy, oz = (float(origin[0]), float(origin[1]), float(origin[2]))

    # Keep y negative to match common AFNI conventions; readers use abs(DELTA).
    dy = -abs(sy)
    ijk_to_dicom = (
        f"{sx} 0.0 0.0 {ox} "
        f"0.0 {dy} 0.0 {oy} "
        f"0.0 0.0 {sz} {oz}"
    )

    head_txt = (
        "type = integer-attribute\n"
        "name = DATASET_DIMENSIONS\n"
        "count = 5\n"
        f"{int(dims[0])} {int(dims[1])} {int(dims[2])} 1 0\n\n"
        "type = integer-attribute\n"
        "name = DATASET_RANK\n"
        "count = 2\n"
        f"3 {int(nvols)}\n\n"
        "type = float-attribute\n"
        "name = DELTA\n"
        "count = 3\n"
        f"{sx} {dy} {sz}\n\n"
        "type = float-attribute\n"
        "name = IJK_TO_DICOM\n"
        "count = 12\n"
        f"{ijk_to_dicom}\n\n"
        "type = float-attribute\n"
        "name = ORIGIN\n"
        "count = 3\n"
        f"{ox} {oy} {oz}\n\n"
        "type = integer-attribute\n"
        "name = BRICK_TYPES\n"
        f"count = {int(nvols)}\n"
        f"{brick_types}\n\n"
        "type = float-attribute\n"
        "name = BRICK_FLOAT_FACS\n"
        f"count = {int(nvols)}\n"
        f"{facs}\n\n"
        "type = string-attribute\n"
        "name = BYTEORDER_STRING\n"
        f"count = {len(byte_order) + 2}\n"
        f"~{byte_order}~\n\n"
        "type = string-attribute\n"
        "name = BRICK_LABS\n"
        f"count = {len(labels)}\n"
        f"{labels}\n"
    )
    Path(head_file).write_text(head_txt, encoding="utf-8")

    raw = arr_write.ravel(order="F").tobytes()
    if data_encoding == "gzip":
        with gzip.open(brik_file, "wb") as f:
            f.write(raw)
    else:
        Path(brik_file).write_bytes(raw)

    return head_file, brik_file
