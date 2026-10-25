"""I/O functions for neuroimaging data."""

import nibabel as nib
import numpy as np
import gzip
from pathlib import Path
from typing import Union, Optional, Dict, Any, Sequence

from .neuro_vol import (
    DenseNeuroVol,
    SparseNeuroVol,
    LogicalNeuroVol,
    neurovol,
    _attach_nibabel_metadata,
    _embed_receipt_extension,
    _restore_nibabel_xforms,
)
from .neuro_vec import DenseNeuroVec, SparseNeuroVec, neurovec
from .neuro_vec import neurovecseq
from .neuro_space import NeuroSpace
from .axis import find_anatomy_3d
from .file_format import find_descriptor, NIFTI, FileFormat
from .meta_info import FileMetaInfo, NIFTIMetaInfo, AFNIMetaInfo
from .afni_io import write_afni_pair
from .orientation import affine_to_axcodes


def _is_afni_descriptor(desc: Optional[FileFormat]) -> bool:
    return desc is not None and desc.file_format.upper() == "AFNI"


def _neurospace_from_afni_meta(meta: AFNIMetaInfo) -> NeuroSpace:
    dim = tuple(int(d) for d in meta.dims[:3])
    spacing = tuple(float(s) for s in meta.spacing[:3])
    origin = tuple(float(o) for o in meta.origin[:3])
    axes = meta.spatial_axes

    # AFNI stores orientation in IJK_TO_DICOM; derive neuroim2-compatible axes
    # when available.
    header = getattr(meta, "afni_header", None)
    ijk_to_dicom = None
    if isinstance(header, dict):
        ijk_to_dicom = header.get("IJK_TO_DICOM", {}).get("content")

    if ijk_to_dicom is not None:
        vals = np.asarray(ijk_to_dicom, dtype=float)
        if vals.size >= 12:
            try:
                mat = vals[:12].reshape(3, 4)
                if mat.shape != (3, 4):
                    raise ValueError("Malformed IJK_TO_DICOM")

                # Match neuroim2's conversion: permute by AXIAL_RAI before decoding.
                # AXIAL_RAI = RIGHT_LEFT, ANT_POST, INF_SUP
                rai_to_lpi = np.array(
                    [
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                )
                tlpi = rai_to_lpi @ mat[:, :3]
                affine = np.eye(4, dtype=float)
                affine[:3, :3] = tlpi
                axis_codes = affine_to_axcodes(
                    affine, labels=[("R", "L"), ("A", "P"), ("S", "I")]
                )
                if len(axis_codes) == 3 and all(
                    c in {"R", "L", "A", "P", "I", "S"} for c in axis_codes
                ):
                    axes = find_anatomy_3d("".join(axis_codes))
            except Exception:
                # keep fallback axes when any unexpected shape/parsing issue occurs
                pass

    trans = np.eye(4, dtype=float)
    trans[0, 0] = spacing[0]
    trans[1, 1] = spacing[1]
    trans[2, 2] = spacing[2]
    trans[:3, 3] = np.asarray(origin)
    return NeuroSpace(dim, spacing=spacing, origin=origin, axes=axes, trans=trans)


_READ_IMAGE_INDEX_NOT_SET = object()


def _read_afni_data(meta: AFNIMetaInfo) -> np.ndarray:
    dtype = meta.get_data_dtype()
    total_elems = int(np.prod(meta.dims))
    if meta.descriptor.data_encoding == "gzip":
        with gzip.open(meta.data_file, "rb") as f:
            buf = f.read()
        data = np.frombuffer(buf, dtype=dtype, count=total_elems)
    else:
        data = np.fromfile(meta.data_file, dtype=dtype, count=total_elems)

    if data.size != total_elems:
        raise ValueError(
            f"AFNI data size mismatch: expected {total_elems} elements, got {data.size}"
        )

    arr = data.reshape(meta.dims, order="F").astype(np.float64, copy=False)

    slope = np.asarray(meta.slope, dtype=float)
    intercept = np.asarray(meta.intercept, dtype=float)
    slope_scalar = float(slope) if slope.ndim == 0 else float(slope.flat[0])
    intercept_scalar = (
        float(intercept) if intercept.ndim == 0 else float(intercept.flat[0])
    )
    if arr.ndim == 4 and slope.size > 1:
        if slope.size != arr.shape[3]:
            raise ValueError(
                f"AFNI BRICK_FLOAT_FACS length {slope.size} does not match nvols {arr.shape[3]}"
            )
        arr = arr * slope.reshape((1, 1, 1, -1))
        if intercept.size == slope.size:
            arr = arr + intercept.reshape((1, 1, 1, -1))
        elif intercept.size == 1:
            arr = arr + intercept_scalar
    else:
        s = slope_scalar if slope.size else 1.0
        i = intercept_scalar if intercept.size else 0.0
        arr = arr * s + i

    return arr


def _resolve_write_format(format_name: str) -> str:
    fmt = str(format_name).upper()
    aliases = {
        "NIFTI": "NIFTI",
        "NIFTI1": "NIFTI",
        "NIFTI-1": "NIFTI",
        "NIFTI2": "NIFTI2",
        "NIFTI-2": "NIFTI2",
        "AFNI": "AFNI",
        "AFNI_GZ": "AFNI_GZ",
        "AFNI-GZ": "AFNI_GZ",
        "AFNIGZ": "AFNI_GZ",
    }
    return aliases.get(fmt, fmt)


_NIFTI_DTYPE_MAP = {
    "FLOAT32": np.float32,
    "FLOAT": np.float32,
    "FLOAT64": np.float64,
    "DOUBLE": np.float64,
    "BINARY": np.uint8,
    "UBYTE": np.uint8,
    "BYTE": np.uint8,
    "SHORT": np.int16,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT": np.int32,
    "INT64": np.int64,
    "INT8": np.int8,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
}


def _source_scaled_storage(values: np.ndarray, source_header, slope_inter):
    """Return integer storage for ``values`` under the source scaling, or None.

    When an image was read from a scaled integer file and its values are still
    exactly ``raw * slope + inter`` for in-range integers, write it back in
    that same datatype and scaling (as neuroim2 does) instead of widening it
    to float64.
    """
    if source_header is None or slope_inter is None:
        return None
    slope, inter = slope_inter
    if slope is None or not np.isfinite(slope) or slope == 0:
        return None
    inter = 0.0 if inter is None or not np.isfinite(inter) else float(inter)
    source_dtype = np.dtype(source_header.get_data_dtype())
    if not np.issubdtype(source_dtype, np.integer):
        return None
    if not np.issubdtype(values.dtype, np.floating) or values.size == 0:
        return None
    if not np.all(np.isfinite(values)):
        return None
    raw = np.rint((values - inter) / slope)
    info = np.iinfo(source_dtype)
    if raw.min() < info.min or raw.max() > info.max:
        return None
    if not np.array_equal(raw * slope + inter, values):
        return None
    return raw.astype(source_dtype), float(slope), inter


def _nifti_storage_data(data: np.ndarray, data_type: Optional[str]):
    """Choose NIfTI storage dtype without prematurely quantizing values."""
    values = np.asanyarray(data)
    if data_type is None:
        target = np.dtype(np.uint8 if values.dtype == np.dtype(bool) else values.dtype)
        stored = values.astype(target) if values.dtype != target else values
        return stored, target

    dtype_key = data_type.upper()
    if dtype_key not in _NIFTI_DTYPE_MAP:
        raise ValueError(f"Unsupported NIfTI data_type: {data_type}")
    target = np.dtype(_NIFTI_DTYPE_MAP[dtype_key])

    if np.issubdtype(target, np.integer):
        if not np.all(np.isfinite(values)):
            raise ValueError("integer NIfTI output requires finite data values")
        info = np.iinfo(target)
        lo = np.min(values)
        hi = np.max(values)
        integral = np.all(np.equal(values, np.floor(values)))
        if integral and lo >= info.min and hi <= info.max:
            # Exact integral values need no slope/intercept and should not be
            # routed through floating quantization.
            return values.astype(target), target
        # Keep the source values floating/in their wider integer dtype.  A
        # target-typed header tells nibabel's array writer to derive the
        # optimal slope/intercept instead of truncating with ndarray.astype.
        return values, target

    return values.astype(target), target


_NIFTI1_MAX_DIM = 32767


def _nifti_image_for_write(
    obj, data: np.ndarray, data_type: Optional[str], nifti2: bool = False
):
    """Build a metadata-preserving NIfTI image with safe storage scaling.

    NIfTI-2 is used when requested, when the source was NIfTI-2, or when a
    dimension does not fit NIfTI-1's int16 ``dim`` field.
    """
    source_header = getattr(obj, "_nibabel_header", None)
    scaled = None
    if data_type is None:
        scaled = _source_scaled_storage(
            np.asanyarray(data),
            source_header,
            getattr(obj, "_nibabel_slope_inter", None),
        )
    if scaled is not None:
        stored, slope, inter = scaled
        target_dtype = stored.dtype
    else:
        stored, target_dtype = _nifti_storage_data(data, data_type)

    use_nifti2 = (
        nifti2
        or isinstance(source_header, nib.Nifti2Header)
        or any(int(d) > _NIFTI1_MAX_DIM for d in np.shape(stored))
    )
    header_cls = nib.Nifti2Header if use_nifti2 else nib.Nifti1Header
    image_cls = nib.Nifti2Image if use_nifti2 else nib.Nifti1Image
    if source_header is None:
        header = header_cls()
    elif type(source_header) is header_cls:
        header = source_header.copy()
    else:
        header = header_cls.from_header(source_header)
    try:
        header.set_data_dtype(target_dtype)
    except nib.spatialimages.HeaderDataError as exc:
        raise ValueError(
            f"dtype {target_dtype} is not representable in NIfTI"
        ) from exc
    # The container data are already in physical/scaled units.  Any source
    # proxy scaling has therefore been consumed; allow nibabel to choose fresh
    # scaling for this output rather than applying it a second time -- unless
    # the values were re-encoded exactly in the source scaling above.
    header.set_slope_inter(None, None)
    image = image_cls(stored, obj.space.affine, header=header)
    if scaled is not None:
        image.header.set_slope_inter(slope, inter)
    _restore_nibabel_xforms(image, obj)
    _embed_receipt_extension(image, getattr(obj, "provenance", None))
    image.header.set_zooms(tuple(float(z) for z in obj.spacing[: stored.ndim]))
    return image


def _refuse_unsupported_nifti_dtype(img) -> None:
    """Refuse complex and RGB payloads instead of silently mangling them."""
    try:
        dtype = img.get_data_dtype()
    except AttributeError:
        return
    if dtype.kind == "c":
        raise ValueError(
            f"Complex-valued images ({dtype}) are not supported: reading them "
            "as real data would discard the imaginary part. Load the file "
            "with nibabel and split real/imaginary or magnitude/phase first."
        )
    if dtype.kind == "V" or dtype.names is not None:
        raise ValueError(
            f"RGB/structured images ({dtype}) are not supported: their "
            "channels cannot be represented as a scalar volume. Load the "
            "file with nibabel and select a channel first."
        )


def _nifti_vec_payload(img):
    """Return ``(data, time_spacing)`` for a 3-D, 4-D or (x, y, z, 1, t) image.

    A 5-D file whose fourth dimension is a singleton is the NIfTI way of
    storing a time series in dim[5]; it is folded to (x, y, z, t) with the
    fifth zoom as the time spacing, as neuroim2 does.
    """
    _refuse_unsupported_nifti_dtype(img)
    shape = tuple(int(d) for d in img.shape)
    zooms = tuple(float(z) for z in img.header.get_zooms())
    if len(shape) == 5:
        if shape[3] != 1:
            raise ValueError(
                f"5-D image of shape {shape} is a multi-feature (hyper) vector; "
                "read_vec only folds files whose fourth dimension is 1"
            )
        data = np.asanyarray(img.dataobj)[:, :, :, 0, :]
        return data, zooms[4] if len(zooms) > 4 else 1.0
    if len(shape) > 5:
        raise ValueError(f"Expected at most 5-D data, got {len(shape)}D")
    data = np.asanyarray(img.dataobj)
    return data, zooms[3] if len(shape) == 4 and len(zooms) > 3 else 1.0


def _nifti_space(img, shape, time_spacing: float = 1.0) -> NeuroSpace:
    """Spatial grid from the image affine, plus a time axis for 4-D shapes."""
    spatial = NeuroSpace.from_affine(img.affine, tuple(shape[:3]))
    if len(shape) == 3:
        return spatial
    return NeuroSpace(
        tuple(int(d) for d in shape),
        spacing=tuple(float(s) for s in spatial.spacing) + (float(time_spacing),),
        trans=spatial.affine,
    )


def read_vol(filename: Union[str, Path], index: int = 0) -> DenseNeuroVol:
    """Read a 3D neuroimaging volume from file.

    Parameters
    ----------
    filename : str or Path
        Path to the neuroimaging file
    index : int, optional
        For 4D files, which 3D volume to extract (0-based)
        Default is 0 (first volume)

    Returns
    -------
    DenseNeuroVol
        The loaded 3D volume

    Notes
    -----
    Unlike R which uses 1-based indexing, Python uses 0-based indexing
    for the volume index parameter."""
    descriptor = find_descriptor(filename)
    if _is_afni_descriptor(descriptor):
        meta = descriptor.read_meta_info(filename)
        assert isinstance(meta, AFNIMetaInfo)
        data = _read_afni_data(meta)
        space = _neurospace_from_afni_meta(meta)
    else:
        # Load with nibabel; values come from dataobj (scaling applied once,
        # stored dtype kept when unscaled).
        img = nib.load(str(filename))
        _refuse_unsupported_nifti_dtype(img)
        if len(img.shape) == 3 and index == 0:
            return DenseNeuroVol.from_nibabel(img)
        if len(img.shape) == 4 and 0 <= index < img.shape[3]:
            data = np.asanyarray(img.dataobj[..., index])
            vol = DenseNeuroVol(data, _nifti_space(img, data.shape))
            _attach_nibabel_metadata(vol, img)
            return vol
        data = np.asanyarray(img.dataobj)

    # Extract single volume if 4D
    if data.ndim == 4:
        if index < 0 or index >= data.shape[3]:
            raise ValueError(
                f"index {index} out of range for 4D data with {data.shape[3]} volumes"
            )
        data = data[:, :, :, index]
    elif index != 0:
        raise ValueError(f"index {index} invalid for 3D data")
    elif data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D data, got {data.ndim}D")

    if not _is_afni_descriptor(descriptor):
        vol = DenseNeuroVol(data, _nifti_space(img, data.shape))
        _attach_nibabel_metadata(vol, img)
        return vol

    # Create and return DenseNeuroVol
    return DenseNeuroVol(data, space)


def write_vol(
    vol: Union[DenseNeuroVol, SparseNeuroVol, LogicalNeuroVol],
    filename: Union[str, Path],
    format: str = "NIFTI",
    data_type: Optional[str] = None,
):
    """Write a neuroimaging volume to file.

    Parameters
    ----------
    vol : NeuroVol
        The volume to write
    filename : str or Path
        Output filename
    format : str, optional
        Output format (currently only "NIFTI" supported)
    data_type : str, optional
        Output data type (e.g., "FLOAT32", "INT16")
        NIfTI preserves the input dtype by default. AFNI infers a compatible
        storage type from the input."""
    format_key = _resolve_write_format(format)

    # Convert to dense if needed
    if isinstance(vol, SparseNeuroVol):
        dense_vol = vol.as_dense()
        data = dense_vol.data
    else:
        data = vol.data

    if format_key in ("NIFTI", "NIFTI2"):
        image = _nifti_image_for_write(
            vol, data, data_type, nifti2=format_key == "NIFTI2"
        )
        nib.save(image, str(filename))
        return

    if format_key in ("AFNI", "AFNI_GZ"):
        if data_type is not None:
            dtype_key = data_type.upper()
            if dtype_key not in _NIFTI_DTYPE_MAP:
                raise ValueError(f"Unsupported AFNI data_type: {data_type}")
            data = data.astype(_NIFTI_DTYPE_MAP[dtype_key])
        data_encoding = "gzip" if format_key == "AFNI_GZ" else "raw"
        spacing = tuple(float(x) for x in vol.spacing[:3])
        origin = tuple(float(x) for x in vol.origin[:3])
        write_afni_pair(
            filename,
            data,
            spacing=spacing,
            origin=origin,
            data_encoding=data_encoding,
            data_type=data_type,
        )
        return

    raise NotImplementedError(
        f"Format '{format}' is not yet supported. "
        "Supported formats: NIFTI, NIFTI2, AFNI, AFNI_GZ."
    )


def _neurospace_from_nifti(nifti_img: nib.Nifti1Image) -> NeuroSpace:
    """Create a 3D NeuroSpace from a NIfTI image.

    Always projects onto the spatial (first 3) axes so ``read_vol`` can pair a
    single 3D volume with a matching 3D space.  ``read_vec`` extends this to
    4D via :meth:`NeuroSpace.add_dim` once the time dimension is known.
    """
    return NeuroSpace.from_affine(
        nifti_img.affine,
        nifti_img.shape[:3],
        header=getattr(nifti_img, "header", None),
    )


def read_header(filename: Union[str, Path]) -> Dict[str, Any]:
    """Read header information from neuroimaging file.

    Parameters
    ----------
    filename : str or Path
        Path to the neuroimaging file

    Returns
    -------
    dict
        Header information including dimensions, spacing, origin, etc."""
    descriptor = find_descriptor(filename)
    if _is_afni_descriptor(descriptor):
        meta = descriptor.read_meta_info(filename)
        assert isinstance(meta, AFNIMetaInfo)
        affine = _neurospace_from_afni_meta(meta).trans
        return {
            "dim": tuple(meta.dims),
            "pixdim": tuple(meta.spacing),
            "spacing": tuple(meta.spacing),
            "origin": tuple(meta.origin),
            "datatype": meta.get_data_dtype(),
            "data_type": str(meta.get_data_dtype()),
            "bitpix": int(meta.bytes_per_element * 8),
            "affine": affine,
            "trans": affine,
            "description": "",
            "descrip": "",
            "qform": {"matrix": affine, "code": 0},
            "qform_code": 0,
            "sform": {"matrix": affine, "code": 0},
            "sform_code": 0,
            "intent_code": 0,
            "intent_name": "",
            "vox_offset": int(meta.data_offset),
            "scl_slope": meta.slope,
            "scl_inter": meta.intercept,
            "cal_min": np.nan,
            "cal_max": np.nan,
            "TR": np.nan,
            "raw": meta.afni_header,
            "afni_header": meta.afni_header,
        }

    img = nib.load(str(filename))
    header = img.header

    # Handle description field
    descrip = header.get("descrip", b"")
    if isinstance(descrip, bytes):
        descrip = descrip.decode("utf-8", errors="ignore").strip("\x00")
    else:
        descrip = str(descrip).strip()

    qform, qform_code = img.get_qform(coded=True)
    sform, sform_code = img.get_sform(coded=True)
    qform = img.affine if qform is None else qform
    sform = img.affine if sform is None else sform

    zooms = header.get_zooms()
    raw_header = {key: header[key].copy() for key in header.keys()}
    scl_slope = float(getattr(img.dataobj, "slope", header.get("scl_slope", 1)))
    scl_inter = float(getattr(img.dataobj, "inter", header.get("scl_inter", 0)))
    raw_header["scl_slope"] = np.asarray(scl_slope, dtype=np.float32)
    raw_header["scl_inter"] = np.asarray(scl_inter, dtype=np.float32)
    tr = float(zooms[3]) if len(zooms) > 3 else np.nan
    intent_name = header.get("intent_name", b"")
    if hasattr(intent_name, "item"):
        intent_name = intent_name.item()
    if isinstance(intent_name, bytes):
        intent_name = intent_name.decode("utf-8", errors="ignore").strip("\x00")
    else:
        intent_name = str(intent_name)

    return {
        "dim": img.shape,
        "pixdim": zooms,
        "spacing": zooms,
        "origin": img.affine[:3, 3],
        "datatype": header.get_data_dtype(),
        "data_type": str(header.get_data_dtype()),
        "bitpix": int(header.get("bitpix", 0)),
        "affine": img.affine,
        "trans": img.affine,
        "description": descrip,
        "descrip": descrip,
        "qform": {"matrix": qform, "code": int(qform_code or 0)},
        "qform_code": int(header.get("qform_code", 0)),
        "sform": {"matrix": sform, "code": int(sform_code or 0)},
        "sform_code": int(header.get("sform_code", 0)),
        "intent_code": int(header.get("intent_code", 0)),
        "intent_name": intent_name,
        "vox_offset": float(header.get("vox_offset", 0)),
        # nibabel NaNs scl_* in the loaded header and keeps the scaling on
        # the data proxy, so report it from there.
        "scl_slope": scl_slope,
        "scl_inter": scl_inter,
        "cal_min": float(header.get("cal_min", np.nan)),
        "cal_max": float(header.get("cal_max", np.nan)),
        "TR": tr,
        "raw": raw_header,
    }


def read_vol_list(filenames: list, index: int = 0) -> list:
    """Read multiple volumes from a list of files.

    Parameters
    ----------
    filenames : list
        List of file paths
    index : int, optional
        For 4D files, which volume to extract (0-based)

    Returns
    -------
    list
        List of DenseNeuroVol objects"""
    return [read_vol(f, index) for f in filenames]


def read_vec(
    filename: Union[str, Path], indices=None, mask=None
) -> Union[DenseNeuroVec, SparseNeuroVec]:
    """Read a 4D neuroimaging vector from file.

    Parameters
    ----------
    filename : str or Path
        Path to the 4D neuroimaging file
    indices : array-like, optional
        Indices of volumes to load (0-based)
        If None, loads all volumes
    mask : LogicalNeuroVol, optional
        Mask for sparse representation

    Returns
    -------
    DenseNeuroVec or SparseNeuroVec
        The loaded 4D vector

    Notes
    -----
    Unlike R which uses 1-based indexing, Python uses 0-based indexing
    for the indices parameter."""
    descriptor = find_descriptor(filename)
    if _is_afni_descriptor(descriptor):
        meta = descriptor.read_meta_info(filename)
        assert isinstance(meta, AFNIMetaInfo)
        data = _read_afni_data(meta)
        if data.ndim == 3:
            data = data[..., np.newaxis]
        space = _neurospace_from_afni_meta(meta)
    else:
        # Load with nibabel
        img = nib.load(str(filename))
        data, time_spacing = _nifti_vec_payload(img)

    if (
        not _is_afni_descriptor(descriptor)
        and indices is None
        and mask is None
        and len(img.shape) == 4
        and img.shape[3] > 1
    ):
        return DenseNeuroVec.from_nibabel(img)

    # Handle dimensionality
    if data.ndim == 3:
        # 3D file, treat as single volume
        data = data[..., np.newaxis]
    elif data.ndim != 4:
        raise ValueError(f"Expected 3D or 4D data, got {data.ndim}D")

    # Extract specified indices
    if indices is not None:
        indices = np.asarray(indices)
        if indices.ndim == 0:
            indices = np.array([indices.item()])
        if np.any(indices < 0) or np.any(indices >= data.shape[3]):
            raise ValueError(f"indices out of range [0, {data.shape[3]-1}]")
        data = data[..., indices]

    if not _is_afni_descriptor(descriptor):
        # Time spacing survives the fold of a 5-D or (x, y, z, 1) file.
        space = _nifti_space(img, data.shape, time_spacing)
    # Update to 4D
    if space.ndim == 3:
        space = space.add_dim(n=1, size=data.shape[3])

    # Create appropriate vector type
    if mask is None:
        vec = DenseNeuroVec(data, space)
        if not _is_afni_descriptor(descriptor):
            _attach_nibabel_metadata(vec, img)
        return vec
    else:
        # Convert to sparse
        if not isinstance(mask, LogicalNeuroVol):
            mask_space = NeuroSpace(
                space.dim[:3],
                spacing=space.spacing[:3],
                origin=space.origin[:3],
                axes=space.axes.drop_dim(),
            )
            mask = LogicalNeuroVol(mask, mask_space)

        # Extract sparse data
        mask_indices = np.where(mask.data.ravel(order="F"))[0]
        data_flat = data.reshape(-1, data.shape[3], order="F")
        sparse_data = data_flat[mask_indices, :].T

    return SparseNeuroVec(sparse_data, space, mask, orientation="time_x_voxels")


def write_vec(
    vec: Union[DenseNeuroVec, SparseNeuroVec],
    filename: Union[str, Path],
    format: str = "NIFTI",
    data_type: Optional[str] = None,
):
    """Write a neuroimaging vector to file.

    Parameters
    ----------
    vec : NeuroVec
        The vector to write
    filename : str or Path
        Output filename
    format : str, optional
        Output format (currently only "NIFTI" supported)
    data_type : str, optional
        Output data type (e.g., "FLOAT32", "INT16")
        NIfTI preserves the input dtype by default. AFNI infers a compatible
        storage type from the input."""
    format_key = _resolve_write_format(format)

    # Convert to dense if needed
    if isinstance(vec, SparseNeuroVec):
        dense_vec = vec.as_dense()
        data = dense_vec.data
    else:
        data = vec.data

    if format_key in ("NIFTI", "NIFTI2"):
        image = _nifti_image_for_write(
            vec, data, data_type, nifti2=format_key == "NIFTI2"
        )
        nib.save(image, str(filename))
        return

    if format_key in ("AFNI", "AFNI_GZ"):
        if data_type is not None:
            dtype_key = data_type.upper()
            if dtype_key not in _NIFTI_DTYPE_MAP:
                raise ValueError(f"Unsupported AFNI data_type: {data_type}")
            data = data.astype(_NIFTI_DTYPE_MAP[dtype_key])
        data_encoding = "gzip" if format_key == "AFNI_GZ" else "raw"
        spacing = tuple(float(x) for x in vec.spacing[:3])
        origin = tuple(float(x) for x in vec.origin[:3])
        write_afni_pair(
            filename,
            data,
            spacing=spacing,
            origin=origin,
            data_encoding=data_encoding,
            data_type=data_type,
        )
        return

    raise NotImplementedError(
        f"Format '{format}' is not yet supported. "
        "Supported formats: NIFTI, NIFTI2, AFNI, AFNI_GZ."
    )


def read_image(
    file_path: Union[str, Path, Sequence[Union[str, Path]]],
    type: str = "auto",
    index: object = _READ_IMAGE_INDEX_NOT_SET,
    indices=None,
    mask=None,
    mode: str = "normal",
):
    """Auto-dispatch image reader based on dimensionality.

    Loads the NIfTI header, inspects ``ndim``, and delegates to
    :func:`read_vol` (3-D) or :func:`read_vec` (4-D).

    Parameters
    ----------
    file_path : str, Path, or list-like of paths
        One or more path(s) to neuroimaging files.
    type : {"auto", "vol", "vec"}
        Dispatch mode:

        * ``"auto"`` (default) - infer from image dimension.
        * ``"vol"`` - force :func:`read_vol`.
        * ``"vec"`` - force :func:`read_vec`.
    index : int or sentinel
        Optional volume index for ``type='vol'`` or single-file ``type='vec'``.
    indices : int sequence, optional
        Optional explicit vectors for ``read_vec``.
    mask : LogicalNeuroVol, optional
        Optional mask forwarded to :func:`read_vec`.
    mode : str
        Currently only ``"normal"`` is supported.

    Returns
    -------
    DenseNeuroVol or DenseNeuroVec (or SparseNeuroVec)
        The loaded image object.

    Raises
    ------
    ValueError
        If the file has fewer than 3 or more than 4 spatial dimensions.

    """
    # Validate requested IO mode.
    if mode != "normal":
        raise NotImplementedError(
            f"read_image mode '{mode}' is not yet supported. Supported modes: normal"
        )

    if type not in {"auto", "vol", "vec"}:
        raise ValueError("type must be one of 'auto', 'vol', or 'vec'")

    if type == "vol":
        if isinstance(file_path, (list, tuple)):
            if len(file_path) != 1:
                raise ValueError("read_image: type='vol' expects a single file_name")
            file_path = file_path[0]

        resolved_index = 0 if index is _READ_IMAGE_INDEX_NOT_SET else index
        return read_vol(file_path, index=resolved_index)

    if isinstance(file_path, (list, tuple)):
        if len(file_path) == 0:
            raise ValueError("read_image requires at least one file path")
        if len(file_path) == 1:
            return read_image(
                file_path[0],
                type=type,
                index=index,
                indices=indices,
                mask=mask,
                mode=mode,
            )
        vecs = [read_vec(fp, indices=indices, mask=mask) for fp in file_path]
        return neurovecseq(vecs) if len(vecs) > 1 else vecs[0]

    has_indices = indices is not None
    if type == "vec" and index is not _READ_IMAGE_INDEX_NOT_SET and not has_indices:
        indices = index

    if type == "vec":
        return read_vec(file_path, indices=indices, mask=mask)

    descriptor = find_descriptor(file_path)
    if descriptor is not None and _is_afni_descriptor(descriptor):
        # AFNI descriptors are not nibabel-readable; defer to our NIfTI/AFNI readers.
        meta = descriptor.read_meta_info(file_path)
        if len(meta.dims) > 3 and meta.dims[3] > 1:
            if indices is None and index is not _READ_IMAGE_INDEX_NOT_SET:
                indices = index
            return read_vec(file_path, indices=indices, mask=mask)
        return read_vol(
            file_path, index=0 if index is _READ_IMAGE_INDEX_NOT_SET else index
        )

    img = nib.load(str(file_path))
    shape = img.shape
    ndim = len(shape)
    has_vec_dim = ndim > 3 and any(dim > 1 for dim in shape[3:])

    if has_vec_dim:
        if indices is None and index is not _READ_IMAGE_INDEX_NOT_SET:
            indices = index
        return read_vec(file_path, indices=indices, mask=mask)
    if ndim == 3 or (ndim > 3 and not has_vec_dim):
        return read_vol(
            file_path, index=0 if index is _READ_IMAGE_INDEX_NOT_SET else index
        )
    else:
        raise ValueError(
            f"Expected 3D or vector-like image, got {len(shape)}D from {file_path}"
        )


def read_volume(
    file_path: Union[str, Path],
    *,
    index: int = 0,
    mask=None,
) -> DenseNeuroVol:
    """Read a file as a 3-D :class:`DenseNeuroVol`.

    Intent-revealing wrapper over ``read_image(..., type="vol")`` with the
    stringly-typed ``type=`` and the hand-rolled index sentinel removed.

    Rank contract (see ``docs/concepts/images.qmd``): a 4-D file is **not**
    refused — the volume reader extracts a single 3-D volume, ``index``
    (default 0). A 3-D file is read directly. This is reader dispatch, not
    a rank gate.
    """
    return read_image(file_path, type="vol", index=index, mask=mask)


def read_series(
    file_path: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    indices=None,
    mask=None,
) -> "DenseNeuroVec":
    """Read a file as a 4-D :class:`DenseNeuroVec` time series.

    Intent-revealing wrapper over ``read_image(..., type="vec")``.

    Rank contract (see ``docs/concepts/images.qmd``): a 3-D file is
    promoted to singleton-time 4-D, shape ``(..., 1)``; a 4-D file is read
    as-is. This is reader dispatch, not a rank gate.
    """
    return read_image(file_path, type="vec", indices=indices, mask=mask)


def load_data(source):
    """Convenience wrapper that calls ``source.load()``.

    Parameters
    ----------
    source : FileSource
        A lazy-loading source object (e.g. :class:`NeuroVolSource`,
        :class:`NeuroVecSource`).

    Returns
    -------
    object
        The materialised neuroimaging object.
    """
    return source.load()


def read_meta_info(filename: Union[str, Path]) -> FileMetaInfo:
    """Read meta information from neuroimaging file.

    Parameters
    ----------
    filename : str or Path
        Path to the neuroimaging file

    Returns
    -------
    FileMetaInfo
        Meta information object (NIFTIMetaInfo or AFNIMetaInfo)"""
    # Find appropriate file format
    descriptor = find_descriptor(filename)
    if descriptor is None:
        raise ValueError(f"Unknown file format for: {filename}")

    # Use format-specific reader
    return descriptor.read_meta_info(filename)
