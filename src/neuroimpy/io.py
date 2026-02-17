"""I/O functions for neuroimaging data.

Direct translation of R's neuroim2 I/O functions.
"""

import nibabel as nib
import numpy as np
import gzip
from pathlib import Path
from typing import Union, Optional, Dict, Any, Sequence

from .neuro_vol import DenseNeuroVol, SparseNeuroVol, LogicalNeuroVol, neurovol
from .neuro_vec import DenseNeuroVec, SparseNeuroVec, neurovec
from .neuro_vec import neurovecseq
from .neuro_space import NeuroSpace
from .axis import find_anatomy_3d
from .file_format import find_descriptor, NIFTI, FileFormat
from .meta_info import FileMetaInfo, NIFTIMetaInfo, AFNIMetaInfo
from .afni_io import write_afni_pair


def _is_afni_descriptor(desc: Optional[FileFormat]) -> bool:
    return desc is not None and desc.file_format.upper() == "AFNI"


def _neurospace_from_afni_meta(meta: AFNIMetaInfo) -> NeuroSpace:
    dim = tuple(int(d) for d in meta.dims[:3])
    spacing = tuple(float(s) for s in meta.spacing[:3])
    origin = tuple(float(o) for o in meta.origin[:3])
    axes = find_anatomy_3d("RAS")
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
    intercept_scalar = float(intercept) if intercept.ndim == 0 else float(intercept.flat[0])
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
        "AFNI": "AFNI",
        "AFNI_GZ": "AFNI_GZ",
        "AFNI-GZ": "AFNI_GZ",
        "AFNIGZ": "AFNI_GZ",
    }
    return aliases.get(fmt, fmt)


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
    for the volume index parameter.
    
    R Equivalent
    ------------
    neuroim2::read_vol
    """
    descriptor = find_descriptor(filename)
    if _is_afni_descriptor(descriptor):
        meta = descriptor.read_meta_info(filename)
        assert isinstance(meta, AFNIMetaInfo)
        data = _read_afni_data(meta)
        space = _neurospace_from_afni_meta(meta)
    else:
        # Load with nibabel
        img = nib.load(str(filename))
        data = img.get_fdata()
    
    # Extract single volume if 4D
    if data.ndim == 4:
        if index < 0 or index >= data.shape[3]:
            raise ValueError(f"index {index} out of range for 4D data with {data.shape[3]} volumes")
        data = data[:, :, :, index]
    elif index != 0:
        raise ValueError(f"index {index} invalid for 3D data")
    elif data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D data, got {data.ndim}D")
    
    if not _is_afni_descriptor(descriptor):
        # Create NeuroSpace from NIfTI header only for valid dimensionality
        space = _neurospace_from_nifti(img)
    
    # Create and return DenseNeuroVol
    return DenseNeuroVol(data, space)


def write_vol(vol: Union[DenseNeuroVol, SparseNeuroVol, LogicalNeuroVol], 
              filename: Union[str, Path],
              format: str = "NIFTI",
              data_type: Optional[str] = None):
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
        Defaults to "FLOAT" for NIfTI, inferred from input for AFNI
        
    R Equivalent
    ------------
    neuroim2::write_vol
    """
    format_key = _resolve_write_format(format)

    # Convert to dense if needed
    if isinstance(vol, SparseNeuroVol):
        dense_vol = vol.as_dense()
        data = dense_vol.data
    else:
        data = vol.data
    
    if data_type is None:
        data_type = None if format_key in ("AFNI", "AFNI_GZ") else "FLOAT"

    # Handle data type conversion
    if data_type is not None:
        dtype_map = {
            "FLOAT32": np.float32,
            "FLOAT": np.float32,
            "FLOAT64": np.float64,
            "DOUBLE": np.float64,
            "BINARY": np.uint8,
            "BYTE": np.uint8,
            "SHORT": np.int16,
            "INT16": np.int16,
            "INT32": np.int32,
            "INT": np.int32,
            "INT8": np.int8,
            "UINT8": np.uint8,
            "UINT16": np.uint16,
            "UINT32": np.uint32,
        }
        dtype_key = data_type.upper()
        if dtype_key in dtype_map:
            data = data.astype(dtype_map[dtype_key])
        else:
            raise ValueError(f"Unsupported NIfTI data_type: {data_type}")
    
    if format_key == "NIFTI":
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(data, vol.trans)
        
        # Set spacing in header
        nifti_img.header.set_zooms(vol.spacing)
        
        # Save
        nib.save(nifti_img, str(filename))
        return

    if format_key in ("AFNI", "AFNI_GZ"):
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
        f"Format '{format}' is not yet supported. Supported formats: NIFTI, AFNI, AFNI_GZ."
    )


def _neurospace_from_nifti(nifti_img: nib.Nifti1Image) -> NeuroSpace:
    """Create NeuroSpace from NIfTI image.
    
    Parameters
    ----------
    nifti_img : nibabel Nifti1Image
        The NIfTI image
        
    Returns
    -------
    NeuroSpace
        Spatial metadata
    """
    # Get dimensions
    dim = nifti_img.shape[:3]  # First 3 dimensions
    
    # Get affine transformation matrix
    affine = nifti_img.affine
    
    # Extract spacing from header (more reliable than from affine)
    spacing = nifti_img.header.get_zooms()[:3]
    
    # Extract origin (translation part of affine)
    origin = affine[:3, 3]
    
    # Try to determine axes from affine
    # This is simplified - full implementation would analyze the rotation matrix
    axes = find_anatomy_3d("RAS")  # Default to RAS for now
    
    # Create NeuroSpace
    return NeuroSpace(dim, spacing=spacing, origin=origin, axes=axes, trans=affine)


def read_header(filename: Union[str, Path]) -> Dict[str, Any]:
    """Read header information from neuroimaging file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the neuroimaging file
        
    Returns
    -------
    dict
        Header information including dimensions, spacing, origin, etc.
        
    R Equivalent
    ------------
    neuroim2::read_header
    """
    descriptor = find_descriptor(filename)
    if _is_afni_descriptor(descriptor):
        meta = descriptor.read_meta_info(filename)
        assert isinstance(meta, AFNIMetaInfo)
        return {
            "dim": tuple(meta.dims),
            "spacing": tuple(meta.spacing),
            "origin": tuple(meta.origin),
            "datatype": meta.get_data_dtype(),
            "bitpix": int(meta.bytes_per_element * 8),
            "affine": _neurospace_from_afni_meta(meta).trans,
            "description": "",
            "qform_code": 0,
            "sform_code": 0,
            "vox_offset": int(meta.data_offset),
            "scl_slope": meta.slope,
            "scl_inter": meta.intercept,
            "afni_header": meta.afni_header,
        }

    img = nib.load(str(filename))
    header = img.header
    
    # Handle description field
    descrip = header.get('descrip', b'')
    if isinstance(descrip, bytes):
        descrip = descrip.decode('utf-8', errors='ignore').strip('\x00')
    else:
        descrip = str(descrip).strip()
    
    return {
        "dim": img.shape,
        "spacing": header.get_zooms(),
        "origin": img.affine[:3, 3],
        "datatype": header.get_data_dtype(),
        "bitpix": int(header.get('bitpix', 0)),
        "affine": img.affine,
        "description": descrip,
        "qform_code": int(header.get('qform_code', 0)),
        "sform_code": int(header.get('sform_code', 0)),
        "vox_offset": float(header.get('vox_offset', 0)),
        "scl_slope": float(header.get('scl_slope', 1)),
        "scl_inter": float(header.get('scl_inter', 0)),
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
        List of DenseNeuroVol objects
        
    R Equivalent
    ------------
    neuroim2::read_vol_list
    """
    return [read_vol(f, index) for f in filenames]


def read_vec(filename: Union[str, Path], indices=None, mask=None) -> Union[DenseNeuroVec, SparseNeuroVec]:
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
    for the indices parameter.
    
    R Equivalent
    ------------
    neuroim2::read_vec
    """
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
        data = img.get_fdata()
    
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
        # Create NeuroSpace from NIfTI header
        space = _neurospace_from_nifti(img)
    # Update to 4D
    if space.ndim == 3:
        space = space.add_dim(n=1, size=data.shape[3])
    
    # Create appropriate vector type
    if mask is None:
        return DenseNeuroVec(data, space)
    else:
        # Convert to sparse
        if not isinstance(mask, LogicalNeuroVol):
            mask_space = NeuroSpace(space.dim[:3],
                                  spacing=space.spacing[:3],
                                  origin=space.origin[:3],
                                  axes=space.axes.drop_dim())
            mask = LogicalNeuroVol(mask, mask_space)
        
        # Extract sparse data
        mask_indices = np.where(mask.data.ravel(order='F'))[0]
        data_flat = data.reshape(-1, data.shape[3], order='F')
        sparse_data = data_flat[mask_indices, :].T
        
    return SparseNeuroVec(sparse_data, space, mask)


def write_vec(vec: Union[DenseNeuroVec, SparseNeuroVec], 
              filename: Union[str, Path],
              format: str = "NIFTI",
              data_type: Optional[str] = None):
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
        Defaults to "FLOAT" for NIfTI, inferred from input for AFNI
        
    R Equivalent
    ------------
    neuroim2::write_vec
    """
    format_key = _resolve_write_format(format)

    # Convert to dense if needed
    if isinstance(vec, SparseNeuroVec):
        dense_vec = vec.as_dense()
        data = dense_vec.data
    else:
        data = vec.data
    
    if data_type is None:
        data_type = None if format_key in ("AFNI", "AFNI_GZ") else "FLOAT"

    # Handle data type conversion
    if data_type is not None:
        dtype_map = {
            "FLOAT32": np.float32,
            "FLOAT": np.float32,
            "FLOAT64": np.float64,
            "DOUBLE": np.float64,
            "BINARY": np.uint8,
            "BYTE": np.uint8,
            "SHORT": np.int16,
            "INT16": np.int16,
            "INT32": np.int32,
            "INT": np.int32,
            "INT8": np.int8,
            "UINT8": np.uint8,
            "UINT16": np.uint16,
            "UINT32": np.uint32,
        }
        dtype_key = data_type.upper()
        if dtype_key in dtype_map:
            data = data.astype(dtype_map[dtype_key])
        else:
            raise ValueError(f"Unsupported NIfTI data_type: {data_type}")
    
    if format_key == "NIFTI":
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(data, vec.trans[:4, :4])
        
        # Set spacing in header
        nifti_img.header.set_zooms(vec.spacing)
        
        # Save
        nib.save(nifti_img, str(filename))
        return

    if format_key in ("AFNI", "AFNI_GZ"):
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
        f"Format '{format}' is not yet supported. Supported formats: NIFTI, AFNI, AFNI_GZ."
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

    R Equivalent
    ------------
    neuroim2::read_image (dispatch on dimensionality)
    """
    # Validate requested IO mode.
    if mode != "normal":
        raise NotImplementedError(
            f"read_image mode '{mode}' is not yet supported. Supported modes: normal"
        )

    if type not in {"auto", "vol", "vec"}:
        raise ValueError(
            "type must be one of 'auto', 'vol', or 'vec'"
        )

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
        return read_vol(file_path, index=0 if index is _READ_IMAGE_INDEX_NOT_SET else index)

    img = nib.load(str(file_path))
    ndim = len(img.shape)

    if ndim == 3:
        return read_vol(file_path, index=0 if index is _READ_IMAGE_INDEX_NOT_SET else index)
    elif ndim == 4:
        if indices is None and index is not _READ_IMAGE_INDEX_NOT_SET:
            indices = index
        return read_vec(file_path, indices=indices, mask=mask)
    else:
        raise ValueError(
            f"Expected 3D or 4D image, got {ndim}D from {file_path}"
        )


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
        Meta information object (NIFTIMetaInfo or AFNIMetaInfo)
        
    R Equivalent
    ------------
    neuroim2::read_meta_info
    """
    # Find appropriate file format
    descriptor = find_descriptor(filename)
    if descriptor is None:
        raise ValueError(f"Unknown file format for: {filename}")
    
    # Use format-specific reader
    return descriptor.read_meta_info(filename)
