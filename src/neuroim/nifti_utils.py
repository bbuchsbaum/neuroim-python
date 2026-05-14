"""NIfTI utilities for header manipulation and quaternion operations.

This module provides functions for creating and manipulating NIfTI headers,
as well as converting between transformation matrices and quaternion
representations.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from .neuro_vol import NeuroVol

# NIfTI data type codes
NIFTI_TYPE_CODES = {
    'BINARY': 2,  # Alias for UINT8
    'UBYTE': 2,  # Alias for UINT8
    'UINT8': 2,
    'INT16': 4,
    'INT32': 8,
    'FLOAT32': 16,
    'FLOAT': 16,  # Alias for FLOAT32
    'COMPLEX64': 32,
    'FLOAT64': 64,
    'DOUBLE': 64,  # Alias for FLOAT64
    'RGB24': 128,
    'INT8': 256,
    'UINT16': 512,
    'UINT32': 768,
    'INT64': 1024,
    'UINT64': 1280,
    'FLOAT128': 1536,
    'COMPLEX128': 1792,
    'COMPLEX256': 2048
}

def create_nifti_header(one_file: bool = True, file_name: Optional[str] = None) -> Dict:
    """Create an empty NIfTI-1 header list.
    
    Initializes a list of fields following the NIfTI-1 specification
    with default or placeholder values. Users typically call this
    internally via as_nifti_header rather than using directly.
    
    Parameters
    ----------
    one_file : bool
        If True, magic is set to "n+1" indicating a single-file (.nii) 
        approach. Otherwise set to "ni1".
    file_name : str, optional
        Optional character string to store in the header, usually
        referencing the intended output file name.
        
    Returns
    -------
    dict
        A dictionary containing approximately 30 fields that comprise
        the NIfTI-1 header structure. Many of these are placeholders
        until filled by downstream usage.
        
    """
    header = {
        # Required fields
        'sizeof_hdr': 348,  # Size of header, must be 348
        'data_type': '',  # Obsolete
        'db_name': '',  # Obsolete
        'extents': 0,  # Obsolete
        'session_error': 0,  # Obsolete
        'regular': 'r',  # Obsolete
        'dim_info': 0,  # MRI slice ordering
        
        # Data array dimensions
        'dim': [3, 1, 1, 1, 1, 1, 1, 1],  # dim[0] is number of dimensions
        
        # Intent parameters
        'intent_p1': 0.0,
        'intent_p2': 0.0,
        'intent_p3': 0.0,
        'intent_code': 0,
        
        # Data type
        'datatype': 0,
        'bitpix': 0,
        'slice_start': 0,
        
        # Grid spacings
        'pixdim': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        
        # Data offset
        'vox_offset': 352.0 if one_file else 0.0,
        
        # Data scaling
        'scl_slope': 1.0,
        'scl_inter': 0.0,
        
        # Slice timing
        'slice_end': 0,
        'slice_code': 0,
        'xyzt_units': 10,  # mm and sec
        
        # Data range
        'cal_max': 0.0,
        'cal_min': 0.0,
        
        # Timing
        'slice_duration': 0.0,
        'toffset': 0.0,
        
        # Unused fields
        'glmax': 0,
        'glmin': 0,
        
        # Data description
        'descrip': file_name[:80] if file_name else '',
        'aux_file': '',
        
        # Coordinate system
        'qform_code': 0,
        'sform_code': 0,
        
        # Quaternion parameters
        'quatern_b': 0.0,
        'quatern_c': 0.0,
        'quatern_d': 0.0,
        'qoffset_x': 0.0,
        'qoffset_y': 0.0,
        'qoffset_z': 0.0,
        
        # Affine matrix
        'srow_x': [0.0, 0.0, 0.0, 0.0],
        'srow_y': [0.0, 0.0, 0.0, 0.0],
        'srow_z': [0.0, 0.0, 0.0, 0.0],
        
        # Intent
        'intent_name': '',
        
        # Magic string
        'magic': 'n+1' if one_file else 'ni1'
    }
    
    return header

def matrix_to_quatern(mat: np.ndarray) -> Dict[str, Union[List[float], float]]:
    """Convert a transformation matrix to a quaternion representation.
    
    Extracts the rotation and scaling components from a 3x3 (or 4x4)
    transformation matrix, normalizes them, and computes the
    corresponding quaternion parameters and a sign factor (qfac)
    indicating whether the determinant is negative.
    
    Parameters
    ----------
    mat : numpy.ndarray
        A numeric matrix with at least the top-left 3x3 portion
        containing rotation/scaling. Often a 4x4 affine transform,
        but only the 3x3 top-left submatrix is used in practice.
        
    Returns
    -------
    dict
        A dictionary with two elements:

        - ``'quaternion'``: A list of length 3, ``[b, c, d]``, which together
          with ``'a'`` (derived internally) represents the rotation.
        - ``'qfac'``: Either +1 or -1, indicating whether the determinant
          of the rotation submatrix is positive or negative.
          
    References
    ----------
    Cox RW. Analysis of Functional NeuroImages (AFNI) and NIfTI-1
    quaternion conventions. https://afni.nimh.nih.gov
    """
    # Extract 3x3 rotation/scaling submatrix
    R = mat[:3, :3].copy()
    
    # Check for zero-length axes and correct
    for i in range(3):
        col_norm = np.linalg.norm(R[:, i])
        if col_norm == 0:
            R[:, i] = 0
            R[i, i] = 1
    
    # Normalize columns to extract pure rotation
    scales = np.linalg.norm(R, axis=0)
    R_normalized = R / scales
    
    # Check determinant
    det = np.linalg.det(R_normalized)
    qfac = 1.0 if det >= 0 else -1.0
    
    # If determinant is negative, negate third column
    if qfac < 0:
        R_normalized[:, 2] = -R_normalized[:, 2]
    
    # Compute quaternion using standard formulas
    # Based on the trace and diagonal elements
    trace = np.trace(R_normalized)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R_normalized[2, 1] - R_normalized[1, 2]) * s
        y = (R_normalized[0, 2] - R_normalized[2, 0]) * s
        z = (R_normalized[1, 0] - R_normalized[0, 1]) * s
    elif R_normalized[0, 0] > R_normalized[1, 1] and R_normalized[0, 0] > R_normalized[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R_normalized[0, 0] - R_normalized[1, 1] - R_normalized[2, 2])
        w = (R_normalized[2, 1] - R_normalized[1, 2]) / s
        x = 0.25 * s
        y = (R_normalized[0, 1] + R_normalized[1, 0]) / s
        z = (R_normalized[0, 2] + R_normalized[2, 0]) / s
    elif R_normalized[1, 1] > R_normalized[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R_normalized[1, 1] - R_normalized[0, 0] - R_normalized[2, 2])
        w = (R_normalized[0, 2] - R_normalized[2, 0]) / s
        x = (R_normalized[0, 1] + R_normalized[1, 0]) / s
        y = 0.25 * s
        z = (R_normalized[1, 2] + R_normalized[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R_normalized[2, 2] - R_normalized[0, 0] - R_normalized[1, 1])
        w = (R_normalized[1, 0] - R_normalized[0, 1]) / s
        x = (R_normalized[0, 2] + R_normalized[2, 0]) / s
        y = (R_normalized[1, 2] + R_normalized[2, 1]) / s
        z = 0.25 * s
    
    # NIfTI uses b, c, d (not including a/w)
    # Also ensure w is positive (NIfTI convention)
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    
    return {
        'quaternion': [x, y, z],  # b, c, d in NIfTI terms
        'qfac': qfac
    }

def quatern_to_matrix(quaternion: List[float], qfac: float = 1.0,
                      qoffset: Optional[List[float]] = None) -> np.ndarray:
    """Convert quaternion parameters back to a transformation matrix.
    
    Parameters
    ----------
    quaternion : list of float
        The quaternion parameters [b, c, d] (a is computed internally)
    qfac : float
        Sign factor, either +1 or -1
    qoffset : list of float, optional
        Translation offsets [x, y, z]
        
    Returns
    -------
    numpy.ndarray
        4x4 transformation matrix
        
    """
    b, c, d = quaternion
    
    # Compute a from the constraint a^2 + b^2 + c^2 + d^2 = 1
    a_sq = 1.0 - (b*b + c*c + d*d)
    a = np.sqrt(max(0.0, a_sq))  # Ensure non-negative
    
    # Build rotation matrix from quaternion
    # Using standard quaternion-to-matrix formulas
    R = np.array([
        [a*a + b*b - c*c - d*d,     2*b*c - 2*a*d,           2*b*d + 2*a*c],
        [2*b*c + 2*a*d,             a*a - b*b + c*c - d*d,   2*c*d - 2*a*b],
        [2*b*d - 2*a*c,             2*c*d + 2*a*b,           a*a - b*b - c*c + d*d]
    ])
    
    # Apply qfac to third column if negative
    if qfac < 0:
        R[:, 2] = -R[:, 2]
    
    # Build 4x4 transformation matrix
    mat = np.eye(4)
    mat[:3, :3] = R
    
    if qoffset is not None:
        mat[:3, 3] = qoffset
    
    return mat

def as_nifti_header(vol: NeuroVol, file_name: str, 
                    one_file: bool = True, data_type: str = "FLOAT") -> Dict:
    """Construct a minimal NIfTI-1 header from a NeuroVol.
    
    Given a NeuroVol object, this function builds a basic NIfTI-1
    header structure, populating essential fields such as dim, pixdim,
    datatype, the affine transform, and the quaternion parameters.
    
    Parameters
    ----------
    vol : NeuroVol
        A NeuroVol specifying dimensions, spacing, and affine transform.
    file_name : str
        File name for the header (used within the header but not
        necessarily to write data).
    one_file : bool
        If True, sets the NIfTI magic to "n+1", implying a single-file
        format (.nii). If False, uses "ni1" (header+image).
    data_type : str
        Character specifying the data representation, e.g. "FLOAT",
        "DOUBLE". The internal code picks an integer NIfTI code.
        
    Returns
    -------
    dict
        A dictionary representing the NIfTI-1 header fields, containing
        elements like dimensions, pixdim, datatype, qform, quaternion,
        qfac, etc. This can be passed to other functions that write or
        manipulate the header.
        
    """
    # Start with empty header
    header = create_nifti_header(one_file, file_name)
    
    # Set dimensions
    ndim = vol.space.ndim
    header['dim'][0] = ndim
    for i in range(ndim):
        header['dim'][i + 1] = int(vol.space.dim[i])
    
    # Set voxel dimensions (pixdim)
    header['pixdim'][0] = 1.0
    for i in range(ndim):
        header['pixdim'][i + 1] = float(vol.space.spacing[i])
    
    # Set data type
    if data_type.upper() in NIFTI_TYPE_CODES:
        header['datatype'] = NIFTI_TYPE_CODES[data_type.upper()]
        # Set bitpix based on data type
        dtype_map = {
            2: 8,    # UINT8
            4: 16,   # INT16
            8: 32,   # INT32
            16: 32,  # FLOAT32
            64: 64,  # FLOAT64
            256: 8,  # INT8
            512: 16, # UINT16
            768: 32, # UINT32
        }
        header['bitpix'] = dtype_map.get(header['datatype'], 0)
    
    # Set coordinate system info
    header['qform_code'] = 1  # Scanner coordinates
    header['sform_code'] = 1  # Scanner coordinates
    
    # Get quaternion from transformation matrix
    quat_info = matrix_to_quatern(vol.space.trans)
    header['quatern_b'] = quat_info['quaternion'][0]
    header['quatern_c'] = quat_info['quaternion'][1]
    header['quatern_d'] = quat_info['quaternion'][2]
    header['pixdim'][0] = quat_info['qfac']  # Store qfac in pixdim[0]
    
    # Set offsets from transformation matrix
    header['qoffset_x'] = vol.space.trans[0, 3]
    header['qoffset_y'] = vol.space.trans[1, 3]
    header['qoffset_z'] = vol.space.trans[2, 3]
    
    # Also set sform (affine matrix rows)
    header['srow_x'] = vol.space.trans[0, :].tolist()
    header['srow_y'] = vol.space.trans[1, :].tolist()
    header['srow_z'] = vol.space.trans[2, :].tolist()
    
    # Set voxel offset for single file
    if one_file:
        header['vox_offset'] = 352.0
    
    return header
