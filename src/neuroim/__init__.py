"""neuroim - Python neuroimaging library"""

import warnings

__version__ = "0.1.0"

# Import Phase 1, Phase 2, Phase 3, Phase 4, and Phase 5 components
from .axis import *
from .axis import find_anatomy_3d
from .neuro_space import *
from .neuro_vol import *
from .neuro_vec import *
from .neuro_slice import *
from .roi import *
from .index_lookup_vol import IndexLookupVol
from .roi_vec_window import ROIVecWindow

# Import I/O functions only if nibabel is available
try:
    from .io import (
        read_vol,
        write_vol,
        read_header,
        read_vol_list,
        read_vec,
        write_vec,
        read_meta_info,
        read_image,
        load_data,
    )

    # File format and metadata
    from .file_format import (
        FileFormat,
        NIFTIFormat,
        AFNIFormat,
        NIFTI,
        NIFTI_GZ,
        NIFTI_PAIR,
        NIFTI_PAIR_GZ,
        AFNI,
        AFNI_GZ,
        find_descriptor,
    )
    from .meta_info import MetaInfo, FileMetaInfo, NIFTIMetaInfo, AFNIMetaInfo
    from .binary_io import BinaryReader, BinaryWriter, ColumnReader
    from .afni_io import (
        parse_niml_element,
        parse_niml_header,
        parse_niml_next,
        parse_niml_file,
        read_niml_data,
        read_afni_header,
        write_afni_pair,
    )

    # NIfTI utilities
    from .nifti_utils import (
        create_nifti_header,
        as_nifti_header,
        matrix_to_quatern,
        quatern_to_matrix,
    )

    # Source factory classes (lazy loaders, depend on io)
    from .sources import (
        FileSource,
        NeuroVolSource,
        NeuroVecSource,
        SparseNeuroVecSource,
        MappedNeuroVecSource,
    )

    # NIfTI extension support
    from .nifti_extension import (
        NiftiExtensionCodes,
        NiftiExtension,
        NiftiExtensionList,
        parse_extensions,
        parse_extension,
        parse_afni_extension,
        ecode_name,
        get_afni_attribute,
        list_afni_attributes,
    )
except ImportError:
    # nibabel not installed, I/O functions will not be available
    pass

# Import spatial filtering
from .kernel import Kernel, gaussian_kernel, spherical_kernel, box_kernel, embed_kernel
from .spatial_filters import (
    gaussian_blur,
    guided_filter,
    bilateral_filter,
    bilateral_filter_vec,
    bilateral_filter_4d,
)
from .graph_filter import (
    cgb_make_graph,
    cgb_filter,
    cgb_smooth,
    cgb_smooth_loro,
    cgb_nuisance,
    laplace_enhance,
)

# Import orthogonal slice extraction
from .orthogonal_slices import (
    extract_orthogonal_slices,
    extract_axial_slice,
    extract_sagittal_slice,
    extract_coronal_slice,
    get_slice_orientation,
    get_world_bounds_for_slice,
)

# Import resampling and reorientation
from .resample import resample, resample_vec, resample_to, reorient

# Import orientation utilities
from .orientation import (
    affine_to_orientation,
    affine_to_axcodes,
    axcodes_to_orientation,
    orientation_to_axcodes,
    orientation_transform,
    axcodes,
    apply_orientation,
    apply_affine,
    append_diag,
    orientation_inverse_affine,
    obliquity,
    voxel_sizes,
    vox2out_vox,
    perm_mat,
    rescale_affine,
)

# Import searchlight functions
from .searchlight import (
    searchlight_iterator as searchlight,
    searchlight_coords,
    random_searchlight,
    bootstrap_searchlight,
    clustered_searchlight,
    ellipsoid_shape,
    cube_shape,
    blobby_shape,
)
from .searchlight_high_level import (
    searchlight as searchlight_apply,
    resampled_searchlight,
    cluster_searchlight_series,
)

# Typed result objects for ROI / searchlight workflows
from .results import (
    Receipt,
    ROIExtractionResult,
    SearchlightResult,
    hash_ndarray,
    hash_neurospace,
)

# Import simulation utilities
from .simulation import simulate_fmri, prepare_confounds, make_time_weights

# Import connected components
from .connected_components import conn_comp, conn_comp_3D, ConnCompResult
from .clustered_neuro_vol import ClusteredNeuroVol
from .clustered_neuro_vec import ClusteredNeuroVec
from .neuro_bucket import NeuroBucket

# Import statistical operations
from .stats import (
    split_blocks,
    split_clusters,
    split_fill,
    split_reduce,
    split_scale,
    partition,
    map_values,
    centroids,
)

# Import data manipulation operations
from .operations import concat, scale_series, mapf, downsample

# Import indexing utilities
from .indexing import (
    linear_access,
    matricized_access,
    from_matvec,
    to_matvec,
    dot_reduce,
)

# Import the explicit neuroim2-style migration namespace without star-exporting it.
from . import compat as compat

# Import memory-mapped variants
from .big_neuro_vec import BigNeuroVec, big_neurovecseq
from .file_backed_neuro_vec import FileBackedNeuroVec, file_backed_neurovec
from .mapped_neuro_vec import (
    MappedNeuroVec,
    mapped_neurovecseq,
    scale_mapper,
    log_mapper,
    threshold_mapper,
)

# Import NeuroHyperVec (5D+ support)
from .neuro_hypervec import (
    NeuroHyperVec,
    DenseNeuroHyperVec,
    SparseNeuroHyperVec,
    concat_features,
    write_neurohypervec,
    read_neurohypervec,
)

# Import visualization / plotting utilities
from .plotting import (
    plot_neuro_vol,
    plot_ortho,
    plot_montage,
    plot_overlay,
    plot_checkerboard,
    plot_edge_overlay,
    map_to_colors,
    resolve_cmap,
)


def from_nibabel(img):
    """Wrap a nibabel SpatialImage-like object as a neuroim volume or vector."""
    shape = getattr(img, "shape", None)
    if shape is None:
        raise TypeError("img must provide a shape attribute")

    ndim = len(shape)
    has_vector_dim = ndim > 3 and any(dim > 1 for dim in shape[3:])
    if has_vector_dim:
        return NeuroVec.from_nibabel(img)
    if ndim >= 3:
        return NeuroVol.from_nibabel(img)
    raise ValueError(f"Expected at least 3D image data, got shape {shape}")


_DEPRECATED_COMPAT_ALIASES = {
    "NeuroVecSeq",
    "createNIfTIHeader",
    "findAnatomy3D",
    "mapToColors",
    "matrixToQuatern",
    "quaternToMatrix",
    "read_hyper_vec",
}


def __getattr__(name):
    if name in _DEPRECATED_COMPAT_ALIASES:
        warnings.warn(
            f"neuroim.{name} is deprecated; use neuroim.compat.{name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(compat, name)
    raise AttributeError(f"module 'neuroim' has no attribute {name!r}")


__all__ = [
    "NeuroSpace",
    "NeuroVol",
    "DenseNeuroVol",
    "SparseNeuroVol",
    "LogicalNeuroVol",
    "NeuroVec",
    "DenseNeuroVec",
    "SparseNeuroVec",
    "BigNeuroVec",
    "FileBackedNeuroVec",
    "MappedNeuroVec",
    "NeuroHyperVec",
    "DenseNeuroHyperVec",
    "SparseNeuroHyperVec",
    "ROICoords",
    "ROIVol",
    "ROIExtractionResult",
    "SearchlightResult",
    "Receipt",
    "spherical_roi",
    "cuboid_roi",
    "searchlight",
    "searchlight_apply",
    "read_image",
    "read_vol",
    "write_vol",
    "read_vec",
    "write_vec",
    "from_nibabel",
    "resample",
    "reorient",
    "plot_neuro_vol",
    "plot_ortho",
    "plot_montage",
    "plot_overlay",
    "ConnCompResult",
    "compat",
]
