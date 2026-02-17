# Feature Parity Tracking: neuroim2 → neuroimpy

## Overall Progress Summary
- **Phase 1 (Core Infrastructure)**: ✅ Complete (100%)
  - NeuroSpace, Axis classes, coordinate transformations
- **Phase 2 (3D Volumes)**: ✅ Complete (100%)
  - NeuroVol hierarchy, basic I/O, all operations
- **Phase 3 (4D Vectors)**: ✅ Complete (100%)
  - NeuroVec hierarchy, dense/sparse variants, all operations
- **Phase 4 (2D Slices)**: ✅ Complete (100%)
  - NeuroSlice class, slice extraction, all operations
- **Phase 5 (ROI)**: ✅ Complete (100%)
  - ROI hierarchy, construction functions, all operations
- **Phase 6 (I/O)**: ✅ Complete (90%)
- **Phase 7 (Filtering)**: ✅ Complete (100%)
- **Phase 8 (Resampling)**: ✅ Complete (100%)
- **Phase 9 (Metadata)**: ✅ Complete (95%)
- **Phase 10 (Searchlight)**: ✅ Complete (100%)
- **Phase 11 (Connected Components)**: ✅ Complete (100%)
- **Phase 12 (Statistical Operations)**: ✅ Complete (100%)
- **Phase 13 (Memory-Mapped Variants)**: ✅ Complete (100%)
- **Phase 14 (Test Conversion)**: 📝 Planned

**Total Features**: ~165 | **Implemented**: ~151 (92%)

## Status Legend
- ✅ Implemented and tested
- 🚧 In progress
- 📝 Planned
- ❌ Not started
- 🔄 Needs refactoring

## 1. Core Data Structures

### Spatial Reference Classes
| R Class | Python Class | Status | Notes |
|---------|-------------|--------|-------|
| NeuroSpace | NeuroSpace | ✅ | Complete with tests |
| AxisSet | AxisSet | ✅ | Base class implemented |
| AxisSet1D | AxisSet1D | ✅ | Complete |
| AxisSet2D | AxisSet2D | ✅ | Complete |
| AxisSet3D | AxisSet3D | ✅ | Complete with drop_dim |
| AxisSet4D | AxisSet4D | ✅ | Complete |
| AxisSet5D | AxisSet5D | ✅ | Complete |
| NamedAxis | NamedAxis | ✅ | Complete with constants |

### 3D Volume Classes
| R Class | Python Class | Status | Notes |
|---------|-------------|--------|-------|
| NeuroVol | NeuroVol | ✅ | Abstract base class complete |
| DenseNeuroVol | DenseNeuroVol | ✅ | Complete with tests |
| SparseNeuroVol | SparseNeuroVol | ✅ | Complete with tests |
| LogicalNeuroVol | LogicalNeuroVol | ✅ | Complete with tests |
| ClusteredNeuroVol | ClusteredNeuroVol | ✅ | Complete with tests |
| IndexLookupVol | - | ✅ | Not needed - functionality in SparseNeuroVec |

### 4D Vector Classes
| R Class | Python Class | Status | Notes |
|---------|-------------|--------|-------|
| NeuroVec | NeuroVec | ✅ | Abstract base class complete |
| DenseNeuroVec | DenseNeuroVec | ✅ | Complete with tests |
| SparseNeuroVec | SparseNeuroVec | ✅ | Complete with tests |
| BigNeuroVec | BigNeuroVec | ✅ | Complete with memory mapping |
| FileBackedNeuroVec | FileBackedNeuroVec | ✅ | Complete with lazy loading |
| MappedNeuroVec | MappedNeuroVec | ✅ | Complete with transformations |
| NeuroVecSeq | neurovecseq | ✅ | Factory function implemented |

### Other Classes
| R Class | Python Class | Status | Notes |
|---------|-------------|--------|-------|
| NeuroHyperVec | NeuroHyperVec | ✅ | Implemented (dense + sparse variants) |
| NeuroSlice | NeuroSlice | ✅ | Complete with tests |
| LatentNeuroVec | - | ✅ | Removed (no R equivalent) |

## 2. ROI Classes

| R Class | Python Class | Status | Notes |
|---------|-------------|--------|-------|
| ROI | ROI | ✅ | Abstract base class |
| ROICoords | ROICoords | ✅ | Complete with indexing |
| ROIVol | ROIVol | ✅ | Complete with conversions |
| ROIVec | ROIVec | ✅ | Complete with column access |
| ROIVolWindow | ROIVolWindow | ✅ | Complete with parent tracking |
| ROIVecWindow | ROIVecWindow | ✅ | Implemented |

## 3. File I/O Functions

### Reading Functions
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| read_vol | read_vol | ✅ | Basic NIfTI support |
| read_vec | read_vec | ✅ | Complete in io.py |
| read_vol_list | read_vol_list | ✅ | In io.py |
| read_header | read_header | ✅ | In io.py |
| read_meta_info | read_meta_info | ✅ | In io.py |

### Writing Functions
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| write_vol | write_vol | ✅ | Basic NIfTI support |
| write_vec | write_vec | ✅ | Complete in io.py |

### Binary I/O
| R Class | Python Class | Status | Notes |
|---------|-------------|--------|-------|
| BinaryReader | BinaryReader | ✅ | Complete with tests |
| BinaryWriter | BinaryWriter | ✅ | Complete with tests |
| ColumnReader | ColumnReader | ✅ | Complete with tests |

## 4. Spatial Operations

### Coordinate Transformations
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| coord_to_grid | coord_to_grid | ✅ | Method of NeuroSpace |
| grid_to_coord | grid_to_coord | ✅ | Method of NeuroSpace |
| coord_to_index | coord_to_index | ✅ | Method of NeuroSpace |
| index_to_coord | index_to_coord | ✅ | Method of NeuroSpace |
| grid_to_index | grid_to_index | ✅ | Method of NeuroSpace |
| index_to_grid | index_to_grid | ✅ | Method of NeuroSpace |
| grid_to_grid | grid_to_grid | ✅ | Method of NeuroSpace |

### Reorientation & Resampling
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| reorient | reorient | ✅ | Complete with orientation support |
| resample | resample | ✅ | Complete with nibabel backend |
| resample_vec | resample_vec | ✅ | For 4D NeuroVec objects |
| perm_mat | perm_mat | ❌ | |

## 5. Data Access Methods

### Element Access
| R Method | Python Method | Status | Notes |
|----------|---------------|--------|-------|
| `[` operator | `__getitem__` | ✅ | Complete for all classes |
| linear_access | linear_access | ❌ | |
| matricized_access | matricized_access | ❌ | |
| series | series | ✅ | Method of NeuroVec |
| series_roi | series_roi | ❌ | |
| vols | vols | ✅ | Method of NeuroVec |
| slices | slices | ✅ | Function implemented |
| sub_vector | sub_vector | ✅ | Method of NeuroVec |

### Data Transformation
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| as.sparse | as_sparse | ✅ | Method of NeuroVec |
| as.dense | as_dense | ✅ | Method of SparseNeuroVec |
| as.mask | as_mask | 📝 | |
| as.matrix | as_matrix | ✅ | Method of DenseNeuroVec |
| scale_series | scale_series | ✅ | Method of DenseNeuroVec |
| concat | concat | ✅ | Method of NeuroVec |

## 6. Image Processing

### Spatial Filtering
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| gaussian_blur | gaussian_blur | ✅ | Complete with mask support |
| bilateral_filter | bilateral_filter | ✅ | Complete with spatial/intensity control |
| bilateral_filter_vec | bilateral_filter_vec | ✅ | Complete for NeuroVec |
| guided_filter | guided_filter | ✅ | Complete with edge preservation |
| laplace_enhance | - | ❌ | Removed per user request |

### Connected Components
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| conn_comp | conn_comp | ✅ | Complete with cluster tables |
| conn_comp_3D | conn_comp_3D | ✅ | Complete for binary masks |

### Kernel Operations
| R Class/Function | Python Class/Function | Status | Notes |
|-----------------|---------------------|--------|-------|
| Kernel | Kernel | ✅ | Complete with weights and coordinates |
| gaussian_kernel | gaussian_kernel | ✅ | Complete with normalization |
| spherical_kernel | spherical_kernel | ✅ | Complete with custom weight functions |
| box_kernel | box_kernel | ✅ | Uniform weight kernel |
| embed_kernel | embed_kernel | ✅ | Embed kernel in NeuroSpace |
| mapf | mapf | ❌ | |

## 7. ROI & Searchlight Operations

### ROI Construction
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| square_roi | square_roi | ✅ | Complete with fixdim |
| cuboid_roi | cuboid_roi | ✅ | Complete with fill/nonzero |
| spherical_roi | spherical_roi | ✅ | Complete with radius check |
| spherical_roi_set | spherical_roi_set | ✅ | Complete with multi-fill |

### Searchlight Iterators
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| searchlight | searchlight | ✅ | Complete with lazy evaluation |
| searchlight_coords | searchlight_coords | ✅ | Returns coordinate matrices |
| random_searchlight | random_searchlight | ✅ | Non-overlapping searchlights |
| bootstrap_searchlight | bootstrap_searchlight | ✅ | Bootstrap sampling |
| clustered_searchlight | clustered_searchlight | ✅ | K-means and pre-defined clusters |

## 8. Statistical Operations

### Split Operations
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| split_blocks | split_blocks | ✅ | Complete in stats.py |
| split_clusters | split_clusters | ✅ | Complete in stats.py |
| split_fill | split_fill | ✅ | Complete in stats.py |
| split_reduce | split_reduce | ✅ | Complete in stats.py |
| split_scale | split_scale | ✅ | Complete in stats.py |

### Other Operations
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| partition | partition | ✅ | Complete in stats.py |
| map_values | map_values | ✅ | Complete in stats.py |
| centroids | centroids | ✅ | Complete in stats.py |

## 9. Metadata & File Format Support

### Metadata Classes
| R Class | Python Class | Status | Notes |
|---------|-------------|--------|-------|
| MetaInfo | MetaInfo | ✅ | Complete |
| FileMetaInfo | FileMetaInfo | ✅ | Complete with all attributes |
| NIFTIMetaInfo | NIFTIMetaInfo | ✅ | Complete with header access |
| AFNIMetaInfo | AFNIMetaInfo | ✅ | Class defined |

### File Format Classes
| R Class | Python Class | Status | Notes |
|---------|-------------|--------|-------|
| FileFormat | FileFormat | ✅ | Complete with methods |
| NIFTIFormat | NIFTIFormat | ✅ | Complete with nibabel backend |
| AFNIFormat | AFNIFormat | 🚧 | HEAD/BRIK metadata + read paths implemented; AFNI write/NIML pending |

## 10. Utility Functions

### Array Operations
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| drop | drop | ❌ | |
| drop_dim | drop_dim | ✅ | Method of NeuroSpace |
| add_dim | add_dim | ✅ | Method of NeuroSpace |
| dim_of | dim_of | ✅ | Method of NeuroSpace |
| which_dim | which_dim | ✅ | Method of NeuroSpace |
| ndim | ndim | ✅ | Property of NeuroSpace |

### Spatial Properties
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| space | space | ✅ | Property in classes |
| spacing | spacing | ✅ | Property of NeuroSpace |
| origin | origin | ✅ | Property of NeuroSpace |
| axes | axes | ✅ | Property of NeuroSpace |
| trans | trans | ✅ | Property of NeuroSpace |
| inverse_trans | inverse | ✅ | Property of NeuroSpace |
| bounds | bounds | ✅ | Method of NeuroSpace |
| centroid | centroid | ✅ | Method of NeuroSpace |

### Data Properties
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| values | values | 📝 | |
| coords | coords | 📝 | |
| indices | indices | 📝 | |
| voxels | voxels | ❌ | |
| lookup | lookup | ❌ | |
| patch_set | patch_set | ❌ | |
| num_clusters | num_clusters | ❌ | |

## 11. Constants

### Anatomical Axes
| R Constant | Python Constant | Status | Notes |
|-----------|-----------------|--------|-------|
| LEFT_RIGHT | LEFT_RIGHT | ✅ | In axis.py |
| RIGHT_LEFT | RIGHT_LEFT | ✅ | In axis.py |
| ANT_POST | ANT_POST | ✅ | In axis.py |
| POST_ANT | POST_ANT | ✅ | In axis.py |
| INF_SUP | INF_SUP | ✅ | In axis.py |
| SUP_INF | SUP_INF | ✅ | In axis.py |
| TIME | TIME | ✅ | In axis.py |
| None | None | ✅ | Built-in |
| NullAxis | NullAxis | ✅ | In axis.py |
| TimeAxis | TimeAxis | ✅ | Alias for TIME |

### Orientation Lists
| R Constant | Python Constant | Status | Notes |
|-----------|-----------------|--------|-------|
| OrientationList2D | OrientationList2D | ✅ | Dict in axis.py |
| OrientationList3D | OrientationList3D | ✅ | Dict in axis.py |

### File Formats
| R Constant | Python Constant | Status | Notes |
|-----------|-----------------|--------|-------|
| NIFTI | NIFTI | ✅ | |
| NIFTI_GZ | NIFTI_GZ | ✅ | |
| NIFTI_PAIR | NIFTI_PAIR | ✅ | |
| NIFTI_PAIR_GZ | NIFTI_PAIR_GZ | ✅ | |
| AFNI | AFNI | ✅ | |
| AFNI_GZ | AFNI_GZ | ✅ | |

## 12. Advanced Features

### Quaternion Support
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| matrixToQuatern | matrix_to_quatern | ✅ | Complete with NIfTI spec compliance |
| quaternToMatrix | quatern_to_matrix | ✅ | Complete with round-trip accuracy |

### NIfTI Header Support
| R Function | Python Function | Status | Notes |
|-----------|-----------------|--------|-------|
| createNIfTIHeader | create_nifti_header | ✅ | Creates NIfTI-1 headers |
| as_nifti_header | as_nifti_header | ✅ | Converts NeuroVol to NIfTI header |

## Progress Summary

- **Total Features**: ~165
- **Implemented**: ~151 (92%)
- **In Progress**: ~0 (0%)
- **Planned**: ~5 (3%)
- **Not Started**: ~9 (5%)

## Phase 1 Completion (✅)
- NeuroSpace class with all coordinate transformations
- Complete AxisSet hierarchy (1D-5D)
- NamedAxis with anatomical constants
- Orientation lists (2D and 3D)
- All dimension manipulation methods
- Comprehensive test suite with R compatibility

## Phase 2 Completion (✅)
- NeuroVol abstract base class with all operations
- DenseNeuroVol with full indexing and arithmetic
- SparseNeuroVol with efficient lookup tables
- LogicalNeuroVol for masks and binary operations
- All data access patterns and conversions
- Complete test coverage

## Phase 3 Completion (✅)
- NeuroVec abstract base class with all required methods
- DenseNeuroVec with full indexing, arithmetic, and operations
- SparseNeuroVec with efficient lookup tables and conversions
- Factory functions (neurovec, neurovecseq)
- Series extraction for single/multiple voxels
- Matrix conversions and scaling operations
- Complete I/O support (read_vec, write_vec)
- Comprehensive test suite matching R's tests

## Phase 4 Completion (✅)
- NeuroSlice class for 2D neuroimaging data
- Slice extraction from 3D volumes (slice function)
- Batch slice extraction (slices function)
- Sparse slice construction support
- Grid/index coordinate conversions
- Arithmetic operations on slices
- Full indexing and data access
- Comprehensive test suite with R compatibility

## Phase 5 Completion (✅)
- ROI abstract base class
- ROICoords with coordinate management and indexing
- ROIVol with data values and conversions (as_sparse, as_logical)
- ROIVec for vector-valued ROIs
- ROIVolWindow for windowed ROI operations
- Factory functions (roicoords, roivol)
- ROI construction functions (square_roi, cuboid_roi, spherical_roi, spherical_roi_set)
- Support for fill values and nonzero filtering
- Comprehensive test suite with full coverage

## Phase 6 Completion (✅)
- FileFormat abstract base class with file matching and path manipulation
- NIFTIFormat class using nibabel as backend for robust NIfTI support
- MetaInfo, FileMetaInfo, and NIFTIMetaInfo classes for metadata management
- BinaryReader, BinaryWriter, and ColumnReader for low-level I/O operations
- Enhanced read_header and read_meta_info functions
- Comprehensive test suite covering all I/O functionality
- Graceful degradation when nibabel is not installed

## Phase 7 Completion (✅)
- Kernel class representing filter kernels with weights and spatial coordinates
- Kernel creation functions: gaussian_kernel, spherical_kernel, box_kernel
- Spatial filtering functions with full R compatibility:
  - gaussian_blur with mask support and configurable window
  - bilateral_filter with spatial and intensity sigma control
  - bilateral_filter_vec for filtering 4D NeuroVec objects
  - guided_filter for edge-preserving smoothing
- embed_kernel function to place kernels in NeuroSpace
- Comprehensive test suite covering all filtering operations
- Parameter validation and error handling

## Phase 8 Completion (✅)
- resample function for NeuroVol with support for nearest, linear, and cubic interpolation
- resample_vec function for 4D NeuroVec objects
- reorient function for remapping grid-to-world coordinates
- Support for all standard anatomical orientations (RAS, LPI, etc.)
- nibabel backend for robust resampling operations
- Graceful degradation when nibabel is not installed
- Comprehensive test suite with parameter validation

## Phase 9 Completion (✅)
- FileFormat abstract base class with methods for file matching and manipulation
- NIFTIFormat and AFNIFormat implementations (AFNIFormat reader pending)
- File format constants: NIFTI, NIFTI_GZ, NIFTI_PAIR, NIFTI_PAIR_GZ, AFNI, AFNI_GZ
- Quaternion conversion functions (matrix_to_quatern, quatern_to_matrix)
- NIfTI header creation and manipulation (create_nifti_header, as_nifti_header)
- Support for NIfTI data type codes and proper header field population
- Comprehensive test suite covering all file format functionality
- Full integration with nibabel backend for robust file operations

## Searchlight Completion (✅)
- searchlight function with lazy evaluation support
- searchlight_coords returning coordinate matrices
- random_searchlight with non-overlapping searchlight generation
- bootstrap_searchlight with random sampling
- clustered_searchlight supporting both k-means and pre-defined clusters
- LazyList implementation for memory-efficient searchlight iteration
- Comprehensive test suite covering all searchlight variants

## Phase 10 Completion (✅)
- searchlight function with lazy evaluation support  
- searchlight_coords returning coordinate matrices
- random_searchlight with non-overlapping searchlight generation
- bootstrap_searchlight with random sampling
- clustered_searchlight supporting both k-means and pre-defined clusters
- LazyList implementation for memory-efficient searchlight iteration
- Comprehensive test suite covering all searchlight variants

## Phase 11 Completion (✅)
- conn_comp function with threshold support and cluster statistics
- conn_comp_3D function for simple binary mask processing
- ConnCompResult dataclass for structured results
- Support for 6/18/26-connectivity patterns in 3D
- Local maxima detection within clusters
- Cluster size and mass computation
- Comprehensive test suite with multiple connectivity options

## Phase 12 Completion (✅)
- split_blocks for dividing data by block indices
- split_clusters for separating data by cluster labels
- split_fill for filling NeuroVec by factor levels
- split_reduce for reducing data across factor levels
- split_scale for within-group scaling
- partition for k-means volume partitioning
- map_values for value remapping with lookup tables
- centroids for computing cluster centers
- Complete test coverage for all statistical operations

## Phase 13 Completion (✅)
- BigNeuroVec for memory-mapped 4D neuroimaging data
- FileBackedNeuroVec for lazy-loading from disk
- MappedNeuroVec for on-the-fly transformations
- Helper functions: big_neurovecseq, file_backed_neurovec, mapped_neurovecseq
- Transformation mappers: scale_mapper, log_mapper, threshold_mapper
- Efficient memory usage for large datasets
- Comprehensive test suite for all variants

## Priority Queue

1. **High Priority** (Phase 14):
   - Complete AFNI format reader implementation
   - Test conversion from R to Python
   - Documentation improvements
   
2. **Medium Priority** (Phase 15+):
   - NeuroVecSeq advanced operations parity
   - API compatibility shims/aliases for selected R naming conventions
   - Performance optimizations
   
3. **Lower Priority** (Future):
   - Advanced clustering algorithms
   - GPU acceleration support
   - Parallel processing utilities
