Changelog
=========

All notable changes to Neuroim will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
- Complete NeuroHyperVec implementation for 5D+ neuroimaging data
- Support for multi-echo fMRI, spectral analysis, and parameter maps
- HDF5 I/O for NeuroHyperVec data
- Comprehensive test suite with 543 passing tests
- R equivalence tests ensuring compatibility with neuroim2
- Memory-mapped array support for large datasets
- Sparse data structure optimizations
- AFNI HEAD/BRIK read/write support, including compressed BRIK round trips
- Lightweight AFNI NIML parser helpers
- neuroim2 compatibility aliases for literal exported R names, including dotted generics and camelCase helpers
- Joint spatial-temporal ``bilateral_filter_4d`` with mask-preserving behavior and R oracle coverage

Fixed
~~~~~
- Fixed all 41 initially failing tests
- Corrected array dimension handling in statistical operations
- Fixed coordinate system transformations
- Resolved ROI extraction issues
- Fixed searchlight API compatibility
- Added spacing validation to prevent invalid NeuroSpace creation
- Fixed Fortran-order voxel mapping in sparse masks, clustered volumes, searchlights, and split/statistical helpers
- Fixed 1D/2D embedded affine handling while preserving invalid-transform validation
- Improved AFNI-style ``clip_level``/``automask`` behavior and ``deoblique`` handling for ``NeuroSpace``

Changed
~~~~~~~
- Standardized method naming (vol_mean → mean)
- Improved add_dim parameter consistency
- Enhanced error messages throughout
- Updated import structure for better organization

[0.1.0] - 2024-01-30
--------------------

Initial release of Neuroim.

Added
~~~~~
- Core data structures: NeuroSpace, NeuroVol, NeuroVec, NeuroSlice
- File I/O support for NIFTI and AFNI formats
- ROI analysis tools (spherical, cubic, coordinate-based)
- Searchlight analysis framework
- Spatial filtering (Gaussian, bilateral, guided)
- Statistical operations (partition, split, reduce)
- Connected components analysis
- Resampling and interpolation
- Memory-efficient sparse representations
- Comprehensive documentation and tutorials
- Migration guide from R neuroim2

Known Issues
~~~~~~~~~~~~
- NIML writing is not yet implemented
- Broader cross-language assertions and large-scale performance envelopes can still be expanded

[0.0.1] - 2023-12-01
--------------------

Pre-release development version.

Added
~~~~~
- Basic project structure
- Initial NeuroVol and NeuroSpace implementations
- NIFTI reading capability
- Basic test framework

Future Releases
----------------

[0.2.0] - Planned
~~~~~~~~~~~~~~~~~

Planned features:

- Expanded NIML write/advanced AFNI coverage
- Parallel searchlight processing
- Enhanced visualization tools
- GPU acceleration options
- Additional statistical methods
- Integration with popular ML frameworks

[0.3.0] - Planned
~~~~~~~~~~~~~~~~~

Planned features:

- Surface-based analysis
- Diffusion imaging support
- Real-time fMRI capabilities
- Advanced connectivity analyses
- Improved performance optimizations

Contributing
------------

See `Contributing Guide <contributing.html>`_ for information on how to contribute to Neuroim.

Versioning
----------

We use `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`__:

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible new features  
- PATCH version for backwards-compatible bug fixes

How to Upgrade
--------------

To upgrade to the latest version::

    pip install --upgrade neuroim

Always check the changelog for breaking changes before upgrading major versions.
