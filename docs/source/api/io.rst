File I/O
========

.. currentmodule:: neuroim

File I/O functions and classes for reading and writing neuroimaging data in formats like NIFTI, AFNI, and others. This module provides high-level functions for data access and low-level format handlers.

High-Level I/O
--------------

Main interface for reading and writing neuroimaging data.

.. automodule:: neuroim.io
   :members:
   :undoc-members:
   :show-inheritance:

File Formats
------------

Format detection and registry for neuroimaging file types.

.. automodule:: neuroim.file_format
   :members:
   :undoc-members:
   :show-inheritance:

Metadata
--------

Metadata extraction and management for neuroimaging files.

.. automodule:: neuroim.meta_info
   :members:
   :undoc-members:
   :show-inheritance:

Binary I/O
----------

Low-level binary reading and writing utilities.

.. automodule:: neuroim.binary_io
   :members:
   :undoc-members:
   :show-inheritance:

AFNI I/O
--------

AFNI format-specific I/O operations.

.. automodule:: neuroim.afni_io
   :members:
   :undoc-members:
   :show-inheritance:

NIFTI Utilities
---------------

NIFTI format utilities and helpers.

.. automodule:: neuroim.nifti_utils
   :members:
   :undoc-members:
   :show-inheritance:

NIFTI Extensions
----------------

NIFTI extension handling for embedded metadata.

.. automodule:: neuroim.nifti_extension
   :members:
   :undoc-members:
   :show-inheritance:

Data Sources
------------

Abstraction layer for data sources including files, URLs, and streams.

.. automodule:: neuroim.sources
   :members:
   :undoc-members:
   :show-inheritance:
