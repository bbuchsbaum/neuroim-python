# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/)
versioning.

## [0.3.0a1] - Unreleased

First alpha. Pre-release cleanup removing legacy/cruft and unpythonic
surfaces before the API is exposed to users. Breaking changes are
expected at this stage and are not separately deprecated.

### Removed

- Deprecated alias packages `neuroimpy` and `pyneuroim`.
- The `_DEPRECATED_COMPAT_ALIASES` top-level camelCase bridge
  (`createNIfTIHeader`, `findAnatomy3D`, `mapToColors`, ...). These
  names live in `neuroim.compat` only.
- Lowercase R-style constructor aliases from the `neuroim` package
  root (`neurospace`, `neurovol`, `neurovec`, `neurovecseq`,
  `neuroslice`, `slices`, `roicoords`, `roivol`, `big_neurovecseq`,
  `file_backed_neurovec`, `mapped_neurovecseq`). The factory functions
  remain available from their submodules.
- `read_image`, `read_vol`, `read_vec`, `read_vol_list`, `load_data`,
  `read_header`, `read_meta_info` from the `neuroim` package root.

### Changed

- `NeuroHyperVec` is now the class (was shadowed by a factory function
  of the same name); construct concrete types with
  `NeuroHyperVec.create(...)` or the `Dense`/`Sparse` classes directly.
- Public readers are `read_volume` / `read_series`. The full I/O
  surface lives in the `neuroim.io` subpackage; neuroim2-style
  `read_vol` / `read_vec` live in `neuroim.compat`.
- Minimum Python raised to 3.10; dependency floors modernized
  (numpy>=1.24, scipy>=1.10, nibabel>=4.0, matplotlib>=3.6,
  scikit-learn>=1.2, h5py>=3.7, joblib>=1.2).
- Packaging now ships only the `neuroim` package (was also installing
  `neuroimpy`/`pyneuroim`/`examples`/`fixtures`).

### Added

- `LICENSE` (MIT) and this changelog.
- `[tool.ruff]` configuration.
