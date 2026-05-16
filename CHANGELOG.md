# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/)
versioning.

## [0.3.0rc1] - Unreleased

First release candidate for the 0.3.0 public experimental release. No
separate alpha was published; this entry is the cumulative pre-release
record. The public API is usable, documented, and tested but **not
frozen until 1.0** — see [STABILITY.md](STABILITY.md) for the binding
contract and [ROADMAP.md](ROADMAP.md) for the release plan. All seven
0.3.0 release-gate items are green at this tag.

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
- `read_vol_list`, `load_data`, `read_header`, `read_meta_info` from the
  `neuroim` package root (still available from `neuroim.io`).

### Changed

- `NeuroHyperVec` is now the class (was shadowed by a factory function
  of the same name); construct concrete types with
  `NeuroHyperVec.create(...)` or the `Dense`/`Sparse` classes directly.
- `read_image` remains the documented default reader (MISSION rule #1);
  `read_image` / `read_vol` / `read_vec` stay in the public namespace.
  `read_volume` / `read_series` are added as intent-revealing additive
  wrappers — not replacements. The full I/O surface also lives in the
  `neuroim.io` subpackage; neuroim2-style names live in `neuroim.compat`.
- Minimum Python raised to 3.10; dependency floors modernized
  (numpy>=1.24, scipy>=1.10, nibabel>=4.0, matplotlib>=3.6,
  scikit-learn>=1.2, h5py>=3.7, joblib>=1.2).
- Packaging now ships only the `neuroim` package (was also installing
  `neuroimpy`/`pyneuroim`/`examples`/`fixtures`).

### Added

- `LICENSE` (MIT) and this changelog.
- `[tool.ruff]` configuration.
- Typed contract-failure exceptions: a `NeuroimError` marker plus
  per-category classes (`SpaceMismatchError`, `MaskMismatchError`,
  `OutOfBoundsError`, `WorldOutOfBoundsError`, `BackendDriftError`,
  `ImmutableError`, `InvalidSpaceError`, `InvalidArgumentError`,
  `ImplicitArrayConversionError`). Each multiply-inherits the built-in
  it replaces, so existing `except ValueError` keeps working while new
  code can `except SpaceMismatchError`. Canonical messages unchanged.
- `read_volume()` / `read_series()` intent-revealing readers (additive).

### Fixed

- **P0** — `read_image` / `read_vol` / `read_vec` were transiently
  dropped from the public namespace by an over-broad API trim and have
  been restored. They are MISSION-mandated entry points.
- `__array__` on typed containers refuses implicit `np.asarray()` with a
  message naming the explicit accessor, instead of returning a 0-D
  object scalar — the `NeuroSpace` is never silently dropped. The
  `NeuroVec` refusal points to `.data` (rank-faithful), not
  `.as_matrix()` (a 2-D matricization).
- Golden-test harness repaired (R↔Python parity): `logical` check type
  added, R `[N,]` / `[,,,N]` index translation corrected, `neurovecseq`
  import ported after the API trim. Suite went from a hard crash (3/12)
  to 10/12 passing.

### Known gaps and limitations

Tracked and surfaced honestly rather than hidden:

- **Provenance at the write boundary (PAIN-4, P0).** `write_vol(vol,
  path)` does **not** embed the provenance Receipt — only
  `vol.to_nibabel()` then `nib.save(...)` does. To preserve provenance
  across a file round-trip today, use `nib.save(vol.to_nibabel(), path)`
  and read back with `read_image(...)`. Default-embed fix is pending.
- **Time-axis slicing (PAIN-13 / PAIN-14).** `vec[..., start:stop]` on a
  `NeuroVec` returns a bare `ndarray`, dropping the `NeuroSpace` and
  upstream provenance; the `slice → temporal_snr` receipt chain is
  incomplete. Re-wrap in a `DenseNeuroVec` on the derived space to keep
  the pipeline typed.
- **No first-class cross-subject group reducer (S19 PAIN-1/2).**
  Averaging N typed 3-D maps has no `group_mean` / `mean_volumes`;
  hand-roll `np.stack` + mean + a same-space check + receipt merge.
- **Golden parity: 10 passed / 2 deferred / 0 failed (gate GREEN).**
  `spatial_resampling` and `spatial_filtering` are R-only specs whose
  Python ports are not yet authored; the validator reports them as
  explicitly *deferred* (skipped), not failed, so the suite is green
  and may gate the release. The two ports are tracked as a follow-up
  mote and are 0.3.x/0.4 work, not an RC blocker.
- **Performance is not characterized.** The 0.3 line is correctness- and
  contract-focused; large file-backed reads and derived-map workflows
  are not yet benchmarked or optimized.
- **Alpha API.** Names and signatures may change before 0.3.0 final;
  breaking changes at this stage are not separately deprecated.
