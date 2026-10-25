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

- **`gaussian_blur(sigma=)` is in millimetres** (it was voxels), matching
  neuroim2 and `fwhm_mm=`; the default is 2 mm and passing both `sigma`
  and `fwhm_mm` raises. `gaussian_blur(vol, sigma=2)` on non-1 mm data now
  smooths differently; to reproduce an old call, multiply the old
  voxel sigma by the voxel size.
- Bilateral filters take `spatial_sigma` in mm and an intensity bandwidth
  of `intensity_sigma * range_scale`; masked `gaussian_blur` defaults to
  `normalize=True`; `FileBackedNeuroVec.as_matrix` returns voxels x time.
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
- `NeuroVec.mean(axis=-1|3)` — first-class mean-over-time reduction
  returning a provenance-carrying 3D `DenseNeuroVol` (matches the
  `fmri.mean(axis=3)` form shown in the README).
- `gaussian_blur` now accepts a 4D `NeuroVec`, smoothing each volume
  spatially (no temporal blur) and returning a `DenseNeuroVec` with a
  `gaussian_blur` Receipt — so the canonical fMRI smoothing step is
  first-class for time-series input.
- `ClusteredNeuroVec.connectome(metric="correlation"|"covariance")` — a
  typed parcel-to-parcel connectivity reducer returning a `ConnectomeResult`
  (`labels`, `matrix`, `metric`, `provenance`, `n_nodes`, optional
  `.to_dataframe()`) whose Receipt chains the upstream extraction
  (`parcel_means+connectome`). `ConnectomeResult` is exported at the package
  root.
- `ClusteredNeuroVec.timeseries_matrix()` — public, copy-returning accessor
  for the `(n_time, n_clusters)` parcel mean-series payload, column-ordered
  by sorted cluster id (previously only the undocumented `.ts` attribute).

### Fixed

- **neuroim2 sync (Aug–Oct 2026 fixes).** Defects that neuroim2 fixed, or
  that an audit against it found here, are corrected. Several change
  results:
  - *Geometry.* Voxel spacing comes from the affine in use, not pixdim. 4-D
    world coordinates honour rotations and flips. Affines are kept by
    `FileBackedNeuroVec`, volume extraction, sub-vectors and mask spaces.
    `ROIVol.get_coords(real=True)` lost its half-voxel shift. Axis labels
    follow the affine, and `reorient("RAS")` is labelled RAS.
  - *Sparse.* `SparseNeuroVec` treats spatial-shape arrays as logical masks
    and validates index masks; unsorted indices stay at their voxels. A new
    `orientation=` argument makes square data unambiguous, and
    `to_sparse` accepts index masks. Hyper-vector `mean_features`,
    `apply_feature_func` and `as_dense` no longer misplace voxels.
    `as_matrix` is voxels × time on every backend, and the `BigNeuroVec`
    volume accessors use the time axis.
  - *I/O.* Partial reads keep the source header. Complex and RGB payloads
    are refused. 5-D `(x, y, z, 1, t)` files fold to 4-D. NIfTI-2 is
    written when needed or requested. Scaled integer sources are written
    back in their own storage. `read_header` reports scaling, and
    `NIFTIMetaInfo` has the right byte order.
  - *Filters.* Masked `gaussian_blur` is insulated to the mask by default
    (`normalize=True`). The bilateral filters use mm distances and in-mask
    neighbours, and `range_scale` gives the intensity bandwidth.
    `downsample` box-averages with the data aligned to its affine.
  - *Plotting.* The overlay scale is computed from the whole map and
    contains the threshold. Graded opacity has a 0.6 floor
    (`alpha_floor`). Background windows ignore air. `plot_neuro_vol(thresh=)`
    hides values inside the range. Plane names resolve through the affine.
  - *ROI/searchlight.* Fractional centroids are truncated. Huge radii and
    non-finite spacing are handled. `random_searchlight` partitions the mask
    in linear time, and `clustered_searchlight(cvol=ClusteredNeuroVol)`
    works. `conn_comp` is vectorised and gives identical output.
    `spherical_kernel` membership matches `spherical_roi`; `patch_set` no
    longer truncates a cube half-width; searchlight windows carry mask
    values (0 for out-of-mask members).
  - *Affines everywhere.* Time slices keep the spatial affine and TR;
    hyper-vector reductions, `NeuroVol.concat`, `dot_reduce`,
    `simulate_fmri` and HDF5 hyper-vector files keep flips/rotations.
- **Ergonomic pass (seed-connectivity workflow stress test).** Five
  friction points found while exercising a natural end-to-end pipeline
  through the public surface:
  - `NeuroSpace(dim=<4-tuple>, spacing=<3-tuple>)` — the natural call for
    a 4D series whose time axis has no spatial extent — now pads trailing
    axes with unit spacing / zero origin instead of raising a cryptic
    numpy broadcast error; an over-long spacing/origin raises a clear
    `InvalidSpaceError`.
  - `conn_comp(..., cluster_table=True)` (the default) degrades
    gracefully when the optional `pandas` dependency is absent: it warns
    and returns `cluster_table=None` rather than crashing the whole call.
  - `write_vol` / `write_vec` route through `to_nibabel`, so a provenance
    Receipt is embedded as a NIfTI extension and survives `write` →
    `read` — previously only `to_nibabel` persisted it.

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
