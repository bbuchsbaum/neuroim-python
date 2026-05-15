# Scenario 12 -- Gaussian spatial smoothing at FWHM = 6 mm

**Task.** Given a 3-D stat map (or any volumetric image) and an optional
brain mask, apply isotropic-mm-space Gaussian smoothing at FWHM = 6 mm.
The single most common preprocessing step in fMRI downstream of motion
correction.

**Why this scenario exists.** S01-S11 all produce *smaller* outputs (time
series, parcel matrices, cluster tables).  S12 is the first scenario for
a *same-shape transformation* (vol -> vol of identical dims).  It also
surfaces a canonical physical-unit pitfall (mm vs voxels) that exists
across implementations -- including, until this scenario landed, a
silent semantic divergence between R neuroim2 and the Python port at the
same call name.

## Verdict

| Axis | Baseline | neuroim after PAIN-1/2/3/4 |
|---|---|---|
| User-facing body statements | ~6 | 1 (`gaussian_blur(vol, fwhm_mm=6.0)`) |
| Same-space mismatch caught | no | yes (inside `gaussian_blur`) |
| Isotropic-mm-space on anisotropic voxels | yes (manual conversion) | yes (`fwhm_mm=` path) |
| Output carries provenance Receipt | no | yes (`method_name="gaussian_blur"`, `radius=fwhm_mm`) |
| Discoverable from `ni.__all__` | n/a | yes |

**neuroim now wins the call-site shape for this workflow** and is the
first scenario to ship with a verified mm-space contract.  The scenario
test `test_legacy_voxel_sigma_is_anisotropic_in_mm_on_anisotropic_voxels`
proves the bug class is real on the legacy scalar-sigma path (3 x 3 x 3.5
mm fixture produces measurably anisotropic mm-space spread), and
`test_fwhm_mm_path_is_isotropic_in_mm_on_anisotropic_voxels` proves the
`fwhm_mm=` path closes it.

## PAINs

### PAIN-1 (P0 mission) -- closed: same-space gate inside `gaussian_blur`

`gaussian_blur(vol, mask=foreign_affine)` now invokes
`verify.assert_same_space(vol, mask)` before any smoothing.  An LR-flipped
mask raises `ValueError("spatial contract mismatch ...")` immediately.

### PAIN-2 (P1 ergonomic / parity) -- closed: `fwhm_mm=` parameter

`gaussian_blur(vol, fwhm_mm=6.0)` is the recommended path.  Internally
converts FWHM-in-mm to per-axis voxel sigma via `vol.space.spacing` and
calls `scipy.ndimage.gaussian_filter` with a per-axis sigma sequence and
the scipy default kernel extent (`truncate=4`).  The legacy scalar
`sigma=` parameter is retained for backward compatibility but is now
documented as voxel-space; the parity hazard against R neuroim2 (whose
C++ kernel at `~/code/neuroim2/src/indexFuns.cpp:174-193` interprets
sigma in mm and applies spacing internally) is called out in the function
docstring.

### PAIN-3 (P1 provenance) -- closed: output carries Receipt

The returned `DenseNeuroVol` carries `.provenance` with
`method_name == "gaussian_blur"`, `radius == fwhm_mm` (or the legacy
sigma when `fwhm_mm` is not supplied), `input_space_hash` from the input
vol's space, and `mask_hash` from the mask payload when supplied.

### PAIN-4 (P2 surface) -- closed: `gaussian_blur` in `ni.__all__`

`neuroim.__all__` now includes `gaussian_blur`.  Budget bumped to 41.
`tests/test_public_api.py::test_public_api_budget_and_required_names`
and `test_dir_tracks_curated_public_surface` updated to reflect the new
canonical preprocessing surface.

## Fixture

Reuses the 32 x 32 x 24 x 40 BOLD from `tests/fixtures/realistic_bold.py`
with voxel size 3 x 3 x 3.5 mm -- enough anisotropy to make the FWHM-in-mm
vs voxel-sigma distinction observable in the spread along z vs x/y.  The
stat map for smoothing is the t=0 slice of the BOLD volume.

## Acceptance state

`PYTHONPATH=src:tests:. python -m pytest tests/test_s12_gaussian_smoothing.py`:

- **6 passed, 0 xfailed** -- numeric parity at FWHM = 6 mm; same-space
  gate; the legacy voxel-sigma anisotropy demonstration; the `fwhm_mm=`
  isotropy fix; provenance Receipt; public namespace export.

Broader regression slice (`test_s11`, `test_s12`, `test_public_api`,
`test_image_processing`, `test_phase7_spatial_filters`,
`test_verifier_blessed_path`, `test_fragility_invariants`): 89 passed.

## Source

- Baseline: `examples/scenarios/12_gaussian_smoothing/baseline_nibabel.py`
- Rewrite: `examples/scenarios/12_gaussian_smoothing/neuroim_version.py`
- Test: `tests/test_s12_gaussian_smoothing.py`
- API surface: `src/neuroim/spatial_filters.py::gaussian_blur` +
  `src/neuroim/__init__.py::__all__`.
