# Scenario 03 — Seed-Sphere Mean: Report

> Compare extracting the **mean BOLD time series** in an 8 mm spherical
> ROI around an MNI coordinate, in raw `nibabel`+`numpy` vs the neuroim
> public API.

## Verdict

**Clean win on every axis** — the rewrite's user-facing function body
is **1 statement** vs the baseline's **15**, no helpers are needed
(the PAIN-1/2/3 fixes from Scenario 01 already landed), and the typed
form ships a populated `Receipt`. Two **provenance-narrowing** pain
points were surfaced and fixed: `Receipt.radius` now records the radius
the caller passed, and `Receipt.method_name` records the public
`series_roi_world` entry point instead of the internal `series_roi`
helper.

## Scoreboard

Counts are AST function-body statements, excluding docstrings.

| Axis | Baseline | Rewrite simple | Rewrite typed |
|---|---:|---:|---:|
| User-facing function body | 15 | **1** | **1** |
| + helpers needed | — | 0 | 0 |
| **Total scenario code** | **15** | **1** | **1** |
| Named operations the user must know | `np.linalg.inv`, `@` homog, `np.round`, spacing-from-affine norm, `np.mgrid`, distance map, mask, `data[mask].mean(axis=0)`, hand-written bounds-check | `series_roi_world(mni, radius=R)`, `.values.mean(axis=1)` | `series_roi_world(mni, radius=R)` |
| Output type | bare `np.ndarray` | bare `np.ndarray` | `ROIExtractionResult` w/ `Receipt` |
| Provenance | none | none | `Receipt(method_name="series_roi_world", radius=R, n_voxels=…, …)` |
| Bounds-check origin | user-written | API (raises) | API (raises) |
| Spacing-weighted distance | user-written, easy to get wrong on rotated affines | from `series_roi_world` | from `series_roi_world` |

`series_roi_world(mni, radius=R)` collapses affine inversion + a
spacing-weighted distance map + masking + bounds-checking into one
named call. The rewrite ergonomics are the cleanest in the suite so
far.

## What the API caught vs what it did not

| Case | Baseline | Rewrite (simple) | Rewrite (typed) |
|---|---|---|---|
| 8 mm sphere at in-bounds MNI | correct | correct (identical bytes) | correct + per-voxel `(T, V=73)` |
| `radius_mm = 0` | nearest-voxel via hand-coded fallback | single voxel | single voxel + `n_voxels=1` |
| Centre maps off the grid | raises (hand-coded) | raises via `series_roi_world` | raises via `series_roi_world` |
| Non-isotropic voxels | correct *only if* user remembers spacing weighting | correct (handled internally) | correct (handled internally) |
| Rotated affine | baseline's `np.linalg.norm(affine[:3,:3], axis=0)` works, but most hand-written sphere code uses `np.diag(...)` and gets it wrong | correct | correct |

The "rotated affine" row is the one nibabel users get burned on
without realising: a sphere built from `np.diag(affine)[:3]` as the
spacing vector is *only* correct for axis-aligned affines.

## Pain points surfaced

### PAIN-1 (P2, fixed) — `Receipt.radius` records `series_roi_world(..., radius=R)`

After `bold.series_roi_world(mni, radius=8.0)`, the returned
`ROIExtractionResult.provenance.radius` is `8.0`.

The radius is a **defining ROI parameter** of the spherical query;
losing it from provenance defeats one of the Receipt's purposes —
namely, being able to reproduce or audit a result without re-reading
the call site.

- **Fix**: `NeuroVec.series_roi_world` threads the caller's `radius`
  into the `series_roi` receipt path.

### PAIN-2 (P3, fixed) — `Receipt.method_name` records `series_roi_world`

The world-coordinate entry point is now visible in the provenance
trail. A caller that runs `series_roi_world(mni, radius=8.0)` receives
`provenance.method_name == "series_roi_world"` rather than the internal
`series_roi` helper name.

- **Fix**: `series_roi_world` passes its public method name into the
  receipt path.

## How to reproduce

```bash
PYTHONPATH=src:tests:. .venv/bin/python -m pytest \
    examples/scenarios/test_s03_seed_sphere_mean.py -q
```

Expect: 7 passed.

## What this scenario does *not* test

- Non-isotropic voxels with rotation in the affine — fixture is
  diagonal. A `45°`-rotated fixture would be a useful scenario follow-up.
- The Receipt **mask_hash**. The current scenario builds the ROI
  from a sphere, not from a mask object; mask hashes are exercised
  in scenario 02 (`02_roi_mean_timeseries`).
- File-backed / lazy reads.

## Follow-ups

| Pain | Suggested issue title | Priority |
|---|---|---|
| Rotated affine sphere fixture | Add a scenario with non-diagonal spatial affine to harden spacing-weighted sphere construction. | P3 |
