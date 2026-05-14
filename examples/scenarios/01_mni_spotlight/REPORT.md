# Scenario 01 — MNI Spotlight: Report

> Compare extracting a BOLD time series at a single world-mm coordinate
> in raw `nibabel`+`numpy` against the neuroim public API.

## Verdict

**Win on read-time, provenance, and total line count after the PAIN-1
and PAIN-3 fixes.** The user-facing function body falls from 10 → 1
statement (simple) or 1 (typed), the affine-inversion idiom collapses
to `series_at_world`, and the typed form ships a `Receipt` for free.
The original scenario exposed three pain points; PAIN-1/PAIN-2/PAIN-3
are now tracked by executable tests.

## Scoreboard

Counts are AST function-body statements, excluding docstrings.

| Axis | Baseline | Rewrite simple | Rewrite typed |
|---|---:|---:|---:|
| User-facing function body | 10 | **1** | **1** |
| + helpers needed *only* because of PAIN-1, PAIN-3 | — | +0 | +0 |
| **Total scenario code** | **10** | **1** | **1** |
| Named operations the user must know | `np.linalg.inv`, `@` homog, `np.round`, `astype(int)`, hand-written bounds-check | `series_at_world` | `series_roi_world` |
| Output type | bare `np.ndarray` | bare `np.ndarray` | `ROIExtractionResult` w/ `Receipt` |
| Provenance | none | none | `Receipt(method_name="series_roi_world", …)` |
| Bounds-check origin | user-written | user-written (filed pain) | user-written (filed pain) |

`np.linalg.inv(affine) @ [x, y, z, 1]` collapses to
`bold.series_at_world(mni)`. The user reads one domain operation instead
of affine algebra plus a manual bounds check.

## What the API caught vs what it did not

| Case | Baseline | Rewrite (simple) | Rewrite (typed) |
|---|---|---|---|
| In-bounds world coord | correct | correct (identical bytes) | correct + Receipt |
| Out-of-bounds world coord | raises (hand-coded) | raises via `series_at_world` | raises via `series_roi_world` |
| Silent numpy negative-index wrap | only because the user wrote the bounds check | **only because the user wrote the bounds check** | **only because the user wrote the bounds check** |
| Mismatched downstream space (compose later) | n/a (bare ndarray loses space) | n/a (bare ndarray loses space) | catchable via `Receipt.diff` |

The "silent numpy negative-index wrap" row is now covered by the
explicit `series_at*` bounds-check tests.

## Pain points surfaced and resolved

### PAIN-1 — resolved: `NeuroVec.series_at_world(mni_xyz)` shortcut

The user previously had to chain `vec.space.world_to_grid(...)` +
`vec.series_at(...)` to do what is conceptually one operation.
`NeuroVec.series_at_world(...)` now collapses the simple form to one
statement, and `NeuroVec.series_roi_world(...)` does the same for the
typed ROI result.

- **Impact**: ergonomics of the most common single-seed query.
- **Status**: fixed by `series_at_world` and `series_roi_world`, routed
  through `world_to_grid` and bounds-checked by default.

### PAIN-2 — resolved: `series_at(x, y, z)` bounds-checks

The previous implementation in `DenseNeuroVec._series` for the
single-voxel branch delegated directly to numpy indexing, so negative
voxel indices could silently wrap. The explicit `series_at*` methods
now reject out-of-bounds coordinates by default.

`series_at_coords` and `series_at_indices` now share the same default
raise-on-OOB contract. Intentional sparse/searchlight-style zero-fill
is still available via `out_of_bounds="zero"`.

- **Impact**: users passing coords derived from world-mm seeds no longer
  have to write the same defensive bounds check as the nibabel baseline.
- **Status**: fixed by default `IndexError` on explicit `series_at*`;
  `series_roi` opts into zero-fill internally to preserve its historical
  ROI contract.

### PAIN-3 — resolved: 4-D `NeuroSpace.world_to_grid` accepts 3-D spatial coords

A `NeuroVec`'s `.space` is 4-D, but world-mm seeds are inherently
spatial. `NeuroSpace.world_to_grid` and `grid_to_world` now accept
3-D spatial coords on N-D spaces and route them through the spatial
affine, so callers no longer need `drop_dim(3)`.

- **Impact**: a 4-D BOLD plus a 3-D MNI coord is the most common
  combination in fMRI Python code.
- **Status**: fixed by spatial-query support on `NeuroSpace` and
  `NeuroVec.spatial_space`.

## How to reproduce

```bash
PYTHONPATH=src:tests:. .venv/bin/python -m pytest \
    examples/scenarios/test_s01_mni_spotlight.py -q
```

Expect: 5 passed.

## What this scenario does *not* test

- File-backed / lazy reads (no `read_image(path, lazy=True)`).
- Multi-coord seeds — that's scenario 02 (spherical ROI at a seed).
- Non-isotropic affines with rotation — both forms use the fixture's
  diagonal affine. A rotated-affine variant is a candidate for a
  later scenario.

## Follow-ups

| Pain | Suggested issue title | Priority |
|---|---|---|
| PAIN-1 | `NeuroVec.series_at_world(mni_xyz)` ergonomic shortcut | fixed |
| PAIN-2 | `series_at` / `series_at_coords` must bounds-check by default | fixed |
| PAIN-3 | 4-D `NeuroSpace.world_to_grid` accepts spatial coords on a 4-D space | fixed |
