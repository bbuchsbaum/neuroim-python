# Scenario 10 — Searchlight Space Mismatch

## Task

Compute a local mean searchlight map from a 4-D BOLD image and a 3-D mask.
The mask has the same voxel shape as the BOLD but may be in a different
spatial frame.

## Verdict

**Falsifies a mission-bearing safety claim.** The baseline needs an explicit
affine check, but once that check is written it refuses a shifted mask. The
neuroim surface should make this validation automatic for searchlight
workflows; currently the acceptance check is xfail-strict because a
same-shape shifted mask can still drive coordinate extraction from the BOLD
grid. The scenario also exposes a lower-priority implementation smell:
`searchlight_apply` calls the deprecated `NeuroVec.series()` method internally.

| Axis | Baseline nibabel | Neuroim today | Expected neuroim |
|---|---|---|---|
| Lines / read-time | Manual sphere traversal and explicit affine guard | One searchlight call | One searchlight call |
| Safety | Refuses shifted mask after a hand-written check | **PAIN-10:** same-shape shifted mask is not rejected | Refuses mask/data space mismatch before sampling |
| Inspectability | Returns anonymous `Nifti1Image` | Returns typed `SearchlightResult` with `Receipt` | Same, only after validated inputs |

## Pain Points Surfaced

### PAIN-10 (P0) — `searchlight_apply` does not reject mask/data space mismatch

**Impact.** Searchlight is a mission-named workflow. If a mask and BOLD share
shape but differ in affine, searchlight neighborhoods are selected in the mask
grid and sampled from the data grid. That can silently scatter every local
statistic while still returning a plausible typed result and receipt.

**Suggested fix.** Before iterating, require `mask.space.compatible_with(data.space)`
for spatial inputs that expose a `space`. Raise a clear `ValueError` mentioning
the mask/data affine or space mismatch. Keep protocol-only inputs supported by
using the same structural `space` check where available.

**Priority.** P0 — mission-bearing silent spatial error in a first-class
analysis workflow.

### PAIN-11 (P2) — `searchlight_apply` still uses deprecated `NeuroVec.series()`

**Impact.** The aligned scenario passes, but pytest reports one
`DeprecationWarning` per searchlight center because `searchlight_apply` calls
`data.series(sl.coords)` internally. Users do not see this in normal Python by
default, but the implementation is leaning on a surface the package itself says
is deprecated.

**Suggested fix.** Replace the internal call with `series_at_coords(sl.coords)`
or the protocol-equivalent path, while preserving the current `(time, voxel)`
array semantics.

**Priority.** P2 — not a correctness bug, but it weakens the curated public API
story and keeps tests noisy.

## Follow-ups

| Pain | Priority | Suggested issue title |
|---|---:|---|
| PAIN-10 | P0 | `searchlight_apply` must reject mask/data space mismatch |
| PAIN-11 | P2 | `searchlight_apply` should stop calling deprecated `NeuroVec.series()` |
