# Scenario 17 - Local Block-Effect Searchlight

## Task

Compute a spherical searchlight map where each in-mask center receives the
local task-minus-rest effect averaged across all voxels in its radius.

## Verdict

**Typed-result and spatial-safety win.** The neuroim workflow is the intended
analysis shape:

```python
result = ni.searchlight_apply(
    mask,
    radius=6.0,
    method=local_effect,
    data=bold,
    nonzero=True,
)
effect_map = result.map_to_volume()
```

The raw nibabel/numpy baseline is numerically equivalent, but it has to own
the full loop over mask centers, mm-radius neighborhood construction,
time-by-voxel extraction, scalar projection, and image reconstruction.

The safety win is clear: a same-shape searchlight mask with the wrong affine
is accepted by the baseline and rejected by neuroim before neighborhood
sampling.

## What It Tests

| Check | Why it matters |
|---|---|
| Numeric parity | The neuroim `SearchlightResult.values` and projected map match the hand-coded baseline. |
| Spatial contract | Neuroim rejects a same-shape wrong-affine mask before local sampling. |
| Typed projection | `.map_to_volume()` is the owned path from center values back to a 3-D map. |
| Signal sanity | The known synthetic target lands in the high-effect tail of the searchlight map. |

## Scorecard

| Axis | Baseline nibabel/numpy | neuroim |
|---|---|---|
| Lines / read-time | Manual center loop, radius math, indexing, scalar projection. | One `searchlight_apply` call plus a named local statistic. |
| Safety | Shape checks only unless affine checks are added by hand. | `searchlight_apply` owns the data/mask same-space check. |
| Inspectability | Anonymous values and image. | `SearchlightResult` carries centers, radius, method name, and Receipt. |
| Projection | Caller scatters values into a NIfTI manually. | `result.map_to_volume()` owns projection and provenance carry-forward. |

## Pain Points Surfaced

None from this slice. The result object records the local method name, the
receipt records the same method name, and `.map_to_volume()` carries that
receipt onto the projected map.

## Follow-ups

| Priority | Suggested title |
|---|---|
| none | no open follow-up |
