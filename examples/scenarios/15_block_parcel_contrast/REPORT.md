# Scenario 15 - Block Parcel Contrast

## Task

Compute a task-minus-rest effect per atlas parcel from a 4-D BOLD run, then
map those parcel effects back into a 3-D volume.

## Verdict

**Clear win on analysis readability and spatial safety.** The raw nibabel/numpy
version is straightforward, but the analysis is buried in array extraction,
label discovery, manual condition validation, and voxel scatter code. The
neuroim version lets the whole analysis sit at the call site:

```python
result = bold.parcel_means(atlas).contrast(condition)
effect_map = result.map_to_volume()
```

The same-space gate is owned by `parcel_means`, so a same-shape atlas with a
wrong affine fails before any parcel averaging. The baseline silently returns
a plausible wrong result unless every caller remembers to write the affine
guard. `contrast()` and `map_to_volume()` own label/effect alignment and
projection back to image space.

## What It Tests

| Check | Why it matters |
|---|---|
| Numeric parity | Neuroim produces the same parcel effects and effect map as raw nibabel/numpy on matched inputs. |
| Spatial contract | Neuroim rejects a same-shape LR-flipped atlas; the baseline accepts it. |
| Typed atlas provenance | The contrast result carries Schaefer fixture provenance. |
| Analysis signal | The winning parcel contains the synthetic target ROI from the fixture. |

## Scorecard

| Axis | Baseline nibabel/numpy | neuroim |
|---|---|---|
| Lines / read-time | More bookkeeping: `dataobj`, label discovery, manual scatter. | The workflow is two analysis-level calls: `parcel_means(...).contrast(...)`, then `map_to_volume()`. |
| Safety | Shape check only in this baseline; affine checks are manual discipline. | `bold.parcel_means(atlas)` enforces same-space. |
| Inspectability | Output image has affine and data only. | `ParcelEffectResult` has chained `Receipt` provenance and typed atlas source metadata. |

## Pain Points Surfaced

### PAIN-1 (P2, closed): no first-class parcel contrast result

Original impact: Scenario 15 needed a small local `ParcelContrast` dataclass
and manual projection from parcel effects back to a 3-D map.

Resolution: `ClusteredNeuroVec.contrast(condition, positive_name="task",
negative_name="rest")` now returns `ParcelEffectResult` with `.effects`,
`.labels`, `.map_to_volume()`, chained `Receipt` provenance, and
`atlas_provenance`.

## Follow-ups

| Priority | Suggested title |
|---|---|
| closed | Add first-class parcel contrast/reduce result for `ClusteredNeuroVec` |
