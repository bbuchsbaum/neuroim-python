# Scenario 02 — ROI mean time series: Report

> Compare extracting the mean BOLD time series across a brain mask in
> raw `nibabel`+`numpy` against the neuroim public API.

## Verdict

**Win on contract semantics; net wash on lines.** The neuroim form
collapses the manual `shape ==`, `np.allclose(affine)`, and
boolean-mask boilerplate into one call, and ships provenance with the
typed form.  PAIN-5 was surfaced by this scenario and is now fixed:
when the mask carries a different affine, `series_roi` raises through
the contract layer instead of scattering values as if the spaces
matched.

## Scoreboard

Counts are AST function-body statements, excluding docstrings.

| Axis | Baseline | Rewrite simple | Rewrite typed |
|---|---:|---:|---:|
| User-facing function body | 11 | **5** | **5** |
| + helpers needed because of PAIN-4 | — | +2 (`_roi_from_mask`) | +2 |
| **Total scenario code** | **11** | **7** | **7** |
| Named operations the user must know | `get_fdata`, manual `shape ==`, `np.allclose(affine)`, boolean-mask, `mean(axis=0)` | `ROICoords`, `series_roi`, `values.mean(axis=1)`, `argwhere` | `ROICoords`, `series_roi`, `argwhere` |
| Output type | bare `np.ndarray` | bare `np.ndarray` | `ROIExtractionResult` w/ `Receipt` |
| Mask-shape mismatch | catches via `mask.shape == bold.shape[:3]` | catches via contract layer | catches via contract layer |
| Mask-affine mismatch (different spatial frame) | catches via `np.allclose(affine)` | catches via contract layer | catches via contract layer |
| Empty-mask | catches | catches | catches |
| Provenance | none | none | `Receipt(method_name="series_roi", n_voxels=...)` |

The headline ergonomics delta is **lines 11 → 7** for total scenario
code: 5 statements in the user-facing call plus the 2-line mask→ROI
helper.  If PAIN-4 is fixed, total drops to ~5.  The headline
**safety** delta is now positive: the rewrite catches the same
shape/affine/empty-mask failures as the baseline and adds typed
provenance.

## What the API caught vs what it did not

| Case | Baseline | Rewrite (simple) | Rewrite (typed) |
|---|---|---|---|
| Matched mask | correct | correct, identical bytes | correct, identical bytes + Receipt |
| Mask shape mismatch | raises | raises (via contract) | raises (via contract) |
| Mask affine mismatch | **raises** (hand-coded) | **raises** (contract layer) | **raises** (contract layer) |
| Empty mask | raises | raises | raises |

The "mask affine mismatch" row is the row the typed-spatial-objects
story exists to remove.  After PAIN-5, removing the user-written
affine compare no longer loses safety against the baseline.

## Pain points surfaced

### PAIN-4 (P3) — `LogicalNeuroVol` has no `to_roi_coords()` helper

To use a mask with the ROI APIs the caller must write
`np.argwhere(np.asarray(mask.data))` themselves.  A
`mask.to_roi_coords()` (or `ROICoords.from_mask(mask)`) one-liner
would remove the 2-line helper the rewrite needs.

`mask.coords()` is a footgun here: it returns *all* `(nx*ny*nz, 3)`
grid coordinates, not just the True voxels.  Callers reasonably
expect a mask to expose its True coords as the primary view.

### PAIN-5 (P0, fixed) — `series_roi` validates mask/data space agreement

This is a **mission-level claim under test**.  From `MISSION.md`
decision rule 4:

> Receipts by default. ROI and searchlight outputs carry provenance
> metadata; silent space/orientation/mask mismatches are caught at
> the contract layer, not in debugging.

The original failure was empirical: with the fixture's `rotated_mask`
(same `.data` as the matched mask but with an **LR-flipped 4×4
affine**):

```python
bold.series_roi(ROICoords(coords, rotated_mask.space)).values
```

returns **identical bytes** to:

```python
bold.series_roi(ROICoords(coords, mask.space)).values
```

The fix routes `NeuroVec.series_roi` and the volume analogue
`roi.values_roi` through `neuroim.verify.assert_same_space` before
extracting.  The acceptance test
`test_mask_space_mismatch_raises_through_neuroim` now runs without an
xfail and passes.

Tracker: **bd-01KRKNSK2A5DVWRHW63637MM1M (P0)**.

This finding justified the scenario suite as a mission-evidence
vector: the ME-2 receipt verifier existed, ME-9 provenance composition
shipped, and ME-3 audit verdicts passed, yet the first realistic
mask-mismatch test falsified the headline mission claim.  The scenario
now guards the fixed contract.
