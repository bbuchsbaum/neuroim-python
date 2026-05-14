# Draw Audit (ME-3)

> Side-by-side comparison of one realistic workflow — per-voxel correlation
> of a BOLD time series with a behavioural regressor inside a mask —
> implemented twice, once in raw `nibabel`+`numpy` and once through the
> public `neuroim` API.
>
> Fixture: `tests/fixtures/realistic_bold.py::make_realistic_bold()` —
> a deterministic 32×32×24×40 float64 BOLD volume with an ellipsoidal mask,
> an LR-flipped mask variant for the bug-class test, and a behavioural
> regressor that correlates `r > 0.5` inside a 3×3×3 target ROI.

## Acceptance summary

| # | Acceptance criterion | Status | Notes |
|---|---|---|---|
| 1 | Line count delta — 30%+ reduction | **PASS** | Function-body lines: 85 → 59 (**−31%**). |
| 2 | Explicit error checks removed | **PASS** | Manual `shape` check, manual mask-index ravel, manual mapping-back-to-3D, and the "trust the affine" comment all removed. |
| 3 | `nibabel` / `numpy.reshape` / manual affine math removed | **PASS** | `nib.`/`np.` references: 15 → 9 (**−40%**). No manual `@ affine`; the spatial contract is carried by `NeuroSpace`. |
| 4 | Bug-class test: baseline silently wrong, rewrite raises | **PASS** | `test_baseline_silently_accepts_wrong_space_mask` and `test_rewrite_raises_on_wrong_space_mask` both pass — the LR-flipped mask produces a plausible-but-wrong correlation map in the baseline and raises `ValueError` from `NeuroSpace.compatible_with` in the rewrite. |
| 5 | Read time qualitative note | **PASS** | See [§Read-time qualitative](#read-time-qualitative). |
| 6 | Memory peak for file-backed input | **DEFERRED** | The fixture is in-memory `DenseNeuroVec`; the file-backed lazy path is exercised by `tests/test_nibabel_interop.py::test_from_nibabel_lazy_does_not_materialize`, not by this audit. |

## Measurements

### Line count
| | `baseline_nibabel.py` | `neuroim_rewrite.py` |
|---|---:|---:|
| Total file lines | 108 | 97 |
| Function body lines | 85 | **59 (−31%)** |
| AST statements in body | 30 | 27 |

The function-body reduction (the apples-to-apples figure) clears the 30%+
target. The biggest single source of removed lines is the mask-index
bookkeeping loop in the baseline:
```python
mask_indices = np.argwhere(mask)
n_vox = mask_indices.shape[0]
masked_series = np.empty((n_vox, nt), dtype=np.float64)
for i, (ix, iy, iz) in enumerate(mask_indices):
    masked_series[i, :] = bold[ix, iy, iz, :]
```
The rewrite collapses that block to:
```python
extract = bold.series_roi(roi)  # default return_legacy=False
```
which both extracts the matrix and carries the coordinate array forward
in the typed `ROIExtractionResult`.

### Explicit-check delta
Each row is a check / sanity assertion the user had to write themselves
in the baseline, and that the rewrite either deletes or moves into a
contract the API checks at the entry point.

| Check | Baseline | Rewrite |
|---|---|---|
| `bold.shape[:3] == mask.shape` | manual `if/raise` | enforced by `NeuroSpace.compatible_with` |
| BOLD↔mask affine agreement | **not checked** (silent bug) | enforced by `NeuroSpace.compatible_with` |
| `regressor.shape == (nt,)` | manual `if/raise` | identical manual check (no improvement) |
| Mask→flat-series mapping | manual indexed loop | replaced by `series_roi` |
| Flat-series→3D-map mapping | manual indexed loop | replaced by `coords` + scatter |
| Output affine | manually passed via `bold_img.affine` | carried by `extract.space` |
| Provenance | none | `extract.provenance` (Receipt) |

The "silent bug" row is the one that matters for the mission claim.
The baseline accepts any mask whose **shape** matches; it silently ignores
the mask's affine. The rewrite makes that mismatch loud at the API surface.

### `nib.` / `np.` references
* Baseline: 15 (`nib.Nifti1Image`, `nib.save`, repeated `np.float64` /
  `np.argwhere` / `np.empty` / `np.full` / `np.linalg.norm` /
  `np.linalg.norm` / `np.asarray`).
* Rewrite: 9 (`np.argwhere` for the mask coords, `np.full` for the output
  grid, two `np.linalg.norm`, `np.asarray`, and the typed-result scatter).

The rewrite still uses numpy for the math (Pearson) — which is correct,
because the mission rule is "don't compete on what nibabel/numpy do well."
What goes away is the numpy/nibabel **plumbing** between extraction and
analysis.

### Bug-class divergence (the mission claim)
Both implementations were run on the same fixture with the LR-flipped
`rotated_mask`. The mask data is identical to the happy-path mask; only
its affine differs.

```text
$ pytest examples/draw_audit/test_audit.py -q
....                                                                     [100%]
4 passed
```

| | Baseline | Rewrite |
|---|---|---|
| Result on `rotated_mask` | finite correlation map, no error | `ValueError: assert_same_space: input_space_hash mismatch` |
| Failure mode | silent wrong numbers | hard fail at the API boundary |
| Time to detect | analyst notices the map "looks weird" hours/days later | immediate |

This is the falsifiable form of the mission claim: a class of bug that
the typical nibabel+numpy code lets through is detected at the spatial
contract surface in the neuroim version.

### Read-time qualitative

A reader scanning the two implementations:

* **Baseline** — has to follow two manual `for i, (ix, iy, iz)` loops to
  reconstruct the mapping from mask voxels back to the 3-D output grid;
  must track which numpy axis order is being used at each step; must
  remember to compare `bold_img.affine` to `mask_img.affine` themselves,
  because nothing does it for them.
* **Rewrite** — reads top-down: assert the spaces match; pull a typed ROI;
  do the math on `extract.values`; map back through `extract.coords`. The
  spatial contract isn't asked-for — it's a precondition of the API.

Subjective read time on the actual code:
* baseline: ~3 minutes to convince yourself it's correct (the loops are
  the load-bearing part);
* rewrite: ~1 minute (the typed objects make the data-flow obvious).

## What still needs work

| Gap | Where |
|---|---|
| Memory-peak measurement on the file-backed lazy path. The current fixture is in-memory; the file-backed lazy story is asserted elsewhere but not in this audit. | A follow-up audit that loads via `read_image(path, lazy=True)` against a >1 GB synthetic fixture. |
| Provenance composition through the full pipeline. `extract.provenance` covers `series_roi`; a real pipeline that does `concat → series_roi → correlation → write` should chain provenance end-to-end so the final `NeuroVol.to_nibabel()` carries the full Receipt chain. ME-9 closed the `concat → series_roi → searchlight` chain; the `correlation → write` step is still receipt-less. | Follow-up: add a small `numeric_op` helper that wraps a callable and threads the upstream Receipt into a downstream one. |

### Resolved after first draft

| Gap | Resolution |
|---|---|
| `verify.assert_same_space` was too strict when one side was 4-D and the other 3-D — it hashed the full `dim` including time, and the rewrite had to work around it by calling `bold.space.compatible_with(mask.space)` directly. | Closed in `bd-01KRKN1QF93ARCKV32620SP02D`: `verify.assert_same_space` now routes through `NeuroSpace.compatible_with` when both inputs expose a concrete space, falling back to hash-strict comparison for Receipt-only inputs. The workaround is gone; `neuroim_rewrite.py` calls `ni.verify.assert_same_space(bold, mask)` directly. 5 new regression tests in `tests/test_verify.py` pin the 4-D-vs-3-D case. |

## What receipts protect (and what they do not)

Receipts are **input-contract guards**, not a universal correctness oracle.
The two test layers in this codebase have different load-bearing roles:

| Layer | Catches |
|---|---|
| **Receipts + verifier** (`Receipt`, `verify.assert_same_space`, `result.require_compatible`) | Silent **input-contract** mismatches: a mask in the wrong space, a data array whose affine disagrees with a downstream mask's affine, a provenance chain that disagrees on which input space produced a result. Mission rule 4 ('receipts by default') refers specifically to this class. |
| **Round-trip / parity tests** (`tests/test_sparse_neurovol_roundtrip.py`, the dtype-parity gates in WP-8/ME-7) | Silent **internal-computation** bugs: a constructor that scrambles its outputs (the SparseNeuroVol F/C-order bug surfaced by ME-7 was exactly this class), a backend that changes dtype, an output that doesn't bit-equal its input on a documented round-trip. |

Receipts cannot detect a constructor that scrambles its outputs because the
inputs were never in question — the scramble is an internal computation
defect, not a contract mismatch. The retro's claim that "receipts didn't
catch the SparseNeuroVol bug" is true but unfair: they were never the
right tool for that class. Both layers are load-bearing; neither
replaces the other.

## Verdict

The rewrite is **shorter (−31% body), more typed, provenance-bearing, and
catches a class of bug the baseline silently misses**.

That is the mission claim. The audit holds.

For the four ME-3 measurement targets that have hard numbers, the deltas
exceed the target. The bug-class assertion (#4) is the most consequential
and is the load-bearing evidence: the rewrite refuses to compute on a
mask whose space disagrees with the data; the baseline does not.

Run `pytest examples/draw_audit/test_audit.py -q` to reproduce.
