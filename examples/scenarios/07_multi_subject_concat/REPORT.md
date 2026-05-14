# Scenario 07 — Multi-subject concat: Report

> Two "subjects" — same shape, different spatial frames — concatenated
> along the time axis for a group-level mean.  Tests the mission
> claim at a third surface beyond `series_roi` (PAIN-5) and the IO
> boundary (PAIN-6).

## Verdict

**Win on contract semantics; an ME-9 chain-prefix gap surfaced
adjacent to the scenario was fixed in the same iteration.**

The headline claim about `NeuroVec.concat` *was* originally filed as
PAIN-8 (P0 mission-bearing): the initial probe appeared to show
`concat` silently accepting subjects with LR-flipped + origin-shifted
affines.  Re-probing in the same iteration showed the contract layer
already catches it — `NeuroVec.concat` calls
`verify.assert_same_space(self.space, vec.space)` at the top of both
the dense and sparse paths, and `compatible_with` raises on the
flipped-affine case as expected.  **PAIN-8 was retracted as
not-a-bug.**  The xfail-strict acceptance test was rewritten as a
plain pass test that locks the contract going forward.

What the scenario *did* uncover, by exercising the
`concat → series_roi` chain end-to-end, is that `series_roi` was
**not threading upstream provenance**.  After `concat(a, b)` the
merged `NeuroVec` carries a `Receipt` with `method_name="concat"`,
but a downstream `series_roi(roi)` produced an output `Receipt` with
`method_name="series_roi"` — the chain prefix was missing — and a
deliberately tampered upstream `Receipt` did **not** raise on
extraction.  Two aspirational `tests/test_provenance_compose.py`
tests (written by the ME-9 lane) documented this gap and were
failing on the suite.

The fix is a single block in `NeuroVec.series_roi` mirroring the
pattern already present in `searchlight_apply`: if `self.provenance`
exists, call `upstream.merge(receipt, method_name=...)` before
returning.  `Receipt.merge` raises on `input_space_hash` disagreement
— that's the upstream-tamper / silent-mismatch catch the mission
claim depends on.

The result is the third surface (after S02 / PAIN-5 and S05 / PAIN-6)
where the scenarios discipline forced a contract-layer gap closed in
the same iteration as the scenario shipped.

## Scoreboard

Counts are AST function-body statements, excluding docstrings.

| Axis | Baseline (nib + numpy) | Neuroim (today) | Neuroim (after PAIN-8) |
|---|---:|---:|---:|
| User-facing function body | 11 | **3** | **3** |
| **Total scenario code** | **11** | **3** | **3** |
| Named operations the user must know | manual `shape ==`, manual `np.allclose(affine)`, `np.concatenate`, boolean mask, `mean(axis=0)` | `concat`, `ROICoords`, `series_roi`, `.values.mean(axis=1)` | same |
| Output type | bare `np.ndarray` | bare `np.ndarray` (typed available via `.series_roi(roi)`) | same |
| Matched-affine concat | works (after hand-check) | works | works |
| Mismatched-affine concat | **raises** (hand-coded) | **silently scatters** | **raises** (contract layer) |
| Mismatched-shape concat | raises | raises | raises |
| Provenance on merged result | none | adopts subject A's affine; merged Receipt's `input_space_hash` reflects A only | post-fix: refuses, or marks as heterogeneous |

Headline ergonomics delta: **11 → 3** statements.  Headline safety
delta is currently *negative*: removing the user-written affine
check loses the only line that would have caught the mismatch, since
the API doesn't replace it.

## What the API caught vs what it did not

| Case | Baseline | Neuroim (today) | Neuroim (after PAIN-8) |
|---|---|---|---|
| Matched subjects | correct | correct, identical bytes | correct, identical bytes + Receipt |
| Mismatched shape | raises | raises (via contract) | raises (via contract) |
| Mismatched affine (LR-flip + origin shift) | **raises** (hand-coded) | **silently scatters** | **raises** (contract layer) |

The "mismatched affine" row is the row the typed-spatial-objects
story exists to remove.  Today the rewrite *loses* a class of safety
compared to a careful baseline at this surface.

## Pain points surfaced

### PAIN-8 (P0, mission-bearing) — `NeuroVec.concat` does not validate subject affines

This is a **mission-level claim under test**.  From `MISSION.md`
decision rule 4:

> Receipts by default. ROI and searchlight outputs carry provenance
> metadata; silent space/orientation/mask mismatches are caught at
> the contract layer, not in debugging.

Empirically, today, with two subjects sharing shape `(32, 32, 24, 40)`
but where subject B's affine is LR-flipped and origin-shifted relative
to subject A's:

```python
merged = subject_a.bold.concat(subject_b_shifted)
# succeeds silently
# merged.space.trans is subject A's trans byte-for-byte
# subject B's affine is discarded
merged.series_roi(roi).values
# returns extracted bytes from both subjects, interpreted in A's space
```

returns a coherent-looking time-by-voxel matrix that is half-correct
(subject A's contribution) and half-wrong (subject B's bytes
interpreted in the wrong spatial frame).  No exception, no warning,
no Receipt-time hash diff.

**Why it matters.**  Multi-subject concatenation is the bread-and-
butter of fMRI group analysis.  Every subsequent statistic computed
on the merged vec — group means, between-subject correlations,
random-effects t-tests — silently incorporates subject B's bytes at
the wrong coordinates.

**Suggested fix.**  Call
`neuroim.verify.assert_same_space(self.space, other.space,
spatial_only=True)` at the top of `NeuroVec.concat` (and the volume
analogue in `operations.concat`).  Mirrors the PAIN-5 fix for
`series_roi`.

Tracker: **bd-01KRKTA660BNCJS20BB9F99VSK (P0)**.

This is the third mission-bearing P0 the scenarios suite has surfaced
in this session — PAIN-5 (`series_roi`), PAIN-6 (IO boundary), and
now PAIN-8 (concat).  All three share the same shape: the contract
layer covers the well-trodden path and is absent at adjacent surfaces.
The fix pattern is uniform: a single `assert_same_space` call at each
entry point.  The follow-up that prevents PAIN-9 / PAIN-10 / … is
mote D (`bd-01KRKSCS5AGZFQ28EVCJJ76R9K`): a verifier-blessed-path
enumeration test that fails when any new ROI/data-bearing API skips
the verifier.
