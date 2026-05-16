# Contract-failure vocabulary — v1

This document specifies the closed set of *canonical refusals* the
`neuroim` public API raises when a spatial, indexing, or provenance
contract is violated. It is a public contract: tests, downstream
tools, and third-party backends may match on the message substrings
and exception classes defined here.

The spec is intentionally narrow. It covers the *contract* failures
the library guarantees — the ones a wrong-but-plausible silent return
would otherwise mask. Ordinary bugs (`KeyError`, NumPy `ValueError`
from a bad reshape, etc.) are out of scope.

---

## Layering model

Spatial-mismatch refusals are produced by a three-level stack. Each
level prepends a prefix and re-raises; the **leaf message is always
present in the final string**, which is the property that makes it
greppable across every operation.

```
NeuroSpace.compatible_with         leaf:  "NeuroSpace mismatch in {spatial dim|affine}: left=…, right=…"
        │  (raises ValueError)
        ▼
verify.assert_same_space           wrap:  "assert_same_space: spatial contract mismatch — {leaf}"
        │  (re-raises ValueError)
        ▼
operations.concat                  wrap:  "spatial contract mismatch: NeuroVecs must have same
                                            spatial dimensions and affine; {assert_same_space msg}"
```

A conforming reader that wants "did a spatial contract fail?" matches
the substring `spatial contract mismatch` **or** the leaf
`NeuroSpace mismatch in`. A reader that wants "which operation
raised?" matches the outermost prefix.

---

## The vocabulary

All strings below are verbatim in the current source. `{…}` marks a
runtime-interpolated field; everything outside `{…}` is literal.

### Spatial-frame refusals

| ID | Exception | Message | Raised by |
|----|-----------|---------|-----------|
| `SPACE-DIM` | `ValueError` | `NeuroSpace mismatch in spatial dim: left={tuple}, right={tuple}` | `NeuroSpace.compatible_with` |
| `SPACE-AFFINE` | `ValueError` | `NeuroSpace mismatch in affine: left={list}, right={list}` | `NeuroSpace.compatible_with` |
| `SPACE-ASSERT` | `ValueError` | `assert_same_space: spatial contract mismatch — {SPACE-DIM\|SPACE-AFFINE}` | `verify.assert_same_space` (structured-space path) |
| `SPACE-HASH` | `ValueError` | `assert_same_space: input_space_hash mismatch` + `  a.input_space_hash = {repr}` + `  b.input_space_hash = {repr}` | `verify.assert_same_space` (receipt-only path) |
| `MASK-HASH` | `ValueError` | `assert_same_mask: mask_hash mismatch` + `  a.mask_hash = {repr}` + `  b.mask_hash = {repr}` | `verify.assert_same_mask` |
| `CONCAT-SPACE` | `ValueError` | `spatial contract mismatch: NeuroVecs must have same spatial dimensions and affine; {SPACE-ASSERT}` | `operations.concat` |

`SPACE-ASSERT` is the message reached by `series_roi`,
`searchlight_apply`, `temporal_snr(mask=…)`, and every other operation
that takes a spatial `mask=`/`roi` argument: they all route through
`verify.assert_same_space` on the structured-space path.

### Off-grid refusals

| ID | Exception | Message | Raised by |
|----|-----------|---------|-----------|
| `IDX-OOB` | `IndexError` | `{coordinate\|index} out of bounds for spatial shape {tuple}: {preview}{… (+N more)}` | `series_at`, `series_at_coords`, `series_at_indices` |
| `WORLD-OOB` | `ValueError` | `world coord {tuple} mm maps to voxel {tuple} which is outside the image grid of shape {tuple}.` | `series_at_world`, `series_roi_world` |

`IDX-OOB` and `WORLD-OOB` are deliberately different exception types.
The voxel/index path is reporting an *indexing* error (`IndexError`);
the world-coordinate path has already performed the affine inversion
and is reporting a *domain* error (`ValueError`). The
`out_of_bounds="zero"` opt-in suppresses both in favour of zero-fill;
the default is `"raise"`.

### Backend refusals

| ID | Exception | Message | Raised by |
|----|-----------|---------|-----------|
| `FB-DRIFT` | `ValueError` | `Volume {idx} has inconsistent affine/space: {underlying}` | `FileBackedNeuroVec` (per-volume load) |

### Immutability refusals

| ID | Exception | Message | Raised by |
|----|-----------|---------|-----------|
| `FROZEN-SPACE` | `AttributeError` | `NeuroSpace is immutable` | `NeuroSpace.__setattr__` after construction |

### Construction refusals

| ID | Exception | Message | Raised by |
|----|-----------|---------|-----------|
| `DIM-POS` | `ValueError` | `All dimensions must be positive` | `NeuroSpace.__init__` |
| `SPACING-POS` | `ValueError` | `Spacing values must be positive` | `NeuroSpace.__init__` |
| `TRANS-3D` | `ValueError` | `trans must be 4x4 matrix for {ndim}D space, got {shape}` | `NeuroSpace.__init__` (ndim ≤ 3) |
| `TRANS-ND` | `ValueError` | `trans must be {n}x{n} matrix for {ndim}D space, got {shape}` | `NeuroSpace.__init__` (ndim > 3) |
| `TRANS-SINGULAR` | `ValueError` | `Transformation matrix must be invertible` | `NeuroSpace.__init__` |
| `OOB-MODE` | `ValueError` | `out_of_bounds must be 'raise' or 'zero'` | `NeuroVec.series_at*` argument validation |

---

## Conformance

A third-party `VoxelSeriesStore` backend (see
[the voxel-series-store protocol](voxel-series-store-protocol.md)) is
**conforming** only if, on the canonical bad input for each applicable
row, it raises the listed exception class and its message contains the
listed literal substring (with `{…}` fields free to vary).

The conformance test for this spec is
`tests/test_contract_failure_vocabulary.py` (added with this spec):
for every operation in the tables above it constructs the canonical
bad input and asserts `pytest.raises(<class>, match=<literal>)`. Any
change that alters a message's literal portion is a spec-breaking
change and bumps this document to v2.

---

## Why this spec exists

"It raises on a bad mask" is only a guarantee if *which* error, with
*what* message, is pinned. Without that, a refactor can silently
downgrade a `ValueError` to a `False` return or change the message so
that downstream `match=` assertions and log scrapers stop firing — and
the mission claim ("silent space/orientation/mask mismatches are
caught at the contract layer") quietly stops being testable. This
vocabulary is what makes the refusal a contract instead of a habit.

See the [Contract failures concept page](../concepts/contract-failures.qmd)
for the narrative version, and `examples/scenarios/` for the runnable
scenarios each refusal was filed against.
