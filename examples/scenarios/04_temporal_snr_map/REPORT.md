# Scenario 04 -- Temporal SNR Map: Report

> Compare computing a masked 3-D temporal SNR map from a 4-D BOLD image in
> raw `nibabel`+`numpy` against the neuroim public API.

## Verdict

**Clean win on every axis (after S04 PAIN-1/PAIN-2 landed).** The
rewrite is now `return bold.temporal_snr(mask=mask)` — one statement.
The spatial contract is checked at the API surface, the reduction is
named, the returned `DenseNeuroVol` rides on the vector's spatial
subspace, and it carries a populated `Receipt` in
`DenseNeuroVol.provenance`. With S05/PAIN-6 also landed (Receipts
embedded in NIfTI ecode-6 extensions on `to_nibabel` / `write_vol`),
the provenance survives a clean-process round-trip through disk.

## Scoreboard

Counts are AST function-body statements, excluding docstrings.

| Axis | Baseline | Rewrite |
|---|---:|---:|
| User-facing function body | 12 | **1** |
| + helpers needed | 0 | 0 |
| **Total scenario code** | **12** | **1** |
| Output type | `nib.Nifti1Image` | `DenseNeuroVol` (+ `.provenance`) |
| Mask-shape mismatch | catches by hand | catches via verifier/spatial contract |
| Mask-affine mismatch | catches by hand (`np.allclose`) | catches via verifier — `"spatial contract mismatch"` |
| Zero-variance voxels | zero-filled | zero-filled |
| Provenance (in-memory) | none | `Receipt(method_name="temporal_snr", mask_hash=…, input_space_hash=…)` |
| Provenance (round-tripped to disk) | none | embedded as NIfTI comment extension via `to_nibabel`/`write_vol` (S05/PAIN-6) |

## What the API Caught

The mask/data compatibility gate carried through to the temporal
reduction path: `bold.temporal_snr(mask=...)` invokes
`verify.assert_same_space` first, so an LR-flipped mask fails before
any numeric work begins. The same `"spatial contract mismatch"` error
shape that the ROI scenarios use applies here.

## Pain Points Surfaced and Fixed

### PAIN-1 (P2, fixed) — `NeuroVec.temporal_snr(mask=...)` is a first-class API

Tracker: `bd-01KRKQ7MXV0SFMZG7F00Q8E72Q` (closed).

The temporal reduction is no longer performed by hand. A new
concrete method on `NeuroVec` validates the mask space, materializes
the dense data once, reduces along time, applies the mask, zero-fills
zero-variance voxels, and wraps the result in a `DenseNeuroVol` on
the vector's `spatial_space`.

### PAIN-2 (P2, fixed) — Temporal-reduction maps carry a `Receipt`

Tracker: `bd-01KRKQ7N1JBPJYXAKGRSZY4MTS` (closed).

`DenseNeuroVol` gained an optional keyword-only `provenance` field.
`temporal_snr` populates it with a `Receipt` whose `method_name` is
`"temporal_snr"`, whose `input_space_hash` is the 3-D spatial space's
hash, and whose `mask_hash` is the supplied mask's content hash. With
S05/PAIN-6 also landed, that Receipt persists through
`to_nibabel`/`write_vol`/`read_image` round-trips.

## How to Reproduce

```bash
PYTHONPATH=src:tests:. .venv/bin/python -m pytest \
    examples/scenarios/test_s04_temporal_snr_map.py -q
```

Expect: 6 passed. (Was 4 passed + 2 xfailed before PAIN-1/PAIN-2
landed; the xfails flipped to plain assertions.)

## What this scenario does *not* test

- Generic temporal reducers (`reduce_time(func, ...)`). The mission
  rule for new API surface is scenario-driven; a generic reducer
  earns its place when a scenario shows the win. `temporal_snr` is
  the canonical case.
- File-backed / lazy reads. The fixture is in-memory; a
  `read_image(path, lazy=True)` variant is a candidate for a later
  scenario.

## Follow-ups

| Item | Suggested issue title | Priority |
|---|---|---|
| Generic temporal reducer | `NeuroVec.reduce_time(func, mask=..., method_name=...)` — earn through a future scenario that needs it | P3 |
