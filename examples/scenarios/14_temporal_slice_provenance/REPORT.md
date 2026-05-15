# Scenario 14 — Time-axis slicing as silent provenance loss: Report

> Drop the first four pre-steady-state volumes from a 4-D BOLD, then
> compute a masked temporal SNR map. The careful raw-`nibabel` user
> writes a sidecar manifest to record the truncation; the `neuroim`
> path should make the manifest unnecessary by carrying a Receipt
> through the slice into the derived map. Today, it does not.

## Verdict

**Falsifies "Receipts by default" at the slicing surface.** The
neuroim form is shorter and ships a typed `DenseNeuroVol` with a
`Receipt`, but two upstream gaps make that Receipt incomplete:

1. `bold[..., 4:]` returns a **bare `ndarray`** (PAIN-13). Both
   `NeuroSpace` and any prior `.provenance` are lost in one
   keystroke.
2. Even with a manual re-wrap to a `DenseNeuroVec`, the downstream
   `temporal_snr` Receipt's `method_name` is indistinguishable from
   one computed on the full series — there is no `TemporalSliceParams`
   in the chain. (PAIN-14).

The combination means a collaborator inspecting the written
`.nii.gz` and its embedded Receipt cannot answer "was this map
computed on the full series or a subset?" — exactly the audit
question Scenario 13 (provenance audit) shows neuroim succeeding at
for resample+tSNR. Slicing is the same shape of question at a
different surface.

## Scoreboard

| Axis | Baseline (raw nib + sidecar) | Neuroim today | Neuroim (after PAIN-13+14) |
|---|---:|---:|---:|
| User-facing function body | 8 | 1 (`bold[..., 4:].temporal_snr(mask=mask)`) | 1 |
| + helpers needed because of pain | 0 | ~10 (manual re-wrap to DenseNeuroVec) | 0 |
| Total scenario code | 8 | ~11 | 1 |
| Output type | `nib.Nifti1Image` | `DenseNeuroVol` w/ `Receipt` | `DenseNeuroVol` w/ chained `Receipt` |
| Records the truncation in provenance? | yes (sidecar) | **no** | yes (Receipt) |
| Survives `nib.save` + fresh-process read? | only if user wrote the sidecar | partially (downstream method, but not the slice) | yes (NIfTI ecode-6 extension) |

## What the API caught vs what it did not

| Case | Baseline | Neuroim today | Neuroim (after PAIN-13+14) |
|---|---|---|---|
| Mask shape/affine mismatch | manual `shape ==` + `np.allclose(affine)` | caught at the temporal_snr API | caught at the temporal_snr API |
| Truncation recorded in provenance | sidecar manifest field `temporal_slice_start` | **no field exists** | `TemporalSliceParams(start, stop, step)` |
| `slice → temporal_snr` recoverable from disk | only if sidecar | **no** | `method_name = "temporal_slice+temporal_snr"` |
| `bold[..., 4:]` preserves type | n/a | **falls through to numpy** | returns `DenseNeuroVec` |

## Pain points surfaced

### PAIN-13 (P1, fixed) — `NeuroVec.__getitem__` with a time-axis slice drops to ndarray

`DenseNeuroVec.__getitem__` had explicit branches for the 4-tuple and
the `(int, int, int)` cases; every other key fell through to
`self.data[key]` and returned the raw `ndarray`. For the very common
`bold[..., :N]` / `bold[..., N:]` / `bold[..., start:stop]` idiom,
this meant:

- the typed `NeuroSpace` was lost;
- any upstream `.provenance` was lost;
- the user had to manually rebuild a `DenseNeuroVec` on a derived
  4-D space to keep the rest of the pipeline typed.

**Fix.** A new branch in the existing `else` clause detects pure
time-axis selection (`result.ndim == 4` and `result.shape[:3] ==
self.data.shape[:3]`) and returns a `DenseNeuroVec` on a derived
4-D `NeuroSpace`. Non-pure-time keys (e.g. a tuple that slices a
spatial axis) keep the original fall-through-to-numpy behavior because
the result is no longer spatially well-defined.

**Tracker:** bd-01KRM193KP32NRAJRMC2Y13RM9 (closed).

### PAIN-14 (P2) — `TemporalSliceParams(OpParams)` missing from the catalogue

`src/neuroim/results.py:OpParams` has subclasses `RoiOpParams`,
`SearchlightParams`, `TemporalReductionParams`, `ConcatParams`,
`ResampleParams`. There is no `TemporalSliceParams`. Without it, the
`slice → temporal_snr` chain cannot be expressed structurally:
`receipt_for(carrier, params=TemporalSliceParams(...), upstream=...)`
has no `params` class to pass.

**Suggested fix.** Add `TemporalSliceParams(OpParams)` with
`start: Optional[int]`, `stop: Optional[int]`, `step: Optional[int]`.
Once PAIN-13 lands, the slicing path attaches
`receipt_for(new_space, params=TemporalSliceParams(start, stop,
step), upstream=self)`; the existing `+`-chaining in `receipt_for`
makes the downstream `temporal_snr` Receipt's `method_name` say
`"temporal_slice+temporal_snr"`, exactly mirroring the resample chain
that S09 already proves works.

**Tracker:** to be filed (chained to PAIN-13).

## How to reproduce

```bash
PYTHONPATH=src:tests:. .venv/bin/python -m pytest \
    examples/scenarios/test_s14_temporal_slice_provenance.py -q
```

Expected (current): **2 passed + 2 xfailed (strict)**. The xfailed
tests are the two falsifying acceptance checks for PAIN-13 and
PAIN-14; both flip to plain assertions once the fix lands.

## Follow-ups

| Pain | Suggested issue title | Priority |
|---|---|---|
| PAIN-13 | `NeuroVec.__getitem__` time-axis slice must return typed `NeuroVec` | P1 |
| PAIN-14 | Add `TemporalSliceParams(OpParams)` and chain via `receipt_for` | P2 |
