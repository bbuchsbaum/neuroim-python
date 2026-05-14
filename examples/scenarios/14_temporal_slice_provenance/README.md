# Scenario 14 — Time-axis slicing as silent provenance loss

## Task

Drop the first four pre-steady-state volumes from a 4-D BOLD, then
compute a masked temporal-SNR map and write it to disk. A skeptical
collaborator who only has the resulting `.nii.gz` should be able to
tell whether the upstream analysis used the full series or a subset.

## Why it is adversarial

`bold[..., start:]` is the most common reshape idiom in 4-D fMRI Python
code (dropping pre-steady-state volumes, hold-out splits, run-by-run
audits). On `DenseNeuroVec` today it falls through to NumPy and returns
a bare `ndarray` — both the typed `NeuroSpace` and any upstream
`.provenance` are lost in one keystroke. A downstream `temporal_snr`
called on a re-typed wrapper does emit a `Receipt`, but it cannot
record the truncation upstream of itself.

So the **mission claim "Receipts by default — silent space/orientation
mismatches are caught at the contract layer, not in debugging"** is
true for the named ops in `OpParams` (`RoiOpParams`, `SearchlightParams`,
`TemporalReductionParams`, `ConcatParams`, `ResampleParams`) but
**empirically false for slicing**, the most-common reshape op.

## Expected falsifying observations

- **PAIN-13 (P1)** — `NeuroVec.__getitem__` with a time-axis slice key
  drops to bare `ndarray`. Type loss is the first failure; provenance
  loss is the second.

- **PAIN-14 (P2)** — even once slicing returns a `NeuroVec` again, the
  Receipt chain from `slice → temporal_snr` needs a `TemporalSliceParams`
  entry, or the recovered map's `method_name` is indistinguishable from
  one computed on the full BOLD. Same shape as PAIN-9 at a different
  surface.

## Run

```bash
PYTHONPATH=src:tests:. python -m pytest \
    examples/scenarios/test_s14_temporal_slice_provenance.py -q
```

Expected (current): 2 passed + 2 xfailed (strict) until PAIN-13 and
PAIN-14 land.
