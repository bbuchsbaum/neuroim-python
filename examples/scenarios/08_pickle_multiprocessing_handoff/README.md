# Scenario 08 — Pickle / Multiprocessing Handoff

Task: compute a masked temporal-SNR map, serialize it as a
joblib/multiprocessing-style pickle payload, and inspect it in a fresh Python
process.

Use this when a workflow farms derived maps out to workers or caches
intermediate results with pickle/joblib. The question is not whether bytes
survive pickling; both nibabel and neuroim can do that. The question is whether
the receiving process can still inspect the typed spatial frame and provenance
without a hand-maintained manifest.

The scenario compares:

- raw `nibabel` + `numpy`, where a bare `Nifti1Image` preserves data and affine
  but carries no method/mask/input-space provenance unless the user manually
  bundles a manifest;
- `neuroim`, where a `DenseNeuroVol` from `temporal_snr(mask=...)` carries
  `.space` and `.provenance` through the pickle boundary.

Run:

```bash
PYTHONPATH=src:tests:. python -m pytest \
  examples/scenarios/test_s08_pickle_multiprocessing_handoff.py -q
```

