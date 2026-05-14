# Scenario 09 — Native-to-Template Provenance

Task: resample a subject-native 4-D BOLD image into a template space, compute
a masked 3-D temporal-SNR map there, and inspect whether the output provenance
records the normalization/resampling step.

This is intentionally adversarial. A careful raw nibabel user writes an
explicit manifest with both source and target spaces. The neuroim rewrite is
shorter, but the current provenance path loses the resample step: the final
`Receipt` says only `temporal_snr` on the template-space result.

Run:

```bash
PYTHONPATH=src:tests:. python -m pytest \
  examples/scenarios/test_s09_native_to_template_provenance.py -q
```

