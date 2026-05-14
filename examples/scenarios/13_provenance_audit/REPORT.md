# Scenario 13 — Pipeline Provenance Audit

## Verdict

**Win on forensic inspectability.**  Raw nibabel can write the same derived
map, but the final NIfTI cannot answer how it was produced without a separate
manifest.  Neuroim embeds a chained Receipt in the NIfTI extension, so a fresh
reader can recover the resample + temporal-SNR lineage from the file alone.

| Axis | Baseline nibabel | neuroim |
|---|---|---|
| Numeric output | resample then temporal SNR | same map |
| Method lineage from final file | unavailable | `resample_vec(...)+temporal_snr` |
| Mask identity from final file | unavailable | recovered `mask_hash` |
| Producer version from final file | unavailable | recovered `neuroim_version` |
| Extra sidecar required | yes, if user writes one | no |

## Why This Matters

This is the careful-user case: the producer did not obviously make a mistake,
and both stacks can compute the map.  The difference appears at handoff time.
A collaborator can inspect a neuroim-produced `.nii.gz` and recover the
pipeline receipt without access to the notebook that produced it.

## Gate

```bash
PYTHONPATH=src:tests:. pytest examples/scenarios/test_s13_provenance_audit.py -q
```

Expected result: three passing tests covering numeric parity, raw-nibabel audit
failure, and neuroim chained-provenance recovery from disk.
