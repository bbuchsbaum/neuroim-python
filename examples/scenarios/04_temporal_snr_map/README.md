# Scenario 04 -- Temporal SNR Map

Task: compute a 3-D temporal signal-to-noise ratio (tSNR) map from a
4-D BOLD image, restricted to a brain mask.

This is a canonical fMRI QC workflow:

1. load or hold a 4-D BOLD image,
2. compute `mean(time) / std(time)` per voxel,
3. keep values only inside a brain mask,
4. return a spatially aligned 3-D image/map.

The scenario is intentionally simple. A competent nibabel user can write
it in a few lines, so neuroim only wins if it provides a clearer spatial
container, mask contract, and provenance story.

## Expected Stress Points

- Does neuroim expose a first-class temporal-reduction method, or does a
  user still reach into `.data` and reconstruct a `DenseNeuroVol` by hand?
- Does the output map carry a receipt explaining the reduction and mask?
- Does mask/data space validation happen before reduction?

Run:

```bash
PYTHONPATH=src:tests:. .venv/bin/python -m pytest \
    examples/scenarios/test_s04_temporal_snr_map.py -q
```
