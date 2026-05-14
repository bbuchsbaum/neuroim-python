# Scenario 10 — Searchlight Space Mismatch

Task: compute a local mean searchlight map from a 4-D BOLD image and a
3-D analysis mask, then verify that the mask and BOLD are in the same
spatial frame before sampling voxel neighborhoods.

This scenario is intentionally adversarial. A careful raw nibabel user
can write the affine check before walking the mask. The neuroim rewrite
should make that check part of the typed searchlight contract: a mask
with the same shape but a shifted affine must not be allowed to drive
coordinate extraction from the BOLD grid.

Run:

```bash
PYTHONPATH=src:tests:. python -m pytest \
  examples/scenarios/test_s10_searchlight_space_mismatch.py -q
```

