# Scenario 12 — File-Backed Affine Drift

Task: load a 4-D BOLD run stored as one 3-D NIfTI file per time point,
compute a masked temporal-SNR map, and refuse the run if any time point
has drifted into a different spatial frame.

This scenario is intentionally adversarial. File-backed split-volume data is
common in pipelines that cache intermediate volumes or stream time points
from disk. A careful raw nibabel user checks every volume's affine before
stacking. The neuroim file-backed vector should make that safety check part
of the backend contract, not silently inherit the first volume's space.

Run:

```bash
PYTHONPATH=src:tests:. python -m pytest \
  examples/scenarios/test_s12_file_backed_affine_drift.py -q
```
