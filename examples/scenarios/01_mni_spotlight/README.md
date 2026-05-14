# Scenario 01 — MNI Spotlight

> Given a 4-D BOLD image and a world-mm coordinate `(x, y, z)`, return
> the BOLD time series at the voxel **nearest** that world coordinate.

This is the absolute "hello world" of seed-based fMRI analysis. It is
the smallest task that still touches the single most error-prone step
in hand-written nibabel code: the **world↔voxel affine transform**.

## Inputs

- `bold` — 4-D BOLD image (`nibabel.Nifti1Image` in the baseline,
  `neuroim.NeuroVec` in the rewrite).
- `mni_xyz` — world coordinate in mm, as `(x, y, z)`.

## Output

- A 1-D time series of length `T` (the temporal dimension of `bold`).

## Behaviour at the edges

| Case | Required behaviour |
|---|---|
| World coord maps to an in-bounds voxel | Return that voxel's time series. |
| World coord maps to an out-of-bounds voxel | Raise `ValueError` with the bad mapping spelled out. |
| `bold` is not 4-D | Raise `ValueError`. |

Out-of-bounds detection is the safety bar both implementations are
held to. The baseline must hand-code it; the rewrite must either get
it from the API or hand-code it (we are honest about which).

## Fixture

`tests/fixtures/realistic_bold.py::make_realistic_bold()`. The known
signal lives at voxel `(16, 22, 16)` ≡ world `(48.0, 66.0, 56.0)` mm.

## Run

```bash
PYTHONPATH=src:tests:. .venv/bin/python -m pytest \
    examples/scenarios/test_s01_mni_spotlight.py -q
```

The runnable test lives one level up because folder names beginning
with a digit are illegal Python module names. See
[`../README.md`](../README.md) for the layout note.

See [`REPORT.md`](REPORT.md) for the verdict and surfaced pain points.
