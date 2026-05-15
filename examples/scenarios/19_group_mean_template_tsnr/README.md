# Scenario 19 — Group-mean tSNR across subjects in template space

## Task

Two synthetic "subjects" are acquired on different native grids.  For each
subject:

1. Resample the native 4-D BOLD into a common template grid.
2. Compute a masked temporal-SNR map in template space.

Then *across subjects*:

3. Average the per-subject tSNR maps into a single group-level tSNR map.

The group map is the canonical second-level QC artifact: every published
multi-subject fMRI paper has a figure of this shape.  The scenario asks
whether a careful raw-`nibabel` user and the `neuroim` public API answer
the same five questions about that map:

| Question | Why it matters |
|---|---|
| Numeric parity with raw `nibabel + numpy + scipy` | The math has to be exactly the same. |
| Same-space gate at the group reduce | A subject with a foreign affine but the *same shape* as the others must be rejected before averaging — the canonical silent-bug shape. |
| Multi-input provenance | The terminal map's Receipt should reflect *both* upstream chains (`resample_vec(...)+temporal_snr`), not just one of them. |
| Round-trip survival | The Receipt must survive `write_vol` + `read_image`, so a collaborator inspecting the saved `.nii.gz` can recover the second-level lineage. |
| Single-call ergonomics | The group reduce should be one named operation, not a hand-rolled stack-mean-wrap loop. |

## Why this scenario exists

S07 (multi-subject concat) pinned the same-space gate on a *4-D* concat.
S09 (native-to-template provenance) pinned the chained Receipt for a
*single subject's* `resample_vec → temporal_snr` derived map.  S15–S18
all stop at single-subject derived maps.

S19 is the first scenario whose **input is N typed derived maps** and
whose **output is one typed derived map with N upstream Receipts**.  It
is the natural climax of the suite: every prior scenario produced one
subject's derived artifact; S19 asks whether the API has a first-class
shape for *combining* those artifacts with provenance intact.

## Layout

```
19_group_mean_template_tsnr/
├── README.md              # this file
├── REPORT.md              # scoreboard, pain points, verdict
├── baseline_nibabel.py    # raw nibabel + numpy + scipy form
└── neuroim_version.py     # neuroim public-API form
```

The runnable acceptance test lives at
`tests/test_s19_group_mean_template_tsnr.py`.

## Running

```bash
PYTHONPATH=src:tests:. .venv/bin/python -m pytest \
    tests/test_s19_group_mean_template_tsnr.py -q
```
