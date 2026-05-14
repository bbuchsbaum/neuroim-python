# Scenario 13 — Pipeline Provenance Audit

> *Forensic recovery: a collaborator hands you a derived 3-D NIfTI and
> walks away.  Without their pipeline code, recover the full history
> from the file alone.*

## The task

You inherit `tsnr_template.nii.gz`.  Answer five questions about it
**from the file on disk only** — no Python notebook, no sidecar
JSON, no Slack thread with the producer:

1. What method produced this map?
2. From what input image's spatial frame?
3. With which mask?
4. What pipeline parameters (resample interpolation order, mask hash, ...)?
5. What library version produced it?

The producer's pipeline was a *chained derived-map workflow* —
the kind that is unaudited by default in every imaging library and
unrecoverable from the output:

```text
native BOLD  -->  resample_vec(target=template_space, order=1)
             -->  temporal_snr(mask=template_mask)
             -->  to_nibabel  -->  write to disk
```

Each stage modifies both the data and its spatial frame.  If the
producer botched any one step (wrong template, wrong mask, wrong
interpolation), the audit is the only way to catch it without rerunning.

## Why this scenario exists

MISSION decision rule 4: *Structural provenance by default. ROI,
searchlight, and derived-map outputs carry provenance metadata;
silent space/orientation/mask mismatches are caught at the contract
layer, not in debugging.*

S05 proved the round-trip for a single-stage derived map.  S09
proved that `resample_vec` records a `ResampleParams` Receipt.  S13
closes the chain: a **multi-stage pipeline** whose final NIfTI
carries the full chained `method_name`, the *terminal* spatial frame,
and every operation's parameters — recoverable from the file alone.

This is the test the mission claim cannot be partial on.  If the
chain breaks anywhere, the audit fails.

## Compared to

- raw `nibabel`+`numpy` (the canonical baseline);
- `nilearn` is a peer here, but at the audit stage every imaging
  library outside neuroim hands the user the same answer: the file
  itself has no policy metadata — read the producer's code.

## Files

- `baseline_nibabel.py`  — runs the same numeric pipeline in raw
  nibabel + numpy + scipy and saves the result.  At audit time,
  returns an empty manifest.
- `neuroim_version.py`   — runs the pipeline through neuroim public
  API.  At audit time, returns a populated manifest with the
  chained method name and the parameters of every stage.
- `../test_s13_provenance_audit.py`  — runnable acceptance test:
  numeric parity on the data, audit-success on the neuroim side,
  audit-failure on the raw-nibabel side.
- `REPORT.md`            — scoreboard and verdict.
