# Scenario 09 — Native-to-template provenance: Report

> Resample a native-space BOLD image to a template, then compute a masked
> temporal-SNR map. The scenario asks whether provenance records both the
> normalization step and the derived-map reduction.

## Verdict

**Originally falsified inspectability for a careful-user workflow; now fixed
and guarded.** neuroim wins on workflow brevity and still checks the template
mask against the resampled BOLD. When this scenario landed, the output
`Receipt` recorded only `method_name == "temporal_snr"`: the native-to-template
resample step was invisible in provenance, and the native source space was not
recoverable from the final result.

PAIN-9 is now fixed: `resample_vec` attaches a provenance receipt that records
the source space hash, target space hash, and interpolation order in the method
path, and `temporal_snr` merges compatible upstream receipts. The acceptance
test now runs without xfail.

This is not a silent numeric scatter like PAIN-5 or the original S07
hypothesis, because the resample does place data into the template grid. It is
a provenance/inspectability bug: the final typed map looks as if it originated
directly in template space.

## Scoreboard

| Axis | Baseline (nib + manifest) | Neuroim today | Neuroim after PAIN-9 |
|---|---:|---:|---:|
| User-facing function body | 3 operations + manifest | **2** | **2** |
| Mask/template spatial check | manual | yes, via `temporal_snr(mask=...)` | yes |
| Resample operation in provenance | manual manifest | recorded | recorded |
| Native source space recoverable | manual manifest | yes, as source hash in method path | yes |
| Output type | bare `Nifti1Image` | typed `DenseNeuroVol` | typed `DenseNeuroVol` |

## Pain points surfaced

### PAIN-9 (P1, fixed) — `resample_vec` provenance was dropped before derived-map receipt

`resample_vec(native_bold, template_bold.space)` returns a `DenseNeuroVec`
whose `.provenance` is `None`. A subsequent
`resampled.temporal_snr(mask=template_mask)` creates a fresh `Receipt` with
`method_name == "temporal_snr"` and an input-space hash for the template-space
resampled vector. The receipt does not record that the data originated in a
different native space or that interpolation/order=1 was applied.

**Why it matters.** Native-to-template normalization is a standard
multi-subject workflow. A collaborator inspecting only the derived map cannot
tell whether it was computed from native data, template data, or data resampled
with a different interpolation policy.

**Suggested fix.** `resample` / `resample_vec` should attach a `Receipt` (or
use the structural provenance builder) that records source space, target space,
interpolation/order, and method name. Downstream derived-map operations should
merge that upstream receipt so the final method path reads like
`resample_vec+temporal_snr` or equivalent structured provenance.

Tracker: **bd-01KRKV5ZW7YNNQNNHCR2969F84 (P1)**.

**Resolution.** `resample` / `resample_vec` now attach receipts named like
`resample_vec(order=1,source=<hash>,target=<hash>)`. `temporal_snr` merges a
compatible upstream receipt, so the final derived map records
`resample_vec(...)+temporal_snr`.
