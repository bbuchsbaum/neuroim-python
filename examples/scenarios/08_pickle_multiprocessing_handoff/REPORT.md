# Scenario 08 — Pickle / multiprocessing handoff: Report

> Compute a masked temporal-SNR map, serialize it through a
> joblib/multiprocessing-style pickle payload, and inspect the result in a
> fresh Python process.

## Verdict

**Win on inspectability; neutral on numeric work.** Raw nibabel can pickle a
derived image's data and affine, but the receiving process cannot tell what
method produced it or which mask/input space were checked unless the caller
hand-bundles a manifest. neuroim's typed temporal-SNR map preserves both
`space` and `Receipt` through the same pickle boundary.

Unlike S02, S05, and S07, this scenario did not surface a mission-bearing P0.
It instead confirms that the current typed-result design creates value at a
common worker/cache boundary.

## Scoreboard

Counts are AST function-body statements for the producer plus the receiving
process inspection helper.

| Axis | Baseline (bare nibabel) | Baseline (careful manifest) | Neuroim |
|---|---:|---:|---:|
| Produce derived map | 10 | 10 + 7 manifest fields | **1** |
| Pickle data + affine | yes | yes | yes |
| Method name survives | no | yes, if hand-written | **yes, computed** |
| Input-space hash survives | no | yes, if hand-written | **yes, computed** |
| Mask hash survives | no | yes, if hand-written | **yes, computed** |
| Receiving process has typed spatial object | no | no | **yes** |

The win is not that pickle itself is hard. The win is that the provenance is
computed by the producing API and travels with the typed result rather than
being a parallel hand-maintained dictionary.

## What the API caught vs what it did not

| Case | Baseline | Neuroim |
|---|---|---|
| Data and affine survive pickle | yes | yes |
| Derived-map provenance survives bare pickle | no | yes |
| User has to keep manifest in sync | yes, for provenance | no |
| Fresh process can inspect method/input/mask | only with manual bundle | yes |

## Pain points surfaced

None. This is a positive-control scenario for the current typed-result and
Receipt design.

## Follow-ups

- Keep this scenario in the 0.3 evidence set. It covers the pickle/joblib
  boundary that sits between in-memory composition (ME-9) and NIfTI
  serialization (S05).
- If future result types add custom non-picklable state, this scenario should
  fail before that regression reaches users.

