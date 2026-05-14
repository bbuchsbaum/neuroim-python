# Scenario 07 — Multi-subject concat with space heterogeneity

Two "subjects" — same shape, different spatial frames (one in a
shifted/LR-flipped affine relative to the other) — concatenated
along the time axis for a group-level mean.

This is the simplest realistic shape of the *multi-subject space-
alignment* hazard that fMRI pipelines hit every day.  Subject A is
in MNI152; subject B was reconstructed in MNI152NLin2009cAsym; the
user `concat`s them and trusts that anything downstream stays
coherent.

The mission claim under test: *"silent space/orientation/mask
mismatches are caught at the contract layer."*  Today the test
falsifies it at a third surface (after PAIN-5 / `series_roi` and
PAIN-6 / IO boundary): `NeuroVec.concat()` silently adopts subject
A's affine and discards subject B's.

The acceptance test lives one level up at
`scenarios/test_s07_multi_subject_concat.py`.  The verdict and pain
points are in `REPORT.md`.
