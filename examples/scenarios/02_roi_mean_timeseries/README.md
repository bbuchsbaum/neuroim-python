# Scenario 02 — ROI mean time series

Compute the mean BOLD time series across all voxels in an anatomical
mask.  One length-`nt` array out, one mask in, one 4-D BOLD in.

This is the workhorse first-pass operation in nearly every fMRI
pipeline — a default-mode-network mean, an ROI control regressor, a
seed-based correlation precursor.  It is also the cleanest test of
the *contract* claim: does the API force the user to validate that
the mask lives in the BOLD's space before extracting, or does it
silently scatter values that happen to fit?

The acceptance test lives one level up at
`scenarios/test_s02_roi_mean_timeseries.py`.  The report and pain
points are in `REPORT.md`.
