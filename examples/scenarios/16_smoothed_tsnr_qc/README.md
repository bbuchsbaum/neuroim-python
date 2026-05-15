# Scenario 16 - Smoothed Temporal-SNR QC Map

Task: compute a masked temporal-SNR map from a 4-D BOLD run, smooth that QC
map at a requested FWHM in millimetres, and report simple summary values.

This is a small but realistic QC workflow. It crosses two operations that
users often chain by hand:

- a temporal reduction from 4-D BOLD to a 3-D map;
- a spatial smoothing step whose units must be millimetres, not voxels;
- a smoothing mask that must live in the same spatial frame as the map.

The scenario asks whether neuroim keeps this workflow readable while owning
the spatial contract checks and provenance mechanics that raw nibabel/numpy
code makes every caller reimplement.

