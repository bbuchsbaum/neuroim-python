# Scenario 15 - Block Parcel Contrast

Task: given a 4-D BOLD run, an integer-labelled atlas, and a binary
task/rest design, compute a task-minus-rest effect per parcel and map the
parcel effects back into a 3-D image.

This is deliberately small, but it is analysis-shaped:

- the BOLD run and atlas must be in the same spatial frame;
- the design vector must match the time axis;
- the output should be interpretable as labelled parcel effects, not just a
  bare array;
- the strongest parcel should recover the synthetic target ROI embedded in
  the fixture.

The scenario asks whether the neuroim path reads like the analysis we meant
to write. The raw nibabel/numpy baseline is numerically equivalent on the
happy path, but it has to manually unpack arrays, discover labels, map parcel
values back to voxels, and remember to enforce spatial contracts.

