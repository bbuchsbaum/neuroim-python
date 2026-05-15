# Scenario 17 - Local Block-Effect Searchlight

Task: compute a spherical searchlight map where each in-mask center receives
the local task-minus-rest effect averaged across all voxels in its
neighborhood.

This is a compact but realistic analysis primitive. It crosses three pieces
that are easy to get subtly wrong in raw nibabel/numpy code:

- iterating only over valid mask centers;
- defining spherical neighborhoods in millimetres, not voxel counts;
- projecting scalar center values back into the same spatial frame with
  provenance.

The scenario asks whether neuroim turns the workflow into a typed result
object with an owned `map_to_volume()` projection, while still matching a
careful nibabel/scipy-style baseline numerically.
