# Scenario 13 -- Cited Nilearn port: seed-to-voxel resting-state connectivity

This scenario is a *cited* port of a specific published Nilearn tutorial.
S06 demonstrated the analytical primitive (seed-to-voxel correlation) on
synthesized data without a source attribution.  S13 fixes the
selection-bias concern from `are-we-winning` by naming the example, the
version, and the line-for-line mapping from the Nilearn version to the
neuroim version.

## Cited example

- **Title:** "Producing single subject maps of seed-to-voxel correlation"
- **URL:** <https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html>
- **Nilearn version:** 0.10.x (stable docs branch as of 2026-05)
- **What it computes:** Pearson correlation between a PCC seed (radius
  8 mm) and every brain voxel of a 4-D resting-state BOLD image,
  producing a 3-D connectivity map.

## CI dependency policy

Nilearn is **not** added as a CI dependency.  The dataset Nilearn
downloads (ADHD200 sample) is not vendored either.  Instead, the
scenario uses the existing `tests/fixtures/realistic_bold.py` synthetic
fixture and *mirrors* the Nilearn tutorial's analytical structure:

| Nilearn primitive | Substitute in this scenario |
|---|---|
| `nilearn.datasets.fetch_development_fmri()` | `tests.fixtures.realistic_bold.make_realistic_bold()` |
| `NiftiSpheresMasker(seeds=[pcc_coords], radius=8)` | manual radius-8 mm sphere mean via voxel-grid mask |
| `NiftiMasker(mask_img=brain_mask)` | direct `bold[mask, :]` indexing |
| `np.dot(brain.T, seed) / (n - 1)` (after z-score) | `_correlate_seed_to_voxels` helper |
| `nib.save(connectivity_img, ...)` | `nib.Nifti1Image(corr_map, affine)` |

The point is that the *source* of the workflow is public and pinnable,
not that Nilearn is imported here.  See `REPORT.md` for the line-for-line
nilearn-to-neuroim mapping table.

## Files

- `baseline_nibabel.py` -- nibabel + numpy implementation that follows the
  Nilearn tutorial's analytical structure.
- `neuroim_version.py` -- neuroim rewrite.
- `REPORT.md` -- line-for-line mapping + verdict.
- Test: `tests/test_s13_cited_nilearn_port.py`.
