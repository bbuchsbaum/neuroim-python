# Scenario 13 -- Cited Nilearn port: seed-to-voxel correlation

**Cited example.**
"[Producing single subject maps of seed-to-voxel
correlation](https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html)" --
Nilearn stable docs, version 0.10.x as of 2026-05.

**Why this scenario exists.**  S06 demonstrated the analytical primitive
without an external attribution.  The `are-we-winning` thread called
that out as a selection-bias risk: "us writing the baseline, us writing
the rewrite, us measuring."  S13 fixes that by naming the published
workflow, pinning its version, and laying out the line-for-line
nilearn->neuroim mapping below.  Nilearn is *not* imported here -- the
mapping is to a hand-rolled `nibabel + numpy` baseline that mirrors the
tutorial's analytical structure.  The scenario is a credibility moat,
not a thesis test.

## Line-for-line mapping

| Nilearn tutorial step (paraphrased) | This scenario's baseline | neuroim rewrite |
|---|---|---|
| `dataset = fetch_development_fmri(...)` | `make_realistic_bold()` (synthesized) | same |
| `pcc_coords = (0, -52, 18)` (MNI) | `_seed_world_xyz(fixture)` (in-fixture coord) | same |
| `seed_masker = NiftiSpheresMasker(seeds=[pcc_coords], radius=8)` | `baseline.extract_seed_time_series(..., radius_mm=8.0)` (manual sphere mean) | `bold.series_roi_world(seed_xyz, radius=8.0).values.mean(axis=1)` |
| `seed_time_series = seed_masker.fit_transform(func_filename)` | (folded into the above) | (folded into the above) |
| `brain_masker = NiftiMasker(mask_img=...).fit()` | `baseline.extract_voxel_time_series(bold_img, mask_img)` (explicit shape + affine check) | `bold.series_roi(ROICoords(coords, mask.space))` (verifier in API) |
| `brain_time_series = brain_masker.fit_transform(...)` | (folded into the above) | (folded into the above) |
| `seed_to_voxel_correlations = (brain_time_series.T @ seed_time_series) / (n_samples - 1)` (after `signal.clean(..., standardize=True)`) | `baseline._pearson_seed_to_voxels` (manual z-score + dot + divide) | identical helper inlined; same formula |
| `corr_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)` | `np.zeros(mask.shape); corr_map[mask] = corr_values; nib.Nifti1Image(corr_map, bold_img.affine)` | `NeuroVol.from_array(corr_values, mask.space, coords=coords)` (typed result) |
| -- (Nilearn does not emit provenance) | -- | `corr_map.provenance = make_receipt(...)` (PAIN-3-style receipt) |

## Verdict

| Axis | Baseline | neuroim |
|---|---|---|
| User-facing body statements | ~6 (4 helpers + main) | ~5 (4 lines of orchestration + receipt) |
| Same-space mismatch caught | yes (explicit `affine` comparison line in `extract_voxel_time_series`) | yes (inside `NeuroVec.series_roi`) |
| Seed OOB safety | yes (the sphere-mask check raises if the sphere doesn't intersect the grid) | yes (`series_roi_world` raises through the world-coord OOB contract) |
| Output map carries receipt | no | yes (`method_name="seed_to_voxel_correlation_map"`, `radius=8.0`, `mask_hash`) |
| Source workflow citable from this repo | yes (README) | yes (README) |

**The cited part is the value here, not the call-site delta.**  S06 and
S13 produce the same numbers via the same analytical primitive.  S13's
distinguishing artifact is the explicit citation + mapping table -- the
thing the `are-we-winning` thread asked for to push past
us-vs-us-on-our-own-fixture credibility.

## Fixture

Reuses `tests/fixtures/realistic_bold.py` (32 x 32 x 24 x 40 BOLD on
3 x 3 x 3.5 mm voxels with an elliptical brain mask) so the scenario
adds no CI dependency.  Seed coordinate is derived from one of the
fixture's `target_roi_centers` so it lands inside the synthetic mask.

## Acceptance state

`PYTHONPATH=src:tests:. python -m pytest tests/test_s13_cited_nilearn_port.py`:

- **5 passed, 0 xfailed** -- numeric parity at in-mask voxels;
  output-affine parity with BOLD; same-space gate on LR-flipped mask;
  world-coord OOB rejection; receipt presence.

## Source

- Cited tutorial:
  <https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html>
- README: `examples/scenarios/13_cited_nilearn_port/README.md`
- Baseline: `examples/scenarios/13_cited_nilearn_port/baseline_nibabel.py`
- Rewrite: `examples/scenarios/13_cited_nilearn_port/neuroim_version.py`
- Test: `tests/test_s13_cited_nilearn_port.py`
