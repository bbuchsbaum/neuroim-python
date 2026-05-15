# Scenario 18 -- Cluster-table from a thresholded stat map

**Task.** Given a 3-D statistic map (z-scores, t-values) and a brain
mask, threshold at `|z| > 2.3` within the mask, label connected
components, and emit the per-cluster table with `cluster_id`,
`n_voxels`, signed `peak_value`, and `peak_x_mm / peak_y_mm /
peak_z_mm`.  Canonical post-stats artifact in every task-fMRI paper.

**Why this scenario exists.**  Scenarios 01-17 produce time series,
parcel matrices, 3-D maps, or scenarios about them.  S18 is the first
scenario whose output shape is a **table** (pandas DataFrame) and the
first to exercise `neuroim.connected_components`.  Five real API gaps
surfaced and have now all been closed.

## Verdict

| Axis | Baseline (scipy + numpy) | neuroim after PAINs 1-5 |
|---|---|---|
| Two-tailed thresholding | yes (manual `np.abs`) | yes (`two_tailed=True`) |
| Mask gating | yes (manual `mask & ...`) | yes (`mask=` parameter) |
| Same-space contract caught | yes (manual `affine` compare) | yes (inside `conn_comp`) |
| Peak coords in world mm | yes (manual `affine @ ijk`) | yes (always-on table columns) |
| Output carries provenance Receipt | no | yes (`ConnCompResult.provenance`) |
| Function discoverable from `ni.__all__` | n/a | yes (`ni.conn_comp`) |
| User-facing body statements | ~6 | ~4 (single `conn_comp` call + column rename) |

**neuroim now wins the call-site shape for this workflow.**  The rewrite
is a single `conn_comp(stat_vol, threshold=..., mask=mask,
two_tailed=True)` call plus a four-line projection onto the canonical
table column names; everything the user used to write against raw
scipy -- absolutize, pre-mask, re-derive peak coords, wrap in
something that carries provenance -- now happens inside the public
API.

## PAINs

### PAIN-1 (P2 surface) -- closed: `conn_comp` in `ni.__all__`

`ni.conn_comp` is now part of the curated public surface.  The result
class `ConnCompResult` has been exported since ME-4; the function that
produces one is now exposed alongside it.  Budget bumped 42 -> 43 in
`tests/test_public_api.py::test_public_api_budget_and_required_names`.

### PAIN-2 (P1 provenance) -- closed: `ConnCompResult.provenance`

`ConnCompResult` gained a `provenance: Optional[Receipt] = None` field.
`conn_comp` populates it via the structural ``receipt_for`` path with
`method_name="conn_comp"`, the input stat map's `input_space_hash`,
the mask `mask_hash` (when supplied), and `radius == threshold`.

### PAIN-3 (P1 contract) -- closed: `mask=` parameter + same-space gate

`conn_comp(x, *, mask=None, ...)` accepts a keyword-only
`LogicalNeuroVol` mask.  When supplied, `verify.assert_same_space(x,
mask)` is invoked first and the suprathreshold map is intersected with
the mask.  An LR-flipped mask raises before clustering runs.

### PAIN-4 (P1 ergonomic) -- closed: `two_tailed=True`

`conn_comp(x, *, two_tailed=True, ...)` thresholds `|x.data| >
threshold` and the resulting `cluster_table` carries signed
`peak_value` columns by looking up the original signed data at each
cluster's local maximum.  Default remains `two_tailed=False` for
backward compatibility with existing callers.

### PAIN-5 (P2 ergonomic) -- closed: always-on `peak_*_mm` columns

`ConnCompResult.cluster_table` always includes `peak_x_mm`,
`peak_y_mm`, `peak_z_mm`, and signed `peak_value` columns alongside
the legacy `index`, `x`, `y`, `z` (centroid mm), `N`, `Area`, and
`value` (mean) columns.  No flag -- additive change.
`tests/test_connected_components.py::test_conn_comp_cluster_table`
updated to expect the new columns.

## Sibling change -- `NeuroVol.values_roi(roi)` method

Filed as the consistency mote (`bd-01KRMNY6RJ3VGR0N6T4Y4C7QZ8`).
`NeuroVec.series_roi(roi)` has had a method form alongside the free
function `ni.series_roi(vec, roi)` since ME-1; the symmetrical method
on the volume side now exists too.  `NeuroVol.values_roi(roi)`
delegates to the free function and inherits the same-space gate.

## Fixture

Deterministic synthetic stat map (16 x 16 x 12, 3 x 3 x 3.5 mm voxels)
with two embedded clusters:

- positive: ``data[10:13, 10:13, 6:9] += 3.5`` (~27 voxels)
- negative: ``data[4:7, 4:7, 3:5] -= 3.2`` (~18 voxels, placed inside
  the ellipsoid brain mask)

Background noise is `N(0, 0.8)`.  Both clusters survive `|z| > 2.3` +
`min_extent=5` after noise.

## Acceptance state

`PYTHONPATH=src:tests:. python -m pytest tests/test_s18_cluster_table_post_stats.py`:

- **8 passed, 0 xfailed.**  Numeric parity row-by-row between baseline
  and rewrite; both paths surface the negative cluster (`two_tailed`
  path through the API); same-space gate on LR-flipped mask;
  `conn_comp` is in `ni.__all__`; `ConnCompResult.provenance` is
  populated; `mask=` parameter exists and triggers the gate;
  `two_tailed=True` is supported and produces both signs; world-mm
  peak columns are present.

Broader regression slice (S11+S12+S13+S18+public_api+image_processing+
phase7_spatial_filters+verifier_blessed_path+fragility_invariants+
roi_results+structural_receipts+connected_components): clean apart
from one pre-existing failure in `test_neurovec_edge_cases.py` (from
another lane's PAIN-13 time-slice work, unrelated to this change).

## Source

- Baseline: `examples/scenarios/18_cluster_table_post_stats/baseline_nibabel.py`
- Rewrite: `examples/scenarios/18_cluster_table_post_stats/neuroim_version.py`
- Test: `tests/test_s18_cluster_table_post_stats.py`
- API surface: `src/neuroim/connected_components.py::conn_comp` +
  `src/neuroim/__init__.py::__all__`.
- Consistency: `src/neuroim/neuro_vol.py::NeuroVol.values_roi` +
  `tests/test_roi_results.py`.
