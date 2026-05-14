# Scenarios

Side-by-side, runnable comparisons of one neuroimaging task implemented
**twice** — once in canonical raw `nibabel`+`numpy`, once through the
`neuroim` public API.

## Why

The mission test (per `MISSION.md` §Decision Rules / Strategic Test) is:

> *Would a competent nibabel user install neuroim-python because it makes
> their real analysis code shorter, safer, and more inspectable?*

The `examples/draw_audit/` folder answers that for **one** flagship
workflow (ROI-time-series + Pearson correlation). The `scenarios/`
folder is the dependency-ordered sweep that complements it: a series
of small, focused experiments that grow from "absolute hello world"
to "full pipeline."

Every scenario is *runnable* (`pytest examples/scenarios/<n>/test_scenario.py`)
so the comparison is reproducible, not aspirational.

## Comparison framework

For each scenario we score three axes:

| Axis | Question |
|---|---|
| **Lines / read-time** | Is the neuroim version shorter and easier to follow? |
| **Safety** | Does the API catch a class of bug the baseline lets through? |
| **Pain points** | What's awkward in the neuroim version that we should fix? |

A scenario is a **win** if it improves on at least one of the first two
and ties or improves on the others. Surfaced pain points are filed as
follow-ups even when the scenario wins overall — they keep us honest.

## Layout

Each scenario folder ships:

```
NN_<name>/
├── README.md              # task brief
├── baseline_nibabel.py    # canonical raw nibabel+numpy form
├── neuroim_version.py     # neuroim public-API form
├── test_scenario.py       # numeric parity + verdict assertions
└── REPORT.md              # scoreboard, pain points, verdict
```

## Index

| # | Scenario | Verdict | Pain points |
|---|---|---|---|
| 01 | **MNI spotlight** — extract a BOLD time series at a single world-mm coordinate | win on read-time / provenance / lines (after PAIN-1/2/3 from S01 landed) | 3 — filed in `01_mni_spotlight/REPORT.md`, all closed |
| 02 | **ROI mean time series** — collapse a 4-D BOLD across a brain mask | win on contract semantics; net wash on lines; **falsifies mission claim on silent space mismatch** | 2 — including **PAIN-5 (P0)**: `series_roi` silently accepts mismatched mask affine. See `02_roi_mean_timeseries/REPORT.md`. |
| 03 | **Seed-sphere mean** — mean BOLD across an 8 mm sphere at an MNI coord | **clean win** (15 → 1 statement, no helpers, typed `Receipt`) | 2 — both narrow provenance gaps (`Receipt.radius` is `None`, `method_name` doesn't distinguish `series_roi_world`). See `03_seed_sphere_mean/REPORT.md`. |
| 04 | **Temporal SNR map** — reduce a 4-D BOLD to a masked 3-D QC map | **clean win** (12 → 1 statement, typed `DenseNeuroVol.provenance`, mask-affine mismatch caught at the API surface) | 2 — both **closed**; `NeuroVec.temporal_snr(mask=...)` is now first-class and the result carries a Receipt. See `04_temporal_snr_map/REPORT.md`. |
| 05 | **Receipts across the IO boundary** — write a derived map, recover provenance in a clean process | **safety win, originally falsified the mission claim; now fixed and guarded** | 2 — **PAIN-6 (P0) closed**: Receipt now embeds as a NIfTI 'comment' extension and `read_image` rehydrates it onto `vol.provenance`. PAIN-7 (P2) subsumed. See `05_receipt_io_boundary/REPORT.md`. |
| 06 | **Public seed-to-voxel correlation** — Nilearn-adjacent seed connectivity workflow yielding a 3-D correlation map | **credibility-gate win**; same correlation math as raw nibabel/numpy, but spatial contracts and derived-map provenance are explicit | none — runnable parity + affine/OOB guards in `test_s06_public_seed_correlation.py`; see `06_public_seed_correlation/REPORT.md`. |
| 07 | **Multi-subject concat** — stack two subjects with mismatched affines for a group mean | **safety win**; `concat` already enforces same-space, but the scenario also uncovered and closed an ME-9 chain-prefix gap in `series_roi` | 1 — **PAIN-8** retracted as not-a-bug after re-probe; the chain-prefix gap was fixed in-iteration. See `07_multi_subject_concat/REPORT.md`. |
| 08 | **Pickle / multiprocessing handoff** — send a derived map through a worker/cache boundary | **win on inspectability** | none — typed `DenseNeuroVol.provenance` survives a fresh-process pickle round-trip; bare nibabel requires a manual manifest. See `08_pickle_multiprocessing_handoff/REPORT.md`. |
| 09 | **Native-to-template provenance** — resample native BOLD to template, then compute tSNR | **safety win, originally falsified inspectability; now fixed and guarded** | 1 — **PAIN-9 (P1) closed**: `resample_vec` now records source/target space hashes and interpolation order, and downstream derived-map receipts thread the normalization step. See `09_native_to_template_provenance/REPORT.md`. |
| 10 | **Searchlight space mismatch** — compute a local mean searchlight map with a shifted same-shape mask | **safety win, originally falsified the mission claim; now fixed and guarded** | 2 — **PAIN-10 (P0) closed**: `searchlight_apply` now calls `assert_same_space(data, mask)` before sampling, mirroring the PAIN-5 fix at `series_roi`. **PAIN-11 (P2) open**: deprecated `NeuroVec.series()` call still internal. See `10_searchlight_space_mismatch/REPORT.md`. |
| 12 | **File-backed affine drift** — compute tSNR from one 3-D file per time point, with one shifted same-shape volume | **falsifies a mission-bearing backend-safety claim** | 1 — **PAIN-12 (P0)**: `FileBackedNeuroVec` should reject per-volume affine drift, not only shape drift. See `12_file_backed_affine_drift/REPORT.md`. |

## Running

```bash
PYTHONPATH=src:tests:. .venv/bin/python -m pytest \
    examples/scenarios/test_s01_mni_spotlight.py -q
```

The fixture is `tests/fixtures/realistic_bold.py::make_realistic_bold()`
— the same deterministic 32×32×24×40 BOLD volume used by the draw audit.

### A note on file layout

The runnable acceptance test for each scenario lives one level up
(`scenarios/test_sNN_<slug>.py`) rather than inside the scenario folder,
because folder names like `01_mni_spotlight` are illegal Python module
names. The in-folder `test_scenario.py` is a pointer to its up-level
runnable counterpart; the `conftest.py` at `scenarios/` excludes
digit-prefixed folders from collection.

## Filing pain points

Every scenario surfaces friction points in the neuroim API. File each
one in **both** places so it is discoverable from the scenario *and*
indexed across the whole suite:

1. **`NN_<slug>/REPORT.md`** — add a `### PAIN-N` section under
   "Pain points surfaced." `N` is local to the scenario (PAIN-1,
   PAIN-2, …). Include: short description, **Impact**, **Suggested fix**,
   and a priority (P0–P3). Also list it in the "Follow-ups" table at
   the end of the report so the priority and a suggested issue title
   are scannable.

2. **mote discussion topic `pain-points-from-scenarios`** — append a
   post that mirrors the REPORT entry, prefixed with the scenario
   number, e.g. `Scenario 02 — PAIN-1: …`. The cross-scenario index
   was seeded by Scenario 01's initial post.

   ```bash
   # Read existing pains
   mote discuss list --topic pain-points-from-scenarios

   # Append new pains as a scenario lands
   mote discuss post --topic pain-points-from-scenarios --body "$(cat <<'EOF'
   Scenario NN — PAIN-N (Px): short title

   Body identical to the REPORT.md entry.
   EOF
   )"
   ```

This split keeps each `REPORT.md` self-contained for an offline
reader while still giving the suite a single durable index of every
pain ever surfaced. The mote topic outlasts any individual scenario
folder; the per-scenario `REPORT.md` is the audit trail for *why*
each pain was filed.
