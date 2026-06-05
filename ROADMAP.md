# Roadmap

This file states the **release plan**. It does not amend the charter.
[VISION.md](VISION.md) and [MISSION.md](MISSION.md) state the durable
destination and decision rules; the live operational rules are pinned in the
mote topic `neuroim-python-pythonic-value`. This document states *how and when
we ship* on the way to a stable 1.0, and how we keep the right to make
breaking changes until then. The stability contract itself lives in
[STABILITY.md](STABILITY.md).

## Philosophy: usable now, not frozen yet

neuroim-python is meant to be installed and experimented with **before** the
API is stable. The two requirements — *a regular, working interface people can
build against today* and *the freedom to change that interface* — are
reconciled by a written contract, not by staying private:

- The public API is **usable, documented, and tested** at every release.
- The public API is **not frozen** until 1.0.
- Breaking changes are allowed **between minor versions** (`0.3 → 0.4`),
  **never within a patch** (`0.3.0 → 0.3.1`).
- A name that will break emits `DeprecationWarning` for **one full minor**
  before removal.
- Every minor before 1.0 is allowed to break; every minor before 1.0 must
  also **shrink** the set of things that can still break. 1.0 is reached by
  attrition of the deprecation surface, not a big-bang freeze.

This is the pre-1.0 posture NumPy and Nilearn used: real, documented, used in
anger, explicitly unstable. See [STABILITY.md](STABILITY.md) for the binding
contract and the deprecation cadence.

## Versioning

`0.MAJORish.PATCH` with the 0.x reading above:

- **patch** (`0.3.0 → 0.3.1`): bug fixes and honest-gap closures only. No
  public API removals or signature breaks. Safe to pin.
- **minor** (`0.3 → 0.4`): may break the public API, with one minor of prior
  `DeprecationWarning`. Read the CHANGELOG before upgrading.
- **1.0.0**: stability commitment — strict SemVer thereafter (public breaks
  only at 2.0).

`neuroim.compat` is governed separately and more strictly: it preserves
documented neuroim2 migration behavior under a hard parity gate and is not
subject to the native API's freedom to change.

## Release plan

| Version | Theme | Exit criteria |
|---|---|---|
| **0.3.0** | First public experimental release | Release gate green (see below). Regular documented interface; [STABILITY.md](STABILITY.md) published; clean `pip install` proven on supported Pythons. Right to break preserved. |
| **0.3.x** | Honest-gap patches | Patch-only, no API breaks: slice→tSNR provenance chain (PAIN-13/14, currently the lone xfail). (`write_vol` receipt embedding — formerly PAIN-4 — has landed.) |
| **0.4.0** | Evidence hardening + first deliberate breaks | Perf/memory benchmark with a ≥1 GB fixture and a nibabel head-to-head; deliberate-misuse quickstart; cited Nilearn port promoted to a credibility-grade README example. First batch of pre-announced breaking changes (e.g. `return_legacy=True` paths raise). |
| **0.5.0** | External validation | Cold-reader docs review; ≥1 outside experimenter runs a public scenario end-to-end and reports; group-level / cross-subject workflow scoped on its own discussion thread. |
| **0.6–0.9** | API stabilization | Per minor: shrink the deprecation surface; freeze one subsystem at a time (spatial model → ROI/searchlight → provenance schema → IO). `Receipt` schema v1 locked. Last allowable break lands by 0.9. |
| **1.0.0** | Stability commitment | Zero outstanding `DeprecationWarning`s. Public API frozen under strict SemVer. Receipt schema versioned and stable. External-validation evidence on record. Performance characterized, not just functional. |

## 0.3.0 release gate

These are packaging/trust items, not features. The 0.3 *evidence* milestone
(`bd-01KRKRZF7A9818NSM5QYQX9YJT`) is already closed: the cited Nilearn port,
4D→3D derived maps, and provenance surviving NIfTI serialization all exist and
pass. What 0.3.0 adds is a trustworthy, installable, honestly-documented
release. Tracked as mote epic `bd-01KRRJKVCKEZXX5FXTB610GRV5`:

1. **Fix the 3 golden-harness failures** — re-sync the concatenation spec
   (stale `neurovecseq`) and remove the hardcoded `pyneuroim` path in the
   golden test harness. A red golden suite cannot gate a release.
2. **Enforce the Python floor** — the suite must run on the declared
   `requires-python >= 3.10`; add a CI matrix (3.10/3.11/3.12) that proves a
   clean `pip install .` + `pytest` from a fresh environment.
3. **Land the in-flight docs-accuracy mote** — correct the overclaimed README
   badges so the documented numbers match reality.
4. **Publish [STABILITY.md](STABILITY.md)** — the 0.x contract, deprecation
   cadence, and the two-lane parity policy, linked from README and packaging
   metadata.
5. **Experimental banner** — a clear "experimental, not stable" note at the
   top of the README and the `Development Status :: 3 - Alpha` classifier in
   `pyproject.toml`.
6. **Reproducible install proof** — `make verify-evidence` run from a clean
   clone, documented in the release notes.
7. **CHANGELOG / release notes** — what works, the stability contract, and the
   known residual gaps stated honestly.

### Not blocking 0.3.0 (named as known limitations in release notes)

- slice→tSNR provenance chain incomplete — PAIN-13/14 (the lone xfail).
- Benchmark evidence is first-cut: no ≥1 GB fixture, no nibabel head-to-head
  timing yet.
- No external/cold-reader validation yet.

## How the right to break is protected operationally

- **Scenario-Evidence Gate** (AGENTS.md): new public API must be pulled by a
  runnable scenario plus a red check — no speculative surface to regret later.
- **`neuroim.compat`** absorbs every legacy/R-shaped name, keeping the native
  API free to change.
- **Charter precedent** (read_image ruling, mote `post-01KRRFFAGGRZFVMB4VS31K81ET`):
  a MISSION-rule change cannot enter through an execution lane; it needs a
  standalone charter slice. This protects the *contract* while leaving the
  *implementation* mutable.
- **[STABILITY.md](STABILITY.md) + 0.x versioning** is the public-facing half
  of the same guarantee: users are told, in writing, not to pin against the
  native API until 1.0.
