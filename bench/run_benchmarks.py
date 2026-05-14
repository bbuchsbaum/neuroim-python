"""First-cut benchmarks for neuroim-python mission claims.

Source: bd-01KRKSCXE4MH9TJCZYNV7FSVF6 (open-discussion-2026-05-14;
are-we-winning post-01KRKS4E1R7BK668GCSFD29QHB losing signal #4).

The "lazy/sparse wins on real data" claim is unproven without numbers.
The typed-result path could be measurably slower per op than the legacy
ndarray path and nobody would know.  This script produces a first set of
measurements on the realistic_bold fixture and writes the result table to
``bench/REPORT.md``.

Measurements shipped here:

  Searchlight wall-time, typed result vs legacy projection
      Same fixture, same method, same radius; only the ``return_legacy``
      flag varies.  Establishes the per-op overhead of the typed
      SearchlightResult and provenance Receipt threading relative to
      the bare DenseNeuroVol projection.

  SparseNeuroVol round-trip wall-time vs DenseNeuroVol baseline
      DenseNeuroVol -> SparseNeuroVol (``as_sparse(mask)``) -> DenseNeuroVol
      (``as_dense()``).  Compared against a same-shape DenseNeuroVol copy
      round-trip as a baseline (so the report quantifies sparse overhead,
      not raw allocation cost).  The masked sparse path is what the
      "lazy/sparse wins on real data" claim depends on.

  Peak traced memory for each timed block
      ``tracemalloc.get_traced_memory()`` peak captured around each run.
      ``tracemalloc`` only measures Python-allocator activity (not raw
      mmap pages from file-backed reads).  A future slice should add a
      ``psutil`` peak-RSS measurement for the file-backed comparison.

Not yet shipped (scaffolded as TODO):

  FileBackedNeuroVec.series_roi peak RSS vs DenseNeuroVec.series_roi
      Requires writing N per-volume NIfTI files to a tmp directory and
      measuring peak resident set with ``psutil`` (or
      ``resource.getrusage(RUSAGE_SELF).ru_maxrss``).  Left as a
      follow-up so the first benchmarks ship with concrete numbers.

Run:
    PYTHONPATH=src:tests:. python bench/run_benchmarks.py

The script writes ``bench/REPORT.md`` and prints the table to stdout.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Make tests/ importable so we can reuse the realistic_bold fixture without
# duplicating the generator.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests"))

import neuroim as ni  # noqa: E402
from fixtures.realistic_bold import make_realistic_bold  # noqa: E402


# ----------------------------------------------------------------------
# Timing helpers
# ----------------------------------------------------------------------


@dataclass
class Measurement:
    """One labelled measurement: wall-time samples + peak traced memory."""

    label: str
    runs: int
    wall_seconds: list[float] = field(default_factory=list)
    peak_bytes: list[int] = field(default_factory=list)
    notes: str = ""

    @property
    def median_wall_ms(self) -> float:
        return 1000.0 * statistics.median(self.wall_seconds)

    @property
    def min_wall_ms(self) -> float:
        return 1000.0 * min(self.wall_seconds)

    @property
    def max_peak_mb(self) -> float:
        return max(self.peak_bytes) / (1024 * 1024)


def measure(label: str, fn: Callable[[], object], *, runs: int = 5, notes: str = "") -> Measurement:
    """Run ``fn`` ``runs`` times.  Record wall-time and tracemalloc peak."""
    m = Measurement(label=label, runs=runs, notes=notes)
    # Warm-up to amortise first-touch costs (e.g. module imports inside the
    # call path, lazy fixtures).  Not counted in stats.
    fn()
    for _ in range(runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        m.wall_seconds.append(elapsed)
        m.peak_bytes.append(peak)
    return m


# ----------------------------------------------------------------------
# Benchmark cases
# ----------------------------------------------------------------------


def bench_searchlight_typed_vs_legacy(bold: ni.DenseNeuroVec, mask: ni.LogicalNeuroVol):
    """Measure searchlight in both return modes on the same input."""
    radius_mm = 6.0
    method = lambda block: float(np.mean(block)) if block.size else float("nan")  # noqa: E731

    typed = measure(
        "searchlight (typed SearchlightResult)",
        lambda: ni.searchlight_apply(mask, radius=radius_mm, method=method, data=bold),
        notes=f"radius={radius_mm}mm, method=mean, return_legacy=False (default)",
    )
    legacy = measure(
        "searchlight (return_legacy=True DenseNeuroVol)",
        lambda: ni.searchlight_apply(mask, radius=radius_mm, method=method, data=bold, return_legacy=True),
        notes=f"radius={radius_mm}mm, method=mean, return_legacy=True (legacy projection)",
    )
    return typed, legacy


def bench_sparse_round_trip(bold: ni.DenseNeuroVec, mask: ni.LogicalNeuroVol):
    """Compare SparseNeuroVol round-trip vs a same-shape dense baseline."""
    # Pick one time-point as a 3-D volume for the round-trip.
    vol_3d = ni.DenseNeuroVol(np.ascontiguousarray(bold.data[..., 0]), mask.space)

    def sparse_round_trip():
        sparse = vol_3d.to_sparse(mask)
        return sparse.to_dense()

    def dense_baseline():
        # The non-sparse baseline: deep-copy + materialize a new DenseNeuroVol.
        # Same data movement (one full read, one full write) without going
        # through the sparse indices layer.
        return ni.DenseNeuroVol(np.array(vol_3d.data, copy=True), vol_3d.space)

    sparse = measure(
        "SparseNeuroVol round-trip (to_sparse(mask) -> to_dense)",
        sparse_round_trip,
        notes=(
            f"shape={vol_3d.shape}, mask voxels={int(np.asarray(mask.data).sum())}"
            f", dtype={vol_3d.data.dtype}"
        ),
    )
    dense = measure(
        "DenseNeuroVol baseline (deep-copy + re-wrap)",
        dense_baseline,
        notes="baseline for sparse round-trip comparison; same shape, no mask projection",
    )
    return sparse, dense


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------


_REPORT_TEMPLATE = """\
# Benchmarks: first-cut measurements

Source: bd-01KRKSCXE4MH9TJCZYNV7FSVF6 (open-discussion-2026-05-14;
are-we-winning losing signal #4).  These numbers are evidence for or
against the "lazy/sparse wins on real data" claim; without them, that
claim has no grounding.

Generated by: `PYTHONPATH=src:tests:. python bench/run_benchmarks.py`

Fixture: `tests.fixtures.realistic_bold.make_realistic_bold()` -- a
deterministic 32x32x24x40 float64 BOLD volume (~7.5 MB raw) with a
brain-shaped LogicalNeuroVol mask.  Single-machine measurements; not
cross-machine.

Platform: {platform}
Python:   {python}
NumPy:    {numpy}

## Results

| Measurement | Runs | Median wall (ms) | Min wall (ms) | Peak traced (MB) |
|---|---:|---:|---:|---:|
{rows}

### Notes per measurement
{notes}

## Interpretation guide

- *Median wall* is the within-process timing, not end-to-end "user runs
  the script" time.  A warm-up call is discarded before timed runs.
- *Peak traced* reports `tracemalloc.get_traced_memory()` only, so it
  captures Python-allocator activity.  Numpy buffers backed by raw
  `malloc` may not appear.  File-backed mmap pages are invisible to
  `tracemalloc`; the file-backed comparison below needs `psutil`.
- Sparse round-trip is timed against a same-shape DenseNeuroVol
  deep-copy baseline so the report quantifies sparse-specific overhead
  rather than raw allocation cost.  A sparse advantage on dense-payload
  data should look the same as the baseline; the advantage shows up
  when payloads are genuinely sparse (the fixture's mask is brain-
  shaped, not whole-volume).

## Not yet measured (follow-up)

- **FileBackedNeuroVec.series_roi peak RSS** vs DenseNeuroVec.series_roi
  on the same realistic_bold fixture.  Requires writing N per-volume
  NIfTI files to a tmp directory and measuring with `psutil` or
  `resource.getrusage(RUSAGE_SELF).ru_maxrss`.  Scaffolded in
  `run_benchmarks.py` as a TODO; left as a follow-up so the first
  benchmarks could ship with concrete numbers.
- **>=1 GB synthetic fixture**.  Jim's are-we-winning prio-3 item
  explicitly named >=1 GB to make the lazy/sparse advantage observable.
  realistic_bold at default shape is ~7.5 MB; a 1-GB fixture would
  need a larger spatial grid or many more timepoints.  Left as a
  follow-up.

## Source

Script: `bench/run_benchmarks.py`
"""


def render_report(measurements: list[Measurement]) -> str:
    rows = []
    notes = []
    for m in measurements:
        rows.append(
            f"| {m.label} | {m.runs} | {m.median_wall_ms:.2f} | "
            f"{m.min_wall_ms:.2f} | {m.max_peak_mb:.2f} |"
        )
        if m.notes:
            notes.append(f"- **{m.label}** -- {m.notes}")
    return _REPORT_TEMPLATE.format(
        platform=platform.platform(),
        python=sys.version.split()[0],
        numpy=np.__version__,
        rows="\n".join(rows),
        notes="\n".join(notes) if notes else "_(no per-measurement notes)_",
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Runs per measurement (default: 5).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_REPO_ROOT / "bench" / "REPORT.md",
        help="Path to write the markdown report (default: bench/REPORT.md).",
    )
    args = parser.parse_args(argv)

    fixture = make_realistic_bold()
    bold, mask = fixture.bold, fixture.mask

    measurements: list[Measurement] = []
    sparse, dense = bench_sparse_round_trip(bold, mask)
    measurements.extend([sparse, dense])
    typed, legacy = bench_searchlight_typed_vs_legacy(bold, mask)
    measurements.extend([typed, legacy])

    report = render_report(measurements)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)

    print(report)
    print(f"\nReport written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
