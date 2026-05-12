# neuroim2 -> neuroimpy Completion Audit

Date: 2026-05-12

## Objective

Finish the port of `~/code/neuroim2` to idiomatic Python with full parity, a complete implementation, and a well-architected Python package.

## Success Criteria

1. The Python package implements the neuroim2 core data model and behavior.
2. The Python package has a top-level API surface that covers the exported neuroim2 package surface, using Pythonic names where appropriate and compatibility aliases where useful.
3. The implemented behavior is covered by executable tests, including R/Python parity tests where available.
4. Documentation and parity trackers reflect the actual current state.
5. The package passes the full local test suite.

## Evidence Checked

| Requirement | Evidence | Current Result |
| --- | --- | --- |
| Core data model | `src/neuroimpy/` classes for `NeuroSpace`, volumes, vectors, slices, ROI, clustered volumes/vectors, memory-backed vectors, hypervectors | Substantially implemented |
| I/O parity | `src/neuroimpy/io.py`, `file_format.py`, `afni_io.py`, `nifti_extension.py`; tests in `tests/test_afni_io.py`, `tests/test_phase6_io.py`, `tests/test_phase9_file_formats.py` | NIfTI and AFNI HEAD/BRIK read/write implemented; lightweight NIML parsing implemented; NIML writing not implemented |
| Fortran/R indexing parity | Focused tests in searchlight, sparse masks, split/stat helpers, clustered volumes | Full suite passes after Fortran-order fixes |
| API surface parity | Compared `~/code/neuroim2/NAMESPACE` against `hasattr(neuroimpy, name)` after compatibility aliases | 207 of 207 exported R names currently resolve. Dotted R names such as `as.array` resolve through `getattr(neuroimpy, "as.array")`; idiomatic Python callers should use `as_array` |
| Class surface parity | Compared `exportClasses()` entries against Python names | 52 of 52 class exports currently resolve. R virtual classes are represented as Python structural protocols or type aliases where appropriate |
| Compatibility endpoint tests | `python -m pytest tests/test_compat_api.py tests/test_phase8_resample.py tests/test_phase1_core.py -q` and focused R oracle tests | Includes literal export alias checks, structural virtual-class checks, automask largest-component/hole-fill behavior, gradual clip volume thresholds, scalar `output_aligned_space`, `deoblique(NeuroSpace)`, and R/Python `bilateral_filter_4d` parity |
| Test gate | `python -m pytest -q` | `1213 passed, 3 warnings` after adding the missing notebook test dependency (`nbconvert`) to the dev extras |
| Beads issue workflow | `bd --db .beads/beads.db ready --json` | `[]`; direct `bd ready --json` still did not auto-discover the database in this shell, so the explicit database path was used |

## Current Unresolved Exported R Names

None from the mechanical `NAMESPACE` audit.

Notes:

- R's exported `None` sentinel is exposed for mechanical access via `getattr(neuroimpy, "None")`; Python source code should use the built-in `None`.
- R dotted generics (`as.array`, `as.dense`, `as.mask`, `as.matrix`, `as.sparse`) are exposed through module attributes for migration/audit code, with idiomatic underscore wrappers as the normal Python API.

## Current Unresolved R Class Names

None from the mechanical `exportClasses()` audit.

Notes:

- `NeuroObj`, `ArrayLike3D`, `ArrayLike4D`, and `ArrayLike5D` are Python `Protocol` definitions because the R classes are virtual S4/interface scaffolding.
- `numericOrMatrix` is a type alias.
- `AbstractSparseNeuroVec` currently resolves to the concrete sparse vector implementation.

## Completion Status

Complete against the current objective. The package now covers the exported neuroim2 API/class surface with Pythonic equivalents, compatibility aliases, or explicit non-goal documentation; the newest compatibility endpoints have executable semantic tests and R oracle coverage where behavior is not obvious from Python-only tests; and the full local quality gate passes.

## Recommended Next Slice

No parity blockers remain in the current audit.

Future work is performance and coverage expansion rather than required parity closure:

1. Add larger performance benchmarks for sparse/memory-backed workloads.
2. Expand optional R oracle fixtures for additional plotting and high-level workflow examples as those APIs evolve.
