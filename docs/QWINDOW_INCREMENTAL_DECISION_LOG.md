# q-window incremental decision log

Date: 2026-06-02

## Decision

Add an optional `robust_incremental` exact-oracle mode that scans only newly
exposed q strips during q-window expansion, while preserving the existing
`robust_al` full-rescan mode as the regression baseline.

## Reason

The robust oracle restored label closure but increased exact-oracle walltime.
The main repeated operation is full q-window rescanning after expansion. An
incremental strip scan directly targets that cost without changing the physics
criterion or acquisition loop.

## Constraints

- Do not change the thermodynamic phase criterion.
- Do not change Delta refinement trigger or final ambiguity tolerances.
- Do not change stable-normal training admission.
- Do not change acquisition formula, weights, candidate domain, or
  StopController.
- Do not submit new AL iterations as part of this refactor.

## Implementation summary

- Added `QScanCache` and q-cache merge helpers.
- Added explicit-q scan helper with optional normal scalar reuse.
- Added `robust_incremental` oracle mode and
  `--enable-incremental-q-expansion`.
- Added per-point timing and workload counters to exact outputs.
- Added rank-level environment snapshot and workload sums in the Slurm exact
  script.
- Added regression and benchmark scripts for report-only validation.
- Added lightweight unit tests for cache merge, q-grid alignment, fallback
  metadata, timing fields, and rank summary schema.

## Required validation before production use

1. Run the unit tests locally.
2. Run `scripts/compare_incremental_qexpansion_regression.py` on selected
   high-\(J_A\), low-\(T\) correction points and stable-normal controls.
3. Confirm phase/trusted/training-eligible labels match `robust_al`.
4. Run `scripts/run_qwindow_incremental_benchmark.py` on HPC to measure runtime
   and q-grid-evaluation savings.
5. Only after the regression report passes should an AL mini-run use
   `robust_incremental`.

## Current status

Implemented as an optional path. Production `robust_al` remains unchanged.
