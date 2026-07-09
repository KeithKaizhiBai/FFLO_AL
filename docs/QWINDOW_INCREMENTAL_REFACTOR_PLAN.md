# q-window incremental refactor plan

Date: 2026-06-02

## Scope

This refactor targets the robust exact-oracle performance path only. It does
not change the thermodynamic phase criterion, Delta tolerances, acquisition
formula, StopController, active-learning control flow, topology logic, or eta
response rules.

## Motivation

The robust oracle currently re-scans the full expanded q-window after every
q-window expansion. In the high-\(J_A\), low-\(T\) region this makes exact
oracle walltime much larger than the original oracle. The retrospective Slurm
audit showed no evidence of traceback, CUDA failure, OOM, or timeout in the
successful 5-iteration mini runs; the dominant cost is expected to be repeated
free-energy grid evaluation inside the robust oracle.

## Design

The new path adds an optional `robust_incremental` oracle mode. The existing
`robust_al` mode remains the full-rescan baseline.

For each exact point:

1. Run the same base scan as before.
2. Diagnose q-window edge risk with the existing logic.
3. If q expansion is requested:
   - baseline `robust_al`: re-scan the whole expanded q-window;
   - `robust_incremental`: scan only newly exposed left/right q strips and
     merge them with the previous q scan cache.
4. If the incremental strip cannot be formed, fall back explicitly to the full
   rescan and record `fallback_full_rescan_reason`.
5. Keep local refinement and Delta guardrail logic unchanged.
6. Save workload counters and timing fields into every exact shard.

## New metadata

Each exact point now records:

- q-point counters:
  `base_q_points_evaluated`, `added_left_q_points_evaluated`,
  `added_right_q_points_evaluated`, `recomputed_q_points`,
  `total_q_points_evaluated`.
- grid-evaluation counters:
  `base_grid_evaluations`, `incremental_q_grid_evaluations`,
  `fallback_full_rescan_grid_evaluations`,
  `delta_refinement_grid_evaluations`,
  `local_refinement_grid_evaluations`,
  `total_estimated_grid_evaluations`.
- runtime counters:
  `point_total_runtime_sec`, `base_scan_runtime_sec`,
  `q_expansion_runtime_sec`, `delta_refinement_runtime_sec`,
  `local_refinement_runtime_sec`, `merge_cache_runtime_sec`,
  `local_minima_detection_runtime_sec`,
  `fallback_full_rescan_runtime_sec`, `other_runtime_sec`.
- mode/fallback counters:
  `incremental_expansion_used`, `fallback_full_rescan_used`,
  `fallback_full_rescan_reason`.

## Regression requirement

Before any full active-learning restart, compare `robust_al` and
`robust_incremental` on representative high-\(J_A\), low-\(T\) correction points
and stable-normal control points. The required checks are:

- phase labels match or any mismatch is explicitly audited;
- trusted/training-eligible flags match;
- \(\Delta F\), \(q_{\rm opt}\), and \(\Delta_{\rm opt}\) differences are
  within numerical tolerance or documented;
- incremental mode reduces q-grid evaluations for expanded-window points.

## Non-goals

This refactor does not attempt branch-resolved topology classification,
MCMC/multistart search, parameter continuation, acquisition simplification, or
new stopping rules.
