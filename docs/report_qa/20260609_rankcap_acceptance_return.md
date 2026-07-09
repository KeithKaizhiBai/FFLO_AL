# Rank-and-Cap K3 Acceptance Return Check

Date: 2026-06-09

## Question

The HPC return directory `local_refinement_rankcap_acceptance_upload/` was
downloaded.  Check whether the one-pipeline acceptance test completed and what
the result means.

## Short Answer

The returned computation completed all required point tasks after local
report-only collection was repaired.  The final regenerated report gives
`acceptance_status = pass` for baseline robust_incremental versus
rank_and_cap_k3 on the fixed 32 points.

## Evidence

- Target-construction tasks: 32 JSON files, all `status=success`.
- Regression tasks: 64 JSON files, all `status=success`.
- Gate A: pass.
- Gate B: pass.
- Gate C: pass.
- 32/32 fixed points complete.
- `max_rankcap_selected_targets = 3`.
- `mandatory_overflow_points = 32`, but overflow was rank-and-capped and did
  not let selected targets exceed 3.
- `phase_label_match_rate = 1`.
- `trusted_exact_match_rate = 1`.
- `training_eligible_exact_match_rate = 1`.
- `q_unresolved_increased_count = 0`.
- `delta_unresolved_increased_count = 0`.
- `timeout_count = 0`.
- `mismatch_point_count = 0`.
- Maximum absolute differences in `DeltaF`, `q_opt`, `Delta_opt`, and
  `positive_delta_gap` are all 0 in the comparison table.

## Workload and Runtime

- Mean local boxes: baseline 6, rank_and_cap_k3 2.75.
- Mean local-refinement runtime: baseline 189.767 s, rank_and_cap_k3 86.9015 s.
- Mean point total runtime: baseline 234.194 s, rank_and_cap_k3 132.314 s.
- Local boxes were reduced by about 54.17%.
- Local-refinement runtime was reduced by about 54.21%.
- Point total runtime was reduced by about 43.50%.
- Mean estimated grid evaluations: baseline 3,523,900; rank_and_cap_k3
  1,961,300.

## Report-Only Collection Fix

The HPC collect step originally failed in `lr_rc_collect-73799.out` because
`pointwise_regression_comparison.csv` was written with a field list that did
not include the baseline/rankcap continuous-value columns.  During local
inspection a second collection-only issue was found: the loader matched both
`point_###.csv` and `point_###_local_box_timing.csv`, which polluted the
pointwise rows.  The report-only script was fixed to:

- include the missing continuous-value columns in the CSV field list;
- read only `point_###.csv` as regression point rows.

No production exact-oracle logic, physical criterion, tolerance, acquisition
function, or active-learning loop was changed.

## Output Paths

- Markdown report:
  `local_refinement_rankcap_acceptance_upload/local_refinement_rankcap_acceptance/local_refinement_rankcap_acceptance_run/reports/local_refinement_rankcap_acceptance/rankcap_acceptance_report.md`
- PDF report:
  `local_refinement_rankcap_acceptance_upload/local_refinement_rankcap_acceptance/local_refinement_rankcap_acceptance_run/reports/local_refinement_rankcap_acceptance/rankcap_acceptance_report.pdf`
- Tables:
  `local_refinement_rankcap_acceptance_upload/local_refinement_rankcap_acceptance/local_refinement_rankcap_acceptance_run/reports/local_refinement_rankcap_acceptance/tables/`
- Figures:
  `local_refinement_rankcap_acceptance_upload/local_refinement_rankcap_acceptance/local_refinement_rankcap_acceptance_run/reports/local_refinement_rankcap_acceptance/figures/`

## Interpretation

rank_and_cap_k3 passes the bounded 32 fixed-point acceptance test against the
baseline robust_incremental path.  This supports moving to a one-iteration AL
validation if the user wants to test the accepted rank-and-cap path in the
active-learning loop.  It does not automatically validate k2, energy-window,
branch reuse, Powell, GPU batching, Hamiltonian cache, mini AL, or full AL.
