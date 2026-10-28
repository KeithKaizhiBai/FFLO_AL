# Rankcap K3 Last-5 Stop-Failure Audit

Date: 2026-06-17

## Question

Why did the rankcap_k3 full-loop run fail formal StopController convergence
after 30 acquisition-selected batches, even though local refinement validation
passed and the final phase-map / boundary-shift metrics were stable?

## Short Answer

The last-five-iteration report-only audit supports the conclusion that the
remaining failure is a late-stage selection / stopping-metric problem, not a
rankcap_k3 local-refinement target-explosion problem.

The final StopController state was:

```text
stop_reason = max_iterations
convergence_pass = false
passed_condition_count = 3
required_pass_count = 4
```

The stable conditions passed:

```text
phase_map_change = 0.0006204676775119246 < 0.002
boundary_shift_normal_sc = 0.002604166666666674 < 0.004167
boundary_shift_uniform_fflo = 0.0
```

The failed conditions were:

```text
label_surprise_rate = 0.18359375 > 0.05
boundary_coverage_p95 = 0.006588078458684216 > 0.00625
```

## Evidence From Last Five Iterations

The audit covers acquisition iterations 26 through 30.  The StopController
label-surprise mismatch counts were:

```text
iter026: 37 / 256 = 0.14453125
iter027: 42 / 256 = 0.1640625
iter028: 34 / 256 = 0.1328125
iter029: 45 / 256 = 0.17578125
iter030: 47 / 256 = 0.18359375
```

The mismatch source is dominated by points predicted as normal before exact
evaluation but labeled as FFLO after exact evaluation under the StopController
phase-label rule based on final `Delta_opt` and `q_opt`.

Final iteration diagnostics:

```text
q_edge_trigger_rate = 0.66015625
rerun_required_rate = 0.36328125
selected_A0_ratio = 0.90164743996301
active_pool_size = 1417
active_pool_fraction = 0.020010450058605057
```

This means the late-stage acquisition is still selecting many high-risk,
q-edge-sensitive or rerun-required points, so the StopController sees persistent
label surprise even after the visible phase map and main boundaries are stable.

## Dataset Label Caveat

For the label-surprise audit, the correct comparison is not the dataset
`phase_label` appended after trust / eligibility filtering.  The StopController
uses an exact-batch phase label reconstructed from exact outputs.  Therefore the
audit computes:

```text
actual_label = phase_label(Delta_opt, q_opt, delta_eps, q_eps)
predicted_label = selected_points_by_pool.csv::predicted_phase_before_exact
```

This reproduces the StopController rates exactly for the last five iterations.

## Report Outputs

```text
rankcap_k3_full_loop/reports/rankcap_k3_last5_stop_audit/
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/last5_selection_stop_audit/
```

Key files:

```text
last5_selection_stop_audit.md
last5_selection_stop_audit.pdf
decision_log.md
tables/last5_selection_decomposition.csv
tables/last5_label_surprise_confusion.csv
tables/last5_stop_failure_root_cause.csv
figures/last5_failed_stop_metrics.png
figures/last5_selected_points_map.png
```

## Do Not Overclaim

Do not claim that rankcap_k3 caused the non-convergence.  The available evidence
supports that rankcap_k3 controlled local boxes and passed oracle validation.

Do not claim that the full loop formally converged.  It stopped by
`max_iterations`.

Do not claim that changing tolerances is required.  This audit did not test any
threshold change and intentionally avoided modifying StopController behavior.

Do not claim that the failure is purely physical complexity.  The audit shows a
specific acquisition / stop-metric pattern: predicted-normal to exact-FFLO
surprise remains high in the last selected batches.

## Recommended Next Step

If formal convergence is required, plan a separate late-stage cleanup validation
that targets the remaining high-surprise / boundary-coverage regime.  Do not
silently alter acquisition, StopController, thresholds, exact oracle, or
rankcap_k3 inside this audit.
