# Phase-II Final Audit Plan

Status: audit_planning

Date: 2026-06-20

## Scope

This plan covers one report-only workflow:

1. Complete a publication-grade hard-risk boundary-impact audit from existing
   tail-continuation artifacts.
2. If and only if that audit passes, build the self-contained Phase-II final
   report.

No full active-learning loop, Slurm submission, production exact-oracle change,
acquisition change, rankcap_k3 change, StopController change, phase-criterion
change, or tolerance change is allowed.

## Input Paths

```text
rankcap_k3_tail_surprise_continuation_results/ML_Phase_512_RankCapK3_TailContinuation/active_runs/active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/dataset_iter035.npz
rankcap_k3_tail_surprise_continuation_results/ML_Phase_512_RankCapK3_TailContinuation/active_runs/active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/iter034/exact_merged_iter034.npz
rankcap_k3_tail_surprise_continuation_results/ML_Phase_512_RankCapK3_TailContinuation/active_runs/active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/iter034/rerun_points.csv
rankcap_k3_tail_surprise_continuation_results/ML_Phase_512_RankCapK3_TailContinuation/active_runs/active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/iter034/selected_points_by_pool.csv
rankcap_k3_tail_surprise_continuation_results/ML_Phase_512_RankCapK3_TailContinuation/active_runs/active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/iter034/monitor_predictions_iter034.npz
rankcap_k3_tail_surprise_continuation_results/ML_Phase_512_RankCapK3_TailContinuation/active_runs/active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/iter034/stop_metrics_iter034.json
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/full_loop_enhanced_report/
reports/trusted_surprise_counterfactual/
reports/rankcap_k3_tail_surprise_continuation_return/
reports/hard_risk_boundary_impact_audit/
```

## Source Control Context

```text
git_commit = f3c277bb419ec4108b40ebba56635ef72c2ad895
working_tree_status = dirty; historical and generated files are present
```

## Gate

Part B starts only if Part A writes:

```text
publication_boundary_audit = pass
need_new_exact_calculation = False
targeted_rerun_count = 0
```

If Part A fails or cannot determine, the workflow stops after the audit and
does not claim publication readiness.
