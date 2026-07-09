# Rankcap K3 Full-Loop Convergence Check

Date: 2026-06-17

## Question

Did the rankcap_k3 full-loop result meet the active-learning convergence
criterion?

## Answer

No.  It passed the corrected rankcap_k3 full-loop validation, but it did not
meet the formal StopController convergence criterion.  The run stopped because
it reached the configured maximum iteration count.

Evidence from:

```text
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/active_runs/active_boundary_discovery_rankcap_k3_full_loop_v1/stop_state.json
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/active_runs/active_boundary_discovery_rankcap_k3_full_loop_v1/stop_metrics_history.json
```

Final stop state:

```text
last_iteration = 30
completed_iterations = 31
stop = true
stop_reason = max_iterations
convergence_pass = false
patience_counter = 0
patience = 4
required_pass_count = 4
passed_condition_count = 3
```

The final iteration passed the low-level phase-map stability criteria:

```text
C1_phase_map_change = true
C2_boundary_shift_normal_sc = true
C3_boundary_shift_uniform_fflo = true
```

It failed two convergence criteria:

```text
C4_label_surprise_rate = false
C5_boundary_coverage_p95 = false
```

Final metric values:

```text
phase_map_change = 0.0006204676775119246
map_tol = 0.002

boundary_shift_normal_sc = 0.002604166666666674
boundary_shift_tol = 0.004166666666666667

boundary_shift_uniform_fflo = 0.0
boundary_shift_tol = 0.004166666666666667

label_surprise_rate = 0.18359375
surprise_tol = 0.05

boundary_coverage_p95 = 0.006588078458684216
coverage_tol = 0.00625
```

Diagnostic criteria also show that acquisition had not become inactive:

```text
selected_A0_ratio = 0.90164743996301
selected_A0_ratio_tol = 0.15

q_edge_trigger_rate = 0.66015625
qedge_rate_tol = 0.01

rerun_required_rate = 0.36328125
rerun_rate_tol = 0.01
```

## Interpretation

The run is a successful optimization validation, not a fully converged active
learning run.  It demonstrates that rankcap_k3 can complete a full 30-batch
active-learning trajectory while preserving the corrected local-refinement
gate and reducing refinement cost.  However, the final selected batch still
contains many informative or difficult points: label surprise remains above
the convergence tolerance, the boundary coverage metric is slightly above its
tolerance, and diagnostic acquisition/rerun rates are still high.

For reports, use the wording:

```text
rankcap_k3 passed full-loop validation and produced a stable, usable phase
diagram, but the active-learning stop criterion was not formally satisfied;
the run stopped at max_iterations.
```

Do not write:

```text
the active-learning loop converged
```

unless a subsequent run stops with `stop_reason = convergence` or equivalent
StopController evidence.
