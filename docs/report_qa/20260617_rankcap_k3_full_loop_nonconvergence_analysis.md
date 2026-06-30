# Rankcap K3 Full-Loop Non-Convergence Analysis

Date: 2026-06-17

## Question

Why did the rankcap_k3 full loop not satisfy the formal StopController
convergence criterion, even though an earlier discovery run stopped within
about 20 iterations?

## Short Answer

The rankcap_k3 run passed full-loop validation, but it did not satisfy formal
active-learning convergence because two stop conditions remained false at the
final iteration:

```text
C4_label_surprise_rate = false
C5_boundary_coverage_p95 = false
```

It stopped at the hard iteration limit:

```text
stop_reason = max_iterations
convergence_pass = false
passed_condition_count = 3
required_pass_count = 4
```

The earlier 20-iteration run is not a like-for-like comparison.  It used an
older q-delta discovery workflow and stopped once the main boundary conditions
were stable enough for four consecutive checks.  It still had a false label
surprise condition at its final iteration.

## Direct Comparison

Earlier converged run:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/
ML_Phase_512_seed_v1/active_runs/active_boundary_discovery_512seed_256x50
```

Final state:

```text
last_iteration = 19
completed_iterations = 20
stop_reason = converged_main_phase_boundaries
patience_counter = 4
passed_condition_count = 4
required_pass_count = 4
```

Final stop metrics:

```text
phase_map_change = 0.000814363826734401
boundary_shift_normal_sc = 0.0026041666666667073
boundary_shift_uniform_fflo = 0.0
label_surprise_rate = 0.0625
boundary_coverage_p95 = 0.0044500019506861985
q_edge_trigger_rate = 0.0078125
rerun_required_rate = 0.1015625
```

Current rankcap_k3 full-loop run:

```text
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/
active_runs/active_boundary_discovery_rankcap_k3_full_loop_v1
```

Final state:

```text
last_iteration = 30
completed_iterations = 31
stop_reason = max_iterations
patience_counter = 0
passed_condition_count = 3
required_pass_count = 4
```

Final stop metrics:

```text
phase_map_change = 0.0006204676775119246
boundary_shift_normal_sc = 0.002604166666666674
boundary_shift_uniform_fflo = 0.0
label_surprise_rate = 0.18359375
boundary_coverage_p95 = 0.006588078458684216
q_edge_trigger_rate = 0.66015625
rerun_required_rate = 0.36328125
```

## Main Causes

1. The phase map itself was already stable.

   The rankcap_k3 run passed:

   ```text
   C1_phase_map_change
   C2_boundary_shift_normal_sc
   C3_boundary_shift_uniform_fflo
   ```

   Thus the non-convergence is not because the coarse phase diagram is
   oscillating visibly.

2. Label surprise remained high.

   The final label surprise was:

   ```text
   label_surprise_rate = 0.18359375
   surprise_tol = 0.05
   ```

   This is far above tolerance.  The selected batch was still producing exact
   labels that the current model did not predict with enough stability.

3. Boundary coverage narrowly missed the threshold.

   The final coverage metric was:

   ```text
   boundary_coverage_p95 = 0.006588078458684216
   coverage_tol = 0.00625
   ```

   This is a small but formal failure.  If this condition had passed, the run
   would still have failed label surprise, but it would have reached the 4/5
   main-condition count.

4. The new robust incremental oracle exposes many q-expansion / hard-risk
   points.

   Dataset-level q-expansion increased strongly:

   ```text
   old 20-iteration run: q_expanded = 79 / 5107
   rankcap_k3 full loop: q_expanded = 2724 / 6880
   ```

   The final StopController diagnostics also show:

   ```text
   q_edge_trigger_rate = 0.66015625
   rerun_required_rate = 0.36328125
   ```

   These diagnostic rates are not part of the 5 main conditions in this
   implementation, but they show that the selected batches remained dominated
   by numerically hard or high-information regions.

5. The full acquisition profile kept exploring hard FFLO/q-risk regions.

   The current run used:

   ```text
   acquisition_profile = full
   oracle_mode = robust_incremental
   rankcap_k3 local refinement
   ```

   The final dataset became FFLO-heavy:

   ```text
   old final: normal=1609, uniform_SC=648, FFLO=2850, total=5107
   new final: normal=1777, uniform_SC=715, FFLO=4388, total=6880
   ```

   This is scientifically useful for discovery, but it keeps selecting
   difficult FFLO/q-boundary points and delays formal StopController
   convergence.

## Interpretation for Reports

The correct wording is:

```text
rankcap_k3 passed full-loop validation and produced a stable phase diagram with
substantially reduced local-refinement cost.  It did not formally converge
under the current StopController; the run stopped at max_iterations because
label surprise remained high and boundary coverage slightly missed tolerance.
```

Avoid:

```text
rankcap_k3 active learning converged
```

## Practical Next Direction

Do not treat this as evidence that rankcap_k3 failed.  The bottleneck is now
active-learning stop/acquisition behavior, not local refinement cost.

Likely next steps:

```text
1. Keep rankcap_k3 as the accepted local-refinement backend.
2. Add or test a late-stage boundary-coverage cleanup mode after discovery.
3. Separate "phase-map stable" from "all hard-risk diagnostics quiet" in the
   report.
4. If formal convergence is required, run a targeted tail stage that directly
   reduces label surprise and boundary_coverage_p95 instead of simply extending
   full acquisition unchanged.
```
