# Stage V-v2 Formal Audit Notes

## Scope

The shared prompt `shared_prompts/inbox/Stage_v2_Audit_prompt.md` was copied to
`shared_prompts/active/CURRENT_TASK_PROMPT.md` and executed as a report-only
audit.

No exact calculation, Delta-q search, active-learning loop, dataset mutation, or
HPC job was run.

## Input

```text
run_id = stagev_v2_multihead_boundary_learning_3d_v1
final_dataset = dataset_iter097
final_exact_iteration = 96
total_samples = 7067
trusted_exact = 6928
normal = 4711
uniform_SC = 83
FFLO = 2273
cFFLO/trivial = 1892
tFFLO/topological = 381
```

## Audit Method

The audit used downloaded cumulative CSV files from `dataset_iter000.csv`
through `dataset_iter097.csv`.  Since online surrogate checkpoints were not
available locally, all iteration comparisons used one consistent offline KNN
exact-label surrogate on a fixed 200000-point 3D audit cloud.

Boundary surfaces were represented by local kNN opposite-label bracket
midpoints, not raw global Delaunay long edges.

## Decision

```text
decision_class = C
stagev_v2_converged = false
main_limiting_factor = topology_surface_coverage
need_new_exact_calculation = true
recommended_tail_batches = 4
recommended_next_action = targeted_topology_surface_coverage_tail_or_acquisition_v3_support_fix
```

## Key Metrics

```text
phase_map_change_last3 = [0.000315, 0.000525, 0.000185]
topology_map_change_last3 = [1.044e-05, 0.000125, 0.000209]
topology_surface_shift_p95_last3 = [0.0, 0.0, 0.0]
topology_surface_coverage_p95_final = 0.036647
trusted_topology_surprise_last3 = [0.0, 0.0, 0.0]
topology_component_count_last3 = [3, 3, 3]
```

## Interpretation

Stage V-v2 completed production and achieved stable phase/topology map proxies
in the final transitions.  However, the final topology-surface support coverage
is not sufficient for a formal convergence claim.  The result should not be
frozen as a final 3D cFFLO/tFFLO surface.  The appropriate next step is a short,
targeted topology-surface coverage tail or an acquisition-v3 correction focused
on support coverage.

## Output

```text
stagev_v2_multihead_boundary_learning_3d_v1_hpc/ML_Phase_StageV_V2_Multihead/reports/stagev_v2_formal_audit_local/
```
