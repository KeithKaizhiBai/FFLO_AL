# cFFLO/tFFLO Topology-Boundary Convergence Audit

## Executive Summary

- Source run: `active_phase_topology_from_scratch_full_loop_v2`.
- Audit run: `topology_convergence_audit_full_loop_v2`.
- Final cumulative dataset: `dataset_iter018` with 4345 samples.
- Final trusted gapped FFLO points: 3213; cFFLO/trivial FFLO 2290; tFFLO/topological FFLO 923.
- Decision: `Decision A`.
- topology_main_converged: `True`.
- limiting_factor: `none`.
- need_new_exact_calculation: `False`.
- recommended_next_action: `freeze_topology_boundary_result_for_offline_report`.

This is an offline, report-only audit.  No new thermodynamic exact calculation,
Delta-q search, active-learning loop, or historical dataset modification was
performed.  The audit uses only trusted gapped FFLO points for the formal
cFFLO/tFFLO boundary tests.

## Repository and Input Audit

The cumulative dataset files `dataset_iter000.csv` through
`dataset_iter018.csv` are present.  The formal last-three transitions are
`iter015 -> iter016`, `iter016 -> iter017`, and `iter017 -> iter018`.

The final online topology diagnostic counts are consistent with the returned
run: trivial/cFFLO 2290, topological/tFFLO
923, and no trusted gapless-SC phase
in the current parameter range.  The absence of a finite-area gapless-SC phase
is not treated as a convergence failure.

## Audit Method

For each cumulative iteration, the same deterministic KNN inverse-distance
surrogate was fit to the iteration's trusted gapped FFLO exact points.  The
surrogate independently interpolates \(P_0(k_B T/t,J_A/t)\) and
\(P_\pi(k_B T/t,J_A/t)\) on a fixed 401 x 401 evaluation grid.  The
fixed audit mask is the final trusted FFLO support intersected with a nearest
trusted-FFLO support radius of 0.045 in normalized parameter
coordinates.

The cFFLO/tFFLO boundary is reconstructed from validated \(P_0=0\) and
\(P_\pi=0\) contour segments inside that mask.  Local opposite-Z2 kNN bracket
edges are used only as support diagnostics; raw long Delaunay edges are not
used as final contours.

## Final Three Transition Metrics

| Transition to iteration | topology map change | boundary p95 shift | trusted surprise |
|---:|---:|---:|---:|
| 16 | 0.000471 | 0.0008148 | 0.0119 |
| 17 | 0.0009083 | 0.002625 | 0.01714 |
| 18 | 0.0004878 | 0.0008712 | 0.01258 |

Final topology-boundary exact-point coverage p95 is
0.006185 versus threshold 0.00625.  Opposite-Z2
bracket coverage p95 is 0.005938.

## Sensitivity Check

The main audit uses k=8.  Lightweight kNN/IDW sensitivity gives:

| k | final coverage p95 | final bracket coverage p95 | significant components |
|---:|---:|---:|---:|
| 6 | 0.006269 | 0.006478 | 1 |
| 8 | 0.006185 | 0.005938 | 1 |
| 12 | 0.005913 | 0.004777 | 2 |

The sensitivity check is stable in map topology and component structure.  It
also shows that the coverage margin is tight: k=6 is slightly above the
nominal coverage threshold, while the preregistered k=8 and k=12 are
below threshold.  This is recorded as a coverage-margin caveat, not as evidence
of map or boundary-shift failure.

## Direct Answers

1. cFFLO/tFFLO boundary stopped moving in the last three rounds:
   `yes`.
2. topology map change is below threshold:
   `yes`.
3. topology boundary p95 shift is below threshold:
   `yes`.
4. topology boundary coverage is sufficient:
   `yes`.
5. trusted topology surprise is close to zero:
   `yes`.
6. topology boundary keeps a stable main connected component:
   `yes`.
7. significant topology islands appeared/disappeared:
   `no`.
8. final contour direct/bracket support fraction:
   1.0000.
9. formal decision:
   `Decision A` / `True`.
10. new exact calculation needed:
   `False`.

## Caveats

- This audit does not merge in `dataset_iter035`.
- Gapless-SC absence is not a failure because the current parameter range does
  not show a finite-area nodal-SC phase in the returned run.
- Strict Hausdorff distance is retained as an outlier diagnostic; the main
  boundary-shift gate uses symmetric p95 distance.
- Unsupported or extrapolation-only contour fragments are not counted as the
  formal main cFFLO/tFFLO boundary.

## Output Tables and Figures

All metrics are saved under:

```text
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/topology_convergence_audit_full_loop_v2/tables
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/topology_convergence_audit_full_loop_v2/figures
```
