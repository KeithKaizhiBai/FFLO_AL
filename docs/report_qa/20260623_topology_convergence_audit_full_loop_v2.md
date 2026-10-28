# Stage III Topology Convergence Audit Notes

Date: 2026-06-23

Source run:

```text
active_phase_topology_from_scratch_full_loop_v2
```

Audit run:

```text
topology_convergence_audit_full_loop_v2
```

Input:

```text
dataset_iter000.csv ... dataset_iter018.csv
final dataset: dataset_iter018
final unique samples: 4345
trusted gapped FFLO points: 3213
cFFLO / trivial FFLO: 2290
tFFLO / topological FFLO: 923
```

The audit is report-only. It does not start new exact calculations, does not
run a new Delta-q search, does not continue active learning, does not modify
historical labels, and does not merge in `dataset_iter035`.

Method:

```text
For each cumulative iteration, fit the same deterministic KNN inverse-distance
surrogate to P0 and Ppi using only that iteration's trusted gapped FFLO exact
points.  Predict on a fixed 401 x 401 normalized grid and fixed final trusted
FFLO support mask.  Extract cFFLO/tFFLO contours from validated P0=0 and Ppi=0
segments.  Use local opposite-Z2 kNN brackets as support diagnostics, not as
the final contour.
```

Main metrics:

```text
last3 topology_map_change:
    0.000471, 0.000908, 0.000488  < 0.002

last3 topology_boundary_shift_p95:
    0.000815, 0.002625, 0.000871  <= 0.004167

final topology_boundary_coverage_p95:
    0.006185  < 0.00625

last3 trusted_topology_surprise:
    0.0119, 0.0171, 0.0126  <= 0.02

significant boundary component count last3:
    1, 1, 1

topological-region component count last3:
    1, 1, 1

final direct/bracket support fraction:
    1.0
```

Decision:

```text
Decision A
topology_main_converged = true
need_new_exact_calculation = false
recommended_next_action = freeze_topology_boundary_result_for_offline_report
```

Sensitivity caveat:

```text
The main configured k=8 audit passes coverage.  k=6 gives final coverage p95
= 0.006269, slightly above the nominal 0.00625 threshold; k=12 gives 0.005913.
The map and component topology remain stable.  Treat this as a tight
coverage-margin caveat, not a map-shift or surprise failure.
```

Do-not-claim:

```text
Do not claim raw Delaunay opposite-label long edges are the topology boundary.
Do not call a missing topology boundary shift zero.
Do not treat no finite-area gapless-SC phase as failure.
Do not merge this cold-start run with dataset_iter035.
Do not claim all possible topology coverage refinements are exhausted; the main
boundary passes under the configured audit, but k sensitivity shows a tight
coverage margin.
```

Output:

```text
active_phase_topology_from_scratch_full_loop_v1_hpc/
  ML_Phase_512_TopoTrivial_FullLoop/
    reports/topology_convergence_audit_full_loop_v2/
```

Re-run validation on 2026-06-23:

```text
python -m py_compile scripts/build_topology_convergence_audit.py
python scripts/build_topology_convergence_audit.py
pdflatex -interaction=nonstopmode topology_convergence_audit_report.tex

Result:
    audit JSON reproduced the Decision A metrics above;
    topology_convergence_audit_report.pdf compiled successfully;
    PDF length = 8 pages.
```
