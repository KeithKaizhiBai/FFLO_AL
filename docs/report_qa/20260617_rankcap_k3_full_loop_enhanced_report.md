# Rankcap K3 Full-Loop Enhanced Report

Date: 2026-06-17

## Question

The returned full-loop package reports:

```text
validation_status = fail
max_local_boxes_refined_count = 24
```

Does this mean the rank_and_cap_k3 full active-learning run failed?

## Answer

No.  The numerical full-loop run completed, and the corrected rank-level
validation passes.

The full-loop package repeats the same report aggregation issue found in the
five-iteration validation package: local-box timing files are rank-level files,
but the package collector drops the rank dimension and groups by
`(iteration, point_id)`.  Since `point_id` is rank-local, rows from different
ranks are incorrectly merged.  This creates the spurious maximum:

```text
8 ranks * 3 boxes = 24
```

Recomputing from the raw files:

```text
iterXXX_local_box_timing_rankYYY_of008.csv
```

with the key:

```text
(iteration, rank, point_id)
```

gives:

```text
corrected_validation_status = pass
actual exact points = 8192
corrected max local boxes = 3
points above 3 boxes = 0
mean local boxes, unweighted by iteration = 2.79296875
mean local boxes, weighted by point = 2.7796630859375
```

## Evidence

Enhanced report artifacts:

```text
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/rankcap_k3_full_loop_enhanced.md
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/rankcap_k3_full_loop_enhanced.pdf
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/full_loop_enhanced_report/
```

Key tables:

```text
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/tables/enhanced_corrected_validation_summary.csv
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/tables/enhanced_corrected_validation_gates.csv
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/tables/enhanced_iteration_recheck.csv
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/tables/enhanced_actual_local_box_point_counts.csv
```

Final dataset:

```text
samples = 6880
normal = 1777
uniform_SC = 715
FFLO = 4388
```

Runtime summary:

```text
estimated total wall time = 36.5713 hours
exact iterations = 31
acquisition-selected batches = 30
mean wall time per exact iteration = 70.7832 min
mean wall time per acquisition batch = 73.1426 min
summed exact-oracle wall time = 34.6863 hours
mean exact-oracle wall time per acquisition iteration = 65.5315 min
mean local-refinement runtime = 88.2856 sec/point
mean point-total runtime = 117.285 sec/point
historical robust-incremental references in package:
    local boxes = 6.0
    local-refinement = 189.767 sec/point
    point-total = 234.194 sec/point
rankcap_k3:
    mean local boxes = 2.79296875
    local-box reduction = 53.4505%
    local-refinement runtime reduction = 53.4768%
    point-total runtime reduction = 49.9199%
```

The total wall time is estimated from the active-loop lock timestamp
`2026-06-10T12:14:54+08:00` to the collector summary timestamp
`2026-06-12T00:49:10.755526+08:00`.  The exact-oracle wall time is computed
from per-iteration maximum rank elapsed time, so it is the better estimate of
GPU-bound scientific work.

## Figures Added

The enhanced report includes:

```text
enhanced_final_phase_diagram.png
enhanced_phase_snapshots.png
enhanced_learning_curve_phase_counts.png
enhanced_training_rerun_curve.png
enhanced_corrected_local_box_gate.png
enhanced_local_box_distribution.png
enhanced_runtime_curve.png
enhanced_rank_runtime_imbalance.png
enhanced_recent_selected_overlay.png
enhanced_exact_eta_revised_boundaries.png
enhanced_exact_walltime_curve.png
enhanced_surrogate_metric_curves.png
enhanced_phase_accuracy_reduction_curve.png
```

## LaTeX Report and ML Curves

The enhanced report is now generated as LaTeX in addition to Markdown:

```text
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/rankcap_k3_full_loop_enhanced.tex
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/rankcap_k3_full_loop_enhanced.pdf
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/full_loop_enhanced_report/rankcap_k3_full_loop_enhanced.tex
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/full_loop_enhanced_report/rankcap_k3_full_loop_enhanced.pdf
```

The package does not store raw per-epoch training loss.  The report therefore
does not claim a training-loss curve.  It plots the available validation-style
surrogate metrics from `metrics_history.json`, including `delta_rmse`,
`q_rmse`, `eta_rmse`, `ic_plus_rmse`, `ic_minus_rmse`, phase accuracy, and
estimated reduction.  Final recorded surrogate metrics are:

```text
metric iteration = 29
delta_rmse = 0.010592307301708126
q_rmse = 0.3707884955840742
eta_rmse = 0.06503987569672262
phase_accuracy = 0.9995535714285714
estimated_reduction = 11.512053571428572
n_exact_calls = 6720.0
```

PDF QA:

```text
pdflatex completed successfully.
pdftoppm rendered the LaTeX PDF successfully.
Visual checks of representative rendered pages confirmed that the summary
tables, phase diagrams, runtime curves, local-box gate, and surrogate metric
curves are present and readable.
```

## Consequence

The full-loop rank_and_cap_k3 result can be used as the completed optimization
evidence for the second-stage report.  The package collector should still be
patched before future runs so merged local-box timing tables preserve rank or
use globally unique point identifiers.

This note does not change the Hamiltonian, phase criterion, acquisition,
StopController, Delta tolerance, final ambiguity tolerance, or exact oracle.
