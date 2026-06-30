# 2026-06-08 Variant Array Return Report Notes

## Question

The point-wise Stage 2-4 variant-suite result returned under
`local_refinement_refactor_hpc_upload_set`.  What did the calculation prove,
what failed, and how should it be reported?

## Short Answer

The returned suite produced a useful diagnostic result but did not pass the
full Stage 2-4 gate.  The package completed 88 of 160 point tasks.  The
fresh `baseline` and `cluster_only` variants both completed all 32 fixed
points and are exactly equivalent on the recorded comparison metrics.  The
three more aggressive variants, `cluster_optional_k3`,
`cluster_optional_k2`, and `cluster_energy_window`, completed only 8 of 32
points each and timed out on the remaining 24 points each.

The completed rows for the optimized variants are numerically equivalent to
the baseline on the eight common completed points.  This is not enough to
accept the optimized-variant gate because those eight points are only the clean
superconducting controls.  The hard categories, including normal-boundary
bands, near-degenerate or Delta-ambiguous points, previous normal-to-FFLO
corrections, q-edge-risk points, rerun-required points, and stable-normal
interior points, all remain unresolved for the optimized variants.

## Evidence

The return status checker reported:

```text
status = ready_to_return_with_validation_failures
successful_tasks = 88 / 160
array_suite_summary.status = fail
task_status = fail
comparison_status = fail
performance_status = pass
```

Per-variant completion:

```text
baseline:              32 / 32 success
cluster_only:          32 / 32 success
cluster_optional_k3:    8 / 32 success, 24 timeout
cluster_optional_k2:    8 / 32 success, 24 timeout
cluster_energy_window:  8 / 32 success, 24 timeout
```

Equivalence:

```text
cluster_only vs baseline:
    common = 32
    missing = 0
    flag mismatches = 0
    max q_opt, Delta_opt, DeltaF differences = 0
    status = pass

optimized variants vs baseline:
    common = 8
    missing = 24 each
    flag mismatches on common rows = 0
    max q_opt, Delta_opt, DeltaF differences on common rows = 0
    status = fail because hard points are missing
```

Performance on the eight points completed by every variant:

```text
baseline mean runtime:              about 3.37 min
cluster_only mean runtime:          about 3.35 min
cluster_optional_k3 mean runtime:   about 45.0 min
cluster_optional_k2 mean runtime:   about 45.0 min
cluster_energy_window mean runtime: about 44.9 min
```

The optimized variants used about 85 local boxes on these completed clean
points, while baseline and `cluster_only` used about 6 local boxes.  Therefore
the intended optimizations currently trigger a much broader local-refinement
search and are roughly 13 times slower on the common completed controls.

## Interpretation

The important accepted result is narrow but useful:

```text
cluster_only can be accepted as physics-equivalent to the fresh baseline on
the full 32-point fixed-point panel.
```

The important rejected result is:

```text
cluster_optional_k3, cluster_optional_k2, and cluster_energy_window cannot be
accepted.  They have not demonstrated equivalence on the hard categories and
show a major performance regression on the completed controls.
```

The failed tasks appear to be Slurm time-limit terminations.  The diagnostic
logs contain time-limit cancellation messages, and the stale JSON rows that
still say `running` are best interpreted as timed-out jobs rather than live
calculations.

## Reporting Use

This result should be presented as a local-refinement engineering audit rather
than a final optimized-algorithm success.  A compact report should show:

1. A task-outcome heatmap by variant and fixed-point id.
2. An equivalence table separating common completed rows from missing rows.
3. A fair runtime comparison on the eight common completed points.
4. A local-box-count comparison explaining why the optimized variants were
   slower.
5. A risk-category success heatmap showing that only clean superconducting
   controls completed for the optimized variants.

The generated report is:

```text
project_history/reports/report_local_refinement_variant_array_return_20260608/
```

## Next Calculation

Do not rerun `baseline` or `cluster_only` for this gate unless the fixed-point
panel changes.  The next useful calculation is either:

```text
rerun only the 72 missing optimized-variant tasks with corrected checkpoint
and time-limit strategy
```

or:

```text
run a smaller diagnostic subset that records why the optimized variants select
about 85 local boxes where baseline selects about 6
```

The second option is scientifically preferable before another full rerun,
because the current bottleneck is algorithmic over-expansion rather than lack
of scheduler wall time alone.
