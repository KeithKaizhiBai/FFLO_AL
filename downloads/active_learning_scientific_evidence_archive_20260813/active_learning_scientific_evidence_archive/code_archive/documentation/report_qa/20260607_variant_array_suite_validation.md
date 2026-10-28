# Variant Array Suite Validation

Date: 2026-06-07

## Question

Why was the Stage 2-4 local-refinement validation redesigned after the serial
variant jobs timed out, and does the new design weaken the validation?

## Short Answer

The validation was not weakened.  The complete set of variants and fixed points
is still recomputed from scratch and compared against a freshly computed
baseline.  The redesign changes the HPC scheduling granularity: instead of one
long serial job per variant, the package now submits one Slurm array over
`variant x fixed_point_id`, writes one result per point task, and aggregates the
point outputs into the same variant-level comparison layout used by the
existing importer.

## What Failed Before

The previous package submitted one GPU job for each variant:

```text
baseline
cluster_only
cluster_optional_k3
cluster_optional_k2
cluster_energy_window
```

`baseline` and `cluster_only` completed in roughly two hours.  The three more
expensive variants ran for 36 hours and timed out.  Because each variant wrote
its pointwise CSV and manifest only after the whole 32-point loop completed,
the timed-out jobs left no useful point-level outputs.  The postprocess job was
submitted with `afterok`, so it became `DependencyNeverSatisfied` and did not
produce a return archive.

This was a checkpointing and scheduling problem, not evidence that the physics
gate was unnecessary.

## New Design

The new package keeps the same runnable variants and fixed-point set:

```text
5 variants x 32 fixed points = 160 array tasks
```

Each task computes exactly one pair:

```text
(variant, fixed_point_id)
```

and writes:

```text
reports/local_refinement_refactor/variant_regression/point_tasks/<variant>/point_XXX.csv
reports/local_refinement_refactor/variant_regression/point_tasks/<variant>/point_XXX.json
reports/local_refinement_refactor/variant_regression/point_tasks/<variant>/point_XXX_local_box_timing.csv
```

The aggregator then reconstructs:

```text
<variant>/<variant>_pointwise.csv
<variant>/<variant>_local_box_timing.csv
<variant>/regression_manifest.json
comparisons/baseline_vs_<variant>/
summary/task_status.csv
summary/missing_or_failed_tasks.csv
summary/equivalence_matrix.csv
summary/array_suite_status.json
performance_report/
decision_log.md
```

The postprocess job uses `afterany`, not `afterok`, so a diagnostic return
archive is produced even if some array tasks fail or time out.

## Validation Meaning

The formal comparison remains a physics-equivalence gate.  Candidate variants
are compared against the newly computed baseline on the same fixed points for:

```text
phase_candidate
q_opt
delta_opt
DeltaF
trusted_exact
training_eligible_exact
q_unresolved
delta_unresolved
rerun_required
local-refinement timing and diagnostic metadata
```

The default numerical thresholds remain:

```text
max_q_opt_abs_diff = 1e-10
max_delta_opt_abs_diff = 1e-10
max_deltaf_abs_diff = 1e-8
```

The redesign does not change physical definitions, numerical thresholds,
feature flags, q/delta safeguards, or the robust-oracle phase criterion.

## Practical HPC Expectations

The package defaults are:

```text
MAX_CONCURRENT = 32
POINT_TIME = 02:00:00
PARTITION = NV_H100
EXCLUDE_NODES = gpuh01
```

If the cluster grants enough concurrent GPU slots, the walltime should be
controlled by the slowest point tasks rather than by five serial 32-point
loops.  If a point is intrinsically slow, it will appear explicitly in
`summary/missing_or_failed_tasks.csv` instead of hiding inside a failed 36-hour
variant job.

## Report Interpretation

This change should be described as an engineering correction to the validation
workflow:

```text
We kept the complete exact-oracle equivalence target, but changed the
calculation from monolithic variant jobs to point-wise, restartable array
tasks.  This makes the gate auditable and resource-aware without relaxing the
scientific comparison.
```
