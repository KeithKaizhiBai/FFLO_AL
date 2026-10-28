# Stage V-v2 Return Report Notes

## Context

The Stage V-v2 multi-head acquisition run returned from GBU-H100 and was
downloaded into the local project tree under:

```text
stagev_v2_multihead_boundary_learning_3d_v1_hpc/
```

Only the final dataset, final iteration folder, logs, and small supporting CSV
files were downloaded.  The full remote active-run directory was not copied.

## Key Results

```text
run_id = stagev_v2_multihead_boundary_learning_3d_v1
final_dataset = dataset_iter097
final_exact_iteration = iter096
total_samples = 7067
trusted_exact = 6928
normal = 4711
uniform_SC = 83
FFLO = 2273
cFFLO/trivial = 1892
tFFLO/topological = 381
```

The production loop completion line in the downloaded logs is:

```text
[done] Stage V-v2 production loop completed through exact iter 096
```

## Acquisition Learning Snapshot

The final reward update is iteration 096.  The final learned reward-model rank
correlations are approximately:

```text
ns = 0.7428
p0 = 0.6485
ppi = 0.7080
uf = 0.5211
gap = 0.0
```

The final acquisition lambda values are:

```text
p0 = 0.7
ppi = 0.7
ns = 0.0
uf = 0.0
gap = 0.0
```

The gap head remains `insufficient_data`, so it should be described as a
diagnostic channel rather than a proven learned gap acquisition channel.

## Report Artifacts

The local return report is:

```text
stagev_v2_multihead_boundary_learning_3d_v1_hpc/ML_Phase_StageV_V2_Multihead/reports/stagev_v2_return_report_local/stagev_v2_return_report.md
stagev_v2_multihead_boundary_learning_3d_v1_hpc/ML_Phase_StageV_V2_Multihead/reports/stagev_v2_return_report_local/stagev_v2_return_report.pdf
stagev_v2_multihead_boundary_learning_3d_v1_hpc/ML_Phase_StageV_V2_Multihead/reports/stagev_v2_return_report_local/decision_log.md
```

The report includes CSV tables and PNG figures under its `tables/` and
`figures/` subdirectories.

## Caveat

This report is a returned-data diagnostic.  It does not prove formal 3D
phase-boundary convergence.  A fixed-grid convergence, topology-support, and
boundary-surface audit is still needed before freezing final 3D boundary
surfaces.
