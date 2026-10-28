# Stage V-v4 Combined Report and Formal Audit

Date: 2026-07-12

## Question

Did the completed Stage V-v4 phase-structure-graph acquisition run converge,
and did it establish an improvement over Stage V-v3?

## Short Answer

No.  Stage V-v4 completed normally and its thermodynamic/topology maps are
stable under the fixed-cloud audit proxy, but topology-surface coverage p95 is
0.020982 versus the 0.00625 gate and trusted topology surprise remains above
0.02.  The formal result is Decision C.  V4 demonstrates an operational graph
model and provisional candidate-node discovery, but a matched V3/V4 audit was
not possible from the downloaded V4-only dataset, so superiority is not
established.

## Main Results

```text
final_dataset = dataset_iter097
total_samples = 7080
trusted_exact = 6727
normal / uniform_SC / FFLO = 3245 / 232 / 3603
trusted_cFFLO / trusted_tFFLO = 2648 / 955
q_unresolved / Delta_unresolved / rerun_required = 0 / 0 / 0
phase_map_change_last3 = [0.00038, 0.00060, 0.00054]
topology_map_change_last3 = [0.0006986, 0.0006986, 0.0005753]
topology_surface_shift_p95_last3 = [0.0, 0.0, 0.0]
topology_surface_coverage_p95_final = 0.0209824344524772
trusted_topology_surprise_last3 = [0.1290323, 0.0303030, 0.0294118]
topology_component_count_last3 = [5, 5, 5]
```

The graph-value model ends with lambda 1.2 and improves reward rank
correlation from 0.427 for the graph prior to 0.514.  The final graph has six
known nodes, four provisional candidate nodes, and twenty edges.  These are
method-level results, not new physical phase claims.

## Logging Caveat

Every reward row has nonblank proposal-source and action-family metadata, but
4362 rows retain stale `source_not_in_allowlist` text from the inherited V3
validator for sources that are valid in V4.  This does not invalidate exact
labels; it should be repaired before another production run.

## Report Artifact

```text
output/pdf/stagev_v4_combined_report_audit/
```

The folder contains the synchronized PDF, LaTeX, Markdown, sixteen CSV tables,
nine PNG figures, JSON decision/summary files, and `decision_log.md`.
