# q-window incremental call graph

Date: 2026-06-02

## Entrypoints

| Entry point | Caller | Output | Notes |
|---|---|---|---|
| `ml_phase.exact_oracle.main` | Slurm exact array or local script | exact shard `.npz` and `.json` | Adds `--oracle-mode robust_incremental` and `--enable-incremental-q-expansion`. |
| `evaluate_points` | CLI, regression scripts, benchmarks | `OracleResult` | Loops over selected points and flushes partial shards. |
| `_confirm_one_point` | `evaluate_points` | `ConfirmedPoint` | Dispatches legacy, `robust_al`, or `robust_incremental`. |

## Robust baseline path

| Function | Input | Output | Current recomputation behavior |
|---|---|---|---|
| `_confirm_one_point_robust` | point, `EtaPhaseConfig`, tolerances | final `ConfirmedPoint` | In `robust_al`, each expansion calls `_run_scan_with_normal` over the full expanded q-window. |
| `_run_scan_with_normal` | point, config | `ScanResult` | Computes SC scan and normal scan. |
| `compute_omega_min_q_batch` | batches, q grid, Delta grid | \(F_{\min}(q)\), \(\Delta_\star(q)\) | Dominant free-energy grid cost. |
| `_build_branch_candidates` | final `ScanResult` | local minima table | Reads final \(F_{\min}(q)\); no new free-energy work. |
| local refinement loop | branch candidates | refined candidates | Re-scans local q-Delta boxes. |
| Delta guardrail | tolerance-sensitive point | positive-Delta scan | Re-scans positive-Delta region. |

## Incremental q-expansion path

| Function | Input | Output | Behavior |
|---|---|---|---|
| `_incremental_q_strips` | previous `ScanResult`, expanded config | left/right q arrays | Builds only newly exposed q points at preserved density. |
| `_run_scan_for_q_vec_with_normal` | explicit q array, config, optional normal scalar | `ScanResult` | Scans arbitrary q arrays. If normal scalar is supplied, skips repeated normal scan. |
| `_scan_to_cache` | `ScanResult` | `QScanCache` | Converts a scan into mergeable arrays. |
| `_merge_q_scan_caches` | cache list | merged `QScanCache` | Sorts q points and de-duplicates overlaps by lower \(\Delta F\). |
| `_cache_to_scan` | merged cache | `ScanResult` | Reconstructs global minimum and edge metadata. |

## Fallback

If the incremental strip cannot be formed, `_confirm_one_point_robust` falls
back to `_run_scan_with_normal` over the full expanded q-window and records:

- `fallback_full_rescan_used = 1`
- `fallback_full_rescan_reason = no_incremental_strip`

No silent fallback is allowed.

## Saved outputs

The exact shard `.npz` now includes per-point timing and workload counters.
The shard `.json` includes rank-level sums for walltime attribution.
