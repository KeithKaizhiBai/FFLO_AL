# Local Refinement Refactor Call Graph

Date: 2026-06-03

The current local-refinement path is inside `ml_phase/exact_oracle.py`.

```text
evaluate_points
  -> _confirm_one_point
      -> _confirm_one_point_robust
          -> _run_scan_with_normal
          -> _diagnose_q_window
          -> _select_expansion_direction
          -> _expand_cfg_keep_density
          -> _run_scan_for_q_vec_with_normal      [robust_incremental strip path]
          -> _merge_q_scan_caches                 [robust_incremental strip path]
          -> _build_branch_candidates
              -> _local_minima_indices
          -> local refinement loop
              -> _run_scan_with_normal            [local box]
          -> positive_delta_config                [Delta guardrail]
          -> _write_branch_candidates_csv
```

Stage 0 does not change this call graph.

Future Stage 1 instrumentation should attach box-level records around the
`local refinement loop` without changing which boxes are refined.

