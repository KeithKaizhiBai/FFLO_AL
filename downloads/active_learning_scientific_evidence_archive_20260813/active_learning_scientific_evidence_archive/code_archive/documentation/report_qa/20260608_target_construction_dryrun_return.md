# Target-Construction Dry-Run Return Check

Date: 2026-06-08

Question: What did the returned 32 fixed-point target-construction-only dry run show?

Answer:

- The HPC return is complete: 32/32 points completed and the aggregate gate status is `pass`.
- The run did not execute local-box scans. Each point JSON records `local_box_scan: not_run`.
- Rank-and-cap variants selected at most 3 targets for every point, with mean selected target count 2.75.
- Baseline and cluster-only selected 6 targets for every point.
- The upstream mandatory-basin population remains broad: clustered variants still have mean mandatory basin count 542.031.
- The selected rank-and-cap targets per variant are 32 global-best, 32 Delta-near-epsilon, and 24 edge-risk targets.
- Energy-window pruning had no effect because `ordinary_count_before_energy_window` was zero for all 32 points.
- This confirms the target-count gate but does not confirm local-box physics equivalence.

Report artifacts:

- `reports/local_refinement_refactor/stage_04_target_construction_dryrun_return/target_construction_dryrun_return_check.md`
- `reports/local_refinement_refactor/stage_04_target_construction_dryrun_return/decision_log.md`
- `reports/local_refinement_refactor/stage_04_target_construction_dryrun_return/tables/`
- `reports/local_refinement_refactor/stage_04_target_construction_dryrun_return/figures/`

Recommended next step:

Run bounded local-box fixed-point regression for the rank-and-cap variants. Do not rerun the old 72 timeout tasks from the pre-fix variant suite.
