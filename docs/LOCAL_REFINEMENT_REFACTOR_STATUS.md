# Local Refinement Refactor Status

Date: 2026-06-03

```text
Stage 0: completed
Stage 1: completed
Stage 2: local-minimal-complete-gpu-variant-pending
Stage 3: local-minimal-complete-gpu-variant-pending
Stage 4: local-minimal-complete-gpu-variant-pending
Stage 5: prototype-local-complete-integration-pending
Stage 6: skeleton-local-complete-integration-pending
Stage 7: variant-runner-and-package-local-complete-gpu-variant-pending
```

## Current Stage

Stage 7 is the GPU batching, Hamiltonian cache, and HPC packaging planning
stage.  Stages 0 and 1 have passed the
returned target GPU/CUDA fixed-point baseline-vs-instrumented gate.  Stages 2,
3, and 4 are locally minimal-complete with synthetic tests, but none has yet
been run as a GPU fixed-point variant.  Stage 5 has local decision primitives
and tests, but is not integrated into the production local-refinement loop.
Stage 6 has local geometry/proxy helpers and tests, but adaptive boxes are not
used by the production local-refinement loop.  Stage 7 variant-runner and
combined package generation are locally complete; GPU fixed-point regressions
for the Stage 2/3/4 variants remain pending.

## Stage 0 Deliverables

- Baseline documentation.
- Fixed-point regression point builder.
- Fixed-point regression runner scaffold.
- Variant comparison scaffold.
- Report-template path guardrails.
- Stage 0 report directory with plan, summaries, and decision log.

## Stage 0 Local Status

Completed locally:

- baseline documentation;
- report-template path guardrails;
- fixed-point regression point builder;
- regression runner dry-run scaffold;
- variant comparison scaffold;
- fixed-point CSV with 32 points across 8 categories;
- local py_compile and pytest checks.

Completed after HPC return:

- exact fixed-point regression on the intended GPU/CUDA environment, as part
  of the imported Stage 1 baseline-vs-instrumented gate.

## Stage 1 Plan Status

Completed:

- Stage 1 instrumentation plan;
- Stage 1 decision log;
- local-box timing schema table;
- local-refinement point-summary schema table.

Completed locally:

- `--enable-local-box-instrumentation` feature flag;
- optional `--local-box-output-file`;
- local-box CSV writer;
- local-refinement summary JSON writer;
- Slurm environment pass-through flag;
- fixed-point regression runner support;
- local-box instrumentation tests.

Prepared for HPC:

- `scripts/package_local_refinement_refactor_hpc.py`;
- `scripts/verify_local_refinement_stage1_gate.py`;
- `scripts/collect_local_refinement_stage1_outputs.py`;
- `scripts/import_local_refinement_stage1_results.py`;
- `scripts/validate_local_refinement_hpc_package.py`;
- `scripts/preflight_local_refinement_stage1_hpc.py`;
- `hpc_packages/local_refinement_refactor_stage01_instrumentation/`;
- `hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz`.
- `hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.sha256`;
- `hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.metadata.json`.

The package runs only fixed-point regression jobs.  It does not run active
learning and does not append training data.
It now includes `scripts/submit_stage1_regression_workflow.sh`, which submits
the Stage 0 baseline job, Stage 1 instrumented job, and an `afterok` dependent
postprocess job for compare, verify, and collect.
The workflow submitter first runs
`scripts/preflight_local_refinement_stage1_hpc.py` inside the extracted package
to check the fixed-point CSV, manifest, key scripts, syntax, and torch/CUDA
visibility snapshot before submitting Slurm jobs.

Stage 1 gate evidence:

- The returned RUN_ROOT archive was downloaded under
  `local_refinement_refactor_stage01_instrumentation/local_refinement_refactor_stage1_run/`
  and imported locally under
  `local_refinement_refactor_stage01_instrumentation/imported_results/stage1_regression_results/`.
- The imported `stage1_gate_status.json` reports `status=pass` with 32
  baseline rows, 32 instrumented rows, 32 comparison rows, 192 local-box rows,
  zero flag mismatches, and zero q/Delta/DeltaF differences.
- The HPC package now includes `scripts/collect_stage1_regression_outputs.sh`,
  which calls the Python collector and bundles logs, regression outputs,
  comparison outputs, gate status, and missing-path metadata for return.
- Returned bundles should be imported locally with
  `scripts/import_local_refinement_stage1_results.py`, which extracts into a
  staging directory and reruns the same artifact-level gate verifier without
  overwriting repository-root files.
- The current Stage 1 HPC package directory and archive pass
  `scripts/validate_local_refinement_hpc_package.py`.
- The package archive has SHA256 and metadata sidecars.  The validator checks
  that both sidecars match the rebuilt tarball before upload.
- The extracted package passes the Stage 1 runtime preflight locally.  This is
  a package/runtime sanity check only; it is not a substitute for the target
  GPU/CUDA regression gate.

Completed:

- exact Stage 0 baseline fixed-point regression on the intended GPU/CUDA
  environment;
- exact Stage 1 instrumented fixed-point regression on the same fixed-point
  set;
- baseline-vs-instrumented comparison with zero mismatch in
  `phase_candidate`, `trusted_exact`, `training_eligible_exact`,
  `q_unresolved`, `delta_unresolved`, and `rerun_required`.

## Stage 2 Plan Status

Completed locally:

- small task decomposition for basin identity, mandatory-risk preservation,
  representative metadata, point-level counters, and disabled-by-default
  feature-flag integration;
- pure `cluster_branch_candidates(...)` helper;
- branch-candidate metadata fields for basin id, cluster size, merged branch
  ids, cluster reason, and mandatory basin reasons;
- point-level metadata fields for clustered basin count, selected refine target
  count, clustering enabled, and merged duplicate count;
- exact-oracle CLI feature flags for future explicit Stage 2 variants;
- synthetic basin-clustering tests.

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py scripts\run_local_refinement_fixed_point_regression.py tests\test_basin_clustering.py
python -m pytest tests\test_basin_clustering.py tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_stage1_gate.py -q
python -m pytest tests -q

Result: full local tests reported 23 passed.
```

Pending:

- GPU fixed-point regression for an explicit Stage 2 clustering variant;
- GPU fixed-point regression for an explicit Stage 3 selective-refinement
  variant;
- GPU fixed-point regression for an explicit Stage 4 energy-pruning variant;
- Stage 5 branch-reuse integration after explicit reuse diagnostics are
  designed;
- Stage 6 adaptive-box production integration after GPU regression design;
- combined HPC package set generation after variant-runner support;
- combined later-stage HPC package set after the remaining local stages are
  implemented and minimally tested.

## Stage 3 Plan Status

Completed locally:

- small task decomposition for legacy rule preservation, mandatory basin
  definition, selective ordinary-cap policy, and default-off integration;
- pure `select_local_refine_targets(...)` helper;
- exact-oracle CLI feature flags for future explicit Stage 3 variants;
- synthetic tests for legacy cap behavior, mandatory keep above cap, optional
  ordinary cap, and strict total-cap mode.

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py tests\test_basin_clustering.py tests\test_selective_refinement.py
python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_stage1_gate.py -q
python -m pytest tests -q

Result: full local tests reported 26 passed.
```

Pending:

- GPU fixed-point regression for an explicit Stage 3 selective-refinement
  variant;
- GPU fixed-point regression for an explicit Stage 4 energy-pruning variant;
- Stage 5 branch-reuse integration after explicit reuse diagnostics are
  designed;
- Stage 6 adaptive-box production integration after GPU regression design;
- combined HPC package set generation after variant-runner support;
- combined later-stage HPC package set after the remaining local stages are
  implemented and minimally tested.

## Stage 4 Plan Status

Completed locally:

- small task decomposition for mandatory marking, ordinary-only pruning,
  selection integration, and point-level counters;
- pure `mark_energy_window_pruning(...)` helper;
- exact-oracle CLI feature flags for future explicit Stage 4 variants;
- branch-candidate `pruned_reason` output;
- point-level `energy_window_pruning_enabled` and `energy_window_pruned_count`
  fields;
- synthetic tests for ordinary pruning, mandatory preservation, default-off
  behavior, and target-selection skipping.

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py scripts\run_local_refinement_fixed_point_regression.py tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py
python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_stage1_gate.py -q
python -m pytest tests -q

Result: full local tests reported 29 passed.
```

Pending:

- GPU fixed-point regression for an explicit Stage 4 energy-pruning variant;
- Stage 5 branch-reuse integration after explicit reuse diagnostics are
  designed;
- Stage 6 adaptive-box production integration after GPU regression design;
- combined HPC package set generation after variant-runner support;
- combined later-stage HPC package set after the remaining local stages are
  implemented and minimally tested.

## Stage 5 Plan Status

Completed locally:

- small task decomposition for stable signatures, reuse acceptance checks, and
  explicit rejection reasons;
- pure `branch_reuse_signature(...)` helper;
- pure `evaluate_branch_reuse_candidate(...)` helper;
- pure `build_branch_reuse_cache_record(...)` helper;
- pure `build_branch_reuse_diagnostic_record(...)` helper;
- synthetic tests for signature stability, accepted matching reuse, signature
  mismatch rejection, lower-competing-branch rejection, cache-record fields,
  and reuse diagnostic records.

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py tests\test_branch_reuse.py tests\test_energy_window_pruning.py tests\test_selective_refinement.py tests\test_basin_clustering.py
python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_branch_reuse.py tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_stage1_gate.py -q
python -m pytest tests -q

Result: full local tests reported 33 passed before the diagnostic-contract update.

Latest diagnostic-contract validation:

python -m py_compile ml_phase\exact_oracle.py tests\test_branch_reuse.py tests\test_adaptive_local_box_skeleton.py
python -m pytest tests\test_branch_reuse.py tests\test_adaptive_local_box_skeleton.py -q

Result: 12 passed.
```

Pending:

- production-loop branch reuse integration;
- GPU fixed-point regression for an explicit branch-reuse variant;
- Stage 6 adaptive-box production integration after GPU regression design.

## Stage 6 Plan Status

Completed locally:

- small task decomposition for basin geometry, representative diagnostics,
  adaptive half-width suggestions, and fixed-box default behavior;
- pure `estimate_basin_geometry(...)` helper;
- pure `adaptive_local_box_half_widths(...)` helper;
- pure `build_adaptive_local_box_diagnostic_record(...)` helper;
- branch-candidate geometry diagnostic fields;
- synthetic tests for width/span/proxy computation, default fixed-box behavior,
  bounded adaptive suggestions, and diagnostic-record reasons.

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py tests\test_adaptive_local_box_skeleton.py tests\test_branch_reuse.py tests\test_energy_window_pruning.py tests\test_selective_refinement.py tests\test_basin_clustering.py
python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_branch_reuse.py tests\test_adaptive_local_box_skeleton.py tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_stage1_gate.py -q
python -m pytest tests -q

Result: full local tests reported 36 passed before the diagnostic-contract update.

Latest diagnostic-contract validation:

python -m py_compile ml_phase\exact_oracle.py tests\test_branch_reuse.py tests\test_adaptive_local_box_skeleton.py
python -m pytest tests\test_branch_reuse.py tests\test_adaptive_local_box_skeleton.py -q

Result: 12 passed.
```

Pending:

- production-loop adaptive box integration;
- GPU fixed-point regression for an explicit adaptive-box variant;
- combined HPC package set generation after variant-runner support.

## Stage 7 Plan Status

Completed locally:

- tensor-shape requirements for future local-box batching;
- batch-evaluation API requirements;
- Hamiltonian cache signature input list;
- profiler hook location list;
- future GPU validation gates;
- combined HPC package set plan.
- explicit fixed-point runner variants:
  `baseline`, `cluster_only`, `cluster_optional_k3`,
  `cluster_optional_k2`, and `cluster_energy_window`;
- variant-suite HPC package generation:
  `hpc_packages/local_refinement_refactor_variant_suite.tar.gz`;
- archive SHA256 and metadata sidecars;
- package-local preflight under the extracted package run root.
- top-level upload-set handoff bundle:
  `hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz`;
- upload-set manifest, run order, and return checklist.
- refreshed goal-run audit report whose machine-readable status is
  `stage2_3_4_gpu_variant_pending`.
- returned-result performance report builder:
  `scripts/build_local_refinement_performance_report.py`;
- independent performance collector:
  `scripts/collect_local_refinement_performance_report.sh`;
- variant-suite postprocess now calls the collector and writes
  `runtime_summary.csv`,
  `local_box_summary.csv`, `performance_summary.json`, and
  `performance_report.md` under the package `RUN_ROOT` before collecting the
  return archive.
- variant-suite importer now writes `import_status` and requires the
  performance report build to pass after extraction, while preserving the
  separate physics-equivalence `gate_status`.
- goal-run audit now requires `gate_status=pass`, `import_status=pass`, and
  `performance_report_status=pass` before reporting Stage 2-4 GPU variant
  validation as passed.
- local-only Stage 7 interface-contract helpers:
  `build_local_box_batch_plan(...)`,
  `build_hamiltonian_cache_signature(...)`,
  `evaluate_hamiltonian_cache_candidate(...)`,
  `build_hamiltonian_cache_diagnostic_record(...)`, and
  `build_local_box_profiler_event(...)`;
- synthetic tests for tensor-shape batch plans, cache signature stability,
  cache hit/miss rejection reasons, and profiler event fields.
- Stage 1 reference-package default runtime output policy now matches the
  package-local run-directory rule:
  `$PACKAGE_ROOT/local_refinement_refactor_stage1_run` when `RUN_ROOT` is
  unset and the extracted package is writable.

Validation:

```text
python -m pytest tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_stage1_gate.py -q
Result: 14 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports/local_refinement_refactor/variant_regression/preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python -m pytest tests -q
Result: 48 passed.

python -m py_compile ml_phase\exact_oracle.py tests\test_gpu_batching_cache_skeleton.py
Result: passed

python -m pytest tests\test_gpu_batching_cache_skeleton.py -q
Result: 6 passed

python -m pytest tests\test_gpu_batching_cache_skeleton.py tests\test_branch_reuse.py tests\test_adaptive_local_box_skeleton.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_goal_run_audit.py -q
Result: 24 passed

python -m pytest tests -q
Result: 59 passed

python -m py_compile scripts\package_local_refinement_refactor_hpc.py scripts\run_local_refinement_fixed_point_regression.py scripts\compare_local_refinement_variants.py scripts\verify_local_refinement_stage1_gate.py scripts\collect_local_refinement_stage1_outputs.py scripts\preflight_local_refinement_stage1_hpc.py tests\test_local_refinement_stage1_gate.py
Result: passed

python -m pytest tests\test_local_refinement_stage1_gate.py -q
Result: 10 passed

python -m pytest tests -q
Result: 60 passed

python scripts\package_local_refinement_refactor_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_stage01_instrumentation.tar.gz

python hpc_packages\local_refinement_refactor_stage01_instrumentation\scripts\validate_local_refinement_hpc_package.py --package-dir hpc_packages\local_refinement_refactor_stage01_instrumentation --archive hpc_packages\local_refinement_refactor_stage01_instrumentation.tar.gz
Result: status=pass

python hpc_packages\local_refinement_refactor_stage01_instrumentation\scripts\preflight_local_refinement_stage1_hpc.py --package-root hpc_packages\local_refinement_refactor_stage01_instrumentation --run-root hpc_packages\local_refinement_refactor_stage01_instrumentation\local_refinement_refactor_stage1_run --output-json hpc_packages\local_refinement_refactor_stage01_instrumentation\local_refinement_refactor_stage1_run\reports\local_refinement_refactor\stage_01_instrumentation\stage1_runtime_preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports\local_refinement_refactor\variant_regression\preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass
```

Pending:

- production-loop integration of Stage 5/6 prototypes after diagnostic
  contracts are wired to emitted records;
- production-loop integration of Stage 7 GPU batching/cache remains deferred;
- upload and run the variant-suite package on the target GPU/CUDA environment;
- compare returned Stage 2/3/4 variant outputs against baseline;
- import/check the returned variant-suite result archive after HPC completion.

## Middle-Plan Stage 1 Basin-Level Risk Refactor

Date: 2026-06-08

Status:

```text
implemented-locally; bounded tests pass; full pytest has unrelated package/audit artifact failures
```

Scope:

```text
This is the Stage 1 in
project_history/plans_and_runbooks/TwoPhase_Optimization_Middle.md, not the
older Stage 1 instrumentation package above.
```

Completed:

- added explicit basin-level risk flags and per-risk member counts to
  clustered basin representatives;
- made mandatory target selection and energy-window pruning prefer basin-level
  risk flags when present;
- preserved legacy candidate-level fallback for unclustered baseline paths;
- added local-box and branch-candidate diagnostic fields for basin risk flags
  and basin selection reasons;
- added a synthetic regression where a non-representative cluster member
  carries Delta-near-epsilon risk and keeps the basin mandatory.

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py
Result: pass.

python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_mandatory_branch_keep.py -q
Result: 12 passed.

python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_mandatory_branch_keep.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_variant_suite_package.py -q
Result: 21 passed.

python -m pytest tests -q
Result: 72 passed, 3 failed.  The failures are in local-refinement package/audit artifact consistency
(`variant_preflight_status=missing`, variant nested GPU script count 1 vs
expected 5, and TwoPhase audit status incomplete vs pending_hpc), not in the
basin-risk tests.
```

Pending:

- Stage 2 rank-and-cap / hard total cap policy;
- regenerated HPC packages after the target-construction policy is stable;
- bounded GPU fixed-point regression before rerunning the 72 missing tasks.

## Middle-Plan Stage 2 Ranked Mandatory Selection and Hard Cap

Date: 2026-06-08

Status:

```text
implemented-locally; selector gate tests pass; fixed-point dry-run still pending
```

Scope:

```text
This is Stage 2 in
project_history/plans_and_runbooks/TwoPhase_Optimization_Middle.md.
It adds an opt-in rank-and-cap target-construction policy without changing
historical failed variants or physical/numerical criteria.
```

Completed:

- added `high_risk_overflow_policy = rank_and_cap`;
- added per-risk caps for edge-risk, Delta-near-epsilon, and near-degenerate
  mandatory basins;
- enforced `max_total_refined_basins` as a hard cap for new rank-and-cap
  variants;
- added diagnostic branch-candidate fields for selected/dropped targets and
  mandatory overflow;
- added `rank_and_cap_k3`, `rank_and_cap_k2`, and
  `rank_and_cap_energy_window` fixed-point regression variants;
- documented the local selector gate under
  `reports/local_refinement_refactor/stage_02_rank_and_cap/`.

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py scripts\run_local_refinement_fixed_point_regression.py
Result: pass.

python -m pytest tests\test_selective_refinement.py tests\test_mandatory_branch_keep.py tests\test_energy_window_pruning.py tests\test_local_refinement_regression_scaffold.py -q
Result: 15 passed.

python -m pytest tests\test_feature_flag_baseline_equivalence.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_performance_report.py -q
Result: 10 passed.

python -m pytest tests\test_selective_refinement.py tests\test_mandatory_branch_keep.py tests\test_energy_window_pruning.py tests\test_local_refinement_regression_scaffold.py tests\test_feature_flag_baseline_equivalence.py -q
Result: 18 passed.
```

Pending:

- Stage 3 target-construction dry-run over all 32 fixed points;
- compare `rank_and_cap_*` target counts against baseline and historical
  `cluster_optional_*` variants;
- regenerate upload packages only after the dry-run target-count gate passes;
- still do not rerun the 72 missing optimized tasks directly.

## Middle-Plan Stage 3 Ordinary Branch Policy Gate

Date: 2026-06-08

Status:

```text
implemented-locally; ordinary energy-window selector gate passes; 32-point dry-run still pending
```

Completed:

- added a rank-and-cap energy-window test confirming ordinary-only pruning;
- confirmed selected mandatory-risk basins are not pruned by the ordinary
  energy-window policy;
- documented the ordinary-policy gate under
  `reports/local_refinement_refactor/stage_03_ordinary_policy/`.

Validation:

```text
python -m pytest tests\test_energy_window_pruning.py tests\test_selective_refinement.py -q
Result: 9 passed.

python -m pytest tests\test_mandatory_branch_keep.py tests\test_local_refinement_regression_scaffold.py tests\test_feature_flag_baseline_equivalence.py -q
Result: 10 passed.
```

Pending:

- true 32 fixed-point target-construction dry-run;
- ordinary count before/after energy-window pruning by risk category;
- bounded GPU test only after the target-count gate passes.

## Middle-Plan Stage 3 HPC Target-Construction Dry-Run Package

Date: 2026-06-08

Status:

```text
packaged-locally; not submitted
```

Completed:

- added a target-construction-only point worker that performs coarse scan,
  q-window expansion, candidate detection, clustering, risk annotation,
  energy-window marking, and final target selection, then stops before
  local-box scans;
- added an aggregator that writes target-construction tables and a return
  archive;
- added an HPC status checker;
- created a complete upload package at
  `hpc_packages/local_refinement_target_construction_dryrun.tar.gz`;
- verified all package shell scripts are LF-only;
- verified Slurm scripts and submit workflow exclude `gpuh01`.

Validation:

```text
python -m py_compile scripts\run_local_refinement_target_construction_point.py scripts\aggregate_local_refinement_target_construction_dryrun.py scripts\check_target_construction_dryrun_hpc_status.py scripts\package_local_refinement_target_construction_dryrun_hpc.py
Result: pass.

python scripts\package_local_refinement_target_construction_dryrun_hpc.py
Result: wrote hpc_packages\local_refinement_target_construction_dryrun.tar.gz.

python scripts\preflight_target_construction_dryrun_hpc.py --package-root . --run-root . --output-json preflight_local.json
Result: status=pass from inside the generated package.

LF shell check
Result: LF_OK.
```

Pending:

- upload the tarball to HPC;
- run `bash scripts/submit_target_construction_dryrun_workflow.sh`;
- inspect the returned
  `local_refinement_target_construction_dryrun_results.tar.gz`;
- proceed to fixed-point GPU local-box regression only if rank-and-cap target
  counts pass the cap gate.
