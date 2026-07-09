# Goal Checkpoint

Date: 2026-06-04

Status: blocked, do not treat as completed.

## 1. Current Goal

```text
继续根据"d:\PhD Study\4th\SC_GBU\Fu_FFLO\project_history\plans_and_runbooks\TwoPhase_Optimization.md"完成后续阶段的工程任务执行。对每一个阶段的每一个大任务，先将其拆分成逻辑上层层递进、交叉级联的小任务，再按照每个小任务进行细致的任务规划，然后执行。对每个阶段，做到在本地跑minimal test，不要每个阶段都停下生成超算包，如果需要超算计算，先进行最小测试，通过后可直接进行下一阶段任务，当前需要验证的计算可写一个超算包。所有阶段完成后，汇总超算包集合打包等待上传。
```

The goal was marked blocked because the same external dependency repeated
across consecutive continuation turns: the Stage 2-4 GPU variant-suite return
archive has not been produced and imported.

## 2. Completed Modifications

Completed local work:

```text
1. Stage 1 instrumentation GPU fixed-point regression package was built,
   returned, imported, and audited as pass.
2. Stage 2-4 local-refinement variant-suite local implementation, tests,
   preflight, packaging, and upload-set verifier were completed locally.
3. Import-path issues were fixed so packages rely on package-local structure
   rather than HPC-specific directory layout.
4. Runtime output policy was hardened so package logs/reports are written under
   RUN_ROOT instead of non-writable upload/root directories.
5. GPU Slurm scripts are checked for gpuh01 exclusion and a real CUDA tensor
   allocation probe.
6. Variant-suite return readiness checker and package-local HPC status checker
   were added and audited.
7. Goal-run and TwoPhase completion audits were added/refined; current audits
   explicitly expose the pending Stage 2-4 GPU return gate.
8. The upload-set now contains a top-level helper,
   run_required_variant_suite.sh, which verifies the upload set, extracts the
   required variant-suite archive, and submits the existing package-local
   fixed-point regression alias.
9. Stage 5 branch reuse, Stage 6 adaptive box, and Stage 7 GPU
   batching/Hamiltonian cache remain deferred by design until the Stage 2-4 GPU
   variant-suite gate passes.
```

No physical definitions, thermodynamic phase criteria, numerical thresholds,
feature flags, or exact-oracle physics conventions were intentionally changed
by the final upload-set handoff helper.

Current authoritative local checks:

```text
python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass; top_level_handoff checks pass; nested packages pass

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass; evidence_matrix row_count=16

python scripts\audit_twophase_optimization_completion.py
Result: status=pending_hpc; requirement_count=15; pass=11; pending_hpc=1; deferred=3
```

Current upload package:

```text
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
sha256 = 1f41a14ecd565c0105518e4611c70996c6ef131017c9cc9b8f8f5351d2200f57
size   = 977568 bytes
package_count = 2
```

Nested required package inside the upload set:

```text
archives/local_refinement_refactor_variant_suite.tar.gz
sha256 = ab90ea157af398a67132fc92a51b30118f18b3e4663bb0b973dd0bc998fe19fc
size   = 530366 bytes
```

## 3. Modified File List

Goal-related source and test files created or modified during the local
refactor/checkpoint sequence include:

```text
ml_phase/acquisition.py
ml_phase/active_refine.py
ml_phase/config.py
ml_phase/exact_oracle.py
ml_phase/hpc.py
ml_phase/report_builder.py
scripts/audit_local_refinement_refactor_goal_run.py
scripts/audit_local_refinement_runbook_tests.py
scripts/audit_local_refinement_stage_reports.py
scripts/audit_twophase_optimization_completion.py
scripts/build_local_refinement_performance_report.py
scripts/build_local_refinement_regression_points.py
scripts/check_local_refinement_variant_suite_return.py
scripts/check_variant_suite_hpc_status.py
scripts/collect_local_refinement_performance_report.sh
scripts/collect_local_refinement_stage1_outputs.py
scripts/compare_local_refinement_variants.py
scripts/import_local_refinement_stage1_results.py
scripts/import_local_refinement_variant_suite_results.py
scripts/package_local_refinement_refactor_hpc.py
scripts/package_local_refinement_upload_set.py
scripts/package_local_refinement_variant_suite_hpc.py
scripts/preflight_local_refinement_stage1_hpc.py
scripts/run_local_refinement_fixed_point_regression.py
scripts/validate_local_refinement_hpc_package.py
scripts/verify_local_refinement_goal_run_report.py
scripts/verify_local_refinement_stage1_gate.py
tests/
docs/DECISIONS.md
docs/NUMERICS_SPEC.md
docs/PROJECT_SUMMARY.md
docs/LOCAL_REFINEMENT_REFACTOR_*.md
docs/QWINDOW_INCREMENTAL_*.md
docs/report_qa/
reports/local_refinement_refactor/
reports/local_refinement_refactor_goal_run/
hpc_packages/
local_refinement_refactor_stage01_instrumentation/
```

Important current generated/upload artifacts:

```text
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.sha256
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.metadata.json
```

The worktree also contains many pre-existing or unrelated dirty entries,
including tracked deletions of older plan/report files and many untracked
project-history/report/package directories. Do not commit blindly; review the
scope before any commit.

## 4. Current Git Diff Summary

Command:

```powershell
git diff --stat
```

Output:

```text
 Active_Learning_Phase_Boundary_Refinement_Plan.md  |  961 ----
 Full_reconstruct_plan.md                           | 1274 ----
 ML_Guidance.md                                     |  438 --
 Phase_boundary_sharp_plan.md                       |  561 --
 QDELTA_REFINEMENT_EXECUTION_PLAN.md                |  345 --
 QDELTA_TARGET_LOGIC_CODE_REWRITE_PLAN.md           |  469 --
 RUN_ORDER_GBU_HPC.md                               |  321 --
 docs/DECISIONS.md                                  |  365 +-
 docs/NUMERICS_SPEC.md                              |  116 +
 docs/PROJECT_SUMMARY.md                            | 6054 +++++++++++++++++++-
 fflo_transition.ipynb                              | 2735 ---------
 finite_T_phase_diagram.m                           |   19 -
 hpc_active_loop.sh                                 |   58 +-
 hpc_run_readme.md                                  |  156 -
 ml_phase/acquisition.py                            |   52 +-
 ml_phase/active_refine.py                          |   45 +-
 ml_phase/config.py                                 |   23 +-
 ml_phase/exact_oracle.py                           | 2377 +++++++-
 ml_phase/hpc.py                                    |   25 +-
 ml_phase/report_builder.py                         |  121 +-
 report_active_learning_r0015_note/README.md        |   28 -
 .../decision_log.md                                |   60 -
 .../figures/exact_eta_revised_boundaries.pdf       |  Bin 69517 -> 0 bytes
 .../point0021_nq12800_response_pathology.png       |  Bin 119666 -> 0 bytes
 .../point0025_nq12800_response_pathology.png       |  Bin 117849 -> 0 bytes
 .../figures/qdensity_eta_vs_nq.png                 |  Bin 117833 -> 0 bytes
 .../figures/qdensity_qic_shift_vs_nq.png           |  Bin 141857 -> 0 bytes
 .../qwindow_positive_eta_qextrema_shift.png        |  Bin 83348 -> 0 bytes
 .../figures/qwindow_two_level_eta_stability.png    |  Bin 96576 -> 0 bytes
 .../figures/selection_focus_curve.png              |  Bin 99466 -> 0 bytes
 .../numerical_reliability_audit.md                 |  212 -
 .../numerical_reliability_audit.pdf                |  Bin 921316 -> 0 bytes
 .../numerical_reliability_audit.tex                |  323 --
 .../tables/qdensity_summary_by_point.csv           |    7 -
 .../tables/response_pathology_summary.csv          |    5 -
 .../phase_qwindow_delta_refinement_config.json     |   56 -
 .../decision_log.md                                |   41 -
 .../input_points/clean_control_points.csv          |   21 -
 .../input_points/combined_phase_audit_points.csv   |  346 --
 .../input_points/delta_sensitive_points.csv        |   97 -
 .../input_points/qwindow_sensitive_points.csv      |  343 --
 .../phase_qwindow_delta_refinement.md              |   29 -
 .../phase_qwindow_delta_refinement.pdf             |  Bin 55426 -> 0 bytes
 .../phase_qwindow_delta_refinement.tex             |   37 -
 report_phase_qwindow_delta_refinement_v1/readme.md |   84 -
 .../reports/phase_qwindow_delta_refinement.md      |   29 -
 .../scripts/collect_phase_audit_results.sh         |   14 -
 .../phase_qwindow_delta_refinement_audit.py        | 1053 ----
 .../scripts/submit_delta_refinement_array.sh       |   25 -
 .../scripts/submit_phase_qwindow_array.sh          |   25 -
 scripts/slurm_active_refine.sh                     |   28 +-
 scripts/slurm_exact_oracle_array.sh                |   36 +-
 52 files changed, 9267 insertions(+), 10147 deletions(-)
```

Command:

```powershell
git status --short
```

High-level status:

```text
Tracked modified files include docs/DECISIONS.md, docs/NUMERICS_SPEC.md,
docs/PROJECT_SUMMARY.md, hpc_active_loop.sh, ml_phase/*.py, and Slurm scripts.

Tracked deleted files include older root plans, notebooks/scripts, and prior
report directories such as report_numerical_reliability_audit_20260523/ and
report_phase_qwindow_delta_refinement_v1/.

Untracked goal-related directories/files include docs/LOCAL_REFINEMENT_*,
docs/QWINDOW_INCREMENTAL_*, docs/report_qa/, hpc_packages/,
local_refinement_refactor_stage01_instrumentation/, project_history/, reports/,
scripts/audit_*.py, scripts/package_local_refinement_*.py,
scripts/check_*variant*.py, scripts/import_local_refinement_*.py, and tests/.
```

## 5. Blocking Reason

The blocker is external to the current local worktree:

```text
Missing Stage 2-4 GPU variant-suite return archive:
local_refinement_refactor_variant_suite_results.tar.gz
```

Current local evidence:

```text
python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending

python scripts\audit_twophase_optimization_completion.py
Result: status=pending_hpc, requirement_count=15, status_counts={pass:11, pending_hpc:1, deferred:3}
```

The pending requirement is:

```text
Stage 2-4 GPU variant-suite return is imported and passed
gate_status=pending; import_status=pending; performance_report_status=pending
```

Local checks already confirmed that the downloaded
local_refinement_refactor_stage01_instrumentation directory is a Stage 1
return/package directory, not the Stage 2-4 variant-suite return:

```text
reports/local_refinement_refactor_goal_run/downloaded_stage1_dir_variant_return_check.json
reports/local_refinement_refactor_goal_run/variant_run_root_return_check.json
```

Both reported:

```text
status=not_ready
missing local_refinement_refactor_variant_suite_results.tar.gz
```

## 6. Waiting External Computation

Upload this file to the cluster:

```text
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
sha256 = 1f41a14ecd565c0105518e4611c70996c6ef131017c9cc9b8f8f5351d2200f57
size   = 977568 bytes
```

Run on the cluster from the directory containing the uploaded tarball:

```bash
tar -xzf local_refinement_refactor_hpc_upload_set.tar.gz
cd local_refinement_refactor_hpc_upload_set
bash run_required_variant_suite.sh
```

The helper does the following:

```bash
python verify_upload_set.py --upload-root "$SCRIPT_DIR"
tar -xzf archives/local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_local_refinement_fixed_point_regression.sh
```

Expected nested package:

```text
local_refinement_refactor_hpc_upload_set/local_refinement_refactor_variant_suite/
```

Expected default run root if RUN_ROOT is not set:

```text
local_refinement_refactor_variant_suite/local_refinement_refactor_variant_suite_run/
```

Expected return archive:

```text
$RUN_ROOT/local_refinement_refactor_variant_suite_results.tar.gz
```

With default RUN_ROOT this is:

```text
local_refinement_refactor_variant_suite/local_refinement_refactor_variant_suite_run/local_refinement_refactor_variant_suite_results.tar.gz
```

If squeue has no visible running or pending jobs but the result archive is not
obvious, run from the extracted variant-suite package:

```bash
python scripts/check_variant_suite_hpc_status.py --package-root . --run-root "${RUN_ROOT:-local_refinement_refactor_variant_suite_run}" --query-scheduler
```

Expected successful cluster-side status:

```text
status = ready_to_return
return_archive.exists = true
```

## 7. How to Continue After External Validation

### If External GPU Run Passes

Download:

```text
local_refinement_refactor_variant_suite_results.tar.gz
```

Then run locally from the repository root:

```powershell
python scripts\check_local_refinement_variant_suite_return.py local_refinement_refactor_variant_suite_results.tar.gz
python scripts\import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
python scripts\audit_local_refinement_refactor_goal_run.py
python scripts\audit_twophase_optimization_completion.py
python scripts\verify_local_refinement_goal_run_report.py
```

Expected pass criteria before resuming later production stages:

```text
check_local_refinement_variant_suite_return.py -> ready_to_import
import_local_refinement_variant_suite_results.py -> gate_status=pass and import_status=pass
audit_local_refinement_refactor_goal_run.py -> no longer stage2_3_4_gpu_variant_pending
audit_twophase_optimization_completion.py -> Stage 2-4 GPU return requirement pass
```

After this passes:

```text
1. Update docs/PROJECT_SUMMARY.md with the import result and package hashes.
2. Update the goal-run report/tables if the audit scripts generated changed
   outputs.
3. Resume the goal and continue Stage 5 branch reuse production integration,
   then Stage 6 adaptive local-box production integration, then Stage 7 GPU
   batching/Hamiltonian cache production integration, following the runbook and
   keeping local minimal tests before further HPC handoff.
```

### If External GPU Run Fails or Produces No Archive

First run the cluster-side status checker:

```bash
cd local_refinement_refactor_hpc_upload_set/local_refinement_refactor_variant_suite
python scripts/check_variant_suite_hpc_status.py --package-root . --run-root "${RUN_ROOT:-local_refinement_refactor_variant_suite_run}" --query-scheduler
```

Interpretation:

```text
ready_to_return:
    Download the reported return archive and follow the pass path above.

pending_or_missing_return_archive:
    Inspect job ids and Slurm logs under RUN_ROOT/logs.  If jobs completed but
    postprocess did not run, rerun the package workflow or postprocess stage.

failed_or_needs_log_review:
    Inspect reported log matches.  Common cases:
    - old NVIDIA driver or gpuh01 appears: confirm gpuh01 exclusion and rerun
      on allowed GPU nodes.
    - cuda_runtime_probe=fail or torch CUDA init error: fix environment/node
      allocation before rerunning.
    - Slurm CANCELLED/FAILED/OUT_OF_MEMORY/TIMEOUT: rerun with corrected
      resource/time settings or inspect the specific failed job.
```

If a return archive exists but local import fails:

```powershell
python scripts\check_local_refinement_variant_suite_return.py <downloaded-return-directory-or-archive>
python scripts\import_local_refinement_variant_suite_results.py <archive>
```

Then inspect:

```text
variant_suite_gate_status.json
variant_suite_import_manifest.json
variant_summary.csv/json
pointwise comparison tables
runtime/performance report outputs
```

Continue based on failure type:

```text
1. Structural archive failure:
   fix collector/package return contents, rebuild package, rerun minimal
   package tests and verifier, then re-upload.

2. Physics-equivalence or gate mismatch:
   inspect pointwise comparison and mismatch tables before changing code.
   Do not relax thresholds silently.  Fix the specific implementation issue,
   rerun local tests, rebuild upload set, and repeat GPU validation.

3. Performance report missing/failing:
   fix report collector/importer companion files, rebuild package, and rerun
   the affected external validation.

4. CUDA/HPC environment failure:
   do not change physics code first.  Fix node selection, environment, or job
   resource settings, then rerun the same validated package.
```

Do not mark the goal complete until every explicit TwoPhase runbook completion
requirement is proven by current artifacts and audits.
