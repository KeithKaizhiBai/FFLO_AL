# 2026-06-23 Stage IV 3D Topology-Aware HPC Package

## Question

How was the Stage IV 3D cold-start topology-aware active-learning package built, and what should be uploaded to HPC?

## Answer

The Stage IV-A package was built as an independent cold-start calculation package:

```text
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz.sha256
```

The run id is:

```text
active_phase_topology_3d_t_ja_mu_from_scratch_v1
```

The output root is:

```text
ML_Phase_StageIV_Topology3D
```

The production config uses:

```text
kBT/t range: [0.0, 0.56]
J_A/t range: [0.0, 2.12]
mu/t range: [-0.5, 1.5]
guard mu/t range: [-1.0, 2.0]
initial seed: 1024 scrambled Sobol 3D points
batch size: 256
acquisition batches: 24
exact oracle: robust_incremental
local refinement: basin-level rank-and-cap K3
topology diagnostics: Pfaffian Z2 plus full-BZ bulk gap
```

The package validation status is:

```text
package_validation_status = pass
archive sha256 = ce18cda0d2347c02f7de76390555aafa53bea018a9091b47518cdb0223ad1244
archive size = 342,472 bytes
package preflight = pass using the production config and a lightweight guard scan
package submit-ready check = pass; package files, config, imports, py_compile, shell encoding, Slurm gpuh01 exclusion, and clean run-directory state are checked before sbatch
package status-checker smoke = pass; missing run directory is reported as run_dir_missing
package return-check smoke = pass; missing returned output is reported as return_path_missing
package failed-rank recovery = pass; recover_stageiv_failed_exact_iter.sh is included for failed exact-array ranks
package command checklist = pass; HPC_COMMANDS_STAGEIV_3D.md records submit monitor recovery collect return-check and post-run audit commands
```

## Implementation Notes

Per-point `mu` now propagates through:

```text
selected_points.csv
ml_phase.hpc shard partitioning
ml_phase.exact_oracle point evaluation
exact shard npz outputs
ml_phase.append_trusted dataset append
topology Pfaffian and bulk-gap diagnostics
```

Existing 2D datasets remain readable because `mu` defaults to the Stage III reference value `0.55` when absent.

The package does not include Stage III datasets or checkpoints.  Stage III data are only used as prior validation context, not as training or initialization data for this Stage IV run.

The package includes both `README.md` and `README_STAGEIV_3D_HPC.md`; both
point to the same Stage IV submit, monitor, read-only status check, collect,
resume, and post-run audit commands.  The standard `README.md` entry is
included for ordinary HPC handoff and archive inspection workflows.

The package also includes `ENVIRONMENT_STAGEIV_3D_HPC.md`, which records the
intended `NV_H100` runtime, `PYTHON_BIN`, `gpuh01` exclusion, CUDA checks,
`REFERENCE_DATASET` validation-only role, and shell/Python encoding policy.

## HPC Submit Command

After upload and extraction:

```bash
cd ~/bkz/Fu_FFLO/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc
export CONFIRM_STAGEIV_FULL_LOOP=1
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
bash scripts/check_stageiv_3d_submit_ready.sh
nohup bash scripts/submit_stageiv_3d_full_loop.sh > active_phase_topology_3d_t_ja_mu_from_scratch_v1.nohup.log 2>&1 &
```

The Slurm scripts include both:

```text
#SBATCH --exclude=gpuh01
runtime hostname guard against gpuh01
```

Shell scripts were normalized to LF, no BOM, and ASCII-safe content.
Local `bash -n` checks were skipped because Windows maps `bash` to WSL and no
WSL distribution is installed on this workstation.  Run `bash -n scripts/*.sh`
on the Linux login node if a second shell syntax check is desired before
submission.

## Local Validation Commands

```text
python -m py_compile ml_phase/stageiv_3d.py scripts/stageiv_3d_select.py scripts/stageiv_3d_preflight.py scripts/stageiv_3d_convergence_audit.py scripts/stageiv_3d_hidden_slice_audit.py scripts/package_stageiv_topology_3d_hpc.py
python scripts/stageiv_3d_preflight.py --output-dir reports/stageiv_3d_preflight_local --n-ja 8 --n-mu 8 --n-k 128
python scripts/package_stageiv_topology_3d_hpc.py
python scripts/stageiv_3d_select.py --config hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/configs/stageiv_3d_smoke.json --mode seed --iteration 0 --output-root reports/stageiv_3d_selector_smoke_output --run-id stageiv_seed_smoke --world-size 2 --partition-strategy cost_aware
python scripts/stageiv_3d_select.py --config reports/stageiv_3d_acq_smoke_config.json --mode acquisition --iteration 1 --output-root reports/stageiv_3d_acq_smoke_output --run-id stageiv_acq_smoke --dataset reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke/dataset_iter001.npz --world-size 2 --partition-strategy cost_aware --device cpu
python scripts/stageiv_3d_convergence_audit.py --run-dir reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke --output-dir reports/stageiv_3d_convergence_audit_smoke --config reports/stageiv_3d_acq_smoke_config.json --audit-cloud-size 512 --no-pdf
python scripts/stageiv_3d_convergence_audit.py --run-dir reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke --output-dir reports/stageiv_3d_convergence_audit_smoke_pdf --config reports/stageiv_3d_acq_smoke_config.json --audit-cloud-size 512
python scripts/stageiv_3d_hidden_slice_audit.py --run-dir reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke --output-dir reports/stageiv_3d_hidden_slice_audit_smoke_missing_ref --config reports/stageiv_3d_acq_smoke_config.json --grid-n 41
python scripts/stageiv_3d_hidden_slice_audit.py --run-dir reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke --output-dir reports/stageiv_3d_hidden_slice_audit_smoke_self_ref --config reports/stageiv_3d_acq_smoke_config.json --reference-dataset reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke/dataset_iter001.npz --grid-n 41
python scripts/stageiv_3d_postrun_bundle.py --run-dir reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke --output-dir reports/stageiv_3d_postrun_bundle_smoke --config reports/stageiv_3d_acq_smoke_config.json --audit-cloud-size 512 --hidden-grid-n 41 --no-pdf
python scripts/stageiv_3d_hpc_status.py --output-root reports/stageiv_3d_status_checker_smoke_output --output-dir reports/stageiv_3d_status_checker_smoke --config reports/stageiv_3d_acq_smoke_config.json
python scripts/stageiv_3d_return_check.py --return-path ML_Phase_StageIV_Topology3D --config hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/configs/stageiv_3d_production.json --output-dir reports/stageiv_3d_return_check_missing_smoke
python scripts/stageiv_3d_return_check.py --return-path hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz --config hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/configs/stageiv_3d_production.json --output-dir reports/stageiv_3d_return_check_package_tar_smoke
python scripts/stageiv_3d_submit_check.py --root hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc --config configs/stageiv_3d_production.json --output-dir hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/reports/stageiv_3d_submit_check_package --python-bin python --allow-missing-manifest
python scripts/package_stageiv_topology_3d_hpc.py --print-json
```

The seed-selection smoke test wrote `kT,JA,mu` columns and partition metadata with `point_columns = ["kT", "JA", "mu"]`.

The acquisition-selection smoke test used a tiny synthetic 3D dataset and wrote 8 selected 3D points with `point_columns = ["kT", "JA", "mu"]`, exercising mixed candidate sources, phase/spectral/coverage channel quotas, stochastic batch selection, and shard partitioning without running exact calculations.

After the Slurm loop finishes on the cluster, first collect the result archive:

```bash
bash scripts/collect_stageiv_3d_results.sh
```

If `squeue` is empty but the run/checkpoint/archive state is unclear, run the
read-only status checker before deciding whether to resume or collect:

```bash
bash scripts/check_stageiv_3d_hpc_status.sh
```

With a known Slurm job id:

```bash
JOB_ID=<job_id> bash scripts/check_stageiv_3d_hpc_status.sh
```

The checker writes:

```text
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_hpc_status/stageiv_3d_hpc_status.json
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_hpc_status/stageiv_3d_hpc_status.md
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_hpc_status/tables/stageiv_iteration_file_status.csv
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_hpc_status/tables/stageiv_dataset_file_status.csv
```

It is report-only and does not submit jobs, merge shards, append datasets, run
exact calculations, or continue active learning.

Before submitting, the package now also includes a read-only submit-ready
checker:

```bash
bash scripts/check_stageiv_3d_submit_ready.sh
```

It writes:

```text
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_submit_check/stageiv_3d_submit_check.json
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_submit_check/stageiv_3d_submit_check.md
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_submit_check/tables/stageiv_submit_check_items.csv
```

The checker is report-only and does not submit Slurm jobs, merge shards,
append datasets, run exact calculations, or continue active learning.

After downloading or extracting returned Stage IV outputs, run the read-only
return checker before the post-run bundle:

```bash
RETURN_PATH=ML_Phase_StageIV_Topology3D bash scripts/check_stageiv_3d_return_bundle.sh
```

The return checker writes:

```text
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_check/stageiv_3d_return_check.json
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_check/stageiv_3d_return_check.md
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_check/tables/stageiv_return_artifacts.csv
```

It validates either a returned directory or tar archive.  It does not extract
archives, submit jobs, merge shards, append datasets, run exact calculations,
or continue active learning.

After the archive is returned and extracted, the package also includes
report-only post-run audits:

```bash
bash scripts/build_stageiv_3d_convergence_audit.sh
REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_hidden_slice_audit.sh
REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_all_postrun_reports.sh
```

The audit command reads returned cumulative Stage IV datasets and writes
Markdown, LaTeX/PDF when available, CSV tables, figures, and a decision JSON.
It does not submit jobs, run exact calculations, continue active learning, or
modify returned datasets.

The local convergence-audit smoke test intentionally used only a one-iteration
smoke output.  It returned:

```text
stageiv_convergence_status = insufficient_history
decision_class = Decision D
need_new_exact_calculation = false
```

This is the expected gate behavior: at least five cumulative datasets are
required for the last-three-transition audit, and missing history is not
converted into false convergence.  PDF generation was checked at:

```text
reports/stageiv_3d_convergence_audit_smoke_pdf/stageiv_3d_convergence_audit.pdf
```

The hidden fixed-mu slice audit was also added as a report-only post-run
validation command.  It requires an external Stage III frozen reference dataset
through `REFERENCE_DATASET`; that reference is used only for validation and is
not merged into Stage IV training.  If `REFERENCE_DATASET` is absent, the audit
returns:

```text
hidden_slice_status = inconclusive
decision_class = Decision D
reason = reference_dataset_missing
need_new_exact_calculation = false
```

The self-reference smoke path checked the full output chain, including strict
JSON cleanup and PDF generation:

```text
reports/stageiv_3d_hidden_slice_audit_smoke_self_ref/stageiv_3d_hidden_slice_audit.pdf
```

The package now also includes a one-command post-run bundle wrapper:

```bash
REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_all_postrun_reports.sh
```

The bundle runs the lightweight post-run summary, the 3D convergence audit,
and the hidden fixed-mu validation audit, then writes:

```text
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_postrun_bundle/stageiv_3d_postrun_bundle_decision.json
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_postrun_bundle/stageiv_3d_postrun_bundle.md
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_postrun_bundle/tables/stageiv_postrun_bundle_components.csv
```

The local bundle smoke intentionally used only the one-iteration smoke output
and no reference dataset.  It returned:

```text
postrun_bundle_status = incomplete_convergence_history
decision_class = Decision D
convergence_status = insufficient_history
hidden_slice_status = inconclusive
need_new_exact_calculation = false
```

The read-only HPC status checker was added after the post-run bundle.  The
local smoke intentionally used a missing output root and returned:

```text
hpc_status = run_dir_missing
next_action = check_upload_path_or_submit_stageiv_full_loop
need_new_exact_calculation = false by construction; the checker never launches exact work
```


The package now also includes a failed-rank recovery command for the common HPC case where one exact-array rank fails while other ranks complete, for example transient CUDA device busy/unavailable on a shared node:

```bash
bash scripts/inspect_stageiv_failed_task.sh <job_id>
ITER=<failed_iteration> FAILED_RANKS=<rank_list> bash scripts/recover_stageiv_failed_exact_iter.sh
START_ITER=<next_iteration> bash scripts/resume_stageiv_3d_full_loop.sh
```

The recovery command resubmits only the listed Slurm array ranks for the failed exact iteration, checks that all expected shard files exist, merges shards, appends trusted exact outputs, and prints the next resume command. It should not be used to rerun a whole iteration unless only the specified ranks failed and the selected-points partition is already present.

The package now includes a concise command checklist:

```text
HPC_COMMANDS_STAGEIV_3D.md
```

It records the exact command sequence for submit-ready checking, production
submission, monitoring, failed-rank recovery, completed-prefix resume,
collection, returned-result checking, and post-run report-only audits.  It also
repeats the key operational restrictions: do not merge Stage III or Phase-II
datasets into Stage IV training, do not treat preflight success as convergence,
do not use `gpuh01`, and do not restart from scratch for a single-rank CUDA
device-busy failure before trying targeted recovery.

The read-only returned-result checker was added after the status checker.  Its
local smoke intentionally used a missing output root and returned:

```text
return_check_status = return_path_missing
return_next_action = download_or_extract_stageiv_results
```

The tar-only smoke against the package archive returned:

```text
return_check_status = tar_only_incomplete
return_next_action = extract_archive_or_recollect_results_then_check_directory
```

The rebuilt package records:

```text
archive sha256 = ce18cda0d2347c02f7de76390555aafa53bea018a9091b47518cdb0223ad1244
archive size = 342,472 bytes
package_validation_status = pass
tar member count = 90
```

## Caveats

This package has not yet been submitted on HPC.  It runs a fixed Stage IV cold-start full loop.  The package now includes both the report-only 3D convergence audit command and the Stage III hidden fixed-mu slice audit command, but neither can prove Stage IV convergence until production HPC cumulative datasets are returned.

Do not merge Stage III `dataset_iter018` or Phase-II `dataset_iter035` into this run.  The run is intended to test cold-start 3D active learning over `(kBT/t, J_A/t, mu/t)`.

The current package implements the cold-start 3D path with mixed candidate
sources: global Sobol, topology opposite-Z2 bracket jitter, thermodynamic
opposite-phase bracket jitter, and mu-edge guard points.  Selection uses
explicit phase/spectral/coverage channel quotas with backfill and records both
`acquisition_channel` and `candidate_source`.  The stability-triggered
acquisition weight switch remains a post-submission improvement; the current
package still uses the configured iteration switch and does not silently claim
3D convergence before the returned results are audited.
