# Local-Refinement Stage 1 HPC Handoff Q&A

Date: 2026-06-03

## Question

What must be uploaded and run on HPC before the local-refinement refactor can
move from Stage 1 instrumentation to Stage 2 basin clustering?

## Short Answer

Upload the Stage 1 fixed-point regression package archive and its two integrity
sidecars, verify the archive checksum on HPC, run the workflow submitter, return
the generated regression-results bundle, and import that bundle locally.  Stage
2 must remain pending until the imported Stage 1 gate status reports
`status=pass`.

## Upload Files

```text
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.sha256
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.metadata.json
```

Current local package validation evidence:

```text
status = pass
directory_file_count = 97
archive_file_count = 97
archive_size_bytes = 383026
archive_sha256 = acd5eef15b273efd6db452f0a66da727fe5b98434d68d15c01b1516bff168d7b
failures = []
```

## HPC Commands

If the extracted package directory is not writable, choose a writable run
directory before preflight and submission:

```bash
export RUN_ROOT="$HOME/local_refinement_refactor_stage1_run"
mkdir -p "$RUN_ROOT"
```

Run the checksum before extracting:

```bash
sha256sum -c local_refinement_refactor_stage01_instrumentation.tar.gz.sha256
```

Then extract and submit the fixed-point regression workflow:

```bash
tar -xzf local_refinement_refactor_stage01_instrumentation.tar.gz
cd local_refinement_refactor_stage01_instrumentation
python scripts/preflight_local_refinement_stage1_hpc.py --package-root . --run-root "$RUN_ROOT"
bash scripts/submit_stage1_regression_workflow.sh
```

The workflow submitter also runs this preflight automatically before `sbatch`.
It checks the extracted package paths, `RUN_MANIFEST.json`, 32 fixed-point rows,
key Python script syntax, package-root imports, and records a torch/CUDA
visibility snapshot.  It does not run exact BdG calculations.

If the cluster requires an explicit partition for the postprocess job:

```bash
POSTPROCESS_SBATCH_ARGS="--partition=NV_H100" bash scripts/submit_stage1_regression_workflow.sh
```

The workflow submits:

```text
1. Stage 0 baseline fixed-point exact regression
2. Stage 1 instrumented fixed-point exact regression
3. afterok postprocess job that runs compare, verify, and collect
```

The two GPU exact-regression Slurm scripts submit to `NV_H100` and exclude the
known bad node:

```text
#SBATCH --exclude=gpuh01
```

They also run a CUDA tensor-allocation probe before the exact oracle starts, so
driver mismatches fail early in the job log.

The package is fixed-point regression only.  It does not run active learning
and does not append training data.

## Return File

After the postprocess job completes, return:

```text
local_refinement_refactor_stage1_regression_results.tar.gz
```

The return archive is written under `RUN_ROOT` when that variable is set.

The collector intentionally writes a return bundle even when the gate fails, so
failed or incomplete HPC attempts still provide logs, missing-path metadata,
and gate status for diagnosis.

## Local Import Command

```text
python scripts/import_local_refinement_stage1_results.py local_refinement_refactor_stage1_regression_results.tar.gz
```

The importer extracts into:

```text
reports/local_refinement_refactor/stage_01_instrumentation/imported_results/
```

It reruns the same artifact-level gate verifier against the extracted bundle
without overwriting repository-root reports or package files.

## Stage 1 Pass Criteria

The Stage 1 gate is a behavioral-equivalence gate for logging-only
instrumentation.  It requires all required artifacts to exist, the expected 32
fixed points to be compared, the instrumented local-box CSV to exist, and these
classification/status mismatch counts to be zero:

```text
phase_candidate
trusted_exact
training_eligible_exact
q_unresolved
delta_unresolved
rerun_required
```

The gate does not change the thermodynamic phase criterion, q-window policy,
Delta ambiguity policy, stable-normal training admission, acquisition formula,
StopController, or active-learning append behavior.

## Current Local Status

The local canonical gate file currently reports `status=fail` because these GPU
exact regression artifacts are missing:

```text
baseline_csv
candidate_csv
baseline_manifest
instrumented_manifest
comparison_summary
pointwise_comparison
mismatch_points
local_box_csv
```

This is expected before the target GPU/CUDA regression jobs have returned.
Stage 2 basin clustering remains pending until returned GPU evidence makes
`stage1_gate_status.json` report `status=pass`.
