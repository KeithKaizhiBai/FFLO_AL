# GBU HPC Usage Summary

This document summarizes the working conventions we have established for using
the GBU Slurm cluster in this project.  It is an operational note, not a physics
definition file.  It must not override `docs/MODEL_SPEC.md`,
`docs/NUMERICS_SPEC.md`, `docs/DECISIONS.md`, or the stage-specific plans.

## 1. Core Usage Pattern

The project uses a package-first HPC workflow:

1. Build a self-contained upload package locally.
2. Verify package metadata, script encoding, and SHA256 hash.
3. Upload the archive to the login node.
4. Extract the package into its own directory.
5. Run a read-only submit-ready check.
6. Submit a Slurm full-loop or regression job.
7. Monitor with `tail`, `squeue`, and `sacct`.
8. Recover failed array ranks when possible.
9. Collect a result archive.
10. Download results and run local post-run audits.

This workflow avoids relying on the local repository after upload.  A valid HPC
package should contain the code, configs, Slurm scripts, collection scripts,
report-only audit scripts, and documentation needed for that calculation.

## 2. Standard Local Package Checks

Before uploading a package, check:

```powershell
Get-FileHash -Algorithm SHA256 <package>.tar.gz
tar -tzf <package>.tar.gz | Select-Object -First 80
```

For Stage IV the current package is:

```text
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624.tar.gz
sha256 = 4799ecfca7ab16c8a340d731dfb74cf0ea44502b2c4582b59523e65df00fcbf4
```

The package should include at least:

```text
AGENTS.md
README*.md
RUN_MANIFEST.json
configs/
docs/
ml_phase/
scripts/
HPC_COMMANDS*.md
```

For active-learning jobs, the package must not contain stale generated
training outputs such as:

```text
ml_phase/active_runs/
ml_phase/datasets/
ml_phase/figures/
ml_phase/reports/
```

Stage IV explicitly uses a cold-start package; Stage III and Phase-II datasets
must not be merged into Stage IV training.

## 3. Shell Encoding and Line Endings

All `.sh` scripts in the package should be Linux-compatible:

```text
first line = #!/bin/bash
CRLF = false
UTF-8 BOM = false
```

The Stage IV package was checked with:

```powershell
$files = Get-ChildItem -Recurse -File -LiteralPath hpc_packages\<package>\scripts -Include *.sh
$rows = foreach ($f in $files) {
  $b = [System.IO.File]::ReadAllBytes($f.FullName)
  [pscustomobject]@{
    Name=$f.Name
    HasCR=($b -contains 13)
    HasUtf8Bom=($b.Length -ge 3 -and $b[0] -eq 239 -and $b[1] -eq 187 -and $b[2] -eq 191)
  }
}
$rows | Format-Table -AutoSize
```

Every checked Stage IV `.sh` file had:

```text
HasCR = False
HasUtf8Bom = False
```

Inside scripts, use explicit UTF-8 environment settings:

```bash
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"
```

## 4. Upload and Extraction

From local PowerShell:

```powershell
scp "<local package path>.tar.gz" sci_bfu@login02:~/bkz/Fu_FFLO/
```

On the login node:

```bash
cd ~/bkz/Fu_FFLO
sha256sum <package>.tar.gz
tar -xzf <package>.tar.gz
cd <package_directory>
```

If extraction reports:

```text
gzip: stdin: unexpected end of file
tar: Unexpected EOF in archive
```

then the upload is incomplete.  Re-upload the archive and do not run a partial
extraction.

## 5. Python Environment

The normal HPC Python environment is:

```bash
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
```

Verify it before manual merge/append/recovery commands:

```bash
"$PYTHON_BIN" -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

Do not accidentally introduce a newline into `PYTHON_BIN`.  This previous form
is wrong:

```bash
export PYTHON_BIN="/public_hw/home/sci_bfu/.conda/envs/my_env/
  bin/python"
```

It can make merge/append commands run with the wrong Python and fail with:

```text
ModuleNotFoundError: No module named 'torch'
```

## 6. Submitting Full-Loop Packages

For Stage IV, after extraction:

```bash
cd ~/bkz/Fu_FFLO/active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624
unset OUTPUT_ROOT RUN_ID CONFIG_JSON
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
bash scripts/check_stageiv_3d_submit_ready.sh
```

Then submit:

```bash
export CONFIRM_STAGEIV_FULL_LOOP=1
nohup bash scripts/submit_stageiv_3d_full_loop.sh > active_phase_topology_3d_t_ja_mu_from_scratch_v1.nohup.log 2>&1 &
```

Use one physical line for the `nohup ... > log 2>&1 &` command.  Splitting after
`>` caused a shell parse error in a previous run.

## 7. Slurm Monitoring

Basic monitoring commands:

```bash
squeue -u sci_bfu
tail -n 120 <run>.nohup.log
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Reason --parsable2
scontrol show job <job_id_or_array_task>
```

For package-specific status:

```bash
bash scripts/check_stageiv_3d_hpc_status.sh
```

Use `sacct` to distinguish:

```text
COMPLETED      successful task
FAILED         process exited nonzero
TIMEOUT        Slurm time limit hit
CANCELLED      batch step was killed, often from timeout
PENDING        waiting for resources or array limit
```

`Reason=JobArrayTaskLimit` usually means the array concurrency cap is active;
it is not itself a calculation failure.

## 8. GPU Node Exclusion

We avoid `gpuh01` and `gpuh14` by default for this project.

Slurm scripts should include:

```bash
#SBATCH --exclude=gpuh01,gpuh14
```

The Stage IV submit script also uses:

```bash
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01,gpuh14}"
sbatch --exclude="${EXCLUDE_NODES}"
```

and the exact Slurm script has a runtime guard:

```bash
host_name="$(hostname 2>/dev/null || true)"
if [ "${host_name%%.*}" = "gpuh01" ] || [ "${host_name%%.*}" = "gpuh14" ]; then
  echo "[error] refusing to run on excluded node ${host_name%%.*}" >&2
  exit 42
fi
```

Evidence:

```text
gpuh01: earlier PyTorch/CUDA driver mismatch.
gpuh14: repeated short exact-array failures with CUDA device busy/unavailable,
        including Stage IV job 82381_0, which failed after 15 seconds while
        sibling ranks completed normally on other nodes.
```

If another node is temporarily unreliable, extend the exclude list only when
there is concrete evidence from `sacct`, Slurm logs, or cluster status.

## 9. Common Failure Modes

### CUDA Device Busy or Unavailable

Typical log:

```text
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

This has occurred when one Slurm array task starts on a node whose GPU is not
available.  If other ranks completed and only one or a few ranks failed, prefer
failed-rank recovery instead of restarting the full loop.

### Time Limit

For Slurm arrays:

```text
TIMEOUT
batch CANCELLED 0:15
```

means the job exceeded Slurm time.  Increasing a running job's time limit may
fail with permission errors.  Later jobs can exceed earlier two-hour behavior if
their submitted or updated limits differ; inspect `scontrol show job` and
`sacct`.

### Wrong Python Environment

If manual post-processing fails with missing `torch`, first check:

```bash
echo "$PYTHON_BIN"
"$PYTHON_BIN" -c "import torch; print(torch.__version__)"
```

Then rerun merge or append commands with the correct `PYTHON_BIN`.

### Partial Archive

If `tar -xzf` reports unexpected EOF, do not continue with that directory.
Re-upload the archive.

## 10. Failed-Rank Recovery

For packages that support failed-rank recovery, use:

```bash
bash scripts/inspect_stageiv_failed_task.sh <job_id>
ITER=<failed_iteration> FAILED_RANKS=<rank_list> bash scripts/recover_stageiv_failed_exact_iter.sh
START_ITER=<next_iteration> bash scripts/resume_stageiv_3d_full_loop.sh
```

Example:

```bash
bash scripts/inspect_stageiv_failed_task.sh 81110
ITER=0 FAILED_RANKS=3 bash scripts/recover_stageiv_failed_exact_iter.sh
START_ITER=1 bash scripts/resume_stageiv_3d_full_loop.sh
```

Use recovery only when:

```text
selected-points partition exists
completed shards are present or preserved
only listed ranks failed
the merge/append boundary for that iteration is not already corrupted
```

Do not rerun the whole full loop just because one exact-array rank hit a
transient CUDA error.

## 11. Merge, Append, and StopController Checks

Manual merge pattern:

```bash
TAG=$(printf "%03d" "$ITER")
NEXT=$(printf "%03d" "$((ITER+1))")

"$PYTHON_BIN" -m ml_phase.hpc \
  --merge \
  --run-dir "$RUN_DIR" \
  --iteration "$ITER" \
  --world-size "$WORLD_SIZE" \
  --positive-delta-gap-tol "$POSITIVE_DELTA_GAP_TOL"
```

Manual append pattern:

```bash
"$PYTHON_BIN" -m ml_phase.append_trusted \
  --dataset "$RUN_DIR/dataset_iter${TAG}.npz" \
  --trusted-exact "$RUN_DIR/iter${TAG}/exact_trusted_iter${TAG}.npz" \
  --output-npz "$RUN_DIR/dataset_iter${NEXT}.npz" \
  --output-csv "$RUN_DIR/dataset_iter${NEXT}.csv" \
  --output-root "$OUTPUT_ROOT"
```

StopController or post-run audits should be report-only unless explicitly
running a production active-learning loop.

## 12. Collection and Return

After completion:

```bash
bash scripts/collect_stageiv_3d_results.sh
```

Stage IV expected result archive:

```text
ML_Phase_StageIV_Topology3D/stageiv_3d_topology_full_loop_results.tar.gz
```

After downloading or extracting locally, run the package checker:

```bash
RETURN_PATH=ML_Phase_StageIV_Topology3D bash scripts/check_stageiv_3d_return_bundle.sh
```

Then run post-run report-only audits:

```bash
REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_all_postrun_reports.sh
```

The reference dataset is validation-only.  It must not be merged into Stage IV
training.

## 13. What Counts as Completion

Package readiness is not scientific completion.

For Stage IV, completion requires at least:

```text
returned production run directory
dataset_iter025.npz
last-five cumulative datasets
3D thermodynamic/topology convergence audit
hidden fixed-mu slice validation
post-run bundle decision
final Stage IV report built from returned production data
```

The current local Stage IV status is:

```text
package_validation_status = pass
handoff_status = ready_for_hpc_submit
goal_status = package_ready_hpc_pending
production_run_returned = false
final_dataset_exists = false
postrun_bundle_exists = false
```

## 14. Do-Not-Do List

- Do not run from a partial extraction.
- Do not use `gpuh01` or `gpuh14`.
- Do not split shell redirection commands across lines.
- Do not use a broken `PYTHON_BIN` path.
- Do not export a stale `OUTPUT_ROOT`, `RUN_ID`, or `CONFIG_JSON` from an older
  active-learning run before launching a self-contained package.  Stage IV 3D
  packages built after 2026-06-24 freeze these identifiers and reject conflicting
  values, but clearing the shell is still the safest habit:

```bash
unset OUTPUT_ROOT RUN_ID CONFIG_JSON
```

- Do not merge Stage III or Phase-II datasets into cold-start Stage IV training.
- Do not treat submit-ready or preflight success as convergence.
- Do not mark missing boundaries or surfaces as zero shift.
- Do not modify thermodynamic criteria, topology formulas, acquisition rules,
  StopController thresholds, or exact-oracle tolerances from an operational HPC
  package.
- Do not restart a full loop for a single transient rank failure before trying
  rank-level recovery.

## 15. Frozen Package Identity Guard

For self-contained production packages, the run identity should be part of the
package, not part of the login-shell environment.  The Stage IV 3D package now
generates every operational shell script with:

```text
expected output_root = ML_Phase_StageIV_Topology3D
expected run_id      = active_phase_topology_3d_t_ja_mu_from_scratch_v1
expected config      = configs/stageiv_3d_production.json
```

If a shell already contains a conflicting `OUTPUT_ROOT`, `RUN_ID`, or
`CONFIG_JSON`, the script exits before selecting points or submitting Slurm
jobs.  This prevents a Stage IV job name from silently writing into an older
2D topology run directory.
