# 2026-06-30 Stage V Slurm-Selection Hotfix

## Context

Stage V run:

```text
run_id = stagev_acqv2_boundary_support_learned_residual_3d_v1
output_root = ML_Phase_StageV_AcqV2
latest good dataset = dataset_iter014
last completed exact iteration = iter013
```

The exact Slurm array for iter013 completed normally.  The outer control log
then stopped with:

```text
[done] Stage V iter 013; wrote dataset_iter014
Terminated
```

There was no `iter014` directory afterward, so the failure happened before the
next exact array submission.  The likely failed step is acquisition selection
for iter014, which trains/scores the Stage V reward model and candidate pool.
That step was still being run by the outer shell on the login node.

## Decision

Move Stage V selection into a separate Slurm job:

```text
scripts/slurm_stagev_acqv2_select.sh
```

The submit loop now does:

```text
selection Slurm job
then exact Slurm array
then merge / append / reward update
```

This avoids long CPU-heavy Python work on the login node.  Both the selection
job and exact array exclude:

```text
gpuh01,gpuh14
```

## What Did Not Change

This hotfix does not change:

```text
exact oracle
Hamiltonian
thermodynamic phase criterion
topology definition
StopController thresholds
acquisition score formula
Delta/q tolerances
completed datasets
```

## Local Outputs

```text
hpc_hotfix_scripts/stagev_acqv2_slurm_selection_hotfix/
hpc_hotfix_scripts/stagev_acqv2_slurm_selection_hotfix.zip
hpc_packages/stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc.tar.gz
reports/stagev_acqv2_hpc_package/stagev_acqv2_package_summary.json
```

The regenerated package archive hash is:

```text
4e98e14ca398afb93e8842bd51d2020e435ed16ad1100011939ed178a64d62c3
```

The hotfix zip was updated to a marker-based v2 after the first application
attempt found that the existing submit script did not match the original exact
text block.  The v2 zip hash is:

```text
0387e2f24b8bd2a7548a5a4bd9a728d3464c25dce64b6a321076d947c718dd3b
```

## Existing HPC Run Recovery Commands

Upload `stagev_acqv2_slurm_selection_hotfix.zip` into the existing Stage V HPC
package root and extract it.  Then run:

```bash
python hpc_hotfix_scripts/stagev_acqv2_slurm_selection_hotfix/apply_stagev_acqv2_slurm_selection_hotfix.py
```

Resume from the latest good dataset:

```bash
export CONFIRM_STAGEV_PRODUCTION=1
START_ITER=14 FINAL_EXACT_ITER=17 nohup bash scripts/resume_stagev_acqv2_full_loop.sh \
  > stagev_resume_iter014_to017_slurmselect.nohup.log 2>&1 &
```

Expected log markers:

```text
[submit] Stage V selection iter 014: job ...
[wait] Stage V exact iter 014-selection: squeue active: ...
[submit] Stage V exact iter 014: job ...
```

If the selection Slurm job fails, inspect:

```bash
sacct -j <selection_job_id> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList,Reason --parsable2
tail -n 200 slurm-select-<selection_job_id>.out
```
