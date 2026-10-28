# Stage V-v2 Slurm Accounting Race Note

Date: 2026-07-03

## Context

During the Stage V-v2 multi-head 3D active-learning production run
(`stagev_v2_multihead_boundary_learning_3d_v1`), the outer loop stopped twice
with messages that looked like Slurm job failures:

```text
[error] Stage V-v2 exact iter 024: job 86408 failed: 86408_0:RUNNING:0:0
[error] Stage V-v2 selection iter 084: job 87599 failed: 87599:RUNNING:0:0
```

Subsequent `sacct` inspection showed these were not real failed calculations.
For example:

```text
87599|stagev2_select|COMPLETED|0:0|00:01:00|2026-07-03T07:57:47|2026-07-03T07:58:47|gpuh11|None
```

The corresponding selection outputs existed:

```text
iter084/selected_points.csv
iter084/selected_points_metadata.csv
iter084/selected_points_rank000_of008.csv
...
iter084/selected_points_rank007_of008.csv
```

Likewise, the earlier exact-array job `86408` later showed all array tasks as
`COMPLETED|0:0`, with all rank shard files present.

## Diagnosis

The likely cause is a Slurm accounting race condition in the HPC wrapper, not a
physics or oracle failure.

The submit script first waits until `squeue` no longer reports the job. It then
immediately queries `sacct`. On this cluster, `sacct` can briefly lag behind
`squeue` and report a just-finished job as:

```text
RUNNING:0:0
```

The current wrapper treats any state other than `COMPLETED` as terminal failure.
Therefore transient accounting states such as `RUNNING`, `PENDING`, or
`COMPLETING` can incorrectly stop the loop.

## Recommended Wrapper Fix

For future packages, update the job wait helper so that:

1. `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, and related
   terminal failure states stop the loop.
2. Transient states such as `RUNNING`, `PENDING`, `CONFIGURING`, and
   `COMPLETING` after `squeue` disappearance trigger a short retry loop rather
   than immediate failure.
3. The wrapper verifies the expected output files before deciding whether a
   completed selection or exact job needs recovery.

This is an operational robustness fix only. It does not change:

```text
Hamiltonian
thermodynamic labels
topology labels
exact oracle
Delta-q search
acquisition score
reward model
numerical tolerances
```

## Report Integration Note

When the Stage V-v2 return report is written, record this as an HPC orchestration
issue:

```text
Two apparent Stage V-v2 stops were traced to Slurm accounting latency: `squeue`
had cleared the job while `sacct` still reported `RUNNING:0:0`. Later accounting
and output-file checks confirmed successful completion. The production data were
continued from the correct iteration rather than restarted. Future wrappers
should retry transient `sacct` states before declaring failure.
```

Do not count these wrapper stops as failed exact calculations, GPU failures, or
evidence of numerical instability.
