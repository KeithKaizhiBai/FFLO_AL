# 2026-06-25 Stage IV gpuh14 Exclusion And Rank-0 Recovery

## Context

The Stage IV 3D cold-start package
`active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624`
started exact iteration 000 as Slurm array job 82381.

## Observed Failure

`sacct -j 82381` showed:

```text
82381_0  stageiv3d_exact  FAILED     1:0  00:00:15  gpuh14
82381_1  stageiv3d_exact  COMPLETED  0:0  03:44:48  gpuh03
82381_2  stageiv3d_exact  COMPLETED  0:0  03:48:35  gpuh05
82381_3  stageiv3d_exact  COMPLETED  0:0  03:44:21  gpuh05
82381_4  stageiv3d_exact  COMPLETED  0:0  03:45:01  gpuh05
82381_5  stageiv3d_exact  COMPLETED  0:0  03:47:27  gpuh05
82381_6  stageiv3d_exact  COMPLETED  0:0  03:44:56  gpuh05
82381_7  stageiv3d_exact  COMPLETED  0:0  03:46:08  gpuh05
```

Rank 0 failed before producing its shard, while ranks 1-7 completed and wrote
`exact_shard_rank001_of008.npz` through `exact_shard_rank007_of008.npz`.

## Decision

Treat `gpuh14` as an excluded GBU GPU node for this project, alongside the
previously excluded `gpuh01`.

Future Stage IV packages should default to:

```bash
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01,gpuh14}"
sbatch --exclude="${EXCLUDE_NODES}"
```

Generated Slurm scripts should also refuse runtime hostnames `gpuh01` and
`gpuh14`.

## Recommended Recovery

Do not restart the full Stage IV loop solely because one rank failed on an
excluded node.  Recover the missing rank 0 shard for iteration 000, merge, append
`dataset_iter001`, then resume from `START_ITER=1`.

This is an operational recovery.  It does not alter thermodynamic phase labels,
topology formulas, acquisition logic, exact-oracle tolerances, or StopController
criteria.
