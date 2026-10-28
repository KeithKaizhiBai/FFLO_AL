# Stage V-v3 Sparse-Arc Boundary-Support HPC Package

Run id:

```text
stagev_v3_sparse_arc_boundary_support_3d_v1
```

This is a strict cold-start Stage V-v3 package.  It does not use Stage IV,
Stage V-v1, Stage V-v2, Stage V-v2-tail checkpoints, coordinates, or labels
for production initialization.

## Smoke

```bash
cd ~/bkz/Fu_FFLO/stagev_v3_sparse_arc_boundary_support_3d_v1_hpc
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
bash scripts/run_stagev_v3_smoke.sh
```

## Production

```bash
export CONFIRM_STAGEV3_PRODUCTION=1
nohup bash scripts/submit_stagev_v3_full_loop.sh > stagev_v3_sparse_arc_boundary_support_3d_v1.nohup.log 2>&1 &
```

Resume:

```bash
export CONFIRM_STAGEV3_PRODUCTION=1
START_ITER=24 nohup bash scripts/resume_stagev_v3_full_loop.sh > stagev_v3_sparse_arc_boundary_support_3d_v1_resume_iter024.nohup.log 2>&1 &
```

Monitor:

```bash
bash scripts/monitor_stagev_v3.sh
squeue -u "$USER"
```

Collect:

```bash
bash scripts/collect_stagev_v3_results.sh
```

## Safety

- Slurm scripts and `sbatch` commands exclude `gpuh01,gpuh14`.
- Runtime guards exit before exact calculation if the node is `gpuh01`.
- Production refuses to run unless `CONFIRM_STAGEV3_PRODUCTION=1`.
- `START_ITER=0` refuses to overwrite an existing run directory.

## Git Snapshot

- commit: `8405b03639043bbc5dc43e154ed7f7465e77cc75`
- dirty worktree: `True`
