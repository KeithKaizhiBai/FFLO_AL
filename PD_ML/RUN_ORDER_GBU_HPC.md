# GBU HPC Upload Bundle Run Order

This bundle is prepared for the Greater Bay University HPC cluster manual 2025 v1.0.

Cluster assumptions from the manual:

- Login node is not for computation-heavy work.
- CPU partitions include `Intel` and `AMD`.
- H100 partition is `NV_H100`.
- Current `sinfo` shows `NV_H100` nodes `gpuh[01-14]`; each H100 node provides NVIDIA H100 GPUs.
- Submit jobs with `sbatch`/`srun`; GPU resources use `--gres=gpu:N`.
- `salloc` is disabled.
- Environment management uses `conda` and `module`.

## 0. Upload and unpack

```bash
tar -xzf gbu_active_learning_hpc_upload.tar.gz
cd gbu_active_learning_hpc_upload
```

If needed, set your Python environment name:

```bash
export CONDA_ENV=your_env_name
```

The current module list shows CUDA as `compiler/cuda/cuda-12.8.1`. If the CUDA module name differs, check `module av` and set:

```bash
export CUDA_MODULE=compiler/cuda/cuda-12.8.1
```

## 1. Check environment

```bash
bash 00_check_env_gbu.sh
```

Required Python packages:

```text
numpy
pandas
scipy
matplotlib
torch with CUDA
```

On login or CPU nodes, `torch.cuda.is_available()` may be `False`. The decisive CUDA check is inside an `NV_H100` job.

## 2. Generate candidate shards

Smoke-test mode:

```bash
export RUN_ID=active_boundary_h100_smoke
export POINTS_PER_ITER=32
export WORLD_SIZE=4
export DRY_RUN_FLAG=--dry-run
export N_ENSEMBLE=2
export REG_EPOCHS=20
export CLS_EPOCHS=20
export BATCH_SIZE=1024
bash 01_prepare_candidate_shards.sh
```

Wait until the CPU job finishes, then check:

```bash
ls ML_Phase/active_runs/${RUN_ID}/iter000
```

You should see `selected_points_rank000_of004.csv` through `selected_points_rank003_of004.csv`.

Production candidate generation:

```bash
export RUN_ID=active_boundary_h100_v1
export POINTS_PER_ITER=512
export WORLD_SIZE=8
export DRY_RUN_FLAG=
export N_ENSEMBLE=5
export REG_EPOCHS=240
export CLS_EPOCHS=240
export BATCH_SIZE=512
bash 01_prepare_candidate_shards.sh
```

## 3. Submit H100 exact oracle

After candidate shards exist:

```bash
export RUN_ID=active_boundary_h100_smoke
export ITER=0
export WORLD_SIZE=4
bash 02_submit_h100_oracle.sh
```

For production:

```bash
export RUN_ID=active_boundary_h100_v1
export ITER=0
export WORLD_SIZE=8
bash 02_submit_h100_oracle.sh
```

Monitor:

```bash
squeue
```

Cancel if needed:

```bash
scancel JOBID
```

## 4. Merge exact shards

After all H100 array tasks finish:

```bash
export RUN_ID=active_boundary_h100_smoke
export ITER=0
export WORLD_SIZE=4
bash 03_merge_h100_shards.sh
```

Production:

```bash
export RUN_ID=active_boundary_h100_v1
export ITER=0
export WORLD_SIZE=8
bash 03_merge_h100_shards.sh
```

Merged output:

```text
ML_Phase/active_runs/<RUN_ID>/iter000/exact_merged_iter000.npz
```

## 5. Build report

```bash
export RUN_ID=active_boundary_h100_smoke
bash 04_build_report.sh
```

Report outputs:

```text
ML_Phase/reports/active_learning_phase_boundary_report.tex
ML_Phase/reports/active_learning_phase_boundary_report.pdf
```

## Notes

- Do not run expensive training or BdG calculations directly on the login node.
- Use `Intel` CPU jobs for model training/acquisition.
- Use `NV_H100` GPU array jobs for exact BdG point evaluation.
- The first run should be the 32-point smoke test before any 512-point production run.
