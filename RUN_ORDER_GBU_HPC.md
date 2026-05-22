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

Use the fixed Python path when conda activation is unreliable:

```bash
export PROJECT_DIR=$PWD
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
export CONDA_ENV=
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

## 2. Recommended: run the active-learning loop driver

The preferred workflow after the `positive_delta_gap` fix is the loop driver:

```bash
export RUN_ID=active_boundary_loop_smoke
export N_ITERS=2
export POINTS_PER_ITER=32
export WORLD_SIZE=4
export N_ENSEMBLE=2
export REG_EPOCHS=20
export CLS_EPOCHS=20
export BATCH_SIZE=1024
export POSITIVE_DELTA_GAP_TOL=1e-8
export EXCLUDE_NODES=gpuh01
bash hpc_active_loop.sh
```

This performs, for each iteration:

```text
train/select candidates
write selected point shards
run H100 exact oracle with q expansion and Delta refinement
merge shards
write exact_merged_iterXXX.npz and exact_trusted_iterXXX.npz
append trusted points into dataset_iterYYY.npz
continue to the next iteration
```

Key outputs:

```text
ML_Phase/active_runs/<RUN_ID>/iterXXX/selected_points.csv
ML_Phase/active_runs/<RUN_ID>/iterXXX/exact_merged_iterXXX.npz
ML_Phase/active_runs/<RUN_ID>/iterXXX/exact_trusted_iterXXX.npz
ML_Phase/active_runs/<RUN_ID>/iterXXX/rerun_points.csv
ML_Phase/active_runs/<RUN_ID>/dataset_iterYYY.npz
ML_Phase/reports/active_learning_phase_boundary_report.tex
```

Current production scale from the warm-start dataset:

```bash
export RUN_ID=active_boundary_loop_512x50_r0075
export START_ITER=0
export N_ITERS=50
export POINTS_PER_ITER=512
export WORLD_SIZE=8
export N_ENSEMBLE=5
export REG_EPOCHS=240
export CLS_EPOCHS=240
export BATCH_SIZE=512
export EXCLUDE_NODES=gpuh01
bash hpc_active_loop.sh
```

The dense-grid acquisition radius in this upload is

```text
diversity_min_dist = existing_min_dist = 0.0075
```

This is half of the previous 0.015 radius. It is intended to reduce overly
coarse exclusion while retaining selected-point diversity and existing-data
deduplication.

If `iter000` has already been completed and `dataset_iter001.npz` exists,
continue from the next iteration instead of restarting:

```bash
export RUN_ID=active_boundary_loop_v1
export START_ITER=1
export N_ITERS=4
export POINTS_PER_ITER=128
export WORLD_SIZE=4
export N_ENSEMBLE=3
export REG_EPOCHS=80
export CLS_EPOCHS=80
export BATCH_SIZE=1024
export POSITIVE_DELTA_GAP_TOL=1e-8
export EXCLUDE_NODES=gpuh01
bash hpc_active_loop.sh
```

`EXCLUDE_NODES=gpuh01` avoids the H100 node that previously failed with an
NVIDIA driver mismatch. Set `EXCLUDE_NODES=` only after confirming that node is
usable with the current PyTorch/CUDA environment.

After the 25-iteration 128-point run, the recommended validation is a short
continuation using the boundary-band semantics:

```bash
export RUN_ID=active_boundary_loop_v1
export START_ITER=25
export N_ITERS=5
export POINTS_PER_ITER=128
export WORLD_SIZE=4
export N_ENSEMBLE=3
export REG_EPOCHS=80
export CLS_EPOCHS=80
export BATCH_SIZE=1024
export POSITIVE_DELTA_GAP_TOL=1e-8
export EXCLUDE_NODES=gpuh01
bash hpc_active_loop.sh
```

Check these files after the run:

```text
ML_Phase/active_runs/<RUN_ID>/iterXXX/merge_summary_iterXXX.json
ML_Phase/active_runs/<RUN_ID>/iterXXX/selection_diagnostics.json
ML_Phase/active_runs/<RUN_ID>/dataset_iterYYY.append.json
ML_Phase/reports/active_learning_phase_boundary_report.tex
```

Expected behavior:

```text
boundary-band normal points are training-eligible;
true rerun-required points exclude finite-resolution boundary-band normal
points;
previously identified boundary-band coordinates are suppressed in later
candidate selection.
```

## 3. Optional: one-iteration smoke workflow

For a single iteration without automatic append/retrain:

```bash
export RUN_ID=active_boundary_qdelta_smoke
export POINTS_PER_ITER=32
export WORLD_SIZE=4
export N_ENSEMBLE=2
export REG_EPOCHS=20
export CLS_EPOCHS=20
export BATCH_SIZE=1024
bash hpc_one_click_submit.sh
```

## 4. Legacy manual sequence

The older manual sequence is retained for debugging if you unpack a bundle that
contains the `00/01/02/03/04` helper scripts.

### Generate candidate shards

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

### Submit H100 exact oracle

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

### Merge exact shards

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

### Build report

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
