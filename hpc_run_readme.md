# HPC Run README

This package is a clean runnable source package. It is intended to start a new
discovery-mode active-learning run from random exact BdG seed points.

It does not include previous `ML_Phase_512` result outputs, previous active
runs, or the warm-start exact dataset.

## What This Package Runs

Default discovery run:

```text
RUN_ID=active_boundary_discovery_512seed_256x50
RUN_MODE=discovery
CANDIDATE_DOMAIN_MODE=full
INITIALIZATION=random_grid
INITIAL_SEED_SIZE=512
  BATCH_SIZE_MAX=256
  SELECTION_MODE=stochastic
  ACTIVE_POOL_QUANTILE=0.90
  ACTIVE_POOL_REL_TO_P95=0.7
  ACTIVE_POOL_RULE=max_threshold
  ACTIVE_POOL_QUANTILE_SCHEDULE=piecewise
  ACTIVE_POOL_MAX_FRACTION_START=0.20
  ACTIVE_POOL_MAX_FRACTION_END=0.05
  B_DELTA_GATE_MODE=normal_sc_competition
  INTERIOR_FILTER_MODE=soft_penalty
  SAMPLING_POWER_SCHEDULE=piecewise
  SAMPLING_POWER=2.0
  SAMPLING_POWER_START=1.5
  SAMPLING_POWER_MID=2.5
  SAMPLING_POWER_END=4.0
  W_EXT_SCHEDULE=piecewise
  START_ITER=0
  N_ITERS=100
WORLD_SIZE=8
N_ENSEMBLE=5
REG_EPOCHS=240
CLS_EPOCHS=240
BATCH_SIZE=512
BOUNDARY_REFINEMENT_MODE=diagnostic
```

The current implemented active-learning code includes:

```text
random exact seed selection at iter000
full retraining on the cumulative accepted dataset each later iteration
full rectangular dense-grid candidate domain
stochastic acquisition sampling with soft observation and batch repulsion
normal/SC-gated Delta-boundary scoring to suppress deep-normal false boundary scores
high-information active-pool gating by A0_for_pool before stochastic sampling
adaptive batch size when the active pool has fewer high-information points
q-window expansion in the exact oracle
Delta refinement near the normal/SC boundary
quality-gate filtering before appending trusted training points
StopController convergence checks after exact merge and trusted append
diagnostic extraction of normal/SC and uniform/FFLO boundary segments
```

Discovery mode explicitly does not use:

```text
warm-start exact data as training initialization
finite-T prior candidate-band pruning
finite_t_band_width
fixed regional quotas
midpoint boundary selection
large-radius historical hard exclusion
```

Boundary extraction is diagnostic only; all post-seed selected exact-call
candidates come from the ML-guided dense-grid acquisition score.

## Run Commands

On the cluster:

```bash
tar -xzf hpc_upload_qdelta_discovery_512seed_256x50_YYYYMMDD_HHMMSS.tar.gz
cd hpc_upload_qdelta_discovery_512seed_256x50_YYYYMMDD_HHMMSS
chmod +x run_discovery_512x50.sh run_discovery_512x50_background.sh
bash run_discovery_512x50.sh
```

To keep the loop running after closing the terminal, use the background
wrapper:

```bash
tar -xzf hpc_upload_qdelta_discovery_512seed_256x50_YYYYMMDD_HHMMSS.tar.gz
cd hpc_upload_qdelta_discovery_512seed_256x50_YYYYMMDD_HHMMSS
chmod +x run_discovery_512x50.sh run_discovery_512x50_background.sh
bash run_discovery_512x50_background.sh
```

The default log name is timestamped. To force a fixed log file:

```bash
LOG_FILE=discovery_active_loop.log bash run_discovery_512x50_background.sh
tail -f discovery_active_loop.log
```

Check job status:

```bash
squeue -u $USER
cat discovery_active_loop.pid
```

The active loop submits candidate-generation and exact-oracle jobs through
Slurm, waits for them, merges exact shards, appends trusted points, and then
continues to the next iteration.

## Override Parameters

Example:

```bash
RUN_ID=my_test_run \
N_ITERS=5 \
INITIAL_SEED_SIZE=128 \
BATCH_SIZE_MAX=64 \
ACTIVE_POOL_QUANTILE=0.90 \
SAMPLING_POWER=2.0 \
WORLD_SIZE=4 \
RANDOM_SEED=7 \
bash run_discovery_512x50.sh
```

The discovery wrapper unsets `WARM_START`, `FINITE_T_BAND_WIDTH`, and
`RESUME_DATASET` so the run starts from random exact seed points.

StopController defaults:

```text
STOP_MIN_ITERATIONS=5
STOP_PATIENCE=4
STOP_MAX_ITERATIONS=N_ITERS
STOP_MAP_TOL=0.002
STOP_SURPRISE_TOL=0.05
```

The controller stops on main phase-boundary convergence. Selected acquisition
score, q-edge rate, and rerun-required rate are diagnostics and cleanup
warnings, not mandatory main-loop stop gates.

## Output

Results are written under:

```text
ML_Phase/active_runs/<RUN_ID>/
ML_Phase/figures/
ML_Phase/reports/
```
