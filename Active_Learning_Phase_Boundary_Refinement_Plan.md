# Active-Learning-Assisted Phase Boundary Refinement Plan

This plan is written for Codex 5.3 to execute directly in this repository.

Target problem: build an active-learning workflow that uses existing exact BdG phase-diagram data as a warm start, trains uncertainty-aware ML models, proposes new points near phase boundaries, calls the exact CUDA solver only on those points, and iterates until the relevant boundaries converge.

## Recommendation

Use the existing `.npz` as the initial training set, then switch to an active-learning loop that computes additional points only where the model is uncertain or where boundary indicators are large.

Do not start from scratch with online learning only.

Reason:

- The existing file already contains `156 x 138 = 21528` exact BdG points over `kT in [0.01, 0.5]` and `JA in [0.01, 1.2]`.
- This is enough to train a useful first surrogate/classifier and to estimate where the current grid already resolves boundaries.
- Starting without it would waste GPU time rediscovering the same coarse structure.
- Pure online learning is useful only after a warm-start model exists; otherwise uncertainty is high almost everywhere and the acquisition function degenerates into coarse scanning.

Recommended strategy:

```text
existing exact .npz
    -> build supervised dataset
    -> train initial surrogate/classifier
    -> score a dense candidate grid
    -> select uncertain/high-gradient/new-boundary points
    -> exact CUDA computation on selected points
    -> append new exact points
    -> retrain
    -> repeat until boundary convergence
```

## Existing Inputs

Use this file as the warm-start dataset:

```text
eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6/
eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6.npz
```

Arrays expected in the file:

```text
kT_vec              shape (n_kt,)
JA_vec              shape (n_ja,)
q_vec               shape (n_q,)
eta_matrix          shape (n_ja, n_kt)
q_opt_matrix        shape (n_ja, n_kt)
delta_opt_matrix    shape (n_ja, n_kt)
ic_plus_matrix      shape (n_ja, n_kt)
ic_minus_matrix     shape (n_ja, n_kt)
delta0              scalar
```

The matrix convention is:

```text
row = JA index
col = kT index
```

## Deliverable

Create a new package:

```text
ml_phase/
    __init__.py
    config.py
    dataset_builder.py
    labels.py
    models.py
    acquisition.py
    exact_oracle.py
    hpc.py
    active_refine.py
    evaluate.py
    plot_active_learning.py
    report_builder.py
```

Create output directories only when needed:

```text
ML_Phase/
    datasets/
    models/
    active_runs/
    figures/
    hpc_jobs/
    reports/
```

Create cluster job scripts:

```text
scripts/
    slurm_active_refine.sh
    slurm_exact_oracle_array.sh
```

Create a LaTeX report template:

```text
report/
    active_learning_phase_boundary_report.tex
    figures/
```

Keep the existing CUDA scripts intact unless a small, clearly isolated helper function must be added.

## Phase Labels and Boundary Indicators

Start with labels that can be computed directly from the existing `.npz`.

Use conservative thresholds in `ml_phase/config.py`:

```python
DELTA_EPS = 1e-3
Q_EPS = 1e-2
ETA_STRONG = 0.5
BOUNDARY_MARGIN = 0.08
```

Initial labels:

```python
if delta_opt < DELTA_EPS:
    phase = "normal"
elif abs(q_opt) < Q_EPS:
    phase = "uniform_SC"
else:
    phase = "FFLO"
```

Diode labels:

```python
eta_sign = "eta_pos" if eta > 0 else "eta_neg" if eta < 0 else "eta_zero"
strong_diode = abs(eta) > ETA_STRONG
```

Boundary indicators:

```text
normal/SC boundary:
    delta_opt close to DELTA_EPS
    or large gradient of delta_opt

uniform/FFLO boundary:
    abs(q_opt) close to Q_EPS
    or large gradient/jump of q_opt

eta sign boundary:
    eta close to 0
    or neighboring grid cells have opposite eta sign

strong diode boundary:
    abs(abs(eta) - ETA_STRONG) small
```

If `tfflo_1d_cuda.py` exposes a reliable `compute_z2` function that can be called pointwise with `q_opt` and `delta_opt`, add optional labels:

```text
trivial_FFLO / topological_FFLO
```

Do not block the first active-learning implementation on `Z2`. Implement `Z2` as a second milestone.

## ML Models

Use models that are easy to train and support uncertainty estimates.

Baseline implementation:

1. Regression model for continuous outputs:

```text
X = [kT, JA]
Y = [delta_opt, q_opt, eta, ic_plus, ic_minus]
```

Use an ensemble of `sklearn.neural_network.MLPRegressor`, `RandomForestRegressor`, or `ExtraTreesRegressor`.

2. Classification model for phase labels:

```text
X = [kT, JA]
Y = phase_label
```

Use `RandomForestClassifier`, `ExtraTreesClassifier`, or an ensemble MLP classifier.

3. Uncertainty:

For tree models:

```text
regression uncertainty = per-tree prediction standard deviation
classification uncertainty = 1 - max(class_probability)
```

For MLP ensembles:

```text
regression uncertainty = ensemble prediction standard deviation
classification uncertainty = entropy or 1 - max(mean_probability)
```

Avoid adding PyTorch training infrastructure in the first version unless scikit-learn is unavailable.

## Acquisition Function

The active learner should score candidate points on a dense candidate grid, not only on the existing grid.

Candidate grid:

```text
kT in [0.0, 0.56]
JA in [0.0, 2.12]
```

But apply physics-aware masks:

- prioritize the existing computed data domain first: `kT <= 0.5`, `JA <= 1.2`;
- allow candidates up to the finite-T boundary extent only when they are near the analytic boundary from `finite_T_phase_diagram.m`;
- avoid negative `kT`.

Candidate scoring:

```text
score =
    w_cls_uncertainty * classification_uncertainty
  + w_reg_uncertainty * normalized_regression_uncertainty
  + w_delta_boundary  * delta_boundary_score
  + w_q_boundary      * q_boundary_score
  + w_eta_boundary    * eta_zero_score
  + w_gradient        * predicted_gradient_score
  + w_diversity       * distance_to_existing_points
```

Default weights:

```python
W_CLS_UNCERTAINTY = 1.0
W_REG_UNCERTAINTY = 0.8
W_DELTA_BOUNDARY = 1.0
W_Q_BOUNDARY = 1.0
W_ETA_BOUNDARY = 0.7
W_GRADIENT = 0.7
W_DIVERSITY = 0.3
```

Define boundary scores so they peak near expected boundaries:

```python
delta_boundary_score = exp(-abs(delta_pred - DELTA_EPS) / delta_scale)
q_boundary_score = exp(-abs(abs(q_pred) - Q_EPS) / q_scale)
eta_zero_score = exp(-abs(eta_pred) / eta_scale)
```

Use non-maximum suppression or k-means clustering in candidate space before selecting points. This prevents the acquisition function from selecting hundreds of nearly identical points on the same small segment.

Default acquisition size:

```text
50 to 200 new exact points per iteration
```

Start with 64 points per iteration.

## Exact CUDA Oracle

Implement `ml_phase/exact_oracle.py` as a thin wrapper around existing code in `eta_phase_diagram_cuda.py`.

Required function:

```python
def evaluate_points(points, cfg=None, device=None) -> dict:
    """
    points: array-like of shape (n_points, 2), columns [kT, JA]

    returns:
        {
            "kT": shape (n_points,),
            "JA": shape (n_points,),
            "eta": shape (n_points,),
            "q_opt": shape (n_points,),
            "delta_opt": shape (n_points,),
            "ic_plus": shape (n_points,),
            "ic_minus": shape (n_points,),
        }
    """
```

Use these existing functions:

```python
from eta_phase_diagram_cuda import (
    EtaPhaseConfig,
    build_q_vec,
    compute_omega_min_q_batch,
    compute_current_from_omega,
    find_eta_from_jq,
)
```

Important implementation detail:

`compute_omega_min_q_batch(kt_batch, ja_batch, ...)` computes the Cartesian product `kt_batch x ja_batch`. For arbitrary active-learning candidate points, first implement the simple robust version:

```text
for each point or small point chunk:
    kt_batch = tensor([kT])
    ja_batch = tensor([JA])
    compute exact output
```

After correctness is verified, optionally optimize by grouping points that share the same `kT` or `JA`.

Do not rewrite the BdG kernel in the ML package.

## Supercomputer and Multi-H100 Execution

The final workflow is intended to run on a supercomputer with one or more NVIDIA H100 GPUs. Design the active-learning code so that CPU-side model training and GPU-side exact BdG computation are separated.

High-level HPC workflow:

```text
login/head node or CPU job:
    build dataset
    train ML models
    score candidate grid
    select new points
    write selected_points_iterXXX.csv
    submit GPU exact-oracle array job

GPU compute nodes with H100:
    read selected point shard
    evaluate exact BdG outputs
    write exact_result_shard_rankYYY_ofZZZ.npz

postprocess job:
    merge exact shards
    append to active-learning dataset
    retrain or schedule next iteration
```

Do not run the full active-learning loop as one monolithic job that keeps all GPUs idle while CPU-side training or plotting happens.

### Point Partitioning

Active-learning selected points must be partitioned before exact CUDA computation.

Add this function in `ml_phase/hpc.py`:

```python
def partition_points(points, world_size, strategy="round_robin"):
    """
    points: array of shape (n_points, 2), columns [kT, JA]
    world_size: number of GPU ranks or SLURM array tasks

    returns:
        list of point arrays, length world_size
    """
```

Supported strategies:

```text
round_robin:
    robust default; balances arbitrary selected points.

contiguous:
    useful for reproducible debugging and small point counts.

cost_aware:
    optional future mode; estimate cost from kT, JA, expected nonzero Delta, or candidate uncertainty.
```

Default to `round_robin`.

Write shards as:

```text
ML_Phase/active_runs/<run_id>/iterXXX/selected_points.csv
ML_Phase/active_runs/<run_id>/iterXXX/selected_points_rank000_of008.csv
ML_Phase/active_runs/<run_id>/iterXXX/selected_points_rank001_of008.csv
...
```

Each GPU rank writes:

```text
ML_Phase/active_runs/<run_id>/iterXXX/exact_shard_rank000_of008.npz
ML_Phase/active_runs/<run_id>/iterXXX/exact_shard_rank001_of008.npz
...
```

The merge step writes:

```text
ML_Phase/active_runs/<run_id>/iterXXX/exact_merged_iterXXX.npz
ML_Phase/active_runs/<run_id>/dataset_iterXXX.npz
```

### Rank-Aware Exact Oracle

`ml_phase/exact_oracle.py` must support both local and HPC modes.

CLI:

```powershell
python -m ml_phase.exact_oracle `
  --points-file ML_Phase/active_runs/<run_id>/iter000/selected_points_rank000_of008.csv `
  --output-file ML_Phase/active_runs/<run_id>/iter000/exact_shard_rank000_of008.npz `
  --device cuda:0
```

It must also support automatic SLURM environment detection:

```text
SLURM_ARRAY_TASK_ID
SLURM_ARRAY_TASK_COUNT
CUDA_VISIBLE_DEVICES
```

If `--rank` and `--world-size` are supplied, use them explicitly. Otherwise infer them from SLURM when available.

Important:

- one process should control one H100 unless a later benchmark shows that multiple processes per GPU improve throughput;
- default precision should match the existing result, currently `float64`;
- write partial results after each point or small chunk so that a time-limit interruption loses minimal work;
- include `rank`, `world_size`, `hostname`, `CUDA_VISIBLE_DEVICES`, and elapsed time in shard metadata.

### SLURM Job Scripts

Add `scripts/slurm_exact_oracle_array.sh` as a template, not a site-specific final script.

Template content should include placeholders:

```bash
#!/bin/bash
#SBATCH --job-name=al_exact
#SBATCH --partition=<gpu_partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7

set -euo pipefail

module purge
# module load cuda
# source <conda_or_venv_activate>

RUN_ID="${RUN_ID:?set RUN_ID}"
ITER="${ITER:?set ITER}"
WORLD_SIZE="${SLURM_ARRAY_TASK_COUNT}"
RANK="${SLURM_ARRAY_TASK_ID}"

python -m ml_phase.exact_oracle \
  --run-id "${RUN_ID}" \
  --iteration "${ITER}" \
  --rank "${RANK}" \
  --world-size "${WORLD_SIZE}" \
  --device cuda:0
```

Add `scripts/slurm_active_refine.sh` for CPU-side training/acquisition and optional job submission. Keep actual `sbatch` submission optional because cluster policies differ.

### Multi-GPU Iteration Modes

Implement two modes in `active_refine.py`:

1. `--mode local`

```text
single process; useful for workstation dry-runs and smoke tests.
```

2. `--mode hpc`

```text
train and acquire points;
write point shards;
print the exact sbatch command to run;
exit unless --submit is explicitly set.
```

Do not auto-submit jobs by default. Require `--submit` for actual `sbatch` submission.

Required HPC commands:

```powershell
python -m ml_phase.active_refine `
  --mode hpc `
  --run-id active_boundary_h100 `
  --iterations 1 `
  --points-per-iter 512 `
  --world-size 8 `
  --dry-run
```

Then, for a real exact pass:

```powershell
python -m ml_phase.active_refine `
  --mode hpc `
  --run-id active_boundary_h100 `
  --iterations 1 `
  --points-per-iter 512 `
  --world-size 8
```

This command should write shards and print the corresponding `sbatch` command.

### Initial H100 Scaling Plan

Use staged scaling:

```text
Stage 0:
    local dry-run, no GPU exact calls.

Stage 1:
    one H100, 8 selected points.

Stage 2:
    one H100, 64 selected points.

Stage 3:
    4 H100 array tasks, 256 selected points.

Stage 4:
    8 H100 array tasks, 512 selected points.

Stage 5:
    production active learning, 5-10 iterations, 512-2048 points per iteration depending on queue time.
```

For each stage, record:

```text
points per GPU
seconds per point
GPU memory peak if available
failure rate
exact-call throughput
boundary improvement per exact call
```

Use these measurements to choose `points_per_iter` and `world_size`.

## Dataset Format

`ml_phase/dataset_builder.py` should convert the warm-start `.npz` into a flat table:

```text
kT, JA, delta_opt, q_opt, eta, ic_plus, ic_minus, phase_label, eta_sign, strong_diode
```

Save as:

```text
ML_Phase/datasets/warm_start_dataset.npz
ML_Phase/datasets/warm_start_dataset.csv
```

Active-learning iterations append new exact points to:

```text
ML_Phase/active_runs/<run_id>/dataset_iter000.npz
ML_Phase/active_runs/<run_id>/dataset_iter001.npz
...
```

Never overwrite old iteration datasets. Each iteration must be reproducible.

## Active Refinement Loop

Implement `ml_phase/active_refine.py`.

CLI:

```powershell
python -m ml_phase.active_refine `
  --warm-start eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6\eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6.npz `
  --run-id active_boundary_v1 `
  --mode local `
  --iterations 5 `
  --points-per-iter 64
```

Loop:

```text
1. Load current exact dataset.
2. Train regression ensemble.
3. Train phase classifier.
4. Generate dense candidate grid.
5. Predict labels and continuous outputs on candidate grid.
6. Compute acquisition scores.
7. Select diverse top candidate points.
8. If `mode=local`, evaluate selected points with exact CUDA oracle.
9. If `mode=hpc`, write selected point shards and exact-oracle job metadata.
10. Merge exact shard results when available.
11. Append new exact points.
12. Save iteration dataset, model metadata, candidate scores, and plots.
13. Update LaTeX report artifacts.
14. Stop if convergence criteria are met.
```

Convergence criteria:

```text
stop if:
    selected candidate scores stop decreasing for 2 iterations
    and boundary position changes by less than 1 candidate-grid spacing
    and validation error on held-out exact points does not improve
```

Do not rely on a single convergence metric.

## Evaluation

Implement `ml_phase/evaluate.py`.

Required outputs:

```text
Delta_opt RMSE
q_opt RMSE
eta RMSE
ic_plus RMSE
ic_minus RMSE
phase classification accuracy
boundary F1 score
number of exact BdG calls
estimated exact-call reduction compared with a full dense grid
```

Hold out 15% of the warm-start exact grid for validation. Use a stratified split over phase labels when possible.

Boundary F1:

1. Convert exact labels to boundary masks using nearest-neighbor grid adjacency.
2. Convert predicted labels to boundary masks.
3. Compare masks after allowing a one-grid-cell tolerance.

## LaTeX Report

In addition to code and figures, create a clear, readable LaTeX report that documents the feature construction, classification definitions, ML models, acquisition function, HPC partitioning, and active-learning results.

Main file:

```text
report/active_learning_phase_boundary_report.tex
```

Output PDF:

```text
ML_Phase/reports/active_learning_phase_boundary_report.pdf
```

The report should be compileable with `pdflatex` or `latexmk`.

Add `ml_phase/report_builder.py` to populate result tables and figure paths from the latest run metadata. Do not hard-code final numerical results in the `.tex` file if they can be loaded from JSON/CSV outputs.

Required report structure:

```text
Title:
    Active-Learning-Assisted Phase Boundary Refinement for 1D Altermagnetic FFLO TSC

1. Physical and Computational Motivation
2. Warm-Start Dataset
3. Feature Construction
4. Feature Classification and Phase Labels
5. ML Surrogate and Classifier Models
6. Uncertainty Quantification
7. Acquisition Function
8. Exact BdG Oracle
9. Multi-H100 Partitioning and Supercomputer Execution
10. Evaluation Metrics
11. Active-Learning Results
12. Discussion and Next Steps
```

### Report Section Details

Feature construction section must define:

```text
raw coordinates:
    kT, JA

regression targets:
    Delta_opt, q_opt, eta, Ic_plus, Ic_minus

optional future physics features:
    minimum BdG gap
    Fermi velocity asymmetry
    normal-state spin splitting
    Z2 formula terms
    OBC edge-mode indicators
```

Feature classification section must include a table:

```text
Class / label             Definition
normal                    Delta_opt < DELTA_EPS
uniform_SC                Delta_opt >= DELTA_EPS and |q_opt| < Q_EPS
FFLO                      Delta_opt >= DELTA_EPS and |q_opt| >= Q_EPS
eta_pos                   eta > 0
eta_neg                   eta < 0
strong_diode              |eta| > ETA_STRONG
```

ML model section must include:

```text
input feature vector
regression targets
classification targets
model family
ensemble size
random seed
training/validation split
normalization policy
```

Acquisition section must include the mathematical score:

```latex
S(x) =
w_c U_c(x)
+ w_r U_r(x)
+ w_\Delta B_\Delta(x)
+ w_q B_q(x)
+ w_\eta B_\eta(x)
+ w_g G(x)
+ w_d D(x).
```

HPC section must include:

```text
number of selected points per iteration
number of H100 tasks
partitioning strategy
points per GPU
seconds per point
merge strategy
failure/restart behavior
```

Results section must include at minimum:

```text
learning curve
uncertainty map
acquisition map
selected points map
phase prediction map
table of exact-call reduction
```

Report figures should be copied or symlinked into:

```text
report/figures/
```

If symlinks are not portable on the cluster filesystem, copy the files.

## Plotting

Implement `ml_phase/plot_active_learning.py`.

Generate:

```text
ML_Phase/figures/<run_id>_iterXXX_eta_exact_points.png
ML_Phase/figures/<run_id>_iterXXX_phase_prediction.png
ML_Phase/figures/<run_id>_iterXXX_uncertainty.png
ML_Phase/figures/<run_id>_iterXXX_acquisition.png
ML_Phase/figures/<run_id>_iterXXX_selected_points.png
ML_Phase/figures/<run_id>_learning_curve.png
```

Plots must show:

- original warm-start points;
- newly selected active-learning points;
- predicted boundaries;
- exact boundary points when available.

Use the same axis convention as `plot_eta_phase_diagram.py`:

```text
x-axis: k_B T / t
y-axis: J_A / t
```

Do not display `kT < 0`.

## Implementation Order

Execute in this order:

1. Create `ml_phase/config.py` and `ml_phase/labels.py`.
2. Create `ml_phase/dataset_builder.py` and verify it loads the existing `.npz`.
3. Create `ml_phase/models.py` using scikit-learn tree ensembles first.
4. Create `ml_phase/acquisition.py`.
5. Create `ml_phase/plot_active_learning.py` for static diagnostics.
6. Create `ml_phase/hpc.py` with point partitioning and shard merge helpers.
7. Create `ml_phase/exact_oracle.py` and test on 1 to 3 points only.
8. Add rank-aware exact-oracle CLI support.
9. Create `ml_phase/active_refine.py`.
10. Add `--mode local`, `--mode hpc`, `--world-size`, and `--submit` controls.
11. Create `scripts/slurm_exact_oracle_array.sh`.
12. Create `scripts/slurm_active_refine.sh`.
13. Create `report/active_learning_phase_boundary_report.tex`.
14. Create `ml_phase/report_builder.py`.
15. Run a dry-run mode that selects points but does not call CUDA.
16. Run one real local/GPU active-learning iteration with 8 points.
17. Run one H100 array smoke test with 4-8 shards.
18. If correct, run 5 iterations with 64-512 points per iteration.

## Dry-Run Mode

`active_refine.py` must support:

```powershell
python -m ml_phase.active_refine --dry-run --iterations 1 --points-per-iter 64
```

Dry-run behavior:

- train on the warm-start dataset;
- score candidate grid;
- select points;
- save plots and candidate CSV;
- do not call `exact_oracle.evaluate_points`.

This is mandatory because exact CUDA calls are expensive.

## First Real Test

Run:

```powershell
python -m ml_phase.active_refine `
  --run-id active_boundary_smoke `
  --iterations 1 `
  --points-per-iter 8
```

Validate:

- all 8 selected points are in physical range;
- exact CUDA oracle returns finite `eta`, `q_opt`, `delta_opt`;
- appended dataset has original points plus 8 new points;
- plots show the selected points near plausible boundaries.

## First H100 Array Test

Run a dry-run partition first:

```powershell
python -m ml_phase.active_refine `
  --mode hpc `
  --run-id active_boundary_h100_smoke `
  --iterations 1 `
  --points-per-iter 32 `
  --world-size 4 `
  --dry-run
```

Validate:

- `selected_points.csv` exists;
- four rank files exist;
- each rank file has 8 points, or differs by at most one point if the count is not divisible;
- no selected point has `kT < 0`;
- shard metadata JSON records `world_size=4`.

Then run the exact array job on H100 using the local site's SLURM settings. After completion, run the merge command:

```powershell
python -m ml_phase.hpc `
  --merge `
  --run-id active_boundary_h100_smoke `
  --iteration 0 `
  --world-size 4
```

Validate:

- all shard files are present;
- merged file contains 32 exact points;
- failed or missing shards are reported clearly;
- dataset append is deterministic and does not duplicate existing points.

## First LaTeX Report Test

After one dry-run or smoke run, build the report:

```powershell
python -m ml_phase.report_builder `
  --run-id active_boundary_h100_smoke `
  --output report/active_learning_phase_boundary_report.tex
```

Then compile:

```powershell
pdflatex -interaction=nonstopmode -output-directory ML_Phase/reports report/active_learning_phase_boundary_report.tex
```

Validate:

- PDF compiles without missing figure errors;
- feature classification table is readable;
- ML model table is readable;
- HPC partitioning section includes `world_size`, selected points, and points per GPU;
- figures are not clipped and axes match `k_B T/t` and `J_A/t`.

## Notes for Codex 5.3

- Prefer simple, correct, inspectable code over complex model infrastructure.
- Do not introduce new deep-learning dependencies unless scikit-learn is unavailable or clearly insufficient.
- Keep all generated data under `ML_Phase/`.
- Do not modify `eta_phase_diagram_cuda.py` except for small reusable helpers if absolutely necessary.
- Treat `Z2` and topological labels as milestone 2. The first milestone is active refinement of normal/SC, uniform/FFLO, and `eta` sign/strong-diode boundaries.
- Always include a dry-run path before triggering expensive exact CUDA calculations.
- Always partition selected exact points before H100 execution; never assume one process should evaluate the full selected set in production.
- Keep CPU-side ML training/acquisition separate from GPU exact evaluation to avoid wasting allocated H100 time.
- Do not auto-submit SLURM jobs unless `--submit` is explicitly passed.
- Write restartable shard outputs so interrupted jobs can resume or be merged partially with clear diagnostics.
- Maintain the LaTeX report as a first-class deliverable, not an afterthought.
- Record every active-learning iteration with a JSON metadata file containing model type, random seed, thresholds, candidate-grid size, selected points, and exact-call count.

## Expected Scientific Narrative

The method should be described as:

```text
We use an existing exact BdG phase diagram as a warm-start dataset. An ensemble surrogate learns the map from thermodynamic parameters to order-parameter, momentum, current, and diode-efficiency observables. Classification uncertainty, regression uncertainty, and physics-informed boundary indicators define an acquisition function. New exact BdG calculations are then concentrated near normal/FFLO, uniform/FFLO, topological-candidate, and diode-efficiency boundaries, reducing the number of exact BdG calls needed for a refined phase diagram.
```

The ML model is not a replacement for the exact solver. It is a scheduler that decides where exact BdG calculations are most valuable.
