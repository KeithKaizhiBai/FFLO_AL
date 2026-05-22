# Continuous-Learning Plan for Active-Learning MLP Surrogates

Date: 2026-05-13

## 1. Purpose

This document clarifies how to extend the current active-learning workflow from
full retraining at every iteration to optional checkpoint-based continuous
learning.

The goal is not to change the physical model or the trusted exact labels. The
ML models remain surrogate schedulers for selecting new exact BdG calculations.
The exact BdG oracle remains the source of physical truth.

## 2. Current Training Logic

Current behavior:

```text
iter000: initialize new MLP ensembles from random seeds; train on warm-start data
iter001: initialize new MLP ensembles from random seeds; train on warm-start + iter000 accepted data
iter002: initialize new MLP ensembles from random seeds; train on warm-start + iter000 + iter001 accepted data
...
```

Each active-learning iteration trains on the cumulative accepted dataset:

```text
current_dataset = warm_start + all previous trusted/training-eligible exact points
```

It does not train only on the newest points.

The current code does not save neural-network weights. It saves:

```text
run_config.json
metrics_history.json
iterXXX/metrics.json
iterXXX/selected_points.csv
iterXXX/candidate_scores.csv
iterXXX/exact_merged_iterXXX.npz
dataset_iterXXX.npz
dataset_iterXXX.append.json
```

These files record data, selected points, exact outputs, metrics, and training
configuration. They do not store `state_dict` values for the MLPs.

## 3. Why Full Retraining Was Reasonable

Full retraining is conservative for this research setting:

```text
1. The training objective at each iteration is exactly tied to the cumulative
   trusted dataset.
2. No previous optimizer state can bias the model toward early active-learning
   mistakes.
3. Validation metrics are easier to compare because each iteration refits from
   a clean initialization protocol.
4. The implementation is simple and auditable.
5. Small MLPs on the current dataset are cheap relative to exact BdG calls.
```

This is a standard active-learning approach: after new labels are acquired, the
surrogate is refitted on the full labeled set.

## 4. Why Continuous Learning May Be Useful

Continuous learning means the next iteration starts from the previous
iteration's trained network weights instead of random initialization.

Possible benefits:

```text
1. Reduce training time when datasets or networks become larger.
2. Preserve a smooth trajectory of learned surrogate functions across
   iterations.
3. Enable analysis of how the surrogate changes as new exact points are added.
4. Allow direct reproduction of acquisition maps from saved checkpoints.
5. Make later report figures more auditable because each plotted surrogate can
   be loaded directly rather than retrained.
```

Possible risks:

```text
1. The model can retain early-iteration bias if later data contradicts earlier
   surrogate structure.
2. Continuing optimization on a changed dataset can over-emphasize old optimizer
   momentum or learning-rate history.
3. Classifier output dimensions depend on the phase classes present in the
   training split; this must be fixed explicitly.
4. Input and target scalers can shift after new data are appended. Reusing
   weights while changing scalers must be handled carefully.
5. If checkpoints are not tied to dataset hashes, it becomes easy to load a
   model trained on the wrong dataset.
```

## 5. Proposed Design

Add continuous learning as an explicit optional mode:

```text
model_init_mode = full_retrain | warm_start_weights
```

Default should remain:

```text
model_init_mode = full_retrain
```

This keeps existing results and behavior reproducible.

### 5.1 Checkpoint Contents

For each iteration, save one checkpoint directory:

```text
ML_Phase/models/<run_id>/iter000/
ML_Phase/models/<run_id>/iter001/
...
```

Each directory should contain:

```text
regressor_member000.pt
regressor_member001.pt
...
classifier_member000.pt
classifier_member001.pt
...
scalers.npz
classes.npy
checkpoint_metadata.json
```

`scalers.npz` should include:

```text
reg_x_mean
reg_x_std
reg_y_mean
reg_y_std
cls_x_mean
cls_x_std
```

`checkpoint_metadata.json` should include:

```text
run_id
iteration
dataset_path
dataset_sha256
n_samples
n_features
n_regression_targets
phase_classes
n_ensemble
hidden_dim
reg_epochs_completed
cls_epochs_completed
lr
weight_decay
batch_size
val_fraction
seed
torch_version
device
created_at
```

### 5.2 Loading Rules

When `model_init_mode = warm_start_weights`:

```text
1. For iter000, initialize from random seeds and train normally.
2. For iterNNN where NNN > 0, load checkpoint from iter(NNN-1).
3. Verify architecture compatibility:
   - input dimension unchanged
   - regression target dimension unchanged
   - hidden_dim unchanged
   - n_ensemble unchanged
   - classifier phase classes unchanged or explicitly remapped
4. Fit new scalers on the current cumulative dataset.
5. Decide how to map old weights under new scaling before continuing training.
6. Continue training on the full cumulative dataset, not only new points.
7. Save a new checkpoint after training.
```

The safest first implementation should keep phase classes fixed to:

```text
normal = 0
uniform_SC = 1
FFLO = 2
```

This avoids classifier output-dimension changes.

### 5.3 Scaler Policy

The regression model uses standardized inputs and standardized regression
targets. The classifier uses standardized inputs.

Changing scalers while reusing weights is nontrivial because the first and last
layers are expressed in scaled coordinates.

Recommended initial policy:

```text
freeze_scalers_with_checkpoint = true
```

That means:

```text
1. iter000 fits input/target scalers.
2. later warm-start-weight iterations reuse the previous checkpoint scalers.
3. new data are transformed through the same scalers.
4. the model continues training on the full cumulative dataset.
```

This is simple and makes the loaded weights mathematically compatible.

Alternative future policy:

```text
refit_scalers_each_iter = true
```

This should only be used if we also implement weight transformation or accept
that the loaded weights are only approximate initial parameters under shifted
coordinates.

## 6. Training Schedule Options

Option A: full retraining baseline.

```text
Every iteration trains from scratch for reg_epochs and cls_epochs.
This is the current behavior.
```

Option B: checkpoint warm-start with full cumulative training.

```text
Every iteration loads previous weights, then trains on the full cumulative
dataset for a smaller number of epochs.
```

Suggested starting values:

```text
reg_epochs_iter0 = 240
cls_epochs_iter0 = 240
reg_epochs_continue = 60
cls_epochs_continue = 60
```

Option C: checkpoint warm-start with mixed replay weighting.

```text
Train on full cumulative data but oversample new points or boundary-band points.
```

This is more complicated and should not be implemented first, because it
changes the statistical weighting of exact data and may distort boundary
interpretation.

## 7. Validation Requirements

Continuous learning is acceptable only if it is compared against the current
full-retraining baseline.

Minimum validation:

```text
1. Run one smoke active-learning loop with full_retrain.
2. Run the same loop with warm_start_weights.
3. Confirm both produce finite predictions and valid candidate scores.
4. Compare validation metrics:
   - Delta RMSE
   - q RMSE
   - eta RMSE
   - Ic+ RMSE
   - Ic- RMSE
   - phase accuracy
5. Compare selected point distributions, not necessarily exact equality.
6. Verify loaded checkpoints reproduce predictions before further training.
7. Verify checkpoint metadata dataset hashes match the intended dataset.
```

Expected behavior:

```text
The selected points need not be identical to full retraining, because optimizer
history changes the surrogate. However, the chosen regions should remain
physically interpretable: normal/SC boundary, uniform/FFLO boundary,
high-uncertainty regions, and q/Delta refinement-risk regions.
```

Failure conditions:

```text
1. NaN or Inf in predictions, uncertainty, or acquisition scores.
2. Classifier loses one phase class because class mapping was not fixed.
3. Checkpoint loads despite dataset hash mismatch.
4. Validation metrics degrade strongly relative to full retraining without a
   clear reason.
5. Acquisition collapses to previously selected or rerun-required points.
```

## 8. Implementation Plan

Step 1: add serialization helpers in `ml_phase/models.py`.

```text
save_model_bundle(bundle, path, metadata)
load_model_bundle(path, cfg, device)
```

Keep them function-based. Do not introduce a manager class.

Step 2: add fixed classifier class support.

```text
phase_classes = [0, 1, 2]
```

The classifier should always output three logits in this project, even if a
small validation split lacks one class.

Step 3: add active-learning config fields.

```text
model_init_mode: "full_retrain"
save_checkpoints: true
continue_reg_epochs: 60
continue_cls_epochs: 60
freeze_checkpoint_scalers: true
```

Step 4: update `ml_phase/active_refine.py`.

Logic:

```text
if iteration == start_iteration or model_init_mode == "full_retrain":
    train from random initialization
else:
    load previous checkpoint
    continue training on current cumulative dataset
save checkpoint after training
```

For resumed HPC runs, checkpoint lookup must respect `start_iteration`.

Step 5: write metadata and hash checks.

Use SHA256 of `dataset_iterXXX.npz` for the dataset used to train each
checkpoint. If the dataset hash does not match, refuse to load unless an
explicit override is added later.

Step 6: add tests or smoke checks.

Suggested local commands:

```text
python -m ml_phase.active_refine --... --iterations 1 --points-per-iter small --dry-run
python -m ml_phase.active_refine --... --model-init-mode warm_start_weights --iterations 2 --points-per-iter small --dry-run
```

The exact command should follow the existing project scripts once the CLI
fields are added.

Step 7: update reports.

The active-learning report should state:

```text
model_init_mode
whether checkpoints were saved
checkpoint path for the latest iteration
whether scalers were frozen or refit
number of continuation epochs
```

## 9. Scientific Interpretation Rules

The neural network parameters are not physical observables. They should not be
interpreted as evidence for a phase boundary by themselves.

Allowed interpretations:

```text
1. Surrogate uncertainty identifies where exact BdG calls are useful.
2. Checkpoint evolution can show how the scheduler changes as new exact labels
   are added.
3. Agreement between full retraining and continuous learning strengthens
   confidence in acquisition stability.
```

Not allowed:

```text
1. Treating neural-network smoothness as proof that a phase boundary is smooth.
2. Treating checkpoint-to-checkpoint parameter drift as a physical phase
   transition.
3. Replacing exact BdG labels with classifier labels in the final phase diagram.
```

## 10. Recommended First Decision

Do not immediately replace full retraining.

Recommended next implementation:

```text
1. Add checkpoint saving first while keeping full_retrain as the default.
2. Confirm that checkpoints can be loaded and reproduce predictions.
3. Add warm_start_weights as an optional experimental mode.
4. Compare full_retrain and warm_start_weights on a small active-learning run.
5. Only then consider using continuous learning for production runs.
```

This staged approach preserves the current scientific baseline while enabling
direct study of whether continuous learning is useful for this project.

## 11. Report Figure Revision Plan

The active-learning report should not only show the latest iteration figures.
For the 512-point run, the latest iteration selected only three points and none
entered the training dataset. Showing only that iteration hides the main useful
output of the run: 999 accepted training-eligible points from the first two
iterations.

Add report figures that distinguish:

```text
warm-start exact points
all selected active-learning points
accepted/training-eligible active-learning points
selected but rerun-required points
finite-resolution normal/SC boundary-band points
```

Recommended figure set:

```text
1. cumulative_selected_points.png
   - background: warm-start phase labels or neutral light density
   - overlay: selected points from every available iteration
   - color or marker by iteration index

2. cumulative_accepted_points.png
   - overlay only points that entered the trusted/training dataset
   - separate marker for boundary-band normal points
   - separate marker for rerun-required rejected points

3. latest_prediction_with_cumulative_points.png
   - background: latest ML phase prediction or uncertainty map
   - overlay: cumulative accepted points, not only latest selected points
```

The report text should state explicitly:

```text
The latest iteration may not represent the useful data added by the run. The
scientific output is the accepted cumulative dataset, while the latest selected
points show only the current state of acquisition saturation.
```

This change is report/diagnostic only. It does not modify the exact solver,
phase definitions, or active-learning selection rule.

## 12. Boundary-Focused Radius Schedule

The current dense-grid exclusion radius is useful for broad exploration, but it
is not an appropriate final rule for boundary precision. If the goal is to learn
and refine phase boundaries, then an active-learning loop should not stop only
because the current radius has saturated candidate selection.

The better interpretation is:

```text
At a given radius, active learning resolves the boundary only down to that
radius-dependent sampling scale. Once this stage saturates, the next step is a
smaller-radius boundary-refinement stage, not immediate termination of the
scientific workflow.
```

However, radius reduction should be boundary-aware. A good staged workflow is:

```text
Stage A: global exploration
    Use a moderate radius to cover disconnected uncertain regions and avoid
    redundant exact calls.

Stage B: boundary extraction
    Extract current normal/SC and uniform_SC/FFLO boundary brackets from the
    accepted exact dataset. Do not treat eta_zero as equal priority unless the
    research question is specifically about diode-response sign changes.

Stage C: local boundary refinement
    Generate targets near boundary brackets or uncertainty ridges. Use a
    smaller boundary-specific radius than the global exploration radius.

Stage D: convergence check
    Compare boundary displacement, near-boundary phase accuracy, and
    Delta/q-specific errors between stages. Continue shrinking the local radius
    only where the boundary displacement remains larger than the target
    tolerance.
```

This means the radius is not a physical tolerance by itself. It is an
acquisition-control parameter. The physical/numerical convergence criterion
should be based on boundary stability and exact-oracle quality flags.

Recommended radius hierarchy:

```text
global exploration radius:
    used for broad candidate diversity;
    can be relatively large.

boundary local radius:
    used only near extracted boundary brackets;
    should be smaller than the global radius.

duplicate-only exclusion:
    prevents exact repeated coordinates;
    should remain active at every stage.
```

For this project, a more suitable next direction is not simply "more loops with
the same acquisition radius." A better direction is:

```text
1. merge or exclude already trusted points from previous active-learning runs;
2. extract accepted-data boundary brackets;
3. sample midpoint/normal-direction targets around those brackets;
4. use a reduced boundary-specific radius;
5. preserve strict quality gates for q-window and Delta ambiguity;
6. measure boundary displacement after each refinement stage.
```

## 13. Correction on Warm-Start Interpretation at High JA

The high-\(J_A\) warm-start data should not be treated as fully reliable phase
boundary evidence. The original warm-start calculation used a fixed q-search
window. It was already expected that for \(J_A/t \gtrsim 1.2\), the true
optimal \(q\) can rapidly move beyond that original fixed range.

Therefore:

```text
1. Similarity between a new ML-predicted boundary and a previous 128-point
   active-learning boundary is not automatically explained by the warm-start
   phase labels.
2. In high-JA regions, the ML surrogate may be predicting a boundary outside
   the accurately resolved warm-start domain.
3. Exact validation with q-window expansion is required before interpreting
   those high-JA ML-predicted boundaries as physical.
```

The report should distinguish:

```text
warm-start resolved region
high-JA q-window-risk region
active-learning exact points with q-window expansion
ML-predicted boundary outside the originally trusted warm-start domain
```

This distinction is essential because the active-learning model can be useful
precisely by proposing boundary candidates outside the original trusted exact
coverage. Those candidates still require exact-oracle confirmation before they
become phase-boundary evidence.

## 14. Detailed Workflow for Boundary-Focused Active Learning

This section turns the design discussion into a staged workflow. It covers both
the current full-retraining baseline and the future optional continuous-learning
mode.

### 14.1 Stage 0: Define Scientific Targets

Inputs:

```text
physical phase labels:
    normal, uniform_SC, FFLO

primary boundaries:
    normal/SC
    uniform_SC/FFLO

secondary response boundaries:
    eta_zero
    strong_diode
```

Decisions:

```text
1. Treat normal/SC and uniform_SC/FFLO as primary phase-boundary targets.
2. Treat eta_zero as a response-function sign boundary, not a thermodynamic
   phase boundary.
3. Treat high-JA q-window-risk regions separately from already trusted
   warm-start regions.
```

Outputs:

```text
target_boundary_types.json
trusted_region_definition.json
```

### 14.2 Stage 1: Build the Initial Dataset

Inputs:

```text
warm_start_dataset.npz
optional previous trusted active-learning datasets
optional historical exclusion points
```

Procedure:

```text
1. Load warm-start exact data.
2. If continuing scientific refinement rather than performing a clean restart,
   merge previous trusted active-learning points.
3. Keep rerun-required points out of the training dataset.
4. Preserve metadata for q-window status, Delta ambiguity, and boundary-band
   normal points.
```

Outputs:

```text
dataset_iter000.npz
dataset_iter000.csv
dataset_iter000_metadata.json
```

Diagnostics:

```text
n_samples
phase counts
q_unresolved count
delta_unresolved count
boundary-band normal count
duplicate coordinate count
high-JA trusted/untrusted coverage
```

### 14.3 Stage 2: Train or Load ML Surrogates

Inputs:

```text
current cumulative accepted dataset
training configuration
optional checkpoint from previous iteration
```

Procedure under the current baseline:

```text
1. Initialize regression MLP ensemble from random seeds.
2. Initialize phase-classifier MLP ensemble from random seeds.
3. Train both on the full cumulative accepted dataset.
4. Save metrics and, after checkpoint support is implemented, save weights.
```

Procedure under future continuous-learning mode:

```text
1. Load previous checkpoint if model_init_mode = warm_start_weights.
2. Verify architecture, class mapping, scaler policy, and dataset hash.
3. Continue training on the full cumulative accepted dataset.
4. Save a new checkpoint with metadata.
```

Outputs:

```text
metrics.json
optional model checkpoint
prediction fields on candidate grid
uncertainty fields on candidate grid
```

Required ML diagnostics:

```text
global validation RMSE for Delta, q, eta, Ic+, Ic-
global phase accuracy
boundary-band validation RMSE for Delta, q, eta
near-boundary phase classification error
classification uncertainty distribution
ensemble regression uncertainty distribution
```

### 14.4 Stage 3: Build Candidate Pools

Use multiple candidate pools instead of one global pool:

```text
global exploration pool:
    broad candidate grid over the full parameter domain

primary boundary pool:
    candidates near extracted normal/SC and uniform_SC/FFLO brackets

high-JA q-risk pool:
    candidates in regions where original fixed q window was unreliable

rerun pool:
    previous rerun-required points that are scientifically worth resolving
```

Each pool should carry its own exclusion rule:

```text
global exploration:
    use moderate diversity radius

primary boundary refinement:
    use smaller boundary-specific radius

high-JA q-risk:
    require q-window expansion and q-quality metadata

rerun pool:
    do not exclude solely by proximity; decide by exact-oracle ambiguity status
```

Outputs:

```text
candidate_scores_global.csv
candidate_scores_boundary.csv
candidate_scores_high_ja_q_risk.csv
candidate_scores_rerun.csv
```

### 14.5 Stage 4: Acquisition and Point Selection

Inputs:

```text
ML predictions
ML uncertainty
boundary brackets
exact-quality metadata
candidate pools
radius schedule
```

Selection objectives:

```text
1. reduce phase-boundary uncertainty;
2. reduce Delta/q regression error near boundaries;
3. validate ML-predicted boundaries outside trusted warm-start regions;
4. resolve q-window and Delta ambiguity where scientifically useful;
5. avoid exact duplicate coordinates.
```

Recommended quota structure:

```text
normal/SC boundary targets: largest quota
uniform_SC/FFLO boundary targets: large quota
high-JA q-risk targets: controlled quota
global exploration targets: smaller quota after main boundaries are known
eta_zero/strong_diode targets: limited quota unless response-function physics
    becomes the main question
rerun-required targets: explicit quota, not accidental reselection
```

Outputs:

```text
selected_points.csv
selection_diagnostics.json
selected_points_by_pool.csv
```

Selection diagnostics:

```text
selected count by pool
selected count by boundary type
minimum distance to accepted exact points
minimum distance to previous selected points
number excluded by duplicate-only rule
number excluded by radius rule
number selected in high-JA q-risk region
expected q-window expansion count
expected Delta-refinement count
```

### 14.6 Stage 5: Exact BdG Oracle Evaluation

Inputs:

```text
selected_points.csv
q-window policy
Delta-refinement policy
exact solver configuration
```

Procedure:

```text
1. Evaluate each selected point with the exact BdG oracle.
2. Expand q window where required, especially in high-JA regions.
3. Refine Delta near the normal/SC boundary.
4. Compare finite positive-Delta solutions against Delta = 0 where needed.
5. Mark each exact point as training-eligible, boundary-band normal, or
   rerun-required.
```

Outputs:

```text
exact_merged_iterXXX.npz
exact_training_iterXXX.npz
exact_trusted_iterXXX.npz
rerun_points.csv
merge_summary_iterXXX.json
```

Exact-oracle diagnostics:

```text
q_unresolved count
q_expanded count
q_edge_hit count
delta_refined count
delta_unresolved count
boundary-band normal count
rerun-required count
training-eligible count
```

### 14.7 Stage 6: Dataset Append and Boundary Extraction

Inputs:

```text
current dataset
exact_training_iterXXX.npz
exact_trusted_iterXXX.npz
merge_summary_iterXXX.json
```

Procedure:

```text
1. Append only training-eligible exact points.
2. Preserve boundary-band normal metadata.
3. Do not append rerun-required ambiguous points.
4. Extract updated normal/SC and uniform_SC/FFLO boundary brackets.
5. Compare updated boundaries with previous-stage boundaries.
```

Outputs:

```text
dataset_iterYYY.npz
dataset_iterYYY.append.json
boundary_segments_iterYYY/
boundary_displacement_iterYYY.json
```

Boundary diagnostics:

```text
boundary segment count by type
boundary displacement relative to previous stage
new boundary coverage in high-JA region
boundary gaps still lacking exact brackets
```

### 14.8 Stage 7: Stopping and Radius-Schedule Decisions

The loop should not stop only because one radius has saturated. Stopping should
require agreement between exact-boundary diagnostics, exact-oracle quality, ML
diagnostics, and acquisition diagnostics.

Use four classes of stopping criteria.

#### A. Exact Boundary Stability

Stop only if:

```text
normal/SC boundary displacement < boundary_position_tol
uniform_SC/FFLO boundary displacement < boundary_position_tol
boundary displacement remains small for n_stable_stages consecutive stages
```

Boundary displacement should be measured on extracted boundary brackets or a
smoothed boundary representation, not by visual inspection.

#### B. Exact-Oracle Quality

Stop only if:

```text
q_unresolved count is below tolerance
Delta_unresolved requiring rerun is below tolerance
rerun-required points do not cluster along unresolved boundary segments
high-JA q-window-risk regions have been explicitly checked or marked unresolved
```

#### C. Boundary-Focused ML Error Convergence

ML loss/error should be included as an auxiliary stopping criterion.

Use boundary-focused metrics, not only global loss:

```text
near-boundary Delta RMSE
near-boundary q RMSE
near-boundary eta RMSE
near-boundary phase classification error
classification uncertainty near extracted boundaries
regression ensemble uncertainty near extracted boundaries
```

Stop only if:

```text
boundary-focused ML metrics plateau over n_stable_stages
global metrics do not show pathological degradation
ML uncertainty near the accepted boundary is below a chosen tolerance or no
longer decreases after smaller-radius refinement
```

Important caveat:

```text
ML loss convergence is not physical convergence by itself. It only means the
surrogate has stopped improving on the current training/validation
distribution. It must be combined with exact-boundary stability and
exact-oracle quality checks.
```

#### D. Acquisition Saturation at the Current Radius

At a fixed radius, saturation means:

```text
selected points below minimum target count
new unique training-eligible samples below threshold
candidate pool exhausted after duplicate and quality exclusions
```

If only D is true, do not stop the scientific workflow. Instead choose one:

```text
1. reduce boundary-specific radius;
2. switch from global exploration to boundary-local refinement;
3. add targeted high-JA q-risk candidates;
4. resolve selected rerun-required points with stricter exact settings;
5. stop only if A, B, and C are also satisfied.
```

### 14.9 Stage 8: Report and Figure Updates

Every report for a boundary-focused run should include:

```text
1. latest dataset phase counts;
2. cumulative selected points;
3. cumulative accepted/training-eligible points;
4. rerun-required points;
5. boundary-band normal points;
6. boundary displacement between stages;
7. boundary-focused ML error curves;
8. exact-oracle quality-flag curves;
9. current radius and radius schedule;
10. high-JA q-window-risk coverage.
```

This makes clear whether the run is improving physical boundary precision or
only exhausting a candidate pool.

### 14.10 Recommended Immediate Next Workflow

Implementation note, 2026-05-14:

```text
The first boundary-local hybrid selection mode has been implemented. It
extracts exact boundary brackets, selects normal_sc and uniform_fflo midpoint
targets with boundary_local_min_dist=0.00375, and fills the remaining budget
with high-JA q-risk/global acquisition candidates. Boundary displacement is
written as diagnostics but is not yet a hard stop parsed by hpc_active_loop.sh.
```

For the next active-learning revision, use this order:

```text
1. Add cumulative selected/accepted-point report figures.
2. Add boundary-focused diagnostics to the report.
3. Add checkpoint saving without changing the default full-retraining behavior.
4. Add boundary extraction and boundary-displacement summaries for each stage.
5. Add a smaller-radius boundary-local refinement mode.
6. Only then test warm_start_weights continuous learning as an optional mode.
```

This keeps the current scientific baseline intact while moving the workflow
toward true phase-boundary convergence rather than simple candidate exhaustion.
