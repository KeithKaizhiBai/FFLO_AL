# ML Training and Active-Learning Q&A

## Question

Why does the active-learning workflow use two MLP ensembles, one for
regression and one for phase classification? If \(\Delta_{\rm opt}=0\) already
identifies the normal state, is the classifier redundant?

## Short Answer

The two networks learn different views of the same exact BdG dataset. The
regressor predicts continuous observables,
\((\Delta_{\rm opt}, q_{\rm opt}, \eta, I_c^+, I_c^-)\), while the classifier
learns the discrete phase boundary between normal, uniform_SC, and FFLO
regions. The classifier does not replace the exact phase rule. It provides
phase probabilities and class uncertainty, which are useful for deciding where
the next exact BdG calculations should be run.

## Technical Notes

In the exact dataset, the phase label is still assigned by the physical rule:

```text
normal:
    Delta_opt < DELTA_EPS

uniform_SC:
    Delta_opt >= DELTA_EPS and abs(q_opt) < Q_EPS

FFLO:
    Delta_opt >= DELTA_EPS and abs(q_opt) >= Q_EPS
```

The classifier is useful because a regression prediction near a threshold is
not the same as an exact threshold decision. Around normal/SC and
uniform_SC/FFLO boundaries, small regression errors in \(\Delta_{\rm opt}\) or
\(q_{\rm opt}\) can move a point across a hard phase threshold. The classifier
learns the phase decision surface directly and exposes uncertainty through
softmax probabilities.

## Report Use

Use this explanation in the methods section describing why the active learner
contains both a continuous surrogate and a phase classifier.

## Question

How do the classification results affect the next selected exact points? What
does acquisition mean here?

## Short Answer

Acquisition is the candidate-selection score used by active learning. It ranks
candidate \((k_B T/t, J_A/t)\) points by how valuable a new exact BdG
calculation would be. Classification contributes phase uncertainty, especially
near boundaries where the classifier gives comparable probabilities to two or
more phases.

## Technical Notes

A typical classification uncertainty term is

\[
U_{\rm cls}(x)=1-\max_c p(c|x).
\]

Large \(U_{\rm cls}\) indicates that the model is unsure whether the candidate
is normal, uniform_SC, or FFLO. These candidates often lie near phase
boundaries and therefore receive higher active-learning priority.

The full acquisition score also uses regression uncertainty, predicted
boundary proximity, numerical-risk terms, and diversity:

```text
classification uncertainty
regression ensemble uncertainty
Delta-boundary score
q-boundary score
eta-zero / strong-diode response score
q-window and high-JA risk
existing-point exclusion
diversity
```

The acquisition score is not a physical observable. It is a scheduling score
for choosing the next exact BdG batch.

## Report Use

Use this explanation in the active-learning workflow section and in the caption
or discussion of the acquisition-flow diagram.

## Question

What is the role of the quality gate?

## Short Answer

The quality gate decides whether a newly computed exact BdG point is reliable
enough to enter the training dataset. It separates clean trusted exact points,
finite-resolution boundary-band points, and points that require rerun or
additional numerical refinement.

## Technical Notes

The quality gate protects the ML dataset from numerical artifacts such as:

```text
q_opt near the scanned q-window edge
unresolved Delta ambiguity near the normal/SC boundary
nonfinite eta or current response
free-energy differences below the adopted tolerance
points requiring expanded q or refined Delta
```

Its outputs are:

```text
trusted exact:
    appended to the training dataset

boundary-band normal:
    appended with finite-resolution metadata

rerun-required:
    not appended; written to a rerun list
```

This step is important because the ML model can only learn the labels it is
given. Unresolved exact points should not silently become clean training
labels.

## Report Use

Use this explanation in the numerical safeguards section and in the workflow
figure discussion.

## Question

The validation report has RMSE values for several observables. Are these RMSEs
summed into one total training loss?

## Short Answer

No. The RMSE values are validation diagnostics. They are not summed with the
classification loss to form a joint training objective.

## Technical Notes

The implementation trains two ensembles separately:

\[
\mathcal{L}_{\rm reg}
  = N^{-1}\sum_i \|\hat y_{i,s}-y_{i,s}\|_2^2,
\]

where \(y_{i,s}\) is the standardized regression target, and

\[
\mathcal{L}_{\rm cls}
  = -N^{-1}\sum_i \log p_{i,y_i}.
\]

The validation RMSEs

\[
{\rm RMSE}(\Delta_{\rm opt}),\quad
{\rm RMSE}(q_{\rm opt}),\quad
{\rm RMSE}(\eta),\quad
{\rm RMSE}(I_c^+),\quad
{\rm RMSE}(I_c^-)
\]

are reported to diagnose which physical output the surrogate predicts well or
poorly. They do not define a single total loss across the regression and
classification branches.

## Report Use

Use this explanation near the ML-training architecture figure to avoid
misinterpreting validation metrics as the optimized objective.

## Question

Why are different loss functions used for continuous observables and phase
labels?

## Short Answer

Continuous BdG observables and phase labels are different statistical objects.
The regression branch predicts real-valued quantities, so it uses MSE on
standardized targets. The classifier predicts a categorical phase distribution,
so it uses cross entropy. This keeps the physical meaning of each prediction
clear, avoids arbitrary distances between phase labels, and gives useful phase
probabilities for active-learning acquisition.

## Technical Notes

For the regression branch,

\[
y_{\rm reg}=(\Delta_{\rm opt},q_{\rm opt},\eta,I_c^+,I_c^-)
\]

is continuous. The training target is standardized, and the loss is

\[
\mathcal{L}_{\rm reg}
  = N^{-1}\sum_i \|\hat y_{i,s}-y_{i,s}\|_2^2.
\]

This is appropriate for real-valued prediction errors and avoids one large
observable dominating the loss only because of its numerical scale.

For the phase classifier, the target is categorical:

```text
normal / uniform_SC / FFLO
```

The loss is

\[
\mathcal{L}_{\rm cls}
  = -N^{-1}\sum_i \log p_{i,y_i}.
\]

Using MSE on integer phase labels would incorrectly imply an ordered distance
between classes, for example that FFLO is "twice as far" from normal as
uniform_SC is. Cross entropy instead treats the output as a probability
distribution over classes and penalizes confident wrong predictions.

The separate losses also avoid an unnecessary multi-task weighting problem. If
one joint loss were used,

```text
L_total = L_reg + lambda * L_cls
```

then the choice of `lambda` would become an extra heuristic that can change the
balance between continuous-observable accuracy and phase-boundary accuracy.
The current implementation keeps the two training objectives separate and
combines their predictions only later in the acquisition score.

## Report Use

Use this explanation in the ML methods section when justifying the two-branch
surrogate architecture and the separate loss functions.

## Question

What is the difference between the two ML loss functions and the acquisition
function?

## Short Answer

The two loss functions train the neural networks. The acquisition function
does not train the networks; it uses the trained networks to rank uncomputed
candidate points. In other words, ML training learns a surrogate map from
known exact data, while acquisition decides where the next exact BdG
calculations should be placed.

## Technical Notes

The regression ensemble is trained by minimizing MSE on standardized
continuous targets:

```text
Delta_opt, q_opt, eta, Ic+, Ic-
```

The classifier ensemble is trained by minimizing cross entropy on the phase
labels:

```text
normal / uniform_SC / FFLO
```

These losses change the neural-network parameters through back-propagation.
The acquisition function is evaluated only after this training step. It
computes a score \(S(x)\) for each candidate point \(x=(k_B T/t,J_A/t)\).
That score is a hand-designed priority score, not a physical observable and
not a trainable loss.

The current acquisition score combines:

```text
classifier uncertainty
regression ensemble uncertainty
Delta-boundary proximity
q-boundary proximity
eta-zero proximity
gradient score
diversity score
q-window risk
Delta-refinement risk
extrapolation risk
```

The default weights are fixed hyperparameters:

```text
w_cls_uncertainty   = 1.0
w_reg_uncertainty   = 0.8
w_delta_boundary    = 1.0
w_q_boundary        = 1.0
w_eta_boundary      = 0.7
w_gradient          = 0.7
w_diversity         = 0.3
w_q_edge_risk       = 0.8
w_delta_refine_risk = 0.8
w_extrapolation     = 0.4
```

These weights can be manually changed between runs, but in the current
implementation they are not learned by gradient descent.

## Report Use

Use this explanation in the methods section to prevent confusion between
training objectives and active-learning point selection.

## Question

Is the active-learning point selection random?

## Short Answer

No. The selected points are chosen by ML-guided ranking. The trained surrogate
predicts observables, phase probabilities, and uncertainties on a finite
candidate grid. The acquisition function then scores all admissible candidates,
sorts them by score, applies exclusion and diversity rules, and keeps the top
batch of points.

## Technical Notes

The selection loop is:

```text
1. Start from the current exact dataset D_n.
2. Train the regression and classification MLP ensembles.
3. Predict on the candidate grid.
4. Compute acquisition score S(x) for each candidate.
5. Mask inadmissible or already-covered candidates.
6. Sort candidates by score.
7. Apply diversity radius within the selected batch.
8. Select POINTS_PER_ITER candidates.
9. Run exact BdG only on those selected candidates.
10. Append only trusted/training-eligible exact outputs to D_{n+1}.
```

There can be randomness in neural-network initialization and shuffled training
batches, and the ensemble uses different seeds. However, the point-selection
principle itself is not random sampling. It is a deterministic ranking step
given the trained ensemble outputs and the configured acquisition weights.

For earlier runs, `POINTS_PER_ITER=128`, so the loop selected 128 high-scoring
points per iteration. In the later production configuration,
`POINTS_PER_ITER=512`, so the same logic selects 512 points per iteration.

## Report Use

Use this explanation when describing the active-learning loop as a scheduler
rather than as random exploration.

## Question

What exactly are the candidate points? Does the ML model predict every point
in the continuous \((k_B T/t,J_A/t)\) plane?

## Short Answer

No. The candidate points are a finite dense grid generated by the code. The ML
model predicts this finite candidate grid cheaply; it does not evaluate a
continuous plane.

## Technical Notes

The current candidate grid is defined in `ml_phase/config.py` as:

```text
kT range: 0.0 to 0.56
JA range: 0.0 to 2.12
n_kt_candidates = 241
n_ja_candidates = 321
```

Thus the full finite grid contains:

```text
241 x 321 = 77361 candidate coordinates
```

Each candidate is a coordinate pair:

```text
(kT_i, JA_j)
```

The MLP ensembles evaluate all these grid coordinates with cheap forward
passes. This does not make the calculation expensive because the model is
small:

```text
regression branch:     2 -> 64 -> 64 -> 5
classification branch: 2 -> 64 -> 64 -> 3
ensemble size: 5
```

The expensive operation is not ML prediction on 77361 points. The expensive
operation is exact BdG minimization. Therefore, after scoring the candidate
grid, only a small selected batch such as 128 or 512 points is sent to the
exact oracle.

Not every point in the rectangular candidate grid is admissible. The code
constructs a physics-aware mask:

```text
candidate_mask:
    kT >= 0
    JA >= JA_min
    JA <= boundary_ja(kT) + finite_t_band_width
```

Here `boundary_ja(kT)` is interpolated from a reference finite-temperature
boundary array, and

```text
finite_t_band_width = 0.08
```

Candidates outside this mask receive score `-inf` and cannot be selected.
Additional exclusions remove exact duplicates, points too close to existing
exact data under the current radius rule, recent selections, and certain
boundary-band cooldown points.

## Report Use

A precise report sentence is:

```text
At each active-learning iteration, we construct a finite dense candidate grid
of 241 x 321 points in the (kBT/t, JA/t) plane. The trained MLP ensembles
evaluate all candidate-grid points cheaply. The acquisition function then
masks out inadmissible or already-covered candidates and selects only the
top-ranked 128 or 512 points for expensive exact BdG evaluation.
```

## Question

How does active learning acquire a direction toward the phase boundary if the
ML loss is not a boundary-position loss?

## Short Answer

The boundary-seeking behavior comes from the acquisition function, not from
the training loss itself. The ML loss trains predictive surrogates for exact
observables and labels. The acquisition function then uses the trained
predictions to favor candidate points near phase thresholds, high uncertainty,
large gradients, and numerical-risk regions.

## Technical Notes

The MLPs do not directly optimize a boundary-displacement loss. They learn:

```text
continuous observables:
    Delta_opt, q_opt, eta, Ic+, Ic-

categorical phase labels:
    normal, uniform_SC, FFLO
```

Boundary orientation enters after training through terms such as:

```text
Delta-boundary score:
    large when Delta_pred is near Delta_eps

q-boundary score:
    large when |q_pred| is near q_eps

classifier uncertainty:
    large when phase probabilities are comparable

gradient score:
    large where predicted Delta, q, or eta changes rapidly
```

Thus the model supplies a learned map, and the acquisition rule interprets
that map according to the scientific target of improving phase-boundary
coverage.

## Report Use

Use this explanation when distinguishing the surrogate-training objective from
the active-learning objective.

## Question

Why did the distance radius \(0.015\) become questionable, and where else does
it appear in the active-learning workflow?

## Short Answer

The radius \(0.015\) was useful as a coarse diversity/exclusion radius for the
dense candidate grid, but it is too crude for explicit boundary-bracket
midpoint refinement and has now been halved to \(0.0075\) for the next
512-point, 50-loop production upload. The radius appears in two connected
places: as the selected-point diversity radius and as the existing-exact-point
exclusion radius. Reusing a dense-grid radius for boundary midpoints can reject
valid refinement targets because a midpoint is naturally close to the exact
points that define the bracket.

## Technical Notes

The earlier active-learning configuration used

```text
diversity_min_dist = 0.015
existing_min_dist = 0.015
```

For the next production upload, this has been changed to

```text
diversity_min_dist = 0.0075
existing_min_dist = 0.0075
```

The distance is not a physical Euclidean length in raw units. It is a
normalized distance in the \((k_B T/t, J_A/t)\) candidate domain:

\[
d =
\sqrt{
    \left(\frac{kT_{\rm cand}-kT_{\rm exact}}{kT_{\max}-kT_{\min}}\right)^2
  + \left(\frac{J_{A,\rm cand}-J_{A,\rm exact}}{J_{A,\max}-J_{A,\min}}\right)^2
}.
\]

With the current candidate ranges,

```text
kT range: 0.00 to 0.56
JA range: 0.00 to 2.12
```

a normalized radius \(0.015\) corresponds roughly to

```text
Delta kT ~= 0.0084 if the displacement is purely horizontal
Delta JA ~= 0.0318 if the displacement is purely vertical
```

This was introduced after repeated exact-coordinate selections remained after
the 4-decimal coordinate exclusion. It is appropriate for dense-grid
acquisition because it prevents the model from spending a full batch on points
that are effectively redundant with existing exact calculations.

However, boundary-bracket refinement is different. A boundary midpoint is
defined between two already computed exact points that straddle a boundary.
Therefore the midpoint is supposed to be close to the existing points. Applying
\(d_{\rm existing}<0.015\) to such midpoints rejected all prioritized midpoint
targets in the boundary-extraction recheck.

The decision recorded as D15 is therefore:

```text
Do not blindly apply dense-grid existing-distance exclusion to
boundary-bracket midpoints.
```

The next production run therefore uses \(0.0075\) for ordinary dense-grid
acquisition, while explicit boundary-bracket refinement still requires a
separate bracket-specific rule for normal_sc and uniform_fflo midpoint targets.

## Report Use

Use this explanation in the active-learning caveats or boundary-refinement
strategy section, especially when explaining why the next targeted refinement
policy should not simply reuse the dense-grid acquisition exclusion rule.

## Question

Does the current active-learning code stop automatically if an iteration adds
no new exact points, or only very few new points?

## Short Answer

This was not present before, but it has now been implemented. The code records
both training-eligible appended points and the actual unique sample increase,
then stops the loop if unique data growth stalls.

## Technical Notes

The HPC active loop still has a requested iteration range:

```text
for iter in START_ITER ... START_ITER + N_ITERS - 1
```

but it can now break out early. After shard merge it calls
`ml_phase.append_trusted`, which writes an
`*.append.json` summary containing:

```text
training_eligible_points_appended
trusted_points_appended
input_samples
output_samples
new_unique_samples_added
```

`hpc_active_loop.sh` reads this JSON summary and applies:

```text
stop if selected_points.csv contains zero candidates
stop if new_unique_samples_added == 0
stop if new_unique_samples_added < MIN_NEW_POINTS_PER_ITER
    for MAX_LOW_APPEND_ITERS consecutive iterations
```

Defaults:

```text
ENABLE_EARLY_STOP=1
MIN_NEW_POINTS_PER_ITER=8
MAX_LOW_APPEND_ITERS=2
```

The boundary recheck found that `dataset_iter039.npz`,
`dataset_iter040.npz`, `dataset_iter041.npz`, and `dataset_iter042.npz` have
the same array-content hash. The new early-stop rule is meant to prevent this
kind of empty or ineffective loop progression from continuing unnoticed.

The local Python active-refinement loop also has matching arguments:

```text
--disable-early-stop
--min-new-points-per-iter
--max-low-append-iters
```

and writes `local_append_summary.json` during local exact runs.

## Report Use

Use this as a reproducibility and workflow-safety note before the next
production active-learning run.

## Question

If the default priority quotas are manually written into the code, where does
ML plus the acquisition function still affect selected points? Why not rely on
acquisition alone instead of adding boundary midpoint refinement?

## Short Answer

The manual quotas decide how the exact-call budget is divided among scientific
tasks. ML acquisition still ranks the dense-grid high-JA and global exploration
points, and it also helped create the cumulative exact dataset from which later
boundary brackets are extracted. Boundary-local midpoint refinement is added
because the current scientific objective is no longer only to discover
interesting regions; it is to shrink already confirmed exact phase-boundary
brackets.

Acquisition alone can identify uncertain or boundary-like regions, but it does
not guarantee bisection-like reduction of a known exact bracket. Midpoint
refinement is a deterministic local-resolution step applied only after exact
data have already bracketed a phase change.

## Technical Notes

Current hybrid selection separates the exact-call budget:

```text
normal_sc midpoint quota
uniform_fflo midpoint quota
high-JA q-risk dense-grid acquisition quota
ordinary global dense-grid acquisition quota
```

The first two quotas are boundary-local. They use exact labels from the current
accepted dataset. The selected point is usually the midpoint between two exact
points that lie on opposite sides of a target phase boundary. This reduces the
geometric uncertainty of that bracket.

The last two quotas are ML/acquisition driven. The MLP ensembles predict
observables and uncertainties on the dense candidate grid, and the acquisition
function ranks those candidates. This part remains essential for:

```text
1. finding boundary regions not already bracketed by exact points;
2. probing high-JA q-window-risk regions beyond the originally trusted warm
   start domain;
3. detecting classifier or regression uncertainty ridges;
4. preventing the workflow from becoming a purely local bisection procedure.
```

Using acquisition alone would be reasonable for broad exploration. It becomes
less ideal once an exact bracket is already known, because the acquisition
score is a heuristic mixture of uncertainty, boundary proximity, gradients,
diversity, q risk, Delta risk, and extrapolation. Its top-scoring point need
not be the point that most efficiently halves a specific exact bracket.

The midpoint rule also has a limitation: if the current bracket is produced by
very sparse or geometrically awkward data, the midpoint may not be the best
physical direction for refinement. In that case a later version should use
boundary-normal or local-curve-aware targets instead of raw midpoint targets.
The current midpoint implementation is therefore a conservative first local
refinement step, not the final mathematical optimum for all boundary shapes.

## Report Use

Use this explanation when distinguishing "budget allocation", "ML acquisition
ranking", and "exact bracket refinement" in the active-learning method section.
