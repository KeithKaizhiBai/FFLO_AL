# Decisions

Last updated: 2026-05-20

## D1. Use Existing Exact Data as Warm Start

Decision:

```text
Use the existing 21528-point exact BdG dataset as the initial supervised
training set.
```

Reason:

```text
Starting active learning from no data would spend expensive GPU time
rediscovering the coarse phase structure. The existing grid already gives a
usable first surrogate and uncertainty map.
```

## D2. Treat ML as Scheduler, Not Solver

Decision:

```text
The ML model proposes exact BdG points. It does not replace exact labels.
```

Reason:

```text
This preserves physics correctness and makes active learning acceptable for
phase-boundary refinement.
```

## D3. Separate CPU-Side Acquisition from H100 Exact Evaluation

Decision:

```text
Train models and select points in a CPU-side job. Run only exact BdG point
evaluation on H100 array jobs.
```

Reason:

```text
Avoid wasting allocated H100 time on CPU-heavy model training, plotting, or
report generation.
```

## D4. Use GBU Partition NV_H100

Decision:

```text
Use partition NV_H100 for H100 GPU jobs.
```

Reason:

```text
The cluster module/sinfo output showed NV_H100 as the available H100 partition.
Earlier Intel partition usage failed on the remote workflow.
```

## D5. Do Not Rely on Conda Activation in SLURM

Decision:

```text
Prefer PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python in
batch scripts.
```

Reason:

```text
The cluster reported "CondaError: Run 'conda init' before 'conda activate'".
Calling the environment Python directly is more robust.
```

## D6. q-Window Must Be Adaptive

Decision:

```text
Future exact oracle outputs must detect q_edge_hit and support q-window
expansion or rerun lists.
```

Reason:

```text
For large J_A, q_opt can grow rapidly. A fixed q range can truncate the true
minimum and create false phase labels.
```

## D7. Delta Must Be Refined Near the Normal/SC Boundary

Decision:

```text
Future exact oracle outputs must refine Delta near Delta_opt ~= DELTA_EPS or
when the free-energy minimum is shallow.
```

Reason:

```text
The normal/SC boundary is sensitive to coarse Delta sampling. Boundary labels
must be based on refined Delta values.
```

## D8. Normal-State q Is Not a Physical q-Edge Failure

Decision:

```text
When Delta_opt < DELTA_EPS and the Delta status is stable, the active-learning
exact oracle sets q_status=not_applicable and q_edge_hit=false.
```

Reason:

```text
In the normal state the free energy is independent of the FFLO order parameter
in the same sense as the superconducting state, so a reported q_opt at q_min is
usually a grid tie artifact rather than a physical q-window truncation.
```

## D9. Merge Only Trusted Exact Points Into the Training Pool

Decision:

```text
HPC shard merge writes both exact_merged_iterXXX.npz and
exact_trusted_iterXXX.npz. Future append logic should use the trusted file by
default.
```

Reason:

```text
Unresolved q-edge and Delta-boundary points should drive reruns or further
refinement, not silently train the surrogate as final exact labels.
```

## D10. Delta=0 Normal Points Require a Positive-Delta Gap Check

Decision:

```text
When Delta_opt = 0, the oracle does not use omega_global = 0 by itself as a
Delta-boundary ambiguity signal. It computes positive_delta_gap from a
strictly positive-Delta scan and marks the point ambiguous only if this gap is
near zero.
```

Reason:

```text
For a true normal-state optimum, the free-energy gain relative to the
Delta=0 baseline is exactly zero by construction. Treating this identity as a
boundary signal misclassifies stable normal points as unresolved.
```

## D11. Exclude gpuh01 from Automatic H100 Active-Learning Jobs

Decision:

```text
Set EXCLUDE_NODES=gpuh01 by default in the automatic H100 submission helpers.
```

Reason:

```text
The completed cluster test showed gpuh01 failing with a PyTorch/CUDA driver
mismatch while other H100 nodes such as gpuh04 and gpuh11 completed the exact
oracle shards.
```

Consequence:

```text
The automatic loop submits both candidate-generation jobs and H100 exact-oracle
arrays with sbatch --exclude=gpuh01 unless the user explicitly overrides
EXCLUDE_NODES. This improves reliability but may reduce available scheduling
capacity by one node.
```

## D12. Treat Sub-Tolerance Delta Boundary Points as Boundary-Band Normal

Decision:

```text
If Delta_opt = 0 and the best strictly positive-Delta state is not lower than
the normal state by more than the adopted energy tolerance, interpret the point
as normal-side or normal/SC boundary-band data rather than as a failed exact
calculation.
```

Reason:

```text
The 25-iteration 128-point active-learning run found 147 Delta-unresolved
points, all with delta_boundary_unresolved;max_delta_refinement_reached. Most
had Delta_opt = 0 and positive_delta_gap in the 1e-9 to 1e-8 range. This is
below or comparable to the chosen physical/numerical resolution for the present
phase diagram, and the positive-Delta state does not show a resolvable
condensation-energy gain.
```

Consequence:

```text
These points should be carried with explicit boundary-band metadata and used as
normal-side constraints for phase-boundary construction. Future active-learning
iterations should avoid repeatedly spending the same budget on these
sub-tolerance points unless the study goal changes to resolving a much finer
normal/SC critical line.
```

## D13. Hard-Exclude Existing Exact Coordinates During Acquisition

Decision:

```text
During candidate selection, hard-exclude dense-grid candidates whose rounded
(kT, JA) coordinate already appears in the current exact training dataset.
Use 4 decimal places by default.
```

Reason:

```text
The 10-iteration continuation after the boundary-band semantic fix showed that
most exact records became training-eligible, but many selected candidates were
already present in the dataset. Repeating exact calls at existing coordinates
does not improve phase-boundary resolution.
```

Consequence:

```text
Existing exact-data exclusion is not relaxed. Recent selected points and
previous boundary-band points use the same 4-decimal cooldown, but those two
cooldowns may be relaxed if fewer than points_per_iter finite candidates remain.
This keeps the active-learning loop productive while avoiding an overly small
rounding threshold such as 8 decimals.
```

## D14. Use the Diversity Radius as the Existing-Data Exclusion Radius

Decision:

```text
In addition to 4-decimal rounded-coordinate exclusion, hard-exclude any
candidate whose normalized distance to the existing exact dataset is smaller
than existing_min_dist = 0.015. This matches the selected-selected diversity
radius.
```

Reason:

```text
The 3-iteration validation after D13 reduced repeated existing coordinates
from 82-97 per 128-point iteration to 1-4, but a few high-JA existing points
still leaked through the rounded-key rule. The selected-selected diversity rule
was already enforcing the same 0.015 normalized radius within each new batch,
so applying that radius to the existing dataset makes the "one point per
radius" rule consistent.
```

Consequence:

```text
The existing-data exclusion is now stronger and will remove much of the
already dense warm-start region from the candidate pool. A check against
dataset_iter038 still leaves more than 1000 finite candidates, so a short
128-point continuation remains feasible. The next validation must confirm that
selected points still follow physically meaningful phase-boundary regions.

This 0.015 radius was later superseded for ordinary dense-grid acquisition by
D17, which halves the radius to 0.0075 for the next 512x50 production upload.
```

## D15. Do Not Blindly Apply Existing-Min-Distance to Boundary-Bracket Midpoints

Decision:

```text
Dense-grid existing-distance rules are valid for ordinary active-learning
candidate selection, but they should not be blindly used to reject explicit
boundary-bracket midpoint refinement targets.
```

Reason:

```text
The boundary extraction recheck on 2026-05-12 found that all prioritized
midpoint targets from the extracted boundary brackets were within the then-used
0.015 existing_min_dist radius of an already computed exact point. This is
expected: a bracket midpoint is geometrically close to the two exact points
that define the bracket. Applying a dense-grid acquisition radius to these
midpoint targets can reject valid normal_sc, uniform_fflo, strong_diode, and
eta_zero candidates.
```

Consequence:

```text
Future boundary refinement must use a boundary-specific target policy. The
policy should quota-limit eta_zero targets, prioritize normal_sc and
uniform_fflo brackets, and either relax the existing-distance rule for bracket
midpoints or replace it with a smaller boundary-specific exclusion radius.
The archived recheck writes both strict radius-checked outputs and
basic-filtered candidate outputs so this decision remains auditable.
```

## D16. Stop Active-Learning Loops When Unique Data Growth Stalls

Decision:

```text
The active-learning loop now stops early when an iteration selects no candidate
points, appends zero new unique training samples, or appends fewer than
MIN_NEW_POINTS_PER_ITER unique samples for MAX_LOW_APPEND_ITERS consecutive
iterations.
```

Reason:

```text
The boundary-extraction recheck showed that dataset_iter039 through
dataset_iter042 can be content-identical when later iterations are
empty-append or ineffective postprocessing aliases. Continuing the loop in
that state wastes exact-oracle budget and creates misleading dataset indices.
```

Consequence:

```text
hpc_active_loop.sh reads the append summary written by ml_phase.append_trusted
and stops by default at zero unique additions or after two consecutive
iterations with fewer than eight new unique samples. The thresholds are
configurable through ENABLE_EARLY_STOP, MIN_NEW_POINTS_PER_ITER, and
MAX_LOW_APPEND_ITERS. The local active_refine loop uses matching configuration
fields and writes local_append_summary.json for traceability.
```

## D17. Halve Dense-Grid Acquisition Radius for the 512x50 Production Upload

Decision:

```text
For the next warm-start production active-learning upload, use
diversity_min_dist = existing_min_dist = 0.0075 instead of 0.015 for ordinary
dense-grid acquisition.
```

Reason:

```text
The 0.015 normalized radius was useful for eliminating repeated or near-repeat
exact calls, but it is coarse relative to the boundary detail now being
targeted. Halving the radius keeps an explicit diversity and existing-data
exclusion rule while allowing denser sampling near normal/SC and
uniform_SC/FFLO boundaries.
```

Consequence:

```text
The production upload defaults to 512 selected points per loop for 50 loops
starting from the warm-start dataset. More candidates can survive the
existing-data exclusion than with the previous 0.015 radius, so the loop should
be less likely to stall from over-aggressive exclusion. D15 still applies:
explicit boundary-bracket midpoint refinement should use a boundary-specific
policy rather than blindly reusing the dense-grid radius.
```

## D18. Add Boundary-Local Hybrid Refinement Mode (Superseded by D19)

Decision:

```text
Add an optional boundary_refinement_mode with values off, hybrid, and local.
The default Python active_refine behavior remains off for backward
compatibility. The current HPC loop wrappers default to hybrid for the next
boundary-refinement production runs.
```

Reason:

```text
Fixed-radius global acquisition can saturate after it has covered the broad
candidate pool, even though exact phase boundaries still need local refinement.
Boundary-bracket midpoints are valid refinement targets precisely because they
lie close to existing exact bracket endpoints, so they require a separate
boundary-local radius instead of the ordinary dense-grid existing-distance
radius.
```

Consequence:

```text
Hybrid mode extracts normal_sc and uniform_fflo boundary brackets from the
current accepted exact dataset, selects midpoint targets with a smaller
boundary_local_min_dist=0.00375, and fills the remaining point budget with
high-JA q-risk and ordinary global acquisition points. Eta-zero and
strong-diode brackets are still extracted for diagnostics, but their default
selection quotas are zero because they are response boundaries rather than the
primary thermodynamic phase boundaries.

This decision is superseded by D19 for the next active-learning stage.
```

## D19. Disable Midpoint-Based Selection for the Next Active-Learning Stage

Decision:

```text
The production active-learning selector no longer uses geometric boundary
midpoints as selected exact-call candidates. Boundary extraction remains
available in diagnostic mode, but selected_points.csv is produced only by
ML-guided dense-grid acquisition.
```

Reason:

```text
Midpoint selection is a deterministic bisection rule on already known exact
brackets. It can narrow a bracket, but it does not let the trained surrogate
decide where a new exact calculation would most reduce uncertainty, boundary
ambiguity, or numerical risk. A large fixed midpoint quota therefore conflicts
with the active-learning objective.
```

Consequence:

```text
boundary_refinement_mode=diagnostic extracts and reports exact boundary
segments without generating midpoint targets. The legacy hybrid/local modes
now raise an error. Candidate selection uses the acquisition score on the
dense grid, a hard exact-coordinate duplicate mask, soft observation repulsion
near existing exact data, and soft batch repulsion among newly selected points.
```

## D20. Use StopController Metrics Instead of Candidate Exhaustion

Decision:

```text
Add an independent StopController for the acquisition-only active-learning
loop. The controller evaluates convergence after each exact-oracle merge and
trusted append, using phase-map stability, main thermodynamic boundary shifts,
label surprise, selected acquisition score saturation, q-edge/rerun rates, and
main-boundary exact-data coverage.
```

Reason:

```text
With soft observation and batch repulsion, the selector can usually continue
assigning nonzero scores to uncomputed dense-grid candidates. Therefore the
existence of more candidates is not a useful convergence criterion. The stop
decision must instead ask whether new exact BdG calls are still changing the
main phase-boundary information or exposing numerical-risk failures.
```

Consequence:

```text
No-available-candidates remains an exceptional stop reason with
stop_reason=no_available_candidates. It is not the main convergence logic.
Eta-zero response boundaries and topology are excluded from the current main
stop rule.

The original 2026-05-14 implementation used q-edge/rerun and boundary coverage
as mandatory gates in a seven-condition rule. This was superseded by D21:
selected A0 ratio and q-edge/rerun rates are now diagnostics/cleanup warnings,
while the main stop rule uses five phase-map and thermodynamic-boundary
conditions.
```

## D21. Add Discovery-Mode Active Learning

Decision:

```text
Add run_mode=discovery as a first-class active-learning workflow distinct from
run_mode=refinement.
```

Reason:

```text
Warm-start refinement measures how efficiently ML improves a known exact phase
diagram. It does not test whether active learning can discover the phase
structure from sparse exact information. Discovery mode is needed for that
benchmark.
```

Consequences:

```text
Discovery mode starts from random exact seed points on the full rectangular
candidate grid, does not load the large warm-start dataset as the initial
training set, and forbids the finite-T prior candidate-band mask.

Refinement mode remains available for warm-start boundary refinement and may
explicitly use candidate_domain_mode=prior_band with finite_t_band_width.

Discovery-mode acquisition uses stochastic score-weighted sampling with dynamic
batch repulsion instead of fixed regional quotas. The main stop controller now
uses the five main phase-map and thermodynamic-boundary conditions; selected
A0 ratio, q-edge rate, and rerun-required rate are diagnostics and cleanup
warnings, not mandatory gates for stopping the main phase-boundary loop.
```

## D22. Gate Discovery Sampling by A0 Main Before Stochastic Draws

Decision:

```text
Discovery-mode active-learning selection now separates physical/model
information value from stochastic sampling weights.  A0_main defines a
high-information active pool on the full candidate grid.  The actual stochastic
draws are then made inside that pool with weights proportional to
(A0_main * R_obs * R_batch)^gamma.
```

Reason:

```text
The first full discovery run converged in the StopController sense, but late
selected points still looked close to random full-domain coverage.  The old
sampler applied stochastic sampling over the broad corrected score field and
used a strong historical R_obs, so selection behaved partly like a coverage
sampler rather than a phase-boundary information sampler.
```

Consequences:

```text
The high-information pool is selected by A0_main quantile and relative-to-p95
thresholds.  R_obs is now a mild historical downweight with default floor 0.5
and length 0.02; it no longer decides whether a point is worth considering.
R_batch remains stronger and only discourages within-batch clustering.

Discovery batches are adaptive: before the minimum-iteration stage the active
pool threshold can relax to maintain a useful exploratory batch, while later
iterations no longer force the batch to fill batch_size_max when only fewer
high-information candidates remain.

Selection diagnostics now track active-pool size, threshold relaxation,
effective sample size, selected A0 concentration, R_obs statistics, and
selected-to-predicted-boundary distances against a random active-pool baseline.
```

## D23. Sharpen Discovery Active Pool with Gated Delta-Boundary Scores

Decision:

```text
Discovery-mode acquisition now uses a normal/SC competition gate for the
Delta-boundary score, a stricter max-threshold active-pool rule, a soft
high-confidence interior penalty, piecewise sampling-power annealing, and
piecewise exploration-weight annealing.
```

Reason:

```text
The 2026-05-20 selection-region diagnostics showed that the late discovery
batch was dominated by predicted normal-interior points.  Component attribution
identified the broad Delta-boundary score as the main source: deep predicted
normal points can have Delta_pred close to zero, making
exp(-abs(Delta_pred - Delta_eps) / delta_scale) large even when the classifier
is highly confident the point is not near the normal/SC boundary.
```

Consequences:

```text
B_delta_raw is still saved for diagnostics, but A_phase uses
B_delta_gated = B_delta_raw * U_NS with U_NS = 4 * P_normal * P_SC.

The high-information pool is built from A0_for_pool rather than raw A0_main.
A0_for_pool includes the gated acquisition components and a soft penalty for
high-confidence, low-information phase interiors.  The active-pool threshold is
max(quantile threshold, relative-p95 threshold) rather than the previous loose
OR rule, and a scheduled active-pool fraction cap can tighten the threshold.

R_obs remains mild and does not decide active-pool membership.  Stochastic
selection still operates inside the score-defined pool and does not introduce
JA/kT region quotas or midpoint targets.

Future discovery reports should inspect B_delta_raw, U_NS, B_delta_gated,
A0_for_pool, active_pool_fraction, selected boundary-band fraction, and
N_eff/active_pool_size to verify that late batches become more boundary
focused.
```
