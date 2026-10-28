# Numerics Specification

Last updated: 2026-05-20

## Exact Solver

The exact solver minimizes BdG free energy over \((\Delta, q)\) at each
\((k_B T/t, J_A/t)\) point, then computes current response, critical currents,
and diode efficiency.

Current warm-start resolution is encoded in the dataset name:

```text
nkt138
nja156
nd400
nq400
nk800
fp64
libcusolver
```

The current data are useful as a warm start but should not be treated as the
final answer in regimes where the q window or Delta resolution is insufficient.

## Required Numerical Safeguards

q-window safeguards:

```text
record q_min, q_max, n_q, q_opt
flag q_edge_hit if q_opt is near q_min or q_max
rerun or exclude q_edge_hit points before trusted training
expand q window for high-J_A candidates, especially JA/t > 1.2
```

Current implementation status:

```text
q metadata and q_edge_hit flags are recorded by the active-learning exact
oracle. The pointwise oracle now supports automatic q-window expansion for
superconducting points only. Stable normal points use q_status=not_applicable
and are not treated as physical q-edge failures.
```

Delta safeguards:

```text
record Delta resolution and refinement level
refine Delta near Delta_opt ~= DELTA_EPS
compare free energy at Delta = 0 and Delta_opt near normal/SC boundary
for Delta_opt = 0, compare the best strictly positive-Delta state against
Delta = 0 before declaring the point ambiguous
do not silently append ambiguous labels
interpret sub-tolerance positive-Delta gaps as a finite-resolution boundary
band, not as a solver failure
```

Current implementation status:

```text
Delta ambiguity metadata and refinement-risk acquisition scores are present.
The pointwise oracle now supports local Delta rescans near the normal/SC
boundary and marks unresolved ambiguous points as untrusted. Stable normal
points are no longer rejected only because their Delta=0 free-energy gain is
identically zero; they are checked with positive_delta_gap.
```

Interpretation after the 25-iteration 128-point active-learning run:

```text
Most Delta-unresolved points have Delta_opt = 0 and positive_delta_gap in the
1e-9 to 1e-8 range. The positive-Delta state is not lower than the normal
state by a resolvable amount. These points should be treated as normal-side or
normal/SC boundary-band points at the adopted numerical precision, with
metadata preserved.
```

Current practical rule:

```text
robust SC:
    free_energy_gap_to_normal < -free_energy_ambiguity_tol

normal-side boundary band:
    Delta_opt = 0
    positive_delta_gap >= 0
    positive_delta_gap <= positive_delta_gap_tol

stable normal:
    Delta_opt = 0
    positive_delta_gap > positive_delta_gap_tol
    q_status = not_applicable
    q_unresolved = false
    training_eligible_exact = true

do not spend repeated active-learning iterations on a boundary-band point
unless a stricter physics question requires resolving below the chosen energy
tolerance.
```

Production robust-oracle label-closure rule:

```text
phase_label, confidence_state, training_eligible_exact, and rerun_required must
be separate metadata fields.  In particular, normal_q_not_applicable is a valid
normal-state q metadata value and must not imply q_unresolved.

For Delta_opt = 0:
    positive_delta_gap > positive_delta_gap_tol
        -> stable normal, confidence_state=trusted, training_eligible_exact=true
    0 <= positive_delta_gap <= positive_delta_gap_tol
        -> normal-side boundary band, confidence_state=boundary_ambiguous,
           training_eligible_exact=true unless a stricter cleanup run is needed

Do not use free_energy_ambiguity_tol as the stable-normal rejection threshold;
that tolerance is too broad for deciding whether a Delta=0 point can train the
normal phase.
```

Active-learning safeguards:

```text
support discovery mode from random exact seed points on the full rectangular
candidate grid
support refinement mode from a warm-start or resume dataset
for discovery mode, do not apply finite-T prior candidate-band pruning
for refinement mode, prior-band pruning is allowed only when explicitly
configured
hold out validation data
track RMSE for Delta, q, eta, I_c^+, I_c^-
track phase accuracy and boundary F1
track selected acquisition score history
deduplicate exact points before appending
write iteration metadata for reproducibility
track repeated selected/rerun coordinates so unresolved boundary-band points do
not consume many iterations without changing the inferred boundary
hard-exclude existing exact coordinates during candidate selection using
4-decimal rounded (kT, JA) keys
hard-exclude exact duplicate coordinates using the 4-decimal rounded key rule
use a soft observation-repulsion factor near existing exact data instead of a
hard distance cutoff
use a soft batch-repulsion factor during greedy batch selection instead of
boundary-bracket midpoint targets
in discovery mode, sample acquisition candidates stochastically with
probability proportional to the corrected acquisition score rather than using
fixed regional quotas
in discovery mode, gate the stochastic candidate pool by A0_main before
sampling; R_obs and R_batch only modify sampling weights inside that
high-information pool
in discovery mode, use normal/SC competition gating for the Delta-boundary
score:
    B_delta_gated = B_delta_raw * 4 * P_normal * P_SC
use A0_for_pool, not raw A0_main, to build the high-information active pool
after high-confidence phase-interior soft penalties
construct the active pool with a max-threshold rule:
    threshold = max(quantile(A0_for_pool), active_pool_rel_to_p95 * p95(A0_for_pool))
anneal active-pool quantile, sampling power, and exploration weight so late
iterations are more boundary-focused than early exploration
cap the fraction of hard-unseen candidates admitted to the active pool by
raising the quantile threshold when necessary
use mild historical observation repulsion by default:
    observation_repulsion_floor = 0.5
    observation_repulsion_length = 0.02
do not force late discovery iterations to fill batch_size_max when the
high-information active pool is smaller than the requested batch
record active-pool size, threshold relaxation, effective sample size, selected
A0 concentration, and selected-to-predicted-boundary distance diagnostics
record B_delta_raw, U_NS, B_delta_gated, U_UF, B_q_raw, B_q_gated,
A0_for_pool, interior_penalty_applied, current sampling power, and current
exploration weight for report diagnostics

boundary diagnostics:
    extract normal/SC and uniform/FFLO exact boundary segments from the current
    accepted dataset
    record boundary counts and displacement diagnostics
    do not turn extracted brackets into midpoint exact-call candidates

stop-controller diagnostics:
    write monitor_predictions_iterXXX.npz on a fixed dense grid
    record predicted_phase_before_exact for every selected point
    after exact merge and trusted append, compute stop metrics from exact
    labels, q-window flags, rerun flags, selected A0, phase-map stability, and
    main-boundary coverage
    exclude eta-zero response boundaries and topology from the current main
    stop rule
    treat selected A0 ratio, q-edge rate, and rerun-required rate as
    diagnostics/cleanup warnings rather than mandatory main-loop stop gates
    do not treat available soft-repulsion candidates as evidence that the loop
    should continue
```

## HPC Safeguards

The H100 exact-oracle jobs should be restartable:

```text
write one shard output per rank
include rank/world_size/hostname/CUDA_VISIBLE_DEVICES metadata
merge only when all expected shards are present
write rerun_points.csv for failed or ambiguous points
write exact_trusted_iterXXX.npz containing only trusted exact points
```

Use `PYTHON_BIN` directly in SLURM scripts when conda activation is unreliable.

## Phase q-window and Delta-refinement Audit Workflow

For high-\(J_A\), low-\(T\) normal/SC boundary-sensitive points, the project now
uses an audit-only phase robustness workflow:

```text
scripts/phase_qwindow_delta_refinement_audit.py
report_phase_qwindow_delta_refinement_v1/
```

This workflow does not redefine the phase criterion, does not modify
active-learning acquisition, and does not append audit-only rerun outputs to
the training dataset.

Phase q-window expansion records:

```text
old q_opt, Delta_opt, DeltaF_min, phase label
expanded q_opt, Delta_opt, DeltaF_min, phase label
phase_q_window_valid
phase_q_window_expanded_checked
q_opt_edge_margin
expanded_window_found_lower_branch
phase_changed_by_q_expansion
uniform_fflo_changed_by_q_expansion
F_min(q) = min_Delta F(Delta, q)
Delta_star(q)
low_energy_local_minima_count
```

The purpose of q-window expansion is not to prove superconductivity after a
lower-free-energy positive-\(\Delta\) state has already been found.  It checks
branch identity, \(q_{\rm opt}\) stability, boundary robustness, and whether
low-energy FFLO local minima are saved for later topology calculations.

Near-zero Delta refinement records:

```text
old_phase
refined_phase
old_Delta_opt
refined_Delta_opt
old_DeltaF
refined_DeltaF
delta_refinement_triggered
delta_refinement_valid
boundary_ambiguous
changed_after_delta_refinement
```

The report must also carry the response caveat:

```text
eta_response_valid is false unless a separate response-level q-window,
q-density, and response-curve pathology audit has validated the point.
```

## Acquisition Profile A/B Support

The active-learning acquisition now supports two explicit profiles through
configuration:

```text
acquisition_profile = full | simple_phase
```

`full` profile:

```text
Use the existing production formula without changing the main branch logic:
A0_main = A_phase + A_numerical + A_explore
```

`simple_phase` profile:

```text
Keep phase-boundary discovery components as the main score and remove
numerical-risk / response terms from A0_main.

U_cls = w_cls_entropy_inner * cls_entropy + w_cls_margin_inner * cls_margin_uncertainty
B_NS = B_delta_raw * U_NS
B_UF = B_q_raw * U_UF
G_phase = 0.5 * grad_delta + 0.5 * grad_q
U_reg_phase = 0.5 * U_delta + 0.5 * U_q

A_phase_simple =
    w_cls_simple  * U_cls
  + w_ns_simple   * B_NS
  + w_uf_simple   * B_UF
  + w_grad_simple * G_phase
  + w_reg_simple  * U_reg_phase

A0_main = A_phase_simple + A_explore_simple
A0_for_pool = A0_main * interior_penalty
Aselect = A0_for_pool * R_obs * R_batch
```

Notes:

```text
Diagnostic fields such as q_edge_risk_score, eta_zero_score, A_response, and
gradient_response are still exported for reporting, but they are excluded from
simple_phase A0_main.
```

## Production Robust Oracle (Discovery Rerun)

The production exact oracle now supports a robust adaptive-box mode for
active-learning exact calls:

```text
oracle_mode = robust_al
search_mode = adaptive_box
```

Core workflow per exact point:

```text
1) initial coarse scan on (Delta, q)
2) q-window coverage diagnostics from F_min(q) and Delta*(q)
3) directional q expansion (left/right/both) with dq preserved or improved
4) low-energy local-minima extraction from F_min(q)
5) local q-Delta box refinement around global/near-degenerate/edge-risk minima
6) near-zero Delta guardrail refinement
7) final minimum selection and unchanged thermodynamic phase criterion
```

Required metadata fields written per point include:

```text
oracle_mode, search_mode
initial_q_min/max, final_q_min/max
initial_n_q/final_n_q, initial_dq/final_dq
q_expansion_count, q_expansion_directions, q_expansion_trigger
q_window_coverage_valid, q_window_unresolved
qopt_edge_hit_initial/final
edge_risk_left/right_initial/final
expanded_window_found_lower_branch
phase_changed_after_q_expansion
local_minima_count, refined_local_minima_count
near_degenerate_branch_count
selected_minimum_rank
branch_candidates_path
delta_refinement_triggered, delta_refinement_valid
boundary_ambiguous, changed_after_delta_refinement
unresolved_reason, exact_status_code, exact_status_name
```

Branch-candidate outputs are audit/topology-preparation artifacts and do not
change the thermodynamic classification rule by themselves.

## Stage V Boundary-Support Acquisition-v2

Stage V introduces a cold-start 3D acquisition strategy for the
\((k_B T,J_A,\mu)\) domain.  It changes only active-learning point selection;
it does not change the exact oracle, thermodynamic phase labels, topology
definitions, or numerical tolerances used by the BdG calculations.

The default production identity is:

```text
run_id = stagev_acqv2_boundary_support_learned_residual_3d_v1
output_root = ML_Phase_StageV_AcqV2
initial_seed_size = 1024
micro_batch_size = 64
max_micro_batches = 96
```

The base acquisition score \(A_0\) combines five boundary channels:

```text
normal_SC
uniform_FFLO
P0_topology
Ppi_topology
gap_nodal
```

For each channel \(s\), the score uses a margin-likelihood factor \(B_s\), a
relative uncertainty factor \(U_s\), and a local sparse-support factor \(H_s\):

```text
B_s = P(|m_s| < tau_s)
U_s = sigma_s / (|mu_s| + sigma_s + eps)
H_s = sparse-support factor from distance to local trusted brackets
```

The channel terms are combined by log-sum-exp:

```text
A0 = logsumexp_s(log(B_s U_s H_s) + alpha_s) * exact_repulsion
```

Boundary support sets are built from local mutual-kNN or local-scale filtered
opposite-label brackets.  Raw global Delaunay long edges are not used as
boundary support.

Stage V also logs a learned residual value model:

```text
A(x) = A0(x) * exp(lambda_t * g_theta(phi(x)))
```

The residual model remains shadow-only while the reward denominator is too
small.  `lambda_t` is zero until the logged reward model improves rank
correlation relative to \(A_0\); if validation worsens, `lambda_t` is reduced.
The current implementation uses an explicit linear reward model over logged
candidate features rather than a hidden black-box optimizer.

## Stage V-v2 Multi-head Boundary-learning Acquisition

Stage V-v2 is a new cold-start 3D active-learning package intended to test
whether Stage V-v1's scalar learned residual can be replaced by per-boundary
learning.  It changes only acquisition and HPC orchestration.  It does not
change the exact oracle, thermodynamic phase rule, Hamiltonian, Pfaffian
formula, q/Delta search, topology classification convention, or numerical
tolerances.

Default production identity:

```text
run_id = stagev_v2_multihead_boundary_learning_3d_v1
output_root = ML_Phase_StageV_V2_Multihead
initial_seed_size = 1024
micro_batch_size = 64
max_micro_batches = 96
```

The run remains strict cold-start:

```text
stageiv_data_used_for_training = false
stagev1_data_used_for_training = false
```

The five boundary families are:

```text
ns   = normal/SC
uf   = uniform-SC/FFLO
p0   = P0 topology zero surface
ppi  = Ppi topology zero surface
gap  = gapped/nodal diagnostic surface
```

For each boundary \(s\), Stage V-v2 keeps the transparent base score:

```text
A0_s = B_s * U_s * H_s
```

and then applies a boundary-specific learned residual:

```text
A_s = A0_s * exp(lambda_s * g_s(phi_s))
```

The learned residuals are independent:

```text
lambda_ns, lambda_uf, lambda_p0, lambda_ppi, lambda_gap
```

A good normal/SC head cannot activate the topology lambdas.  If a boundary head
has insufficient reward support or underperforms its \(A0_s\) baseline, its
lambda remains zero and the channel falls back to \(A0_s\).

Before combining channels, Stage V-v2 rank-normalizes each boundary score
within the current candidate pool.  The final score is a log-sum-exp over
rank-normalized boundary contributions with automatic alpha priorities:

```text
A_total = logsumexp_s(alpha_s + ranknorm(log A_s)) + source_correction
```

The alpha priorities are updated from boundary-support deficits rather than
manual quotas.  Missing-boundary states are represented explicitly as
`missing_boundary`, `insufficient_support`, `physically_absent`, or
`not_yet_discovered`; missing boundary is never encoded as zero shift.

Stage V-v2 also applies proposal-source density correction so a large
`global_sobol` proposal pool cannot dominate selection solely by volume.  The
micro-batch selector remains stochastic and records selection propensity,
candidate source, dominant boundary, diversity factor, per-boundary raw score,
rank-normalized score, alpha, lambda, and final contribution.

The exact output schema is not modified in this package.  The existing field
`free_energy_gap_to_normal` is used as the stored \(F_{\rm SC}-F_{\rm normal}\)
margin.  Existing topology fields `topology_p0`, `topology_ppi`, and
`topology_bulk_gap` are used by the acquisition diagnostics.
