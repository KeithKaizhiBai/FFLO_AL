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

do not spend repeated active-learning iterations on a boundary-band point
unless a stricter physics question requires resolving below the chosen energy
tolerance.
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
