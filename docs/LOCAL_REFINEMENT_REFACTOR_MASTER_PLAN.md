# Local Refinement Refactor Master Plan

Date: 2026-06-03

## Scope

This refactor targets only the performance path around local refinement in the
robust exact oracle. It must preserve the thermodynamic phase criterion,
Delta-refinement tolerances, stable-normal admission, acquisition formula,
StopController, topology policy, eta-response policy, and active-learning
control flow.

## Baseline

The current working baseline is the `robust_incremental` oracle mode:

- q-window expansion can use incremental strip scans.
- local refinement is still the same fixed-box refinement path.
- `max_refined_minima = 6`.
- basin clustering is off.
- energy-window pruning beyond the existing low-energy target selection is off.
- branch reuse is off.
- adaptive local boxes are off.
- GPU local-box batching and Hamiltonian caching are not implemented.

The existing `robust_al` full-rescan path remains a regression reference.

## Stage Plan

### Stage 0: Baseline Freeze

Goals:

1. Record the current baseline configuration and constraints.
2. Create a fixed-point regression point set.
3. Add regression scripts that compare later variants against the baseline.
4. Add report-template path guardrails so a missing report template does not
   hide completed numerical work.

No local-refinement optimization is enabled in Stage 0.

### Stage 1: Box-Level Instrumentation

Add logging-only local-box records:

- branch id and rank before refinement;
- selection reason;
- local box bounds and grid size;
- box runtime;
- refined minimum;
- whether the box changed the global minimum or phase label.

No pruning, clustering, reuse, or adaptive boxes are enabled in Stage 1.

### Stage 2: Basin Clustering

Cluster duplicate coarse minima into basin representatives. Mandatory-risk
basins must not be removed.

### Stage 3: Selective Refinement

Refine mandatory-risk basins plus a limited number of ordinary basins. This is
the first stage where the number of local refinement boxes can change.

### Stage 4: Energy-Window Pruning

Prune ordinary high-energy basins only after box-level instrumentation and
fixed-point regression show that they do not change the selected minimum.

### Stage 5: Branch Reuse

Prototype reuse of refined branch results across q-window expansion levels.
Reuse must be explicit and must record rejection reasons.

### Stage 6: Adaptive Local Box Skeleton

Record basin width and curvature proxies, then prototype configurable adaptive
local boxes. Fixed local boxes remain the baseline.

### Stage 7: GPU Batching and Hamiltonian Cache Planning

Plan tensor-shape logging, batching APIs, and cache signatures. This stage does
not change production physics until separate GPU regression passes.

## Invariants

Every stage must keep these invariants unless explicitly reauthorized:

- phase labels match the baseline or mismatches are audited point-by-point;
- `trusted_exact` matches the baseline;
- `training_eligible_exact` matches the baseline;
- no silent fallback;
- no silent reuse;
- no production active-learning restart without regression evidence.

