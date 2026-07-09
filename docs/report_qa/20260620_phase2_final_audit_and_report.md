# 2026-06-20 Phase-II Final Audit and Report

## Question

Can the rankcap_k3 Phase-II active-learning result be frozen as the main
thermodynamic phase-map result, and what caveats remain?

## Short Answer

Yes for the main phase map.  The final tail-continuation endpoint satisfies the
trusted-surprise StopController gate, and the hard-risk boundary-impact audit v2
passes the publication boundary gate:

```text
main_phase_converged = True
publication_boundary_audit = pass
audit_decision = Decision A
need_new_exact_calculation = False
targeted_rerun_count = 0
```

The numerical hard-risk frontier remains active and must be shown as an
uncertainty layer, not as definitive phase labels.

## Evidence

```text
final dataset = dataset_iter035
total samples = 7434
normal = 1867
uniform_SC = 715
FFLO = 4852
phase_map_change = 0.0016287 < 0.002
normal/SC boundary shift = 0.0041667 <= 0.0041667
uniform/FFLO boundary shift = 0
boundary_coverage_p95 = 0.0046875 < 0.00625
trusted surprise = 0/127
all-selected surprise = 75/256
hard-risk surprise = 75/129
```

Hard-risk audit v2:

```text
hard-risk total = 129
rerun_required = 110
non-rerun hard-risk = 19
boundary-near hard-risk = 88
local single/cluster p95 shift = 0
strict local Hausdorff diagnostic = 0.8125
significant local Hausdorff gate value = 0
meaningful topology change = False
targeted rerun count = 0
```

## Interpretation

The strict Hausdorff diagnostic can become large when an isolated hard-risk
counterfactual point creates a tiny disconnected boundary fragment on the dense
monitor grid.  The audit records this strict value, but the publication gate
requires a significant affected boundary arc or significant main-boundary
topology change before treating it as boundary-moving.  Under that gate, no
single-point or continuous-cluster hard-risk scenario moves the main boundaries
beyond tolerance.

## Outputs

```text
reports/hard_risk_boundary_impact_audit_v2/
report_phase2_robust_al_final_202606/
report_phase2_robust_al_final_202606/reproduction_manifest.json
```

## Recommended Next Stage

Freeze the Phase-II main phase-map result and move to branch-resolved topology
classification, hidden-ground-truth evaluation, multi-seed benchmarking, and
final publication figure polishing.  Do not launch another full active-learning
loop for the main phase-map result unless a later physics question explicitly
requires it.
