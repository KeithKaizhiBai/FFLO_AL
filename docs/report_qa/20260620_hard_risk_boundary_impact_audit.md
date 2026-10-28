# Hard-Risk Boundary-Impact Audit Note

Date: 2026-06-20

## Context

The rankcap_k3 tail continuation reached main-boundary convergence under the
trusted-surprise StopController gate, but `publication_ready` remained false
because the hard-risk numerical frontier had not yet been audited for boundary
impact.

Final tail-continuation status:

```text
main_phase_converged = true
trusted surprise = 0 / 127
hard-risk surprise = 75 / 129
rerun_required_count = 110
publication_ready = false
publication_ready_reason = hard_risk_boundary_impact_not_audited
```

## Audit Method

The audit used only existing local artifacts:

```text
dataset_iter035.npz
iter034/exact_merged_iter034.npz
iter034/rerun_points.csv
iter034/selected_points_by_pool.csv
iter034/monitor_predictions_iter034.npz
iter034/stop_metrics_iter034.json
```

It performed offline label-flip tests on copied dense-grid labels.  It did not
modify datasets, acquisition, exact oracle, StopController, phase criteria,
tolerances, or Slurm state.

Hard-risk points were defined as:

```text
rerun_required
or not trusted_exact
or not training_eligible_exact
or q_unresolved
or delta_unresolved
```

## Main Results

```text
hard-risk total = 129
boundary-near hard-risk points = 88
deep/far interior hard-risk points = 17
potentially boundary-moving points = 0
single-point exceeds boundary tolerance = 0 / 88
local hard-risk clusters = 37
local clusters exceeding boundary tolerance = 0 / 37
targeted_rerun_points.csv rows = 0
```

Boundary shifts:

```text
local single/cluster worst normal/SC p95 shift = 0
local single/cluster worst uniform/FFLO p95 shift = 0
strict global-stress worst normal/SC p95 shift = 0.002604
strict global-stress worst uniform/FFLO p95 shift = 0.663439
boundary-shift tolerance = 0.004167
```

The large uniform/FFLO stress value occurs only in the deliberately
synchronized all-boundary-near SC/FFLO stress test.  It is a nonlocal stress
test that can create artificial distant uniform/FFLO islands along the
normal/SC frontier.  It is not evidence that any single hard-risk point or
continuous local hard-risk segment moves the main boundaries.

## Decision

```text
Decision = A
need_new_exact_calculation = no
targeted rerun point count = 0
```

Interpretation:

```text
No full active-learning rerun and no targeted cleanup exact calculation is
required by this audit.  The main phase map can be treated as publication-ready
with an explicit hard-risk uncertainty band/marker layer.  Provisional
hard-risk labels should not be silently promoted to definitive phase labels.
```

## Report Outputs

```text
reports/hard_risk_boundary_impact_audit/hard_risk_boundary_impact_audit.md
reports/hard_risk_boundary_impact_audit/hard_risk_boundary_impact_audit.pdf
reports/hard_risk_boundary_impact_audit/decision_log.md
reports/hard_risk_boundary_impact_audit/tables/
reports/hard_risk_boundary_impact_audit/figures/
```

## Caveats

```text
1. The all-selected surprise metric remains high and should be reported as an
   acquisition-difficulty / hard-risk-frontier diagnostic, not as global phase
   instability.
2. Hard-risk provisional labels are not definitive physics labels.
3. The final phase-map figure should mark hard-risk uncertainty points or bands.
4. This audit supports moving forward without new exact calculation, but it
   does not eliminate the numerical-frontier caveat.
```
