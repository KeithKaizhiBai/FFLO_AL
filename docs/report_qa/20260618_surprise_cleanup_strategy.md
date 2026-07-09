# 2026-06-18 Surprise Cleanup Strategy

## Question

Why did the rankcap_k3 full loop stabilize the main phase map but fail formal
StopController convergence, and what optimization direction should be tested
next?

## Answer

The main remaining contradiction is label surprise, not local-refinement cost
or phase-map drift.  The rankcap_k3 full loop passed the corrected local-box
cap validation and reached stable StopController phase-map and boundary-shift
metrics:

```text
phase_map_change = 0.0006204676775119246 < 0.002
boundary_shift_normal_sc = 0.002604166666666674 < 0.004166666666666667
boundary_shift_uniform_fflo = 0.0 < 0.004166666666666667
```

Formal convergence failed because:

```text
label_surprise_rate = 0.18359375 > 0.05
boundary_coverage_p95 = 0.006588078458684216 > 0.00625
```

The surprise decomposition audit shows that late-stage surprise is dominated
by `predicted normal -> exact FFLO` points.  Across audited acquisition
iterations this channel is strongly associated with q-window expansion and
rerun-heavy exact results:

```text
normal_to_FFLO surprise count = 814
qedge_or_expanded_rate = 0.9963144963144963
rerun_rate = 0.9434889434889435
```

This supports a cleanup strategy rather than another ordinary full-discovery
loop.  The cleanup should be opt-in and bounded.  It should start from the
completed full-loop dataset, focus on the remaining surprise channels and
boundary-coverage gaps, and explicitly monitor q-edge/rerun-heavy selections.

## Implementation Decision

An opt-in acquisition profile named `surprise_cleanup` was added.  It does not
change default `full` or `simple_phase` behavior.  It keeps phase/boundary
uncertainty active, removes q-edge risk as a positive numerical reward, and
records a q-edge penalty factor so cleanup selections remain auditable.

Do not change:

```text
thermodynamic phase criterion
exact oracle
rankcap_k3
StopController thresholds
Delta/q tolerances
default full acquisition profile
```

## Evidence Files

```text
reports/surprise_decomposition_audit/surprise_decomposition_audit.md
reports/surprise_decomposition_audit/surprise_decomposition_audit.pdf
reports/surprise_decomposition_audit/tables/surprise_by_iteration.csv
reports/surprise_decomposition_audit/tables/surprise_by_phase_transition.csv
reports/surprise_decomposition_audit/tables/cleanup_target_recommendations.csv
```

## Recommended Next Calculation

Run a one-batch cleanup validation from the completed full-loop dataset using:

```text
ACQUISITION_PROFILE=surprise_cleanup
START_ITER=31
N_ITERS=1
RUN_ID=<new cleanup validation id>
```

The validation should pass only if label surprise decreases, boundary coverage
improves, and q-edge/rerun-heavy selection does not dominate the cleanup batch.
