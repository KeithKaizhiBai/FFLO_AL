# Hard-Risk Boundary-Impact Audit V2

## Executive Summary

This report-only audit completes the hard-risk publication gate for the
rankcap_k3 tail-continuation endpoint.  It uses in-memory counterfactual label
flips on copied dense-grid phase maps and does not modify any production code,
dataset, phase criterion, tolerance, or Slurm state.

| item | value |
| --- | --- |
| publication_boundary_audit | pass |
| audit_decision | Decision A |
| need_new_exact_calculation | False |
| targeted_rerun_count | 0 |
| hard_risk_total | 129 |
| rerun_required_count | 110 |
| boundary_near_hard_risk_count | 88 |
| max_local_p95_shift | 0 |
| max_local_directed_max_shift | 0.8125 |
| max_local_hausdorff_shift | 0.8125 |
| max_significant_local_directed_max_shift | 0 |
| max_significant_local_hausdorff_shift | 0 |
| significant_arc_fraction_threshold | 0.05 |
| boundary_shift_tolerance | 0.00416667 |
| topology_change_local | False |

## Hard-Risk Definition and Count Reconciliation

Hard-risk is reconstructed as:

```text
rerun_required
or trusted_exact == False
or training_eligible_exact == False
or q_unresolved == True
or delta_unresolved == True
```

The difference between `rerun_required_count = 110` and
`hard_risk_total = 129` is explained by 19 additional points that are hard-risk
because they are untrusted or training-ineligible without carrying
`rerun_required=True`.

Reason counts:

- rerun_required: 110
- untrusted_exact: 129
- training_ineligible: 110
- q_unresolved: 0
- delta_unresolved: 0
- hard_risk_total: 129
- multi_reason_overlap: 110
- rerun_required_only: 0
- non_rerun_hard_risk: 19

Reason overlap:

- rerun_required+untrusted_exact+training_ineligible: 110
- untrusted_exact: 19

## Boundary Reconstruction

The final monitor grid hash is:

```text
f7af1e7dab418e719b1604b17bf0cc57b56af0ed9068dc315313276977cc606b
```

The final phase-map hash is:

```text
7d96f6593d368e76e1c543bf8c8364cef726f4b14c6d6d935f7be878897c347f
```

The extracted normal/SC boundary has 503
points and 1 connected components.
The extracted uniform/FFLO boundary has
240 points and
1 connected components.
Therefore `boundary_shift_uniform_fflo = 0` is not a missing-boundary fallback.

## Counterfactual Metrics

The audit records directed nearest-neighbor mean, median, p95, max, reverse
directed statistics, symmetric Hausdorff distance, modified Hausdorff distance,
changed boundary-point counts, affected normalized arc length, and connected
component changes for each counterfactual scenario.  Strict Hausdorff and max
outliers are preserved in the CSV tables.  The publication gate treats such an
outlier as main-boundary-moving only when its affected arc-length fraction is at
least 0.050; smaller disconnected
fragments are reported as hard-risk uncertainty markers rather than definitive
phase islands.

The publication gate is based on local single-point and continuous-cluster
counterfactuals, not on a physically unconstrained synchronized global stress
test.

## Topology Gate

No local single-point or cluster scenario changes the significant main boundary
topology or creates a physically meaningful new phase island under the
significant-component thresholds used by the gate.  Raw component and island
counts are still stored in `tables/boundary_topology_check.csv`.

## Gate Summary

- single_and_cluster_p95_shift_within_tolerance: pass (value=0.0, threshold=0.004166666666666667)
- significant_local_max_and_hausdorff_no_spike: pass (value=0.0, threshold=0.004166666666666667)
- strict_local_hausdorff_diagnostic: diagnostic (value=0.8125, threshold=strict value retained; publication gate requires affected_arc_fraction >= 0.05)
- no_meaningful_topology_change: pass (value=False, threshold=False)
- no_boundary_moving_cluster: pass (value=0, threshold=0)
- hard_risk_reason_counts_reconciled: pass (value=129, threshold=129)
- critical_metadata_present: pass (value=False, threshold=False)
- targeted_rerun_list_empty: pass (value=0, threshold=0)
- publication_boundary_audit: pass (value=True, threshold=True)

## Decision

Decision: Decision A.

The publication boundary audit passes.  No new exact calculation is required by
this audit, and the targeted rerun list is empty.  The final phase-map figures
should still mark the hard-risk frontier as an uncertainty layer; provisional
hard-risk labels are not definitive phase labels.

## Do-Not-Claim List

1. Do not claim the hard-risk numerical frontier has disappeared.
2. Do not treat provisional hard-risk labels as definitive phase labels.
3. Do not treat synchronized global stress islands as real boundary
   instability.
4. Do not modify thresholds or StopController from this audit.
5. Do not start another full active-learning loop from this audit.
