# Rankcap K3 Five-Iteration Validation Recheck

Date: 2026-06-10

## Question

The returned five-iteration validation package reported:

```text
validation_status = fail
max_local_boxes_refined_count = 24
```

Does this mean rank_and_cap_k3 failed the local-box cap in the closed-loop
validation?

## Answer

No.  The fail is a report aggregation false positive.

The raw rank-level local-box timing files are named:

```text
iterXXX_local_box_timing_rankYYY_of008.csv
```

Inside each file, `point_id` is rank-local.  The package collector merged these
files into `local_box_rows.csv` after inserting only `iteration`, not `rank`.
It then computed:

```python
local_box_df.groupby(["iteration", "point_id"]).size().max()
```

This incorrectly merges different ranks that share the same local `point_id`.
In the five-iteration return package, this produced a spurious maximum of
`8 ranks * 3 boxes = 24`.

Recomputing from the raw rank files with:

```text
(iteration, rank, point_id)
```

shows:

```text
actual exact points checked: 1792
local boxes per point: only 2 or 3
corrected max local boxes: 3
points above 3: 0
corrected validation_status: pass
```

## Evidence

The corrected report is:

```text
reports/rankcap_k3_5iter_validation_recheck/rankcap_k3_5iter_validation_recheck.md
reports/rankcap_k3_5iter_validation_recheck/rankcap_k3_5iter_validation_recheck.pdf
```

Key tables:

```text
reports/rankcap_k3_5iter_validation_recheck/tables/actual_local_box_point_counts.csv
reports/rankcap_k3_5iter_validation_recheck/tables/corrected_validation_gates.csv
reports/rankcap_k3_5iter_validation_recheck/tables/original_report_discrepancy.csv
```

The final dataset still has all three phases:

```text
normal = 608
uniform_SC = 102
FFLO = 982
samples = 1692
```

## Consequence

The five-iteration closed-loop validation supports rank_and_cap_k3 for the next
active-learning step.  The active-loop report collector should be patched before
using any full-loop package `validation_status` field as a decision gate.

This note does not change the Hamiltonian, phase criterion, acquisition,
StopController, Delta tolerance, final ambiguity tolerance, or exact oracle.
