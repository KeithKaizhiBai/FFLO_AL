# Rank-Cap K3 One-Iteration Validation Return

Date: 2026-06-09

## Question

What did the returned one-iteration active-learning validation show for `rank_and_cap_k3`?

## Answer

The returned run passed the one-iteration validation:

```text
validation_status=pass
run_id=active_boundary_discovery_rankcap_k3_one_iter_validation_v1
```

The workflow executed the intended semantics:

```text
iter000: random seed exact batch
append: dataset_iter001
iter001: exactly one acquisition-selected batch
append: dataset_iter002
stop
```

There was no `iter002` acquisition batch.

The final dataset grew monotonically:

```text
dataset_iter000: 0 samples
dataset_iter001: 506 samples
dataset_iter002: 744 samples
```

Final phase counts were:

```text
normal=377
uniform_SC=13
FFLO=354
```

The one acquisition-selected exact batch had:

```text
selected points=256
training_eligible_count=238
training_eligible_fraction=0.929688
rerun_required_fraction=0.0703125
q_unresolved_count=0
delta_unresolved_count=0
```

Rank-and-cap workload metrics for the acquisition batch were:

```text
mean_local_boxes_refined_count=2.79297
max_local_boxes_refined_count=3
mean_local_refinement_runtime_sec=88.3859
mean_point_total_runtime_sec=112.359
```

The logs and metadata showed:

```text
exact shards complete: confirmed
merge and append: confirmed
final report generation: confirmed
fallback_full_rescan_runtime_sec_sum=0 for both iter000 and iter001
no traceback
no OOM
no CUDA initialization failure
no timeout or cancellation
```

Rank runtime imbalance was small:

```text
iter000 max/min elapsed ratio=1.0186
iter001 max/min elapsed ratio=1.0654
```

The report-only collector was rerun locally from returned data to regenerate:

```text
reports/rankcap_k3_one_iter_validation/rankcap_k3_one_iter_validation.md
reports/rankcap_k3_one_iter_validation/rankcap_k3_one_iter_validation.pdf
reports/rankcap_k3_one_iter_validation/tables/*.csv
reports/rankcap_k3_one_iter_validation/figures/*.png
```

No physical calculation was rerun.

## Decision

The result supports proceeding to a 3-5 iteration mini AL validation for `rank_and_cap_k3` only.

It does not validate:

```text
full-length AL
k2
energy-window pruning
branch reuse
Powell
adaptive box
GPU batching
Hamiltonian cache
```

Those remain separate decisions.
