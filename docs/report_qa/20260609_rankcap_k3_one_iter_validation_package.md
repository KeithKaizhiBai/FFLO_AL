# Rank-Cap K3 One-Iteration Validation Package

Date: 2026-06-09

## Question

How is the one-iteration active-learning validation for `rank_and_cap_k3` packaged, and what exactly will it validate?

## Answer

The package `hpc_packages/rankcap_k3_one_iter_validation.tar.gz` contains a complete HPC workflow for one active-learning validation of the accepted `rank_and_cap_k3` local-refinement path.  It does not run mini AL or full AL.  It runs:

```text
iter000: random seed exact batch
append: dataset_iter001
iter001: exactly one acquisition-selected batch
append: dataset_iter002
stop
```

The run id is:

```text
active_boundary_discovery_rankcap_k3_one_iter_validation_v1
```

The output root is:

```text
ML_Phase_512_Speed_20260602
```

The active oracle settings are:

```text
acquisition_profile=full
oracle_mode=robust_incremental
incremental_q_expansion=True
enable_basin_clustering=True
enable_selective_refinement=True
max_refined_minima=3
max_optional_refined_basins=3
mandatory_basins_can_exceed_cap=False
high_risk_overflow_policy=rank_and_cap
```

The intentionally disabled paths are:

```text
k2
energy-window pruning
branch reuse
adaptive box
Powell
GPU batching
Hamiltonian cache
acquisition changes
phase-criterion changes
tolerance changes
StopController changes
```

The package excludes `gpuh01` by default:

```text
EXCLUDE_NODES=gpuh01
```

The background HPC command from the extracted package root is:

```bash
nohup bash scripts/submit_rankcap_k3_one_iter_validation.sh > rankcap_k3_one_iter_validation.nohup.log 2>&1 &
```

The expected return archive after successful collection is:

```text
ML_Phase_512_Speed_20260602/rankcap_k3_one_iter_validation_results.tar.gz
```

The local preflight passed before packaging.  The uploaded package includes the preflight markdown, preflight CSV, the rankcap fixed-point acceptance summary, and the target-logic audit material needed for report context.

## Validation Scope

The returned report should answer whether `rank_and_cap_k3` remains healthy inside the real active-learning data flow:

```text
exact shards complete
merge and append complete
dataset size grows monotonically
training_eligible appended is nonzero
normal / uniform-SC / FFLO remain represented
uniform-SC does not disappear
q_unresolved and delta_unresolved remain controlled
rerun_required fraction remains controlled
mean local_boxes_refined_count <= 3.2
max local_boxes_refined_count <= 3 unless explicitly reported fallback exists
local-refinement runtime is below the robust_incremental reference
no traceback, OOM, CUDA init failure, silent fallback, or silent mismatch
```

## Current Status

```text
validation_status=pending_hpc_run
```

The package is ready, but no HPC result has been returned yet.  Do not enter 3-5 iteration mini AL or full AL until the returned one-iteration validation report passes.
