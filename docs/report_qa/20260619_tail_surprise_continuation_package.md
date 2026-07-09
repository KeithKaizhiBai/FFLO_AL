# Tail Surprise Continuation Package

Date: 2026-06-19

## Question

Should the next computation be an independent package that starts from the
downloaded rankcap_k3 full-loop tail and runs a few more active-learning
iterations to test whether surprise decreases?

## Answer

Yes.  The short trusted-surprise validation from scratch tested that the new
StopController path and rankcap_k3 package machinery run, but it did not answer
the late-stage question from the full loop.  The relevant test is a continuation
from the full-loop endpoint:

```text
source run:
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/
  active_runs/active_boundary_discovery_rankcap_k3_full_loop_v1

restart dataset:
dataset_iter031.npz
```

The continuation package should be self-contained.  It must include the
restart dataset, previous monitor predictions, stop and metrics history, run
configuration, tail artifacts, and all runnable code needed by Slurm.  It
should not rely on paths outside the uploaded package.

## Package

```text
hpc_packages/rankcap_k3_tail_surprise_continuation_v1.tar.gz
```

Default run settings:

```text
START_ITER=31
N_ITERS=5
STOP_SURPRISE_MODE=trusted
TRUSTED_SURPRISE_MIN_DENOMINATOR=64
TRUSTED_SURPRISE_MIN_FRACTION=0.25
EXCLUDE_NODES=gpuh01
```

## Interpretation Goal

The returned run should separate three possibilities:

```text
1. trusted surprise stays low while all-selected surprise remains high:
   all-selected surprise is mainly acquisition difficulty / numerical frontier.

2. trusted surprise rises above tolerance:
   clean trusted labels still expose phase-model errors near the boundary.

3. surprise is acceptable but boundary coverage remains failed:
   boundary coverage, not surprise, is the next cleanup target.
```

## Non-Changes

This package does not change:

```text
thermodynamic phase criterion
Delta refinement trigger tolerance
final ambiguity tolerance
acquisition formula
candidate-domain strategy
rankcap_k3 local-refinement policy
StopController thresholds
k2
energy-window pruning
branch reuse
Powell
adaptive box
GPU batching
Hamiltonian cache
```
