# Local-Refinement Stage 1 CUDA Node Exclusion Q&A

Date: 2026-06-03

## Question

Why did the Stage 1 fixed-point regression fail with:

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 12080)
```

## Short Answer

The job was scheduled onto a GPU node whose NVIDIA driver is incompatible with
the PyTorch/CUDA environment used by the package.  Project decision D11 already
records that `gpuh01` should be excluded from automatic H100 jobs because it
has shown this driver mismatch.

## Fix

The Stage 1 package generator now adds this directive to both GPU exact
regression Slurm scripts:

```text
#SBATCH --exclude=gpuh01
```

The generated baseline and instrumented exact jobs also run a CUDA runtime
probe before starting the fixed-point solver:

```text
torch.empty(1, device="cuda")
```

If the CUDA runtime probe fails, the job exits before the exact oracle begins,
and the failure is written to the job environment log under `RUN_ROOT/logs/`.

## What Did Not Change

This is scheduling/runtime protection only.  It does not change the
Hamiltonian, exact-oracle parameters, fixed-point input set, q-window policy,
Delta-refinement policy, local-box instrumentation, or Stage 1 gate criteria.

## Required Action

Upload the rebuilt Stage 1 package and rerun the workflow.  Do not reuse the
old package whose GPU Slurm scripts lack the `gpuh01` exclusion.
