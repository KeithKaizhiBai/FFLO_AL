# Phase 3 plan — deterministic figure regeneration

## Audit

Confirm that each main figure can be built from frozen endpoints, retained
selection histories, audit tables, or canonical model/numerics specifications.

## Plan

1. Build a shared colorblind-aware style and deterministic plotting utilities.
2. Prepare derived plotting tables without changing frozen labels.
3. Generate PDF, SVG, and PNG candidates for Figures 1–4.
4. Add one Supplementary support/sensitivity audit candidate.

## Critique

The main risks are unsupported interpolation, obscuring thin phase support,
mistaking Stage III brackets for a physical boundary, and visually implying
that the Stage IV topology surface is converged.

## Non-changes

No Hamiltonian, phase criterion, topology rule, endpoint, exact evaluation,
HPC state, or frozen archive file is modified.
