# Rank-and-Cap Refactor Notes

Date: 2026-06-08

## Question

Why not directly change the old `cluster_optional_k3`, `cluster_optional_k2`,
and `cluster_energy_window` variants?

## Answer

Those variants are now part of the failure evidence.  They showed that K=3/K=2
did not reduce local-refinement target count because mandatory-risk basins were
allowed to bypass the total cap.  Changing their meaning would make the old HPC
return report harder to interpret.

The safer design is to preserve the old variants and introduce new explicit
variants:

```text
rank_and_cap_k3
rank_and_cap_k2
rank_and_cap_energy_window
```

These variants opt into:

```text
high_risk_overflow_policy = rank_and_cap
mandatory_basins_can_exceed_cap = False
max_edge_risk_basins = 1
max_delta_near_eps_basins = 2
max_near_degenerate_basins = 2
```

## What Changed

Mandatory basins are no longer all kept under the new policy.  The selector
keeps the global best basin, then ranks and caps edge-risk,
Delta-near-epsilon, and near-degenerate basins before filling remaining slots
with ordinary optional basins.

The target list is capped by `max_total_refined_basins`.  Branch-candidate CSVs
record whether each candidate was selected, why mandatory candidates were
dropped, and which overflow policy was active.

## What Was Not Changed

No thermodynamic phase criterion, Delta tolerance, final ambiguity tolerance,
acquisition logic, or Slurm behavior was changed.

## Report Use

This supports a narrative distinction between method failure and implementation
failure: top-k and energy-window ideas were not disproven physically.  The
observed failure was that mandatory target construction bypassed the intended
cap.  The next report should present rank-and-cap as a target-construction
correction that must pass a 32-point dry-run before any expensive GPU rerun.
