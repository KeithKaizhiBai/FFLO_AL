# Basin-Level Risk Refactor Q&A

## Question

Why do we need a basin-level risk refactor before trying to fix the local-box
target explosion?

## Short Answer

The returned variant-array audit showed that optimized local-refinement
variants were dominated by mandatory targets. Before applying a hard cap or a
rank-and-cap policy, the code must make the selection unit explicit: a target
should be a basin representative with aggregated risk metadata, not an
unexplained raw candidate row.

## Technical Notes

The updated local-refinement target path keeps the existing physical and
numerical criteria unchanged. It adds basin-level metadata after clustering:

- `basin_risk_flags`
- `basin_has_global_best`
- `basin_has_edge_risk`
- `basin_has_delta_near_epsilon`
- `basin_has_near_degenerate`
- `basin_has_low_energy_window`
- per-risk member counts

Mandatory target selection and energy-window pruning now prefer these
basin-level fields when they exist. If clustering is not enabled, the old
candidate-level logic remains the fallback, so the baseline feature path is not
silently redefined.

This refactor does not solve the overflow by itself. The observed target
explosion still requires a later Stage 2 policy such as
`high_risk_overflow_policy = rank_and_cap` or explicit hard total cap
enforcement.

## Report Use

Use this as the narrative bridge between the target-logic audit and the
rank-and-cap optimization stage:

```text
Stage 1 changed the unit of explanation from raw candidates to basins. It made
mandatory target construction auditable, but intentionally left the overflow
policy unchanged so that the next stage can test rank-and-cap without mixing it
with risk-annotation changes.
```

