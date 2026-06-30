# 2026-06-08 Local-Refinement Target Logic Audit

## Question

Why did the selective local-refinement variants produce about 85 local boxes on
completed common points, and should the 72 missing optimized-variant tasks be
rerun directly?

## Answer

The count of about 85 is confirmed to be completed refined local boxes on
completed rows, not raw candidate count.  The source is
`local_boxes_refined_count`, produced in `ml_phase/exact_oracle.py` as
`len(refined_rows)` after the local-box scan loop.  The local-box timing CSV
also contains one row per completed local-box scan, and the completed
selective variants have 681 rows over 8 completed points, giving 85.125 boxes
per point.

The immediate implementation-level cause is mandatory overflow.  In the
selective path, `select_local_refine_targets(...)` first keeps all mandatory
basins, then applies `max_optional_refined_basins` only to ordinary optional
basins.  The current optimized variants set
`mandatory_basins_can_exceed_cap=True`, so `max_total_refined_basins=6` does
not cap the mandatory list.  Completed selective points are dominated by
`Delta_near_epsilon` mandatory targets, so K=3 and K=2 do not reduce the final
target count.

Energy-window pruning is confirmed to be ordinary-only.  Therefore it cannot
reduce the target count when mandatory branches dominate.  This explains why
`cluster_energy_window` behaves like the K variants on completed points and
has `energy_window_pruned_count=0`.

Hard-risk timeout points do not contain enough returned metadata to prove
their exact selected target count.  They have startup JSON only; no point CSV,
NPZ, or local-box timing CSV was flushed.  Therefore hard-risk timeout target
explosion is supported by the completed clean controls and by the code path,
but not directly proven for the timed-out points.

## Practical Conclusion

Do not directly rerun the 72 missing optimized-variant tasks yet.  The minimum
safe next step is to add a target-construction-only gate or dry-run that saves
raw candidates, clustered basins, mandatory reasons, optional candidates, and
the final selected target list before any local-box scan.  Then test a
rank-and-cap mandatory overflow policy before spending more GPU time.

## Report Outputs

```text
reports/local_refinement_target_logic_audit/target_logic_audit.md
reports/local_refinement_target_logic_audit/target_logic_audit.pdf
reports/local_refinement_target_logic_audit/tables/
reports/local_refinement_target_logic_audit/figures/
```
