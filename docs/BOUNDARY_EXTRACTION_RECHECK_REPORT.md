# Boundary Extraction Recheck Report

Date: 2026-05-12

## Purpose

This report rechecks the explicit boundary-extraction step that was performed
near the previous interrupted work session. The goal is to verify that the
archived boundary output is reproducible, that the input dataset is correctly
identified, and that the next targeted refinement list is not blindly dominated
by noisy or overrepresented boundary types.

## Inputs

Rechecked dataset family:

```text
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/active_runs/active_boundary_loop_v1/dataset_iter039.npz
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/active_runs/active_boundary_loop_v1/dataset_iter040.npz
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/active_runs/active_boundary_loop_v1/dataset_iter041.npz
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/active_runs/active_boundary_loop_v1/dataset_iter042.npz
```

All four files have the same array-content hash and contain 24083 samples.
Therefore `dataset_iter042` is a postprocessing/empty-append alias of
`dataset_iter039`, not a dataset with three additional H100 exact iterations.

## Default Reproduction

Default command:

```bash
python -m ml_phase.extract_phase_boundaries \
  --dataset hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/active_runs/active_boundary_loop_v1/dataset_iter042.npz \
  --output-dir hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/boundaries/recheck_iter042_default \
  --kt-bin-width 0.005 \
  --max-local-spacing 0.035 \
  --max-refinement-points 512
```

The archived counts were reproduced:

```text
n_exact_points: 24083
phase counts: normal=7011, uniform_SC=405, FFLO=16667
boundary segments: 8090
eta_zero: 7787
normal_sc: 122
strong_diode: 70
uniform_fflo: 111
confidence: high=5643, medium=2410, low=37
```

## Independent Checks

Dataset consistency:

```text
kT_negative_points: 0
nonfinite kT/JA/Delta/q/eta/Ic points: 0
phase-threshold mismatches: 0
needs_rerun_exact: 0
q_unresolved: 0
q_expanded: 27
delta_unresolved: 110
delta_boundary_band_normal: 110
duplicate coordinates, rounded to 4 decimals: 1
duplicate coordinates, rounded to 8 decimals: 0
```

Boundary predicate audit:

```text
normal_sc predicate failures: 0
uniform_fflo predicate failures: 0
eta_zero predicate failures: 0
strong_diode predicate failures: 0
boundary interpolation outside segment: 0
low-confidence rows without severe reason: 0
high-confidence rows with risk reason: 0
```

## Parameter Sensitivity

```text
case                         eta_zero  normal_sc  strong_diode  uniform_fflo
kt_bin=0.0025 spacing=0.035      7840        205            76           206
kt_bin=0.0050 spacing=0.035      7787        122            70           111
kt_bin=0.0100 spacing=0.035      7759        136            72            58
kt_bin=0.0050 spacing=0.020      7787        122            70           111
kt_bin=0.0050 spacing=0.050      7787        122            70           111
```

Interpretation:

```text
eta_zero is overwhelmingly numerous and remains around 7759-7840 segments.
normal_sc and uniform_fflo are more sensitive to kt_bin_width.
max_local_spacing changes confidence labels but not the extracted segments.
```

The default extraction is mechanically valid, but the resulting all-boundary
target list should not be used directly for the next H100 refinement because
`eta_zero` dominates the segment pool.

## Targeted Refinement Outputs

Audit outputs were written under:

```text
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/boundaries/recheck_iter042_audit/
```

Important files:

```text
boundary_recheck_summary.json
boundary_parameter_sensitivity.csv
boundary_extraction_recheck_report.md
targeted_refinement_prioritized_basic_filtered.csv
targeted_refinement_prioritized_radius_checked.csv
target_normal_sc_all_candidates.csv
target_uniform_fflo_all_candidates.csv
target_strong_diode_all_candidates.csv
target_eta_zero_all_candidates.csv
```

The planned strict `existing_min_dist=0.015` check rejects all midpoint
targets:

```text
radius-checked recommended points: 0
basic-filtered points before hard radius exclusion: 168
```

This is not a numerical failure. It means that the dense-grid acquisition
exclusion radius is too strict for boundary-bracket midpoint refinement. A
midpoint between two exact bracket points is expected to be close to existing
points by construction. Future boundary refinement should either:

```text
1. explicitly relax the existing-distance rule for bracket midpoints; or
2. define a smaller boundary-specific exclusion radius; or
3. refine only low-confidence/large-spacing brackets where the midpoint
   distance is physically meaningful.
```

Until this is decided, use `targeted_refinement_prioritized_basic_filtered.csv`
as an inspectable candidate list, not as an automatic H100 submission list.

## Conclusion

The archived boundary extraction is reproducible and internally consistent.
The interrupted work did not corrupt the extracted default boundary CSVs.
However, the refinement-policy layer is not yet finalized:

```text
1. eta_zero must be quota-limited or filtered before H100 refinement;
2. normal_sc and uniform_fflo should receive explicit quotas;
3. existing_min_dist=0.015 should not be blindly applied to bracket midpoint
   refinement;
4. a boundary-specific target-selection policy should be implemented before
   launching the next production H100 run.
```
