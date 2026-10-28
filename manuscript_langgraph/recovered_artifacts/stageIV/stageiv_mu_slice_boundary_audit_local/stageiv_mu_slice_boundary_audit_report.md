# Stage IV-A Mu-Slice Thermodynamic-Boundary Audit

This report is generated from existing Stage IV-A artifacts only. No exact calculation, Delta-q search, active-learning continuation, dataset edit, or label change was performed.

## Executive Summary

- final samples: `7081`
- thermodynamic counts: normal `2350`, uniform-SC `141`, FFLO `4590`
- topology counts: trivial `3265`, topological `1466`, gapless-SC `0`, unresolved `0`
- final_classification: `Class A - broad-bin projection artifact`
- recommended_next_action: `curve_extraction_fix_only`
- need_new_exact_calculation: `False`
- recommended_tail_candidate_count: `0`
- candidate_count_generated_for_traceability: `24`

## Direct Answers

1. Suspicious panels: `['0.50_0.83', '0.83_1.17', '1.17_1.50']`.
2. Broad-bin projection artifact: `partial`.
3. Narrow fixed-mu slices were generated with adaptive half-widths from 0.025 to 0.075.
4. Local bracket support computed; sampling support limited: `False`.
5. Thermodynamic margin support: `partial`.
6. Smooth-fit artifact: `medium`.
7. Sampling insufficiency: `False`.
8. Single-band/lower-mu relation: `partial`, `True`.
9. Hard-risk correlation: `partial`.
10. Overall class: `Class A - broad-bin projection artifact`.
11. Need new exact calculation: `False`.
12. Recommended next action: `curve_extraction_fix_only`.
13. Stage IV-A remains scientifically useful but not formally converged.

## Figures
- `figures/fig01_original_wide_bin_phase_atlas_highlighted.png`
- `figures/fig02_wide_bin_points_colored_by_mu.png`
- `figures/fig03_narrow_fixed_mu_exact_label_atlas.png`
- `figures/fig04_narrow_fixed_mu_atlas_with_brackets.png`
- `figures/fig05_anomaly_bin_local_bracket_support.png`
- `figures/fig06_smooth_curve_vs_bracket_supported_curve.png`
- `figures/fig07_delta_opt_anomaly_bins.png`
- `figures/fig08_free_energy_margin_anomaly_bins.png`
- `figures/fig09_weak_margin_hard_risk_overlay.png`
- `figures/fig10_boundary_reconstruction_sensitivity.png`
- `figures/fig11_acquisition_channel_support_by_mu_bin.png`
- `figures/fig12_selected_points_anomaly_bins_by_iteration.png`
- `figures/fig13_normal_sc_brackets_single_band_projection.png`
- `figures/fig14_phase_tail_candidates_for_unsupported_arcs.png`
- `figures/fig15_final_decision_summary_panel.png`

## Caveats

- Smooth fits are diagnostic only.
- Single-band corridor is diagnostic only.
- Candidate tail points are not submitted by this report.
- Hidden fixed-mu Stage III reference remains missing.
