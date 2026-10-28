# Stage IV-A 3D Return-Run Audit

This is a report-only audit generated from the returned Stage IV-A 3D topology-aware active-learning run. It does not modify exact labels, does not launch new exact calculations, and does not continue the active-learning loop.

## Executive Summary

- source_run_id: `active_phase_topology_3d_t_ja_mu_from_scratch_v1`
- final_dataset: `D:\PhD Study\4th\SC_GBU\Fu_FFLO\active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\ML_Phase_StageIV_Topology3D\active_runs\active_phase_topology_3d_t_ja_mu_from_scratch_v1\dataset_iter025.npz`
- final_dataset_sha256: `7acab805115bcfcae56e94e2541bfdaa727de7767f2d7dd34330f163e2695842`
- final samples: `7081`
- phase counts: normal `2350`, uniform-SC `141`, FFLO `4590`
- topology counts: not_applicable `2350`, trivial `3265`, topological `1466`, gapless-SC `0`, unresolved `0`
- decision_class: `Decision D`
- recommended_next_action: `collect_missing_stageiii_reference_then_choose_same_window_topology_tail_or_lower_mu_extension_from_edge_audit`

## Direct Answers

1. Did Stage IV-A recover the Stage III fixed-mu reference slice? `inconclusive`. The Stage III frozen reference dataset is missing in the returned package, so recovery cannot be determined from current artifacts.
2. Why is 3D topology convergence failing? Main limiter: `trusted_topology_surprise`; secondary limiters: `coverage, surface_shift, component_instability, mu_edge_limitation`.
3. Is the failure mainly coverage, surprise, component instability, or mu-edge limitation? `trusted_topology_surprise` is the primary numerical blocker; coverage, surface shift, trusted topology surprise, component instability, and lower-mu edge contact all remain material diagnostics.
4. Is `mu/t in [-0.5, 1.5]` enough? `cannot determine as complete`; lower edge is range-limited: `True`, upper edge range-limited: `False`.
5. Which mu boundary limits? Lower-mu boundary. The upper boundary has no tFFLO contact in the edge bands used here.
6. Does tFFLO overlap the normal-state single-band corridor? `supported`; tFFLO single-pair fraction `0.9653979238754326`.
7. Are multiple topology components physical or artifacts? `cannot determine`. Component counts are not stable in the last three iterations, so the report treats extra components as unresolved diagnostics rather than final physical islands.
8. Did spectral/topology acquisition receive intended quota? `confirmed`. Per-iteration metadata is available and shows phase/spectral/coverage quota fulfillment.
9. Are hard-risk/training-ineligible points concentrated on topology surface? `not_concentrated_on_topology_surface`.
10. Next step: `collect_missing_stageiii_reference_then_choose_same_window_topology_tail_or_lower_mu_extension_from_edge_audit`.
11. How many new exact points if any? `2_to_4_after_reference_collection` tail batches, quota `spectral/topology_surface_coverage_focused; preserve phase quota` if proceeding after missing-reference artifacts are collected.
12. Is from-scratch restart necessary? `False`.

## Key Failure Metrics

|metric|iteration_or_transition|value|threshold|pass|failure_class|threshold_ratio|denominator|n_surprise|
|---|---|---|---|---|---|---|---|---|
|topology_volume_map_change|22->23|0.0014122480421106|0.002|True|||||
|topology_volume_map_change|23->24|0.0019225839528326|0.002|True|||||
|topology_volume_map_change|24->25|0.002561475409836|0.002|False|map_change||||
|topology_surface_shift_p95|22->23|0.0055835372535258|0.004167|False|surface_shift||||
|topology_surface_shift_p95|23->24|0.0059066811860211|0.004167|False|surface_shift||||
|topology_surface_shift_p95|24->25|0.0059961957165026|0.004167|False|surface_shift||||
|topology_surface_coverage_p95|25|0.015317527016765|0.00625|False|coverage|2.4508043226823997|||
|trusted_topology_surprise|22->23|0.065|0.02|False|trusted_surprise||200|13|
|trusted_topology_surprise|23->24|0.1122994652406417|0.02|False|trusted_surprise||187|21|
|trusted_topology_surprise|24->25|0.140625|0.02|False|trusted_surprise||192|27|
|component_stability|last3|3;4;3|constant component count|False|component_instability||||

## Hidden Slice Validation

|status|decision_class|reason|mu_reference|reference_dataset|evaluated_iterations|need_new_exact_calculation|stageiv_dataset_rows_in_provenance|
|---|---|---|---|---|---|---|---|
|inconclusive|Decision D|reference_dataset_missing|0.55|None||False|26|

## Mu-Edge Contact

|edge|mu_edge|width|trusted_tfflo_count|trusted_cfflo_count|topology_boundary_proxy_count|range_limited|phase|mu_min_observed|mu_max_observed|count|
|---|---|---|---|---|---|---|---|---|---|---|
|lower|-0.5|0.02|58|79|114|True|||||
|upper|1.5|0.02|0|19|0|False|||||
|lower|-0.5|0.04|91|126|175|True|||||
|upper|1.5|0.04|0|28|0|False|||||
|lower|-0.5|0.08|130|231|263|True|||||
|upper|1.5|0.08|0|58|0|False|||||
|lower|-0.5|0.1|145|262|298|True|||||
|upper|1.5|0.1|0|64|0|False|||||
|lower|-0.5|0.15|178|338|375|True|||||
|upper|1.5|0.15|0|80|1|True|||||
|summary|||||||tFFLO|-0.5|1.283287945560402|1445|
|summary|||||||cFFLO|-0.5|1.5|3145|

## Single-Band Corridor Overlap

|group|denominator|inside_single_pair_corridor|outside_single_pair_corridor|inside_fraction|operational_definition|
|---|---|---|---|---|---|
|tFFLO|1445|1395|50|0.9653979238754326|single_pair_diagnostic == 1, equivalent to one pair of normal-state Fermi crossings in the preflight scan|
|cFFLO|3145|1122|2023|0.3567567567567568|single_pair_diagnostic == 1, equivalent to one pair of normal-state Fermi crossings in the preflight scan|
|topology_boundary_proxy|3231|2212|1019|0.6846177653977097|single_pair_diagnostic == 1, equivalent to one pair of normal-state Fermi crossings in the preflight scan|
|preflight_scan|3072|2029|1043|0.6604817708333334|computed on the normal-state (JA, mu) guard grid, independent of thermodynamic labels|

## Hard-Risk Distribution

|group|count|hard_risk_count|on_topology_boundary_proxy|inside_single_band_corridor|
|---|---|---|---|---|
|trusted_exact_false|314|314|0|168|
|training_eligible_false|0|0|0|0|
|rerun_required|0|0|0|0|
|q_unresolved|0|0|0|0|
|delta_unresolved|0|0|0|0|
|fflo_topology_untrusted|0|0|0|0|
|hard_risk_phase_normal|314|314|0|168|
|hard_risk_phase_uniform_SC|0|0|0|0|
|hard_risk_phase_FFLO|0|0|0|0|
|hard_risk_total|314|314|0|168|

## Acquisition Forensics

|acquisition_channel|iterations_present|total_selected|mean_per_iteration|mean_score|
|---|---|---|---|---|
|coverage|24|923|38.458333333333336|0.20747747637927216|
|phase|24|2097|87.375|0.3484935708852762|
|spectral|24|3124|130.16666666666666|0.4373914483517006|
|unknown|1|1024|1024.0|nan|

## Figures

- `figures/phase_3d_transparent.png`
- `figures/topology_3d_transparent.png`
- `figures/phase_mu_slice_atlas.png`
- `figures/topology_mu_slice_atlas.png`
- `figures/stageiv_convergence_failure_summary.png`
- `figures/stageiv_convergence_failure_summary.pdf`
- `figures/stageiv_mu_edge_contact.png`
- `figures/stageiv_mu_edge_contact.pdf`
- `figures/stageiv_mu_edge_boundary_proxy.png`
- `figures/stageiv_mu_edge_boundary_proxy.pdf`
- `figures/stageiv_single_band_overlap.png`
- `figures/stageiv_single_band_overlap.pdf`
- `figures/stageiv_single_band_overlap_counts.png`
- `figures/stageiv_single_band_overlap_counts.pdf`
- `figures/stageiv_acquisition_channel_forensics.png`
- `figures/stageiv_acquisition_channel_forensics.pdf`
- `figures/stageiv_hard_risk_distribution.png`
- `figures/stageiv_hard_risk_distribution.pdf`
- `figures/stageiv_next_action_decision_panel.png`
- `figures/stageiv_next_action_decision_panel.pdf`

## Machine-Readable Outputs

Tables are under `tables`.

## Caveats

- Smoothed visual surfaces are diagnostic only and are not used as final physical topology surfaces.
- The normal-state single-band corridor is a guard diagnostic and is not a topology label.
- Stage III hidden-slice validation is blocked until the frozen Stage III reference artifact is supplied.
- No new exact calculation was run by this audit.
