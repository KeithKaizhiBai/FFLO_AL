# Figure 2 specific validation

- Status: **PASS**
- Checks: **21/21**
- Physics recomputed: **no**

| Result | Check | Detail |
|---|---|---|
| PASS | `source_hash:endpoint` | /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/manuscript_langgraph/recovered_artifacts/stageII/dataset_iter035.npz |
| PASS | `source_hash:cold_start` | /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/downloads/active_learning_scientific_evidence_archive_20260813/active_learning_scientific_evidence_archive/data_archive/stage_i/active_boundary_discovery_rankcap_k3_full_loop_v1/selection_history/iter000/selected_points.csv |
| PASS | `source_hash:cold_start_labels` | /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/downloads/active_learning_scientific_evidence_archive_20260813/active_learning_scientific_evidence_archive/data_archive/stage_i/active_boundary_discovery_rankcap_k3_full_loop_v1/dataset_iter031.npz |
| PASS | `source_hash:selection_history_index` | /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/figures/fig2_strict_data/canonical/fig2_selection_history_sources.json |
| PASS | `source_hash:stop_metrics` | /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/manuscript_langgraph/recovered_artifacts/stageII/run_bundle/stop_metrics_history.json |
| PASS | `source_hash:hard_risk` | /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/manuscript_langgraph/recovered_artifacts/stageII/figure_data/hard_risk_boundary_impact_audit_v2/hard_risk_point_manifest.csv |
| PASS | `canonical:fig2_final_exact_samples.csv` | rows=7434; expected=7434 |
| PASS | `canonical:fig2_cold_start_exact_calls.csv` | rows=512; expected=512 |
| PASS | `canonical:fig2_exact_call_allocation.csv` | rows=9216; expected=9216 |
| PASS | `canonical:fig2_raw_convergence_metrics.csv` | rows=35; expected=35 |
| PASS | `canonical:fig2_hard_risk_points.csv` | rows=129; expected=129 |
| PASS | `canonical:fig2_phase_counts.csv` | rows=3; expected=3 |
| PASS | `stage2_phase_invariants` | total=7434; counts={'FFLO': 4852, 'normal': 1867, 'uniform_SC': 715} |
| PASS | `cold_start_reconciliation` | selected=512; stage1_exact=506; retained_stage2=506; unmatched=6 |
| PASS | `selection_history_complete` | iterations=0-34; calls=9216 |
| PASS | `hard_risk_separate_layer` | risk_manifest=129; risk_overlay=19; phases=['FFLO', 'normal', 'uniform_SC'] |
| PASS | `raw_convergence_complete` | iterations=35; final_phase_map_change=0.00162872765347 |
| PASS | `phase_mapping_frozen` | {'normal': 0, 'uniform_SC': 1, 'FFLO': 2} |
| PASS | `no_unauthorized_transform` | [] |
| PASS | `review_outputs_present` | fig2_strict_data.svg, fig2_strict_data.pdf, fig2_strict_data.png, fig2_contact_sheet.pdf, fig2_contact_sheet.png, fig2_strict_data_grayscale.png |
| PASS | `pdf_single_page_print_size` | Page size:       504.567 x 340.157 pts |
