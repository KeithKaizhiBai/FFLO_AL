#!/usr/bin/env python3
"""Prepare immutable plotting tables and strict-data contracts from frozen evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANON = HERE / "canonical"
STAGE2 = ROOT / "manuscript_langgraph/recovered_artifacts/stageII"
ARCHIVE = ROOT / "downloads/active_learning_scientific_evidence_archive_20260813/active_learning_scientific_evidence_archive"
STAGE1_RUN = ARCHIVE / "data_archive/stage_i/active_boundary_discovery_rankcap_k3_full_loop_v1"
STAGE2_RUN = STAGE2 / "run_bundle"
HARD_RISK = STAGE2 / "figure_data/hard_risk_boundary_impact_audit_v2/hard_risk_point_manifest.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path, *, version: str, fields: list[str], units: dict[str, str], row_count: int | None = None) -> dict[str, Any]:
    if row_count is None:
        if path.suffix == ".csv":
            row_count = len(pd.read_csv(path))
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                row_count = int(data["x"].shape[0])
        elif path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            row_count = len(value) if isinstance(value, list) else None
    return {"path": str(path.resolve()), "sha256": sha256(path), "version": version,
            "row_count": row_count, "fields": fields, "units": units}


def transform_defaults() -> dict[str, dict[str, Any]]:
    return {name: {"enabled": False, "authorized": False}
            for name in ("interpolation", "smoothing", "boundary_extraction", "aggregation", "normalization")}


def phase_name(code: int) -> str:
    return {0: "normal", 1: "uniform_SC", 2: "FFLO"}[int(code)]


def prepare_stage2() -> dict[str, Any]:
    CANON.mkdir(parents=True, exist_ok=True)
    endpoint = STAGE2 / "dataset_iter035.npz"
    with np.load(endpoint, allow_pickle=False) as d:
        x = np.asarray(d["x"], float)
        y = np.asarray(d["y_reg"], float)
        phase = np.asarray(d["y_phase"], int)
        final = pd.DataFrame({
            "kT": x[:, 0], "JA": x[:, 1], "Delta_opt": y[:, 0], "q_opt": y[:, 1],
            "eta": y[:, 2], "Ic_plus": y[:, 3], "Ic_minus": y[:, 4],
            "phase_code": phase, "phase_name": [phase_name(v) for v in phase],
            "trusted_exact": np.asarray(d["trusted_exact"], int),
            "training_eligible_exact": np.asarray(d["training_eligible_exact"], int),
            "needs_rerun_exact": np.asarray(d["needs_rerun_exact"], int),
            "q_unresolved": np.asarray(d["q_unresolved"], int),
            "delta_unresolved": np.asarray(d["delta_unresolved"], int),
            "delta_boundary_ambiguous": np.asarray(d["delta_boundary_ambiguous"], int),
        })
    counts = final["phase_name"].value_counts().to_dict()
    expected = {"normal": 1867, "uniform_SC": 715, "FFLO": 4852}
    if len(final) != 7434 or counts != expected:
        raise RuntimeError(f"Stage II invariants mismatch: total={len(final)}, counts={counts}")

    risk = pd.read_csv(HARD_RISK)
    risk_keys = {(round(float(r.kBT), 12), round(float(r.JA), 12)): r for r in risk.itertuples()}
    final["hard_risk"] = [int((round(a, 12), round(b, 12)) in risk_keys) for a, b in final[["kT", "JA"]].to_numpy()]
    final_path = CANON / "fig2_final_exact_samples.csv"
    final.to_csv(final_path, index=False, float_format="%.16g")

    seed_path = STAGE1_RUN / "selection_history/iter000/selected_points.csv"
    seed = pd.read_csv(seed_path)
    stage1_endpoint = STAGE1_RUN / "dataset_iter031.npz"
    with np.load(stage1_endpoint, allow_pickle=False) as d1:
        stage1_x = np.asarray(d1["x"], float)
        stage1_phase = np.asarray(d1["y_phase"], int)
    stage1_lookup = {(round(a, 12), round(b, 12)): i for i, (a, b) in enumerate(stage1_x)}
    endpoint_lookup = {(round(a, 12), round(b, 12)): i for i, (a, b) in enumerate(final[["kT", "JA"]].to_numpy())}
    seed_rows = []
    for row in seed.itertuples(index=False):
        key = (round(float(row.kT), 12), round(float(row.JA), 12))
        stage1_idx = stage1_lookup.get(key)
        idx = endpoint_lookup.get(key)
        seed_rows.append({"kT": row.kT, "JA": row.JA,
                          "stage1_exact_match": stage1_idx is not None,
                          "endpoint_match": idx is not None,
                          "phase_code": int(stage1_phase[stage1_idx]) if stage1_idx is not None else -1,
                          "phase_name": phase_name(stage1_phase[stage1_idx]) if stage1_idx is not None else "exact_result_unavailable"})
    cold = pd.DataFrame(seed_rows)
    if len(cold) != 512 or int(cold.stage1_exact_match.sum()) != 506 or int(cold.endpoint_match.sum()) != 506:
        raise RuntimeError(f"Cold-start reconciliation mismatch: rows={len(cold)}, stage1_matches={cold.stage1_exact_match.sum()}, endpoint_matches={cold.endpoint_match.sum()}")
    if not (cold.stage1_exact_match == cold.endpoint_match).all():
        raise RuntimeError("Stage I and Stage II cold-start retention masks differ")
    cold_path = CANON / "fig2_cold_start_exact_calls.csv"
    cold.to_csv(cold_path, index=False, float_format="%.16g")

    selection_records, source_records = [], []
    for iteration in range(35):
        base = STAGE1_RUN if iteration <= 30 else STAGE2_RUN
        path = base / f"selection_history/iter{iteration:03d}/selected_points.csv"
        df = pd.read_csv(path)
        if len(df) != (512 if iteration == 0 else 256):
            raise RuntimeError(f"Unexpected selection count at iter{iteration:03d}: {len(df)}")
        for call_index, row in enumerate(df.itertuples(index=False)):
            selection_records.append({"iteration": iteration, "call_index": call_index,
                                      "kT": float(row.kT), "JA": float(row.JA),
                                      "call_role": "cold_start" if iteration == 0 else "active_selection"})
        source_records.append(file_record(path, version=f"selection_iter{iteration:03d}",
                                          fields=["kT", "JA"], units={"kT": "kBT/t", "JA": "JA/t"}))
    allocation = pd.DataFrame(selection_records)
    allocation_path = CANON / "fig2_exact_call_allocation.csv"
    allocation.to_csv(allocation_path, index=False, float_format="%.16g")
    selection_manifest = CANON / "fig2_selection_history_sources.json"
    selection_manifest.write_text(json.dumps(source_records, indent=2) + "\n", encoding="utf-8")

    stop_path = STAGE2_RUN / "stop_metrics_history.json"
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    convergence_rows = []
    for item in stop:
        m, th = item["metrics"], item["thresholds"]
        convergence_rows.append({
            "iteration": item["iteration"],
            "phase_map_change": m.get("phase_map_change"),
            "normal_sc_boundary_shift": m.get("boundary_shift_normal_sc"),
            "uniform_fflo_boundary_shift": m.get("boundary_shift_uniform_fflo"),
            "trusted_surprise_rate": m.get("label_surprise_trusted", m.get("label_surprise_rate")),
            "all_selected_surprise_rate": m.get("label_surprise_all_selected", m.get("label_surprise_rate")),
            "hard_risk_surprise_rate": m.get("label_surprise_hard_risk"),
            "map_threshold": th.get("map_tol"),
            "boundary_shift_threshold": th.get("boundary_shift_tol"),
            "surprise_threshold": th.get("surprise_tol"),
        })
    conv = pd.DataFrame(convergence_rows)
    if list(conv.iteration) != list(range(35)):
        raise RuntimeError("Stage II iteration history is incomplete")
    conv_path = CANON / "fig2_raw_convergence_metrics.csv"
    conv.to_csv(conv_path, index=False, float_format="%.16g")

    risk_path = CANON / "fig2_hard_risk_points.csv"
    risk.to_csv(risk_path, index=False, float_format="%.16g")
    phase_counts_path = CANON / "fig2_phase_counts.csv"
    pd.DataFrame([{"phase": k, "count": v} for k, v in expected.items()]).to_csv(phase_counts_path, index=False)

    inventory = {
        "schema_version": 1,
        "figure_id": "FIG-02-THERMO-STRICT",
        "archive_read_only": True,
        "physics_recomputed": False,
        "transformations": {"interpolation": False, "smoothing": False, "boundary_extraction": False,
                            "gridding": False, "resampling": False, "aggregation": False},
        "sources": {
            "endpoint": file_record(endpoint, version="dataset_iter035", fields=list(np.load(endpoint).files),
                                    units={"x[:,0]": "kBT/t", "x[:,1]": "JA/t", "y_phase": "code"}),
            "cold_start": file_record(seed_path, version="selection_iter000", fields=["kT", "JA"],
                                      units={"kT": "kBT/t", "JA": "JA/t"}),
            "cold_start_labels": file_record(stage1_endpoint, version="stageI_dataset_iter031", fields=list(np.load(stage1_endpoint).files),
                                              units={"x[:,0]": "kBT/t", "x[:,1]": "JA/t", "y_phase": "code"}),
            "selection_history_index": file_record(selection_manifest, version="selection_history_iter000_034",
                                                    fields=["path", "sha256", "version", "row_count", "fields", "units"],
                                                    units={"row_count": "exact calls"}),
            "stop_metrics": file_record(stop_path, version="stop_metrics_iter000_034",
                                        fields=["iteration", "metrics", "thresholds", "conditions"],
                                        units={"iteration": "index", "metrics": "raw metric units"}),
            "hard_risk": file_record(HARD_RISK, version="hard_risk_boundary_impact_audit_v2",
                                     fields=risk.columns.tolist(), units={"kBT": "kBT/t", "JA": "JA/t"}),
        },
        "canonical_outputs": {p.name: {"sha256": sha256(p), "rows": len(pd.read_csv(p))}
                              for p in (final_path, cold_path, allocation_path, conv_path, risk_path, phase_counts_path)},
        "invariants": {"total": len(final), "phase_counts": expected, "cold_start_selected": len(cold),
                       "cold_start_exact_labels": int(cold.stage1_exact_match.sum()),
                       "cold_start_in_endpoint": int(cold.endpoint_match.sum()), "iterations": 35,
                       "selection_calls_total": len(allocation), "hard_risk": len(risk),
                       "hard_risk_retained_in_endpoint": int(final.hard_risk.sum())},
    }
    inventory_path = HERE / "fig2_data_manifest.yaml"
    inventory_path.write_text(yaml.safe_dump(inventory, sort_keys=False), encoding="utf-8")
    return {"inventory": inventory, "paths": {"endpoint": endpoint, "seed": seed_path, "stage1_endpoint": stage1_endpoint,
            "selection_manifest": selection_manifest, "stop": stop_path, "risk": HARD_RISK},
            "canonical": {"final": final_path, "cold": cold_path, "allocation": allocation_path,
                          "convergence": conv_path, "risk": risk_path}}


def panel(panel_id: str, role: str, priority: str, panel_type: str, sources: list[dict],
          input_rows: int, *, phase_mapping: dict | None = None, expected_counts: dict | None = None,
          convergence: dict | None = None) -> dict[str, Any]:
    return {
        "panel_id": panel_id, "claim_ids": ["CLM-2D-THERMO"], "scientific_role": role,
        "visual_priority": priority, "panel_type": panel_type, "source_artifacts": sources,
        "source_versions": [x["version"] for x in sources], "source_hashes": [x["sha256"] for x in sources],
        "source_scripts": ["prepare_inputs.py", "build_fig2.py"],
        "coordinate_system": "Cartesian", "axis_fields": ["kT", "JA"],
        "axis_units": {"kT": "kBT/t", "JA": "JA/t"},
        "axis_orientation": {"x": "increasing_right", "y": "increasing_up"},
        "parameter_ranges": {"kT": [0.0, 0.56], "JA": [0.0, 2.12]},
        "input_row_count": input_rows, "selected_row_count": input_rows, "excluded_row_count": 0,
        "selection_filters": [], "excluded_data": [], "missing_data_policy": "preserve_and_mask",
        "label_fields": ["phase_code"], "label_semantics": "frozen exact BdG phase code",
        "phase_mapping": phase_mapping or {}, "topology_mapping": {}, "trust_mask": "explicit source fields",
        "transformations": transform_defaults(), "uncertainty_layer": "separate",
        "hard_risk_layer": "separate_open_rings_not_phase_identity", "convergence_status": convergence or {},
        "sensitivity_notes": [], "expected_counts": expected_counts or {}, "expected_reference_values": {},
        "numerical_tolerances": {}, "scientific_invariants": ["exact labels immutable", "kT >= 0"],
        "allowed_claims": ["audited thermodynamic endpoint", "active learning allocates exact calls"],
        "required_caveats": ["hard-risk remains an uncertainty layer"],
        "forbidden_claims": ["all numerical uncertainty eliminated", "ML replaces the BdG solver"],
        "forbidden_operations": ["interpolation", "smoothing", "boundary re-extraction", "manual label change"],
        "style_tokens": "al_fflo_manuscript_v1", "legend_semantics": {
            "registered_phase_encoding": {"normal": "neutral.secondary", "uniform_SC": "physical.highlight", "FFLO": "physical.dark"},
            "rendered_phase_encoding": {"normal": "neutral.secondary", "uniform_SC": "physical.highlight", "FFLO": "physical.dark"}},
        "exact_vs_derived_encoding": {"visually_distinct": True}, "primary_visual_object": panel_id,
        "validators": ["strict_data_checker", "fig2_scientific_validator"],
        "human_review_items": ["panel supports claim without overstatement"], "output_artifacts": []}


def build_contract(prepared: dict[str, Any]) -> None:
    inv = prepared["inventory"]; sources = inv["sources"]
    counts = inv["invariants"]["phase_counts"]
    panels = [
        panel("cold_start", "sparse random-grid exact cold start", "tertiary", "EXACT_SAMPLE_DATA",
              [sources["cold_start"], sources["cold_start_labels"], sources["endpoint"]], 512),
        panel("exact_call_allocation", "iteration-resolved allocation of exact BdG calls", "secondary", "EXACT_SAMPLE_DATA",
              [sources["selection_history_index"]], inv["invariants"]["selection_calls_total"]),
        panel("final_phase_map", "exact-sample thermodynamic phase reconstruction", "primary", "EXACT_SAMPLE_DATA",
              [sources["endpoint"], sources["hard_risk"]], 7434,
              phase_mapping={"normal": 0, "uniform_SC": 1, "FFLO": 2}, expected_counts=counts),
        panel("raw_convergence", "raw configured thermodynamic convergence evidence", "secondary", "DERIVED_NUMERICAL_DATA",
              [sources["stop_metrics"]], 35, convergence={"main_metric_representation": "raw_physical_metrics",
              "raw_trace_visible": True, "iteration_index": [0, 34]}),
    ]
    panels[-1]["transformations"]["derivation_formula"] = {
        "enabled": True, "authorized": True, "formula": "direct field extraction from archived stop_metrics_history.json",
        "implementation": "prepare_inputs.py", "parameters": {}, "validator": "endpoint_exact_match"}
    contract = {
        "schema_version": 3, "figure_id": "FIG-02-THERMO-STRICT", "figure_type": "manuscript",
        "template": "multi_panel_template",
        "scientific_message": "Starting from sparse exact observations, active learning reallocates exact BdG calls and reaches an audited 2D Normal/uniform-SC/FFLO endpoint while preserving the hard-risk layer.",
        "visual_message": "The final exact-sample thermodynamic map dominates; allocation and raw convergence support it, while the cold start remains contextual.",
        "primary_object": {"id": "final_phase_map", "asset_type": "phase_map", "scientific_role": "audited thermodynamic endpoint", "visual_role": "primary"},
        "primary_visual_object": {"id": "final_phase_map", "scientific_role": "exact-sample thermodynamic phase map", "expected_rank": 1},
        "secondary_objects": [
            {"id": "exact_call_allocation", "asset_type": "active_learning_trajectory", "scientific_role": "exact-call allocation", "visual_role": "secondary"},
            {"id": "raw_convergence", "asset_type": "convergence_plot", "scientific_role": "raw convergence audit", "visual_role": "secondary"}],
        "tertiary_information": [{"id": "cold_start", "scientific_role": "initial exact design", "visual_role": "tertiary"}],
        "secondary_visual_objects": ["exact_call_allocation", "raw_convergence"], "tertiary_visual_objects": ["cold_start"],
        "reading_order": ["final_phase_map", "exact_call_allocation", "raw_convergence", "cold_start"],
        "visual_narrative": {"enabled": True, "narrative_regions": {
            "final_phase_map": ["final_phase_map"], "exact_call_allocation": ["exact_call_allocation"],
            "raw_convergence": ["raw_convergence"], "cold_start": ["cold_start"]},
            "scientific_asset_fill_ratio_audits": []},
        "saliency_expectation": {"method": "blurred_region_visual_mass", "expected_ranking": ["final_phase_map", "exact_call_allocation", "raw_convergence", "cold_start"], "severity": "warning"},
        "forbidden_visual_priorities": ["equal_weight_dashboard", "hard_risk_as_fourth_phase", "cold_start_as_primary"],
        "data_sources": [str(p) for p in prepared["paths"].values()],
        "assets": [
            {"id": "final_phase_map", "type": "phase_map", "source": "canonical/fig2_final_exact_samples.csv", "schematic": False, "bbox": "final_phase_map", "features": ["consistent_phase_colors", "common_axes_for_comparison", "boundary_semantics"]},
            {"id": "exact_call_allocation", "type": "active_learning_trajectory", "source": "canonical/fig2_exact_call_allocation.csv", "schematic": False, "bbox": "exact_call_allocation", "features": ["iteration_or_budget", "sampled_points", "common_domain"]},
            {"id": "raw_convergence", "type": "convergence_plot", "source": "canonical/fig2_raw_convergence_metrics.csv", "schematic": False, "bbox": "raw_convergence", "features": ["raw_metric", "units_or_definition", "iteration_or_budget_axis"]},
            {"id": "cold_start", "type": "exact_sample_data", "source": "canonical/fig2_cold_start_exact_calls.csv", "schematic": False, "bbox": "cold_start"}],
        "layout": {"canvas": {"viewbox": [0, 0, 1600, 1100], "print_mm": [178.0, 120.0]},
                   "template_profile": "phase_map", "primary_area_fraction_min": 0.42,
                   "bounding_boxes": "geometry_manifest.json", "alignment_anchors": ["shared_2d_domain", "shared_iteration_axis"],
                   "declared_overlays": ["hard_risk_open_rings_over_exact_phase_samples"], "declared_exceptions": [],
                   "area_requirements": [{"id": "hero_map_area", "object_id": "final_phase_map", "reference": "canvas", "minimum_fraction": 0.42}]},
        "color_semantics": {"palette": "al_fflo_manuscript_v1", "mappings": {"phase": "physical", "exact_calls": "selected", "hard_risk": "uncertainty"}, "allowed_extra_hex": [],
                            "contrast_pairs": [{"id": "body_on_white", "role": "body_text", "foreground": "neutral.body", "background": "neutral.white"}]},
        "text_budget": {"maximum_words_in_artwork": 105, "maximum_words_per_panel": 26, "minimum_font_size_pt": 7.0, "move_excess_to": "caption_or_supplement"},
        "required_elements": ["sparse_cold_start", "iteration_resolved_exact_calls", "three_phase_exact_endpoint", "separate_hard_risk_layer", "raw_convergence_metrics"],
        "required_labels": ["Sparse cold start", "Exact-call allocation", "Final audited thermodynamic endpoint", "Raw convergence evidence", "Normal", "Uniform SC", "FFLO", "hard-risk"],
        "forbidden_elements": ["interpolated_phase_field", "smoothed_boundary", "hard_risk_as_phase", "default_matplotlib_palette"],
        "forbidden_claims": ["ML solves the BdG phase diagram", "all numerical uncertainty is eliminated", "complete phase discovery"],
        "validation_requirements": {"hard_fail": True, "require_geometry_manifest": True, "require_editable_svg": True,
            "require_pdf": True, "require_png_preview": True, "require_grayscale_review": True, "require_engine_preflight": True,
            "engine_preflight_record": "engine_preflight.json", "strict_palette": True},
        "provenance": {"physics_recomputed": False, "parameters_recorded": True, "commit_hash_when_available": True},
        "approval": {"gate": "GATE_FIG2_HUMAN_SCIENCE_REVIEW", "state": "HUMAN_WAIT"},
        "strict_data": {"enabled": True, "track": "STRICT_DATA_FIGURE_TRACK", "claim_freeze": ["CLM-2D-THERMO"],
            "panels": panels, "output_artifacts": ["fig2_strict_data.svg", "fig2_strict_data.pdf", "fig2_strict_data.png", "geometry_manifest.json", "asset_manifest.json", "fig2_scientific_validation.md", "fig2_visual_narrative_audit.md", "fig2_export_validation.md"]}}
    (HERE / "fig2_panel_contract.yaml").write_text(yaml.safe_dump(contract, sort_keys=False, width=120), encoding="utf-8")


def write_evidence_readiness(prepared: dict[str, Any]) -> None:
    inv = prepared["inventory"]
    text = f"""# Figure 2 evidence readiness

## Decision

**READY for `GATE_FIG2_HUMAN_SCIENCE_REVIEW`.** The review package is reconstructable from frozen artifacts without new physics calculations.

## Available evidence

- Frozen Stage II endpoint: `{inv['sources']['endpoint']['path']}` ({inv['invariants']['total']} exact rows).
- True archived cold start: {inv['invariants']['cold_start_selected']} selected calls; {inv['invariants']['cold_start_exact_labels']} have frozen Stage I exact labels and the same {inv['invariants']['cold_start_in_endpoint']} coordinates remain in the final Stage II endpoint.
- Complete selection history: iterations 0-34, {inv['invariants']['selection_calls_total']} calls including the cold start.
- Raw stop metrics: 35 archived iteration records.
- Hard-risk manifest: {inv['invariants']['hard_risk']} archived exact calls, kept as a separate numerical layer; {inv['invariants']['hard_risk_retained_in_endpoint']} coordinate matches remain in the final trusted endpoint.

## Independently reproduced invariants

- Normal: {inv['invariants']['phase_counts']['normal']}
- Uniform SC: {inv['invariants']['phase_counts']['uniform_SC']}
- FFLO: {inv['invariants']['phase_counts']['FFLO']}
- Total: {inv['invariants']['total']}

## Evidence limitation carried into the artwork

The exported archive does not contain a final dense monitor-prediction field or the coordinates of both final thermodynamic boundaries. Archived raster maps and boundary hashes/counts are provenance evidence but cannot support a new exact vector boundary reconstruction. Therefore panel c uses exact samples only and does not infer, interpolate, smooth, or fit boundary curves. Phase adjacency remains visible, but no newly reconstructed smooth boundary is claimed.

The six cold-start selections without a retained Stage I exact result remain explicitly unmatched; they are also absent from the final trusted Stage II endpoint and are not assigned inferred phase labels.
"""
    (HERE / "fig2_evidence_readiness.md").write_text(text, encoding="utf-8")


def main() -> int:
    prepared = prepare_stage2()
    build_contract(prepared)
    write_evidence_readiness(prepared)
    print(json.dumps({"status": "prepared", "total": prepared["inventory"]["invariants"]["total"],
                      "phase_counts": prepared["inventory"]["invariants"]["phase_counts"],
                      "cold_start": prepared["inventory"]["invariants"]["cold_start_selected"],
                      "iterations": prepared["inventory"]["invariants"]["iterations"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
