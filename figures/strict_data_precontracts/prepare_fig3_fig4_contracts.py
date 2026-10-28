#!/usr/bin/env python3
"""Build Fig.3/Fig.4 evidence manifests and pre-render contracts only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from figures.figure_engine.builders.figure_build import preflight


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE3 = ROOT / "manuscript_langgraph/recovered_artifacts/stageIII"
STAGE4 = ROOT / "manuscript_langgraph/recovered_artifacts/stageIV"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path, version: str, units: dict[str, str]) -> dict[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            rows, fields = int(data["x"].shape[0]), list(data.files)
    elif path.suffix == ".csv":
        table = pd.read_csv(path)
        rows, fields = len(table), table.columns.tolist()
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = len(value) if isinstance(value, list) else 1
        fields = list(value) if isinstance(value, dict) else []
    return {"path": str(path.resolve()), "sha256": sha256(path), "version": version,
            "row_count": rows, "fields": fields, "units": units}


def transforms() -> dict[str, dict[str, Any]]:
    return {name: {"enabled": False, "authorized": False}
            for name in ("interpolation", "smoothing", "boundary_extraction", "aggregation", "normalization")}


def panel(panel_id: str, role: str, priority: str, panel_type: str, sources: list[dict[str, Any]],
          axes: list[str], units: dict[str, str], input_rows: int, selected: int, excluded: int,
          *, phase_mapping: dict | None = None, topology_mapping: dict | None = None,
          counts: dict | None = None, caveats: list[str] | None = None) -> dict[str, Any]:
    is_3d = len(axes) == 3
    item = {
        "panel_id": panel_id, "claim_ids": ["CLM-3D-EXTENSION" if is_3d else "CLM-2D-TOPOLOGY"],
        "scientific_role": role, "visual_priority": priority, "panel_type": panel_type,
        "source_artifacts": sources, "source_versions": [x["version"] for x in sources],
        "source_hashes": [x["sha256"] for x in sources],
        "source_scripts": ["prepare_fig3_fig4_contracts.py", "future figure-specific builder"],
        "coordinate_system": "Cartesian", "axis_fields": axes, "axis_units": units,
        "axis_orientation": {name: "increasing" for name in axes},
        "parameter_ranges": {}, "input_row_count": input_rows, "selected_row_count": selected,
        "excluded_row_count": excluded, "selection_filters": [], "excluded_data": [],
        "missing_data_policy": "preserve_and_mask", "label_fields": [],
        "label_semantics": "frozen exact/audit fields", "phase_mapping": phase_mapping or {},
        "topology_mapping": topology_mapping or {}, "trust_mask": "topology_trusted == 1 and y_phase == 2",
        "transformations": transforms(), "uncertainty_layer": "separate",
        "hard_risk_layer": "separate_not_phase_identity", "convergence_status": {},
        "sensitivity_notes": [], "expected_counts": counts or {}, "expected_reference_values": {},
        "numerical_tolerances": {}, "scientific_invariants": ["frozen labels are immutable"],
        "allowed_claims": ["higher-dimensional exploration" if is_3d else "Decision A under the configured audit"],
        "required_caveats": caveats or [],
        "forbidden_claims": (["complete 3D phase diagram", "fully converged 3D topology atlas"] if is_3d
                             else ["all possible topology refinements exhausted", "raw Delaunay edges are the boundary"]),
        "forbidden_operations": ["manual label change", "unauthorized interpolation", "unauthorized smoothing"],
        "style_tokens": "al_fflo_manuscript_v1", "legend_semantics": {
            "registered_phase_encoding": {"normal": "neutral.secondary", "uniform_SC": "physical.highlight", "FFLO": "physical.dark"},
            "rendered_phase_encoding": {"normal": "neutral.secondary", "uniform_SC": "physical.highlight", "FFLO": "physical.dark"}},
        "exact_vs_derived_encoding": {"visually_distinct": True}, "primary_visual_object": panel_id,
        "validators": ["strict_data_checker", "future_figure_specific_validator"],
        "human_review_items": ["claim wording", "trust-mask visibility", "caveat visibility"],
        "output_artifacts": [],
    }
    if topology_mapping:
        item["label_fields"] = ["topology_z2", "topology_bulk_gap", "topology_trusted"]
        item["sensitivity_notes"] = (["k=8 audit is the configured main result", "k=6 coverage p95=0.006269 exceeds 0.00625"]
                                     if not is_3d else ["no k=6 sensitivity closure is available for Stage IV; strict 3D topology convergence remains open"])
    elif phase_mapping:
        item["label_fields"] = ["y_phase"]
    return item


def base_contract(figure_id: str, message: str, primary: str, secondary: list[str], tertiary: list[str],
                  data_sources: list[str], panels: list[dict[str, Any]], claim: str, gate: str,
                  gate_state: str, is_3d: bool) -> dict[str, Any]:
    objects = [primary] + secondary + tertiary
    return {
        "schema_version": 3, "figure_id": figure_id, "figure_type": "manuscript",
        "template": "multi_panel_template", "scientific_message": message,
        "visual_message": f"{primary} is primary; supporting panels expose provenance, sensitivity, and convergence status.",
        "primary_object": {"id": primary, "asset_type": "phase_map", "scientific_role": message, "visual_role": "primary"},
        "primary_visual_object": {"id": primary, "scientific_role": message, "expected_rank": 1},
        "secondary_objects": [{"id": x, "asset_type": "data_panel", "scientific_role": x, "visual_role": "secondary"} for x in secondary],
        "tertiary_information": [{"id": x, "scientific_role": x, "visual_role": "tertiary"} for x in tertiary],
        "secondary_visual_objects": secondary, "tertiary_visual_objects": tertiary,
        "reading_order": objects, "visual_narrative": {"enabled": True,
            "narrative_regions": {x: [x] for x in objects}, "scientific_asset_fill_ratio_audits": []},
        "saliency_expectation": {"method": "blurred_region_visual_mass", "expected_ranking": objects, "severity": "warning"},
        "forbidden_visual_priorities": ["equal_weight_dashboard", "caveat_hidden_in_caption"],
        "data_sources": data_sources,
        "assets": [{"id": x, "type": "phase_map" if x == primary else "exact_sample_data",
                    "source": panels[i]["source_artifacts"][0]["path"], "schematic": False, "bbox": x,
                    "features": (["consistent_phase_colors", "common_axes_for_comparison", "boundary_semantics"] if x == primary else [])}
                   for i, x in enumerate(objects)],
        "layout": {"canvas": {"viewbox": [0, 0, 1600, 1100], "print_mm": [178.0, 120.0]},
                   "template_profile": "phase_map", "primary_area_fraction_min": 0.42,
                   "bounding_boxes": "future_geometry_manifest.json", "alignment_anchors": ["shared_parameter_axes"],
                   "declared_overlays": [], "declared_exceptions": [], "area_requirements": [
                       {"id": "primary_area", "object_id": primary, "reference": "canvas", "minimum_fraction": 0.42}]},
        "color_semantics": {"palette": "al_fflo_manuscript_v1", "mappings": {
            "phase": "physical", "topology_cfflo": "topology_cfflo", "topology_transition": "topology_transition",
            "topology_tfflo": "topology_tfflo"}, "allowed_extra_hex": [], "contrast_pairs": [
                {"id": "body_on_white", "role": "body_text", "foreground": "neutral.body", "background": "neutral.white"}]},
        "text_budget": {"maximum_words_in_artwork": 110, "maximum_words_per_panel": 28,
                        "minimum_font_size_pt": 7.0, "move_excess_to": "caption_or_supplement"},
        "required_elements": objects, "required_labels": [],
        "forbidden_elements": ["unauthorized_interpolation", "hidden_nonconvergence"],
        "forbidden_claims": (["complete 3D phase diagram", "fully converged 3D topology atlas"] if is_3d
                             else ["all refinements exhausted", "gap closing as stable phase"]),
        "validation_requirements": {"hard_fail": True, "require_geometry_manifest": True,
            "require_editable_svg": True, "require_pdf": True, "require_png_preview": True,
            "require_grayscale_review": True, "require_engine_preflight": True,
            "engine_preflight_record": "future_engine_preflight.json", "strict_palette": True},
        "provenance": {"physics_recomputed": False, "parameters_recorded": True, "commit_hash_when_available": True},
        "approval": {"gate": gate, "state": gate_state},
        "strict_data": {"enabled": True, "track": "STRICT_DATA_FIGURE_TRACK", "claim_freeze": [claim],
                        "panels": panels, "output_artifacts": [f"future_{figure_id.lower()}.svg", f"future_{figure_id.lower()}.pdf"]},
    }


def build_fig3() -> tuple[dict, dict]:
    endpoint = record(STAGE3 / "dataset_iter018.npz", "dataset_iter018",
                      {"x[:,0]": "kBT/t", "x[:,1]": "JA/t", "topology_z2": "Z2 code", "topology_bulk_gap": "t"})
    audit = STAGE3 / "figure_data/topology_convergence_audit_full_loop_v2"
    boundary = record(audit / "topology_boundary_points_by_iteration.csv", "topology_boundary_points_v2",
                      {"kT": "kBT/t", "JA": "JA/t", "x_norm": "normalized", "y_norm": "normalized"})
    map_change = record(audit / "topology_convergence_metrics_by_iteration.csv", "topology_map_change_v2", {"topology_map_change": "fraction"})
    shift = record(audit / "topology_boundary_shift_by_iteration.csv", "topology_boundary_shift_v2", {"symmetric_p95": "normalized distance"})
    coverage = record(audit / "topology_boundary_coverage.csv", "topology_boundary_coverage_v2", {"coverage_p95": "normalized distance"})
    surprise = record(audit / "topology_surprise_by_iteration.csv", "topology_surprise_v2", {"trusted_topology_surprise": "fraction"})
    sensitivity = record(audit / "topology_sensitivity_summary.csv", "topology_sensitivity_v2", {"final_coverage_p95": "normalized distance"})
    with np.load(Path(endpoint["path"]), allow_pickle=False) as d:
        mask = (d["y_phase"] == 2) & (d["topology_trusted"] == 1)
        counts = {"cFFLO": int(((d["topology_z2"] == 0) & mask).sum()),
                  "tFFLO": int(((d["topology_z2"] == 1) & mask).sum())}
    topo = {"cFFLO": 0, "tFFLO": 1, "gap_closing": "diagnostic_only_not_stable_phase"}
    panels = [
        panel("topology_exact_map", "trusted gapped FFLO topology-resolved exact samples", "primary", "EXACT_SAMPLE_DATA",
              [endpoint], ["kT", "JA"], {"kT": "kBT/t", "JA": "JA/t"}, 4345, 3213, 1132,
              topology_mapping=topo, counts=counts,
              caveats=["k=6 coverage p95=0.006269 slightly exceeds threshold", "gap closing is a diagnostic transition, not a stable phase"]),
        panel("audited_boundary_geometry", "archived k=8 topology-boundary geometry with support audit", "secondary", "DERIVED_BOUNDARY_GEOMETRY",
              [boundary], ["kT", "JA"], {"kT": "kBT/t", "JA": "JA/t"}, 11200, 646, 10554,
              topology_mapping=topo, caveats=["raw brackets are support diagnostics only"]),
        panel("raw_topology_convergence", "raw map-change, shift, coverage, and surprise traces", "secondary", "DERIVED_NUMERICAL_DATA",
              [map_change, shift, coverage, surprise], ["iteration", "metric"], {"iteration": "index", "metric": "native units"}, 73, 73, 0),
        panel("k_sensitivity", "k=6/8/12 topology-boundary sensitivity", "tertiary", "DERIVED_NUMERICAL_DATA",
              [sensitivity], ["k", "metric"], {"k": "neighbors", "metric": "native units"}, 3, 3, 0,
              topology_mapping=topo, caveats=["k=6 coverage p95=0.006269 slightly exceeds threshold"]),
    ]
    panels[1]["transformations"]["boundary_extraction"] = {"enabled": True, "authorized": True,
        "algorithm": "archived deterministic IDW contour extraction", "grid": "401x401",
        "connectivity": "archived component connectivity", "component_rule": "archived v2 component filter",
        "support_definition": "support radius 0.045; two-sided/bracket audit", "validator": "archived support and sensitivity tables"}
    for p in panels[2:]:
        p["transformations"]["derivation_formula"] = {"enabled": True, "authorized": True,
            "formula": "direct selection of archived audit fields", "implementation": "future builder",
            "parameters": {}, "validator": "source-row and endpoint exact match"}
    panels[2]["convergence_status"] = {"main_metric_representation": "raw_physical_metrics", "raw_trace_visible": True,
        "configured_decision": "Decision A", "k8_coverage_p95": 0.006185, "threshold": 0.00625}
    manifest = {"schema_version": 1, "figure_id": "FIG-03-STAGEIII-TOPOLOGY-STRICT", "physics_recomputed": False,
                "render_authorized": False, "sources": {x["version"]: x for x in [endpoint, boundary, map_change, shift, coverage, surprise, sensitivity]},
                "invariants": {"total": 4345, "trusted_FFLO": 3213, **counts, "k8_coverage_p95": 0.006185,
                               "k6_coverage_p95": 0.006269, "decision": "A"}}
    contract = base_contract("FIG-03-STAGEIII-TOPOLOGY-STRICT",
        "Topology-aware exact sampling resolves cFFLO/tFFLO structure and its configured convergence audit.",
        "topology_exact_map", ["audited_boundary_geometry", "raw_topology_convergence"], ["k_sensitivity"],
        [x["path"] for x in manifest["sources"].values()], panels, "CLM-2D-TOPOLOGY",
        "GATE_FIG3_RENDER_AUTHORIZATION", "BLOCKED_BY_FIG2_HUMAN_GATE", False)
    return manifest, contract


def build_fig4() -> tuple[dict, dict]:
    endpoint = record(STAGE4 / "dataset_iter025.npz", "dataset_iter025",
                      {"x[:,0]": "kBT/t", "x[:,1]": "JA/t", "x[:,2]": "mu/t", "topology_z2": "Z2 code"})
    return_tables = STAGE4 / "stageiv_3d_return_report_local/tables"
    key = record(return_tables / "return_report_key_metrics.csv", "stageiv_return_key_metrics", {"value": "mixed native units"})
    slice_metrics = record(STAGE4 / "figure_data/stageiv_mu_slice_boundary_audit_local/stageiv_narrow_mu_slice_metrics.csv",
                           "stageiv_narrow_mu_slice_metrics", {"mu_center": "mu/t", "slice_half_width": "mu/t"})
    failure = record(STAGE4 / "stageiv_3d_audit_local/tables/stageiv_3d_convergence_failure_decomposition.csv",
                     "stageiv_convergence_failure_decomposition", {"value": "native units", "threshold": "native units"})
    surface = record(return_tables / "boundary_surface_diagnostics.csv", "stageiv_diagnostic_surface_registry",
                     {"edge_threshold_normalized": "normalized distance"})
    topo = {"cFFLO": 0, "tFFLO": 1, "gap_closing": "diagnostic_only_not_stable_phase"}
    caveat = ["the 3D topology surface is not formally closed by the strict convergence criteria used for the 2D topology result"]
    panels = [
        panel("exact_3d_thermodynamic_cloud", "3D extension exact-sample thermodynamic observations", "primary", "EXACT_SAMPLE_DATA",
              [endpoint], ["kT", "JA", "mu"], {"kT": "kBT/t", "JA": "JA/t", "mu": "mu/t"}, 7081, 7081, 0,
              phase_mapping={"normal": 0, "uniform_SC": 1, "FFLO": 2},
              counts={"normal": 2350, "uniform_SC": 141, "FFLO": 4590}, caveats=caveat),
        panel("exact_3d_topology_support", "trusted 3D cFFLO/tFFLO exact support", "secondary", "EXACT_SAMPLE_DATA",
              [endpoint], ["kT", "JA", "mu"], {"kT": "kBT/t", "JA": "JA/t", "mu": "mu/t"}, 7081, 4731, 2350,
              topology_mapping=topo, counts={"cFFLO": 3265, "tFFLO": 1466}, caveats=caveat),
        panel("fixed_mu_slice_audit", "archived support-restricted fixed-mu slice audit", "secondary", "DERIVED_NUMERICAL_DATA",
              [slice_metrics, endpoint], ["kT", "JA", "mu"], {"kT": "kBT/t", "JA": "JA/t", "mu": "mu/t"}, 20, 20, 0,
              caveats=caveat + ["slice bins are explicit; no continuous surface is implied"]),
        panel("strict_3d_nonclosure", "raw failed strict 3D convergence gates", "tertiary", "DERIVED_NUMERICAL_DATA",
              [failure, key], ["iteration", "metric", "mu"], {"iteration": "index", "metric": "native units", "mu": "mu/t"}, 33, 33, 0,
              caveats=caveat),
    ]
    for p in panels[2:]:
        p["transformations"]["derivation_formula"] = {"enabled": True, "authorized": True,
            "formula": "direct archived table field selection", "implementation": "future builder",
            "parameters": {}, "validator": "source-row and endpoint exact match"}
    panels[3]["convergence_status"] = {"main_metric_representation": "raw_physical_metrics", "raw_trace_visible": True,
        "configured_decision": "not_converged", "coverage_p95": 0.015317527016765012, "coverage_threshold": 0.00625}
    manifest = {"schema_version": 1, "figure_id": "FIG-04-STAGEIV-3D-STRICT", "physics_recomputed": False,
        "render_authorized": False, "continuous_surface_authorized": False,
        "sources": {x["version"]: x for x in [endpoint, key, slice_metrics, failure, surface]},
        "invariants": {"total": 7081, "phase_counts": {"normal": 2350, "uniform_SC": 141, "FFLO": 4590},
            "topology_counts": {"cFFLO": 3265, "tFFLO": 1466}, "strict_3d_convergence": "not_converged",
            "topology_surface_coverage_p95": 0.015317527016765012, "coverage_threshold": 0.00625,
            "diagnostic_surface_status": "blocked_from_main_artwork_without_explicit_authorization"}}
    contract = base_contract("FIG-04-STAGEIV-3D-STRICT",
        "Reliability-aware active learning extends to higher-dimensional exploration and reveals thermodynamic and topology-aware structures, without claiming strict 3D topology closure.",
        "exact_3d_thermodynamic_cloud", ["exact_3d_topology_support", "fixed_mu_slice_audit"], ["strict_3d_nonclosure"],
        [x["path"] for x in manifest["sources"].values()], panels, "CLM-3D-EXTENSION",
        "GATE_FIG4_RENDER_AUTHORIZATION", "BLOCKED_BY_FIG2_HUMAN_GATE", True)
    return manifest, contract


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    for number, builder in ((3, build_fig3), (4, build_fig4)):
        manifest, contract = builder()
        manifest_path = HERE / f"fig{number}_data_manifest.yaml"
        contract_path = HERE / f"fig{number}_panel_contract.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, width=120), encoding="utf-8")
        contract_path.write_text(yaml.safe_dump(contract, sort_keys=False, width=120), encoding="utf-8")
        preflight(contract_path)
    (HERE / "fig3_evidence_readiness.md").write_text("""# Figure 3 evidence readiness

**CONTRACT_READY; rendering is not authorized before Fig.2 human approval.**

The frozen endpoint contains 4345 rows. Applying the explicit trusted gapped FFLO mask gives 3213 points: 2290 cFFLO and 923 tFFLO. The archived k=8 audit records Decision A and coverage p95 0.006185 below the 0.00625 threshold. The k=6 sensitivity result is 0.006269 and must remain visible. Archived boundary geometry is derived evidence with its extraction parameters and support audit; local brackets remain diagnostics only. Gap closing must be shown as a transition diagnostic, not a stable third topology phase.

No Fig.3 SVG, PDF, or PNG was rendered in this stage.
""", encoding="utf-8")
    (HERE / "fig4_evidence_readiness.md").write_text("""# Figure 4 evidence readiness

**CONTRACT_READY_WITH_CAVEAT; rendering is not authorized before Fig.2 human approval.**

The frozen 3D endpoint contains 7081 exact observations: 2350 Normal, 141 uniform SC, and 4590 FFLO; topology fields contain 3265 cFFLO/trivial and 1466 tFFLO/topological points among 4731 applicable trusted samples. The fixed-mu audit provides 20 explicit support-restricted slices.

Strict 3D topology closure is not achieved: final coverage p95 is 0.0153175 versus 0.00625, surprise and component stability fail, and the chemical-potential domain is limited. Existing RBF/KNN surfaces are diagnostic only. The precontract therefore authorizes exact point clouds, explicit fixed-mu slices, and raw nonclosure metrics, but blocks a continuous 3D phase/topology surface unless a later human gate explicitly authorizes that diagnostic layer.

No Fig.4 SVG, PDF, or PNG was rendered in this stage.
""", encoding="utf-8")
    print(json.dumps({"status": "CONTRACT_READY", "fig3_rendered": False, "fig4_rendered": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
