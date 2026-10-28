#!/usr/bin/env python3
"""Panel-type router and structural checks for strict-data figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PANEL_TYPES = {
    "EXACT_SAMPLE_DATA",
    "DERIVED_NUMERICAL_DATA",
    "DERIVED_BOUNDARY_GEOMETRY",
    "INTERPOLATED_FIELD",
    "SCHEMATIC_ANNOTATION",
    "HYBRID_DATA_SCHEMATIC",
}

BASE_GATES = [
    "GATE_DATA_SCOPE_FREEZE",
    "GATE_SOURCE_PROVENANCE",
    "GATE_DATA_SEMANTICS",
    "GATE_REPRODUCIBLE_RECONSTRUCTION",
    "GATE_SCIENTIFIC_INVARIANTS",
    "GATE_VISUAL_NARRATIVE",
    "GATE_STYLE_AND_LAYOUT",
    "GATE_EXPORT_INTEGRITY",
    "GATE_HUMAN_SCIENCE_REVIEW",
    "GATE_HUMAN_VISUAL_REVIEW",
    "GATE_FIGURE_FREEZE",
]

REQUIRED_PANEL_FIELDS = {
    "panel_id", "claim_ids", "scientific_role", "visual_priority", "panel_type",
    "source_artifacts", "source_versions", "source_hashes", "source_scripts",
    "coordinate_system", "axis_fields", "axis_units", "axis_orientation",
    "parameter_ranges", "input_row_count", "selection_filters", "excluded_data",
    "missing_data_policy", "label_fields", "label_semantics", "phase_mapping",
    "topology_mapping", "trust_mask", "transformations", "uncertainty_layer",
    "hard_risk_layer", "convergence_status", "sensitivity_notes", "expected_counts",
    "expected_reference_values", "numerical_tolerances", "scientific_invariants",
    "allowed_claims", "required_caveats", "forbidden_claims", "forbidden_operations",
    "style_tokens", "legend_semantics", "exact_vs_derived_encoding",
    "primary_visual_object", "validators", "human_review_items", "output_artifacts",
}


@dataclass(frozen=True)
class PanelRoute:
    panel_id: str
    panel_type: str
    gates: tuple[str, ...]


def route_panel(panel: dict[str, Any]) -> PanelRoute:
    panel_type = str(panel.get("panel_type", ""))
    if panel_type not in PANEL_TYPES:
        raise ValueError(f"unknown panel_type {panel_type!r}")
    gates = list(BASE_GATES)
    if panel_type in {"DERIVED_NUMERICAL_DATA", "DERIVED_BOUNDARY_GEOMETRY", "INTERPOLATED_FIELD"}:
        gates.insert(3, "GATE_TRANSFORMATION_AUTHORIZATION")
    return PanelRoute(str(panel.get("panel_id", "")), panel_type, tuple(dict.fromkeys(gates)))


def validate_panel_contract(panel: dict[str, Any]) -> list[str]:
    """Return stable machine-readable violation codes."""
    errors: list[str] = []
    missing = sorted(REQUIRED_PANEL_FIELDS - set(panel))
    if missing:
        errors.append("missing_required_fields")
        return errors
    try:
        route_panel(panel)
    except ValueError:
        errors.append("unknown_panel_type")
        return errors

    if not panel.get("axis_units"):
        errors.append("missing_units")
    if not panel.get("source_versions") or not panel.get("source_hashes"):
        errors.append("missing_source_version_or_hash")
    if panel.get("input_row_count") is None:
        errors.append("missing_row_count")
    if panel.get("missing_data_policy") in {None, "", "convert_to_phase"}:
        errors.append("invalid_missing_data_policy")

    transforms = panel.get("transformations", {})
    for name in ("interpolation", "smoothing", "boundary_extraction", "aggregation", "normalization"):
        item = transforms.get(name, {"enabled": False, "authorized": False})
        if item.get("enabled") and not item.get("authorized"):
            errors.append(f"unauthorized_{name}")

    if panel["panel_type"] == "INTERPOLATED_FIELD":
        item = transforms.get("interpolation", {})
        if not (item.get("enabled") and item.get("authorized") and item.get("method") and item.get("validator")):
            errors.append("unauthorized_interpolation")
    if panel["panel_type"] == "DERIVED_BOUNDARY_GEOMETRY":
        item = transforms.get("boundary_extraction", {})
        required = ("algorithm", "grid", "connectivity", "component_rule", "support_definition", "validator")
        if not item.get("authorized") or any(not item.get(key) for key in required):
            errors.append("unauthorized_boundary_extraction")
    if panel["panel_type"] == "SCHEMATIC_ANNOTATION" and panel.get("quantitative_claim") is not False:
        errors.append("schematic_quantitative_claim")
    if panel["panel_type"] == "HYBRID_DATA_SCHEMATIC" and not panel.get("exact_vs_derived_encoding", {}).get("visually_distinct"):
        errors.append("exact_and_interpolated_layers_indistinguishable")

    mapping = panel.get("phase_mapping") or {}
    if mapping:
        if mapping != {"normal": 0, "uniform_SC": 1, "FFLO": 2}:
            errors.append("phase_mapping_invalid")
        if "hard_risk" in mapping:
            errors.append("hard_risk_as_fourth_phase")
        counts = panel.get("expected_counts", {})
        phase_counts = [counts.get(name) for name in ("normal", "uniform_SC", "FFLO")]
        if all(value is not None for value in phase_counts) and sum(map(int, phase_counts)) != int(panel.get("input_row_count", -1)):
            errors.append("wrong_phase_count")
        legend = panel.get("legend_semantics", {})
        registered = legend.get("registered_phase_encoding")
        rendered = legend.get("rendered_phase_encoding")
        if registered and rendered and registered != rendered:
            errors.append("phase_color_swap")

    topo = panel.get("topology_mapping") or {}
    if topo:
        if not panel.get("trust_mask"):
            errors.append("missing_trusted_topology_mask")
        if topo.get("cFFLO") != 0 or topo.get("tFFLO") != 1:
            errors.append("topology_mapping_invalid")
        if topo.get("gap_closing") in {0, 1, "stable_phase"}:
            errors.append("gap_closing_labeled_as_phase")
        if not any("k=6" in str(note) or "k6" in str(note) for note in panel.get("sensitivity_notes", [])):
            errors.append("missing_k6_sensitivity")

    if panel.get("convergence_status"):
        conv = panel["convergence_status"]
        if conv.get("main_metric_representation") == "normalized_gate_ratio_only":
            errors.append("normalized_gate_ratio_used_as_main_convergence")
        if transforms.get("smoothing", {}).get("enabled") and not conv.get("raw_trace_visible"):
            errors.append("unauthorized_smoothing")

    forbidden_text = " ".join(map(str, panel.get("forbidden_claims", []))).lower()
    allowed_text = " ".join(map(str, panel.get("allowed_claims", []))).lower()
    high_dimensional = len(panel.get("axis_fields", [])) >= 3 or "3d" in str(panel.get("scientific_role", "")).lower()
    if high_dimensional:
        for phrase in ("complete 3d phase diagram", "fully converged 3d topology surface", "complete 3d topology atlas"):
            if phrase in allowed_text:
                errors.append("3d_complete_atlas_claim")
        caveats = " ".join(map(str, panel.get("required_caveats", []))).lower()
        evidence_scope_tokens = ("finite", "discrete", "continuous surface", "atlas")
        if not any(token in caveats for token in evidence_scope_tokens):
            errors.append("3d_missing_evidence_scope_caveat")

    conv = panel.get("convergence_status") or {}
    if conv.get("mode") == "diagnostic_only":
        if conv.get("threshold_line_visible") is True:
            errors.append("unsupported_threshold_line_in_diagnostic_mode")
        if conv.get("pass_fail_labels_visible") is True:
            errors.append("unsupported_pass_fail_label_in_diagnostic_mode")
    return sorted(set(errors))
