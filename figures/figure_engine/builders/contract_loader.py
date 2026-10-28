#!/usr/bin/env python3
"""Canonical figure-contract loader and schema validator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .rule_loader import RuleBundle, load_rule_bundle, resolve_template


REQUIRED_FIELDS = {
    "schema_version",
    "figure_id",
    "figure_type",
    "template",
    "scientific_message",
    "primary_object",
    "secondary_objects",
    "tertiary_information",
    "data_sources",
    "assets",
    "layout",
    "color_semantics",
    "text_budget",
    "required_elements",
    "required_labels",
    "forbidden_elements",
    "forbidden_claims",
    "validation_requirements",
    "provenance",
    "approval",
}

SUPPORTED_SCHEMA_VERSIONS = {1, 2, 3}
VISUAL_NARRATIVE_FIELDS = {
    "visual_message",
    "primary_visual_object",
    "reading_order",
    "secondary_visual_objects",
    "tertiary_visual_objects",
    "saliency_expectation",
    "forbidden_visual_priorities",
}


class ContractError(ValueError):
    """Raised when a figure contract is missing required scientific metadata."""


@dataclass(frozen=True)
class FigureContract:
    path: Path
    data: dict[str, Any]
    rules: RuleBundle

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    @property
    def figure_id(self) -> str:
        return str(self.data["figure_id"])

    @property
    def visual_narrative_enabled(self) -> bool:
        return visual_narrative_enabled(self.data)


def _read(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"Figure contract not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ContractError(f"Invalid figure contract YAML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError(f"Figure contract must be a mapping: {path}")
    return data


def normalize_legacy_contract(data: dict[str, Any]) -> dict[str, Any]:
    """Return a non-mutating migration view for older Fig.1-Fig.4 contracts."""
    normalized = dict(data)
    normalized.setdefault("schema_version", 1)
    normalized.setdefault("figure_type", "manuscript")
    normalized.setdefault("template", "multi_panel_template")
    normalized.setdefault("data_sources", normalized.get("source_datasets", []))
    normalized.setdefault("forbidden_elements", normalized.get("forbidden", []))
    normalized.setdefault("forbidden_claims", normalized.get("forbidden", []))
    normalized.setdefault("required_elements", normalized.get("allowed", []))
    normalized.setdefault("required_labels", [])
    normalized.setdefault("secondary_objects", [])
    normalized.setdefault("tertiary_information", [])
    panels = normalized.get("panels", {})
    if "primary_object" not in normalized:
        first = next(iter(panels), "primary") if isinstance(panels, dict) else "primary"
        normalized["primary_object"] = {
            "id": first,
            "asset_type": "legacy_panel",
            "scientific_role": str(panels.get(first, first)) if isinstance(panels, dict) else str(first),
            "visual_role": "primary",
        }
    normalized.setdefault("assets", [])
    normalized.setdefault("layout", {"template_profile": "multi_panel", "bounding_boxes": {}})
    normalized.setdefault("color_semantics", {"palette": "al_fflo_manuscript_v1", "mappings": {}})
    normalized["color_semantics"] = dict(normalized["color_semantics"])
    normalized["color_semantics"].setdefault("contrast_pairs", [
        {"id": "legacy_body_on_white", "role": "body_text",
         "foreground": "neutral.body", "background": "neutral.white"}
    ])
    normalized.setdefault("text_budget", {"minimum_font_size_pt": 7.0})
    normalized.setdefault("validation_requirements", {"hard_fail": True})
    normalized.setdefault("approval", {"gate": normalized.get("gate", "HUMAN_GATE"), "state": "HUMAN_WAIT"})
    return normalized


def visual_narrative_enabled(data: dict[str, Any]) -> bool:
    """Return whether v2 narrative fields are mandatory for this contract."""
    if data.get("schema_version") in {2, 3}:
        return True
    narrative = data.get("visual_narrative", {})
    return isinstance(narrative, dict) and narrative.get("enabled") is True


def strict_data_enabled(data: dict[str, Any]) -> bool:
    strict = data.get("strict_data", {})
    return data.get("schema_version") == 3 or (isinstance(strict, dict) and strict.get("enabled") is True)


def validate_strict_data_contract(data: dict[str, Any]) -> list[str]:
    if not strict_data_enabled(data):
        return []
    errors: list[str] = []
    strict = data.get("strict_data")
    if not isinstance(strict, dict) or strict.get("enabled") is not True:
        return ["schema-v3 contract must define strict_data.enabled=true"]
    if strict.get("track") != "STRICT_DATA_FIGURE_TRACK":
        errors.append("strict_data.track must be STRICT_DATA_FIGURE_TRACK")
    if not strict.get("claim_freeze"):
        errors.append("strict_data.claim_freeze is required")
    panels = strict.get("panels")
    if not isinstance(panels, list) or not panels:
        errors.append("strict_data.panels must be a non-empty list")
        return errors
    from .panel_router import validate_panel_contract
    ids = []
    for panel in panels:
        if not isinstance(panel, dict):
            errors.append("every strict-data panel must be a mapping")
            continue
        ids.append(panel.get("panel_id"))
        violations = validate_panel_contract(panel)
        if violations:
            errors.append(f"strict-data panel {panel.get('panel_id')!r}: {violations}")
    if len(ids) != len(set(ids)):
        errors.append("strict-data panel ids must be unique")
    if not isinstance(strict.get("output_artifacts"), list) or not strict.get("output_artifacts"):
        errors.append("strict_data.output_artifacts must be a non-empty list")
    return errors


def validate_visual_narrative_contract(data: dict[str, Any]) -> list[str]:
    """Validate schema-v2/opt-in visual-narrative declarations."""
    if not visual_narrative_enabled(data):
        return []
    errors: list[str] = []
    missing = sorted(VISUAL_NARRATIVE_FIELDS - set(data))
    if missing:
        errors.append(f"visual narrative is enabled but fields are missing: {missing}")
        return errors
    if not str(data.get("visual_message", "")).strip() or data.get("visual_message") == "REQUIRED":
        errors.append("visual_message is empty")

    narrative = data.get("visual_narrative")
    if not isinstance(narrative, dict):
        errors.append("visual_narrative must be a mapping")
        narrative = {}
    elif data.get("schema_version") == 1 and narrative.get("enabled") is not True:
        errors.append("schema-v1 visual_narrative must set enabled=true")

    primary = data.get("primary_visual_object")
    primary_id = None
    if not isinstance(primary, dict) or not {"id", "scientific_role", "expected_rank"}.issubset(primary):
        errors.append("primary_visual_object must define id, scientific_role, and expected_rank")
    else:
        primary_id = primary.get("id")
        if not str(primary_id or "").strip():
            errors.append("primary_visual_object.id must be non-empty")
        if primary.get("expected_rank") != 1:
            errors.append("primary_visual_object.expected_rank must be 1")

    reading = data.get("reading_order")
    if not isinstance(reading, list) or len(reading) < 2 or not all(isinstance(value, str) and value for value in reading):
        errors.append("reading_order must contain at least two non-empty string ids")
        reading = []
    else:
        if len(reading) != len(set(reading)):
            errors.append("reading_order ids must be unique")
        if primary_id and reading[0] != primary_id:
            errors.append("reading_order must begin with primary_visual_object.id")

    for key in ("secondary_visual_objects", "tertiary_visual_objects", "forbidden_visual_priorities"):
        value = data.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            errors.append(f"{key} must be a list of non-empty strings")

    secondary = data.get("secondary_visual_objects", [])
    tertiary = data.get("tertiary_visual_objects", [])
    if isinstance(secondary, list) and isinstance(tertiary, list):
        overlap = sorted(set(secondary) & set(tertiary))
        if overlap:
            errors.append(f"secondary and tertiary visual objects overlap: {overlap}")
        if primary_id in set(secondary) | set(tertiary):
            errors.append("primary visual object may not also be secondary or tertiary")

    regions = narrative.get("narrative_regions", {}) if isinstance(narrative, dict) else {}
    if not isinstance(regions, dict):
        errors.append("visual_narrative.narrative_regions must be a mapping")
        regions = {}
    else:
        for region_id, members in regions.items():
            if not isinstance(region_id, str) or not region_id or not isinstance(members, list) or not members \
                    or not all(isinstance(item, str) and item for item in members):
                errors.append("each narrative region must map a non-empty id to a non-empty object-id list")
    resolvable_regions = set(regions)
    if primary_id:
        resolvable_regions.add(primary_id)
    unresolved_reading = sorted(set(reading) - resolvable_regions)
    if unresolved_reading:
        errors.append(f"reading_order ids missing from narrative_regions: {unresolved_reading}")
    narrative_role_ids = set(data.get("secondary_visual_objects", [])) | set(data.get("tertiary_visual_objects", []))
    unresolved_roles = sorted(narrative_role_ids - resolvable_regions)
    if unresolved_roles:
        errors.append(f"secondary/tertiary visual objects missing from narrative_regions: {unresolved_roles}")

    area_requirements = data.get("layout", {}).get("area_requirements", []) if isinstance(data.get("layout"), dict) else []
    primary_area_requirements = [item for item in area_requirements if isinstance(item, dict) and item.get("object_id") == primary_id]
    if not primary_area_requirements:
        errors.append("primary_visual_object must have an explicit layout.area_requirements entry")

    saliency = data.get("saliency_expectation")
    if not isinstance(saliency, dict) or not {"method", "expected_ranking", "severity"}.issubset(saliency):
        errors.append("saliency_expectation must define method, expected_ranking, and severity")
    else:
        ranking = saliency.get("expected_ranking")
        if not isinstance(ranking, list) or not ranking or not all(isinstance(item, str) and item for item in ranking) \
                or ranking[0] != primary_id:
            errors.append("saliency_expectation.expected_ranking must begin with the primary visual object")
        elif len(ranking) != len(set(ranking)):
            errors.append("saliency_expectation.expected_ranking ids must be unique")
        elif set(ranking) - resolvable_regions:
            errors.append(f"saliency expected ids missing from narrative_regions: {sorted(set(ranking) - resolvable_regions)}")
        if saliency.get("severity") != "warning":
            errors.append("saliency_expectation.severity must be warning")

    audits = narrative.get("scientific_asset_fill_ratio_audits", []) if isinstance(narrative, dict) else []
    if not isinstance(audits, list):
        errors.append("visual_narrative.scientific_asset_fill_ratio_audits must be a list")
    else:
        for audit in audits:
            required = {"id", "object_id", "content_bbox_field", "minimum", "severity"}
            if not isinstance(audit, dict) or not required.issubset(audit):
                errors.append("each fill-ratio audit must define id, object_id, content_bbox_field, minimum, and severity")
                continue
            try:
                minimum = float(audit["minimum"])
            except (TypeError, ValueError):
                minimum = -1
            if not 0 < minimum <= 1:
                errors.append(f"fill-ratio audit {audit.get('id')!r} minimum must be in (0,1]")
            if audit.get("severity") != "hard_fail":
                errors.append(f"fill-ratio audit {audit.get('id')!r} severity must be hard_fail")
    return errors


def validate_contract(data: dict[str, Any], rules: RuleBundle) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        errors.append(f"missing required fields: {missing}")
    if data.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f"unsupported schema_version: {data.get('schema_version')!r}")
    if not str(data.get("scientific_message", "")).strip() or data.get("scientific_message") == "REQUIRED":
        errors.append("scientific_message is empty")
    primary = data.get("primary_object")
    if not isinstance(primary, dict) or not {"id", "asset_type", "scientific_role", "visual_role"}.issubset(primary):
        errors.append("primary_object must define id, asset_type, scientific_role, and visual_role")
    elif primary.get("visual_role") != "primary":
        errors.append("primary_object.visual_role must be 'primary'")
    if not isinstance(data.get("data_sources"), list):
        errors.append("data_sources must be a list")
    elif not data.get("data_sources"):
        errors.append("data_sources must declare at least one source or governing specification")
    if not isinstance(data.get("assets"), list):
        errors.append("assets must be a list")
    if not isinstance(data.get("forbidden_elements"), list):
        errors.append("forbidden_elements must be a list")
    if not isinstance(data.get("forbidden_claims"), list):
        errors.append("forbidden_claims must be a list")
    if not isinstance(data.get("required_elements"), list) or not isinstance(data.get("required_labels"), list):
        errors.append("required_elements and required_labels must be lists")

    role_objects = []
    for key, expected_role in (("secondary_objects", "secondary"), ("tertiary_information", "tertiary")):
        values = data.get(key)
        if not isinstance(values, list):
            errors.append(f"{key} must be a list")
            continue
        for item in values:
            if not isinstance(item, dict) or not {"id", "scientific_role", "visual_role"}.issubset(item):
                errors.append(f"every {key} entry must define id, scientific_role, and visual_role")
            elif item.get("visual_role") != expected_role:
                errors.append(f"{key} entry {item.get('id')!r} must have visual_role={expected_role!r}")
            else:
                role_objects.append(item)

    assets = data.get("assets", []) if isinstance(data.get("assets"), list) else []
    asset_ids, duplicate_ids = [], []
    for asset in assets:
        if not isinstance(asset, dict) or not {"id", "type", "source", "schematic", "bbox"}.issubset(asset):
            errors.append("every asset must define id, type, source, schematic, and bbox")
            continue
        if not isinstance(asset.get("schematic"), bool):
            errors.append(f"asset {asset.get('id')!r} schematic must be boolean")
        asset_id = asset.get("id")
        if asset_id in asset_ids:
            duplicate_ids.append(asset_id)
        asset_ids.append(asset_id)
    if duplicate_ids:
        errors.append(f"duplicate asset ids: {sorted(set(duplicate_ids))}")
    declared_role_ids = {primary.get("id")} if isinstance(primary, dict) else set()
    declared_role_ids.update(item.get("id") for item in data.get("secondary_objects", []) if isinstance(item, dict))
    missing_assets = sorted(value for value in declared_role_ids if value not in asset_ids)
    if missing_assets:
        errors.append(f"visual-role objects missing asset declarations: {missing_assets}")

    layout = data.get("layout")
    if not isinstance(layout, dict):
        errors.append("layout must be a mapping")
    else:
        for key in ("canvas", "template_profile", "bounding_boxes", "alignment_anchors"):
            if key not in layout:
                errors.append(f"layout must define {key}")
        if not isinstance(layout.get("alignment_anchors"), list) or not layout.get("alignment_anchors"):
            errors.append("layout.alignment_anchors must be a non-empty list")
    palette = data.get("color_semantics", {}).get("palette")
    if palette != rules["palette"].get("palette_id"):
        errors.append(f"color_semantics.palette must be {rules['palette'].get('palette_id')!r}")
    contrast_pairs = data.get("color_semantics", {}).get("contrast_pairs")
    if not isinstance(contrast_pairs, list) or not contrast_pairs:
        errors.append("color_semantics.contrast_pairs must declare at least one audited pair")
    try:
        resolve_template(str(data.get("template", "")))
    except Exception as exc:
        errors.append(str(exc))
    approval = data.get("approval", {})
    if not isinstance(approval, dict) or not approval.get("gate") or not approval.get("state"):
        errors.append("approval must define gate and state")
    minimum = data.get("text_budget", {}).get("minimum_font_size_pt", 0)
    global_min = rules["typography"]["minimum_print_size_pt"]["journal_width_178mm"]["annotation"]
    if data.get("figure_type") == "manuscript" and float(minimum or 0) < float(global_min):
        errors.append(f"text_budget.minimum_font_size_pt cannot be below {global_min}")
    provenance = data.get("provenance", {})
    if not isinstance(provenance, dict) or "physics_recomputed" not in provenance or "parameters_recorded" not in provenance:
        errors.append("provenance must declare physics_recomputed and parameters_recorded")
    errors.extend(validate_visual_narrative_contract(data))
    errors.extend(validate_strict_data_contract(data))
    return errors


def load_contract(path: str | Path, *, allow_legacy: bool = False, rules: RuleBundle | None = None) -> FigureContract:
    contract_path = Path(path).expanduser().resolve()
    raw = _read(contract_path)
    data = normalize_legacy_contract(raw) if allow_legacy else raw
    bundle = rules or load_rule_bundle()
    errors = validate_contract(data, bundle)
    if errors:
        raise ContractError(f"Invalid contract {contract_path}: " + "; ".join(errors))
    return FigureContract(path=contract_path, data=data, rules=bundle)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("contract")
    parser.add_argument("--allow-legacy", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.contract, allow_legacy=args.allow_legacy)
    print(f"valid contract {contract.figure_id}: {contract.path}")
