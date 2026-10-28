#!/usr/bin/env python3
"""Validate that visual roles are backed by declared scientific assets."""

from __future__ import annotations

from .common import Check, check


def validate(contract: dict, rules: dict) -> list[Check]:
    assets = contract.get("assets", [])
    asset_ids = {asset.get("id") for asset in assets if isinstance(asset, dict)}
    required_ids = {contract.get("primary_object", {}).get("id")}
    required_ids.update(item.get("id") for item in contract.get("secondary_objects", []))
    required_ids.discard(None)
    missing = sorted(required_ids - asset_ids)

    malformed = []
    for asset in assets:
        missing_fields = sorted({"id", "type", "source", "schematic", "bbox"} - set(asset))
        if missing_fields:
            malformed.append([asset.get("id"), missing_fields])
        if asset.get("schematic") is False and not str(asset.get("source", "")).strip():
            malformed.append([asset.get("id"), ["data_source"]])

    quantitative_schematic = []
    quantitative_roles = {"phase_map", "topology_map", "convergence_plot", "active_learning_trajectory",
                          "boundary_extraction", "slice_atlas", "high_dimensional_visualization"}
    for asset in assets:
        if asset.get("type") in quantitative_roles and asset.get("schematic") is True:
            quantitative_schematic.append(asset.get("id"))

    feature_issues = []
    asset_rules = rules["asset"]
    for asset in assets:
        rule = asset_rules.get(str(asset.get("type")), {})
        required = set(rule.get("required", [])) if isinstance(rule, dict) else set()
        declared = set(asset.get("features", []))
        missing_features = sorted(required - declared)
        if missing_features:
            feature_issues.append([asset.get("id"), asset.get("type"), missing_features])

    return [
        check("asset", "declared_asset_semantics", not missing and not malformed,
              "Every visual-role object has a well-formed asset declaration.",
              evidence={"missing_role_assets": missing, "malformed": malformed}),
        check("asset", "schematic_not_quantitative_evidence", not quantitative_schematic,
              "Quantitative scientific asset types are not supplied by schematic generators.",
              evidence=quantitative_schematic),
        check("asset", "required_scientific_features", not feature_issues,
              "Every typed scientific asset declares the features required by asset_rules.yaml.",
              evidence=feature_issues),
    ]
