#!/usr/bin/env python3
"""Load and validate the canonical figure-engine constitution and styles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ENGINE_ROOT = Path(__file__).resolve().parents[1]
CONSTITUTION = ENGINE_ROOT / "constitution"
STYLES = ENGINE_ROOT / "styles"
TEMPLATES = ENGINE_ROOT / "templates"

RULE_FILES = {
    "principles": CONSTITUTION / "scientific_figure_principles.yaml",
    "visual_narrative": CONSTITUTION / "visual_narrative_rules.yaml",
    "typography": CONSTITUTION / "typography_rules.yaml",
    "layout": CONSTITUTION / "layout_rules.yaml",
    "spacing": CONSTITUTION / "spacing_rules.yaml",
    "color": CONSTITUTION / "color_rules.yaml",
    "arrow": CONSTITUTION / "arrow_rules.yaml",
    "annotation": CONSTITUTION / "annotation_rules.yaml",
    "asset": CONSTITUTION / "asset_rules.yaml",
    "validation": CONSTITUTION / "validation_rules.yaml",
    "strict_data": CONSTITUTION / "strict_data_rules.yaml",
    "enforcement": CONSTITUTION / "enforcement_registry.yaml",
    "palette": STYLES / "palette.yaml",
    "fonts": STYLES / "fonts.yaml",
    "svg_style": STYLES / "svg_style.yaml",
}


class RuleError(ValueError):
    """Raised when canonical rules are absent, malformed, or contradictory."""


@dataclass(frozen=True)
class RuleBundle:
    root: Path
    rules: dict[str, dict[str, Any]]
    files: dict[str, Path]
    hashes: dict[str, str]

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self.rules[key]

    @property
    def schema_versions(self) -> dict[str, int]:
        return {name: int(rule["schema_version"]) for name, rule in self.rules.items()}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuleError(f"Missing canonical rule file: {path}") from exc
    except yaml.YAMLError as exc:
        raise RuleError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuleError(f"Rule file must contain a mapping: {path}")
    if value.get("schema_version") != 1:
        raise RuleError(f"Unsupported schema_version in {path}: {value.get('schema_version')!r}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_cross_file(bundle: dict[str, dict[str, Any]]) -> None:
    strict = bundle["strict_data"]
    if strict.get("authority") != "canonical" or strict.get("track_id") != "STRICT_DATA_FIGURE_TRACK":
        raise RuleError("Strict-data constitution must be canonical and active")
    if strict.get("hard_risk_is_phase") is not False:
        raise RuleError("Hard risk may not be registered as a thermodynamic phase")
    if strict.get("transformation_defaults", {}).get("interpolation", {}).get("authorized") is not False:
        raise RuleError("Strict-data interpolation must default to unauthorized")

    narrative = bundle["visual_narrative"]
    if narrative.get("authority") != "canonical" or narrative.get("integration", {}).get("status") != "active":
        raise RuleError("Visual-narrative constitution must be canonical and active")
    if narrative["saliency"]["severity"]["unexpected_saliency_rank"] != "warning":
        raise RuleError("Saliency-rank mismatch must remain warning-only")
    if narrative["saliency"]["severity"]["perceptual_first_fixation"] != "human_review":
        raise RuleError("First fixation must remain an explicit human-review judgment")

    required_levels = set(bundle["color"]["required_levels"])
    for family, values in bundle["palette"]["families"].items():
        missing = sorted(required_levels - set(values))
        if missing:
            raise RuleError(f"Palette family {family!r} is missing levels {missing}")

    arrow = bundle["arrow"]["profiles"]["main_workflow"]
    widths = arrow["widths_px"]
    if not (widths["tail"] < widths["mid"] < widths["near_head"]):
        raise RuleError("Main-workflow ribbon widths must increase toward the target")
    opacity = arrow["opacities"]
    if not (opacity["tail"] < opacity["mid"] < opacity["head"]):
        raise RuleError("Main-workflow ribbon opacity must increase toward the target")

    spacing = bundle["spacing"]
    if spacing["clearance_px"]["independent_cards_minimum"] < spacing["internal_padding_px"]["hard_minimum"]:
        raise RuleError("Independent-card clearance cannot be below internal hard padding")

    typography = bundle["typography"]["font_stack"]
    fonts = bundle["fonts"]
    sans_style = [fonts["sans"]["preferred"], *fonts["sans"]["fallback"]]
    if sans_style != typography["sans_preferred"]:
        raise RuleError("fonts.yaml sans stack conflicts with typography_rules.yaml")
    math_style = {fonts["math"]["preferred"], *fonts["math"]["fallback"]}
    if not set(typography["math_preferred"]).issubset(math_style):
        raise RuleError("fonts.yaml math stack conflicts with typography_rules.yaml")

    annotation = bundle["annotation"]
    if annotation["labels"]["minimum_label_clearance_px"] != spacing["clearance_px"]["label_to_label_minimum"]:
        raise RuleError("Annotation label clearance conflicts with spacing rules")
    if annotation["labels"]["text_to_leader_clearance_px"] != spacing["clearance_px"]["label_to_leader_minimum"]:
        raise RuleError("Annotation leader clearance conflicts with spacing rules")

    semantic_families = set(bundle["color"]["semantic_categories"].values())
    missing_semantic_families = sorted(semantic_families - set(bundle["palette"]["families"]))
    if missing_semantic_families:
        raise RuleError(f"Color semantics reference missing palette families: {missing_semantic_families}")

    schematic = _read_yaml(TEMPLATES / "schematic_template.yaml")
    if schematic["layout"]["primary_area_fraction_min"] != bundle["layout"]["template_profiles"]["schematic"]["primary_area_fraction_min"]:
        raise RuleError("Schematic template primary-area threshold conflicts with layout rules")

    enforcement = bundle["enforcement"]
    allowed_levels = set(enforcement["levels"])
    for name, entry in enforcement["rules"].items():
        if entry.get("level") not in allowed_levels or not str(entry.get("implementation", "")).strip():
            raise RuleError(f"Enforcement registry entry {name!r} is incomplete")


def load_rule_bundle() -> RuleBundle:
    """Load the complete canonical rule set. Figure builders must call this."""
    rules = {name: _read_yaml(path) for name, path in RULE_FILES.items()}
    _validate_cross_file(rules)
    return RuleBundle(
        root=ENGINE_ROOT,
        rules=rules,
        files=dict(RULE_FILES),
        hashes={name: _sha256(path) for name, path in RULE_FILES.items()},
    )


def resolve_template(name: str) -> Path:
    path = TEMPLATES / (name if name.endswith(".yaml") else f"{name}.yaml")
    if not path.is_file():
        raise RuleError(f"Unknown figure template: {name}")
    return path


if __name__ == "__main__":
    loaded = load_rule_bundle()
    print(f"loaded {len(loaded.rules)} canonical rule files from {loaded.root}")
