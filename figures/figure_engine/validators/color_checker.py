#!/usr/bin/env python3
"""Validate extracted SVG colors against the project semantic palette."""

from __future__ import annotations

import re

from .common import Check, check


def _approved(rules: dict, contract: dict) -> set[str]:
    palette = rules["palette"]
    values = set()
    for family in palette["families"].values():
        values.update(str(value).upper() for value in family.values())
    values.update(str(value).upper() for value in palette["neutral"].values())
    for scale in palette.get("continuous", {}).values():
        values.update(str(value).upper() for value in scale)
    values.update(str(value).upper() for value in contract.get("color_semantics", {}).get("allowed_extra_hex", []))
    return values


def _resolve_color(value: str, palette: dict) -> str:
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return value.upper()
    parts = value.split(".")
    if len(parts) == 2 and parts[0] == "neutral":
        return str(palette["neutral"][parts[1]]).upper()
    if len(parts) == 2 and parts[0] in palette["families"]:
        return str(palette["families"][parts[0]][parts[1]]).upper()
    raise ValueError(f"unknown semantic color token {value!r}")


def _relative_luminance(value: str) -> float:
    rgb = [int(value[index:index + 2], 16) / 255.0 for index in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    first, second = _relative_luminance(foreground), _relative_luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def _contrast_check(contract: dict, rules: dict) -> Check:
    pairs = contract.get("color_semantics", {}).get("contrast_pairs", [])
    thresholds = rules["color"]["contrast"]
    threshold_by_role = {
        "body_text": float(thresholds["body_text_on_background_min_ratio"]),
        "large_text": float(thresholds["large_text_on_background_min_ratio"]),
        "key_line": float(thresholds["key_line_against_background_min_ratio"]),
    }
    evidence, failures = [], []
    for pair in pairs:
        try:
            role = pair["role"]
            foreground = _resolve_color(str(pair["foreground"]), rules["palette"])
            background = _resolve_color(str(pair["background"]), rules["palette"])
            threshold = threshold_by_role[role]
            ratio = _contrast_ratio(foreground, background)
            record = {"id": pair.get("id"), "role": role, "foreground": foreground,
                      "background": background, "ratio": round(ratio, 3), "minimum": threshold}
            evidence.append(record)
            if ratio < threshold:
                failures.append(record)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append({"pair": pair, "error": str(exc)})
    return check(
        "color", "minimum_contrast", bool(pairs) and not failures,
        f"Registered contrast pairs: {len(pairs)}; failures: {len(failures)}.",
        evidence={"pairs": evidence, "failures": failures},
    )


def validate(svg_summary: dict, contract: dict, rules: dict) -> list[Check]:
    colors = set(svg_summary.get("hex_colors", []))
    approved = _approved(rules, contract)
    unknown = sorted(colors - approved)
    mappings = contract.get("color_semantics", {}).get("mappings", {})
    required_families = set(mappings.values())
    missing_families = sorted(name for name in required_families if name not in rules["palette"]["families"])
    strict = bool(contract.get("validation_requirements", {}).get("strict_palette", False))
    return [
        check("color", "semantic_mappings", not missing_families, "Every contract color role maps to a canonical semantic family.", evidence=missing_families),
        check("color", "approved_palette", not unknown or not strict, f"Unknown SVG colors: {len(unknown)}; strict palette={strict}.", severity="hard_fail" if strict else "warning", evidence=unknown),
        _contrast_check(contract, rules),
    ]
