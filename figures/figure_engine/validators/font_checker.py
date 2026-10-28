#!/usr/bin/env python3
"""Validate manifest typography against the canonical font constitution."""

from __future__ import annotations

import re

from .common import Check, check


def validate(manifest: dict, contract: dict, rules: dict) -> list[Check]:
    typography = rules["typography"]
    text = [obj for obj in manifest.get("objects", []) if obj.get("type") == "text"]
    minimum = float(contract["text_budget"]["minimum_font_size_pt"])
    below = [[obj.get("id"), obj.get("font_size_pt")] for obj in text if obj.get("font_size_pt") is not None and float(obj["font_size_pt"]) < minimum]
    condensed = []
    nonuniform = []
    skewed = []
    preferred_sans = [value.lower() for value in typography["font_stack"]["sans_preferred"]]
    unknown = []
    for obj in text:
        family = str(obj.get("font_family", ""))
        stretch = str(obj.get("font_stretch", "normal"))
        if re.search(r"condensed|narrow|compressed", family + " " + stretch, re.I):
            condensed.append([obj.get("id"), family, stretch])
        transform = obj.get("transform", {}) or {}
        sx, sy = float(transform.get("scale_x", 1)), float(transform.get("scale_y", 1))
        if abs(abs(sx) - abs(sy)) > 1e-3:
            nonuniform.append([obj.get("id"), sx, sy])
        if abs(float(transform.get("skew_x", 0))) > 1e-3 or abs(float(transform.get("skew_y", 0))) > 1e-3:
            skewed.append(obj.get("id"))
        if obj.get("role") != "math" and family and not any(name in family.lower() for name in preferred_sans):
            unknown.append([obj.get("id"), family])

    summary = manifest.get("font_audit_summary", {})
    if summary.get("minimum_font_size_pt") is not None and float(summary["minimum_font_size_pt"]) < minimum:
        below.append(["font_audit_summary", summary["minimum_font_size_pt"]])
    if summary.get("condensed_count", 0):
        condensed.append(["font_audit_summary", summary["condensed_count"]])
    if summary.get("nonuniform_scaling_count", 0):
        nonuniform.append(["font_audit_summary", summary["nonuniform_scaling_count"]])

    math_families = {str(obj.get("font_family", "")) for obj in text if obj.get("role") == "math"}
    math_families.update(summary.get("math_families", []))
    preferred_math = typography["font_stack"]["math_preferred"]
    math_preferred_used = not math_families or any(any(pref.lower() in family.lower() for pref in preferred_math) for family in math_families)
    fallback_used = bool(math_families) and not math_preferred_used

    return [
        check("typography", "minimum_font_size", not below, f"All registered text is at least {minimum:g} pt.", evidence=below),
        check("typography", "normal_width_fonts", not condensed, "No condensed/narrow/compressed fonts are used.", evidence=condensed),
        check("typography", "nonuniform_text_scaling", not nonuniform, "No text uses non-uniform scaling.", evidence=nonuniform),
        check("typography", "skewed_text", not skewed, "No text uses skew transforms.", evidence=skewed),
        check("typography", "preferred_math_font", math_preferred_used, "Preferred STIX math stack is used for new output; legacy fallbacks are reported.", severity="warning", evidence={"families": sorted(math_families), "fallback_used": fallback_used}),
        check("typography", "sans_font_stack", not unknown, "Non-math text uses the approved normal-width sans stack.", severity="warning", evidence=unknown),
    ]

