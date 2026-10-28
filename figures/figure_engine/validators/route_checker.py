#!/usr/bin/env python3
"""Validate workflow-arrow and annotation-route geometry against engine rules."""

from __future__ import annotations

from ..builders.layout_engine import BBox
from .common import Check, check


def _inside(point, box: BBox, clearance: float = 0.0) -> bool:
    return (
        box.x - clearance <= point[0] <= box.right + clearance
        and box.y - clearance <= point[1] <= box.bottom + clearance
    )


def validate(manifest: dict, rules: dict) -> list[Check]:
    objects = {obj["id"]: obj for obj in manifest.get("objects", [])}
    cards = {obj_id: BBox.from_mapping(obj["bbox"]) for obj_id, obj in objects.items()
             if obj.get("type") == "card" and obj.get("bbox")}
    main = [route for route in manifest.get("routes", []) if route.get("type") == "main_workflow"]
    profile = rules["arrow"]["profiles"]["main_workflow"]
    endpoint_issues, style_issues, crossing_issues = [], [], []

    for route in main:
        route_id = route.get("id")
        source, target = route.get("source"), route.get("target")
        if source not in objects or target not in objects or source == target:
            endpoint_issues.append([route_id, source, target])

        widths = route.get("widths_px", {})
        opacities = route.get("opacities", {})
        samples = route.get("samples", [])
        expected_widths = profile["widths_px"]
        expected_opacities = profile["opacities"]
        width_values = [widths.get("tail"), widths.get("mid"), widths.get("near_head")]
        opacity_values = [opacities.get("tail"), opacities.get("mid"), opacities.get("head")]
        monotonic_width = all(value is not None for value in width_values) and width_values[0] < width_values[1] < width_values[2]
        monotonic_opacity = all(value is not None for value in opacity_values) and opacity_values[0] < opacity_values[1] < opacity_values[2]
        token_match = widths == expected_widths and opacities == expected_opacities
        enough_samples = len(samples) >= int(profile["minimum_samples"])
        if not (monotonic_width and monotonic_opacity and token_match and enough_samples):
            style_issues.append({
                "id": route_id, "widths": widths, "opacities": opacities,
                "samples": len(samples), "expected_widths": expected_widths,
                "expected_opacities": expected_opacities,
            })

        for sample in samples[1:-1]:
            point = sample.get("center")
            if not point:
                continue
            for card_id, box in cards.items():
                if card_id in {source, target}:
                    continue
                if _inside(point, box):
                    crossing_issues.append([route_id, card_id, point])
                    break
            if crossing_issues and crossing_issues[-1][0] == route_id:
                break

    annotation_lengths = []
    maximum = float(rules["arrow"]["profiles"]["annotation_leader"]["hard_maximum_length_px"])
    for route in manifest.get("routes", []):
        if route.get("type") in {"leader", "annotation_leader"} and float(route.get("length", 0)) > maximum:
            annotation_lengths.append([route.get("id"), route.get("length"), maximum])

    return [
        check("arrow", "workflow_source_target", not endpoint_issues,
              "Every main workflow route has distinct registered source and target roles.", evidence=endpoint_issues),
        check("arrow", "workflow_profile", not style_issues,
              "Main workflow routes inherit tapered width, opacity, and sampling tokens from arrow_rules.yaml.", evidence=style_issues),
        check("arrow", "workflow_card_crossing", not crossing_issues,
              "Main workflow route interiors do not pass through unrelated cards.", evidence=crossing_issues),
        check("arrow", "annotation_length", not annotation_lengths,
              f"Annotation leaders do not exceed {maximum:g} px.", evidence=annotation_lengths),
    ]
