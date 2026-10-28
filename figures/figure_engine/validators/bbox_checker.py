#!/usr/bin/env python3
"""Validate containment, clipping, gutters, and declared bbox hierarchy."""

from __future__ import annotations

from ..builders.layout_engine import BBox, pairwise, visual_area_fraction
from .common import Check, check


def validate(manifest: dict, contract: dict, rules: dict) -> list[Check]:
    canvas = BBox.from_mapping(manifest["canvas"])
    objects = manifest.get("objects", [])
    indexed = {obj["id"]: obj for obj in objects}
    boxes = {obj["id"]: BBox.from_mapping(obj["bbox"]) for obj in objects if obj.get("bbox")}
    clip_tolerance = float(rules["layout"]["containment"]["declared_antialias_overhang_px"])
    clipped = [object_id for object_id, box in boxes.items() if not canvas.contains(box, clip_tolerance)]

    containment = []
    for obj in objects:
        parent = obj.get("parent")
        if not parent or parent in {"canvas", "root"} or obj.get("allow_parent_overflow"):
            continue
        if parent not in boxes:
            containment.append([obj["id"], parent, "missing parent bbox"])
        elif not boxes[parent].contains(boxes[obj["id"]], clip_tolerance):
            containment.append([obj["id"], parent, "outside parent"])

    cards = [(obj["id"], boxes[obj["id"]], obj) for obj in objects if obj.get("type") == "card" and obj.get("independent", True)]
    minimum_gap = float(rules["spacing"]["clearance_px"]["independent_cards_minimum"])
    gap_issues, observed_min = [], float("inf")
    for (left_id, left, left_obj), (right_id, right, right_obj) in pairwise(cards):
        declared = set(left_obj.get("allow_overlap_with", [])) | set(right_obj.get("allow_overlap_with", []))
        if right_id in declared or left_id in declared:
            continue
        gap = left.gap(right)
        observed_min = min(observed_min, gap)
        if left.overlaps(right) or gap < minimum_gap:
            gap_issues.append([left_id, right_id, gap])

    primary_ids = {contract["primary_object"]["id"]}
    primary_ids.update(obj.get("id") for obj in contract.get("secondary_objects", []) if obj.get("visual_role") == "primary")
    primary_boxes = [boxes[item] for item in primary_ids if item in boxes]
    role_boxes = [box for object_id, box in boxes.items() if indexed[object_id].get("visual_weight_object", indexed[object_id].get("type") == "card")]
    fraction = visual_area_fraction(primary_boxes, role_boxes) if role_boxes else 0.0
    profile = contract.get("layout", {}).get("template_profile", "multi_panel")
    template_threshold = rules["layout"]["template_profiles"].get(
        profile, rules["layout"]["template_profiles"]["multi_panel"]
    )["primary_area_fraction_min"]
    contract_threshold = contract.get("layout", {}).get("primary_area_fraction_min", template_threshold)
    threshold = max(float(template_threshold), float(contract_threshold))
    hierarchy_ok = bool(primary_boxes) and fraction >= threshold

    area_issues = []
    for requirement in contract.get("layout", {}).get("area_requirements", []):
        object_id = requirement.get("object_id")
        reference_id = requirement.get("reference", "canvas")
        minimum = float(requirement.get("minimum_fraction", 0))
        object_box = boxes.get(object_id)
        reference_box = canvas if reference_id == "canvas" else boxes.get(reference_id)
        actual = object_box.area / reference_box.area if object_box and reference_box and reference_box.area else 0.0
        if object_box is None or reference_box is None or actual < minimum:
            area_issues.append({"id": requirement.get("id"), "object_id": object_id,
                                "reference": reference_id, "actual": actual, "minimum": minimum})

    return [
        check("geometry", "canvas_clipping", not clipped, "All registered objects fit the canvas.", evidence=clipped),
        check("geometry", "parent_containment", not containment, "All registered child objects fit their parent boxes.", evidence=containment),
        check("geometry", "independent_card_gutters", not gap_issues, f"Independent cards retain at least {minimum_gap:g} px clearance.", evidence={"issues": gap_issues, "observed_min": observed_min}),
        check("layout", "primary_visual_area", hierarchy_ok,
              f"Primary registered area fraction {fraction:.4f}; required minimum {threshold:.4f}.",
              evidence={"primary_ids": sorted(primary_ids), "primary_registered": sorted(item for item in primary_ids if item in boxes)}),
        check("layout", "declared_area_requirements", not area_issues,
              "All contract-declared scientific subobjects meet their minimum area fractions.", evidence=area_issues),
    ]
