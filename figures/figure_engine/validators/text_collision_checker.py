#!/usr/bin/env python3
"""Detect text/text and text/leader collisions from a geometry manifest."""

from __future__ import annotations

from math import hypot

from ..builders.layout_engine import BBox, pairwise
from .common import Check, check


def _point_segment_distance(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0 and dy == 0:
        return hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    q = (a[0] + t * dx, a[1] + t * dy)
    return hypot(p[0] - q[0], p[1] - q[1])


def _segment_box_distance(a, b, box: BBox) -> float:
    if _segment_intersects_box(a, b, box):
        return 0.0
    corners = [(box.x, box.y), (box.right, box.y), (box.right, box.bottom), (box.x, box.bottom)]
    return min(_point_segment_distance(corner, a, b) for corner in corners)


def _orientation(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p, tolerance: float = 1e-9) -> bool:
    return (
        min(a[0], b[0]) - tolerance <= p[0] <= max(a[0], b[0]) + tolerance
        and min(a[1], b[1]) - tolerance <= p[1] <= max(a[1], b[1]) + tolerance
        and abs(_orientation(a, b, p)) <= tolerance
    )


def _segments_intersect(a, b, c, d) -> bool:
    first = _orientation(a, b, c)
    second = _orientation(a, b, d)
    third = _orientation(c, d, a)
    fourth = _orientation(c, d, b)
    if first * second < 0 and third * fourth < 0:
        return True
    return any((
        _on_segment(a, b, c), _on_segment(a, b, d),
        _on_segment(c, d, a), _on_segment(c, d, b),
    ))


def _segment_intersects_box(a, b, box: BBox) -> bool:
    if box.contains(BBox(a[0], a[1], 0, 0)) or box.contains(BBox(b[0], b[1], 0, 0)):
        return True
    corners = [(box.x, box.y), (box.right, box.y), (box.right, box.bottom), (box.x, box.bottom)]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return any(_segments_intersect(a, b, start, end) for start, end in edges)


def validate(manifest: dict, *, label_clearance: float = 10.0, leader_clearance: float = 5.0,
             border_clearance: float = 4.0, text_plot_clearance: float = 6.0) -> list[Check]:
    objects = manifest.get("objects", [])
    indexed = {obj["id"]: obj for obj in objects}
    boxes = {obj["id"]: BBox.from_mapping(obj["bbox"]) for obj in objects if obj.get("bbox")}
    texts = [(obj["id"], BBox.from_mapping(obj["bbox"]), obj) for obj in objects if obj.get("type") == "text"]
    overlap_issues, clearance_issues = [], []
    for (left_id, left, left_obj), (right_id, right, right_obj) in pairwise(texts):
        allowed = set(left_obj.get("allow_overlap_with", [])) | set(right_obj.get("allow_overlap_with", []))
        if right_id in allowed or left_id in allowed:
            continue
        if left.overlaps(right):
            overlap_issues.append([left_id, right_id])
        elif left_obj.get("role") == "phase_label" and right_obj.get("role") == "phase_label" and left.gap(right) < label_clearance:
            clearance_issues.append([left_id, right_id, left.gap(right)])

    border_issues = []
    for text_id, box, obj in texts:
        parent_id = obj.get("parent")
        if parent_id not in boxes:
            continue
        parent = boxes[parent_id]
        clearances = [box.x - parent.x, box.y - parent.y, parent.right - box.right, parent.bottom - box.bottom]
        if min(clearances) < border_clearance:
            border_issues.append([text_id, parent_id, min(clearances)])

    plot_issues = []
    plots = [(obj["id"], boxes[obj["id"]]) for obj in objects if obj.get("type") == "plot" and obj["id"] in boxes]
    for text_id, text_box, text_obj in texts:
        allowed = set(text_obj.get("allow_overlap_with", []))
        for plot_id, plot_box in plots:
            if plot_id in allowed:
                continue
            if text_box.overlaps(plot_box) or text_box.gap(plot_box) < text_plot_clearance:
                plot_issues.append([text_id, plot_id, text_box.gap(plot_box)])

    leader_issues = []
    leaders = [route for route in manifest.get("routes", []) if route.get("type") in {"leader", "annotation_leader"}]
    for route in leaders:
        if route.get("type") not in {"leader", "annotation_leader"}:
            continue
        a, b = route.get("start"), route.get("end")
        if not a or not b:
            leader_issues.append([route.get("id"), "missing endpoints"])
            continue
        own = route.get("target_text")
        for text_id, box, _ in texts:
            if text_id == own:
                continue
            distance = _segment_box_distance(a, b, box)
            if distance < leader_clearance:
                leader_issues.append([route.get("id"), text_id, distance])

    leader_crossings = []
    for index, left in enumerate(leaders):
        for right in leaders[index + 1:]:
            a, b, c, d = left.get("start"), left.get("end"), right.get("start"), right.get("end")
            if not all((a, b, c, d)):
                continue
            shared_endpoint = any(hypot(p[0] - q[0], p[1] - q[1]) < 1e-6 for p in (a, b) for q in (c, d))
            if not shared_endpoint and _segments_intersect(a, b, c, d):
                leader_crossings.append([left.get("id"), right.get("id")])

    return [
        check("geometry", "text_text_overlap", not overlap_issues, "No undeclared text-text overlaps.", evidence=overlap_issues),
        check("geometry", "text_border_clearance", not border_issues, f"Registered text clears its parent border by at least {border_clearance:g} px.", evidence=border_issues),
        check("geometry", "text_plot_clearance", not plot_issues, f"Text clears unrelated plots by at least {text_plot_clearance:g} px unless an overlay is declared.", evidence=plot_issues),
        check("geometry", "phase_label_clearance", not clearance_issues, f"Phase-label clearance is at least {label_clearance:g} px.", evidence=clearance_issues),
        check("geometry", "leader_text_clearance", not leader_issues, f"Annotation leaders clear unrelated text by at least {leader_clearance:g} px.", evidence=leader_issues),
        check("geometry", "leader_leader_crossing", not leader_crossings, "Annotation leaders do not cross one another.", evidence=leader_crossings),
    ]
