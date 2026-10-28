#!/usr/bin/env python3
"""Validate compact thermodynamic-landscape geometry and its control fixtures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
CONTRACT = yaml.safe_load((HERE / "landscape_contract.yaml").read_text(encoding="utf-8"))
RULES = CONTRACT["thermodynamic_landscape_geometry"]


def overlaps(first: dict, second: dict) -> bool:
    return not (
        first["x"] + first["width"] <= second["x"]
        or second["x"] + second["width"] <= first["x"]
        or first["y"] + first["height"] <= second["y"]
        or second["y"] + second["height"] <= first["y"]
    )


def validate(metrics: dict) -> dict:
    aspect = float(metrics["projected_surface_aspect_ratio"])
    height_width = float(metrics["projected_surface_height_width_ratio"])
    fill = float(metrics["foreground_fill_fraction"])
    whitespace = float(metrics["internal_whitespace_fraction"])
    triangle = float(metrics["dominant_minima_projected_triangle_fraction"])
    relief = float(metrics["vertical_relief_fraction"])
    labels = metrics.get("labels", [])
    label_pairs = [
        (labels[index]["id"], labels[other]["id"])
        for index in range(len(labels))
        for other in range(index + 1, len(labels))
        if overlaps(labels[index]["bbox"], labels[other]["bbox"])
    ]
    longest_leader = max((float(label.get("leader_length_px", math.inf)) for label in labels), default=math.inf)
    local_labels = all(float(label.get("leader_length_px", math.inf)) <= 45 for label in labels)
    axis_visible = (
        float(metrics.get("delta_axis_projected_length_fraction", 0)) >= 0.18
        and float(metrics.get("q_axis_projected_length_fraction", 0)) >= 0.18
        and float(metrics.get("projected_axis_angle_deg", 0)) >= 25
    )
    hard = {
        "projected_surface_aspect_ratio": aspect <= RULES["projected_surface_aspect_ratio"]["hard_maximum"],
        "projected_surface_height_width_ratio": height_width >= RULES["projected_surface_height_width_ratio"]["hard_minimum"],
        "foreground_fill_fraction": fill >= RULES["foreground_fill_fraction"]["hard_minimum"],
        "internal_whitespace_fraction": whitespace <= RULES["internal_whitespace_fraction"]["hard_maximum"],
        "dominant_minima_projected_triangle_fraction": triangle >= RULES["dominant_minima_projected_triangle_fraction"]["hard_minimum"],
        "vertical_relief_fraction": relief >= RULES["vertical_relief_fraction"]["hard_minimum"],
        "phase_labels_do_not_overlap": not label_pairs,
        "phase_labels_remain_local": local_labels,
        "leader_hard_maximum": longest_leader <= 45,
        "no_nonuniform_transform": not bool(metrics.get("nonuniform_transform", True)),
        "both_parameter_directions_visible": axis_visible,
        "prohibited_text_absent": not metrics.get("prohibited_text_found", []),
    }
    preferred = {
        "aspect_in_preferred_range": RULES["projected_surface_aspect_ratio"]["preferred_minimum"] <= aspect <= RULES["projected_surface_aspect_ratio"]["preferred_maximum"],
        "height_width_preferred": height_width >= RULES["projected_surface_height_width_ratio"]["preferred_minimum"],
        "fill_preferred": fill >= RULES["foreground_fill_fraction"]["preferred_minimum"],
        "whitespace_preferred": whitespace <= RULES["internal_whitespace_fraction"]["preferred_maximum"],
        "vertical_relief_preferred": relief >= RULES["vertical_relief_fraction"]["preferred_minimum"],
    }
    hard_failures = [name for name, passed in hard.items() if not passed]
    warnings = [name for name, passed in preferred.items() if not passed]
    return {
        "candidate_id": metrics.get("candidate_id"),
        "status": "pass" if not hard_failures else "fail",
        "hard_checks": hard,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "label_overlap_pairs": label_pairs,
        "longest_leader_px": longest_leader,
        "human_review_required": True,
    }


def self_test() -> dict:
    negative = json.loads((HERE / "fixtures" / "elongated_negative.json").read_text(encoding="utf-8"))
    control = json.loads((HERE / "fixtures" / "compact_control.json").read_text(encoding="utf-8"))
    negative_result = validate(negative)
    control_result = validate(control)
    expected_negative = {
        "projected_surface_aspect_ratio",
        "projected_surface_height_width_ratio",
        "foreground_fill_fraction",
        "internal_whitespace_fraction",
        "dominant_minima_projected_triangle_fraction",
        "vertical_relief_fraction",
        "phase_labels_do_not_overlap",
        "no_nonuniform_transform",
        "both_parameter_directions_visible",
        "prohibited_text_absent",
    }
    detected = set(negative_result["hard_failures"])
    passed = expected_negative.issubset(detected) and control_result["status"] == "pass"
    return {
        "status": "pass" if passed else "fail",
        "negative_status": negative_result["status"],
        "negative_expected_failures": sorted(expected_negative),
        "negative_detected_failures": sorted(detected),
        "control_status": control_result["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.metrics:
        result = validate(json.loads(args.metrics.read_text(encoding="utf-8")))
    else:
        parser.error("provide a metrics JSON or --self-test")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
