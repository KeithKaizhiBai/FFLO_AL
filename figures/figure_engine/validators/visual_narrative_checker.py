#!/usr/bin/env python3
"""Validate physical-first reading order with honest hard/warning/human boundaries."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from ..builders.contract_loader import visual_narrative_enabled
from ..builders.layout_engine import BBox
from .common import Check, check


def _area(box: dict[str, Any]) -> float:
    return float(box["width"]) * float(box["height"])


def _contained(outer: dict[str, Any], inner: dict[str, Any], tolerance: float = 1e-6) -> bool:
    return (
        float(inner["x"]) >= float(outer["x"]) - tolerance
        and float(inner["y"]) >= float(outer["y"]) - tolerance
        and float(inner["x"]) + float(inner["width"]) <= float(outer["x"]) + float(outer["width"]) + tolerance
        and float(inner["y"]) + float(inner["height"]) <= float(outer["y"]) + float(outer["height"]) + tolerance
    )


def _region_metrics(image_path: Path, manifest: dict, region_members: dict[str, list[str]], rules: dict) -> dict:
    image = Image.open(image_path).convert("RGB")
    gray = ImageOps.grayscale(image)
    sigma = float(rules["visual_narrative"]["saliency"]["preprocessing"]["gaussian_blur_sigma_fraction_of_canvas_width"]) * image.width
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=sigma))
    pixels = list(gray.crop((0, 0, image.width, min(4, image.height))).getdata())
    pixels += list(gray.crop((0, max(0, image.height - 4), image.width, image.height)).getdata())
    pixels += list(gray.crop((0, 0, min(4, image.width), image.height)).getdata())
    pixels += list(gray.crop((max(0, image.width - 4), 0, image.width, image.height)).getdata())
    background = float(statistics.median(pixels)) if pixels else 255.0

    canvas = manifest["canvas"]
    sx = image.width / float(canvas["width"])
    sy = image.height / float(canvas["height"])
    indexed = {item["id"]: item for item in manifest.get("objects", [])}
    metrics: dict[str, dict[str, float]] = {}
    for region_id, members in region_members.items():
        integrated = 0.0
        detail = 0.0
        sample_count = 0
        for object_id in members:
            obj = indexed.get(object_id)
            if not obj or not obj.get("bbox"):
                continue
            box = obj["bbox"]
            crop_box = (
                max(0, round(float(box["x"]) * sx)),
                max(0, round(float(box["y"]) * sy)),
                min(image.width, round((float(box["x"]) + float(box["width"])) * sx)),
                min(image.height, round((float(box["y"]) + float(box["height"])) * sy)),
            )
            raw_crop = gray.crop(crop_box)
            blurred_crop = blurred.crop(crop_box)
            raw_values = list(raw_crop.getdata())
            blur_values = list(blurred_crop.getdata())
            integrated += sum(abs(background - value) for value in blur_values)
            detail += sum(abs(raw - smooth) for raw, smooth in zip(raw_values, blur_values))
            sample_count += len(blur_values)
        if sample_count:
            metrics[region_id] = {
                "integrated_visual_mass": integrated,
                "mean_contrast": integrated / sample_count,
                "detail_density": detail / sample_count,
                "sample_count": sample_count,
            }
    return {
        "method": "Gaussian-blurred grayscale distance from median outer-border luminance",
        "gaussian_sigma_px": sigma,
        "background_luminance": background,
        "image_size_px": list(image.size),
        "regions": metrics,
        "ranking": sorted(metrics, key=lambda item: metrics[item]["integrated_visual_mass"], reverse=True),
    }


def validate(manifest: dict | None, contract: dict, rules: dict, png_path: Path | None = None) -> tuple[list[Check], dict]:
    if not visual_narrative_enabled(contract):
        return [], {"enabled": False}

    narrative = contract["visual_narrative"]
    primary_spec = contract["primary_visual_object"]
    primary_id = primary_spec["id"]
    manifest = manifest or {"canvas": {"x": 0, "y": 0, "width": 1, "height": 1}, "objects": []}
    indexed = {item["id"]: item for item in manifest.get("objects", [])}
    primary = indexed.get(primary_id)
    checks: list[Check] = []

    primary_is_scientific = bool(
        primary
        and primary.get("type") != "card"
        and (
            primary.get("scientific_subobject") is True
            or "scientific" in str(primary.get("role", ""))
            or primary.get("type") in {"plot", "map", "landscape", "landscape_3d", "topology", "surface", "slice_atlas"}
        )
    )
    checks.append(check(
        "visual_narrative", "narrative_primary_scientific_subobject", primary_is_scientific,
        "The primary visual object resolves to a registered scientific subobject, never its parent card.",
        evidence={"primary_visual_object": primary_id, "resolved_object": primary},
    ))

    area_requirements = [item for item in contract.get("layout", {}).get("area_requirements", [])
                         if item.get("object_id") == primary_id]
    area_evidence = []
    area_pass = bool(primary and primary.get("bbox") and area_requirements)
    canvas = BBox.from_mapping(manifest["canvas"])
    boxes = {item["id"]: BBox.from_mapping(item["bbox"]) for item in manifest.get("objects", []) if item.get("bbox")}
    for requirement in area_requirements:
        reference_id = requirement.get("reference", "canvas")
        reference = canvas if reference_id == "canvas" else boxes.get(reference_id)
        obj = boxes.get(primary_id)
        actual = obj.area / reference.area if obj and reference and reference.area else 0.0
        minimum = float(requirement.get("minimum_fraction", 0))
        passed = bool(obj and reference and actual >= minimum)
        area_pass = area_pass and passed
        area_evidence.append({"reference": reference_id, "actual": actual, "minimum": minimum, "passed": passed})
    checks.append(check(
        "visual_narrative", "narrative_primary_area", area_pass,
        "The primary scientific subobject satisfies its explicit contract area requirement.",
        evidence=area_evidence,
    ))

    region_members = {key: list(value) for key, value in narrative.get("narrative_regions", {}).items()}
    region_members.setdefault(primary_id, [primary_id])
    reading = contract["reading_order"]
    unresolved_regions = {
        region_id: [member for member in region_members.get(region_id, []) if member not in indexed]
        for region_id in reading
        if region_id not in region_members or any(member not in indexed for member in region_members.get(region_id, []))
    }
    reading_pass = bool(reading and reading[0] == primary_id and not unresolved_regions)
    checks.append(check(
        "visual_narrative", "narrative_reading_order", reading_pass,
        "The declared reading order begins with the primary scientific subobject and all narrative regions resolve.",
        evidence={"reading_order": reading, "region_members": region_members, "unresolved": unresolved_regions},
    ))

    fill_results = []
    for audit in narrative.get("scientific_asset_fill_ratio_audits", []):
        obj = indexed.get(audit["object_id"])
        object_box = obj.get("bbox") if obj else None
        content_box = obj.get(audit["content_bbox_field"]) if obj else None
        ratio = _area(content_box) / _area(object_box) if object_box and content_box and _area(object_box) else 0.0
        passed = bool(object_box and content_box and _contained(object_box, content_box) and ratio >= float(audit["minimum"]))
        fill_results.append({
            "id": audit["id"], "object_id": audit["object_id"], "ratio": ratio,
            "minimum": float(audit["minimum"]), "contained": bool(object_box and content_box and _contained(object_box, content_box)),
            "passed": passed,
        })
    checks.append(check(
        "visual_narrative", "scientific_asset_fill_ratio", all(item["passed"] for item in fill_results),
        "Declared scientific content fills the asset bbox above its figure-specific minimum.",
        evidence=fill_results,
    ))

    saliency = {}
    if png_path and png_path.is_file():
        saliency = _region_metrics(png_path, manifest, region_members, rules)
        expected = contract["saliency_expectation"]["expected_ranking"]
        observed = saliency["ranking"]
        rank_pass = bool(observed and observed[0] == expected[0])
        checks.append(check(
            "visual_narrative", "narrative_saliency_rank", rank_pass,
            "Blurred saliency is a warning-only proxy; it never certifies first fixation or aesthetic quality.",
            severity="warning", evidence={"expected": expected, "observed": observed},
        ))
        metrics = saliency["regions"]
        tertiary_values = [metrics[item]["mean_contrast"] for item in contract.get("tertiary_visual_objects", []) if item in metrics]
        primary_contrast = metrics.get(primary_id, {}).get("mean_contrast")
        contrast_pass = primary_contrast is not None and (not tertiary_values or primary_contrast >= statistics.median(tertiary_values))
        checks.append(check(
            "visual_narrative", "narrative_contrast_hierarchy", contrast_pass,
            "Primary mean contrast should not fall below the median declared tertiary-region contrast.",
            severity="warning", evidence={"primary": primary_contrast, "tertiary": tertiary_values},
        ))
        interval = rules["visual_narrative"]["saliency"]["equal_weight_proxy"]["comparable_mass_ratio_interval"]
        minimum_count = int(rules["visual_narrative"]["saliency"]["equal_weight_proxy"]["minimum_competing_regions_for_warning"])
        primary_mass = metrics.get(primary_id, {}).get("integrated_visual_mass", 0.0)
        comparable = []
        if primary_mass:
            comparable = [item for item, value in metrics.items()
                          if item != primary_id and float(interval[0]) <= value["integrated_visual_mass"] / primary_mass <= float(interval[1])]
        checks.append(check(
            "visual_narrative", "narrative_equal_weight_regions", len(comparable) < minimum_count,
            "Four or more regions with near-primary mass trigger an equal-weight warning.",
            severity="warning", evidence={"comparable_regions": comparable, "minimum_for_warning": minimum_count},
        ))
    else:
        checks.append(check(
            "visual_narrative", "narrative_saliency_evidence", False,
            "No final PNG was supplied; saliency remains unevaluated and cannot be reported as an automatic pass.",
            severity="warning",
        ))

    checks.append(check(
        "visual_narrative", "narrative_first_fixation_human_review", False,
        "Human review is required to decide whether actual first fixation matches the declared reading order.",
        severity="human_review", evidence={"expected_first_fixation": primary_id},
    ))
    hard_failures = [item for item in checks if item.severity == "hard_fail" and not item.passed]
    warnings = [item for item in checks if item.severity == "warning" and not item.passed]
    human = [item for item in checks if item.severity == "human_review"]
    payload = {
        "schema_version": 1,
        "enabled": True,
        "status": "pass" if not hard_failures else "fail",
        "hard_failure_count": len(hard_failures),
        "warning_count": len(warnings),
        "human_review_count": len(human),
        "primary_visual_object": primary_id,
        "reading_order": reading,
        "saliency": saliency,
        "checks": [item.as_dict() for item in checks],
        "automatic_aesthetic_approval": False,
    }
    return checks, payload


def write_report(payload: dict, output_dir: Path) -> None:
    if not payload.get("enabled"):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visual_narrative_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Visual narrative report", "",
        f"- Hard-rule status: **{payload['status'].upper()}**",
        f"- Hard failures: **{payload['hard_failure_count']}**",
        f"- Warnings: **{payload['warning_count']}**",
        f"- Human-review judgments: **{payload['human_review_count']}**",
        f"- Primary visual object: `{payload['primary_visual_object']}`",
        f"- Declared reading order: `{' -> '.join(payload['reading_order'])}`", "",
    ]
    ranking = payload.get("saliency", {}).get("ranking", [])
    if ranking:
        lines.extend(["## Warning-oriented saliency proxy", ""])
        lines.extend(f"{index}. `{name}`" for index, name in enumerate(ranking[:3], start=1))
        lines.append("")
    lines.extend(["| Result | Severity | Check | Detail |", "|---|---|---|---|"])
    for item in payload["checks"]:
        lines.append(f"| {'pass' if item['passed'] else 'review/fail'} | {item['severity']} | `{item['name']}` | {item['detail']} |")
    lines.extend([
        "", "Saliency and first fixation are not scientific criteria and are not automatic aesthetic approval.",
        "Automated validation does not pass a human proposal or figure gate.", "",
    ])
    (output_dir / "visual_narrative_report.md").write_text("\n".join(lines), encoding="utf-8")
