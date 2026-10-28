#!/usr/bin/env python3
"""Run the reusable hard-fail scientific figure validation pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image

from ..builders.contract_loader import ContractError, FigureContract, load_contract
from . import asset_checker, bbox_checker, color_checker, font_checker, route_checker, scientific_checker, strict_data_checker, svg_checker, text_collision_checker, visual_narrative_checker
from .common import Check, check


PDFINFO_CANDIDATES = [
    Path("/Users/baikaizhi/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdfinfo"),
    Path("/opt/homebrew/bin/pdfinfo"),
]


def _resolve(contract: FigureContract, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (contract.path.parent / path).resolve()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _output_checks(contract: FigureContract, svg: Path, pdf: Path | None, png: Path | None) -> list[Check]:
    requirements = contract["validation_requirements"]
    checks = [check("output", "editable_svg_present", svg.is_file(), str(svg))]
    if requirements.get("require_pdf", True):
        checks.append(check("output", "pdf_present", bool(pdf and pdf.is_file()), str(pdf)))
    if requirements.get("require_png_preview", True):
        checks.append(check("output", "png_preview_present", bool(png and png.is_file()), str(png)))
    if png and png.is_file():
        with Image.open(png) as image:
            checks.append(check("output", "png_resolution", image.width >= 1600 and image.height >= 1000,
                                f"PNG preview is {image.width}x{image.height} px."))
            corner = image.convert("RGB").getpixel((image.width - 2, 2))
            checks.append(check("output", "white_background", min(corner) >= 245,
                                f"Top-right corner RGB={corner}."))
    if pdf and pdf.is_file():
        pdfinfo = next((path for path in PDFINFO_CANDIDATES if path.is_file()), None)
        if pdfinfo:
            result = subprocess.run([str(pdfinfo), str(pdf)], capture_output=True, text=True, check=False)
            page_size = next((line for line in result.stdout.splitlines() if line.startswith("Page size:")), "")
            pages = next((line for line in result.stdout.splitlines() if line.startswith("Pages:")), "")
            checks.append(check("output", "pdf_single_page", pages.rstrip().endswith("1"), pages or "pdfinfo unavailable"))
            checks.append(check("output", "pdf_page_size_recorded", bool(page_size), page_size or "page size missing"))
        else:
            checks.append(check("output", "pdf_metadata", True, "pdfinfo unavailable; PDF existence checked only.", severity="warning"))
    return checks


def _integration_checks(contract: FigureContract) -> list[Check]:
    requirement = contract["validation_requirements"]
    record = _resolve(contract, requirement.get("engine_preflight_record"))
    if not requirement.get("require_engine_preflight", True):
        return []
    if not record or not record.is_file():
        return [check("integration", "engine_preflight", False, f"Missing engine preflight record: {record}")]
    try:
        payload = _load_json(record)
        figure_matches = payload.get("figure_id") == contract.figure_id
        hashes_match = payload.get("rule_hashes") == contract.rules.hashes
        contract_matches = Path(str(payload.get("contract", ""))).resolve() == contract.path
        valid = figure_matches and hashes_match and contract_matches
        evidence = {
            "record": payload,
            "figure_matches": figure_matches,
            "hashes_match_current_rules": hashes_match,
            "contract_matches": contract_matches,
        }
        return [check("integration", "engine_preflight", valid,
                      f"Engine preflight record matches the current contract and all canonical rule hashes: {record}",
                      evidence=evidence)]
    except Exception as exc:
        return [check("integration", "engine_preflight", False, f"Invalid engine preflight record: {exc}")]


def _legacy_evidence_checks(contract: FigureContract) -> list[Check]:
    checks: list[Check] = []
    for item in contract["validation_requirements"].get("legacy_evidence", []):
        path = _resolve(contract, item.get("path"))
        name = item.get("name", Path(path or "evidence").stem)
        expected_status = item.get("expected_status", "pass")
        if not path or not path.is_file():
            checks.append(check("evidence", name, False, f"Missing legacy evidence: {path}"))
            continue
        try:
            payload = _load_json(path)
            actual = payload
            for key in item.get("status_path", ["status"]):
                actual = actual[key]
            passed = str(actual).lower() == str(expected_status).lower()
            checks.append(check("evidence", name, passed, f"{path.name}: status={actual!r}; expected={expected_status!r}"))
        except Exception as exc:
            checks.append(check("evidence", name, False, f"Could not inspect {path}: {exc}"))
    return checks


def lint_figure(*, contract_path: str | Path, svg_path: str | Path, geometry_path: str | Path | None,
                output_dir: str | Path, pdf_path: str | Path | None = None, png_path: str | Path | None = None,
                report_prefix: str = "figure_validation_report") -> dict:
    contract = load_contract(contract_path)
    svg = _resolve(contract, svg_path)
    pdf = _resolve(contract, pdf_path)
    png = _resolve(contract, png_path)
    geometry = _resolve(contract, geometry_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks: list[Check] = []

    checks.extend(_output_checks(contract, svg, pdf, png))
    checks.extend(_integration_checks(contract))
    checks.extend(_legacy_evidence_checks(contract))
    checks.extend(asset_checker.validate(contract.data, contract.rules.rules))
    checks.extend(strict_data_checker.validate(contract.data, contract.path.parent))

    svg_checks, svg_summary = svg_checker.inspect(svg, contract.data)
    checks.extend(svg_checks)
    checks.extend(color_checker.validate(svg_summary, contract.data, contract.rules.rules))
    checks.extend(scientific_checker.validate(svg_summary.get("raw_text", ""), contract.data))

    manifest = None
    if geometry and geometry.is_file():
        try:
            manifest = _load_json(geometry)
            checks.append(check("geometry", "geometry_manifest", True, str(geometry)))
        except Exception as exc:
            checks.append(check("geometry", "geometry_manifest", False, f"Invalid geometry manifest: {exc}"))
    elif contract["validation_requirements"].get("require_geometry_manifest", True):
        checks.append(check("geometry", "geometry_manifest", False, f"Missing geometry manifest: {geometry}"))

    if manifest:
        checks.extend(bbox_checker.validate(manifest, contract.data, contract.rules.rules))
        checks.extend(text_collision_checker.validate(
            manifest,
            label_clearance=float(contract.rules["spacing"]["clearance_px"]["label_to_label_minimum"]),
            leader_clearance=float(contract.rules["spacing"]["clearance_px"]["label_to_leader_minimum"]),
            border_clearance=float(contract.rules["spacing"]["clearance_px"]["text_to_border_minimum"]),
            text_plot_clearance=float(contract.rules["spacing"]["clearance_px"]["text_to_plot_minimum"]),
        ))
        checks.extend(font_checker.validate(manifest, contract.data, contract.rules.rules))
        checks.extend(route_checker.validate(manifest, contract.rules.rules))

    narrative_checks, narrative_payload = visual_narrative_checker.validate(
        manifest, contract.data, contract.rules.rules, png_path=png
    )
    checks.extend(narrative_checks)
    visual_narrative_checker.write_report(narrative_payload, output)

    hard_failures = [item for item in checks if item.severity == "hard_fail" and not item.passed]
    warnings = [item for item in checks if item.severity == "warning" and not item.passed]
    human_reviews = [item for item in checks if item.severity == "human_review"]
    report = {
        "schema_version": 1,
        "figure_id": contract.figure_id,
        "contract": str(contract.path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine_root": str(contract.rules.root),
        "rule_hashes": contract.rules.hashes,
        "status": "pass" if not hard_failures else "fail",
        "hard_failure_count": len(hard_failures),
        "warning_count": len(warnings),
        "human_review_count": len(human_reviews),
        "checks_passed": sum(item.passed for item in checks),
        "checks_total": len(checks),
        "checks": [item.as_dict() for item in checks],
        "gate": contract["approval"],
    }
    json_path = output / f"{report_prefix}.json"
    md_path = output / f"{report_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# {contract.figure_id} validation report", "",
        f"- Status: **{report['status'].upper()}**",
        f"- Hard failures: **{report['hard_failure_count']}**",
        f"- Warnings: **{report['warning_count']}**",
        f"- Human-review judgments: **{report['human_review_count']}**",
        f"- Checks passed: **{report['checks_passed']}/{report['checks_total']}**", "",
        "| Result | Severity | Category | Check | Detail |", "|---|---|---|---|---|",
    ]
    for item in checks:
        result = "pass" if item.passed else "fail"
        detail = item.detail.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {result} | {item.severity} | {item.category} | `{item.name}` | {detail} |")
    lines.extend(["", "Automated validation does not pass the human approval gate.", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--svg", required=True)
    parser.add_argument("--geometry")
    parser.add_argument("--pdf")
    parser.add_argument("--png")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-prefix", default="figure_validation_report")
    args = parser.parse_args()
    try:
        report = lint_figure(
            contract_path=args.contract, svg_path=args.svg, geometry_path=args.geometry,
            pdf_path=args.pdf, png_path=args.png, output_dir=args.output_dir,
            report_prefix=args.report_prefix,
        )
    except (ContractError, ValueError, FileNotFoundError) as exc:
        print(f"figure lint input error: {exc}")
        return 2
    print(json.dumps({"figure_id": report["figure_id"], "status": report["status"],
                      "hard_failures": report["hard_failure_count"], "warnings": report["warning_count"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
