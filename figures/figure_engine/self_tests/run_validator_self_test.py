#!/usr/bin/env python3
"""Run negative and corrected-control tests against the full figure lint."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from figures.figure_engine.validators.figure_lint import lint_figure  # noqa: E402


EXPECTED_INVALID_FAILURES = {
    "approved_palette",
    "prohibited_claims",
    "independent_card_gutters",
    "primary_visual_area",
    "declared_area_requirements",
    "text_text_overlap",
    "text_border_clearance",
    "text_plot_clearance",
    "leader_leader_crossing",
    "minimum_font_size",
    "normal_width_fonts",
    "nonuniform_text_scaling",
    "workflow_profile",
    "workflow_card_crossing",
    "annotation_length",
    "narrative_primary_scientific_subobject",
    "scientific_asset_fill_ratio",
}


def run_case(name: str, svg: str, geometry: str) -> dict:
    directory = HERE / name
    return lint_figure(
        contract_path=directory / "figure_contract.yaml",
        svg_path=svg,
        geometry_path=geometry,
        output_dir=HERE / "output" / name,
        report_prefix="figure_validation_report",
    )


def main() -> int:
    invalid = run_case("invalid_figure", "invalid_figure.svg", "invalid_geometry.json")
    corrected = run_case("corrected_figure", "corrected_figure.svg", "corrected_geometry.json")
    failed_names = {item["name"] for item in invalid["checks"] if item["severity"] == "hard_fail" and not item["passed"]}
    missing = sorted(EXPECTED_INVALID_FAILURES - failed_names)
    unexpected_corrected = [item for item in corrected["checks"] if item["severity"] == "hard_fail" and not item["passed"]]
    passed = invalid["status"] == "fail" and not missing and corrected["status"] == "pass" and not unexpected_corrected
    payload = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "invalid_fixture_status": invalid["status"],
        "detected_expected_failures": sorted(EXPECTED_INVALID_FAILURES & failed_names),
        "missing_expected_failures": missing,
        "additional_detected_failures": sorted(failed_names - EXPECTED_INVALID_FAILURES),
        "corrected_fixture_status": corrected["status"],
        "unexpected_corrected_failures": unexpected_corrected,
    }
    output = HERE / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "validator_self_test_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Validator self-test summary", "",
        f"- Overall self-test: **{payload['status'].upper()}**",
        f"- Intentionally invalid fixture: **{invalid['status'].upper()}**",
        f"- Corrected control fixture: **{corrected['status'].upper()}**", "",
        "## Expected violations detected", "",
    ]
    lines.extend(f"- `{name}`" for name in payload["detected_expected_failures"])
    lines.extend(["", "## Missing expected violations", ""])
    lines.extend([*(f"- `{name}`" for name in missing), *( ["- none"] if not missing else [])])
    lines.extend(["", "The synthetic fixtures are validator tests, not manuscript figures.", ""])
    (output / "validator_self_test_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
