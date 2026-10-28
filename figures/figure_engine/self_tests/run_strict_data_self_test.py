#!/usr/bin/env python3
"""Run strict-data panel-router negative and control fixtures."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from figures.figure_engine.builders.panel_router import validate_panel_contract  # noqa: E402


def set_path(data: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    target = data
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def fixture(base: dict, kind: str | None) -> dict:
    panel = copy.deepcopy(base)
    if kind == "topology":
        panel["scientific_role"] = "trusted_topology_map"
        panel["topology_mapping"] = {"cFFLO": 0, "tFFLO": 1, "gap_closing": "transition_only"}
        panel["trust_mask"] = "topology_trusted & topology_applicable"
        panel["sensitivity_notes"] = ["configured k=8", "k=6 sensitivity retained"]
    elif kind == "high_dimensional":
        panel["scientific_role"] = "3D higher-dimensional exploration"
        panel["axis_fields"] = ["kT", "JA", "mu"]
        panel["axis_units"]["mu"] = "mu/t"
        panel["required_caveats"] = ["finite sampled domain with discrete support and no inferred continuous surface"]
    elif kind == "diagnostic_only":
        panel["scientific_role"] = "archived numerical stability diagnostic"
        panel["convergence_status"] = {
            "mode": "diagnostic_only",
            "threshold_line_visible": False,
            "pass_fail_labels_visible": False,
            "raw_trace_visible": True,
        }
    elif kind == "hybrid":
        panel["panel_type"] = "HYBRID_DATA_SCHEMATIC"
    return panel


def main() -> int:
    root = HERE / "strict_data_fixtures"
    base = yaml.safe_load((root / "control_panel.yaml").read_text(encoding="utf-8"))
    cases = yaml.safe_load((root / "negative_cases.yaml").read_text(encoding="utf-8"))["cases"]
    control_errors = validate_panel_contract(base)
    results = []
    for case in cases:
        panel = fixture(base, case.get("fixture"))
        set_path(panel, case["mutation"], case.get("value"))
        errors = validate_panel_contract(panel)
        results.append({"id": case["id"], "expected": case["expected"], "errors": errors,
                        "detected": case["expected"] in errors})
    passed = not control_errors and all(item["detected"] for item in results)
    payload = {"schema_version": 1, "status": "pass" if passed else "fail",
               "control_errors": control_errors, "negative_cases": results}
    output = HERE / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "strict_data_self_test.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Strict-data self-test", "", f"Status: **{payload['status'].upper()}**", "",
             f"Control errors: `{control_errors}`", "", "| Fixture | Expected | Detected |", "|---|---|---|"]
    lines.extend(f"| `{x['id']}` | `{x['expected']}` | {'yes' if x['detected'] else 'no'} |" for x in results)
    (output / "strict_data_self_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "negative_cases": len(results)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
