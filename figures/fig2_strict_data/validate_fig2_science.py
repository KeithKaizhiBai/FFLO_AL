#!/usr/bin/env python3
"""Figure-2-specific scientific and provenance validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
CANON = HERE / "canonical"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = yaml.safe_load((HERE / "fig2_data_manifest.yaml").read_text(encoding="utf-8"))
    contract = yaml.safe_load((HERE / "fig2_panel_contract.yaml").read_text(encoding="utf-8"))
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for name, source in manifest["sources"].items():
        path = Path(source["path"])
        add(f"source_hash:{name}", path.is_file() and sha256(path) == source["sha256"], str(path))
    for name, record in manifest["canonical_outputs"].items():
        path = CANON / name
        rows = len(pd.read_csv(path)) if path.is_file() else -1
        add(f"canonical:{name}", path.is_file() and sha256(path) == record["sha256"] and rows == record["rows"],
            f"rows={rows}; expected={record['rows']}")

    final = pd.read_csv(CANON / "fig2_final_exact_samples.csv")
    cold = pd.read_csv(CANON / "fig2_cold_start_exact_calls.csv")
    allocation = pd.read_csv(CANON / "fig2_exact_call_allocation.csv")
    convergence = pd.read_csv(CANON / "fig2_raw_convergence_metrics.csv")
    risk = pd.read_csv(CANON / "fig2_hard_risk_points.csv")
    counts = final.phase_name.value_counts().to_dict()
    expected = {"normal": 1867, "uniform_SC": 715, "FFLO": 4852}
    add("stage2_phase_invariants", len(final) == 7434 and counts == expected,
        f"total={len(final)}; counts={counts}")
    add("cold_start_reconciliation", len(cold) == 512 and int(cold.stage1_exact_match.sum()) == 506
        and int(cold.endpoint_match.sum()) == 506 and (cold.stage1_exact_match == cold.endpoint_match).all(),
        f"selected={len(cold)}; stage1_exact={int(cold.stage1_exact_match.sum())}; retained_stage2={int(cold.endpoint_match.sum())}; unmatched={int((~cold.stage1_exact_match.astype(bool)).sum())}")
    per_iteration = allocation.groupby("iteration").size().to_dict()
    add("selection_history_complete", list(sorted(per_iteration)) == list(range(35)) and per_iteration.get(0) == 512
        and all(per_iteration.get(i) == 256 for i in range(1, 35)) and len(allocation) == 9216,
        f"iterations={min(per_iteration)}-{max(per_iteration)}; calls={len(allocation)}")
    add("hard_risk_separate_layer", len(risk) == 129 and int(final.hard_risk.sum()) == 19
        and set(final.phase_name.unique()) == set(expected),
        f"risk_manifest={len(risk)}; risk_overlay={int(final.hard_risk.sum())}; phases={sorted(final.phase_name.unique())}")
    add("raw_convergence_complete", convergence.iteration.tolist() == list(range(35)) and len(convergence) == 35,
        f"iterations={len(convergence)}; final_phase_map_change={convergence.iloc[-1].phase_map_change:.12g}")

    panels = {panel["panel_id"]: panel for panel in contract["strict_data"]["panels"]}
    final_panel = panels["final_phase_map"]
    add("phase_mapping_frozen", final_panel["phase_mapping"] == expected.__class__({"normal": 0, "uniform_SC": 1, "FFLO": 2}),
        str(final_panel["phase_mapping"]))
    forbidden_enabled = []
    for panel_id, panel in panels.items():
        for operation in ("interpolation", "smoothing", "boundary_extraction", "normalization"):
            if panel["transformations"].get(operation, {}).get("enabled"):
                forbidden_enabled.append(f"{panel_id}:{operation}")
    add("no_unauthorized_transform", not forbidden_enabled, str(forbidden_enabled))

    outputs = ["fig2_strict_data.svg", "fig2_strict_data.pdf", "fig2_strict_data.png",
               "fig2_contact_sheet.pdf", "fig2_contact_sheet.png", "fig2_strict_data_grayscale.png"]
    add("review_outputs_present", all((HERE / name).is_file() and (HERE / name).stat().st_size > 0 for name in outputs),
        ", ".join(outputs))
    pdfinfo = subprocess.run(["pdfinfo", str(HERE / "fig2_strict_data.pdf")], capture_output=True, text=True, check=False).stdout
    add("pdf_single_page_print_size", "Pages:           1" in pdfinfo and "504.567 x 340.157 pts" in pdfinfo,
        next((line for line in pdfinfo.splitlines() if line.startswith("Page size:")), "missing"))

    failures = [item for item in checks if not item["passed"]]
    result = {"schema_version": 1, "figure_id": "FIG-02-THERMO-STRICT",
              "status": "pass" if not failures else "fail", "physics_recomputed": False,
              "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "checks": checks}
    (HERE / "fig2_specific_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Figure 2 specific validation", "", f"- Status: **{result['status'].upper()}**",
             f"- Checks: **{result['checks_passed']}/{result['checks_total']}**", "- Physics recomputed: **no**", "",
             "| Result | Check | Detail |", "|---|---|---|"]
    lines.extend(f"| {'PASS' if x['passed'] else 'FAIL'} | `{x['name']}` | {x['detail']} |" for x in checks)
    (HERE / "fig2_specific_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": f"{result['checks_passed']}/{result['checks_total']}"}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
