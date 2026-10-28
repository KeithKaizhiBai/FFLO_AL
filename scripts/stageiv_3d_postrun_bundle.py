from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.stageiv_3d import STAGEIV_RUN_ID, StageIV3DConfig
from scripts.stageiv_3d_convergence_audit import run_audit as run_convergence_audit
from scripts.stageiv_3d_hidden_slice_audit import build_audit as run_hidden_slice_audit
from scripts.stageiv_3d_postrun_report import build_report as run_postrun_summary


def json_clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def component_row(name: str, output_dir: Path, decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "component": name,
        "output_dir": str(output_dir),
        "decision_class": decision.get("decision_class", ""),
        "status": decision.get("stageiv_convergence_status", decision.get("hidden_slice_status", decision.get("postrun_status", ""))),
        "need_new_exact_calculation": decision.get("need_new_exact_calculation", ""),
        "recommended_next_action": decision.get("recommended_next_action", ""),
        "reason": decision.get("reason", ""),
    }


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    rows = [[str(col) for col in df.columns]]
    for _, row in df.iterrows():
        rows.append([str(row[col]) for col in df.columns])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def fmt(parts: list[str]) -> str:
        return "| " + " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts))) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    return "\n".join([fmt(rows[0]), sep, *[fmt(r) for r in rows[1:]]])


def bundle_decision(
    *,
    run_dir: Path,
    output_dir: Path,
    reference_dataset: Path | None,
    summary_decision: dict[str, Any],
    convergence_decision: dict[str, Any],
    hidden_decision: dict[str, Any],
) -> dict[str, Any]:
    convergence_status = str(convergence_decision.get("stageiv_convergence_status", "unknown"))
    hidden_status = str(hidden_decision.get("hidden_slice_status", "unknown"))
    hidden_passed = bool(hidden_decision.get("hidden_slice_passed", False))
    hidden_required_available = reference_dataset is not None

    if convergence_status in {"insufficient_history", "inconclusive"}:
        status = "incomplete_convergence_history"
        decision_class = "Decision D"
        next_action = "collect_returned_stageiv_history_and_rerun_postrun_bundle"
    elif not hidden_required_available:
        status = "hidden_slice_reference_missing"
        decision_class = "Decision D"
        next_action = "rerun_with_reference_dataset_for_hidden_fixed_mu_validation"
    elif hidden_status == "inconclusive":
        status = "hidden_slice_inconclusive"
        decision_class = "Decision D"
        next_action = "inspect_hidden_slice_inputs_before_stageiv_claims"
    elif convergence_status == "preliminary_pass" and hidden_passed:
        status = "postrun_bundle_pass"
        decision_class = "Decision A"
        next_action = "build_stageiv_final_report_from_bundle_outputs"
    elif convergence_status == "near_converged_coverage_limited" and hidden_passed:
        status = "near_converged_coverage_limited"
        decision_class = "Decision B"
        next_action = "review_coverage_margin_before_targeted_tail_batches"
    else:
        status = "postrun_bundle_not_passed"
        decision_class = "Decision C"
        next_action = "inspect_failed_stageiv_component_reports"

    need_exact = bool(convergence_decision.get("need_new_exact_calculation", False)) or bool(hidden_decision.get("need_new_exact_calculation", False))
    return {
        "run_id": "stageiv_3d_postrun_bundle",
        "source_run_id": convergence_decision.get("source_run_id", STAGEIV_RUN_ID),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "latest_dataset": summary_decision.get("latest_dataset"),
        "latest_dataset_iteration": summary_decision.get("latest_dataset_iteration"),
        "sample_count": summary_decision.get("sample_count"),
        "postrun_bundle_status": status,
        "decision_class": decision_class,
        "convergence_status": convergence_status,
        "hidden_slice_status": hidden_status,
        "hidden_slice_passed": hidden_passed,
        "hidden_reference_dataset": str(reference_dataset) if reference_dataset else None,
        "need_new_exact_calculation": need_exact,
        "recommended_next_action": next_action,
        "component_decisions": {
            "postrun_summary": summary_decision,
            "convergence_audit": convergence_decision,
            "hidden_slice_audit": hidden_decision,
        },
        "caveats": [
            "This bundle only aggregates report-only Stage IV post-run audits.",
            "It does not run thermodynamic exact calculations, Delta-q searches, or active-learning iterations.",
            "A missing hidden fixed-mu reference keeps the Stage IV capability claim inconclusive.",
        ],
    }


def write_markdown(output_dir: Path, decision: dict[str, Any], component_table: pd.DataFrame) -> None:
    lines = [
        "# Stage IV 3D Post-Run Bundle",
        "",
        "## Executive Summary",
        "",
        f"- source_run_id: `{decision.get('source_run_id')}`",
        f"- latest_dataset_iteration: `{decision.get('latest_dataset_iteration')}`",
        f"- sample_count: `{decision.get('sample_count')}`",
        f"- postrun_bundle_status: `{decision.get('postrun_bundle_status')}`",
        f"- decision_class: `{decision.get('decision_class')}`",
        f"- convergence_status: `{decision.get('convergence_status')}`",
        f"- hidden_slice_status: `{decision.get('hidden_slice_status')}`",
        f"- hidden_slice_passed: `{decision.get('hidden_slice_passed')}`",
        f"- need_new_exact_calculation: `{decision.get('need_new_exact_calculation')}`",
        f"- recommended_next_action: `{decision.get('recommended_next_action')}`",
        "",
        "## Component Reports",
        "",
        df_to_markdown(component_table),
        "",
        "## Caveats",
        "",
    ]
    for caveat in decision.get("caveats", []):
        lines.append(f"- {caveat}")
    lines.extend(
        [
            "",
            "## Output Layout",
            "",
            "- `postrun_summary/` contains the lightweight returned-dataset count summary.",
            "- `convergence_audit/` contains the 3D thermodynamic/topology convergence audit.",
            "- `hidden_slice_audit/` contains the fixed-mu hidden validation audit.",
            "- `stageiv_3d_postrun_bundle_decision.json` is the machine-readable aggregate decision.",
            "",
        ]
    )
    write_text(output_dir / "stageiv_3d_postrun_bundle.md", "\n".join(lines))


def run_bundle(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = StageIV3DConfig.from_json(args.config) if args.config else StageIV3DConfig()

    summary_dir = output_dir / "postrun_summary"
    convergence_dir = output_dir / "convergence_audit"
    hidden_dir = output_dir / "hidden_slice_audit"

    summary_decision = run_postrun_summary(args.run_dir, summary_dir)
    convergence_decision = run_convergence_audit(
        args.run_dir,
        convergence_dir,
        cfg,
        audit_cloud_size=args.audit_cloud_size,
        support_radius=args.support_radius,
        neighbor_k=args.neighbor_k,
        surface_max_distance=args.surface_max_distance,
        component_radius=args.component_radius,
        component_min_size=args.component_min_size,
        build_pdf=not args.no_pdf,
    )
    hidden_args = SimpleNamespace(
        source_run_id=args.source_run_id,
        mu_reference=args.mu_reference,
        grid_n=args.hidden_grid_n,
        knn_k=args.hidden_knn_k,
        support_radius=args.hidden_support_radius,
        last_n=args.hidden_last_n,
        component_min_size=args.hidden_component_min_size,
        phase_map_tol=args.hidden_phase_map_tol,
        topology_map_tol=args.hidden_topology_map_tol,
        boundary_shift_tol=args.hidden_boundary_shift_tol,
        coverage_tol=args.hidden_coverage_tol,
        topology_overlap_min=args.hidden_topology_overlap_min,
    )
    hidden_decision = run_hidden_slice_audit(args.run_dir, hidden_dir, args.config, args.reference_dataset, hidden_args)

    decision = bundle_decision(
        run_dir=args.run_dir,
        output_dir=output_dir,
        reference_dataset=args.reference_dataset,
        summary_decision=summary_decision,
        convergence_decision=convergence_decision,
        hidden_decision=hidden_decision,
    )
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    component_table = pd.DataFrame(
        [
            component_row("postrun_summary", summary_dir, summary_decision),
            component_row("convergence_audit", convergence_dir, convergence_decision),
            component_row("hidden_slice_audit", hidden_dir, hidden_decision),
        ]
    )
    component_table.to_csv(tables_dir / "stageiv_postrun_bundle_components.csv", index=False)
    write_json(output_dir / "stageiv_3d_postrun_bundle_decision.json", decision)
    write_markdown(output_dir, decision, component_table)
    return decision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all report-only Stage IV 3D post-run audits and aggregate decisions.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--reference-dataset", type=Path, default=None)
    p.add_argument("--source-run-id", default=STAGEIV_RUN_ID)
    p.add_argument("--audit-cloud-size", type=int, default=20000)
    p.add_argument("--support-radius", type=float, default=0.075)
    p.add_argument("--neighbor-k", type=int, default=8)
    p.add_argument("--surface-max-distance", type=float, default=0.18)
    p.add_argument("--component-radius", type=float, default=0.035)
    p.add_argument("--component-min-size", type=int, default=12)
    p.add_argument("--hidden-grid-n", type=int, default=201)
    p.add_argument("--hidden-knn-k", type=int, default=8)
    p.add_argument("--hidden-support-radius", type=float, default=0.075)
    p.add_argument("--hidden-last-n", type=int, default=5)
    p.add_argument("--hidden-component-min-size", type=int, default=24)
    p.add_argument("--hidden-phase-map-tol", type=float, default=0.01)
    p.add_argument("--hidden-topology-map-tol", type=float, default=0.01)
    p.add_argument("--hidden-boundary-shift-tol", type=float, default=0.004167)
    p.add_argument("--hidden-coverage-tol", type=float, default=0.00625)
    p.add_argument("--hidden-topology-overlap-min", type=float, default=0.95)
    p.add_argument("--mu-reference", type=float, default=None)
    p.add_argument("--no-pdf", action="store_true")
    return p.parse_args()


def main() -> None:
    decision = run_bundle(parse_args())
    print(json.dumps(json_clean(decision), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
