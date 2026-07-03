from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.config import ActiveLearningConfig
from ml_phase.stop_controller import StopConfig, label_surprise_metrics


DEFAULT_RUN_DIR = (
    ROOT
    / "rankcap_k3_full_loop"
    / "ML_Phase_512_RankCapK3_FullLoop"
    / "active_runs"
    / "active_boundary_discovery_rankcap_k3_full_loop_v1"
)
DEFAULT_OUT_DIR = ROOT / "reports" / "stopcontroller_surprise_replay"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_cfg(run_dir: Path) -> ActiveLearningConfig:
    raw = read_json(run_dir / "run_config.json", {})
    cfg_raw = raw.get("active_learning_config", {}) if isinstance(raw, dict) else {}
    if not isinstance(cfg_raw, dict):
        cfg_raw = {}
    return ActiveLearningConfig.from_dict(cfg_raw)


def stop_config_from_saved(saved: dict[str, Any], min_denominator: int, min_fraction: float) -> StopConfig:
    raw = saved.get("stop_config", {}) if isinstance(saved.get("stop_config"), dict) else {}
    return StopConfig(
        min_iterations=int(raw.get("min_iterations", 5)),
        patience=int(raw.get("patience", 4)),
        max_iterations=int(raw.get("max_iterations", saved.get("completed_iterations", 0) or 50)),
        max_exact_calls=raw.get("max_exact_calls"),
        warmup_reference_iters=int(raw.get("warmup_reference_iters", 3)),
        map_tol=float(raw.get("map_tol", 0.002)),
        boundary_shift_tol=raw.get("boundary_shift_tol"),
        surprise_tol=float(raw.get("surprise_tol", 0.05)),
        selected_a0_ratio_tol=float(raw.get("selected_a0_ratio_tol", 0.15)),
        qedge_rate_tol=float(raw.get("qedge_rate_tol", 0.01)),
        rerun_rate_tol=float(raw.get("rerun_rate_tol", 0.01)),
        coverage_tol=raw.get("coverage_tol"),
        allow_missing_boundary=bool(raw.get("allow_missing_boundary", False)),
        required_pass_count=int(raw.get("required_pass_count", saved.get("required_pass_count", 4))),
        stop_surprise_mode=str(raw.get("stop_surprise_mode", "all_selected")),
        trusted_surprise_min_denominator=int(min_denominator),
        trusted_surprise_min_fraction=float(min_fraction),
    )


def condition_from_rate(rate: float | None, available: bool, tol: float, denominator_valid: bool = True) -> bool:
    if not denominator_valid:
        return False
    if not available or rate is None:
        return False
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and value < float(tol))


def replay(
    history: list[dict[str, Any]],
    run_dir: Path,
    cfg: ActiveLearningConfig,
    mode: str,
    min_denominator: int,
    min_fraction: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patience_counter = 0
    for saved in sorted(history, key=lambda item: int(item.get("iteration", -1))):
        it = int(saved.get("iteration", -1))
        stop_cfg = stop_config_from_saved(saved, min_denominator, min_fraction)
        stop_cfg = replace(stop_cfg, stop_surprise_mode=mode)
        iter_dir = run_dir / f"iter{it:03d}"
        surprise = label_surprise_metrics(iter_dir, it, cfg, stop_cfg)
        if mode == "all_selected":
            metric = surprise["all_selected"]
            denominator_valid = bool(metric["available"])
            metric_name = "label_surprise_all_selected"
        elif mode == "trusted":
            metric = surprise["trusted"]
            denominator_valid = bool(surprise["trusted_surprise_denominator_valid"])
            metric_name = "label_surprise_trusted"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        saved_conditions = saved.get("conditions", {})
        c1 = bool(saved_conditions.get("C1_phase_map_change", False))
        c2 = bool(saved_conditions.get("C2_boundary_shift_normal_sc", False))
        c3 = bool(saved_conditions.get("C3_boundary_shift_uniform_fflo", False))
        c4 = condition_from_rate(metric["rate"], bool(metric["available"]), stop_cfg.surprise_tol, denominator_valid)
        c5 = bool(saved_conditions.get("C5_boundary_coverage_p95", False))
        passed = int(sum([c1, c2, c3, c4, c5]))
        completed = int(saved.get("completed_iterations", it + 1))
        convergence_pass = bool(completed >= int(stop_cfg.min_iterations) and passed >= int(stop_cfg.required_pass_count))
        patience_counter = patience_counter + 1 if convergence_pass else 0
        hard_stop = bool(saved.get("hard_stop", False))
        would_stop = bool(hard_stop or patience_counter >= int(stop_cfg.patience))
        if patience_counter >= int(stop_cfg.patience):
            stop_reason = "converged_main_phase_boundaries"
        elif hard_stop:
            stop_reason = str(saved.get("stop_reason") or "max_iterations")
        else:
            stop_reason = ""
        rows.append(
            {
                "iteration": it,
                "stop_surprise_mode": mode,
                "surprise_metric_name": metric_name,
                "surprise_rate": metric["rate"],
                "n_surprise": int(metric["n_surprise"]),
                "n_denominator": int(metric["n_denominator"]),
                "denominator_fraction_selected": metric["denominator_fraction_selected"],
                "trusted_surprise_denominator_valid": bool(surprise["trusted_surprise_denominator_valid"]),
                "trusted_surprise_min_denominator": int(min_denominator),
                "trusted_surprise_min_fraction": float(min_fraction),
                "label_surprise_pass": c4,
                "phase_map_pass": c1,
                "normal_sc_shift_pass": c2,
                "uniform_fflo_shift_pass": c3,
                "boundary_coverage_pass": c5,
                "passed_condition_count": passed,
                "required_pass_count": int(stop_cfg.required_pass_count),
                "convergence_pass": convergence_pass,
                "patience_counter": patience_counter,
                "would_stop": would_stop,
                "stop_reason": stop_reason,
                "saved_label_surprise_rate": saved.get("metrics", {}).get("label_surprise_rate"),
                "saved_passed_condition_count": saved.get("passed_condition_count"),
                "saved_convergence_pass": saved.get("convergence_pass"),
                "saved_patience_counter": saved.get("patience_counter"),
                "saved_stop": saved.get("stop"),
                "saved_stop_reason": saved.get("stop_reason"),
                "hard_risk_surprise_rate": surprise["hard_risk"]["rate"],
                "hard_risk_n_denominator": int(surprise["hard_risk"]["n_denominator"]),
                "hard_risk_fraction_selected": surprise["hard_risk_fraction_selected"],
                "numerical_frontier_status": surprise["numerical_frontier_status"],
            }
        )
    return rows


def discrepancy_rows(current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in current_rows:
        saved_rate = row["saved_label_surprise_rate"]
        rate = row["surprise_rate"]
        if saved_rate is None or rate is None:
            diff = None
            surprise_match = saved_rate is None and rate is None
        else:
            diff = abs(float(saved_rate) - float(rate))
            surprise_match = diff <= 0.0
        out.append(
            {
                "iteration": row["iteration"],
                "surprise_match": surprise_match,
                "abs_surprise_discrepancy": diff,
                "passed_count_match": int(row["saved_passed_condition_count"]) == int(row["passed_condition_count"]),
                "convergence_pass_match": bool(row["saved_convergence_pass"]) == bool(row["convergence_pass"]),
                "patience_counter_match": int(row["saved_patience_counter"]) == int(row["patience_counter"]),
                "stop_match": bool(row["saved_stop"]) == bool(row["would_stop"]),
            }
        )
    return out


def earliest_summary(rows_by_mode: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, rows in rows_by_mode.items():
        first_pass = next((r for r in rows if bool(r["convergence_pass"])), None)
        first_stop = next((r for r in rows if r["stop_reason"] == "converged_main_phase_boundaries"), None)
        final = rows[-1] if rows else {}
        remaining = []
        if final and not bool(final.get("label_surprise_pass")):
            remaining.append(str(final.get("surprise_metric_name")))
        if final and not bool(final.get("boundary_coverage_pass")):
            remaining.append("boundary_coverage_p95")
        out.append(
            {
                "scenario": name,
                "earliest_iteration_meeting_required_pass_count": None if first_pass is None else int(first_pass["iteration"]),
                "earliest_counterfactual_stop_iteration": None if first_stop is None else int(first_stop["iteration"]),
                "whether_stop_occurs_before_final": bool(first_stop is not None and int(first_stop["iteration"]) < int(final.get("iteration", 0))),
                "remaining_failed_conditions_at_final": ";".join(remaining),
                "final_passed_condition_count": final.get("passed_condition_count"),
                "final_surprise_rate": final.get("surprise_rate"),
                "final_denominator_count": final.get("n_denominator"),
            }
        )
    return out


def write_markdown(out_dir: Path, summary: list[dict[str, Any]], discrepancies: list[dict[str, Any]]) -> Path:
    current_ok = all(
        bool(r["surprise_match"])
        and bool(r["passed_count_match"])
        and bool(r["convergence_pass_match"])
        and bool(r["patience_counter_match"])
        and bool(r["stop_match"])
        for r in discrepancies
    )
    lines = [
        "# StopController Surprise-Mode Replay",
        "",
        "This report replays saved StopController artifacts without rerunning exact BdG calculations.",
        "",
        f"- current all-selected reconstruction: {'pass' if current_ok else 'fail'}",
        "",
        "## Earliest Stop Summary",
        "",
        "| scenario | first required pass | earliest stop | final pass count | final surprise | final denominator | remaining final blockers |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario']} | {row['earliest_iteration_meeting_required_pass_count']} | "
            f"{row['earliest_counterfactual_stop_iteration']} | {row['final_passed_condition_count']} | "
            f"{row['final_surprise_rate']} | {row['final_denominator_count']} | "
            f"{row['remaining_failed_conditions_at_final']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`all_selected` is the backward-compatible current StopController mode. "
            "`trusted` uses only trusted, training-eligible, non-rerun, q-resolved, and delta-resolved exact labels for C4. "
            "Hard-risk surprise remains recorded as a numerical-frontier diagnostic.",
            "",
            "No StopController threshold, physical phase criterion, exact-oracle path, acquisition weight, or rankcap_k3 behavior is changed by this replay.",
        ]
    )
    path = out_dir / "replay_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_pdf(md_path: Path, pdf_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return
    text = md_path.read_text(encoding="utf-8")
    chunks = [text[i : i + 2600] for i in range(0, len(text), 2600)] or [""]
    with PdfPages(pdf_path) as pdf:
        for chunk in chunks:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
            ax.axis("off")
            ax.text(0, 1, chunk, va="top", ha="left", family="monospace", fontsize=7, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay StopController all-selected and trusted surprise modes.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trusted-surprise-min-denominator", type=int, default=64)
    parser.add_argument("--trusted-surprise-min-fraction", type=float, default=0.25)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.output_dir.resolve()
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    history = read_json(run_dir / "stop_metrics_history.json", [])
    if not history:
        raise SystemExit(f"Missing stop history: {run_dir / 'stop_metrics_history.json'}")
    cfg = load_cfg(run_dir)
    current = replay(history, run_dir, cfg, "all_selected", args.trusted_surprise_min_denominator, args.trusted_surprise_min_fraction)
    trusted = replay(history, run_dir, cfg, "trusted", args.trusted_surprise_min_denominator, args.trusted_surprise_min_fraction)
    rows_by_mode = {"all_selected": current, "trusted": trusted}
    discrepancies = discrepancy_rows(current)
    sensitivity: list[dict[str, Any]] = []
    for min_den, min_frac in [(16, 0.0), (32, 0.0), (64, 0.0), (64, 0.25)]:
        replay_rows = replay(history, run_dir, cfg, "trusted", min_den, min_frac)
        summary = earliest_summary({f"trusted_min{min_den}_frac{min_frac:g}": replay_rows})[0]
        summary["trusted_surprise_min_denominator"] = min_den
        summary["trusted_surprise_min_fraction"] = min_frac
        sensitivity.append(summary)

    summary_rows = earliest_summary(rows_by_mode)
    write_csv(tables_dir / "current_mode_reconstruction.csv", current)
    write_csv(tables_dir / "trusted_mode_reconstruction.csv", trusted)
    write_csv(tables_dir / "denominator_sensitivity.csv", sensitivity)
    write_csv(tables_dir / "patience_reconstruction.csv", current + trusted)
    write_csv(tables_dir / "earliest_stop_summary.csv", summary_rows)
    write_csv(tables_dir / "discrepancy_check.csv", discrepancies)
    md = write_markdown(out_dir, summary_rows, discrepancies)
    write_pdf(md, out_dir / "replay_report.pdf")
    decision = [
        "# StopController Surprise Replay Decision Log",
        "",
        f"- run_dir: `{run_dir}`",
        f"- current all-selected reconstruction: {'pass' if all(r['surprise_match'] and r['passed_count_match'] and r['convergence_pass_match'] and r['patience_counter_match'] and r['stop_match'] for r in discrepancies) else 'fail'}",
        "- trusted replay uses the same thresholds and patience, replacing only the C4 surprise input.",
        "- no exact BdG calculation was rerun.",
    ]
    (out_dir / "decision_log.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(md), "tables": str(tables_dir), "current_reconstruction_rows": len(current)}, indent=2))


if __name__ == "__main__":
    main()
