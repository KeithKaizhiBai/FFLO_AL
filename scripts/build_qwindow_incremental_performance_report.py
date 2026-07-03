from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_rank_jsons(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("iter*/exact_shard_rank*_of*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            row = {"error": "json_decode_error"}
        row["source_file"] = str(path)
        row["iteration"] = path.parent.name
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(run_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_rank_jsons(run_dir)
    rank_table = output_dir / "rank_runtime_summary.csv"
    df.to_csv(rank_table, index=False)
    if df.empty:
        summary = pd.DataFrame(
            [{"status": "no_rank_jsons_found", "run_dir": str(run_dir)}]
        )
    else:
        numeric_cols = [
            c
            for c in [
                "elapsed_sec",
                "n_points",
                "point_total_runtime_sec_sum",
                "base_scan_runtime_sec_sum",
                "q_expansion_runtime_sec_sum",
                "delta_refinement_runtime_sec_sum",
                "local_refinement_runtime_sec_sum",
                "fallback_full_rescan_runtime_sec_sum",
                "total_q_points_evaluated",
                "total_estimated_grid_evaluations",
                "incremental_expansion_used_count",
                "fallback_full_rescan_used_count",
            ]
            if c in df.columns
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        summary = (
            df.groupby("iteration", dropna=False)[numeric_cols]
            .sum(min_count=1)
            .reset_index()
            if numeric_cols
            else pd.DataFrame({"iteration": sorted(df["iteration"].unique())})
        )
    summary_table = output_dir / "iteration_runtime_summary.csv"
    summary.to_csv(summary_table, index=False)
    md_lines = [
        "# q-window incremental performance report",
        "",
        f"- run_dir: `{run_dir}`",
        f"- rank summary table: `{rank_table}`",
        f"- iteration summary table: `{summary_table}`",
        "",
        "This report is generated from existing exact shard JSON files only.",
        "Missing timing fields mean the corresponding run predates the instrumentation.",
    ]
    (output_dir / "qwindow_incremental_performance.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a report-only timing summary from exact shard JSON files.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qwindow_incremental_performance"))
    args = parser.parse_args()
    build_report(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
