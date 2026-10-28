from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _default_run_root() -> Path:
    env_run_root = os.environ.get("RUN_ROOT")
    if env_run_root:
        return Path(env_run_root)
    if os.access(ROOT, os.W_OK):
        return ROOT / "local_refinement_refactor_stage1_run"
    fallback = os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or os.environ.get("HOME")
    if fallback:
        return Path(fallback) / "local_refinement_refactor_stage1_run"
    return ROOT / "local_refinement_refactor_stage1_run"


def _run_path(path: Path, run_root: Path) -> Path:
    return path if path.is_absolute() else run_root / path


KEY_COLUMNS = ["point_id", "kT", "JA"]
FLAG_COLUMNS = [
    "phase_candidate",
    "trusted_exact",
    "training_eligible_exact",
    "q_unresolved",
    "delta_unresolved",
    "rerun_required",
]
FLOAT_COLUMNS = ["q_opt", "delta_opt", "DeltaF"]


def _load_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(r["point_id"]): r for r in rows}


def compare(baseline_csv: Path, candidate_csv: Path, output_dir: Path) -> dict[str, object]:
    base = _load_rows(baseline_csv)
    cand = _load_rows(candidate_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    mismatch_rows: list[dict[str, object]] = []
    for point_id in sorted(set(base) & set(cand)):
        b = base[point_id]
        c = cand[point_id]
        row: dict[str, object] = {
            "point_id": point_id,
            "kT": b.get("kT", ""),
            "JA": b.get("JA", ""),
        }
        any_mismatch = False
        for col in FLAG_COLUMNS:
            match = b.get(col) == c.get(col)
            row[f"{col}_baseline"] = b.get(col, "")
            row[f"{col}_candidate"] = c.get(col, "")
            row[f"{col}_match"] = int(match)
            any_mismatch = any_mismatch or (not match)
        for col in FLOAT_COLUMNS:
            bv = float(b.get(col, "nan"))
            cv = float(c.get(col, "nan"))
            diff = abs(bv - cv)
            row[f"{col}_baseline"] = bv
            row[f"{col}_candidate"] = cv
            row[f"{col}_abs_diff"] = diff
        row["any_flag_mismatch"] = int(any_mismatch)
        rows.append(row)
        if any_mismatch:
            mismatch_rows.append(row)

    def write(path: Path, table: list[dict[str, object]]) -> None:
        if not table:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)

    write(output_dir / "pointwise_comparison.csv", rows)
    write(output_dir / "mismatch_points.csv", mismatch_rows)
    summary = {
        "baseline_csv": str(baseline_csv),
        "candidate_csv": str(candidate_csv),
        "n_common_points": len(rows),
        "n_missing_in_candidate": len(set(base) - set(cand)),
        "n_extra_in_candidate": len(set(cand) - set(base)),
        "flag_mismatch_count": len(mismatch_rows),
        "max_deltaf_abs_diff": float(np.nanmax([r["DeltaF_abs_diff"] for r in rows])) if rows else float("nan"),
        "max_q_opt_abs_diff": float(np.nanmax([r["q_opt_abs_diff"] for r in rows])) if rows else float("nan"),
        "max_delta_opt_abs_diff": float(np.nanmax([r["delta_opt_abs_diff"] for r in rows])) if rows else float("nan"),
    }
    (output_dir / "variant_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "regression_report.md").write_text(
        "# Local-refinement variant comparison\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary.items())
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local-refinement variant outputs against a baseline CSV.")
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/local_refinement_regression"))
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root or _default_run_root()
    summary = compare(
        _run_path(args.baseline_csv, run_root),
        _run_path(args.candidate_csv, run_root),
        _run_path(args.output_dir, run_root),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
