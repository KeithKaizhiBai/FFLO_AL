from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eta_phase_diagram_cuda import EtaPhaseConfig
from ml_phase.exact_oracle import evaluate_points


def _load_points(path: Path, limit: int | None) -> np.ndarray:
    df = pd.read_csv(path)
    points = df[["kT", "JA"]].to_numpy(dtype=np.float64)
    if limit is not None:
        points = points[: int(limit)]
    return points


def _result_frame(prefix: str, result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_id": np.arange(result.kT.size, dtype=int),
            "kT": result.kT,
            "JA": result.JA,
            f"{prefix}_phase_candidate": result.phase_candidate,
            f"{prefix}_q_opt": result.q_opt,
            f"{prefix}_delta_opt": result.delta_opt,
            f"{prefix}_deltaf": result.free_energy_gap_to_normal,
            f"{prefix}_q_expanded": result.q_expanded,
            f"{prefix}_q_unresolved": result.q_unresolved,
            f"{prefix}_trusted_exact": result.trusted_exact,
            f"{prefix}_training_eligible_exact": result.training_eligible_exact,
            f"{prefix}_point_runtime_sec": result.point_total_runtime_sec,
            f"{prefix}_total_q_points_evaluated": result.total_q_points_evaluated,
            f"{prefix}_total_estimated_grid_evaluations": result.total_estimated_grid_evaluations,
            f"{prefix}_incremental_expansion_used": result.incremental_expansion_used,
            f"{prefix}_fallback_full_rescan_used": result.fallback_full_rescan_used,
        }
    )


def run_regression(points_file: Path, output_dir: Path, limit: int | None, device: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = _load_points(points_file, limit)
    cfg = EtaPhaseConfig()
    common = dict(
        points=points,
        cfg=cfg,
        device=device,
        save_every=0,
        output_file=None,
        oracle_mode="robust_al",
        enable_delta_refinement=True,
        enable_q_expansion=True,
    )
    baseline = evaluate_points(**common, enable_incremental_q_expansion=False)
    incremental = evaluate_points(**common, oracle_mode="robust_incremental", enable_incremental_q_expansion=True)

    base_df = _result_frame("baseline", baseline)
    inc_df = _result_frame("incremental", incremental).drop(columns=["kT", "JA"])
    merged = base_df.merge(inc_df, on="point_id", how="inner")
    merged["phase_match"] = merged["baseline_phase_candidate"] == merged["incremental_phase_candidate"]
    merged["trusted_match"] = merged["baseline_trusted_exact"] == merged["incremental_trusted_exact"]
    merged["training_eligible_match"] = (
        merged["baseline_training_eligible_exact"] == merged["incremental_training_eligible_exact"]
    )
    merged["deltaf_abs_diff"] = np.abs(merged["baseline_deltaf"] - merged["incremental_deltaf"])
    merged["q_opt_abs_diff"] = np.abs(merged["baseline_q_opt"] - merged["incremental_q_opt"])
    merged["delta_opt_abs_diff"] = np.abs(merged["baseline_delta_opt"] - merged["incremental_delta_opt"])
    merged.to_csv(output_dir / "incremental_qexpansion_regression.csv", index=False)

    summary = {
        "points_file": str(points_file),
        "n_points": int(points.shape[0]),
        "device": device,
        "phase_mismatch_count": int((~merged["phase_match"]).sum()),
        "trusted_mismatch_count": int((~merged["trusted_match"]).sum()),
        "training_eligible_mismatch_count": int((~merged["training_eligible_match"]).sum()),
        "max_deltaf_abs_diff": float(np.nanmax(merged["deltaf_abs_diff"])) if len(merged) else float("nan"),
        "baseline_runtime_sec_sum": float(np.nansum(merged["baseline_point_runtime_sec"])),
        "incremental_runtime_sec_sum": float(np.nansum(merged["incremental_point_runtime_sec"])),
    }
    (output_dir / "regression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "regression_summary.md").write_text(
        "# Incremental q-expansion regression\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary.items())
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare robust full-rescan and incremental q-expansion outputs.")
    parser.add_argument("--points-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qwindow_incremental_regression"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_regression(args.points_file, args.output_dir, args.limit, args.device)


if __name__ == "__main__":
    main()
