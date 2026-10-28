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
    return points[:limit] if limit is not None else points


def _runtime_table(result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_id": np.arange(result.kT.size, dtype=int),
            "kT": result.kT,
            "JA": result.JA,
            "phase_candidate": result.phase_candidate,
            "q_expansion_count": result.q_expansion_count,
            "q_expansion_directions": result.q_expansion_directions,
            "q_unresolved": result.q_unresolved,
            "delta_refinement_triggered": result.delta_refinement_triggered,
            "trusted_exact": result.trusted_exact,
            "training_eligible_exact": result.training_eligible_exact,
            "point_total_runtime_sec": result.point_total_runtime_sec,
            "base_scan_runtime_sec": result.base_scan_runtime_sec,
            "q_expansion_runtime_sec": result.q_expansion_runtime_sec,
            "delta_refinement_runtime_sec": result.delta_refinement_runtime_sec,
            "local_refinement_runtime_sec": result.local_refinement_runtime_sec,
            "fallback_full_rescan_runtime_sec": result.fallback_full_rescan_runtime_sec,
            "total_q_points_evaluated": result.total_q_points_evaluated,
            "total_estimated_grid_evaluations": result.total_estimated_grid_evaluations,
            "incremental_expansion_used": result.incremental_expansion_used,
            "fallback_full_rescan_used": result.fallback_full_rescan_used,
            "fallback_full_rescan_reason": result.fallback_full_rescan_reason,
        }
    )


def run_benchmark(points_file: Path, output_dir: Path, limit: int | None, device: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = _load_points(points_file, limit)
    result = evaluate_points(
        points=points,
        cfg=EtaPhaseConfig(),
        device=device,
        save_every=0,
        output_file=None,
        oracle_mode="robust_incremental",
        enable_q_expansion=True,
        enable_delta_refinement=True,
        enable_incremental_q_expansion=True,
    )
    table = _runtime_table(result)
    table.to_csv(output_dir / "qwindow_incremental_benchmark_points.csv", index=False)
    summary = {
        "points_file": str(points_file),
        "n_points": int(points.shape[0]),
        "device": device,
        "total_runtime_sec": float(np.nansum(result.point_total_runtime_sec)),
        "mean_runtime_sec": float(np.nanmean(result.point_total_runtime_sec)) if result.kT.size else float("nan"),
        "q_expanded_count": int(np.sum(result.q_expanded)),
        "incremental_expansion_used_count": int(np.sum(result.incremental_expansion_used)),
        "fallback_full_rescan_used_count": int(np.sum(result.fallback_full_rescan_used)),
        "total_q_points_evaluated": int(np.sum(result.total_q_points_evaluated)),
        "total_estimated_grid_evaluations": int(np.sum(result.total_estimated_grid_evaluations)),
    }
    (output_dir / "qwindow_incremental_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "qwindow_incremental_benchmark.md").write_text(
        "# q-window incremental benchmark\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary.items())
        + "\n\nThis benchmark is report-only and does not append active-learning data.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark robust incremental q-window expansion on selected exact points.")
    parser.add_argument("--points-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qwindow_incremental_benchmark"))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_benchmark(args.points_file, args.output_dir, args.limit, args.device)


if __name__ == "__main__":
    main()
