from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANT_CONFIGS: dict[str, dict[str, object]] = {
    "baseline": {
        "enable_basin_clustering": False,
        "enable_selective_refinement": False,
        "energy_window_pruning_enabled": False,
    },
    "cluster_only": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": False,
        "energy_window_pruning_enabled": False,
    },
    "cluster_optional_k3": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": True,
        "max_optional_refined_basins": 3,
        "mandatory_basins_can_exceed_cap": True,
        "energy_window_pruning_enabled": False,
    },
    "cluster_optional_k2": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": True,
        "max_optional_refined_basins": 2,
        "mandatory_basins_can_exceed_cap": True,
        "energy_window_pruning_enabled": False,
    },
    "cluster_energy_window": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": True,
        "max_optional_refined_basins": 3,
        "mandatory_basins_can_exceed_cap": True,
        "energy_window_pruning_enabled": True,
        "local_refine_pruning_energy_window": None,
    },
    "rank_and_cap_k3": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": True,
        "max_optional_refined_basins": 3,
        "mandatory_basins_can_exceed_cap": False,
        "high_risk_overflow_policy": "rank_and_cap",
        "max_edge_risk_basins": 1,
        "max_delta_near_eps_basins": 2,
        "max_near_degenerate_basins": 2,
        "energy_window_pruning_enabled": False,
    },
    "rank_and_cap_k2": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": True,
        "max_optional_refined_basins": 2,
        "mandatory_basins_can_exceed_cap": False,
        "high_risk_overflow_policy": "rank_and_cap",
        "max_edge_risk_basins": 1,
        "max_delta_near_eps_basins": 2,
        "max_near_degenerate_basins": 2,
        "energy_window_pruning_enabled": False,
    },
    "rank_and_cap_energy_window": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": True,
        "max_optional_refined_basins": 3,
        "mandatory_basins_can_exceed_cap": False,
        "high_risk_overflow_policy": "rank_and_cap",
        "max_edge_risk_basins": 1,
        "max_delta_near_eps_basins": 2,
        "max_near_degenerate_basins": 2,
        "energy_window_pruning_enabled": True,
        "local_refine_pruning_energy_window": None,
    },
}

PROTOTYPE_VARIANTS: dict[str, str] = {
    "cluster_energy_reuse": (
        "Stage 5 branch reuse is currently a tested decision prototype only; "
        "it is not integrated into the production exact-oracle local-refinement loop."
    ),
}


def supported_variant_names() -> list[str]:
    return sorted(VARIANT_CONFIGS)


def resolve_variant_config(variant_name: str) -> dict[str, object]:
    if variant_name in VARIANT_CONFIGS:
        return dict(VARIANT_CONFIGS[variant_name])
    if variant_name in PROTOTYPE_VARIANTS:
        raise ValueError(f"Variant {variant_name!r} is not runnable yet. {PROTOTYPE_VARIANTS[variant_name]}")
    known = ", ".join(supported_variant_names())
    raise ValueError(f"Unknown local-refinement regression variant {variant_name!r}. Supported variants: {known}.")


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


def _package_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _run_path(path: Path, run_root: Path) -> Path:
    return path if path.is_absolute() else run_root / path


def _load_points(path: Path, limit: int | None) -> tuple[np.ndarray, list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[: int(limit)]
    points = np.array([[float(r["kT"]), float(r["JA"])] for r in rows], dtype=np.float64)
    return points, rows


def _write_dry_run(
    output_dir: Path,
    points_file: Path,
    rows: list[dict[str, str]],
    variant_name: str,
    variant_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    categories: dict[str, int] = {}
    for row in rows:
        category = row.get("category", "unknown")
        categories[category] = categories.get(category, 0) + 1
    manifest = {
        "mode": "dry_run",
        "variant_name": variant_name,
        "points_file": str(points_file),
        "n_points": len(rows),
        "category_counts": categories,
        "variant_config": variant_config,
        "note": "No exact-oracle calculation was run.",
    }
    (output_dir / "regression_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "regression_summary.md").write_text(
        "# Local-refinement fixed-point regression dry run\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in manifest.items())
        + "\n",
        encoding="utf-8",
    )


def _result_frame(result) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = int(result.kT.size)

    def int_value(name: str, index: int, default: int = 0) -> int:
        values = getattr(result, name, None)
        if values is None:
            return int(default)
        return int(values[index])

    for i in range(n):
        rows.append(
            {
                "point_id": i,
                "kT": float(result.kT[i]),
                "JA": float(result.JA[i]),
                "phase_candidate": int(result.phase_candidate[i]),
                "q_opt": float(result.q_opt[i]),
                "delta_opt": float(result.delta_opt[i]),
                "DeltaF": float(result.free_energy_gap_to_normal[i]),
                "trusted_exact": int(result.trusted_exact[i]),
                "training_eligible_exact": int(result.training_eligible_exact[i]),
                "q_unresolved": int(result.q_unresolved[i]),
                "delta_unresolved": int(result.delta_unresolved[i]),
                "rerun_required": int(result.rerun_required[i]),
                "local_minima_detected_count": int_value("local_minima_detected_count", i),
                "clustered_basin_count": int_value("clustered_basin_count", i),
                "selected_refine_target_count": int_value("selected_refine_target_count", i),
                "basin_clustering_enabled": int_value("basin_clustering_enabled", i),
                "basin_clustering_merged_count": int_value("basin_clustering_merged_count", i),
                "energy_window_pruning_enabled": int_value("energy_window_pruning_enabled", i),
                "energy_window_pruned_count": int_value("energy_window_pruned_count", i),
                "local_boxes_refined_count": int_value("local_boxes_refined_count", i),
                "local_refinement_reused_count": int_value("local_refinement_reused_count", i),
                "point_total_runtime_sec": float(result.point_total_runtime_sec[i]),
                "local_refinement_runtime_sec": float(result.local_refinement_runtime_sec[i]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_regression(
    points_file: Path,
    output_dir: Path,
    limit: int | None,
    device: str,
    dry_run: bool,
    variant_name: str,
    enable_local_box_instrumentation: bool = False,
    run_root: Path | None = None,
) -> None:
    run_root = run_root or _default_run_root()
    points_file = _package_path(points_file)
    output_dir = _run_path(output_dir, run_root)
    variant_config = resolve_variant_config(variant_name)
    points, source_rows = _load_points(points_file, limit)
    if dry_run:
        _write_dry_run(output_dir, points_file, source_rows, variant_name=variant_name, variant_config=variant_config)
        return

    from eta_phase_diagram_cuda import EtaPhaseConfig
    from ml_phase.exact_oracle import evaluate_points

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
        enable_local_box_instrumentation=bool(enable_local_box_instrumentation),
        local_box_output_file=(
            output_dir / f"{variant_name}_local_box_timing.csv"
            if enable_local_box_instrumentation
            else None
        ),
        **variant_config,
    )
    rows = _result_frame(result)
    for row, src in zip(rows, source_rows):
        row["source_category"] = src.get("category", "unknown")
        row["source_run"] = src.get("source_run", "unknown")
    _write_csv(output_dir / f"{variant_name}_pointwise.csv", rows)
    summary = {
        "mode": "exact",
        "variant_name": variant_name,
        "points_file": str(points_file),
        "n_points": int(points.shape[0]),
        "device": device,
        "phase_counts": {str(k): int(np.sum(result.phase_candidate == k)) for k in np.unique(result.phase_candidate)},
        "trusted_count": int(np.sum(result.trusted_exact.astype(bool))),
        "training_eligible_count": int(np.sum(result.training_eligible_exact.astype(bool))),
        "rerun_required_count": int(np.sum(result.rerun_required.astype(bool))),
        "total_runtime_sec": float(np.nansum(result.point_total_runtime_sec)),
        "local_refinement_runtime_sec": float(np.nansum(result.local_refinement_runtime_sec)),
        "variant_config": variant_config,
        "enable_local_box_instrumentation": bool(enable_local_box_instrumentation),
        "local_box_output_file": str(output_dir / f"{variant_name}_local_box_timing.csv")
        if enable_local_box_instrumentation
        else "N/A",
    }
    (output_dir / "regression_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "regression_summary.md").write_text(
        "# Local-refinement fixed-point regression\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary.items())
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local-refinement fixed-point regression.")
    parser.add_argument("--points-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/local_refinement_refactor/stage_00_baseline/regression"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant-name", type=str, default="baseline")
    parser.add_argument("--enable-local-box-instrumentation", action="store_true")
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root or _default_run_root()
    points_file = _package_path(args.points_file)
    output_dir = _run_path(args.output_dir, run_root)
    run_regression(
        points_file,
        output_dir,
        args.limit,
        args.device,
        args.dry_run,
        args.variant_name,
        enable_local_box_instrumentation=bool(args.enable_local_box_instrumentation),
        run_root=run_root,
    )
    print(f"Wrote local-refinement regression outputs: {output_dir}")


if __name__ == "__main__":
    main()
