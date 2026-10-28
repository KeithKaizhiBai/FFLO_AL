from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_local_refinement_performance_report import build_performance_report
from scripts import package_local_refinement_variant_suite_hpc as package_script


def _write_pointwise(path: Path, variant: str, *, runtime_scale: float, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rows.append(
            {
                "point_id": i,
                "kT": 0.1 + 0.1 * i,
                "JA": 0.2 + 0.1 * i,
                "phase_candidate": 0 if i == 0 else 1,
                "q_opt": 0.0 if i == 0 else 0.1 * i,
                "delta_opt": 0.0 if i == 0 else 0.2,
                "DeltaF": 1.0e-5 if i == 0 else -1.0e-3,
                "trusted_exact": 1 if i != 2 else 0,
                "training_eligible_exact": 1,
                "q_unresolved": 0,
                "delta_unresolved": 0,
                "rerun_required": 1 if i == 2 else 0,
                "local_minima_detected_count": 4 + i,
                "clustered_basin_count": 3,
                "selected_refine_target_count": 2,
                "basin_clustering_enabled": 1 if variant != "baseline" else 0,
                "basin_clustering_merged_count": i,
                "energy_window_pruning_enabled": 1 if variant == "cluster_energy_window" else 0,
                "energy_window_pruned_count": 1 if variant == "cluster_energy_window" and i == 1 else 0,
                "local_boxes_refined_count": 6 - i,
                "local_refinement_reused_count": 0,
                "point_total_runtime_sec": runtime_scale * (10.0 + i),
                "local_refinement_runtime_sec": runtime_scale * (7.0 + i),
                "source_category": "synthetic",
                "source_run": "test",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_local_box_timing(path: Path, *, runtime_scale: float, n_points: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for point_id in range(n_points):
        for branch_id in range(2):
            rows.append(
                {
                    "point_id": point_id,
                    "branch_id": branch_id,
                    "selection_reason": "global_best" if branch_id == 0 else "within_low_energy_window",
                    "box_runtime_sec": runtime_scale * (1.0 + 0.1 * branch_id),
                    "changed_global_minimum": 1 if point_id == 0 and branch_id == 0 else 0,
                    "changed_phase_label": 0,
                    "near_degenerate_after_refine": 1 if point_id == 2 else 0,
                    "reused_from_previous_scan": 0,
                    "pruned_reason": "" if branch_id == 0 else "outside_energy_window",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, variant: str, *, runtime_scale: float, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mode": "exact",
                "variant_name": variant,
                "n_points": n,
                "total_runtime_sec": runtime_scale * sum(10.0 + i for i in range(n)),
                "local_refinement_runtime_sec": runtime_scale * sum(7.0 + i for i in range(n)),
                "variant_config": package_script.resolve_variant_config(variant),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_result_root(root: Path) -> None:
    runtime_scales = {
        "baseline": 1.0,
        "cluster_only": 0.9,
        "cluster_optional_k3": 0.8,
        "cluster_optional_k2": 0.75,
        "cluster_energy_window": 0.7,
    }
    for variant in package_script.RUNNABLE_VARIANTS:
        variant_root = root / "reports" / "local_refinement_refactor" / "variant_regression" / variant
        scale = runtime_scales[variant]
        _write_pointwise(variant_root / f"{variant}_pointwise.csv", variant, runtime_scale=scale)
        _write_local_box_timing(variant_root / f"{variant}_local_box_timing.csv", runtime_scale=scale)
        _write_manifest(variant_root / "regression_manifest.json", variant, runtime_scale=scale)


def test_build_local_refinement_performance_report(tmp_path):
    result_root = tmp_path / "returned"
    output_dir = result_root / "reports" / "local_refinement_refactor" / "variant_regression" / "performance_report"
    _write_result_root(result_root)

    summary = build_performance_report(result_root, output_dir, variants=package_script.RUNNABLE_VARIANTS)

    assert summary["status"] == "pass"
    assert (output_dir / "runtime_summary.csv").exists()
    assert (output_dir / "local_box_summary.csv").exists()
    assert (output_dir / "performance_report.md").exists()
    assert (output_dir / "performance_summary.json").exists()

    rows = list(csv.DictReader((output_dir / "runtime_summary.csv").open(encoding="utf-8")))
    baseline = next(row for row in rows if row["variant"] == "baseline")
    energy = next(row for row in rows if row["variant"] == "cluster_energy_window")
    assert float(baseline["total_runtime_speedup_vs_baseline"]) == pytest.approx(1.0)
    assert float(energy["total_runtime_speedup_vs_baseline"]) > 1.0
    assert float(energy["energy_window_pruned_count_sum"]) == pytest.approx(1.0)

    box_rows = list(csv.DictReader((output_dir / "local_box_summary.csv").open(encoding="utf-8")))
    baseline_box = next(row for row in box_rows if row["variant"] == "baseline")
    assert int(baseline_box["local_box_rows"]) == 6
    assert float(baseline_box["boxes_per_point"]) == pytest.approx(2.0)
    assert int(baseline_box["changed_global_minimum_rows"]) == 1


def test_build_local_refinement_performance_report_fails_missing_pointwise(tmp_path):
    result_root = tmp_path / "returned"
    output_dir = result_root / "reports" / "local_refinement_refactor" / "variant_regression" / "performance_report"
    _write_result_root(result_root)
    (
        result_root
        / "reports"
        / "local_refinement_refactor"
        / "variant_regression"
        / "cluster_optional_k2"
        / "cluster_optional_k2_pointwise.csv"
    ).unlink()

    with pytest.raises(FileNotFoundError, match="missing pointwise CSV"):
        build_performance_report(result_root, output_dir, variants=package_script.RUNNABLE_VARIANTS)

    assert (output_dir / "performance_summary.json").exists()
