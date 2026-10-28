from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from scripts.build_local_refinement_regression_points import build_points
from scripts.run_local_refinement_fixed_point_regression import resolve_variant_config, run_regression


def _write_fake_shard(path: Path) -> None:
    path.parent.mkdir(parents=True)
    n = 5
    np.savez(
        path,
        kT=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        JA=np.array([0.0, 0.2, 0.4, 0.8, 1.2]),
        phase_candidate=np.array([0, 1, 1, 0, 1]),
        q_opt=np.array([0.0, 0.0, 0.2, 0.0, 0.4]),
        delta_opt=np.array([0.0, 0.3, 0.2, 0.0, 0.1]),
        free_energy_gap_to_normal=np.array([1e-5, -1e-3, -2e-3, 1e-9, -5e-4]),
        trusted_exact=np.array([1, 1, 1, 0, 1]),
        training_eligible_exact=np.array([1, 1, 1, 1, 1]),
        rerun_required=np.array([0, 0, 0, 0, 1]),
        q_expanded=np.array([0, 0, 1, 0, 1]),
        q_unresolved=np.array([0, 0, 0, 0, 1]),
        q_status=np.array([0, 1, 2, 0, 3]),
        qopt_edge_hit_initial=np.array([0, 0, 1, 0, 1]),
        delta_boundary_ambiguous=np.array([0, 0, 0, 1, 1]),
        delta_boundary_band_normal=np.array([0, 0, 0, 1, 0]),
        phase_changed_after_q_expansion=np.array([0, 0, 1, 0, 1]),
        near_degenerate_branch_count=np.array([0, 0, 0, 1, 2]),
        local_boxes_refined_count=np.full(n, 6),
        local_minima_detected_count=np.arange(n) + 1,
    )


def test_regression_point_builder_and_dry_run(tmp_path):
    run_dir = tmp_path / "active_boundary_discovery_robust_incremental_full_acq_v1"
    _write_fake_shard(run_dir / "iter004" / "exact_merged_iter004.npz")
    points_csv = tmp_path / "fixed_points.csv"

    rows = build_points([run_dir], points_csv, max_per_category=3, q_eps=1e-2)
    assert points_csv.exists()
    assert rows
    categories = {str(r["category"]) for r in rows}
    assert "stable_normal_interior" in categories
    assert "clean_uniform_sc" in categories
    assert "clean_fflo" in categories

    out_dir = tmp_path / "dry_run"
    run_regression(points_csv, out_dir, limit=None, device="cpu", dry_run=True, variant_name="baseline")
    manifest = json.loads((out_dir / "regression_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "dry_run"
    assert manifest["n_points"] == len(list(csv.DictReader(points_csv.open(encoding="utf-8"))))


def test_fixed_point_runner_resolves_explicit_variant_flags(tmp_path):
    points_csv = tmp_path / "fixed_points.csv"
    with points_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "source_run", "kT", "JA"])
        writer.writeheader()
        writer.writerow({"category": "stub", "source_run": "test", "kT": "0.1", "JA": "0.2"})

    out_dir = tmp_path / "cluster_energy_window_dry_run"
    run_regression(
        points_csv,
        out_dir,
        limit=None,
        device="cpu",
        dry_run=True,
        variant_name="cluster_energy_window",
    )
    manifest = json.loads((out_dir / "regression_manifest.json").read_text(encoding="utf-8"))

    assert manifest["variant_name"] == "cluster_energy_window"
    assert manifest["variant_config"]["enable_basin_clustering"] is True
    assert manifest["variant_config"]["enable_selective_refinement"] is True
    assert manifest["variant_config"]["max_optional_refined_basins"] == 3
    assert manifest["variant_config"]["energy_window_pruning_enabled"] is True


def test_fixed_point_runner_resolves_rank_and_cap_variant_flags(tmp_path):
    points_csv = tmp_path / "fixed_points.csv"
    with points_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "source_run", "kT", "JA"])
        writer.writeheader()
        writer.writerow({"category": "stub", "source_run": "test", "kT": "0.1", "JA": "0.2"})

    out_dir = tmp_path / "rank_and_cap_k3_dry_run"
    run_regression(
        points_csv,
        out_dir,
        limit=None,
        device="cpu",
        dry_run=True,
        variant_name="rank_and_cap_k3",
    )
    manifest = json.loads((out_dir / "regression_manifest.json").read_text(encoding="utf-8"))

    assert manifest["variant_name"] == "rank_and_cap_k3"
    assert manifest["variant_config"]["enable_basin_clustering"] is True
    assert manifest["variant_config"]["enable_selective_refinement"] is True
    assert manifest["variant_config"]["mandatory_basins_can_exceed_cap"] is False
    assert manifest["variant_config"]["high_risk_overflow_policy"] == "rank_and_cap"
    assert manifest["variant_config"]["max_edge_risk_basins"] == 1
    assert manifest["variant_config"]["max_delta_near_eps_basins"] == 2
    assert manifest["variant_config"]["max_near_degenerate_basins"] == 2


def test_fixed_point_runner_rejects_unintegrated_reuse_variant():
    with pytest.raises(ValueError, match="not runnable yet"):
        resolve_variant_config("cluster_energy_reuse")


def test_fixed_point_runner_passes_variant_config_to_exact_oracle(tmp_path, monkeypatch):
    points_csv = tmp_path / "fixed_points.csv"
    with points_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "source_run", "kT", "JA"])
        writer.writeheader()
        writer.writerow({"category": "stub", "source_run": "test", "kT": "0.1", "JA": "0.2"})

    fake_eta = types.ModuleType("eta_phase_diagram_cuda")

    class EtaPhaseConfig:
        pass

    fake_eta.EtaPhaseConfig = EtaPhaseConfig

    seen_kwargs: dict[str, object] = {}
    fake_exact = types.ModuleType("ml_phase.exact_oracle")

    class Result:
        def __init__(self, points: np.ndarray):
            n = int(points.shape[0])
            self.kT = points[:, 0]
            self.JA = points[:, 1]
            self.phase_candidate = np.zeros(n, dtype=int)
            self.q_opt = np.zeros(n)
            self.delta_opt = np.zeros(n)
            self.free_energy_gap_to_normal = np.zeros(n)
            self.trusted_exact = np.ones(n, dtype=int)
            self.training_eligible_exact = np.ones(n, dtype=int)
            self.q_unresolved = np.zeros(n, dtype=int)
            self.delta_unresolved = np.zeros(n, dtype=int)
            self.rerun_required = np.zeros(n, dtype=int)
            self.local_minima_detected_count = np.zeros(n, dtype=int)
            self.local_boxes_refined_count = np.zeros(n, dtype=int)
            self.local_refinement_reused_count = np.zeros(n, dtype=int)
            self.point_total_runtime_sec = np.zeros(n)
            self.local_refinement_runtime_sec = np.zeros(n)

    def evaluate_points(points, cfg, **kwargs):
        assert isinstance(cfg, EtaPhaseConfig)
        seen_kwargs.update(kwargs)
        return Result(points)

    fake_exact.evaluate_points = evaluate_points
    monkeypatch.setitem(sys.modules, "eta_phase_diagram_cuda", fake_eta)
    monkeypatch.setitem(sys.modules, "ml_phase.exact_oracle", fake_exact)

    out_dir = tmp_path / "exact_stub"
    run_regression(
        points_csv,
        out_dir,
        limit=None,
        device="cpu",
        dry_run=False,
        variant_name="cluster_optional_k2",
    )

    assert seen_kwargs["enable_basin_clustering"] is True
    assert seen_kwargs["enable_selective_refinement"] is True
    assert seen_kwargs["max_optional_refined_basins"] == 2
    assert seen_kwargs["mandatory_basins_can_exceed_cap"] is True
    assert seen_kwargs["energy_window_pruning_enabled"] is False
    manifest = json.loads((out_dir / "regression_manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant_config"]["max_optional_refined_basins"] == 2
