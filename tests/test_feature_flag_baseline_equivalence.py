from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import numpy as np

from ml_phase import exact_oracle
from ml_phase.exact_oracle import ConfirmedPoint, evaluate_points
from scripts.run_local_refinement_fixed_point_regression import resolve_variant_config, run_regression


def _confirmed_point(kT: float, JA: float) -> ConfirmedPoint:
    return ConfirmedPoint(
        kT=float(kT),
        JA=float(JA),
        eta=0.0,
        q_opt=0.0,
        delta_opt=0.2,
        ic_plus=0.0,
        ic_minus=0.0,
        phase_candidate=1,
        q_status=1,
        q_min=-0.5,
        q_max=0.5,
        n_q=10,
        q_index=5,
        q_edge_distance=0.5,
        q_edge_hit=0,
        q_refinement_level=0,
        q_expanded=0,
        q_unresolved=0,
        delta_status=0,
        delta_min=0.0,
        delta_max=1.0,
        n_delta=10,
        n_delta_refined=0,
        delta_refinement_level=0,
        delta_boundary_ambiguous=0,
        delta_refined=0,
        delta_unresolved=0,
        free_energy_gap_to_normal=-1.0e-3,
        positive_delta_gap=float("nan"),
        positive_delta_checked=0,
        exact_status_code=0,
        exact_status_name="trusted",
        trusted_exact=1,
    )


def _write_points_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "source_run", "kT", "JA"])
        writer.writeheader()
        writer.writerow({"category": "stub", "source_run": "test", "kT": "0.1", "JA": "0.2"})


def test_exact_oracle_default_feature_flags_match_baseline_path(monkeypatch):
    seen_kwargs: dict[str, object] = {}

    def fake_confirm_one_point(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return _confirmed_point(args[0], args[1])

    monkeypatch.setattr(exact_oracle, "_confirm_one_point", fake_confirm_one_point)
    evaluate_points(points=np.array([[0.1, 0.2]], dtype=float), output_file=None, save_every=0)

    assert seen_kwargs["enable_basin_clustering"] is False
    assert seen_kwargs["enable_selective_refinement"] is False
    assert seen_kwargs["max_optional_refined_basins"] == 3
    assert seen_kwargs["mandatory_basins_can_exceed_cap"] is True
    assert seen_kwargs["energy_window_pruning_enabled"] is False
    assert seen_kwargs["local_refine_pruning_energy_window"] is None
    assert seen_kwargs["local_box_records"] is None


def test_regression_runner_baseline_variant_passes_feature_flags_disabled(tmp_path, monkeypatch):
    points_csv = tmp_path / "fixed_points.csv"
    _write_points_csv(points_csv)

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

    def evaluate_points_stub(points, cfg, **kwargs):
        assert isinstance(cfg, EtaPhaseConfig)
        seen_kwargs.update(kwargs)
        return Result(points)

    fake_exact.evaluate_points = evaluate_points_stub
    monkeypatch.setitem(sys.modules, "eta_phase_diagram_cuda", fake_eta)
    monkeypatch.setitem(sys.modules, "ml_phase.exact_oracle", fake_exact)

    out_dir = tmp_path / "baseline_exact_stub"
    run_regression(
        points_csv,
        out_dir,
        limit=None,
        device="cpu",
        dry_run=False,
        variant_name="baseline",
        run_root=tmp_path,
    )

    assert seen_kwargs["enable_basin_clustering"] is False
    assert seen_kwargs["enable_selective_refinement"] is False
    assert seen_kwargs["energy_window_pruning_enabled"] is False
    assert seen_kwargs["enable_local_box_instrumentation"] is False


def test_baseline_variant_config_keeps_all_local_refinement_flags_off():
    baseline = resolve_variant_config("baseline")

    assert baseline == {
        "enable_basin_clustering": False,
        "enable_selective_refinement": False,
        "energy_window_pruning_enabled": False,
    }
