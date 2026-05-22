from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.config import ActiveLearningConfig
from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC
from ml_phase.stop_controller import StopConfig, evaluate_stop


def _toy_grid(cfg: ActiveLearningConfig) -> tuple[np.ndarray, tuple[int, int]]:
    kt = np.linspace(cfg.kt_min, cfg.kt_max, cfg.n_kt_candidates)
    ja = np.linspace(cfg.ja_min, cfg.ja_max, cfg.n_ja_candidates)
    kt_mesh, ja_mesh = np.meshgrid(kt, ja, indexing="xy")
    return np.stack([kt_mesh.ravel(), ja_mesh.ravel()], axis=1), ja_mesh.shape


def _stable_phase_map(shape: tuple[int, int]) -> np.ndarray:
    n_ja, n_kt = shape
    phase = np.full((n_ja, n_kt), PHASE_FFLO, dtype=np.int64)
    phase[:, : max(1, n_kt // 3)] = PHASE_NORMAL
    phase[:, max(1, n_kt // 3) : max(2, 2 * n_kt // 3)] = PHASE_UNIFORM_SC
    return phase.ravel()


def _phase_to_exact(phase: np.ndarray, delta_eps: float, q_eps: float) -> tuple[np.ndarray, np.ndarray]:
    phase = np.asarray(phase, dtype=np.int64)
    delta = np.full(phase.shape, 10.0 * delta_eps, dtype=np.float64)
    q = np.full(phase.shape, 2.0 * q_eps, dtype=np.float64)
    delta[phase == PHASE_NORMAL] = 0.0
    q[phase == PHASE_NORMAL] = 0.0
    q[phase == PHASE_UNIFORM_SC] = 0.0
    return delta, q


def _write_dataset(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    y_reg = np.zeros((n, 5), dtype=np.float64)
    np.savez(
        path,
        x=points.astype(np.float64),
        y_reg=y_reg,
        y_phase=np.zeros(n, dtype=np.int64),
        y_eta_sign=np.zeros(n, dtype=np.int64),
        y_strong_diode=np.zeros(n, dtype=np.int64),
    )


def _write_iteration(
    run_dir: Path,
    iteration: int,
    cfg: ActiveLearningConfig,
    phase_pred: np.ndarray,
    a0_mean: float,
    qedge: bool = False,
    rerun: bool = False,
    eta_offset: float = 0.0,
) -> Path:
    points, shape = _toy_grid(cfg)
    iter_dir = run_dir / f"iter{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        iter_dir / f"monitor_predictions_iter{iteration:03d}.npz",
        grid_points=points,
        full_shape=np.asarray(shape, dtype=np.int64),
        phase_pred=phase_pred.astype(np.int64),
        A0_main=np.full(points.shape[0], a0_mean, dtype=np.float64),
        score=np.full(points.shape[0], a0_mean, dtype=np.float64),
        candidate_mask=np.ones(points.shape[0], dtype=np.int8),
    )
    selected_idx = np.linspace(0, points.shape[0] - 1, 8, dtype=np.int64)
    selected = points[selected_idx]
    selected_phase = phase_pred[selected_idx]
    pd.DataFrame(
        {
            "selection_rank": np.arange(1, selected.shape[0] + 1),
            "selection_source": ["acquisition"] * selected.shape[0],
            "selection_pool": ["acquisition"] * selected.shape[0],
            "grid_index": selected_idx,
            "kT": selected[:, 0],
            "JA": selected[:, 1],
            "A0_main": np.full(selected.shape[0], a0_mean),
            "predicted_phase_before_exact": selected_phase,
        }
    ).to_csv(iter_dir / "selected_points_by_pool.csv", index=False)

    delta, q = _phase_to_exact(selected_phase, cfg.delta_eps, cfg.q_eps)
    n = selected.shape[0]
    np.savez(
        iter_dir / f"exact_merged_iter{iteration:03d}.npz",
        kT=selected[:, 0],
        JA=selected[:, 1],
        delta_opt=delta,
        q_opt=q,
        eta=np.full(n, eta_offset, dtype=np.float64),
        ic_plus=np.zeros(n, dtype=np.float64),
        ic_minus=np.zeros(n, dtype=np.float64),
        q_expanded=np.full(n, int(qedge), dtype=np.int8),
        q_unresolved=np.zeros(n, dtype=np.int8),
        q_edge_hit=np.full(n, int(qedge), dtype=np.int8),
        needs_rerun_exact=np.full(n, int(rerun), dtype=np.int8),
        training_eligible_exact=np.full(n, int(not rerun), dtype=np.int8),
    )
    dataset_path = run_dir / f"dataset_iter{iteration + 1:03d}.npz"
    _write_dataset(dataset_path, points)
    return dataset_path


def _stop_config() -> StopConfig:
    return StopConfig(
        min_iterations=2,
        patience=2,
        max_iterations=20,
        warmup_reference_iters=2,
        map_tol=0.002,
        boundary_shift_tol=1e-12,
        surprise_tol=0.05,
        selected_a0_ratio_tol=0.15,
        qedge_rate_tol=0.01,
        rerun_rate_tol=0.01,
        coverage_tol=0.07,
    )


def test_stable_boundaries_stop_after_patience() -> None:
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        cfg = ActiveLearningConfig(n_kt_candidates=9, n_ja_candidates=7)
        _, shape = _toy_grid(cfg)
        phase = _stable_phase_map(shape)
        stop_cfg = _stop_config()
        stopped = False
        for i, a0 in enumerate([1.0, 1.0, 0.1]):
            dataset = _write_iteration(run_dir, i, cfg, phase, a0_mean=a0)
            result = evaluate_stop(run_dir, i, dataset, cfg, stop_cfg)
            stopped = bool(result["stop"])
        assert stopped, "stable phase map and boundaries should stop after patience"


def test_soft_candidates_do_not_block_convergence_stop() -> None:
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        cfg = ActiveLearningConfig(n_kt_candidates=9, n_ja_candidates=7)
        _, shape = _toy_grid(cfg)
        phase = _stable_phase_map(shape)
        stop_cfg = _stop_config()
        for i, a0 in enumerate([1.0, 1.0, 0.1]):
            dataset = _write_iteration(run_dir, i, cfg, phase, a0_mean=a0)
            result = evaluate_stop(run_dir, i, dataset, cfg, stop_cfg)
        assert result["metrics"]["selected_A0_mean"] is not None
        assert result["stop"], "converged metrics should stop even when selected candidates exist"


def test_eta_change_does_not_block_stop() -> None:
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        cfg = ActiveLearningConfig(n_kt_candidates=9, n_ja_candidates=7)
        _, shape = _toy_grid(cfg)
        phase = _stable_phase_map(shape)
        stop_cfg = _stop_config()
        for i, eta in enumerate([0.0, 10.0, -10.0]):
            dataset = _write_iteration(run_dir, i, cfg, phase, a0_mean=1.0 if i < 2 else 0.1, eta_offset=eta)
            result = evaluate_stop(run_dir, i, dataset, cfg, stop_cfg)
        assert result["stop"], "eta-only response changes must not enter the main convergence stop"


def test_high_qedge_rate_becomes_cleanup_warning() -> None:
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        cfg = ActiveLearningConfig(n_kt_candidates=9, n_ja_candidates=7)
        _, shape = _toy_grid(cfg)
        phase = _stable_phase_map(shape)
        stop_cfg = _stop_config()
        for i, a0 in enumerate([1.0, 1.0, 0.1, 0.1]):
            dataset = _write_iteration(run_dir, i, cfg, phase, a0_mean=a0, qedge=True)
            result = evaluate_stop(run_dir, i, dataset, cfg, stop_cfg)
        assert result["stop"], "main phase-boundary convergence should stop even if q-edge cleanup remains"
        assert result["numerical_cleanup_warning"], "high q-edge rate should be reported as a cleanup warning"


def test_bad_boundary_coverage_is_one_main_condition() -> None:
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        cfg = ActiveLearningConfig(n_kt_candidates=9, n_ja_candidates=7)
        points, shape = _toy_grid(cfg)
        phase = _stable_phase_map(shape)
        stop_cfg = _stop_config()
        for i, a0 in enumerate([1.0, 1.0, 0.1, 0.1]):
            dataset = _write_iteration(run_dir, i, cfg, phase, a0_mean=a0)
            _write_dataset(dataset, points[:1])
            result = evaluate_stop(run_dir, i, dataset, cfg, stop_cfg)
        assert not result["conditions"]["C5_boundary_coverage_p95"]
        assert result["stop"], "coverage is one of five main conditions, not a mandatory gate"


def main() -> None:
    test_stable_boundaries_stop_after_patience()
    test_soft_candidates_do_not_block_convergence_stop()
    test_eta_change_does_not_block_stop()
    test_high_qedge_rate_becomes_cleanup_warning()
    test_bad_boundary_coverage_is_one_main_condition()
    print("stop-controller checks passed")


if __name__ == "__main__":
    main()
