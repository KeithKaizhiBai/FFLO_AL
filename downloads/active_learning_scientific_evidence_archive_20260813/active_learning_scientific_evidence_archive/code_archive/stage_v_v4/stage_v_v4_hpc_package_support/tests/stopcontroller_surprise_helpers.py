from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_phase.config import ActiveLearningConfig
from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC
from ml_phase.stop_controller import StopConfig, evaluate_stop


def toy_config() -> ActiveLearningConfig:
    return ActiveLearningConfig(n_kt_candidates=9, n_ja_candidates=7)


def toy_grid(cfg: ActiveLearningConfig) -> tuple[np.ndarray, tuple[int, int]]:
    kt = np.linspace(cfg.kt_min, cfg.kt_max, int(cfg.n_kt_candidates))
    ja = np.linspace(cfg.ja_min, cfg.ja_max, int(cfg.n_ja_candidates))
    kt_mesh, ja_mesh = np.meshgrid(kt, ja, indexing="xy")
    return np.stack([kt_mesh.ravel(), ja_mesh.ravel()], axis=1), ja_mesh.shape


def stable_phase_map(shape: tuple[int, int]) -> np.ndarray:
    n_ja, n_kt = shape
    phase = np.full((n_ja, n_kt), PHASE_FFLO, dtype=np.int64)
    phase[:, : max(1, n_kt // 3)] = PHASE_NORMAL
    phase[:, max(1, n_kt // 3) : max(2, 2 * n_kt // 3)] = PHASE_UNIFORM_SC
    return phase.ravel()


def phase_to_exact(phase: np.ndarray, cfg: ActiveLearningConfig) -> tuple[np.ndarray, np.ndarray]:
    phase = np.asarray(phase, dtype=np.int64)
    delta = np.full(phase.shape, 10.0 * float(cfg.delta_eps), dtype=np.float64)
    q = np.full(phase.shape, 2.0 * float(cfg.q_eps), dtype=np.float64)
    delta[phase == PHASE_NORMAL] = 0.0
    q[phase == PHASE_NORMAL] = 0.0
    q[phase == PHASE_UNIFORM_SC] = 0.0
    return delta, q


def write_dataset(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(points.shape[0])
    np.savez(
        path,
        x=points.astype(np.float64),
        y_reg=np.zeros((n, 5), dtype=np.float64),
        y_phase=np.zeros(n, dtype=np.int64),
        y_eta_sign=np.zeros(n, dtype=np.int64),
        y_strong_diode=np.zeros(n, dtype=np.int64),
    )


def stop_config(mode: str = "all_selected", min_denominator: int = 1, min_fraction: float = 0.0) -> StopConfig:
    return StopConfig(
        min_iterations=1,
        patience=1,
        max_iterations=20,
        warmup_reference_iters=1,
        map_tol=1.0,
        boundary_shift_tol=1.0,
        surprise_tol=0.05,
        selected_a0_ratio_tol=0.15,
        qedge_rate_tol=0.01,
        rerun_rate_tol=0.01,
        coverage_tol=1.0,
        required_pass_count=1,
        stop_surprise_mode=mode,
        trusted_surprise_min_denominator=int(min_denominator),
        trusted_surprise_min_fraction=float(min_fraction),
    )


def write_iteration(
    run_dir: Path,
    iteration: int,
    cfg: ActiveLearningConfig,
    rows: list[dict[str, object]],
) -> Path:
    points, shape = toy_grid(cfg)
    iter_dir = run_dir / f"iter{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    phase_map = stable_phase_map(shape)
    np.savez(
        iter_dir / f"monitor_predictions_iter{iteration:03d}.npz",
        grid_points=points,
        full_shape=np.asarray(shape, dtype=np.int64),
        phase_pred=phase_map.astype(np.int64),
        A0_main=np.full(points.shape[0], 0.1, dtype=np.float64),
        score=np.full(points.shape[0], 0.1, dtype=np.float64),
        candidate_mask=np.ones(points.shape[0], dtype=np.int8),
    )

    selected_idx = np.arange(len(rows), dtype=np.int64)
    selected_points = points[selected_idx]
    pred_phase = np.asarray([int(row.get("pred", PHASE_NORMAL)) for row in rows], dtype=np.int64)
    exact_phase = np.asarray([int(row.get("exact", row.get("pred", PHASE_NORMAL))) for row in rows], dtype=np.int64)
    pd.DataFrame(
        {
            "selection_rank": np.arange(1, len(rows) + 1),
            "selection_source": ["acquisition"] * len(rows),
            "selection_pool": ["acquisition"] * len(rows),
            "grid_index": selected_idx,
            "kT": selected_points[:, 0],
            "JA": selected_points[:, 1],
            "A0_main": np.full(len(rows), 0.1),
            "predicted_phase_before_exact": pred_phase,
        }
    ).to_csv(iter_dir / "selected_points_by_pool.csv", index=False)

    delta, q = phase_to_exact(exact_phase, cfg)
    def flag(name: str, default: bool) -> np.ndarray:
        return np.asarray([bool(row.get(name, default)) for row in rows], dtype=np.int8)

    np.savez(
        iter_dir / f"exact_merged_iter{iteration:03d}.npz",
        kT=selected_points[:, 0],
        JA=selected_points[:, 1],
        delta_opt=delta,
        q_opt=q,
        eta=np.zeros(len(rows), dtype=np.float64),
        ic_plus=np.zeros(len(rows), dtype=np.float64),
        ic_minus=np.zeros(len(rows), dtype=np.float64),
        q_expanded=flag("q_expanded", False),
        q_unresolved=flag("q_unresolved", False),
        q_edge_hit=flag("q_edge_hit", False),
        needs_rerun_exact=flag("needs_rerun_exact", False),
        rerun_required=flag("rerun_required", False),
        trusted_exact=flag("trusted_exact", True),
        training_eligible_exact=flag("training_eligible_exact", True),
        delta_unresolved=flag("delta_unresolved", False),
    )

    dataset = run_dir / f"dataset_iter{iteration + 1:03d}.npz"
    write_dataset(dataset, points)
    return dataset


def evaluate_rows(
    tmp_path: Path,
    rows: list[dict[str, object]],
    mode: str = "all_selected",
    min_denominator: int = 1,
    min_fraction: float = 0.0,
) -> dict:
    cfg = toy_config()
    dataset = write_iteration(tmp_path, 0, cfg, rows)
    return evaluate_stop(
        tmp_path,
        0,
        dataset,
        cfg,
        stop_config(mode=mode, min_denominator=min_denominator, min_fraction=min_fraction),
    )
