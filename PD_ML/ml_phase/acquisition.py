from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import ActiveLearningConfig


@dataclass
class CandidateGrid:
    points: np.ndarray
    kt_values: np.ndarray
    ja_values: np.ndarray
    full_shape: Tuple[int, int]
    candidate_mask: np.ndarray
    prioritize_mask: np.ndarray


def finite_t_phase_boundary_arrays() -> Tuple[np.ndarray, np.ndarray]:
    t = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55, 0.56], dtype=np.float64)
    ja = np.array([2.12, 1.733, 1.5, 1.32, 1.16, 0.9, 0.78, 0.733, 0.7, 0.667, 0.59, 0.4, 0.178, 0.0], dtype=np.float64)
    return t, ja


def _boundary_ja_at_kt(kt: np.ndarray, boundary_t: np.ndarray, boundary_ja: np.ndarray) -> np.ndarray:
    return np.interp(kt, boundary_t, boundary_ja, left=boundary_ja[0], right=boundary_ja[-1])


def build_candidate_grid(cfg: ActiveLearningConfig) -> CandidateGrid:
    kt_vals = np.linspace(cfg.kt_min, cfg.kt_max, cfg.n_kt_candidates, dtype=np.float64)
    ja_vals = np.linspace(cfg.ja_min, cfg.ja_max, cfg.n_ja_candidates, dtype=np.float64)
    kt_mesh, ja_mesh = np.meshgrid(kt_vals, ja_vals, indexing="xy")

    # Physics-aware mask: avoid negative kT and keep points under analytic finite-T boundary plus a margin.
    b_t, b_ja = finite_t_phase_boundary_arrays()
    boundary_ja = _boundary_ja_at_kt(kt_mesh, b_t, b_ja)
    candidate_mask = (kt_mesh >= 0.0) & (ja_mesh >= cfg.ja_min) & (ja_mesh <= (boundary_ja + cfg.finite_t_band_width))

    prioritize_mask = (kt_mesh <= cfg.prioritize_kt_max) & (ja_mesh <= cfg.prioritize_ja_max)
    points = np.stack([kt_mesh.ravel(), ja_mesh.ravel()], axis=1)
    return CandidateGrid(
        points=points,
        kt_values=kt_vals,
        ja_values=ja_vals,
        full_shape=ja_mesh.shape,
        candidate_mask=candidate_mask.ravel(),
        prioritize_mask=prioritize_mask.ravel(),
    )


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _gradient_score(
    value_flat: np.ndarray,
    grid_shape: Tuple[int, int],
    kt_values: np.ndarray,
    ja_values: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(value_flat, dtype=np.float64).reshape(grid_shape)
    d_ja, d_kt = np.gradient(arr, ja_values, kt_values, edge_order=1)
    g = np.sqrt(d_ja**2 + d_kt**2)
    return _normalize_01(g.ravel())


def _distance_score(candidates: np.ndarray, existing_points: np.ndarray) -> np.ndarray:
    if existing_points.size == 0:
        return np.ones(candidates.shape[0], dtype=np.float64)
    # Avoid scipy dependency on the cluster. Compute nearest-neighbor distances
    # in chunks to keep memory bounded for dense candidate grids.
    dist = np.full(candidates.shape[0], np.inf, dtype=np.float64)
    chunk = 4096
    existing = np.asarray(existing_points, dtype=np.float64)
    for start in range(0, candidates.shape[0], chunk):
        cand = candidates[start : start + chunk]
        d2 = np.sum((cand[:, None, :] - existing[None, :, :]) ** 2, axis=2)
        dist[start : start + chunk] = np.sqrt(np.min(d2, axis=1))
    return _normalize_01(dist)


def compute_acquisition_scores(
    cfg: ActiveLearningConfig,
    grid: CandidateGrid,
    predictions: Dict[str, np.ndarray],
    existing_points: np.ndarray,
) -> Dict[str, np.ndarray]:
    reg_mean = np.asarray(predictions["reg_mean"], dtype=np.float64)
    reg_std = np.asarray(predictions["reg_std"], dtype=np.float64)
    cls_unc = np.asarray(predictions["cls_uncertainty"], dtype=np.float64)

    delta_pred = reg_mean[:, 0]
    q_pred = reg_mean[:, 1]
    eta_pred = reg_mean[:, 2]

    reg_unc = np.mean(reg_std, axis=1)
    reg_unc_n = _normalize_01(reg_unc)
    cls_unc_n = _normalize_01(cls_unc)

    delta_boundary = np.exp(-np.abs(delta_pred - cfg.delta_eps) / max(cfg.delta_scale, 1e-6))
    q_boundary = np.exp(-np.abs(np.abs(q_pred) - cfg.q_eps) / max(cfg.q_scale, 1e-6))
    eta_boundary = np.exp(-np.abs(eta_pred) / max(cfg.eta_scale, 1e-6))

    grad_eta = _gradient_score(eta_pred, grid.full_shape, grid.kt_values, grid.ja_values)
    grad_delta = _gradient_score(delta_pred, grid.full_shape, grid.kt_values, grid.ja_values)
    gradient_score = 0.5 * (grad_eta + grad_delta)

    diversity = _distance_score(grid.points, np.asarray(existing_points, dtype=np.float64))

    score = (
        cfg.w_cls_uncertainty * cls_unc_n
        + cfg.w_reg_uncertainty * reg_unc_n
        + cfg.w_delta_boundary * delta_boundary
        + cfg.w_q_boundary * q_boundary
        + cfg.w_eta_boundary * eta_boundary
        + cfg.w_gradient * gradient_score
        + cfg.w_diversity * diversity
    )

    # Soft priority to already-covered physical domain.
    score = score + 0.25 * grid.prioritize_mask.astype(np.float64)
    score = np.where(grid.candidate_mask, score, -np.inf)

    return {
        "score": score,
        "cls_uncertainty": cls_unc_n,
        "reg_uncertainty": reg_unc_n,
        "delta_boundary_score": delta_boundary,
        "q_boundary_score": q_boundary,
        "eta_zero_score": eta_boundary,
        "gradient_score": gradient_score,
        "diversity_score": diversity,
    }


def select_top_diverse(
    points: np.ndarray,
    scores: np.ndarray,
    k: int,
    min_dist: float,
    kt_range: Tuple[float, float],
    ja_range: Tuple[float, float],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if points.shape[0] != scores.shape[0]:
        raise ValueError("points and scores length mismatch")

    order = np.argsort(scores)[::-1]
    selected: list[int] = []

    k_lo, k_hi = kt_range
    j_lo, j_hi = ja_range
    scale = np.array([max(k_hi - k_lo, 1e-12), max(j_hi - j_lo, 1e-12)], dtype=np.float64)

    normalized = (points - np.array([k_lo, j_lo], dtype=np.float64)) / scale
    for idx in order:
        if not np.isfinite(scores[idx]):
            continue
        if len(selected) >= k:
            break
        if not selected:
            selected.append(int(idx))
            continue
        d = normalized[selected] - normalized[idx]
        mind = float(np.sqrt(np.sum(d**2, axis=1)).min())
        if mind >= min_dist:
            selected.append(int(idx))

    if len(selected) < k:
        used = set(selected)
        for idx in order:
            if len(selected) >= k:
                break
            if idx in used:
                continue
            if not np.isfinite(scores[idx]):
                continue
            selected.append(int(idx))
    return np.array(selected, dtype=np.int64)
