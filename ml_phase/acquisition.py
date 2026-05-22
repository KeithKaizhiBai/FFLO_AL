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
    t = np.array(
        [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55, 0.56],
        dtype=np.float64,
    )
    ja = np.array([2.12, 1.733, 1.5, 1.32, 1.16, 0.9, 0.78, 0.733, 0.7, 0.667, 0.59, 0.4, 0.178, 0.0], dtype=np.float64)
    return t, ja


def _boundary_ja_at_kt(kt: np.ndarray, boundary_t: np.ndarray, boundary_ja: np.ndarray) -> np.ndarray:
    return np.interp(kt, boundary_t, boundary_ja, left=boundary_ja[0], right=boundary_ja[-1])


def build_candidate_grid(cfg: ActiveLearningConfig) -> CandidateGrid:
    kt_vals = np.linspace(cfg.kt_min, cfg.kt_max, cfg.n_kt_candidates, dtype=np.float64)
    ja_vals = np.linspace(cfg.ja_min, cfg.ja_max, cfg.n_ja_candidates, dtype=np.float64)
    kt_mesh, ja_mesh = np.meshgrid(kt_vals, ja_vals, indexing="xy")

    base_mask = (kt_mesh >= float(cfg.kt_min)) & (kt_mesh <= float(cfg.kt_max))
    base_mask &= (ja_mesh >= float(cfg.ja_min)) & (ja_mesh <= float(cfg.ja_max))
    if str(cfg.candidate_domain_mode) == "full":
        candidate_mask = base_mask
    elif str(cfg.candidate_domain_mode) == "prior_band":
        if cfg.finite_t_band_width is None:
            raise ValueError("prior_band candidate domain requires finite_t_band_width.")
        b_t, b_ja = finite_t_phase_boundary_arrays()
        boundary_ja = _boundary_ja_at_kt(kt_mesh, b_t, b_ja)
        candidate_mask = base_mask & (ja_mesh <= (boundary_ja + float(cfg.finite_t_band_width)))
    else:
        raise ValueError(f"Unknown candidate_domain_mode: {cfg.candidate_domain_mode}")

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
    out = np.zeros_like(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out
    lo = float(np.nanmin(x[finite]))
    hi = float(np.nanmax(x[finite]))
    if hi <= lo:
        return out
    out[finite] = (x[finite] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _gradient_score(
    value_flat: np.ndarray,
    grid_shape: Tuple[int, int],
    kt_values: np.ndarray,
    ja_values: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(value_flat, dtype=np.float64).reshape(grid_shape)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    d_ja, d_kt = np.gradient(arr, ja_values, kt_values, edge_order=1)
    g = np.sqrt(d_ja**2 + d_kt**2)
    return _normalize_01(g.ravel())


def normalized_min_distance(
    points: np.ndarray,
    reference_points: np.ndarray,
    kt_range: Tuple[float, float],
    ja_range: Tuple[float, float],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    reference_points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return np.empty((0,), dtype=np.float64)
    if reference_points.size == 0:
        return np.full(points.shape[0], np.inf, dtype=np.float64)

    kt_lo, kt_hi = kt_range
    ja_lo, ja_hi = ja_range
    offset = np.array([kt_lo, ja_lo], dtype=np.float64)
    scale = np.array([max(kt_hi - kt_lo, 1e-12), max(ja_hi - ja_lo, 1e-12)], dtype=np.float64)
    points_n = (points - offset) / scale
    reference_n = (reference_points - offset) / scale

    try:
        from scipy.spatial import cKDTree  # type: ignore

        dist, _ = cKDTree(reference_n).query(points_n, k=1)
        return np.asarray(dist, dtype=np.float64)
    except Exception:
        dist = np.full(points_n.shape[0], np.inf, dtype=np.float64)
        chunk = 512
        for start in range(0, points_n.shape[0], chunk):
            cand = points_n[start : start + chunk]
            d2 = np.sum((cand[:, None, :] - reference_n[None, :, :]) ** 2, axis=2)
            dist[start : start + chunk] = np.sqrt(np.min(d2, axis=1))
        return dist


def observation_repulsion(d_obs_min: np.ndarray, ell: float, floor: float) -> np.ndarray:
    if ell <= 0.0:
        raise ValueError("observation_repulsion_length must be positive.")
    if not (0.0 <= floor < 1.0):
        raise ValueError("observation_repulsion_floor must satisfy 0 <= floor < 1.")
    d = np.asarray(d_obs_min, dtype=np.float64)
    out = floor + (1.0 - floor) * (1.0 - np.exp(-((d / ell) ** 2)))
    out = np.where(np.isfinite(d), out, 1.0)
    return np.clip(out, floor, 1.0)


def _finite_percentiles(values: np.ndarray, percentiles: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(p)}": None for p in percentiles}
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in percentiles}


def _piecewise_value(
    iteration: int | None,
    start: float,
    mid: float,
    end: float,
    mid_iter: int,
    end_iter: int,
) -> float:
    if iteration is None:
        return float(start)
    if int(iteration) < int(mid_iter):
        return float(start)
    if int(iteration) < int(end_iter):
        return float(mid)
    return float(end)


def _sampling_power_for_iteration(cfg: ActiveLearningConfig, iteration: int | None) -> float:
    schedule = str(cfg.sampling_power_schedule)
    if schedule == "piecewise":
        return _piecewise_value(
            iteration,
            start=float(cfg.sampling_power_start),
            mid=float(cfg.sampling_power_mid),
            end=float(cfg.sampling_power_end),
            mid_iter=int(cfg.sampling_power_mid_iter),
            end_iter=int(cfg.sampling_power_end_iter),
        )
    if schedule != "linear" or iteration is None:
        return float(cfg.sampling_power)
    denom = max(int(cfg.active_selection_min_iterations), 1)
    t = float(np.clip(float(iteration) / float(denom), 0.0, 1.0))
    return float(cfg.sampling_power_start) + t * (float(cfg.sampling_power_end) - float(cfg.sampling_power_start))


def _w_ext_for_iteration(cfg: ActiveLearningConfig, iteration: int | None) -> float:
    if str(cfg.w_ext_schedule) == "piecewise":
        return _piecewise_value(
            iteration,
            start=float(cfg.w_ext_start),
            mid=float(cfg.w_ext_mid),
            end=float(cfg.w_ext_end),
            mid_iter=int(cfg.w_ext_mid_iter),
            end_iter=int(cfg.w_ext_end_iter),
        )
    return float(cfg.w_extrapolation)


def _active_pool_quantile_for_iteration(cfg: ActiveLearningConfig, iteration: int | None) -> float:
    if str(cfg.active_pool_quantile_schedule) == "piecewise":
        return _piecewise_value(
            iteration,
            start=float(cfg.active_pool_quantile_start),
            mid=float(cfg.active_pool_quantile_mid),
            end=float(cfg.active_pool_quantile_end),
            mid_iter=int(cfg.active_pool_quantile_mid_iter),
            end_iter=int(cfg.active_pool_quantile_end_iter),
        )
    return float(cfg.active_pool_quantile)


def _active_pool_max_fraction_for_iteration(cfg: ActiveLearningConfig, iteration: int | None) -> float:
    if iteration is None or int(iteration) <= 0:
        return float(cfg.active_pool_max_fraction_start)
    t = float(np.clip(float(iteration) / max(int(cfg.active_pool_max_fraction_end_iter), 1), 0.0, 1.0))
    return float(cfg.active_pool_max_fraction_start) + t * (
        float(cfg.active_pool_max_fraction_end) - float(cfg.active_pool_max_fraction_start)
    )


def _interior_penalty_for_iteration(cfg: ActiveLearningConfig, iteration: int | None) -> float:
    if iteration is not None and int(iteration) >= int(cfg.interior_penalty_start_iter):
        return float(cfg.interior_penalty_late)
    return float(cfg.interior_penalty_early)


def _build_active_pool(
    a0_for_pool: np.ndarray,
    hard_unseen: np.ndarray,
    cfg: ActiveLearningConfig,
    iteration: int | None,
    requested_min: int,
) -> tuple[np.ndarray, dict[str, object]]:
    a0 = np.asarray(a0_for_pool, dtype=np.float64)
    hard = np.asarray(hard_unseen, dtype=bool)
    finite_unseen = hard & np.isfinite(a0)
    finite_values = a0[finite_unseen]
    q_requested = _active_pool_quantile_for_iteration(cfg, iteration)
    if finite_values.size == 0:
        return np.zeros_like(hard, dtype=bool), {
            "active_pool_size": 0,
            "active_pool_fraction": 0.0,
            "active_pool_rule": str(cfg.active_pool_rule),
            "active_pool_quantile_requested": float(q_requested),
            "active_pool_quantile_used": None,
            "active_pool_rel_to_p95": float(cfg.active_pool_rel_to_p95),
            "active_pool_threshold_quantile": None,
            "active_pool_threshold_rel_p95": None,
            "active_pool_threshold_final": None,
            "active_pool_threshold_relaxed": False,
            "active_pool_fraction_cap": _active_pool_max_fraction_for_iteration(cfg, iteration),
            "active_pool_fraction_cap_tightened": False,
            "active_pool_available_count": 0,
            "active_pool_warning": "no finite unseen A0_main candidates",
            "unseen_A0_for_pool_percentiles": _finite_percentiles(finite_values, [50, 75, 90, 95, 98, 99]),
        }

    quantiles = [float(q_requested)]
    if iteration is not None and int(iteration) < int(cfg.active_selection_min_iterations):
        q = float(q_requested)
        while q - 0.02 >= float(cfg.active_pool_min_quantile) - 1e-12:
            q = round(q - 0.02, 10)
            if q not in quantiles:
                quantiles.append(q)
        if float(cfg.active_pool_min_quantile) not in quantiles:
            quantiles.append(float(cfg.active_pool_min_quantile))

    p95 = float(np.percentile(finite_values, 95))
    rel_threshold = float(cfg.active_pool_rel_to_p95) * p95
    finite_count = int(finite_values.size)
    chosen = np.zeros_like(hard, dtype=bool)
    chosen_q = None
    chosen_q_threshold = None
    chosen_threshold = None
    relaxed = False
    quantile_rule_count = 0
    relative_rule_count = 0
    overlap_count = 0
    for q in quantiles:
        q_threshold = float(np.quantile(finite_values, q))
        if str(cfg.active_pool_rule) == "legacy_or":
            pool = finite_unseen & ((a0 >= q_threshold) | (a0 >= rel_threshold))
            threshold = min(q_threshold, rel_threshold)
        else:
            threshold = max(q_threshold, rel_threshold)
            pool = finite_unseen & (a0 >= threshold)
        chosen = pool
        chosen_q = q
        chosen_q_threshold = q_threshold
        chosen_threshold = threshold
        if int(np.sum(pool)) >= int(requested_min) or q <= float(cfg.active_pool_min_quantile) + 1e-12:
            relaxed = bool(q != float(q_requested))
            break

    cap = _active_pool_max_fraction_for_iteration(cfg, iteration)
    cap_tightened = False
    if finite_count > 0 and int(np.sum(chosen)) / max(finite_count, 1) > cap:
        q_cap = float(chosen_q if chosen_q is not None else q_requested)
        while q_cap < 0.995 - 1e-12:
            q_cap = min(0.995, round(q_cap + 0.005, 10))
            q_threshold = float(np.quantile(finite_values, q_cap))
            if str(cfg.active_pool_rule) == "legacy_or":
                threshold = min(q_threshold, rel_threshold)
                pool = finite_unseen & ((a0 >= q_threshold) | (a0 >= rel_threshold))
            else:
                threshold = max(q_threshold, rel_threshold)
                pool = finite_unseen & (a0 >= threshold)
            if (
                int(np.sum(pool)) / max(finite_count, 1) <= cap
                or q_cap >= 0.995 - 1e-12
                or (requested_min > 0 and int(np.sum(pool)) < int(requested_min))
            ):
                if requested_min > 0 and int(np.sum(pool)) < int(requested_min) and int(np.sum(chosen)) >= int(requested_min):
                    break
                chosen = pool
                chosen_q = q_cap
                chosen_q_threshold = q_threshold
                chosen_threshold = threshold
                cap_tightened = True
                break
            chosen = pool
            chosen_q = q_cap
            chosen_q_threshold = q_threshold
            chosen_threshold = threshold
            cap_tightened = True

    if chosen_q_threshold is not None:
        quantile_rule = finite_unseen & (a0 >= float(chosen_q_threshold))
        relative_rule = finite_unseen & (a0 >= rel_threshold)
        quantile_rule_count = int(np.sum(quantile_rule))
        relative_rule_count = int(np.sum(relative_rule))
        overlap_count = int(np.sum(quantile_rule & relative_rule))

    return chosen, {
        "active_pool_size": int(np.sum(chosen)),
        "active_pool_available_count": int(np.sum(chosen)),
        "active_pool_fraction": float(np.sum(chosen) / max(finite_count, 1)),
        "active_pool_rule": str(cfg.active_pool_rule),
        "active_pool_quantile_requested": float(q_requested),
        "active_pool_quantile_used": float(chosen_q) if chosen_q is not None else None,
        "active_pool_rel_to_p95": float(cfg.active_pool_rel_to_p95),
        "active_pool_threshold_quantile": chosen_q_threshold,
        "active_pool_threshold_rel_p95": rel_threshold,
        "active_pool_threshold_final": chosen_threshold,
        "active_pool_threshold_relaxed": relaxed,
        "active_pool_fraction_cap": float(cap),
        "active_pool_fraction_cap_tightened": bool(cap_tightened),
        "active_pool_quantile_rule_count": quantile_rule_count,
        "active_pool_relative_p95_rule_count": relative_rule_count,
        "active_pool_rule_overlap_count": overlap_count,
        "active_pool_warning": "" if int(np.sum(chosen)) > 0 else "active pool is empty after A0_main gating",
        "unseen_A0_for_pool_percentiles": _finite_percentiles(finite_values, [50, 75, 90, 95, 98, 99]),
    }


def _effective_sample_size(probs: np.ndarray) -> float | None:
    p = np.asarray(probs, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0.0)]
    if p.size == 0:
        return None
    denom = float(np.sum(p * p))
    if denom <= 0.0:
        return None
    return float(1.0 / denom)


def _phase_components(predictions: Dict[str, np.ndarray], cfg: ActiveLearningConfig) -> dict[str, np.ndarray]:
    probs = np.asarray(predictions.get("phase_proba"), dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        cls_unc = _normalize_01(np.asarray(predictions["cls_uncertainty"], dtype=np.float64))
        return {
            "P_normal": np.full_like(cls_unc, 0.5),
            "P_uniform": np.full_like(cls_unc, 0.25),
            "P_FFLO": np.full_like(cls_unc, 0.25),
            "P_SC": np.ones_like(cls_unc),
            "P_max": np.full_like(cls_unc, 0.5),
            "U_NS": np.ones_like(cls_unc),
            "U_UF": np.ones_like(cls_unc),
            "cls_entropy": cls_unc,
            "cls_margin_uncertainty": cls_unc,
            "cls_uncertainty_mix": cls_unc,
        }

    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    entropy /= np.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
    sorted_probs = np.sort(probs, axis=1)
    if probs.shape[1] >= 2:
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        margin = np.ones(probs.shape[0], dtype=np.float64)
    margin_unc = np.exp(-((margin / max(float(cfg.cls_margin_tau), 1e-12)) ** 2))
    cls_mix = float(cfg.w_cls_entropy_inner) * entropy + float(cfg.w_cls_margin_inner) * margin_unc

    p_normal = probs[:, 0] if probs.shape[1] >= 1 else np.zeros(probs.shape[0], dtype=np.float64)
    if probs.shape[1] >= 3:
        p_uniform = probs[:, 1]
        p_fflo = probs[:, 2]
        p_sc = p_uniform + p_fflo
    elif probs.shape[1] == 2:
        p_uniform = probs[:, 1]
        p_fflo = np.zeros(probs.shape[0], dtype=np.float64)
        p_sc = 1.0 - probs[:, 0]
    else:
        p_uniform = np.zeros(probs.shape[0], dtype=np.float64)
        p_fflo = np.zeros(probs.shape[0], dtype=np.float64)
        p_sc = np.ones(probs.shape[0], dtype=np.float64)
    u_ns = np.clip(4.0 * p_normal * p_sc, 0.0, 1.0)
    u_uf = np.clip(4.0 * p_uniform * p_fflo, 0.0, 1.0)

    return {
        "P_normal": np.clip(p_normal, 0.0, 1.0),
        "P_uniform": np.clip(p_uniform, 0.0, 1.0),
        "P_FFLO": np.clip(p_fflo, 0.0, 1.0),
        "P_SC": np.clip(p_sc, 0.0, 1.0),
        "P_max": np.clip(np.max(probs, axis=1), 0.0, 1.0),
        "U_NS": u_ns,
        "U_UF": u_uf,
        "cls_entropy": np.clip(entropy, 0.0, 1.0),
        "cls_margin_uncertainty": np.clip(margin_unc, 0.0, 1.0),
        "cls_uncertainty_mix": np.clip(cls_mix, 0.0, 1.0),
    }


def compute_acquisition_scores(
    cfg: ActiveLearningConfig,
    grid: CandidateGrid,
    predictions: Dict[str, np.ndarray],
    existing_points: np.ndarray,
    iteration: int | None = None,
) -> Dict[str, np.ndarray]:
    reg_mean = np.asarray(predictions["reg_mean"], dtype=np.float64)
    reg_std = np.asarray(predictions["reg_std"], dtype=np.float64)

    delta_pred = reg_mean[:, 0]
    q_pred = reg_mean[:, 1]
    eta_pred = reg_mean[:, 2]

    phase = _phase_components(predictions, cfg)
    p_sc = phase["P_SC"]
    u_ns = phase["U_NS"]
    u_uf = phase["U_UF"]

    u_delta = _normalize_01(reg_std[:, 0])
    u_q = _normalize_01(reg_std[:, 1])
    u_eta = _normalize_01(reg_std[:, 2])
    u_icp = _normalize_01(reg_std[:, 3]) if reg_std.shape[1] > 3 else np.zeros_like(u_eta)
    u_icm = _normalize_01(reg_std[:, 4]) if reg_std.shape[1] > 4 else np.zeros_like(u_eta)
    u_reg_phase = 0.5 * u_delta + 0.5 * u_q
    u_reg_response = 0.6 * u_eta + 0.2 * u_icp + 0.2 * u_icm

    delta_boundary_raw = np.exp(-np.abs(delta_pred - cfg.delta_eps) / max(cfg.delta_scale, 1e-6))
    if str(cfg.b_delta_gate_mode) == "normal_sc_competition":
        delta_boundary = delta_boundary_raw * u_ns
    else:
        delta_boundary = delta_boundary_raw
    q_boundary_raw = np.exp(-np.abs(np.abs(q_pred) - cfg.q_eps) / max(cfg.q_scale, 1e-6))
    if str(cfg.q_boundary_gate_mode) == "uf_competition":
        q_boundary_sc = u_uf * q_boundary_raw
    else:
        q_boundary_sc = p_sc * q_boundary_raw
    eta_boundary_raw = np.exp(-np.abs(eta_pred) / max(cfg.eta_scale, 1e-6))
    eta_boundary_response = p_sc * eta_boundary_raw

    grad_eta = _gradient_score(eta_pred, grid.full_shape, grid.kt_values, grid.ja_values)
    grad_delta = _gradient_score(delta_pred, grid.full_shape, grid.kt_values, grid.ja_values)
    grad_q = _gradient_score(np.abs(q_pred), grid.full_shape, grid.kt_values, grid.ja_values) * p_sc
    gradient_phase = 0.5 * grad_delta + 0.5 * grad_q
    gradient_response = grad_eta * p_sc

    kt = grid.points[:, 0]
    ja = grid.points[:, 1]
    q_margin_lo = np.abs(q_pred - cfg.q_window_safe_min)
    q_margin_hi = np.abs(cfg.q_window_safe_max - q_pred)
    q_edge_distance = np.minimum(q_margin_lo, q_margin_hi)
    q_edge_risk_raw = np.exp(-np.maximum(q_edge_distance, 0.0) / max(cfg.q_edge_margin, 1e-6))
    high_ja_risk = _normalize_01(np.maximum(ja - cfg.high_ja_q_risk_start, 0.0))
    q_edge_risk_sc = p_sc * np.maximum(q_edge_risk_raw, high_ja_risk)

    extrapolation_raw = np.maximum(
        _normalize_01(np.maximum(kt - cfg.prioritize_kt_max, 0.0)),
        _normalize_01(np.maximum(ja - cfg.prioritize_ja_max, 0.0)),
    )
    extrapolation_uncertain = extrapolation_raw * np.maximum(phase["cls_uncertainty_mix"], u_reg_phase)
    w_ext_current = _w_ext_for_iteration(cfg, iteration)

    a_phase = (
        float(cfg.w_cls_mix) * phase["cls_uncertainty_mix"]
        + float(cfg.w_reg_phase) * u_reg_phase
        + float(cfg.w_delta_boundary) * delta_boundary
        + float(cfg.w_q_boundary_sc) * q_boundary_sc
        + float(cfg.w_gradient_phase) * gradient_phase
    )
    a_numerical = float(cfg.w_q_edge_risk) * q_edge_risk_sc
    a_explore = float(w_ext_current) * extrapolation_uncertain
    a_response = (
        float(cfg.w_eta_response) * eta_boundary_response
        + float(cfg.w_gradient_response) * gradient_response
        + float(cfg.w_reg_response) * u_reg_response
    )
    a0_main = a_phase + a_numerical + a_explore

    high_confidence_interior = (
        (phase["P_max"] > float(cfg.p_conf_threshold))
        & (u_ns < float(cfg.u_ns_low))
        & (u_uf < float(cfg.u_uf_low))
        & (gradient_phase < float(cfg.g_phase_low))
        & (q_edge_risk_sc < float(cfg.e_q_low))
        & (extrapolation_uncertain < float(cfg.e_ext_low))
    )
    interior_penalty_value = _interior_penalty_for_iteration(cfg, iteration)
    if str(cfg.interior_filter_mode) == "soft_penalty":
        interior_penalty = np.where(high_confidence_interior, interior_penalty_value, 1.0)
        active_pool_candidate_mask = np.ones_like(high_confidence_interior, dtype=bool)
    elif str(cfg.interior_filter_mode) == "hard_exclude":
        interior_penalty = np.where(high_confidence_interior, 0.0, 1.0)
        active_pool_candidate_mask = ~high_confidence_interior
    else:
        interior_penalty = np.ones_like(a0_main, dtype=np.float64)
        active_pool_candidate_mask = np.ones_like(high_confidence_interior, dtype=bool)
    a0_for_pool = a0_main * interior_penalty

    score = np.where(grid.candidate_mask & active_pool_candidate_mask, a0_for_pool, -np.inf)

    return {
        "score": score,
        "A0_main": a0_main,
        "A0_main_raw": a0_main,
        "A0_for_pool": a0_for_pool,
        "A_phase": a_phase,
        "A_numerical": a_numerical,
        "A_explore": a_explore,
        "A_response": a_response,
        "cls_uncertainty": phase["cls_uncertainty_mix"],
        "cls_entropy": phase["cls_entropy"],
        "cls_margin_uncertainty": phase["cls_margin_uncertainty"],
        "P_SC": p_sc,
        "P_normal": phase["P_normal"],
        "P_uniform": phase["P_uniform"],
        "P_FFLO": phase["P_FFLO"],
        "P_max": phase["P_max"],
        "U_NS": u_ns,
        "U_UF": u_uf,
        "reg_uncertainty": u_reg_phase,
        "U_delta": u_delta,
        "U_q": u_q,
        "U_eta": u_eta,
        "U_ic_plus": u_icp,
        "U_ic_minus": u_icm,
        "U_reg_phase": u_reg_phase,
        "U_reg_response": u_reg_response,
        "B_delta_raw": delta_boundary_raw,
        "U_NS_boundary_gate": u_ns,
        "B_delta_gated": delta_boundary,
        "delta_boundary_score": delta_boundary,
        "q_boundary_score": q_boundary_sc,
        "B_q_raw": q_boundary_raw,
        "B_q_gated": q_boundary_sc,
        "q_boundary_raw": q_boundary_raw,
        "eta_zero_score": eta_boundary_response,
        "eta_zero_raw": eta_boundary_raw,
        "gradient_score": gradient_phase,
        "gradient_response": gradient_response,
        "diversity_score": np.ones_like(a0_main),
        "q_edge_risk_score": q_edge_risk_sc,
        "q_edge_risk_raw": q_edge_risk_raw,
        "delta_refine_risk_score": np.zeros_like(a0_main),
        "extrapolation_risk_score": extrapolation_uncertain,
        "extrapolation_raw": extrapolation_raw,
        "w_ext_current": np.full_like(a0_main, float(w_ext_current), dtype=np.float64),
        "interior_penalty": interior_penalty,
        "interior_penalty_value": np.full_like(a0_main, float(interior_penalty_value), dtype=np.float64),
        "interior_penalty_applied": high_confidence_interior.astype(np.int8),
        "high_confidence_interior": high_confidence_interior.astype(np.int8),
        "active_pool_candidate_mask": active_pool_candidate_mask.astype(np.int8),
    }


def select_acquisition_batch(
    points: np.ndarray,
    scores: Dict[str, np.ndarray],
    k: int,
    cfg: ActiveLearningConfig,
    rng_seed: int | None = None,
    iteration: int | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    points = np.asarray(points, dtype=np.float64)
    score = np.asarray(scores["score"], dtype=np.float64)
    a0_main = np.asarray(scores.get("A0_main", score), dtype=np.float64)
    a0_for_pool = np.asarray(scores.get("A0_for_pool", a0_main), dtype=np.float64)
    r_obs = np.asarray(scores.get("R_obs", np.ones_like(score)), dtype=np.float64)
    if points.shape[0] != score.shape[0]:
        raise ValueError("points and scores length mismatch")

    ell = float(cfg.batch_repulsion_length)
    floor = float(cfg.batch_repulsion_floor)
    if ell <= 0.0:
        raise ValueError("batch_repulsion_length must be positive.")
    if not (0.0 <= floor < 1.0):
        raise ValueError("batch_repulsion_floor must satisfy 0 <= floor < 1.")

    offset = np.array([cfg.kt_min, cfg.ja_min], dtype=np.float64)
    scale = np.array([max(cfg.kt_max - cfg.kt_min, 1e-12), max(cfg.ja_max - cfg.ja_min, 1e-12)], dtype=np.float64)
    coords_norm = (points - offset) / scale

    selected: list[int] = []
    rows: list[dict[str, object]] = []
    r_batch = np.ones(points.shape[0], dtype=np.float64)
    selected_mask = np.zeros(points.shape[0], dtype=bool)

    rng = np.random.default_rng(int(cfg.random_seed) if rng_seed is None else int(rng_seed))
    requested_batch = max(0, int(k))
    if str(cfg.run_mode) == "discovery":
        min_batch = (
            int(cfg.batch_size_min_before_min_iter)
            if iteration is not None and int(iteration) < int(cfg.active_selection_min_iterations)
            else int(cfg.batch_size_min_after_min_iter)
        )
    else:
        min_batch = int(cfg.batch_size_min)
    min_batch = max(0, min(min_batch, requested_batch))

    hard_unseen = np.isfinite(a0_for_pool)
    if "candidate_excluded" in scores:
        hard_unseen &= ~np.asarray(scores["candidate_excluded"], dtype=bool)
    else:
        hard_unseen &= np.isfinite(score)
    if "active_pool_candidate_mask" in scores:
        hard_unseen &= np.asarray(scores["active_pool_candidate_mask"], dtype=bool)
    active_pool, pool_info = _build_active_pool(
        a0_for_pool=a0_for_pool,
        hard_unseen=hard_unseen,
        cfg=cfg,
        iteration=iteration,
        requested_min=min_batch,
    )
    sampling_power = _sampling_power_for_iteration(cfg, iteration)

    scores["active_pool_mask"] = active_pool.astype(np.int8)
    scores["Aselect_initial"] = np.where(active_pool, a0_for_pool * r_obs, -np.inf)
    scores["score"] = np.where(active_pool, a0_for_pool * r_obs, -np.inf)

    def append_row(rank: int, idx: int, final_score_value: float, probability: float, source: str) -> None:
        rows.append(
            {
                "selection_rank": int(rank + 1),
                "selection_source": source,
                "selection_pool": "acquisition",
                "boundary_type": "",
                "grid_index": int(idx),
                "kT": float(points[idx, 0]),
                "JA": float(points[idx, 1]),
                "A0_main": float(scores.get("A0_main", a0_main)[idx]),
                "A0_main_raw": float(scores.get("A0_main_raw", a0_main)[idx]),
                "A0_for_pool": float(scores.get("A0_for_pool", a0_for_pool)[idx]),
                "A_phase": float(scores.get("A_phase", np.zeros_like(score))[idx]),
                "A_numerical": float(scores.get("A_numerical", np.zeros_like(score))[idx]),
                "A_explore": float(scores.get("A_explore", np.zeros_like(score))[idx]),
                "A_response": float(scores.get("A_response", np.zeros_like(score))[idx]),
                "R_obs": float(scores.get("R_obs", np.ones_like(score))[idx]),
                "R_batch": float(r_batch[idx]),
                "Aselect": float(final_score_value),
                "final_score": float(final_score_value),
                "selection_score": float(final_score_value),
                "sampling_probability_before_pick": float(probability),
                "active_pool_quantile_used": pool_info.get("active_pool_quantile_used"),
                "active_pool_threshold_quantile": pool_info.get("active_pool_threshold_quantile"),
                "active_pool_threshold_rel_p95": pool_info.get("active_pool_threshold_rel_p95"),
                "active_pool_threshold_final": pool_info.get("active_pool_threshold_final"),
                "active_pool_size": pool_info.get("active_pool_size"),
                "active_pool_fraction": pool_info.get("active_pool_fraction"),
                "cls_uncertainty": float(scores.get("cls_uncertainty", np.zeros_like(score))[idx]),
                "cls_uncertainty_mix": float(scores.get("cls_uncertainty", np.zeros_like(score))[idx]),
                "cls_entropy": float(scores.get("cls_entropy", np.zeros_like(score))[idx]),
                "cls_margin_uncertainty": float(scores.get("cls_margin_uncertainty", np.zeros_like(score))[idx]),
                "P_SC": float(scores.get("P_SC", np.ones_like(score))[idx]),
                "P_normal": float(scores.get("P_normal", np.zeros_like(score))[idx]),
                "P_uniform": float(scores.get("P_uniform", np.zeros_like(score))[idx]),
                "P_FFLO": float(scores.get("P_FFLO", np.zeros_like(score))[idx]),
                "U_NS": float(scores.get("U_NS", np.zeros_like(score))[idx]),
                "U_UF": float(scores.get("U_UF", np.zeros_like(score))[idx]),
                "U_delta": float(scores.get("U_delta", np.zeros_like(score))[idx]),
                "U_q": float(scores.get("U_q", np.zeros_like(score))[idx]),
                "U_eta": float(scores.get("U_eta", np.zeros_like(score))[idx]),
                "U_ic_plus": float(scores.get("U_ic_plus", np.zeros_like(score))[idx]),
                "U_ic_minus": float(scores.get("U_ic_minus", np.zeros_like(score))[idx]),
                "U_reg_phase": float(scores.get("U_reg_phase", np.zeros_like(score))[idx]),
                "U_reg_response": float(scores.get("U_reg_response", np.zeros_like(score))[idx]),
                "B_delta_raw": float(scores.get("B_delta_raw", np.zeros_like(score))[idx]),
                "B_delta_gated": float(scores.get("B_delta_gated", np.zeros_like(score))[idx]),
                "delta_boundary_score": float(scores.get("delta_boundary_score", np.zeros_like(score))[idx]),
                "B_delta": float(scores.get("delta_boundary_score", np.zeros_like(score))[idx]),
                "B_q_raw": float(scores.get("B_q_raw", np.zeros_like(score))[idx]),
                "B_q_gated": float(scores.get("B_q_gated", np.zeros_like(score))[idx]),
                "q_boundary_score": float(scores.get("q_boundary_score", np.zeros_like(score))[idx]),
                "B_q_SC": float(scores.get("q_boundary_score", np.zeros_like(score))[idx]),
                "gradient_score": float(scores.get("gradient_score", np.zeros_like(score))[idx]),
                "G_phase": float(scores.get("gradient_score", np.zeros_like(score))[idx]),
                "q_edge_risk_score": float(scores.get("q_edge_risk_score", np.zeros_like(score))[idx]),
                "E_q_SC": float(scores.get("q_edge_risk_score", np.zeros_like(score))[idx]),
                "extrapolation_risk_score": float(scores.get("extrapolation_risk_score", np.zeros_like(score))[idx]),
                "E_ext_uncertain": float(scores.get("extrapolation_risk_score", np.zeros_like(score))[idx]),
                "interior_penalty": float(scores.get("interior_penalty", np.ones_like(score))[idx]),
                "interior_penalty_applied": int(scores.get("interior_penalty_applied", np.zeros_like(score, dtype=np.int8))[idx]),
                "high_confidence_interior": int(scores.get("high_confidence_interior", np.zeros_like(score, dtype=np.int8))[idx]),
                "sampling_power": float(sampling_power),
                "w_ext_current": float(scores.get("w_ext_current", np.full_like(score, np.nan))[idx]),
            }
        )

    last_probs = np.zeros(points.shape[0], dtype=np.float64)
    break_reason = "selected_batch_size_reached"
    for rank in range(requested_batch):
        final_score = a0_for_pool * r_obs * r_batch
        final_score[selected_mask] = -np.inf
        eligible = active_pool & np.isfinite(final_score) & (final_score > 0.0)
        if not np.any(eligible):
            break_reason = "no_positive_Aselect_in_active_pool"
            break

        if str(cfg.selection_mode) == "stochastic":
            weights = np.zeros_like(final_score, dtype=np.float64)
            weights[eligible] = np.maximum(final_score[eligible], 0.0) ** sampling_power
            total = float(np.sum(weights))
            if not np.isfinite(total) or total <= 0.0:
                break_reason = "sampling_weights_degenerate"
                break
            probs = weights / total
            last_probs = probs
            idx = int(rng.choice(np.arange(points.shape[0]), p=probs))
            probability = float(probs[idx])
            source = "acquisition_stochastic"
        else:
            idx = int(np.nanargmax(np.where(eligible, final_score, -np.inf)))
            probability = 1.0
            source = "acquisition"
        if not np.isfinite(final_score[idx]):
            break
        selected.append(idx)
        selected_mask[idx] = True
        append_row(rank, idx, float(final_score[idx]), probability, source)

        dist = np.sqrt(np.sum((coords_norm - coords_norm[idx]) ** 2, axis=1))
        factor = floor + (1.0 - floor) * (1.0 - np.exp(-((dist / ell) ** 2)))
        r_batch *= np.clip(factor, floor, 1.0)

    if len(selected) < requested_batch and break_reason == "selected_batch_size_reached":
        break_reason = "active_pool_exhausted_before_batch_max"
    elif len(selected) == requested_batch:
        break_reason = "selected_batch_size_reached"
    neff = _effective_sample_size(last_probs)
    underfilled = bool(len(selected) < requested_batch)
    summary = {
        **pool_info,
        "requested_batch_size": int(requested_batch),
        "selected_batch_size": int(len(selected)),
        "active_pool_available_count": int(pool_info.get("active_pool_available_count") or pool_info.get("active_pool_size") or 0),
        "batch_was_underfilled": underfilled,
        "underfill_reason": "" if not underfilled else break_reason,
        "batch_size_min_before_min_iter": int(cfg.batch_size_min_before_min_iter),
        "batch_size_min_after_min_iter": int(cfg.batch_size_min_after_min_iter),
        "allow_underfilled_batch_after_min_iter": bool(cfg.allow_underfilled_batch_after_min_iter),
        "adaptive_min_batch_used": int(min_batch),
        "sampling_power_used": float(sampling_power),
        "N_eff": neff,
        "N_eff_over_active_pool_size": (
            float(neff / max(int(pool_info.get("active_pool_size") or 0), 1)) if neff is not None else None
        ),
        "selected_less_than_max_reason": "" if len(selected) >= requested_batch else break_reason,
    }
    scores["_selection_summary"] = summary
    scores["R_batch_final"] = r_batch
    if not selected:
        return np.empty((0, 2), dtype=np.float64), rows
    return points[np.asarray(selected, dtype=np.int64)], rows
