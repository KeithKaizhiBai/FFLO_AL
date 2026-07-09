from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ActiveLearningConfig
from .dataset_builder import FlatDataset, load_flat_dataset
from .hpc import write_point_shards
from .labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC
from .models import predict_models, train_models
from .stageiv_3d import min_distance_3d
from .topology_oracle import TopologyModelParams, TopologyPfaffianOracle


STAGEV_RUN_ID = "stagev_acqv2_boundary_support_learned_residual_3d_v1"
STAGEV_OUTPUT_ROOT = "ML_Phase_StageV_AcqV2"


@dataclass(frozen=True)
class StageVConfig:
    run_id: str = STAGEV_RUN_ID
    output_root: str = STAGEV_OUTPUT_ROOT
    kt_min: float = 0.0
    kt_max: float = 0.56
    ja_min: float = 0.0
    ja_max: float = 2.12
    mu_min: float = -0.5
    mu_max: float = 1.5
    guard_mu_min: float = -1.0
    guard_mu_max: float = 2.0
    t: float = 1.0
    u: float = 1.6
    lambda_ry: float = 0.6
    lambda_rz: float = 0.6
    initial_seed_size: int = 1024
    micro_batch_size: int = 64
    max_micro_batches: int = 96
    candidate_pool_size: int = 65536
    random_seed: int = 20260628
    model_ensemble: int = 5
    reg_epochs: int = 160
    cls_epochs: int = 160
    hidden_dim: int = 96
    q_eps: float = 1.0e-2
    gap_tol: float = 1.0e-8
    tau_ns: float = 1.0e-3
    tau_uf: float = 2.0e-2
    tau_p0: float = 2.0e-2
    tau_ppi: float = 2.0e-2
    tau_gap: float = 1.0
    support_radius: float = 0.065
    exact_repulsion_radius: float = 0.035
    diversity_radius: float = 0.045
    diversity_floor: float = 0.03
    bracket_neighbor_k: int = 12
    bracket_max_distance: float = 0.20
    global_candidate_fraction: float = 0.55
    bracket_candidate_fraction: float = 0.25
    sparse_fill_candidate_fraction: float = 0.10
    mu_edge_candidate_fraction: float = 0.10
    bracket_jitter_scale: float = 0.018
    softmax_temperature: float = 0.18
    gumbel_noise_scale: float = 1.0
    learned_min_reward_samples: int = 256
    learned_initial_lambda: float = 0.1
    learned_lambda_max: float = 0.7
    learned_validation_margin: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path) -> "StageVConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in fields})

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.kt_min, self.ja_min, self.mu_min], dtype=np.float64)
        hi = np.array([self.kt_max, self.ja_max, self.mu_max], dtype=np.float64)
        return lo, hi


def active_config_from_stagev(cfg: StageVConfig) -> ActiveLearningConfig:
    return ActiveLearningConfig(
        run_mode="discovery",
        candidate_domain_mode="full",
        initialization="sobol_scrambled",
        initial_seed_size=int(cfg.initial_seed_size),
        batch_size_max=int(cfg.micro_batch_size),
        points_per_iter=int(cfg.micro_batch_size),
        iterations=int(cfg.max_micro_batches + 1),
        output_root=str(cfg.output_root),
        random_seed=int(cfg.random_seed),
        seed=int(cfg.random_seed),
        n_ensemble=int(cfg.model_ensemble),
        reg_epochs=int(cfg.reg_epochs),
        cls_epochs=int(cfg.cls_epochs),
        hidden_dim=int(cfg.hidden_dim),
        kt_min=float(cfg.kt_min),
        kt_max=float(cfg.kt_max),
        ja_min=float(cfg.ja_min),
        ja_max=float(cfg.ja_max),
        acquisition_profile="topo_trivial",
    )


def topology_params_from_stagev(cfg: StageVConfig) -> TopologyModelParams:
    return TopologyModelParams(t=float(cfg.t), lambda_ry=float(cfg.lambda_ry), lambda_rz=float(cfg.lambda_rz), mu=0.0)


def sobol_points_3d(n: int, cfg: StageVConfig, seed_offset: int = 0) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.empty((0, 3), dtype=np.float64)
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=int(cfg.random_seed) + int(seed_offset))
    unit = engine.draw(n).cpu().numpy().astype(np.float64, copy=False)
    lo, hi = cfg.bounds()
    return lo + unit * (hi - lo)


def normalized_coordinates(points: np.ndarray, cfg: StageVConfig) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo, hi = cfg.bounds()
    return (points - lo) / np.maximum(hi - lo, 1.0e-12)


def clip_to_bounds(points: np.ndarray, cfg: StageVConfig) -> np.ndarray:
    lo, hi = cfg.bounds()
    return np.minimum(np.maximum(np.asarray(points, dtype=np.float64).reshape(-1, 3), lo), hi)


def dataset_points_3d(dataset: FlatDataset, cfg: StageVConfig) -> np.ndarray:
    x = np.asarray(dataset.x, dtype=np.float64)
    if x.shape[1] == 3:
        return x
    mu = np.asarray(dataset.records.get("mu", np.zeros(x.shape[0])), dtype=np.float64)
    return np.column_stack([x[:, 0], x[:, 1], mu])


def _trusted_mask(dataset: FlatDataset) -> np.ndarray:
    n = dataset.x.shape[0]
    trusted = np.asarray(dataset.records.get("trusted_exact", np.ones(n, dtype=np.int8))).astype(bool)
    eligible = np.asarray(dataset.records.get("training_eligible_exact", np.ones(n, dtype=np.int8))).astype(bool)
    rerun = np.asarray(dataset.records.get("needs_rerun_exact", np.zeros(n, dtype=np.int8))).astype(bool)
    q_unresolved = np.asarray(dataset.records.get("q_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)
    delta_unresolved = np.asarray(dataset.records.get("delta_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)
    return trusted & eligible & ~rerun & ~q_unresolved & ~delta_unresolved


def _mutual_opposite_midpoints(
    points: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    cfg: StageVConfig,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels)
    valid = np.asarray(valid, dtype=bool)
    idx_valid = np.flatnonzero(valid)
    if idx_valid.size < 2:
        return np.empty((0, 3), dtype=np.float64)
    pts = points[idx_valid]
    labs = labels[idx_valid]
    pts_n = normalized_coordinates(pts, cfg)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        k = min(int(cfg.bracket_neighbor_k) + 1, pts.shape[0])
        dist, nn = cKDTree(pts_n).query(pts_n, k=k)
    except Exception:
        dmat = np.sqrt(np.sum((pts_n[:, None, :] - pts_n[None, :, :]) ** 2, axis=2))
        nn = np.argsort(dmat, axis=1)[:, : min(int(cfg.bracket_neighbor_k) + 1, pts.shape[0])]
        dist = np.take_along_axis(dmat, nn, axis=1)
    neighbor_sets = [set(np.ravel(row[1:]).astype(int).tolist()) for row in nn]
    rows: list[np.ndarray] = []
    for i in range(pts.shape[0]):
        for d, j0 in zip(np.ravel(dist[i])[1:], np.ravel(nn[i])[1:]):
            j = int(j0)
            if j <= i or i not in neighbor_sets[j]:
                continue
            if labs[i] == labs[j]:
                continue
            if not np.isfinite(d) or float(d) > float(cfg.bracket_max_distance):
                continue
            rows.append(0.5 * (pts[i] + pts[j]))
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def build_boundary_support_sets(dataset: FlatDataset, cfg: StageVConfig) -> dict[str, np.ndarray]:
    points = dataset_points_3d(dataset, cfg)
    phase = np.asarray(dataset.y_phase, dtype=np.int64)
    trusted = _trusted_mask(dataset)
    topo_trusted = np.asarray(dataset.records.get("topology_trusted", np.zeros(points.shape[0], dtype=np.int8))).astype(bool)
    spectral_code = np.asarray(dataset.records.get("topology_spectral_status_code", np.full(points.shape[0], -1)), dtype=np.int64)
    p0 = np.asarray(dataset.records.get("topology_p0", np.full(points.shape[0], np.nan)), dtype=np.float64)
    ppi = np.asarray(dataset.records.get("topology_ppi", np.full(points.shape[0], np.nan)), dtype=np.float64)
    topo_ok = trusted & topo_trusted & (phase != PHASE_NORMAL) & (spectral_code == 0)
    gapless = np.asarray(dataset.records.get("topology_label_code", np.full(points.shape[0], -1)), dtype=np.int64) == 2
    gapped = topo_ok & ~gapless
    return {
        "normal_sc": _mutual_opposite_midpoints(points, phase != PHASE_NORMAL, trusted, cfg),
        "uniform_fflo": _mutual_opposite_midpoints(points, phase == PHASE_FFLO, trusted & (phase != PHASE_NORMAL), cfg),
        "p0_topology": _mutual_opposite_midpoints(points, np.signbit(p0), topo_ok & np.isfinite(p0), cfg),
        "ppi_topology": _mutual_opposite_midpoints(points, np.signbit(ppi), topo_ok & np.isfinite(ppi), cfg),
        "gap_nodal": _mutual_opposite_midpoints(points, gapless, trusted & topo_trusted & (phase != PHASE_NORMAL), cfg),
        "trusted_exact": points[trusted],
        "trusted_sc": points[trusted & (phase != PHASE_NORMAL)],
        "trusted_topology": points[topo_ok],
        "trusted_gapped_sc": points[gapped],
    }


def _knn_field_predict(points: np.ndarray, ref_points: np.ndarray, ref_values: np.ndarray, cfg: StageVConfig, k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ref_points = np.asarray(ref_points, dtype=np.float64).reshape(-1, 3)
    ref_values = np.asarray(ref_values, dtype=np.float64).ravel()
    valid = np.isfinite(ref_values)
    ref_points = ref_points[valid]
    ref_values = ref_values[valid]
    if ref_points.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=np.float64), np.ones(points.shape[0], dtype=np.float64)
    p = normalized_coordinates(points, cfg)
    r = normalized_coordinates(ref_points, cfg)
    kk = min(max(int(k), 1), ref_points.shape[0])
    try:
        from scipy.spatial import cKDTree  # type: ignore

        dist, idx = cKDTree(r).query(p, k=kk)
    except Exception:
        dmat = np.sqrt(np.sum((p[:, None, :] - r[None, :, :]) ** 2, axis=2))
        idx = np.argsort(dmat, axis=1)[:, :kk]
        dist = np.take_along_axis(dmat, idx, axis=1)
    idx = np.asarray(idx)
    dist = np.asarray(dist, dtype=np.float64)
    if idx.ndim == 1:
        idx = idx[:, None]
        dist = dist[:, None]
    weights = 1.0 / np.maximum(dist, 1.0e-6) ** 2
    vals = ref_values[idx]
    mean = np.sum(weights * vals, axis=1) / np.maximum(np.sum(weights, axis=1), 1.0e-12)
    var = np.sum(weights * (vals - mean[:, None]) ** 2, axis=1) / np.maximum(np.sum(weights, axis=1), 1.0e-12)
    spread = np.sqrt(np.maximum(var, 0.0))
    nearest = np.min(dist, axis=1)
    uncertainty = spread + 0.25 * nearest
    return mean.astype(np.float64), np.maximum(uncertainty, 1.0e-6).astype(np.float64)


def predict_stagev_fields(
    dataset: FlatDataset,
    candidate_points: np.ndarray,
    cfg: StageVConfig,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate_points = np.asarray(candidate_points, dtype=np.float64).reshape(-1, 3)
    active_cfg = active_config_from_stagev(cfg)
    bundle = train_models(dataset_points_3d(dataset, cfg), dataset.y_reg, dataset.y_phase, active_cfg, device=device)
    pred = predict_models(bundle, candidate_points)
    reg_mean = np.asarray(pred["reg_mean"], dtype=np.float64)
    reg_std = np.asarray(pred["reg_std"], dtype=np.float64)
    phase_probs = np.asarray(pred.get("phase_proba", np.empty((candidate_points.shape[0], 0))), dtype=np.float64)
    phase_pred = np.asarray(pred.get("phase_pred", np.full(candidate_points.shape[0], -1)), dtype=np.int64)
    classes = np.asarray(bundle.classifier.classes_, dtype=np.int64)

    def prob_for(cls: int) -> np.ndarray:
        out = (phase_pred == cls).astype(np.float64)
        where = np.flatnonzero(classes == cls)
        if where.size and phase_probs.shape[1] == classes.size:
            out = phase_probs[:, int(where[0])]
        return out

    p_normal = prob_for(PHASE_NORMAL)
    p_uniform = prob_for(PHASE_UNIFORM_SC)
    p_fflo = prob_for(PHASE_FFLO)
    p_sc = np.clip(p_uniform + p_fflo, 0.0, 1.0)
    points = dataset_points_3d(dataset, cfg)
    records = dataset.records
    f_gap = np.asarray(records.get("free_energy_gap_to_normal", np.full(points.shape[0], np.nan)), dtype=np.float64)
    m_ns, s_ns = _knn_field_predict(candidate_points, points, f_gap, cfg)
    delta_pred = np.maximum(reg_mean[:, 0], 0.0)
    q_pred = reg_mean[:, 1]
    q_std = np.maximum(reg_std[:, 1], 1.0e-6)
    m_uf = np.abs(q_pred) - float(cfg.q_eps)
    s_uf = q_std
    pf = TopologyPfaffianOracle(topology_params_from_stagev(cfg))
    p0, ppi, product, pf_margin = pf.analytic_pfaffians(delta_pred, q_pred, candidate_points[:, 1], mu=candidate_points[:, 2])
    p0_ref = np.asarray(records.get("topology_p0", np.full(points.shape[0], np.nan)), dtype=np.float64)
    ppi_ref = np.asarray(records.get("topology_ppi", np.full(points.shape[0], np.nan)), dtype=np.float64)
    _, p0_unc_knn = _knn_field_predict(candidate_points, points, p0_ref, cfg)
    _, ppi_unc_knn = _knn_field_predict(candidate_points, points, ppi_ref, cfg)
    s_p0 = np.maximum(p0_unc_knn, 0.05 * (np.abs(p0) + 1.0))
    s_ppi = np.maximum(ppi_unc_knn, 0.05 * (np.abs(ppi) + 1.0))
    bulk_gap = np.asarray(records.get("topology_bulk_gap", np.full(points.shape[0], np.nan)), dtype=np.float64)
    log_gap_ref = np.log(np.maximum(bulk_gap, float(cfg.gap_tol)) / max(float(cfg.gap_tol), 1.0e-12))
    m_gap, s_gap = _knn_field_predict(candidate_points, points, log_gap_ref, cfg)
    frame = pd.DataFrame(
        {
            "kT": candidate_points[:, 0],
            "JA": candidate_points[:, 1],
            "mu": candidate_points[:, 2],
            "p_normal": p_normal,
            "p_uniform_SC": p_uniform,
            "p_FFLO": p_fflo,
            "p_SC": p_sc,
            "pred_delta": delta_pred,
            "pred_q": q_pred,
            "m_NS": m_ns,
            "sigma_NS": s_ns,
            "m_UF": m_uf,
            "sigma_UF": s_uf,
            "m_P0": p0,
            "sigma_P0": s_p0,
            "m_Ppi": ppi,
            "sigma_Ppi": s_ppi,
            "m_gap": m_gap,
            "sigma_gap": s_gap,
            "pf_product_pred": product,
            "pfaffian_margin_pred": pf_margin,
        }
    )
    summary = {"candidate_count": int(candidate_points.shape[0]), "classes": classes.astype(int).tolist()}
    return frame, summary


def boundary_likelihood(mean: np.ndarray, sigma: np.ndarray, tau: float) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1.0e-12)
    tau = max(float(tau), 1.0e-12)
    upper = (tau - mean) / (np.sqrt(2.0) * sigma)
    lower = (-tau - mean) / (np.sqrt(2.0) * sigma)
    erf_vec = np.vectorize(math.erf)
    return np.clip(0.5 * (erf_vec(upper) - erf_vec(lower)), 0.0, 1.0)


def uncertainty_factor(mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 0.0)
    return np.clip(sigma / (np.abs(mean) + sigma + 1.0e-12), 0.0, 1.0)


def support_sparsity(points: np.ndarray, support_points: np.ndarray, cfg: StageVConfig) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if support_points is None or np.asarray(support_points).size == 0:
        return np.ones(points.shape[0], dtype=np.float64), np.full(points.shape[0], np.inf, dtype=np.float64)
    d = min_distance_3d(points, np.asarray(support_points, dtype=np.float64).reshape(-1, 3), cfg)  # type: ignore[arg-type]
    h = 1.0 - np.exp(-((d / max(float(cfg.support_radius), 1.0e-12)) ** 2))
    return np.clip(h, 0.0, 1.0), d


def automatic_boundary_priorities(history: dict[str, dict[str, float]] | None, boundary_names: list[str]) -> dict[str, float]:
    if not history:
        return {name: 0.0 for name in boundary_names}
    raw: dict[str, float] = {}
    for name in boundary_names:
        h = history.get(name, {})
        signal = (
            max(float(h.get("trusted_surprise", 0.0)), 0.0)
            + max(float(h.get("coverage_deficit", 0.0)), 0.0)
            + max(float(h.get("surface_shift", 0.0)), 0.0)
            + max(float(h.get("component_instability", 0.0)), 0.0)
        )
        raw[name] = math.log1p(signal)
    mean = float(np.mean(list(raw.values()))) if raw else 0.0
    return {k: float(v - mean) for k, v in raw.items()}


def score_stagev_a0(
    candidate_features: pd.DataFrame,
    support_sets: dict[str, np.ndarray],
    cfg: StageVConfig,
    priority_history: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = candidate_features.copy()
    points = frame[["kT", "JA", "mu"]].to_numpy(float)
    boundaries = [
        ("normal_sc", "m_NS", "sigma_NS", cfg.tau_ns, "normal_sc"),
        ("uniform_fflo", "m_UF", "sigma_UF", cfg.tau_uf, "uniform_fflo"),
        ("p0_topology", "m_P0", "sigma_P0", cfg.tau_p0, "p0_topology"),
        ("ppi_topology", "m_Ppi", "sigma_Ppi", cfg.tau_ppi, "ppi_topology"),
        ("gap_nodal", "m_gap", "sigma_gap", cfg.tau_gap, "gap_nodal"),
    ]
    alpha = automatic_boundary_priorities(priority_history, [b[0] for b in boundaries])
    log_terms: list[np.ndarray] = []
    support_counts: dict[str, int] = {}
    for name, m_key, s_key, tau, support_key in boundaries:
        b_like = boundary_likelihood(frame[m_key].to_numpy(float), frame[s_key].to_numpy(float), tau)
        u = uncertainty_factor(frame[m_key].to_numpy(float), frame[s_key].to_numpy(float))
        h, d = support_sparsity(points, support_sets.get(support_key, np.empty((0, 3))), cfg)
        if name in {"p0_topology", "ppi_topology", "gap_nodal"}:
            b_like *= frame.get("p_SC", pd.Series(np.ones(len(frame)))).to_numpy(float)
        a = np.clip(b_like * u * h, 0.0, np.inf)
        frame[f"B_{name}"] = b_like
        frame[f"U_{name}"] = u
        frame[f"H_{name}"] = h
        frame[f"support_distance_{name}"] = d
        frame[f"A_{name}"] = a
        support_counts[name] = int(np.asarray(support_sets.get(support_key, np.empty((0, 3)))).reshape(-1, 3).shape[0])
        log_terms.append(np.log(np.maximum(a, 1.0e-300)) + float(alpha.get(name, 0.0)))
    stack = np.vstack(log_terms) if log_terms else np.full((1, len(frame)), -np.inf)
    m = np.max(stack, axis=0)
    a0 = np.exp(m) * np.sum(np.exp(stack - m), axis=0)
    d_exact = min_distance_3d(points, support_sets.get("trusted_exact", np.empty((0, 3))), cfg)  # type: ignore[arg-type]
    exact_repulsion = 1.0 - np.exp(-((d_exact / max(float(cfg.exact_repulsion_radius), 1.0e-12)) ** 2))
    frame["nearest_exact_distance"] = d_exact
    frame["exact_repulsion"] = np.clip(exact_repulsion, 0.0, 1.0)
    frame["A0"] = a0 * frame["exact_repulsion"].to_numpy(float)
    summary = {"boundary_priorities": alpha, "support_counts": support_counts, "a0_max": float(np.nanmax(frame["A0"])) if len(frame) else 0.0}
    return frame, summary


def generate_stagev_candidate_pool(dataset: FlatDataset, cfg: StageVConfig, iteration: int) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.random_seed) + 7919 * int(iteration))
    n = int(cfg.candidate_pool_size)
    fractions = {
        "global_sobol": float(cfg.global_candidate_fraction),
        "boundary_support_jitter": float(cfg.bracket_candidate_fraction),
        "sparse_fill": float(cfg.sparse_fill_candidate_fraction),
        "mu_edge_guard": float(cfg.mu_edge_candidate_fraction),
    }
    denom = sum(max(v, 0.0) for v in fractions.values()) or 1.0
    counts = {k: int(round(n * max(v, 0.0) / denom)) for k, v in fractions.items()}
    counts["global_sobol"] += n - sum(counts.values())
    blocks: list[np.ndarray] = []
    sources: list[str] = []
    global_points = sobol_points_3d(counts["global_sobol"], cfg, seed_offset=1000 + int(iteration))
    blocks.append(global_points)
    sources += ["global_sobol"] * global_points.shape[0]
    supports = build_boundary_support_sets(dataset, cfg)
    support_union = np.vstack([v for k, v in supports.items() if k in {"normal_sc", "uniform_fflo", "p0_topology", "ppi_topology"} and v.size]) if any(
        v.size for k, v in supports.items() if k in {"normal_sc", "uniform_fflo", "p0_topology", "ppi_topology"}
    ) else np.empty((0, 3))
    count = counts["boundary_support_jitter"]
    if support_union.size and count > 0:
        idx = rng.integers(0, support_union.shape[0], size=count)
        lo, hi = cfg.bounds()
        jitter = rng.normal(0.0, float(cfg.bracket_jitter_scale), size=(count, 3)) * (hi - lo)
        pts = clip_to_bounds(support_union[idx] + jitter, cfg)
    else:
        pts = sobol_points_3d(count, cfg, seed_offset=2000 + int(iteration))
    blocks.append(pts)
    sources += ["boundary_support_jitter"] * pts.shape[0]
    sparse = sobol_points_3d(counts["sparse_fill"], cfg, seed_offset=3000 + int(iteration))
    blocks.append(sparse)
    sources += ["sparse_fill"] * sparse.shape[0]
    edge = sobol_points_3d(counts["mu_edge_guard"], cfg, seed_offset=4000 + int(iteration))
    if edge.shape[0]:
        lo, hi = cfg.bounds()
        half = edge.shape[0] // 2
        width = 0.04 * (hi[2] - lo[2])
        edge[:half, 2] = lo[2] + rng.random(half) * width
        edge[half:, 2] = hi[2] - rng.random(edge.shape[0] - half) * width
    blocks.append(edge)
    sources += ["mu_edge_guard"] * edge.shape[0]
    points = np.vstack([b for b in blocks if b.size]) if blocks else np.empty((0, 3))
    source = np.asarray(sources, dtype=object)
    rounded = np.round(points, 8)
    _, first = np.unique(rounded, axis=0, return_index=True)
    first = np.sort(first)
    points = points[first]
    source = source[first]
    meta = pd.DataFrame({"candidate_source": source})
    return points, meta, {"candidate_source_counts": meta["candidate_source"].value_counts().to_dict(), "requested_candidate_counts": counts}


def train_linear_value_model(features: pd.DataFrame, rewards: np.ndarray, feature_columns: list[str], l2: float = 1.0e-3) -> dict[str, Any]:
    x = features[feature_columns].to_numpy(float)
    y = np.asarray(rewards, dtype=np.float64).ravel()
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if np.sum(valid) < max(8, len(feature_columns) + 2):
        return {"status": "insufficient_data", "feature_columns": feature_columns, "coef": [], "intercept": 0.0}
    x = x[valid]
    y = y[valid]
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1.0e-12] = 1.0
    xs = (x - mean) / std
    design = np.column_stack([np.ones(xs.shape[0]), xs])
    reg = l2 * np.eye(design.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ y)
    pred = design @ beta
    corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 1.0e-12 and np.std(y) > 1.0e-12 else 0.0
    return {
        "status": "trained",
        "feature_columns": feature_columns,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "intercept": float(beta[0]),
        "coef": beta[1:].tolist(),
        "train_rank_correlation": corr,
        "sample_count": int(x.shape[0]),
    }


def predict_linear_value_model(model: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    if model.get("status") != "trained":
        return np.zeros(len(features), dtype=np.float64)
    cols = list(model["feature_columns"])
    frame = features.copy()
    for col in cols:
        if col not in frame:
            frame[col] = 0.0
    x = frame[cols].to_numpy(float)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    return float(model["intercept"]) + ((x - mean) / np.maximum(std, 1.0e-12)) @ coef


def update_lambda_t(prev_lambda: float, validation: dict[str, float], cfg: StageVConfig) -> float:
    n = int(validation.get("reward_sample_count", 0))
    if n < int(cfg.learned_min_reward_samples):
        return 0.0
    delta = float(validation.get("rank_correlation_delta_vs_a0", 0.0))
    unstable = bool(validation.get("bad_sampling_detected", 0.0))
    if unstable or delta < -float(cfg.learned_validation_margin):
        return max(0.0, 0.5 * float(prev_lambda))
    if delta > float(cfg.learned_validation_margin):
        return min(float(cfg.learned_lambda_max), max(float(prev_lambda), float(cfg.learned_initial_lambda)) + 0.05)
    return min(float(cfg.learned_lambda_max), max(0.0, float(prev_lambda)))


def select_micro_batch(
    scored: pd.DataFrame,
    cfg: StageVConfig,
    rng: np.random.Generator | None = None,
    learned_values: np.ndarray | None = None,
    lambda_t: float = 0.0,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    rng = rng or np.random.default_rng(int(cfg.random_seed))
    frame = scored.copy().reset_index(drop=True)
    g = np.zeros(len(frame), dtype=np.float64) if learned_values is None else np.asarray(learned_values, dtype=np.float64).ravel()
    base_score = np.asarray(frame["A0"], dtype=np.float64)
    final_score = base_score * np.exp(float(lambda_t) * np.clip(g, -5.0, 5.0))
    final_score = np.where(np.isfinite(final_score) & (final_score > 0), final_score, 0.0)
    coords = frame[["kT", "JA", "mu"]].to_numpy(float)
    coords_n = normalized_coordinates(coords, cfg)
    diversity = np.ones(len(frame), dtype=np.float64)
    selected: list[int] = []
    rows: list[dict[str, Any]] = []
    for rank in range(int(cfg.micro_batch_size)):
        score = final_score * diversity
        eligible = score > 0
        if not np.any(eligible):
            break
        logit = np.log(np.maximum(score, 1.0e-300)) / max(float(cfg.softmax_temperature), 1.0e-12)
        gumbel = -np.log(-np.log(np.clip(rng.random(len(frame)), 1.0e-12, 1.0 - 1.0e-12)))
        pick_score = np.where(eligible, logit + float(cfg.gumbel_noise_scale) * gumbel, -np.inf)
        idx = int(np.argmax(pick_score))
        probs = np.exp(logit - np.max(logit[eligible]))
        full_prob = np.zeros(len(frame), dtype=np.float64)
        full_prob[eligible] = probs[eligible] / np.maximum(np.sum(probs[eligible]), 1.0e-300)
        selected.append(idx)
        row = frame.iloc[idx].to_dict()
        row.update(
            {
                "selection_rank": rank + 1,
                "g_theta": float(g[idx]),
                "lambda_t": float(lambda_t),
                "final_A": float(final_score[idx]),
                "selection_probability": float(full_prob[idx]),
                "diversity_factor_before_pick": float(diversity[idx]),
            }
        )
        rows.append(row)
        final_score[idx] = 0.0
        d = np.sqrt(np.sum((coords_n - coords_n[idx]) ** 2, axis=1))
        factor = float(cfg.diversity_floor) + (1.0 - float(cfg.diversity_floor)) * (
            1.0 - np.exp(-((d / max(float(cfg.diversity_radius), 1.0e-12)) ** 2))
        )
        diversity *= np.clip(factor, float(cfg.diversity_floor), 1.0)
    selected_points = coords[np.asarray(selected, dtype=np.int64)] if selected else np.empty((0, 3), dtype=np.float64)
    meta = pd.DataFrame(rows)
    summary = {
        "selected_batch_size": int(selected_points.shape[0]),
        "requested_micro_batch_size": int(cfg.micro_batch_size),
        "lambda_t": float(lambda_t),
        "mean_selection_probability": float(meta["selection_probability"].mean()) if not meta.empty else 0.0,
    }
    return selected_points, meta, summary


def compute_point_rewards(selected: pd.DataFrame, exact: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exact = exact if exact is not None else pd.DataFrame()
    for i, row in selected.reset_index(drop=True).iterrows():
        trusted = bool(exact.iloc[i].get("trusted_exact", True)) if i < len(exact) else True
        training = bool(exact.iloc[i].get("training_eligible_exact", True)) if i < len(exact) else True
        phase_surprise = float(row.get("B_normal_sc", 0.0)) * float(row.get("U_normal_sc", 0.0))
        topology_surprise = max(float(row.get("B_p0_topology", 0.0)), float(row.get("B_ppi_topology", 0.0)))
        support_gain = max(float(row.get("H_normal_sc", 0.0)), float(row.get("H_p0_topology", 0.0)), float(row.get("H_ppi_topology", 0.0)))
        margin = max(float(row.get("B_normal_sc", 0.0)), float(row.get("B_uniform_fflo", 0.0)), topology_surprise)
        redundant = 1.0 if float(row.get("nearest_exact_distance", np.inf)) < 0.01 else 0.0
        untrusted_penalty = 1.0 if not (trusted and training) else 0.0
        reward = 0.30 * phase_surprise + 0.25 * topology_surprise + 0.25 * support_gain + 0.20 * margin - 0.30 * redundant - 0.60 * untrusted_penalty
        rows.append(
            {
                "reward_bracket": support_gain,
                "reward_surprise": phase_surprise + topology_surprise,
                "reward_support": support_gain,
                "reward_margin": margin,
                "penalty_redundant": redundant,
                "penalty_untrusted": untrusted_penalty,
                "reward_scalar": float(reward),
            }
        )
    return pd.DataFrame(rows)


def write_empty_stagev_dataset(run_dir: Path, iteration: int = 0) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    npz_path = run_dir / f"dataset_iter{iteration:03d}.npz"
    csv_path = run_dir / f"dataset_iter{iteration:03d}.csv"
    payload = {
        "x": np.empty((0, 3), dtype=np.float64),
        "y_reg": np.empty((0, 5), dtype=np.float64),
        "y_phase": np.empty((0,), dtype=np.int64),
        "y_eta_sign": np.empty((0,), dtype=np.int64),
        "y_strong_diode": np.empty((0,), dtype=np.int64),
        "mu": np.empty((0,), dtype=np.float64),
    }
    if not npz_path.exists():
        np.savez(npz_path, **payload)
    if not csv_path.exists():
        pd.DataFrame(columns=["kT", "JA", "mu", "delta_opt", "q_opt", "eta", "ic_plus", "ic_minus", "phase_label"]).to_csv(csv_path, index=False)
    return npz_path, csv_path


def write_stagev_selection(
    run_dir: Path,
    iteration: int,
    selected_points: np.ndarray,
    metadata: pd.DataFrame,
    cfg: StageVConfig,
    world_size: int,
    partition_strategy: str,
    summary: dict[str, Any],
) -> None:
    iter_dir = run_dir / f"iter{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_points, columns=["kT", "JA", "mu"]).to_csv(iter_dir / "selected_points.csv", index=False)
    metadata.to_csv(iter_dir / "selected_points_metadata.csv", index=False)
    write_point_shards(run_dir, iteration, selected_points, world_size=world_size, strategy=partition_strategy)
    (iter_dir / "stagev_selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (iter_dir / "stagev_config_snapshot.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_stagev_dataset(path: Path) -> FlatDataset:
    return load_flat_dataset(path)
