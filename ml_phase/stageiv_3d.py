from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from eta_phase_diagram_cuda import EtaPhaseConfig

from .config import ActiveLearningConfig
from .dataset_builder import FlatDataset, load_flat_dataset
from .hpc import write_point_shards
from .labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC
from .models import predict_models, train_models
from .topology_oracle import TopologyModelParams, TopologyPfaffianOracle


STAGEIV_RUN_ID = "active_phase_topology_3d_t_ja_mu_from_scratch_v1"
STAGEIV_OUTPUT_ROOT = "ML_Phase_StageIV_Topology3D"


@dataclass(frozen=True)
class StageIV3DConfig:
    run_id: str = STAGEIV_RUN_ID
    output_root: str = STAGEIV_OUTPUT_ROOT
    kt_min: float = 0.0
    kt_max: float = 0.56
    ja_min: float = 0.0
    ja_max: float = 2.12
    mu_min: float = -0.5
    mu_max: float = 1.5
    mu_reference: float = 0.55
    guard_mu_min: float = -1.0
    guard_mu_max: float = 2.0
    t: float = 1.0
    u: float = 1.6
    lambda_ry: float = 0.6
    lambda_rz: float = 0.6
    initial_seed_size: int = 1024
    batch_size: int = 256
    max_acquisition_batches: int = 24
    candidate_pool_size: int = 65536
    random_seed: int = 20260623
    model_ensemble: int = 5
    reg_epochs: int = 180
    cls_epochs: int = 180
    hidden_dim: int = 96
    topology_phase_weight_early: float = 0.45
    topology_spectral_weight_early: float = 0.40
    coverage_weight_early: float = 0.15
    topology_phase_weight_late: float = 0.25
    topology_spectral_weight_late: float = 0.60
    coverage_weight_late: float = 0.15
    thermodynamic_stability_switch_iter: int = 12
    topo_pf_margin_scale: float = 0.02
    observation_repulsion_length: float = 0.04
    batch_repulsion_length: float = 0.045
    batch_repulsion_floor: float = 0.02
    selection_power: float = 3.0
    global_candidate_fraction: float = 0.70
    topology_bracket_candidate_fraction: float = 0.15
    thermodynamic_bracket_candidate_fraction: float = 0.10
    mu_edge_candidate_fraction: float = 0.05
    bracket_neighbor_k: int = 8
    bracket_max_distance: float = 0.20
    bracket_jitter_scale: float = 0.018
    mu_edge_guard_width: float = 0.04

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path) -> "StageIV3DConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in fields})

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.kt_min, self.ja_min, self.mu_min], dtype=np.float64)
        hi = np.array([self.kt_max, self.ja_max, self.mu_max], dtype=np.float64)
        return lo, hi


def active_config_from_stageiv(cfg: StageIV3DConfig) -> ActiveLearningConfig:
    return ActiveLearningConfig(
        run_mode="discovery",
        candidate_domain_mode="full",
        initialization="sobol_scrambled",
        initial_seed_size=int(cfg.initial_seed_size),
        batch_size_max=int(cfg.batch_size),
        points_per_iter=int(cfg.batch_size),
        iterations=int(cfg.max_acquisition_batches + 1),
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


def topology_params_from_stageiv(cfg: StageIV3DConfig) -> TopologyModelParams:
    return TopologyModelParams(
        t=float(cfg.t),
        lambda_ry=float(cfg.lambda_ry),
        lambda_rz=float(cfg.lambda_rz),
        mu=float(cfg.mu_reference),
    )


def exact_config_from_stageiv(cfg: StageIV3DConfig) -> EtaPhaseConfig:
    return EtaPhaseConfig(
        t=float(cfg.t),
        lambda_ry=float(cfg.lambda_ry),
        lambda_rz=float(cfg.lambda_rz),
        mu=float(cfg.mu_reference),
        u=float(cfg.u),
    )


def sobol_points_3d(n: int, cfg: StageIV3DConfig, seed_offset: int = 0) -> np.ndarray:
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=int(cfg.random_seed) + int(seed_offset))
    unit = engine.draw(n).cpu().numpy().astype(np.float64, copy=False)
    lo, hi = cfg.bounds()
    return lo + unit * (hi - lo)


def normalized_coordinates(points: np.ndarray, cfg: StageIV3DConfig) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo, hi = cfg.bounds()
    return (points - lo) / np.maximum(hi - lo, 1.0e-12)


def min_distance_3d(points: np.ndarray, ref: np.ndarray, cfg: StageIV3DConfig) -> np.ndarray:
    points_n = normalized_coordinates(points, cfg)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1, 3)
    if points_n.size == 0:
        return np.empty((0,), dtype=np.float64)
    if ref.size == 0:
        return np.full(points_n.shape[0], np.inf, dtype=np.float64)
    ref_n = normalized_coordinates(ref, cfg)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        dist, _ = cKDTree(ref_n).query(points_n, k=1)
        return np.asarray(dist, dtype=np.float64)
    except Exception:
        out = np.full(points_n.shape[0], np.inf, dtype=np.float64)
        for s in range(0, points_n.shape[0], 1024):
            chunk = points_n[s : s + 1024]
            d2 = np.sum((chunk[:, None, :] - ref_n[None, :, :]) ** 2, axis=2)
            out[s : s + chunk.shape[0]] = np.sqrt(np.min(d2, axis=1))
        return out


def dataset_points_3d(dataset: FlatDataset, cfg: StageIV3DConfig) -> np.ndarray:
    x = np.asarray(dataset.x, dtype=np.float64)
    if x.shape[1] == 3:
        return x
    mu = np.asarray(dataset.records.get("mu", np.full(x.shape[0], cfg.mu_reference)), dtype=np.float64)
    return np.column_stack([x[:, 0], x[:, 1], mu])


def clip_to_stageiv_bounds(points: np.ndarray, cfg: StageIV3DConfig) -> np.ndarray:
    lo, hi = cfg.bounds()
    return np.minimum(np.maximum(np.asarray(points, dtype=np.float64).reshape(-1, 3), lo), hi)


def stageiv_dataset_records(dataset: FlatDataset) -> pd.DataFrame:
    records = dict(dataset.records)
    if "mu" not in records:
        if dataset.x.shape[1] >= 3:
            records["mu"] = dataset.x[:, 2]
        else:
            records["mu"] = np.full(dataset.x.shape[0], 0.55, dtype=np.float64)
    return pd.DataFrame(records)


def topology_context_3d(dataset: FlatDataset) -> dict[str, np.ndarray]:
    x = dataset.x
    if x.shape[1] < 3:
        mu = dataset.records.get("mu", np.full(x.shape[0], 0.55, dtype=np.float64))
        x = np.column_stack([x[:, 0], x[:, 1], mu])
    topo_label = np.asarray(dataset.records.get("topology_label_code", np.full(x.shape[0], -1)), dtype=np.int64)
    topo_trusted = np.asarray(dataset.records.get("topology_trusted", np.zeros(x.shape[0])), dtype=np.int64).astype(bool)
    spectral = np.asarray(dataset.records.get("topology_spectral_status_code", np.full(x.shape[0], -1)), dtype=np.int64)
    phase = np.asarray(dataset.y_phase, dtype=np.int64)
    base = topo_trusted & (phase != PHASE_NORMAL) & (spectral == 0)
    return {
        "trusted_topology_points": x[base],
        "trivial_points": x[base & (topo_label == 0)],
        "topological_points": x[base & (topo_label == 1)],
        "gapless_points": x[topo_trusted & (topo_label == 2)],
    }


def _local_opposite_pairs(
    points: np.ndarray,
    labels: np.ndarray,
    cfg: StageIV3DConfig,
    *,
    max_pairs: int,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels)
    if points.shape[0] < 2 or max_pairs <= 0:
        return np.empty((0, 2), dtype=np.int64)
    points_n = normalized_coordinates(points, cfg)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        k = min(int(cfg.bracket_neighbor_k) + 1, points.shape[0])
        dist, idx = cKDTree(points_n).query(points_n, k=k)
        pairs: list[tuple[float, int, int]] = []
        for i in range(points.shape[0]):
            for d, j in zip(np.ravel(dist[i]), np.ravel(idx[i])):
                j = int(j)
                if j <= i:
                    continue
                if labels[i] == labels[j]:
                    continue
                if not np.isfinite(d) or float(d) > float(cfg.bracket_max_distance):
                    continue
                pairs.append((float(d), i, j))
        pairs.sort(key=lambda row: row[0])
        return np.asarray([(i, j) for _, i, j in pairs[: int(max_pairs)]], dtype=np.int64)
    except Exception:
        pairs = []
        for i in range(points.shape[0]):
            d = np.sqrt(np.sum((points_n - points_n[i]) ** 2, axis=1))
            for j in np.argsort(d)[1 : int(cfg.bracket_neighbor_k) + 1]:
                if j <= i or labels[i] == labels[j] or d[j] > float(cfg.bracket_max_distance):
                    continue
                pairs.append((float(d[j]), i, int(j)))
        pairs.sort(key=lambda row: row[0])
        return np.asarray([(i, j) for _, i, j in pairs[: int(max_pairs)]], dtype=np.int64)


def _jitter_midpoints(
    points: np.ndarray,
    pairs: np.ndarray,
    cfg: StageIV3DConfig,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    if count <= 0 or pairs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    idx = rng.integers(0, pairs.shape[0], size=int(count))
    a = points[pairs[idx, 0]]
    b = points[pairs[idx, 1]]
    mid = 0.5 * (a + b)
    lo, hi = cfg.bounds()
    span = hi - lo
    jitter = rng.normal(0.0, float(cfg.bracket_jitter_scale), size=mid.shape) * span
    return clip_to_stageiv_bounds(mid + jitter, cfg)


def _split_candidate_counts(cfg: StageIV3DConfig) -> dict[str, int]:
    total = int(cfg.candidate_pool_size)
    fractions = {
        "global_sobol": max(float(cfg.global_candidate_fraction), 0.0),
        "topology_bracket_jitter": max(float(cfg.topology_bracket_candidate_fraction), 0.0),
        "thermodynamic_bracket_jitter": max(float(cfg.thermodynamic_bracket_candidate_fraction), 0.0),
        "mu_edge_guard": max(float(cfg.mu_edge_candidate_fraction), 0.0),
    }
    denom = sum(fractions.values()) or 1.0
    counts = {k: int(round(total * v / denom)) for k, v in fractions.items()}
    delta = total - sum(counts.values())
    counts["global_sobol"] += delta
    return counts


def generate_stageiv_candidate_pool(
    dataset: FlatDataset,
    cfg: StageIV3DConfig,
    iteration: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.random_seed) + 7717 * int(iteration))
    counts = _split_candidate_counts(cfg)
    blocks: list[np.ndarray] = []
    labels: list[str] = []

    global_points = sobol_points_3d(counts["global_sobol"], cfg, seed_offset=1000 + int(iteration))
    blocks.append(global_points)
    labels.extend(["global_sobol"] * global_points.shape[0])

    data_points = dataset_points_3d(dataset, cfg)
    context = topology_context_3d(dataset)
    topo_points = np.vstack([context["trivial_points"], context["topological_points"]]) if (
        context["trivial_points"].size and context["topological_points"].size
    ) else np.empty((0, 3), dtype=np.float64)
    topo_labels = np.concatenate(
        [
            np.zeros(context["trivial_points"].shape[0], dtype=np.int8),
            np.ones(context["topological_points"].shape[0], dtype=np.int8),
        ]
    ) if topo_points.shape[0] else np.empty((0,), dtype=np.int8)
    topo_pairs = _local_opposite_pairs(topo_points, topo_labels, cfg, max_pairs=counts["topology_bracket_jitter"])
    topo_jitter = _jitter_midpoints(topo_points, topo_pairs, cfg, rng, counts["topology_bracket_jitter"])
    blocks.append(topo_jitter)
    labels.extend(["topology_bracket_jitter"] * topo_jitter.shape[0])

    phase_labels = np.asarray(dataset.y_phase, dtype=np.int64)
    thermo_pairs = _local_opposite_pairs(data_points, phase_labels, cfg, max_pairs=counts["thermodynamic_bracket_jitter"])
    thermo_jitter = _jitter_midpoints(data_points, thermo_pairs, cfg, rng, counts["thermodynamic_bracket_jitter"])
    blocks.append(thermo_jitter)
    labels.extend(["thermodynamic_bracket_jitter"] * thermo_jitter.shape[0])

    edge_count = counts["mu_edge_guard"]
    edge_points = sobol_points_3d(edge_count, cfg, seed_offset=2000 + int(iteration))
    if edge_count:
        lo, hi = cfg.bounds()
        half = edge_count // 2
        width = max(float(cfg.mu_edge_guard_width), 1.0e-6) * (hi[2] - lo[2])
        edge_points[:half, 2] = lo[2] + rng.random(half) * width
        edge_points[half:, 2] = hi[2] - rng.random(edge_count - half) * width
    blocks.append(edge_points)
    labels.extend(["mu_edge_guard"] * edge_points.shape[0])

    points = np.vstack([b for b in blocks if b.size]) if any(b.size for b in blocks) else np.empty((0, 3), dtype=np.float64)
    source = np.asarray(labels, dtype=object)
    if points.shape[0] < int(cfg.candidate_pool_size):
        missing = int(cfg.candidate_pool_size) - int(points.shape[0])
        backfill = sobol_points_3d(missing, cfg, seed_offset=3000 + int(iteration))
        points = np.vstack([points, backfill]) if points.size else backfill
        source = np.concatenate([source, np.asarray(["global_sobol_backfill"] * missing, dtype=object)])
    if points.shape[0] == 0:
        return points, pd.DataFrame(columns=["candidate_source"]), {"candidate_source_counts": {}}
    rounded = np.round(points, decimals=8)
    _, first_idx = np.unique(rounded, axis=0, return_index=True)
    first_idx = np.sort(first_idx)
    points = points[first_idx]
    source = source[first_idx]
    meta = pd.DataFrame({"candidate_source": source})
    summary = {
        "candidate_source_counts": meta["candidate_source"].value_counts().to_dict(),
        "requested_candidate_source_counts": counts,
        "topology_bracket_pair_count": int(topo_pairs.shape[0]),
        "thermodynamic_bracket_pair_count": int(thermo_pairs.shape[0]),
    }
    return points, meta, summary


def _empty_float(n: int) -> np.ndarray:
    return np.empty((int(n),), dtype=np.float64)


def write_empty_stageiv_dataset(run_dir: Path, iteration: int = 0) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    npz_path = run_dir / f"dataset_iter{iteration:03d}.npz"
    csv_path = run_dir / f"dataset_iter{iteration:03d}.csv"
    payload = {
        "x": np.empty((0, 3), dtype=np.float64),
        "y_reg": np.empty((0, 5), dtype=np.float64),
        "y_phase": np.empty((0,), dtype=np.int64),
        "y_eta_sign": np.empty((0,), dtype=np.int64),
        "y_strong_diode": np.empty((0,), dtype=np.int64),
        "mu": _empty_float(n),
    }
    if not npz_path.exists():
        np.savez(npz_path, **payload)
    if not csv_path.exists():
        pd.DataFrame(columns=["kT", "JA", "mu", "delta_opt", "q_opt", "eta", "ic_plus", "ic_minus", "phase_label"]).to_csv(
            csv_path,
            index=False,
        )
    return npz_path, csv_path


def _phase_probabilities(
    pred: dict[str, np.ndarray],
    classes: np.ndarray | None,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(pred.get("phase_proba", np.empty((n, 0))), dtype=np.float64)
    phase_pred = np.asarray(pred.get("phase_pred", np.full(n, -1)), dtype=np.int64)
    p_normal = (phase_pred == PHASE_NORMAL).astype(np.float64)
    p_uniform = (phase_pred == PHASE_UNIFORM_SC).astype(np.float64)
    p_fflo = (phase_pred == PHASE_FFLO).astype(np.float64)
    if classes is not None and probs.shape[1] == len(classes):
        for col, cls in enumerate(np.asarray(classes, dtype=np.int64)):
            if cls == PHASE_NORMAL:
                p_normal = probs[:, col]
            elif cls == PHASE_UNIFORM_SC:
                p_uniform = probs[:, col]
            elif cls == PHASE_FFLO:
                p_fflo = probs[:, col]
    p_sc = np.clip(p_uniform + p_fflo, 0.0, 1.0)
    return p_normal, p_uniform, p_fflo, p_sc


def score_stageiv_candidates(
    dataset: FlatDataset,
    candidate_points: np.ndarray,
    cfg: StageIV3DConfig,
    iteration: int,
    device: torch.device | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    active_cfg = active_config_from_stageiv(cfg)
    bundle = train_models(dataset.x, dataset.y_reg, dataset.y_phase, active_cfg, device=device)
    pred = predict_models(bundle, candidate_points)
    reg_mean = np.asarray(pred["reg_mean"], dtype=np.float64)
    reg_std = np.asarray(pred["reg_std"], dtype=np.float64)
    delta_pred = np.maximum(reg_mean[:, 0], 0.0)
    q_pred = reg_mean[:, 1]
    n = candidate_points.shape[0]
    p_normal, p_uniform, p_fflo, p_sc = _phase_probabilities(pred, bundle.classifier.classes_, n)

    u_cls = np.asarray(pred["cls_entropy"], dtype=np.float64)
    u_reg_phase = np.mean(reg_std[:, :2], axis=1)
    u_reg_phase = u_reg_phase / max(float(np.nanpercentile(u_reg_phase, 95)), 1.0e-12)
    phase_score = np.clip(0.65 * u_cls + 0.35 * np.clip(u_reg_phase, 0.0, 1.0), 0.0, 1.0)

    pf = TopologyPfaffianOracle(topology_params_from_stageiv(cfg))
    p0, ppi, product, margin = pf.analytic_pfaffians(
        delta_pred,
        q_pred,
        candidate_points[:, 1],
        mu=candidate_points[:, 2],
    )
    margin_score = np.exp(-np.clip(margin, 0.0, np.inf) / max(float(cfg.topo_pf_margin_scale), 1.0e-12)) * p_sc

    context = topology_context_3d(dataset)
    d_trivial = min_distance_3d(candidate_points, context["trivial_points"], cfg)
    d_topological = min_distance_3d(candidate_points, context["topological_points"], cfg)
    has_bracket = np.isfinite(d_trivial) & np.isfinite(d_topological)
    z2_edge = np.zeros(n, dtype=np.float64)
    if np.any(has_bracket):
        near_both = np.exp(-np.minimum(d_trivial, d_topological) / 0.08)
        balanced = np.exp(-np.abs(d_trivial - d_topological) / 0.08)
        z2_edge = near_both * balanced * p_sc
    spectral_score = np.clip(np.maximum(margin_score, z2_edge), 0.0, 1.0)

    d_existing = min_distance_3d(candidate_points, dataset_points_3d(dataset, cfg), cfg)
    coverage_score = np.clip(1.0 - np.exp(-((d_existing / max(float(cfg.observation_repulsion_length), 1.0e-12)) ** 2)), 0.0, 1.0)

    if int(iteration) >= int(cfg.thermodynamic_stability_switch_iter):
        w_phase = float(cfg.topology_phase_weight_late)
        w_spectral = float(cfg.topology_spectral_weight_late)
        w_cov = float(cfg.coverage_weight_late)
    else:
        w_phase = float(cfg.topology_phase_weight_early)
        w_spectral = float(cfg.topology_spectral_weight_early)
        w_cov = float(cfg.coverage_weight_early)
    score = w_phase * phase_score + w_spectral * spectral_score + w_cov * coverage_score
    keys = np.round(candidate_points, decimals=6)
    existing = np.round(dataset_points_3d(dataset, cfg), decimals=6)
    existing_set = {tuple(row) for row in existing}
    duplicate = np.array([tuple(row) in existing_set for row in keys], dtype=bool)
    score = np.where(duplicate, -np.inf, score)
    scores = {
        "score": score,
        "A_phase": phase_score,
        "A_spectral": spectral_score,
        "A_coverage": coverage_score,
        "topology_pfaffian_p0_pred": p0,
        "topology_pfaffian_ppi_pred": ppi,
        "topology_pfaffian_product_pred": product,
        "topology_pfaffian_margin_pred": margin,
        "P_normal": p_normal,
        "P_uniform": p_uniform,
        "P_FFLO": p_fflo,
        "P_SC": p_sc,
        "candidate_duplicate_existing": duplicate.astype(np.int8),
        "distance_to_existing_3d": d_existing,
        "distance_to_trivial_3d": d_trivial,
        "distance_to_topological_3d": d_topological,
    }
    summary = {
        "iteration": int(iteration),
        "candidate_count": int(candidate_points.shape[0]),
        "duplicate_count": int(np.sum(duplicate)),
        "topology_context_counts": {k: int(v.shape[0]) for k, v in context.items()},
        "weights": {"phase": w_phase, "spectral": w_spectral, "coverage": w_cov},
    }
    return scores, summary


def _channel_sequence(cfg: StageIV3DConfig, iteration: int) -> list[str]:
    if int(iteration) >= int(cfg.thermodynamic_stability_switch_iter):
        weights = {
            "phase": float(cfg.topology_phase_weight_late),
            "spectral": float(cfg.topology_spectral_weight_late),
            "coverage": float(cfg.coverage_weight_late),
        }
    else:
        weights = {
            "phase": float(cfg.topology_phase_weight_early),
            "spectral": float(cfg.topology_spectral_weight_early),
            "coverage": float(cfg.coverage_weight_early),
        }
    total = sum(max(v, 0.0) for v in weights.values()) or 1.0
    quotas = {k: int(round(int(cfg.batch_size) * max(v, 0.0) / total)) for k, v in weights.items()}
    quotas["coverage"] += int(cfg.batch_size) - sum(quotas.values())
    seq: list[str] = []
    for channel in ["phase", "spectral", "coverage"]:
        seq.extend([channel] * max(int(quotas[channel]), 0))
    return seq[: int(cfg.batch_size)]


def select_stageiv_batch(
    candidate_points: np.ndarray,
    scores: dict[str, np.ndarray],
    cfg: StageIV3DConfig,
    iteration: int,
    candidate_metadata: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    score = np.asarray(scores["score"], dtype=np.float64).copy()
    channel_scores = {
        "phase": np.asarray(scores["A_phase"], dtype=np.float64).copy(),
        "spectral": np.asarray(scores["A_spectral"], dtype=np.float64).copy(),
        "coverage": np.asarray(scores["A_coverage"], dtype=np.float64).copy(),
    }
    source = (
        candidate_metadata["candidate_source"].astype(str).to_numpy()
        if candidate_metadata is not None and "candidate_source" in candidate_metadata
        else np.asarray(["unknown"] * candidate_points.shape[0], dtype=object)
    )
    selected: list[int] = []
    rows: list[dict[str, Any]] = []
    coords_n = normalized_coordinates(candidate_points, cfg)
    r_batch = np.ones(candidate_points.shape[0], dtype=np.float64)
    rng = np.random.default_rng(int(cfg.random_seed) + int(iteration) * 1000003)
    channel_seq = _channel_sequence(cfg, iteration)
    quota_fulfilled = {"phase": 0, "spectral": 0, "coverage": 0, "backfill": 0}
    for rank in range(int(cfg.batch_size)):
        requested_channel = channel_seq[rank] if rank < len(channel_seq) else "backfill"
        base = channel_scores.get(requested_channel, score)
        final = base * r_batch
        eligible = np.isfinite(final) & (final > 0.0)
        backfilled = False
        if not np.any(eligible):
            final = score * r_batch
            eligible = np.isfinite(final) & (final > 0.0)
            backfilled = True
        if not np.any(eligible):
            break
        weights = np.zeros_like(final)
        weights[eligible] = np.maximum(final[eligible], 0.0) ** float(cfg.selection_power)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            final = score * r_batch
            eligible = np.isfinite(final) & (final > 0.0)
            weights = np.zeros_like(final)
            weights[eligible] = np.maximum(final[eligible], 0.0) ** float(cfg.selection_power)
            total = float(np.sum(weights))
            backfilled = True
        if not np.isfinite(total) or total <= 0.0:
            break
        prob = weights / total
        idx = int(rng.choice(np.arange(candidate_points.shape[0]), p=prob))
        selected.append(idx)
        point = candidate_points[idx]
        actual_channel = "backfill" if backfilled else requested_channel
        quota_fulfilled[actual_channel] = quota_fulfilled.get(actual_channel, 0) + 1
        rows.append(
            {
                "selection_rank": rank + 1,
                "selection_source": f"stageiv_3d_{actual_channel}_quota",
                "acquisition_channel": actual_channel,
                "requested_acquisition_channel": requested_channel,
                "candidate_source": str(source[idx]),
                "kT": float(point[0]),
                "JA": float(point[1]),
                "mu": float(point[2]),
                "score": float(score[idx]),
                "channel_score_before_pick": float(final[idx]),
                "A_phase": float(scores["A_phase"][idx]),
                "A_spectral": float(scores["A_spectral"][idx]),
                "A_coverage": float(scores["A_coverage"][idx]),
                "P_normal": float(scores["P_normal"][idx]),
                "P_uniform": float(scores["P_uniform"][idx]),
                "P_FFLO": float(scores["P_FFLO"][idx]),
                "P_SC": float(scores["P_SC"][idx]),
                "topology_pfaffian_margin_pred": float(scores["topology_pfaffian_margin_pred"][idx]),
                "distance_to_existing_3d": float(scores["distance_to_existing_3d"][idx]),
                "sampling_probability_before_pick": float(prob[idx]),
            }
        )
        score[idx] = -np.inf
        for arr in channel_scores.values():
            arr[idx] = -np.inf
        dist = np.sqrt(np.sum((coords_n - coords_n[idx]) ** 2, axis=1))
        factor = float(cfg.batch_repulsion_floor) + (1.0 - float(cfg.batch_repulsion_floor)) * (
            1.0 - np.exp(-((dist / max(float(cfg.batch_repulsion_length), 1.0e-12)) ** 2))
        )
        r_batch *= np.clip(factor, float(cfg.batch_repulsion_floor), 1.0)
    selected_points = candidate_points[np.asarray(selected, dtype=np.int64)] if selected else np.empty((0, 3), dtype=np.float64)
    summary = {
        "requested_batch_size": int(cfg.batch_size),
        "selected_batch_size": int(selected_points.shape[0]),
        "underfilled": bool(selected_points.shape[0] < int(cfg.batch_size)),
        "channel_requested_counts": {k: int(channel_seq.count(k)) for k in ["phase", "spectral", "coverage"]},
        "channel_fulfilled_counts": {k: int(v) for k, v in quota_fulfilled.items()},
    }
    return selected_points, pd.DataFrame(rows), summary


def write_stageiv_selection(
    run_dir: Path,
    iteration: int,
    selected_points: np.ndarray,
    metadata: pd.DataFrame,
    cfg: StageIV3DConfig,
    world_size: int,
    partition_strategy: str,
    summary: dict[str, Any],
) -> None:
    iter_dir = run_dir / f"iter{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_points, columns=["kT", "JA", "mu"]).to_csv(iter_dir / "selected_points.csv", index=False)
    metadata.to_csv(iter_dir / "selected_points_metadata.csv", index=False)
    write_point_shards(run_dir, iteration, selected_points, world_size=world_size, strategy=partition_strategy)
    (iter_dir / "stageiv_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (iter_dir / "stageiv_config_snapshot.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def normal_band_count_scan(
    cfg: StageIV3DConfig,
    n_ja: int = 96,
    n_mu: int = 128,
    n_k: int = 2048,
) -> pd.DataFrame:
    ja_vals = np.linspace(float(cfg.ja_min), float(cfg.ja_max), int(n_ja), dtype=np.float64)
    mu_vals = np.linspace(float(cfg.guard_mu_min), float(cfg.guard_mu_max), int(n_mu), dtype=np.float64)
    k = np.linspace(-math.pi, math.pi, int(n_k), endpoint=False, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    sin_k = np.sin(k)
    cos_k = np.cos(k)
    for ja in ja_vals:
        d_norm = np.sqrt((cfg.lambda_ry * sin_k) ** 2 + (cfg.lambda_rz * sin_k + ja * cos_k) ** 2)
        band_plus_base = cfg.t * cos_k + d_norm
        band_minus_base = cfg.t * cos_k - d_norm
        for mu in mu_vals:
            e_plus = band_plus_base - mu
            e_minus = band_minus_base - mu
            crossings = 0
            for e in (e_plus, e_minus):
                crossings += int(np.sum(np.signbit(e) != np.signbit(np.roll(e, -1))))
            min_abs = float(min(np.min(np.abs(e_plus)), np.min(np.abs(e_minus))))
            rows.append(
                {
                    "JA": float(ja),
                    "mu": float(mu),
                    "fermi_crossing_count": int(crossings),
                    "fermi_point_pair_count": int(crossings // 2),
                    "min_abs_band_energy": min_abs,
                    "single_pair_diagnostic": int(crossings == 2),
                    "multi_pair_diagnostic": int(crossings > 2),
                    "no_fs_diagnostic": int(crossings == 0),
                    "band_plus_min": float(np.min(e_plus)),
                    "band_plus_max": float(np.max(e_plus)),
                    "band_minus_min": float(np.min(e_minus)),
                    "band_minus_max": float(np.max(e_minus)),
                }
            )
    return pd.DataFrame(rows)
