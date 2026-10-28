from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset_builder import FlatDataset
from .labels import PHASE_FFLO, PHASE_NORMAL
from .stageiv_3d import min_distance_3d
from .stagev_v2 import (
    BOUNDARY_NAMES,
    BOUNDARY_SCORE_SUFFIX,
    StageV2Config,
    build_boundary_support_sets,
    combine_multihead_scores,
    compute_per_boundary_rewards,
    fit_multihead_value_models,
    generate_stagev_candidate_pool,
    initial_alpha_state,
    initial_lambda_state,
    load_stagev2_state,
    predict_stagev_fields,
    score_stagev_a0,
    select_micro_batch_v2,
    sobol_points_3d,
    topology_channel_diagnostics,
    update_boundary_alphas,
    validation_summary_by_boundary,
    write_empty_stagev_dataset,
    write_stagev2_state,
    write_stagev_selection,
)
from .stagev_acqv2 import clip_to_bounds, normalized_coordinates


STAGEV3_RUN_ID = "stagev_v3_sparse_arc_boundary_support_3d_v1"
STAGEV3_OUTPUT_ROOT = "ML_Phase_StageV_V3_SparseArc"

TOPOLOGY_BOUNDARIES = ("p0", "ppi")
ALLOWED_PROPOSAL_SOURCES = {
    "stagev3_scrambled_sobol_seed",
    "initial_sobol",
    "global_sobol",
    "p0_zero_surface_sparse_patch",
    "ppi_zero_surface_sparse_patch",
    "opposite_Z2_bracket_gap",
    "topology_surface_normal_jitter",
    "topology_surface_tangent_jitter",
    "component_boundary_guard",
    "sparse_patch_fill_distance",
    "gap_topology_relevant",
    "ns_surface",
    "uf_surface",
    "fallback",
    "unknown_explicit",
}


@dataclass(frozen=True)
class StageV3Config(StageV2Config):
    run_id: str = STAGEV3_RUN_ID
    output_root: str = STAGEV3_OUTPUT_ROOT
    random_seed: int = 20260706
    stagev2_schema_version: str = "stagev_v3_sparse_arc_acq_1"
    topology_sparse_quantile: float = 0.95
    topology_sparse_patch_radius: float = 0.055
    topology_sparse_candidate_radius: float = 0.080
    topology_surface_jitter_scale: float = 0.020
    topology_surface_tangent_scale: float = 0.030
    topology_coverage_weight: float = 1.75
    topology_coverage_floor: float = 0.05
    topology_min_surface_samples: int = 32
    gap_relevance_floor: float = 0.05
    gap_relevance_power: float = 1.0
    sparse_surface_max_edges: int = 6000
    sparse_replay_top_contributors: int = 256
    proposal_source_validation_strict: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path) -> "StageV3Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in fields})


def _trusted_training_mask(dataset: FlatDataset) -> np.ndarray:
    n = int(dataset.x.shape[0])
    trusted = np.asarray(dataset.records.get("trusted_exact", np.ones(n, dtype=np.int8))).astype(bool)
    eligible = np.asarray(dataset.records.get("training_eligible_exact", np.ones(n, dtype=np.int8))).astype(bool)
    rerun = np.asarray(dataset.records.get("needs_rerun_exact", np.zeros(n, dtype=np.int8))).astype(bool)
    q_bad = np.asarray(dataset.records.get("q_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)
    d_bad = np.asarray(dataset.records.get("delta_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)
    return trusted & eligible & ~rerun & ~q_bad & ~d_bad


def _topology_mask(dataset: FlatDataset) -> np.ndarray:
    n = int(dataset.x.shape[0])
    phase = np.asarray(dataset.y_phase, dtype=np.int64)
    topo_trusted = np.asarray(dataset.records.get("topology_trusted", np.zeros(n, dtype=np.int8))).astype(bool)
    spectral = np.asarray(dataset.records.get("topology_spectral_status_code", np.full(n, -1)), dtype=np.int64)
    p0 = np.asarray(dataset.records.get("topology_p0", np.full(n, np.nan)), dtype=np.float64)
    ppi = np.asarray(dataset.records.get("topology_ppi", np.full(n, np.nan)), dtype=np.float64)
    return _trusted_training_mask(dataset) & topo_trusted & (phase == PHASE_FFLO) & (spectral == 0) & np.isfinite(p0) & np.isfinite(ppi)


def _pair_edges(points_n: np.ndarray, k: int, max_distance: float) -> list[tuple[int, int, float]]:
    if points_n.shape[0] < 2:
        return []
    kk = min(max(int(k) + 1, 2), points_n.shape[0])
    try:
        from scipy.spatial import cKDTree  # type: ignore

        dist, nn = cKDTree(points_n).query(points_n, k=kk)
    except Exception:
        dmat = np.sqrt(np.sum((points_n[:, None, :] - points_n[None, :, :]) ** 2, axis=2))
        nn = np.argsort(dmat, axis=1)[:, :kk]
        dist = np.take_along_axis(dmat, nn, axis=1)
    nn = np.asarray(nn)
    dist = np.asarray(dist, dtype=np.float64)
    if nn.ndim == 1:
        nn = nn[:, None]
        dist = dist[:, None]
    neighbor_sets = [set(np.ravel(row[1:]).astype(int).tolist()) for row in nn]
    edges: list[tuple[int, int, float]] = []
    for i in range(points_n.shape[0]):
        for d, j0 in zip(np.ravel(dist[i])[1:], np.ravel(nn[i])[1:]):
            j = int(j0)
            if j <= i or i not in neighbor_sets[j]:
                continue
            if not np.isfinite(d) or float(d) > float(max_distance):
                continue
            edges.append((i, j, float(d)))
    return edges


def _component_ids(points_n: np.ndarray, radius: float) -> np.ndarray:
    n = int(points_n.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return int(i)

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    try:
        from scipy.spatial import cKDTree  # type: ignore

        pairs = cKDTree(points_n).query_pairs(float(radius))
        for i, j in pairs:
            union(int(i), int(j))
    except Exception:
        dmat = np.sqrt(np.sum((points_n[:, None, :] - points_n[None, :, :]) ** 2, axis=2))
        ii, jj = np.where(np.triu(dmat <= float(radius), k=1))
        for i, j in zip(ii, jj):
            union(int(i), int(j))
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    unique = {root: idx for idx, root in enumerate(sorted(set(roots.tolist())))}
    return np.array([unique[int(root)] for root in roots], dtype=np.int64)


def build_sparse_topology_surface(dataset: FlatDataset, cfg: StageV3Config) -> pd.DataFrame:
    points = np.asarray(dataset.x, dtype=np.float64).reshape(-1, 3)
    mask = _topology_mask(dataset)
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return pd.DataFrame(columns=_surface_columns())
    pts = points[idx]
    pts_n = normalized_coordinates(pts, cfg)
    p0 = np.asarray(dataset.records.get("topology_p0"), dtype=np.float64)[idx]
    ppi = np.asarray(dataset.records.get("topology_ppi"), dtype=np.float64)[idx]
    z2 = np.asarray(dataset.records.get("topology_label_code", np.full(points.shape[0], -1)), dtype=np.int64)[idx]
    edges = _pair_edges(pts_n, int(cfg.bracket_neighbor_k), float(cfg.bracket_max_distance))
    rows: list[dict[str, Any]] = []
    for i, j, d in edges:
        p0_cross = bool(np.signbit(p0[i]) != np.signbit(p0[j]))
        ppi_cross = bool(np.signbit(ppi[i]) != np.signbit(ppi[j]))
        z2_cross = bool(z2[i] != z2[j])
        if not (z2_cross or p0_cross or ppi_cross):
            continue
        midpoint = 0.5 * (pts[i] + pts[j])
        rows.append(
            {
                "kT": float(midpoint[0]),
                "JA": float(midpoint[1]),
                "mu": float(midpoint[2]),
                "edge_i": int(idx[i]),
                "edge_j": int(idx[j]),
                "edge_length_norm": float(d),
                "p0_cross": p0_cross,
                "ppi_cross": ppi_cross,
                "z2_cross": z2_cross,
                "dominant_pfaffian": "P0" if p0_cross and not ppi_cross else ("Ppi" if ppi_cross and not p0_cross else "mixed"),
            }
        )
        if len(rows) >= int(cfg.sparse_surface_max_edges):
            break
    if not rows:
        return pd.DataFrame(columns=_surface_columns())
    frame = pd.DataFrame(rows)
    surface_points = frame[["kT", "JA", "mu"]].to_numpy(float)
    support_points = pts
    support_distance = min_distance_3d(surface_points, support_points, cfg)
    frame["support_distance"] = support_distance
    frame["bracket_distance"] = 0.5 * frame["edge_length_norm"].to_numpy(float)
    frame["coverage_metric"] = np.maximum(frame["support_distance"].to_numpy(float), frame["bracket_distance"].to_numpy(float))
    sparse_threshold = float(np.nanquantile(frame["coverage_metric"], float(cfg.topology_sparse_quantile)))
    frame["sparse_threshold"] = sparse_threshold
    frame["is_sparse_patch"] = frame["coverage_metric"].to_numpy(float) >= sparse_threshold
    comp = _component_ids(normalized_coordinates(surface_points, cfg), float(cfg.topology_sparse_patch_radius))
    frame["component_id"] = comp
    component_size = frame.groupby("component_id")["component_id"].transform("size")
    frame["component_size"] = component_size.astype(int)
    return frame


def _surface_columns() -> list[str]:
    return [
        "kT",
        "JA",
        "mu",
        "edge_i",
        "edge_j",
        "edge_length_norm",
        "p0_cross",
        "ppi_cross",
        "z2_cross",
        "dominant_pfaffian",
        "support_distance",
        "bracket_distance",
        "coverage_metric",
        "sparse_threshold",
        "is_sparse_patch",
        "component_id",
        "component_size",
    ]


def expected_topology_coverage_improvement(
    candidates: np.ndarray,
    surface: pd.DataFrame,
    cfg: StageV3Config,
) -> pd.DataFrame:
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 3)
    n = int(candidates.shape[0])
    if surface.empty:
        return pd.DataFrame(
            {
                "C_topo_raw": np.zeros(n),
                "C_topo": np.full(n, float(cfg.topology_coverage_floor)),
                "topology_sparse_nearest_distance": np.full(n, np.inf),
                "topology_sparse_component": np.full(n, -1, dtype=np.int64),
                "topology_sparse_surface_available": np.zeros(n, dtype=np.int8),
            }
        )
    sparse = surface[surface["is_sparse_patch"].astype(bool)].copy()
    if sparse.empty:
        sparse = surface.nlargest(min(len(surface), max(1, int(cfg.sparse_replay_top_contributors))), "coverage_metric").copy()
    surf_pts = sparse[["kT", "JA", "mu"]].to_numpy(float)
    surf_n = normalized_coordinates(surf_pts, cfg)
    cand_n = normalized_coordinates(candidates, cfg)
    old = sparse["coverage_metric"].to_numpy(float)
    radius = max(float(cfg.topology_sparse_candidate_radius), 1.0e-12)
    raw = np.zeros(n, dtype=np.float64)
    nearest = np.full(n, np.inf, dtype=np.float64)
    component = np.full(n, -1, dtype=np.int64)
    comp_values = sparse["component_id"].to_numpy(dtype=np.int64)
    for i in range(n):
        d = np.sqrt(np.sum((surf_n - cand_n[i]) ** 2, axis=1))
        nearest[i] = float(np.nanmin(d)) if d.size else np.inf
        if d.size:
            component[i] = int(comp_values[int(np.nanargmin(d))])
        local = d <= radius
        if not np.any(local):
            continue
        improvement = np.maximum(old[local] - np.minimum(old[local], d[local]), 0.0)
        weights = np.maximum(old[local], 1.0e-12)
        raw[i] = float(np.sum(improvement * weights) / np.maximum(np.sum(weights), 1.0e-12))
    scale = float(np.nanpercentile(raw[raw > 0], 95)) if np.any(raw > 0) else 1.0
    c = np.clip(raw / max(scale, 1.0e-12), 0.0, 1.0)
    factor = np.maximum(float(cfg.topology_coverage_floor), c)
    return pd.DataFrame(
        {
            "C_topo_raw": raw,
            "C_topo": factor,
            "topology_sparse_nearest_distance": nearest,
            "topology_sparse_component": component,
            "topology_sparse_surface_available": np.ones(n, dtype=np.int8),
        }
    )


def apply_stagev3_sparse_scores(
    scored: pd.DataFrame,
    dataset: FlatDataset,
    cfg: StageV3Config,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    frame = scored.copy().reset_index(drop=True)
    surface = build_sparse_topology_surface(dataset, cfg)
    points = frame[["kT", "JA", "mu"]].to_numpy(float)
    topo = expected_topology_coverage_improvement(points, surface, cfg)
    frame = pd.concat([frame.reset_index(drop=True), topo.reset_index(drop=True)], axis=1)
    for suffix in ("p0_topology", "ppi_topology"):
        if f"A_{suffix}" not in frame:
            frame[f"A_{suffix}"] = 0.0
        frame[f"A_{suffix}_pre_stagev3"] = frame[f"A_{suffix}"].to_numpy(float)
        frame[f"A_{suffix}"] = frame[f"A_{suffix}"].to_numpy(float) * (
            float(cfg.topology_coverage_weight) * frame["C_topo"].to_numpy(float)
        )
        frame[f"H_{suffix}"] = np.maximum(frame.get(f"H_{suffix}", 0.0), frame["C_topo"].to_numpy(float))
    p0_rel = frame.get("B_p0_topology", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    ppi_rel = frame.get("B_ppi_topology", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    topo_rel = np.maximum.reduce([p0_rel, ppi_rel, frame["C_topo"].to_numpy(float)])
    gap_relevance = np.clip(topo_rel, float(cfg.gap_relevance_floor), 1.0) ** float(cfg.gap_relevance_power)
    frame["gap_topology_relevance"] = gap_relevance
    if "A_gap_nodal" in frame:
        frame["A_gap_nodal_pre_stagev3"] = frame["A_gap_nodal"].to_numpy(float)
        frame["A_gap_nodal"] = frame["A_gap_nodal"].to_numpy(float) * gap_relevance
    summary = {
        "topology_sparse_surface_samples": int(surface.shape[0]),
        "topology_sparse_patch_samples": int(surface["is_sparse_patch"].sum()) if not surface.empty else 0,
        "topology_sparse_component_count": int(surface["component_id"].nunique()) if not surface.empty else 0,
        "topology_sparse_coverage_p95": float(surface["coverage_metric"].quantile(0.95)) if not surface.empty else None,
        "topology_c_topo_available": bool(not surface.empty),
        "topology_c_topo_mean": float(frame["C_topo"].mean()) if len(frame) else 0.0,
        "gap_topology_relevance_mean": float(frame["gap_topology_relevance"].mean()) if len(frame) else 0.0,
    }
    return frame, summary, surface


def generate_sparse_topology_proposals(
    dataset: FlatDataset,
    cfg: StageV3Config,
    iteration: int,
    count: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.random_seed) + 9173 * int(iteration))
    surface = build_sparse_topology_surface(dataset, cfg)
    if count <= 0:
        return np.empty((0, 3), dtype=np.float64), pd.DataFrame({"candidate_source": []}), {"sparse_proposal_status": "disabled"}
    if surface.empty or surface.shape[0] < int(cfg.topology_min_surface_samples):
        pts = sobol_points_3d(count, cfg, seed_offset=7000 + int(iteration))
        meta = pd.DataFrame({"candidate_source": ["fallback_global_sobol"] * pts.shape[0]})
        meta["candidate_source"] = "fallback"
        return pts, meta, {"sparse_proposal_status": "insufficient_topology_surface_support", "sparse_surface_samples": int(surface.shape[0])}
    sparse = surface[surface["is_sparse_patch"].astype(bool)].copy()
    if sparse.empty:
        sparse = surface.nlargest(min(count, len(surface)), "coverage_metric").copy()
    weights = np.maximum(sparse["coverage_metric"].to_numpy(float), 1.0e-12)
    weights = weights / np.sum(weights)
    base_idx = rng.choice(np.arange(sparse.shape[0]), size=count, replace=True, p=weights)
    base = sparse.iloc[base_idx][["kT", "JA", "mu"]].to_numpy(float)
    lo, hi = cfg.bounds()
    span = hi - lo
    normals = rng.normal(0.0, 1.0, size=(count, 3))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    tangents = rng.normal(0.0, 1.0, size=(count, 3))
    tangents -= np.sum(tangents * normals, axis=1, keepdims=True) * normals
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1.0e-12)
    mode = np.arange(count) % 5
    jitter = np.zeros_like(base)
    jitter[mode == 0] = normals[mode == 0] * float(cfg.topology_surface_jitter_scale) * span
    jitter[mode == 1] = -normals[mode == 1] * float(cfg.topology_surface_jitter_scale) * span
    jitter[mode == 2] = tangents[mode == 2] * float(cfg.topology_surface_tangent_scale) * span
    jitter[mode == 3] = rng.normal(0.0, float(cfg.topology_surface_jitter_scale), size=(np.sum(mode == 3), 3)) * span
    jitter[mode == 4] = 0.5 * rng.normal(0.0, float(cfg.topology_surface_jitter_scale), size=(np.sum(mode == 4), 3)) * span
    pts = clip_to_bounds(base + jitter, cfg)
    pf = sparse.iloc[base_idx]["dominant_pfaffian"].astype(str).to_numpy()
    sources = np.full(count, "sparse_patch_fill_distance", dtype=object)
    sources[(mode == 0) & (pf == "P0")] = "p0_zero_surface_sparse_patch"
    sources[(mode == 0) & (pf == "Ppi")] = "ppi_zero_surface_sparse_patch"
    sources[mode == 1] = "opposite_Z2_bracket_gap"
    sources[mode == 2] = "topology_surface_tangent_jitter"
    sources[mode == 3] = "topology_surface_normal_jitter"
    sources[mode == 4] = "component_boundary_guard"
    meta = pd.DataFrame({"candidate_source": sources, "surface_component_id": sparse.iloc[base_idx]["component_id"].to_numpy(dtype=np.int64)})
    return pts, meta, {"sparse_proposal_status": "ok", "sparse_surface_samples": int(surface.shape[0])}


def generate_stagev3_candidate_pool(dataset: FlatDataset, cfg: StageV3Config, iteration: int) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    base_points, base_meta, base_summary = generate_stagev_candidate_pool(dataset, cfg, iteration)
    n_sparse = max(int(0.20 * int(cfg.candidate_pool_size)), int(cfg.micro_batch_size) * 8)
    sparse_points, sparse_meta, sparse_summary = generate_sparse_topology_proposals(dataset, cfg, iteration, n_sparse)
    points = np.vstack([base_points, sparse_points]) if sparse_points.size else base_points
    meta = pd.concat([base_meta, sparse_meta], ignore_index=True) if not sparse_meta.empty else base_meta.copy()
    if points.shape[0] != meta.shape[0]:
        meta = pd.DataFrame({"candidate_source": ["unknown_explicit"] * points.shape[0], "source_failure_reason": "candidate_meta_length_mismatch"})
    rounded = np.round(points, 8)
    _, first = np.unique(rounded, axis=0, return_index=True)
    first = np.sort(first)
    points = points[first]
    meta = meta.iloc[first].reset_index(drop=True)
    meta = validate_proposal_sources(meta, strict=False)
    summary = {
        **base_summary,
        **sparse_summary,
        "stagev3_candidate_count": int(points.shape[0]),
        "stagev3_candidate_source_counts": meta["candidate_source"].value_counts().to_dict(),
    }
    return points, meta, summary


def validate_proposal_sources(meta: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    frame = meta.copy()
    if "candidate_source" not in frame:
        frame["candidate_source"] = "unknown_explicit"
        frame["source_failure_reason"] = "missing_candidate_source_column"
    source = frame["candidate_source"].astype(object)
    bad = source.isna() | (source.astype(str).str.strip() == "") | source.astype(str).str.lower().isin({"nan", "none"})
    if "source_failure_reason" not in frame:
        frame["source_failure_reason"] = ""
    frame.loc[bad, "candidate_source"] = "unknown_explicit"
    frame.loc[bad, "source_failure_reason"] = "blank_or_nan_source"
    unknown_allowed = ~frame["candidate_source"].isin(ALLOWED_PROPOSAL_SOURCES)
    frame.loc[unknown_allowed, "source_failure_reason"] = frame.loc[unknown_allowed, "candidate_source"].astype(str).map(
        lambda x: f"source_not_in_allowlist:{x}"
    )
    frame.loc[unknown_allowed, "candidate_source"] = "unknown_explicit"
    remaining_bad = frame["candidate_source"].isna() | (frame["candidate_source"].astype(str).str.strip() == "")
    if strict and bool(remaining_bad.any()):
        raise ValueError("proposal source attribution validation failed")
    return frame


def compute_stagev3_per_boundary_rewards(selected: pd.DataFrame, exact: pd.DataFrame | None = None) -> pd.DataFrame:
    base = compute_per_boundary_rewards(selected, exact)
    selected = selected.reset_index(drop=True).copy()
    for boundary in TOPOLOGY_BOUNDARIES:
        cov = selected.get("C_topo", pd.Series(np.zeros(len(selected)))).to_numpy(dtype=np.float64)
        raw_gain = selected.get("C_topo_raw", pd.Series(np.zeros(len(selected)))).to_numpy(dtype=np.float64)
        sparse_dist = selected.get("topology_sparse_nearest_distance", pd.Series(np.full(len(selected), np.inf))).to_numpy(dtype=np.float64)
        bracket = selected.get("created_topology_bracket", pd.Series(np.zeros(len(selected)))).to_numpy(dtype=np.float64)
        near_sparse = np.where(np.isfinite(sparse_dist), np.exp(-((sparse_dist / 0.08) ** 2)), 0.0)
        coverage_local = np.clip(cov, 0.0, 1.0)
        coverage_p95 = np.clip(raw_gain / max(np.nanpercentile(raw_gain[raw_gain > 0], 95) if np.any(raw_gain > 0) else 1.0, 1.0e-12), 0.0, 1.0)
        add = 0.35 * coverage_local + 0.25 * coverage_p95 + 0.20 * near_sparse + 0.20 * bracket
        raw_col = f"reward_{boundary}_raw"
        base[f"reward_{boundary}_coverage_local"] = coverage_local
        base[f"reward_{boundary}_coverage_p95"] = coverage_p95
        base[f"reward_{boundary}_component_support"] = near_sparse
        base[raw_col] = base.get(raw_col, 0.0) + add
    gap_rel = selected.get("gap_topology_relevance", pd.Series(np.zeros(len(selected)))).to_numpy(dtype=np.float64)
    if "reward_gap_raw" in base:
        base["reward_gap_topology_relevance"] = gap_rel
        base["reward_gap_raw"] = base["reward_gap_raw"].to_numpy(float) * np.clip(gap_rel, 0.0, 1.0)
    for boundary in BOUNDARY_NAMES:
        col = f"reward_{boundary}_raw"
        if col not in base:
            continue
        vals = base[col].to_numpy(dtype=np.float64)
        med = float(np.nanmedian(vals)) if vals.size else 0.0
        q75 = float(np.nanpercentile(vals, 75)) if vals.size else 1.0
        q25 = float(np.nanpercentile(vals, 25)) if vals.size else 0.0
        scale = max(q75 - q25, float(np.nanstd(vals)) if vals.size else 0.0, 1.0e-6)
        base[f"reward_{boundary}_normalized"] = (vals - med) / scale
    return base


def update_stagev3_lambdas(
    previous: dict[str, float] | None,
    validation: dict[str, Any],
    cfg: StageV3Config,
) -> dict[str, float]:
    prev = {**initial_lambda_state(cfg), **(previous or {})}
    out: dict[str, float] = {}
    for boundary in BOUNDARY_NAMES:
        n = int(validation.get(f"reward_sample_count_{boundary}", validation.get("reward_sample_count", 0)))
        delta = float(validation.get(f"rank_correlation_delta_vs_a0_{boundary}", 0.0))
        improved = float(validation.get(f"scientific_metric_improved_{boundary}", 1.0))
        coverage_improved = float(validation.get(f"scientific_metric_improved_coverage_{boundary}", 1.0))
        value = float(prev.get(boundary, 0.0))
        if n < int(cfg.learned_min_reward_samples):
            out[boundary] = 0.0
        elif boundary in TOPOLOGY_BOUNDARIES and coverage_improved <= 0.0:
            out[boundary] = max(0.0, 0.5 * value)
        elif improved < 0.0 or delta < -float(cfg.learned_validation_margin):
            out[boundary] = max(0.0, 0.5 * value)
        elif delta > float(cfg.learned_validation_margin):
            out[boundary] = min(float(cfg.learned_lambda_max), max(value, float(cfg.learned_initial_lambda)) + 0.05)
        else:
            out[boundary] = min(float(cfg.learned_lambda_max), max(0.0, value))
    return out


def sparse_arc_replay_from_dataset(dataset: FlatDataset, cfg: StageV3Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    surface = build_sparse_topology_surface(dataset, cfg)
    if surface.empty:
        return surface, pd.DataFrame(), {"status": "no_surface"}
    contributors = surface.nlargest(min(int(cfg.sparse_replay_top_contributors), len(surface)), "coverage_metric").copy()
    patch_table = (
        surface.groupby(["component_id", "dominant_pfaffian"], dropna=False)
        .agg(
            point_count=("component_id", "size"),
            coverage_median=("coverage_metric", "median"),
            coverage_p95=("coverage_metric", lambda x: float(np.nanpercentile(x, 95))),
            edge_length_median=("edge_length_norm", "median"),
            kT_min=("kT", "min"),
            kT_max=("kT", "max"),
            JA_min=("JA", "min"),
            JA_max=("JA", "max"),
            mu_min=("mu", "min"),
            mu_max=("mu", "max"),
        )
        .reset_index()
        .sort_values(["coverage_p95", "point_count"], ascending=[False, False])
    )
    summary = {
        "status": "ok",
        "surface_samples": int(surface.shape[0]),
        "sparse_patch_samples": int(surface["is_sparse_patch"].sum()),
        "component_count": int(surface["component_id"].nunique()),
        "coverage_p50": float(surface["coverage_metric"].quantile(0.50)),
        "coverage_p90": float(surface["coverage_metric"].quantile(0.90)),
        "coverage_p95": float(surface["coverage_metric"].quantile(0.95)),
        "p0_cross_fraction": float(surface["p0_cross"].mean()),
        "ppi_cross_fraction": float(surface["ppi_cross"].mean()),
    }
    return contributors, patch_table, summary


__all__ = [
    "ALLOWED_PROPOSAL_SOURCES",
    "STAGEV3_OUTPUT_ROOT",
    "STAGEV3_RUN_ID",
    "StageV3Config",
    "apply_stagev3_sparse_scores",
    "build_boundary_support_sets",
    "build_sparse_topology_surface",
    "combine_multihead_scores",
    "compute_stagev3_per_boundary_rewards",
    "expected_topology_coverage_improvement",
    "fit_multihead_value_models",
    "generate_stagev3_candidate_pool",
    "initial_alpha_state",
    "initial_lambda_state",
    "load_stagev2_state",
    "predict_stagev_fields",
    "score_stagev_a0",
    "select_micro_batch_v2",
    "sobol_points_3d",
    "sparse_arc_replay_from_dataset",
    "topology_channel_diagnostics",
    "update_boundary_alphas",
    "update_stagev3_lambdas",
    "validation_summary_by_boundary",
    "validate_proposal_sources",
    "write_empty_stagev_dataset",
    "write_stagev2_state",
    "write_stagev_selection",
]
