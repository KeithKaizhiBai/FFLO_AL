from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset_builder import FlatDataset
from .labels import PHASE_FFLO, PHASE_NAMES, PHASE_NORMAL, PHASE_UNIFORM_SC
from .stageiv_3d import min_distance_3d
from .stagev_acqv2 import clip_to_bounds, normalized_coordinates
from .stagev_v2 import (
    BOUNDARY_NAMES,
    StageV2Config,
    build_boundary_support_sets,
    combine_multihead_scores,
    compute_per_boundary_rewards,
    fit_multihead_value_models,
    generate_stagev_candidate_pool,
    initial_alpha_state,
    initial_lambda_state,
    load_stagev2_state,
    predict_linear_value_model,
    predict_stagev_fields,
    score_stagev_a0,
    select_micro_batch_v2,
    sobol_points_3d,
    topology_channel_diagnostics,
    train_linear_value_model,
    update_boundary_alphas,
    validation_summary_by_boundary,
    write_empty_stagev_dataset,
    write_stagev2_state,
    write_stagev_selection,
)
from .stagev_v3 import (
    StageV3Config,
    apply_stagev3_sparse_scores,
    build_sparse_topology_surface,
    compute_stagev3_per_boundary_rewards,
    generate_sparse_topology_proposals,
    validate_proposal_sources as validate_stagev3_proposal_sources,
)


STAGEV4_RUN_ID = "stagev_v4_phase_structure_graph_acquisition_3d_v1"
STAGEV4_OUTPUT_ROOT = "ML_Phase_StageV_V4_GraphAcq"

ACTION_FAMILY_BY_SOURCE = {
    "stagev4_scrambled_sobol_seed": "global_exploration",
    "initial_sobol": "global_exploration",
    "global_sobol": "global_exploration",
    "sparse_parameter_region": "global_exploration",
    "known_boundary_NS": "known_boundary_refinement",
    "known_boundary_UF": "known_boundary_refinement",
    "known_boundary_P0": "known_boundary_refinement",
    "known_boundary_Ppi": "known_boundary_refinement",
    "known_boundary_gap": "known_boundary_refinement",
    "p0_zero_surface_sparse_patch": "known_boundary_refinement",
    "ppi_zero_surface_sparse_patch": "known_boundary_refinement",
    "opposite_Z2_bracket_gap": "known_boundary_refinement",
    "topology_surface_normal_jitter": "known_boundary_refinement",
    "topology_surface_tangent_jitter": "known_boundary_refinement",
    "component_boundary_guard": "known_boundary_refinement",
    "sparse_patch_fill_distance": "known_boundary_refinement",
    "gap_topology_relevant": "known_boundary_refinement",
    "ns_surface": "known_boundary_refinement",
    "uf_surface": "known_boundary_refinement",
    "descriptor_OOD": "candidate_phase_validation",
    "model_disagreement": "candidate_phase_validation",
    "component_birth": "candidate_phase_validation",
    "candidate_phase_interior": "candidate_phase_validation",
    "candidate_phase_boundary": "candidate_phase_validation",
    "branch_competition": "oracle_anomaly_audit",
    "persistent_surprise": "oracle_anomaly_audit",
    "oracle_anomaly": "oracle_anomaly_audit",
    "fallback": "global_exploration",
    "unknown_explicit": "global_exploration",
    "boundary_support_jitter": "known_boundary_refinement",
    "mu_edge_guard": "global_exploration",
    "sparse_fill": "global_exploration",
}

ALLOWED_ACTION_FAMILIES = {
    "global_exploration",
    "known_boundary_refinement",
    "candidate_phase_validation",
    "oracle_anomaly_audit",
}

GRAPH_FEATURE_COLUMNS = [
    "graph_value_prior",
    "graph_uncertainty_score",
    "node_uncertainty_score",
    "edge_uncertainty_score",
    "support_uncertainty_score",
    "component_uncertainty_score",
    "reliability_uncertainty_score",
    "oracle_anomaly_score",
    "nearest_graph_node_distance",
    "nearest_graph_edge_distance",
    "candidate_phase_score",
    "action_family_code",
]


@dataclass(frozen=True)
class StageV4Config(StageV3Config):
    run_id: str = STAGEV4_RUN_ID
    output_root: str = STAGEV4_OUTPUT_ROOT
    random_seed: int = 20260708
    stagev2_schema_version: str = "stagev_v4_phase_structure_graph_acq_1"
    graph_update_interval: int = 3
    graph_min_trusted_points: int = 128
    candidate_phase_min_points: int = 10
    candidate_phase_min_persistence: int = 2
    descriptor_ood_quantile: float = 0.98
    graph_component_radius: float = 0.085
    graph_edge_radius: float = 0.105
    graph_candidate_fraction: float = 0.25
    graph_value_weight: float = 1.35
    graph_value_lambda_max: float = 1.2
    graph_model_min_reward_samples: int = 512
    graph_model_validation_margin: float = 0.02
    graph_shadow_mode: bool = False
    candidate_phase_shadow_mode: bool = False
    known_boundary_baseline_only: bool = False
    proposal_source_validation_strict: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path) -> "StageV4Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in fields})


def trusted_mask(dataset: FlatDataset) -> np.ndarray:
    n = int(dataset.x.shape[0])
    trusted = np.asarray(dataset.records.get("trusted_exact", np.ones(n, dtype=np.int8))).astype(bool)
    eligible = np.asarray(dataset.records.get("training_eligible_exact", np.ones(n, dtype=np.int8))).astype(bool)
    rerun = np.asarray(dataset.records.get("needs_rerun_exact", np.zeros(n, dtype=np.int8))).astype(bool)
    q_bad = np.asarray(dataset.records.get("q_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)
    d_bad = np.asarray(dataset.records.get("delta_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)
    return trusted & eligible & ~rerun & ~q_bad & ~d_bad


def _record_array(dataset: FlatDataset, name: str, default: float | int = np.nan) -> np.ndarray:
    n = int(dataset.x.shape[0])
    if name in dataset.records:
        arr = np.asarray(dataset.records[name])
        if arr.shape[0] == n:
            return arr
    return np.full(n, default)


def _safe_float_array(dataset: FlatDataset, names: list[str], default: float = np.nan) -> np.ndarray:
    for name in names:
        arr = _record_array(dataset, name, default)
        if np.isfinite(np.asarray(arr, dtype=np.float64)).any():
            return np.asarray(arr, dtype=np.float64)
    return np.full(int(dataset.x.shape[0]), default, dtype=np.float64)


def build_observation_descriptors(dataset: FlatDataset, cfg: StageV4Config) -> pd.DataFrame:
    points = np.asarray(dataset.x, dtype=np.float64).reshape(-1, 3)
    n = int(points.shape[0])
    phase = np.asarray(dataset.y_phase, dtype=np.int64)
    delta = _safe_float_array(dataset, ["delta_opt", "Delta_opt"], default=np.nan)
    if not np.isfinite(delta).any() and dataset.y_reg.shape[1] >= 1:
        delta = np.asarray(dataset.y_reg[:, 0], dtype=np.float64)
    q = _safe_float_array(dataset, ["q_opt"], default=np.nan)
    if not np.isfinite(q).any() and dataset.y_reg.shape[1] >= 2:
        q = np.asarray(dataset.y_reg[:, 1], dtype=np.float64)
    f_gap = _safe_float_array(
        dataset,
        ["free_energy_gap_to_normal", "F_SC_minus_F_normal", "free_energy_margin"],
        default=np.nan,
    )
    p0 = _safe_float_array(dataset, ["topology_p0", "P0"], default=np.nan)
    ppi = _safe_float_array(dataset, ["topology_ppi", "Ppi"], default=np.nan)
    pf_product = np.where(np.isfinite(p0) & np.isfinite(ppi), p0 * ppi, np.nan)
    bulk_gap = _safe_float_array(dataset, ["topology_bulk_gap", "bulk_gap"], default=np.nan)
    topo = np.asarray(_record_array(dataset, "topology_label_code", -1), dtype=np.int64)
    spectral = np.asarray(_record_array(dataset, "topology_spectral_status_code", -1), dtype=np.int64)
    tmask = trusted_mask(dataset)
    topo_trusted = np.asarray(_record_array(dataset, "topology_trusted", 0)).astype(bool)
    hard_risk = (
        ~tmask
        | np.asarray(_record_array(dataset, "needs_rerun_exact", 0)).astype(bool)
        | np.asarray(_record_array(dataset, "q_unresolved", 0)).astype(bool)
        | np.asarray(_record_array(dataset, "delta_unresolved", 0)).astype(bool)
    )
    phase_name = np.array([PHASE_NAMES.get(int(v), "unknown") for v in phase], dtype=object)
    node_label = phase_name.astype(object)
    cfflo = (phase == PHASE_FFLO) & (topo == 0) & topo_trusted & (spectral == 0)
    tfflo = (phase == PHASE_FFLO) & (topo == 1) & topo_trusted & (spectral == 0)
    node_label[cfflo] = "cFFLO_trivial"
    node_label[tfflo] = "tFFLO_topological"
    node_label[(phase == PHASE_FFLO) & ~(cfflo | tfflo)] = "FFLO_topology_unresolved"
    basin_gap = _safe_float_array(dataset, ["deltaF_to_second_basin", "basin_deltaF_12"], default=np.nan)
    q_sep = _safe_float_array(dataset, ["q_separation_to_second_basin", "basin_q_separation_12"], default=np.nan)
    pf_stack = np.vstack([np.abs(p0), np.abs(ppi)])
    pf_margin = np.full(n, np.nan, dtype=np.float64)
    pf_valid = np.isfinite(pf_stack).any(axis=0)
    if bool(pf_valid.any()):
        pf_margin[pf_valid] = np.min(np.where(np.isfinite(pf_stack[:, pf_valid]), pf_stack[:, pf_valid], np.inf), axis=0)
    descriptor = pd.DataFrame(
        {
            "point_index": np.arange(n, dtype=np.int64),
            "kT": points[:, 0],
            "JA": points[:, 1],
            "mu": points[:, 2],
            "phase_label": phase,
            "phase_name": phase_name,
            "node_label": node_label,
            "Delta_opt": delta,
            "q_opt": q,
            "F_SC_minus_F_normal": f_gap,
            "free_energy_margin": np.abs(f_gap),
            "P0": p0,
            "Ppi": ppi,
            "pf_product": pf_product,
            "pfaffian_margin": pf_margin,
            "bulk_gap": bulk_gap,
            "topology_label_code": topo,
            "spectral_status_code": spectral,
            "deltaF_to_second_basin": basin_gap,
            "q_separation_to_second_basin": q_sep,
            "trusted_exact": tmask,
            "topology_trusted": topo_trusted,
            "hard_risk": hard_risk,
            "descriptor_available": True,
        }
    )
    descriptor["branch_competition_score"] = np.where(
        np.isfinite(descriptor["deltaF_to_second_basin"]),
        np.exp(-np.abs(descriptor["deltaF_to_second_basin"].to_numpy(float)) / 1.0e-3),
        0.0,
    )
    descriptor["descriptor_ood_score"] = descriptor_ood_score(descriptor)
    descriptor["candidate_phase_flag"] = (
        (descriptor["descriptor_ood_score"] >= float(descriptor["descriptor_ood_score"].quantile(cfg.descriptor_ood_quantile)))
        & descriptor["trusted_exact"].astype(bool)
        & ~descriptor["hard_risk"].astype(bool)
    )
    return descriptor


def descriptor_ood_score(desc: pd.DataFrame) -> np.ndarray:
    cols = ["Delta_opt", "q_opt", "F_SC_minus_F_normal", "P0", "Ppi", "bulk_gap", "branch_competition_score"]
    mat = desc[cols].to_numpy(dtype=np.float64)
    score = np.zeros(mat.shape[0], dtype=np.float64)
    for label in sorted(set(desc["node_label"].astype(str))):
        mask = desc["node_label"].astype(str).to_numpy() == label
        if int(mask.sum()) < 8:
            continue
        sub = mat[mask]
        finite_cols = np.isfinite(sub).any(axis=0)
        if not bool(finite_cols.any()):
            continue
        med = np.zeros(sub.shape[1], dtype=np.float64)
        scale = np.ones(sub.shape[1], dtype=np.float64)
        med[finite_cols] = np.nanmedian(sub[:, finite_cols], axis=0)
        scale[finite_cols] = np.nanpercentile(np.abs(sub[:, finite_cols] - med[finite_cols]), 75, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1.0e-9), scale, 1.0)
        z = np.nan_to_num((mat[mask] - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        score[mask] = np.sqrt(np.mean(z * z, axis=1))
    if np.nanmax(score) > 0:
        score = score / np.nanmax(score)
    return np.clip(score, 0.0, 1.0)


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


def build_phase_structure_graph(dataset: FlatDataset, cfg: StageV4Config) -> dict[str, Any]:
    desc = build_observation_descriptors(dataset, cfg)
    trusted = desc["trusted_exact"].astype(bool).to_numpy()
    points = desc[["kT", "JA", "mu"]].to_numpy(dtype=np.float64)
    points_n = normalized_coordinates(points, cfg)
    node_rows: list[dict[str, Any]] = []
    point_node = np.full(len(desc), "unassigned", dtype=object)
    for label in sorted(set(desc["node_label"].astype(str))):
        mask = trusted & (desc["node_label"].astype(str).to_numpy() == label)
        if int(mask.sum()) == 0:
            continue
        ids = _component_ids(points_n[mask], float(cfg.graph_component_radius))
        local_indices = np.flatnonzero(mask)
        for comp in sorted(set(ids.tolist())):
            comp_idx = local_indices[ids == comp]
            sub = desc.iloc[comp_idx]
            if label.startswith("candidate_phase"):
                node_kind = "candidate"
            elif bool(sub["candidate_phase_flag"].mean() > 0.55) and len(sub) >= int(cfg.candidate_phase_min_points):
                node_kind = "candidate"
                label = "candidate_phase_1"
            else:
                node_kind = "known"
            node_id = f"{label}_c{int(comp)}"
            point_node[comp_idx] = node_id
            hard_rate = float(sub["hard_risk"].mean()) if len(sub) else 1.0
            support_p95 = _support_p95(sub[["kT", "JA", "mu"]].to_numpy(float), cfg)
            node_rows.append(
                {
                    "node_id": node_id,
                    "node_kind": node_kind,
                    "identity": label,
                    "point_count": int(len(sub)),
                    "trusted_count": int(sub["trusted_exact"].sum()),
                    "hard_risk_fraction": hard_rate,
                    "candidate_phase_score": float(sub["descriptor_ood_score"].mean()),
                    "support_p95": support_p95,
                    "component_id": int(comp),
                    "node_uncertainty": float(np.clip((1.0 / np.sqrt(max(len(sub), 1))) + hard_rate + min(support_p95, 1.0), 0.0, 3.0)),
                    "status": "provisional_candidate_node" if node_kind == "candidate" else "known_node",
                }
            )
    desc["graph_node_id"] = point_node
    edge_rows: list[dict[str, Any]] = []
    trusted_idx = np.flatnonzero(trusted)
    if trusted_idx.size >= 2:
        edges = _pair_edges(points_n[trusted_idx], int(cfg.bracket_neighbor_k), float(cfg.graph_edge_radius))
        for i0, j0, d in edges:
            i = int(trusted_idx[i0])
            j = int(trusted_idx[j0])
            ni = str(desc.iloc[i]["graph_node_id"])
            nj = str(desc.iloc[j]["graph_node_id"])
            if ni == nj or ni == "unassigned" or nj == "unassigned":
                continue
            edge_type = edge_type_from_nodes(str(desc.iloc[i]["node_label"]), str(desc.iloc[j]["node_label"]))
            midpoint = 0.5 * (points[i] + points[j])
            edge_rows.append(
                {
                    "edge_id": f"{ni}__{nj}",
                    "node_a": ni,
                    "node_b": nj,
                    "edge_type": edge_type,
                    "kT": float(midpoint[0]),
                    "JA": float(midpoint[1]),
                    "mu": float(midpoint[2]),
                    "edge_length_norm": float(d),
                    "hard_risk_edge": bool(desc.iloc[i]["hard_risk"] or desc.iloc[j]["hard_risk"]),
                }
            )
    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(edge_rows)
    if not edges.empty:
        grouped = (
            edges.groupby(["node_a", "node_b", "edge_type"], dropna=False)
            .agg(
                edge_support_count=("edge_id", "size"),
                edge_length_p50=("edge_length_norm", "median"),
                edge_length_p95=("edge_length_norm", lambda x: float(np.nanpercentile(x, 95))),
                hard_risk_fraction=("hard_risk_edge", "mean"),
                kT=("kT", "mean"),
                JA=("JA", "mean"),
                mu=("mu", "mean"),
            )
            .reset_index()
        )
        grouped["edge_uncertainty"] = np.clip(
            1.0 / np.sqrt(np.maximum(grouped["edge_support_count"].to_numpy(float), 1.0))
            + grouped["edge_length_p95"].to_numpy(float)
            + grouped["hard_risk_fraction"].to_numpy(float),
            0.0,
            3.0,
        )
        edges = grouped
    summary = graph_uncertainty_summary(nodes, edges, desc)
    return {"descriptors": desc, "nodes": nodes, "edges": edges, "summary": summary}


def _support_p95(points: np.ndarray, cfg: StageV4Config) -> float:
    if points.shape[0] < 2:
        return 1.0
    pn = normalized_coordinates(points, cfg)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        d, _ = cKDTree(pn).query(pn, k=2)
        nearest = d[:, 1]
    except Exception:
        dmat = np.sqrt(np.sum((pn[:, None, :] - pn[None, :, :]) ** 2, axis=2))
        dmat[dmat == 0] = np.inf
        nearest = np.min(dmat, axis=1)
    return float(np.nanpercentile(nearest, 95))


def edge_type_from_nodes(a: str, b: str) -> str:
    labels = {a, b}
    if "normal" in labels and any(x in labels for x in {"uniform_SC", "FFLO_topology_unresolved", "cFFLO_trivial", "tFFLO_topological"}):
        return "known_boundary_NS"
    if "uniform_SC" in labels and any(x in labels for x in {"FFLO_topology_unresolved", "cFFLO_trivial", "tFFLO_topological"}):
        return "known_boundary_UF"
    if labels == {"cFFLO_trivial", "tFFLO_topological"}:
        return "known_boundary_topology"
    if any(x.startswith("candidate_phase") for x in labels):
        return "candidate_boundary"
    if "FFLO_topology_unresolved" in labels:
        return "unresolved_boundary"
    return "anomaly_boundary"


def graph_uncertainty_summary(nodes: pd.DataFrame, edges: pd.DataFrame, desc: pd.DataFrame) -> dict[str, Any]:
    candidate_nodes = int(nodes["node_kind"].eq("candidate").sum()) if not nodes.empty else 0
    known_nodes = int(nodes["node_kind"].eq("known").sum()) if not nodes.empty else 0
    edge_count = int(edges.shape[0]) if not edges.empty else 0
    hard_risk_rate = float(desc["hard_risk"].mean()) if len(desc) else 1.0
    node_u = float(nodes["node_uncertainty"].sum()) if not nodes.empty else 1.0
    edge_u = float(edges["edge_uncertainty"].sum()) if not edges.empty else 1.0
    candidate_u = float(candidate_nodes)
    total = node_u + edge_u + candidate_u + hard_risk_rate
    return {
        "known_node_count": known_nodes,
        "candidate_node_count": candidate_nodes,
        "edge_count": edge_count,
        "graph_uncertainty_total": float(total),
        "node_uncertainty_total": node_u,
        "edge_uncertainty_total": edge_u,
        "candidate_uncertainty_total": candidate_u,
        "hard_risk_rate": hard_risk_rate,
    }


def generate_graph_candidate_proposals(
    dataset: FlatDataset,
    graph: dict[str, Any],
    cfg: StageV4Config,
    iteration: int,
    count: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.random_seed) + 13007 * int(iteration))
    if count <= 0:
        return np.empty((0, 3), dtype=np.float64), pd.DataFrame({"candidate_source": []}), {"graph_proposal_status": "disabled"}
    lo, hi = cfg.bounds()
    span = hi - lo
    nodes = graph.get("nodes", pd.DataFrame())
    edges = graph.get("edges", pd.DataFrame())
    desc = graph.get("descriptors", pd.DataFrame())
    pts: list[np.ndarray] = []
    sources: list[str] = []

    def add(center: np.ndarray, source: str, scale: float = 0.035) -> None:
        jitter = rng.normal(0.0, scale, size=3) * span
        pts.append(np.clip(center + jitter, lo, hi))
        sources.append(source)

    if not edges.empty:
        weights = np.maximum(edges.get("edge_uncertainty", pd.Series(np.ones(len(edges)))).to_numpy(float), 1.0e-9)
        weights = weights / np.sum(weights)
        for _ in range(max(1, count // 3)):
            row = edges.iloc[int(rng.choice(np.arange(len(edges)), p=weights))]
            add(row[["kT", "JA", "mu"]].to_numpy(float), source_for_edge_type(str(row.get("edge_type", "candidate_boundary"))), scale=0.030)
    if not nodes.empty and bool(nodes["node_kind"].eq("candidate").any()):
        cand = nodes[nodes["node_kind"].eq("candidate")]
        weights = np.maximum(cand.get("node_uncertainty", pd.Series(np.ones(len(cand)))).to_numpy(float), 1.0e-9)
        weights = weights / np.sum(weights)
        for _ in range(max(1, count // 5)):
            row = cand.iloc[int(rng.choice(np.arange(len(cand)), p=weights))]
            members = desc[desc["graph_node_id"].astype(str).eq(str(row["node_id"]))]
            if members.empty:
                continue
            center = members[["kT", "JA", "mu"]].mean().to_numpy(float)
            add(center, "candidate_phase_interior", scale=0.025)
    if not desc.empty and bool(desc["branch_competition_score"].gt(0.4).any()):
        branch = desc[desc["branch_competition_score"].gt(0.4)]
        for _ in range(max(1, count // 8)):
            row = branch.iloc[int(rng.integers(0, len(branch)))]
            add(row[["kT", "JA", "mu"]].to_numpy(float), "branch_competition", scale=0.025)
    while len(pts) < count:
        sob = sobol_points_3d(1, cfg, seed_offset=9000 + int(iteration) + len(pts))
        pts.append(sob[0])
        sources.append("sparse_parameter_region")
    arr = np.vstack(pts[:count])
    meta = pd.DataFrame({"candidate_source": sources[:count]})
    meta = validate_stagev4_proposal_sources(meta, strict=False)
    return arr, meta, {
        "graph_proposal_status": "ok",
        "graph_proposal_count": int(arr.shape[0]),
        "graph_proposal_source_counts": meta["candidate_source"].value_counts().to_dict(),
    }


def source_for_edge_type(edge_type: str) -> str:
    return {
        "known_boundary_NS": "known_boundary_NS",
        "known_boundary_UF": "known_boundary_UF",
        "known_boundary_topology": "known_boundary_P0",
        "known_boundary_gap": "known_boundary_gap",
        "candidate_boundary": "candidate_phase_boundary",
        "anomaly_boundary": "oracle_anomaly",
        "unresolved_boundary": "persistent_surprise",
    }.get(edge_type, "fallback")


def generate_stagev4_candidate_pool(dataset: FlatDataset, cfg: StageV4Config, iteration: int) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    base_points, base_meta, base_summary = generate_stagev_candidate_pool(dataset, cfg, iteration)
    sparse_count = max(int(0.16 * int(cfg.candidate_pool_size)), int(cfg.micro_batch_size) * 6)
    sparse_points, sparse_meta, sparse_summary = generate_sparse_topology_proposals(dataset, cfg, iteration, sparse_count)
    graph = build_phase_structure_graph(dataset, cfg)
    graph_count = max(int(float(cfg.graph_candidate_fraction) * int(cfg.candidate_pool_size)), int(cfg.micro_batch_size) * 8)
    graph_points, graph_meta, graph_summary = generate_graph_candidate_proposals(dataset, graph, cfg, iteration, graph_count)
    points = np.vstack([base_points, sparse_points, graph_points])
    meta = pd.concat([base_meta, sparse_meta, graph_meta], ignore_index=True)
    if points.shape[0] != meta.shape[0]:
        meta = pd.DataFrame({"candidate_source": ["unknown_explicit"] * points.shape[0], "source_failure_reason": "candidate_meta_length_mismatch"})
    rounded = np.round(points, 8)
    _, first = np.unique(rounded, axis=0, return_index=True)
    first = np.sort(first)
    points = points[first]
    meta = meta.iloc[first].reset_index(drop=True)
    meta = validate_stagev4_proposal_sources(meta, strict=False)
    summary = {
        **base_summary,
        **sparse_summary,
        **graph_summary,
        **{f"graph_{k}": v for k, v in graph["summary"].items()},
        "stagev4_candidate_count": int(points.shape[0]),
        "stagev4_candidate_source_counts": meta["candidate_source"].value_counts().to_dict(),
        "stagev4_action_family_counts": meta["action_family"].value_counts().to_dict(),
    }
    return points, meta, summary, graph


def validate_stagev4_proposal_sources(meta: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    frame = validate_stagev3_proposal_sources(meta.copy(), strict=False)
    if "proposal_source" not in frame:
        if "proposal_source" in meta:
            frame["proposal_source"] = meta["proposal_source"].astype(str).to_numpy()
        else:
            frame["proposal_source"] = frame["candidate_source"].astype(str)
    source = frame["candidate_source"].astype(str)
    unknown = source.eq("unknown_explicit") & meta.get("candidate_source", pd.Series([""] * len(frame))).astype(str).isin(ACTION_FAMILY_BY_SOURCE)
    if bool(unknown.any()):
        frame.loc[unknown, "candidate_source"] = meta.loc[unknown, "candidate_source"].astype(str)
    proposal_source = frame["proposal_source"].astype(str)
    proposal_unknown = proposal_source.eq("unknown_explicit") & meta.get("proposal_source", meta.get("candidate_source", pd.Series([""] * len(frame)))).astype(str).isin(ACTION_FAMILY_BY_SOURCE)
    if bool(proposal_unknown.any()):
        src = meta.get("proposal_source", meta.get("candidate_source")).astype(str)
        frame.loc[proposal_unknown, "proposal_source"] = src.loc[proposal_unknown]
    frame["proposal_source"] = frame["proposal_source"].fillna(frame["candidate_source"]).astype(str)
    frame.loc[frame["proposal_source"].str.strip().eq(""), "proposal_source"] = frame.loc[
        frame["proposal_source"].str.strip().eq(""), "candidate_source"
    ].astype(str)
    frame["action_family"] = frame["candidate_source"].map(ACTION_FAMILY_BY_SOURCE).fillna("global_exploration")
    bad_family = ~frame["action_family"].isin(ALLOWED_ACTION_FAMILIES)
    frame.loc[bad_family, "action_family"] = "global_exploration"
    if "source_failure_reason" not in frame:
        frame["source_failure_reason"] = ""
    remaining_bad = frame["candidate_source"].isna() | frame["candidate_source"].astype(str).str.strip().eq("")
    remaining_bad = remaining_bad | frame["proposal_source"].isna() | frame["proposal_source"].astype(str).str.strip().eq("")
    remaining_bad = remaining_bad | frame["action_family"].isna() | frame["action_family"].astype(str).str.strip().eq("")
    if strict and bool(remaining_bad.any()):
        raise ValueError("Stage V-v4 proposal/action-family validation failed")
    return frame


def attach_graph_context_scores(scored: pd.DataFrame, graph: dict[str, Any], cfg: StageV4Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = scored.copy().reset_index(drop=True)
    candidates = frame[["kT", "JA", "mu"]].to_numpy(dtype=np.float64)
    nodes = graph.get("nodes", pd.DataFrame())
    edges = graph.get("edges", pd.DataFrame())
    if nodes.empty:
        node_points = np.empty((0, 3))
        node_unc = np.empty((0,))
    else:
        node_points = graph.get("descriptors", pd.DataFrame()).groupby("graph_node_id")[["kT", "JA", "mu"]].mean().reset_index()
        node_lookup = nodes.set_index("node_id")["node_uncertainty"].to_dict()
        node_unc = node_points["graph_node_id"].map(node_lookup).fillna(0.0).to_numpy(float)
        node_points = node_points[["kT", "JA", "mu"]].to_numpy(float)
    if edges.empty:
        edge_points = np.empty((0, 3))
        edge_unc = np.empty((0,))
    else:
        edge_points = edges[["kT", "JA", "mu"]].to_numpy(float)
        edge_unc = edges["edge_uncertainty"].to_numpy(float)
    node_dist, node_score = nearest_weighted_uncertainty(candidates, node_points, node_unc, cfg)
    edge_dist, edge_score = nearest_weighted_uncertainty(candidates, edge_points, edge_unc, cfg)
    support_score = np.clip(frame.get("nearest_exact_distance", pd.Series(np.zeros(len(frame)))).to_numpy(float) / 0.08, 0.0, 1.0)
    reliability_score = np.clip(1.0 - frame.get("exact_repulsion", pd.Series(np.ones(len(frame)))).to_numpy(float), 0.0, 1.0)
    action_family = frame.get("action_family", pd.Series(["global_exploration"] * len(frame))).astype(str)
    family_code = action_family.map({name: i for i, name in enumerate(sorted(ALLOWED_ACTION_FAMILIES))}).fillna(0).to_numpy(float)
    family_prior = action_family.map(
        {
            "global_exploration": 0.45,
            "known_boundary_refinement": 0.80,
            "candidate_phase_validation": 0.95,
            "oracle_anomaly_audit": 0.70,
        }
    ).fillna(0.45).to_numpy(float)
    candidate_phase_score = np.where(action_family.eq("candidate_phase_validation"), np.maximum(node_score, support_score), 0.0)
    oracle_anomaly = np.where(action_family.eq("oracle_anomaly_audit"), np.maximum(reliability_score, support_score), 0.0)
    graph_score = np.clip(
        0.22 * family_prior
        + 0.20 * edge_score
        + 0.18 * node_score
        + 0.18 * support_score
        + 0.12 * candidate_phase_score
        + 0.10 * oracle_anomaly,
        0.0,
        1.0,
    )
    frame["nearest_graph_node_distance"] = node_dist
    frame["nearest_graph_edge_distance"] = edge_dist
    frame["node_uncertainty_score"] = node_score
    frame["edge_uncertainty_score"] = edge_score
    frame["support_uncertainty_score"] = support_score
    frame["component_uncertainty_score"] = np.maximum(node_score, edge_score)
    frame["reliability_uncertainty_score"] = reliability_score
    frame["oracle_anomaly_score"] = oracle_anomaly
    frame["candidate_phase_score"] = candidate_phase_score
    frame["action_family_code"] = family_code
    frame["graph_value_prior"] = family_prior
    frame["graph_uncertainty_score"] = graph_score
    frame["graph_value_score"] = graph_score
    return frame, {
        "graph_value_score_mean": float(np.nanmean(graph_score)) if len(graph_score) else 0.0,
        "graph_value_score_p95": float(np.nanpercentile(graph_score, 95)) if len(graph_score) else 0.0,
        "candidate_phase_candidate_count": int(action_family.eq("candidate_phase_validation").sum()),
        "oracle_anomaly_candidate_count": int(action_family.eq("oracle_anomaly_audit").sum()),
    }


def nearest_weighted_uncertainty(candidates: np.ndarray, support: np.ndarray, weights: np.ndarray, cfg: StageV4Config) -> tuple[np.ndarray, np.ndarray]:
    if support.shape[0] == 0:
        return np.full(candidates.shape[0], 1.0), np.zeros(candidates.shape[0])
    cand_n = normalized_coordinates(candidates, cfg)
    sup_n = normalized_coordinates(support, cfg)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        d, idx = cKDTree(sup_n).query(cand_n, k=1)
    except Exception:
        dmat = np.sqrt(np.sum((cand_n[:, None, :] - sup_n[None, :, :]) ** 2, axis=2))
        idx = np.argmin(dmat, axis=1)
        d = dmat[np.arange(cand_n.shape[0]), idx]
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(np.nanmax(w)) if w.size else 1.0, 1.0e-12)
    proximity = np.exp(-((np.asarray(d, dtype=np.float64) / 0.10) ** 2))
    return np.asarray(d, dtype=np.float64), np.clip(proximity * w[np.asarray(idx, dtype=int)], 0.0, 1.0)


def load_stagev4_state(run_dir: Path, cfg: StageV4Config) -> dict[str, Any]:
    path = run_dir / "stagev4_acquisition_state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    state = load_stagev2_state(run_dir, cfg)
    state.setdefault("graph_value_model", {"status": "insufficient_data", "feature_columns": GRAPH_FEATURE_COLUMNS})
    state.setdefault("graph_lambda", 0.0)
    return state


def write_stagev4_state(run_dir: Path, state: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stagev4_acquisition_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_stagev2_state(run_dir, state)


def apply_graph_value_model(scored: pd.DataFrame, state: dict[str, Any], cfg: StageV4Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = scored.copy()
    if bool(cfg.known_boundary_baseline_only):
        frame["graph_value_model_prediction"] = 0.0
        frame["graph_lambda"] = 0.0
        frame["A_total_v4"] = frame.get("A_total_v2", pd.Series(np.ones(len(frame)))).to_numpy(float)
        return frame, {"graph_value_model_status": "disabled_known_boundary_baseline", "graph_lambda": 0.0}
    model = state.get("graph_value_model", {})
    pred = predict_graph_value_model(model, frame)
    graph_lambda = float(state.get("graph_lambda", 0.0))
    base = frame.get("A_total_v2", pd.Series(np.ones(len(frame)))).to_numpy(float)
    prior = frame.get("graph_uncertainty_score", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    learned = np.clip(pred, -3.0, 3.0)
    boost = 1.0 + float(cfg.graph_value_weight) * prior
    if not bool(cfg.graph_shadow_mode):
        boost *= np.exp(graph_lambda * learned)
    frame["graph_value_model_prediction"] = pred
    frame["graph_lambda"] = graph_lambda
    frame["A_total_v4"] = base * np.clip(boost, 1.0e-6, 1.0e6)
    frame["A_total_v2"] = frame["A_total_v4"].to_numpy(float)
    return frame, {
        "graph_value_model_status": model.get("status", "missing"),
        "graph_lambda": graph_lambda,
        "graph_value_prediction_mean": float(np.nanmean(pred)) if len(pred) else 0.0,
        "graph_boost_mean": float(np.nanmean(boost)) if len(boost) else 0.0,
    }


def train_graph_value_model(history: pd.DataFrame, cfg: StageV4Config) -> dict[str, Any]:
    frame = history.copy()
    for col in GRAPH_FEATURE_COLUMNS:
        if col not in frame:
            frame[col] = 0.0
    if "reward_graph_normalized" not in frame:
        frame["reward_graph_normalized"] = 0.0
    model = train_linear_value_model(
        frame,
        frame["reward_graph_normalized"].to_numpy(dtype=np.float64),
        GRAPH_FEATURE_COLUMNS,
    )
    model["reward_column"] = "reward_graph_normalized"
    model["model_kind"] = "stagev4_graph_value_linear"
    return model


def predict_graph_value_model(model: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    frame = features.copy()
    for col in GRAPH_FEATURE_COLUMNS:
        if col not in frame:
            frame[col] = 0.0
    return predict_linear_value_model(model, frame)


def compute_stagev4_rewards(selected: pd.DataFrame, exact: pd.DataFrame | None = None) -> pd.DataFrame:
    base = compute_stagev3_per_boundary_rewards(selected, exact)
    frame = selected.reset_index(drop=True).copy()
    trusted = frame.get("trusted_exact", pd.Series(np.ones(len(frame)))).fillna(1).astype(float).to_numpy()
    if exact is not None and "trusted_exact" in exact:
        trusted = exact["trusted_exact"].fillna(0).astype(float).to_numpy()
    graph_prior = frame.get("graph_uncertainty_score", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    edge = frame.get("edge_uncertainty_score", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    node = frame.get("node_uncertainty_score", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    support = frame.get("support_uncertainty_score", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    candidate = frame.get("candidate_phase_score", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    anomaly = frame.get("oracle_anomaly_score", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    action = frame.get("action_family", pd.Series(["global_exploration"] * len(frame))).astype(str)
    family_bonus = action.map(
        {
            "global_exploration": 0.20,
            "known_boundary_refinement": 0.45,
            "candidate_phase_validation": 0.65,
            "oracle_anomaly_audit": 0.55,
        }
    ).fillna(0.20).to_numpy(float)
    raw = trusted * (0.25 * graph_prior + 0.20 * edge + 0.15 * node + 0.15 * support + 0.15 * candidate + 0.10 * anomaly + family_bonus)
    duplicate_penalty = 1.0 - np.clip(frame.get("exact_repulsion", pd.Series(np.ones(len(frame)))).to_numpy(float), 0.0, 1.0)
    raw = raw - 0.25 * duplicate_penalty
    med = float(np.nanmedian(raw)) if raw.size else 0.0
    scale = max(float(np.nanpercentile(raw, 75) - np.nanpercentile(raw, 25)) if raw.size else 0.0, float(np.nanstd(raw)) if raw.size else 0.0, 1.0e-6)
    base["reward_graph_raw"] = raw
    if raw.size < 2:
        base["reward_graph_normalized"] = raw
    else:
        base["reward_graph_normalized"] = (raw - med) / scale
    base["reward_graph_node"] = node
    base["reward_graph_edge"] = edge
    base["reward_graph_support"] = support
    base["reward_graph_candidate"] = candidate
    base["reward_graph_anomaly"] = anomaly
    return base


def update_graph_lambda(previous: float | None, validation: dict[str, Any], cfg: StageV4Config) -> float:
    value = float(previous or 0.0)
    n = int(validation.get("reward_sample_count_graph", validation.get("reward_sample_count", 0)))
    delta = float(validation.get("rank_correlation_delta_vs_a0_graph", 0.0))
    if n < int(cfg.graph_model_min_reward_samples):
        return 0.0
    if delta < -float(cfg.graph_model_validation_margin):
        return max(0.0, 0.5 * value)
    if delta > float(cfg.graph_model_validation_margin):
        return min(float(cfg.graph_value_lambda_max), max(value, 0.2) + 0.05)
    return min(float(cfg.graph_value_lambda_max), max(0.0, value))


def add_graph_validation(validation: dict[str, Any], history: pd.DataFrame, model: dict[str, Any]) -> dict[str, Any]:
    out = dict(validation)
    if history.empty or "reward_graph_normalized" not in history:
        out.update(
            {
                "reward_sample_count_graph": 0,
                "rank_correlation_a0_graph": 0.0,
                "rank_correlation_model_graph": 0.0,
                "rank_correlation_delta_vs_a0_graph": 0.0,
            }
        )
        return out
    y = history["reward_graph_normalized"].to_numpy(float)
    a0 = history.get("graph_uncertainty_score", pd.Series(np.zeros(len(history)))).to_numpy(float)
    pred = predict_graph_value_model(model, history)
    out["reward_sample_count_graph"] = int(np.sum(np.isfinite(y)))
    out["rank_correlation_a0_graph"] = _rank_corr(a0, y)
    out["rank_correlation_model_graph"] = _rank_corr(pred, y)
    out["rank_correlation_delta_vs_a0_graph"] = out["rank_correlation_model_graph"] - out["rank_correlation_a0_graph"]
    return out


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 4:
        return 0.0
    ra = pd.Series(a[valid]).rank(method="average").to_numpy(float)
    rb = pd.Series(b[valid]).rank(method="average").to_numpy(float)
    if np.std(ra) < 1.0e-12 or np.std(rb) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def stagev4_leakage_guard_summary() -> dict[str, Any]:
    return {
        "cold_start": True,
        "stageiii_data_used_for_training": False,
        "stageiv_data_used_for_training": False,
        "stagev1_data_used_for_training": False,
        "stagev2_data_used_for_training": False,
        "stagev2_tail_data_used_for_training": False,
        "stagev3_data_used_for_training": False,
    }


__all__ = [
    "ACTION_FAMILY_BY_SOURCE",
    "ALLOWED_ACTION_FAMILIES",
    "GRAPH_FEATURE_COLUMNS",
    "STAGEV4_OUTPUT_ROOT",
    "STAGEV4_RUN_ID",
    "StageV4Config",
    "add_graph_validation",
    "apply_graph_value_model",
    "apply_stagev3_sparse_scores",
    "attach_graph_context_scores",
    "build_boundary_support_sets",
    "build_observation_descriptors",
    "build_phase_structure_graph",
    "combine_multihead_scores",
    "compute_stagev4_rewards",
    "fit_multihead_value_models",
    "generate_stagev4_candidate_pool",
    "initial_alpha_state",
    "initial_lambda_state",
    "load_stagev4_state",
    "predict_stagev_fields",
    "score_stagev_a0",
    "select_micro_batch_v2",
    "sobol_points_3d",
    "stagev4_leakage_guard_summary",
    "topology_channel_diagnostics",
    "train_graph_value_model",
    "update_boundary_alphas",
    "update_graph_lambda",
    "validation_summary_by_boundary",
    "validate_stagev4_proposal_sources",
    "write_empty_stagev_dataset",
    "write_stagev4_state",
    "write_stagev_selection",
]
