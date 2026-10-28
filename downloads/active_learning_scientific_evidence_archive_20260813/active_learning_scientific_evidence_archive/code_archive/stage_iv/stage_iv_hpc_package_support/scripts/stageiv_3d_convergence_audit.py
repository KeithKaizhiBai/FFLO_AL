from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.labels import PHASE_FFLO, PHASE_NAMES, PHASE_NORMAL, PHASE_UNIFORM_SC
from ml_phase.stageiv_3d import STAGEIV_RUN_ID, StageIV3DConfig, sobol_points_3d


TOPOLOGY_LABEL_NAMES = {
    -1: "not_applicable",
    0: "trivial",
    1: "topological",
    2: "gapless_SC",
    3: "unresolved",
}

MAP_CHANGE_TOL = 0.002
SURFACE_SHIFT_TOL = 0.004166666666666667
COVERAGE_TOL = 0.00625
SURPRISE_TOL = 0.02


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_from_dataset(path: Path) -> int:
    return int(path.stem.replace("dataset_iter", ""))


def find_datasets(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("dataset_iter*.npz"), key=iter_from_dataset)
    if not paths:
        raise FileNotFoundError(f"No dataset_iter*.npz files found under {run_dir}")
    return paths


def arr_bool(z: Any, name: str, n: int, default: bool) -> np.ndarray:
    if name in z.files:
        return np.asarray(z[name]).astype(bool)
    return np.full(n, bool(default), dtype=bool)


def arr_float(z: Any, name: str, n: int, default: float = np.nan) -> np.ndarray:
    if name in z.files:
        return np.asarray(z[name], dtype=np.float64)
    return np.full(n, float(default), dtype=np.float64)


def arr_int(z: Any, name: str, n: int, default: int) -> np.ndarray:
    if name in z.files:
        return np.asarray(z[name], dtype=np.int64)
    return np.full(n, int(default), dtype=np.int64)


def load_dataset(path: Path, cfg: StageIV3DConfig) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        x_raw = np.asarray(z["x"], dtype=np.float64)
        n = int(x_raw.shape[0])
        if x_raw.shape[1] >= 3:
            x3 = x_raw[:, :3]
        else:
            mu = arr_float(z, "mu", n, float(cfg.mu_reference))
            x3 = np.column_stack([x_raw[:, 0], x_raw[:, 1], mu])
        trusted = arr_bool(z, "trusted_exact", n, True)
        training = arr_bool(z, "training_eligible_exact", n, True)
        if "rerun_required" in z.files:
            rerun = arr_bool(z, "rerun_required", n, False)
        else:
            rerun = arr_bool(z, "needs_rerun_exact", n, False)
        out = {
            "path": path,
            "iteration": iter_from_dataset(path),
            "x": x3,
            "y_phase": arr_int(z, "y_phase", n, -1),
            "trusted_exact": trusted,
            "training_eligible_exact": training,
            "rerun_required": rerun,
            "q_unresolved": arr_bool(z, "q_unresolved", n, False),
            "delta_unresolved": arr_bool(z, "delta_unresolved", n, False),
            "topology_label_code": arr_int(z, "topology_label_code", n, -1),
            "topology_trusted": arr_bool(z, "topology_trusted", n, False),
            "topology_spectral_status_code": arr_int(z, "topology_spectral_status_code", n, -1),
            "topology_p0": arr_float(z, "topology_p0", n),
            "topology_ppi": arr_float(z, "topology_ppi", n),
            "topology_bulk_gap": arr_float(z, "topology_bulk_gap", n),
        }
        if "point_id" in z.files:
            out["point_id"] = np.asarray(z["point_id"])
        else:
            out["point_id"] = None
    return out


def point_keys(data: dict[str, Any]) -> np.ndarray:
    point_id = data.get("point_id")
    if point_id is not None:
        return np.asarray(point_id).astype(str)
    x = np.asarray(data["x"], dtype=np.float64)
    return np.asarray([f"{p[0]:.8f}|{p[1]:.8f}|{p[2]:.8f}" for p in x], dtype=object)


def normalized(points: np.ndarray, cfg: StageIV3DConfig) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo, hi = cfg.bounds()
    return (points - lo) / np.maximum(hi - lo, 1.0e-12)


def phase_counts(y: np.ndarray) -> dict[str, int]:
    return {PHASE_NAMES.get(int(code), str(int(code))): int(np.sum(y == int(code))) for code in sorted(set(y.tolist()))}


def topology_counts(topo: np.ndarray) -> dict[str, int]:
    return {
        TOPOLOGY_LABEL_NAMES.get(int(code), str(int(code))): int(np.sum(topo == int(code)))
        for code in sorted(set(topo.tolist()))
    }


def trusted_phase_mask(data: dict[str, Any]) -> np.ndarray:
    return (
        np.asarray(data["trusted_exact"]).astype(bool)
        & np.asarray(data["training_eligible_exact"]).astype(bool)
        & ~np.asarray(data["rerun_required"]).astype(bool)
        & ~np.asarray(data["q_unresolved"]).astype(bool)
        & ~np.asarray(data["delta_unresolved"]).astype(bool)
    )


def trusted_gapped_topology_mask(data: dict[str, Any]) -> np.ndarray:
    topo = np.asarray(data["topology_label_code"], dtype=np.int64)
    spectral = np.asarray(data["topology_spectral_status_code"], dtype=np.int64)
    return (
        trusted_phase_mask(data)
        & np.asarray(data["topology_trusted"]).astype(bool)
        & np.isin(topo, [0, 1])
        & ((spectral == 0) | (spectral < 0))
    )


def nearest_query(query: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1, 3)
    if query.shape[0] == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)
    if ref.shape[0] == 0:
        return np.full(query.shape[0], np.inf), np.full(query.shape[0], -1, dtype=np.int64)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        dist, idx = cKDTree(ref).query(query, k=1)
        return np.asarray(dist, dtype=np.float64), np.asarray(idx, dtype=np.int64)
    except Exception:
        dist = np.full(query.shape[0], np.inf)
        idx = np.full(query.shape[0], -1, dtype=np.int64)
        for start in range(0, query.shape[0], 1024):
            chunk = query[start : start + 1024]
            d2 = np.sum((chunk[:, None, :] - ref[None, :, :]) ** 2, axis=2)
            local = np.argmin(d2, axis=1)
            dist[start : start + chunk.shape[0]] = np.sqrt(np.min(d2, axis=1))
            idx[start : start + chunk.shape[0]] = local
        return dist, idx


def predict_nearest_labels(
    cloud: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    support_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return np.full(cloud.shape[0], -999, dtype=np.int64), np.zeros(cloud.shape[0], dtype=bool)
    dist, idx = nearest_query(cloud, points)
    valid = np.isfinite(dist) & (dist <= float(support_radius)) & (idx >= 0)
    pred = np.full(cloud.shape[0], -999, dtype=np.int64)
    pred[valid] = labels[idx[valid]]
    return pred, valid


def opposite_midpoints(
    points: np.ndarray,
    labels: np.ndarray,
    cfg: StageIV3DConfig,
    *,
    k: int,
    max_distance: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels, dtype=np.int64)
    if points.shape[0] < 2:
        return np.empty((0, 3), dtype=np.float64)
    pts = normalized(points, cfg)
    pairs: list[tuple[float, int, int]] = []
    try:
        from scipy.spatial import cKDTree  # type: ignore

        kk = min(int(k) + 1, pts.shape[0])
        dist, idx = cKDTree(pts).query(pts, k=kk)
        for i in range(pts.shape[0]):
            for d, j in zip(np.ravel(dist[i]), np.ravel(idx[i])):
                j = int(j)
                if j <= i:
                    continue
                if labels[i] == labels[j]:
                    continue
                if float(d) > float(max_distance):
                    continue
                pairs.append((float(d), i, j))
    except Exception:
        for i in range(pts.shape[0]):
            d = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
            for j in np.argsort(d)[1 : int(k) + 1]:
                if j <= i or labels[i] == labels[j] or d[j] > float(max_distance):
                    continue
                pairs.append((float(d[j]), i, int(j)))
    if not pairs:
        return np.empty((0, 3), dtype=np.float64)
    pairs.sort(key=lambda row: row[0])
    unique: dict[tuple[int, int], tuple[float, int, int]] = {}
    for d, i, j in pairs:
        unique[(i, j)] = (d, i, j)
    rows = list(unique.values())
    return np.asarray([0.5 * (points[i] + points[j]) for _, i, j in rows], dtype=np.float64)


def surface_points(data: dict[str, Any], cfg: StageIV3DConfig, surface: str, k: int, max_distance: float) -> np.ndarray:
    x = np.asarray(data["x"], dtype=np.float64)
    phase = np.asarray(data["y_phase"], dtype=np.int64)
    mask_phase = trusted_phase_mask(data)
    if surface == "normal_sc":
        mask = mask_phase & np.isin(phase, [PHASE_NORMAL, PHASE_UNIFORM_SC, PHASE_FFLO])
        labels = (phase[mask] != PHASE_NORMAL).astype(np.int64)
        return opposite_midpoints(x[mask], labels, cfg, k=k, max_distance=max_distance)
    if surface == "uniform_fflo":
        mask = mask_phase & np.isin(phase, [PHASE_UNIFORM_SC, PHASE_FFLO])
        labels = (phase[mask] == PHASE_FFLO).astype(np.int64)
        return opposite_midpoints(x[mask], labels, cfg, k=k, max_distance=max_distance)
    if surface == "topology":
        mask = trusted_gapped_topology_mask(data)
        labels = np.asarray(data["topology_label_code"], dtype=np.int64)[mask]
        return opposite_midpoints(x[mask], labels, cfg, k=k, max_distance=max_distance)
    raise ValueError(f"Unknown surface {surface}")


def p95_symmetric_shift(a: np.ndarray, b: np.ndarray, cfg: StageIV3DConfig) -> dict[str, Any]:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return {
            "status": "missing_surface",
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "n_current": int(a.shape[0]),
            "n_previous": int(b.shape[0]),
        }
    an = normalized(a, cfg)
    bn = normalized(b, cfg)
    d_ab, _ = nearest_query(an, bn)
    d_ba, _ = nearest_query(bn, an)
    d = np.concatenate([d_ab, d_ba])
    return {
        "status": "ok",
        "median": float(np.nanmedian(d)),
        "p90": float(np.nanpercentile(d, 90)),
        "p95": float(np.nanpercentile(d, 95)),
        "max": float(np.nanmax(d)),
        "n_current": int(a.shape[0]),
        "n_previous": int(b.shape[0]),
    }


def coverage(surface: np.ndarray, support: np.ndarray, cfg: StageIV3DConfig) -> dict[str, Any]:
    if surface.shape[0] == 0 or support.shape[0] == 0:
        return {"status": "missing_surface", "median": np.nan, "p90": np.nan, "p95": np.nan, "max": np.nan, "n": int(surface.shape[0])}
    d, _ = nearest_query(normalized(surface, cfg), normalized(support, cfg))
    return {
        "status": "ok",
        "median": float(np.nanmedian(d)),
        "p90": float(np.nanpercentile(d, 90)),
        "p95": float(np.nanpercentile(d, 95)),
        "max": float(np.nanmax(d)),
        "n": int(surface.shape[0]),
    }


def component_count(points: np.ndarray, cfg: StageIV3DConfig, radius: float, min_size: int) -> dict[str, Any]:
    if points.shape[0] == 0:
        return {"component_count": 0, "significant_component_count": 0, "largest_component_size": 0}
    pts = normalized(points, cfg)
    parent = np.arange(pts.shape[0], dtype=np.int64)

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

        pairs = cKDTree(pts).query_pairs(r=float(radius))
        for i, j in pairs:
            union(int(i), int(j))
    except Exception:
        for i in range(pts.shape[0]):
            d = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
            for j in np.where((d <= float(radius)) & (d > 0))[0]:
                union(i, int(j))
    roots = np.asarray([find(i) for i in range(pts.shape[0])])
    _, counts = np.unique(roots, return_counts=True)
    return {
        "component_count": int(counts.size),
        "significant_component_count": int(np.sum(counts >= int(min_size))),
        "largest_component_size": int(np.max(counts)) if counts.size else 0,
    }


def load_selected_metadata(run_dir: Path, iteration: int) -> pd.DataFrame:
    path = run_dir / f"iter{iteration:03d}" / "selected_points_metadata.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def new_row_mask(prev: dict[str, Any], cur: dict[str, Any]) -> np.ndarray:
    prev_keys = set(point_keys(prev).tolist())
    cur_keys = point_keys(cur)
    return np.asarray([k not in prev_keys for k in cur_keys], dtype=bool)


def surprise_metrics(prev: dict[str, Any], cur: dict[str, Any], cfg: StageIV3DConfig, support_radius: float) -> dict[str, Any]:
    new_mask = new_row_mask(prev, cur)
    x_new = np.asarray(cur["x"], dtype=np.float64)[new_mask]
    if x_new.shape[0] == 0:
        return {
            "new_rows": 0,
            "trusted_phase_denominator": 0,
            "trusted_phase_surprise": np.nan,
            "trusted_topology_denominator": 0,
            "trusted_topology_surprise": np.nan,
            "trusted_topology_status": "insufficient_surprise_support",
        }
    prev_phase_mask = trusted_phase_mask(prev)
    phase_pred, phase_valid = predict_nearest_labels(
        normalized(x_new, cfg),
        normalized(np.asarray(prev["x"])[prev_phase_mask], cfg),
        np.asarray(prev["y_phase"])[prev_phase_mask],
        support_radius,
    )
    cur_phase = np.asarray(cur["y_phase"])[new_mask]
    cur_phase_trusted = trusted_phase_mask(cur)[new_mask]
    phase_den = cur_phase_trusted & phase_valid
    phase_surprise = int(np.sum(phase_den & (phase_pred != cur_phase)))

    prev_topo_mask = trusted_gapped_topology_mask(prev)
    topo_pred, topo_valid = predict_nearest_labels(
        normalized(x_new, cfg),
        normalized(np.asarray(prev["x"])[prev_topo_mask], cfg),
        np.asarray(prev["topology_label_code"])[prev_topo_mask],
        support_radius,
    )
    cur_topo = np.asarray(cur["topology_label_code"])[new_mask]
    cur_topo_trusted = trusted_gapped_topology_mask(cur)[new_mask]
    topo_den = cur_topo_trusted & topo_valid
    topo_surprise = int(np.sum(topo_den & (topo_pred != cur_topo)))
    topo_den_count = int(np.sum(topo_den))
    return {
        "new_rows": int(np.sum(new_mask)),
        "trusted_phase_denominator": int(np.sum(phase_den)),
        "trusted_phase_n_surprise": int(phase_surprise),
        "trusted_phase_surprise": float(phase_surprise / max(int(np.sum(phase_den)), 1)) if np.sum(phase_den) else np.nan,
        "trusted_topology_denominator": topo_den_count,
        "trusted_topology_n_surprise": int(topo_surprise),
        "trusted_topology_surprise": float(topo_surprise / max(topo_den_count, 1)) if topo_den_count else np.nan,
        "trusted_topology_status": "ok" if topo_den_count >= 16 else "insufficient_surprise_support",
    }


def iteration_summary_rows(datasets: list[dict[str, Any]], cfg: StageIV3DConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in datasets:
        x = np.asarray(data["x"], dtype=np.float64)
        phase = np.asarray(data["y_phase"], dtype=np.int64)
        topo = np.asarray(data["topology_label_code"], dtype=np.int64)
        topo_mask = trusted_gapped_topology_mask(data)
        trusted = trusted_phase_mask(data)
        mu = x[:, 2] if x.shape[0] else np.asarray([], dtype=np.float64)
        pc = phase_counts(phase)
        tc = topology_counts(topo)
        rows.append(
            {
                "iteration": int(data["iteration"]),
                "dataset_path": str(data["path"]),
                "dataset_sha256": sha256_file(Path(data["path"])),
                "sample_count": int(x.shape[0]),
                "trusted_phase_count": int(np.sum(trusted)),
                "fflo_count": int(np.sum(phase == PHASE_FFLO)),
                "trusted_gapped_topology_count": int(np.sum(topo_mask)),
                "normal_count": pc.get("normal", 0),
                "uniform_SC_count": pc.get("uniform_SC", 0),
                "FFLO_count": pc.get("FFLO", 0),
                "trivial_count": tc.get("trivial", 0),
                "topological_count": tc.get("topological", 0),
                "gapless_SC_count": tc.get("gapless_SC", 0),
                "unresolved_topology_count": tc.get("unresolved", 0),
                "mu_min": float(np.nanmin(mu)) if mu.size else np.nan,
                "mu_max": float(np.nanmax(mu)) if mu.size else np.nan,
                "mu_mean": float(np.nanmean(mu)) if mu.size else np.nan,
                "mu_lower_edge_contact_count": int(np.sum(mu <= cfg.mu_min + 0.02 * (cfg.mu_max - cfg.mu_min))) if mu.size else 0,
                "mu_upper_edge_contact_count": int(np.sum(mu >= cfg.mu_max - 0.02 * (cfg.mu_max - cfg.mu_min))) if mu.size else 0,
            }
        )
    return rows


def make_audit_cloud(cfg: StageIV3DConfig, n: int) -> np.ndarray:
    return sobol_points_3d(int(n), cfg, seed_offset=900001)


def map_change_rows(datasets: list[dict[str, Any]], cfg: StageIV3DConfig, cloud: np.ndarray, support_radius: float) -> list[dict[str, Any]]:
    predictions: dict[int, dict[str, Any]] = {}
    for data in datasets:
        phase_mask = trusted_phase_mask(data)
        topo_mask = trusted_gapped_topology_mask(data)
        phase_pred, phase_valid = predict_nearest_labels(
            normalized(cloud, cfg),
            normalized(np.asarray(data["x"])[phase_mask], cfg),
            np.asarray(data["y_phase"])[phase_mask],
            support_radius,
        )
        topo_pred, topo_valid = predict_nearest_labels(
            normalized(cloud, cfg),
            normalized(np.asarray(data["x"])[topo_mask], cfg),
            np.asarray(data["topology_label_code"])[topo_mask],
            support_radius,
        )
        predictions[int(data["iteration"])] = {
            "phase_pred": phase_pred,
            "phase_valid": phase_valid,
            "topo_pred": topo_pred,
            "topo_valid": topo_valid,
        }
    rows: list[dict[str, Any]] = []
    for prev, cur in zip(datasets[:-1], datasets[1:]):
        pi = int(prev["iteration"])
        ci = int(cur["iteration"])
        p = predictions[pi]
        c = predictions[ci]
        phase_common = p["phase_valid"] & c["phase_valid"]
        topo_common = p["topo_valid"] & c["topo_valid"]
        rows.append(
            {
                "transition_from_iteration": pi,
                "transition_to_iteration": ci,
                "phase_valid_common_count": int(np.sum(phase_common)),
                "phase_valid_common_fraction": float(np.mean(phase_common)) if phase_common.size else np.nan,
                "phase_volume_map_change": float(np.mean(p["phase_pred"][phase_common] != c["phase_pred"][phase_common])) if np.any(phase_common) else np.nan,
                "topology_valid_common_count": int(np.sum(topo_common)),
                "topology_valid_common_fraction": float(np.mean(topo_common)) if topo_common.size else np.nan,
                "topology_volume_map_change": float(np.mean(p["topo_pred"][topo_common] != c["topo_pred"][topo_common])) if np.any(topo_common) else np.nan,
                "newly_valid_topology_fraction": float(np.mean(c["topo_valid"] & ~p["topo_valid"])) if c["topo_valid"].size else np.nan,
                "no_longer_valid_topology_fraction": float(np.mean(p["topo_valid"] & ~c["topo_valid"])) if c["topo_valid"].size else np.nan,
            }
        )
    return rows


def surface_tables(
    datasets: list[dict[str, Any]],
    cfg: StageIV3DConfig,
    *,
    k: int,
    max_distance: float,
    support_radius: float,
    component_radius: float,
    component_min_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    surfaces: dict[tuple[int, str], np.ndarray] = {}
    for data in datasets:
        for name in ["normal_sc", "uniform_fflo", "topology"]:
            surfaces[(int(data["iteration"]), name)] = surface_points(data, cfg, name, k, max_distance)
    coverage_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for data in datasets:
        phase_support = np.asarray(data["x"])[trusted_phase_mask(data)]
        topo_support = np.asarray(data["x"])[trusted_gapped_topology_mask(data)]
        for name in ["normal_sc", "uniform_fflo", "topology"]:
            surf = surfaces[(int(data["iteration"]), name)]
            support = topo_support if name == "topology" else phase_support
            cov = coverage(surf, support, cfg)
            comp = component_count(surf, cfg, component_radius, component_min_size)
            coverage_rows.append({"iteration": int(data["iteration"]), "surface": name, **cov})
            component_rows.append({"iteration": int(data["iteration"]), "surface": name, **comp})
    shift_rows: list[dict[str, Any]] = []
    for prev, cur in zip(datasets[:-1], datasets[1:]):
        for name in ["normal_sc", "uniform_fflo", "topology"]:
            vals = p95_symmetric_shift(
                surfaces[(int(cur["iteration"]), name)],
                surfaces[(int(prev["iteration"]), name)],
                cfg,
            )
            shift_rows.append({"transition_from_iteration": int(prev["iteration"]), "transition_to_iteration": int(cur["iteration"]), "surface": name, **vals})
    return shift_rows, coverage_rows, component_rows


def acquisition_channel_rows(run_dir: Path, datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in datasets:
        it = int(data["iteration"])
        meta = load_selected_metadata(run_dir, it)
        if meta.empty:
            rows.append({"iteration": it, "metadata_available": False})
            continue
        for channel, sub in meta.groupby(meta.get("acquisition_channel", pd.Series(["unknown"] * len(meta))).astype(str)):
            rows.append(
                {
                    "iteration": it,
                    "metadata_available": True,
                    "acquisition_channel": channel,
                    "count": int(len(sub)),
                    "mean_score": float(pd.to_numeric(sub.get("score", np.nan), errors="coerce").mean()),
                }
            )
    return rows


def mu_domain_rows(datasets: list[dict[str, Any]], cfg: StageIV3DConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    width = 0.02 * (float(cfg.mu_max) - float(cfg.mu_min))
    for data in datasets:
        x = np.asarray(data["x"], dtype=np.float64)
        topo_mask = trusted_gapped_topology_mask(data)
        topo_label = np.asarray(data["topology_label_code"], dtype=np.int64)
        tfflo_mu = x[topo_mask & (topo_label == 1), 2] if x.shape[0] else np.asarray([])
        all_mu = x[:, 2] if x.shape[0] else np.asarray([])
        rows.append(
            {
                "iteration": int(data["iteration"]),
                "mu_min_observed": float(np.nanmin(all_mu)) if all_mu.size else np.nan,
                "mu_max_observed": float(np.nanmax(all_mu)) if all_mu.size else np.nan,
                "trusted_tfflo_count": int(tfflo_mu.size),
                "trusted_tfflo_near_lower_mu_count": int(np.sum(tfflo_mu <= float(cfg.mu_min) + width)) if tfflo_mu.size else 0,
                "trusted_tfflo_near_upper_mu_count": int(np.sum(tfflo_mu >= float(cfg.mu_max) - width)) if tfflo_mu.size else 0,
                "mu_range_limited_lower": bool(np.any(tfflo_mu <= float(cfg.mu_min) + width)) if tfflo_mu.size else False,
                "mu_range_limited_upper": bool(np.any(tfflo_mu >= float(cfg.mu_max) - width)) if tfflo_mu.size else False,
            }
        )
    return rows


def surprise_rows(datasets: list[dict[str, Any]], cfg: StageIV3DConfig, support_radius: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prev, cur in zip(datasets[:-1], datasets[1:]):
        vals = surprise_metrics(prev, cur, cfg, support_radius)
        rows.append({"transition_from_iteration": int(prev["iteration"]), "transition_to_iteration": int(cur["iteration"]), **vals})
    return rows


def final_decision(
    datasets: list[dict[str, Any]],
    map_rows: list[dict[str, Any]],
    shift_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    surprise: list[dict[str, Any]],
    mu_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(datasets) < 5:
        return {
            "run_id": "stageiv_3d_convergence_audit",
            "source_run_id": STAGEIV_RUN_ID,
            "final_iteration": int(datasets[-1]["iteration"]),
            "stageiv_convergence_status": "insufficient_history",
            "decision_class": "Decision D",
            "need_new_exact_calculation": False,
            "recommended_next_action": "run_or_return_more_stageiv_iterations",
            "caveats": ["At least five cumulative datasets are required for last-three-transition audit."],
        }
    last_iters = [int(d["iteration"]) for d in datasets[-4:]]
    last_transitions = set(zip(last_iters[:-1], last_iters[1:]))
    map_last = [r for r in map_rows if (r["transition_from_iteration"], r["transition_to_iteration"]) in last_transitions]
    topo_shift_last = [r for r in shift_rows if r["surface"] == "topology" and (r["transition_from_iteration"], r["transition_to_iteration"]) in last_transitions]
    topo_surprise_last = [r for r in surprise if (r["transition_from_iteration"], r["transition_to_iteration"]) in last_transitions]
    final_iter = int(datasets[-1]["iteration"])
    final_topo_cov = [r for r in coverage_rows if r["surface"] == "topology" and int(r["iteration"]) == final_iter]
    final_topo_comp = [r for r in component_rows if r["surface"] == "topology" and int(r["iteration"]) == final_iter]
    last_comp = [r for r in component_rows if r["surface"] == "topology" and int(r["iteration"]) in last_iters[1:]]

    map_vals = [float(r["topology_volume_map_change"]) for r in map_last if np.isfinite(float(r["topology_volume_map_change"]))]
    shift_vals = [float(r["p95"]) for r in topo_shift_last if r["status"] == "ok" and np.isfinite(float(r["p95"]))]
    surprise_vals = [
        float(r["trusted_topology_surprise"])
        for r in topo_surprise_last
        if r.get("trusted_topology_status") == "ok" and np.isfinite(float(r["trusted_topology_surprise"]))
    ]
    coverage_val = float(final_topo_cov[0]["p95"]) if final_topo_cov and np.isfinite(float(final_topo_cov[0]["p95"])) else np.nan
    component_counts = [int(r["significant_component_count"]) for r in last_comp]
    component_stable = bool(component_counts and len(set(component_counts)) == 1 and component_counts[-1] > 0)
    map_pass = len(map_vals) == 3 and all(v < MAP_CHANGE_TOL for v in map_vals)
    shift_pass = len(shift_vals) == 3 and all(v <= SURFACE_SHIFT_TOL for v in shift_vals)
    surprise_pass = len(surprise_vals) == 3 and all(v <= SURPRISE_TOL for v in surprise_vals)
    coverage_pass = np.isfinite(coverage_val) and coverage_val < COVERAGE_TOL
    if map_pass and shift_pass and surprise_pass and coverage_pass and component_stable:
        status = "preliminary_pass"
        decision_class = "Decision A"
        action = "freeze_stageiv_result_after_full_surface_visual_review"
    elif map_pass and shift_pass and surprise_pass and np.isfinite(coverage_val) and coverage_val < 1.25 * COVERAGE_TOL:
        status = "near_converged_coverage_limited"
        decision_class = "Decision B"
        action = "consider_1_to_3_stageiv_spectral_tail_batches_after_review"
    elif not map_pass or not shift_pass or not surprise_pass:
        status = "not_converged"
        decision_class = "Decision C"
        action = "inspect_failed_surfaces_before_more_exact_work"
    else:
        status = "inconclusive"
        decision_class = "Decision D"
        action = "inspect_missing_surface_or_support_metadata"
    final_mu = mu_rows[-1] if mu_rows else {}
    mu_range_limited = bool(final_mu.get("mu_range_limited_lower", False) or final_mu.get("mu_range_limited_upper", False))
    return {
        "run_id": "stageiv_3d_convergence_audit",
        "source_run_id": STAGEIV_RUN_ID,
        "final_iteration": final_iter,
        "stageiv_convergence_status": status,
        "decision_class": decision_class,
        "topology_volume_map_change_last3": map_vals,
        "topology_surface_shift_p95_last3": shift_vals,
        "topology_surface_coverage_p95_final": coverage_val,
        "trusted_topology_surprise_last3": surprise_vals,
        "topology_surface_component_count_last3": component_counts,
        "topology_component_stable": component_stable,
        "mu_range_limited": mu_range_limited,
        "mu_domain_complete": not mu_range_limited,
        "need_new_exact_calculation": decision_class in {"Decision B", "Decision C"},
        "recommended_next_action": action,
        "caveats": [
            "This audit uses report-only nearest-neighbor/KNN proxy surfaces and must be paired with visual slice/surface review.",
            "Missing surfaces are never converted to zero shift.",
            "Stage III fixed-mu hidden-slice validation is reported separately if reference data are supplied.",
        ],
    }


def write_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)


def plot_lines(frame: pd.DataFrame, x: str, y_cols: list[str], path: Path, title: str, threshold: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col in y_cols:
        if col in frame.columns:
            ax.plot(frame[x], frame[col], marker="o", label=col)
    if threshold is not None:
        ax.axhline(threshold, ls="--", c="k", lw=1, label="threshold")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_counts(frame: pd.DataFrame, cols: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col in cols:
        if col in frame.columns:
            ax.plot(frame["iteration"], frame[col], marker="o", label=col)
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def make_figures(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    fig_dir = output_dir / "figures"
    summary = tables["stageiv_iteration_summary"]
    if not summary.empty:
        plot_counts(summary, ["sample_count", "trusted_phase_count", "trusted_gapped_topology_count"], fig_dir / "dataset_growth.png", "Stage IV dataset growth")
        plot_counts(summary, ["normal_count", "uniform_SC_count", "FFLO_count"], fig_dir / "phase_counts.png", "Thermodynamic phase counts")
        plot_counts(summary, ["trivial_count", "topological_count", "gapless_SC_count", "unresolved_topology_count"], fig_dir / "topology_counts.png", "Topology diagnostic counts")
        plot_lines(summary, "iteration", ["mu_min", "mu_max", "mu_mean"], fig_dir / "mu_coverage.png", "Observed mu coverage")
    map_frame = tables["stageiv_volume_map_change"]
    if not map_frame.empty:
        plot_lines(map_frame, "transition_to_iteration", ["phase_volume_map_change", "topology_volume_map_change"], fig_dir / "volume_map_change.png", "3D volume-map change proxy", MAP_CHANGE_TOL)
    shift = tables["stageiv_surface_shift"]
    if not shift.empty:
        topo = shift[shift["surface"] == "topology"]
        plot_lines(topo, "transition_to_iteration", ["median", "p95", "max"], fig_dir / "topology_surface_shift.png", "Topology surface shift proxy", SURFACE_SHIFT_TOL)
    cov = tables["stageiv_surface_coverage"]
    if not cov.empty:
        topo = cov[cov["surface"] == "topology"]
        plot_lines(topo, "iteration", ["median", "p95", "max"], fig_dir / "topology_surface_coverage.png", "Topology surface coverage proxy", COVERAGE_TOL)
    surprise = tables["stageiv_surprise_by_iteration"]
    if not surprise.empty:
        plot_lines(surprise, "transition_to_iteration", ["trusted_phase_surprise", "trusted_topology_surprise"], fig_dir / "trusted_surprise.png", "Trusted surprise proxy", SURPRISE_TOL)


def fmt_value(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.6g}"


def write_markdown(output_dir: Path, decision: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
    summary = tables["stageiv_iteration_summary"]
    final = summary.iloc[-1].to_dict() if not summary.empty else {}
    lines = [
        "# Stage IV 3D Convergence Audit",
        "",
        "This is a report-only post-run audit. It does not launch exact calculations, Delta-q search, or active-learning iterations.",
        "",
        "## Executive Summary",
        "",
        f"- source_run_id: `{decision.get('source_run_id')}`",
        f"- final_iteration: `{decision.get('final_iteration')}`",
        f"- status: `{decision.get('stageiv_convergence_status')}`",
        f"- decision_class: `{decision.get('decision_class')}`",
        f"- need_new_exact_calculation: `{decision.get('need_new_exact_calculation')}`",
        f"- recommended_next_action: `{decision.get('recommended_next_action')}`",
        "",
        "## Final Dataset",
        "",
        f"- sample_count: `{final.get('sample_count', 'n/a')}`",
        f"- trusted_phase_count: `{final.get('trusted_phase_count', 'n/a')}`",
        f"- trusted_gapped_topology_count: `{final.get('trusted_gapped_topology_count', 'n/a')}`",
        f"- observed mu range: `{fmt_value(final.get('mu_min', np.nan))}` to `{fmt_value(final.get('mu_max', np.nan))}`",
        "",
        "## Last-Three Metrics",
        "",
        f"- topology_volume_map_change_last3: `{decision.get('topology_volume_map_change_last3', [])}`",
        f"- topology_surface_shift_p95_last3: `{decision.get('topology_surface_shift_p95_last3', [])}`",
        f"- topology_surface_coverage_p95_final: `{fmt_value(decision.get('topology_surface_coverage_p95_final', np.nan))}`",
        f"- trusted_topology_surprise_last3: `{decision.get('trusted_topology_surprise_last3', [])}`",
        f"- topology_surface_component_count_last3: `{decision.get('topology_surface_component_count_last3', [])}`",
        "",
        "## Caveats",
        "",
    ]
    for caveat in decision.get("caveats", []):
        lines.append(f"- {caveat}")
    lines += [
        "",
        "## Outputs",
        "",
        "- `tables/*.csv` contain the machine-readable audit metrics.",
        "- `figures/*.png` and `figures/*.pdf` contain the generated diagnostic plots.",
        "- `stageiv_3d_convergence_decision.json` contains the formal post-run decision payload.",
        "",
        "## Do-Not-Claim",
        "",
        "- Do not claim Stage IV production convergence from this audit if the status is `insufficient_history` or `inconclusive`.",
        "- Do not treat a missing 3D surface as zero shift.",
        "- Do not merge Stage III fixed-mu datasets into this cold-start run.",
        "- Do not treat nearest-neighbor proxy surfaces as publication meshes without visual/slice review.",
    ]
    write_text(output_dir / "stageiv_3d_convergence_audit.md", "\n".join(lines) + "\n")


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )


def write_latex_and_pdf(output_dir: Path, decision: dict[str, Any]) -> None:
    fig_dir = output_dir / "figures"
    figures = [
        ("dataset_growth.png", "Stage IV dataset growth."),
        ("phase_counts.png", "Thermodynamic phase counts."),
        ("topology_counts.png", "Topology diagnostic counts."),
        ("mu_coverage.png", "Observed mu coverage."),
        ("volume_map_change.png", "3D volume-map change proxy."),
        ("topology_surface_shift.png", "Topology-surface shift proxy."),
        ("topology_surface_coverage.png", "Topology-surface coverage proxy."),
        ("trusted_surprise.png", "Trusted surprise proxy."),
    ]
    body = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.8in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\title{Stage IV 3D Convergence Audit}",
        r"\author{Report-only post-run audit}",
        r"\date{\today}",
        r"\maketitle",
        r"\section*{Executive Summary}",
        f"Source run: \\texttt{{{latex_escape(str(decision.get('source_run_id')))}}}. "
        f"Status: \\texttt{{{latex_escape(str(decision.get('stageiv_convergence_status')))}}}. "
        f"Decision: \\texttt{{{latex_escape(str(decision.get('decision_class')))}}}. "
        f"Need new exact calculation: \\texttt{{{latex_escape(str(decision.get('need_new_exact_calculation')))}}}.",
        r"\section*{Key Metrics}",
        f"Topology volume map change last3: \\texttt{{{latex_escape(str(decision.get('topology_volume_map_change_last3', [])))}}}.",
        f"Topology surface p95 shift last3: \\texttt{{{latex_escape(str(decision.get('topology_surface_shift_p95_last3', [])))}}}.",
        f"Final topology coverage p95: \\texttt{{{latex_escape(fmt_value(decision.get('topology_surface_coverage_p95_final', np.nan)))}}}.",
        f"Trusted topology surprise last3: \\texttt{{{latex_escape(str(decision.get('trusted_topology_surprise_last3', [])))}}}.",
        r"\section*{Figures}",
    ]
    for name, caption in figures:
        if (fig_dir / name).exists():
            body += [
                r"\begin{figure}[H]",
                r"\centering",
                f"\\includegraphics[width=0.92\\linewidth]{{figures/{name}}}",
                f"\\caption{{{latex_escape(caption)}}}",
                r"\end{figure}",
            ]
    body += [
        r"\section*{Caveats}",
        r"\begin{itemize}",
    ]
    for caveat in decision.get("caveats", []):
        body.append(f"\\item {latex_escape(str(caveat))}")
    body += [
        r"\end{itemize}",
        r"\end{document}",
    ]
    tex = output_dir / "stageiv_3d_convergence_audit.tex"
    write_text(tex, "\n".join(body) + "\n")
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        write_json(output_dir / "pdf_build_status.json", {"status": "skipped", "reason": "pdflatex_not_found"})
        return
    result = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", tex.name],
        cwd=output_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    write_json(
        output_dir / "pdf_build_status.json",
        {"status": "pass" if result.returncode == 0 else "fail", "returncode": result.returncode, "stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-4000:]},
    )


def run_audit(
    run_dir: Path,
    output_dir: Path,
    cfg: StageIV3DConfig,
    *,
    audit_cloud_size: int,
    support_radius: float,
    neighbor_k: int,
    surface_max_distance: float,
    component_radius: float,
    component_min_size: int,
    build_pdf: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [load_dataset(p, cfg) for p in find_datasets(run_dir)]
    cloud = make_audit_cloud(cfg, audit_cloud_size)
    iteration_rows = iteration_summary_rows(datasets, cfg)
    map_rows = map_change_rows(datasets, cfg, cloud, support_radius)
    shift_rows, coverage_rows, component_rows = surface_tables(
        datasets,
        cfg,
        k=neighbor_k,
        max_distance=surface_max_distance,
        support_radius=support_radius,
        component_radius=component_radius,
        component_min_size=component_min_size,
    )
    surprise = surprise_rows(datasets, cfg, support_radius)
    acq_rows = acquisition_channel_rows(run_dir, datasets)
    mu_rows = mu_domain_rows(datasets, cfg)
    decision = final_decision(datasets, map_rows, shift_rows, coverage_rows, component_rows, surprise, mu_rows)
    decision["run_dir"] = str(run_dir)
    decision["audit_cloud_size"] = int(audit_cloud_size)
    decision["support_radius"] = float(support_radius)
    decision["neighbor_k"] = int(neighbor_k)
    decision["surface_max_distance"] = float(surface_max_distance)
    decision["component_radius"] = float(component_radius)
    decision["component_min_size"] = int(component_min_size)
    decision["final_dataset_path"] = str(datasets[-1]["path"])
    decision["final_dataset_sha256"] = sha256_file(Path(datasets[-1]["path"]))

    tables = {
        "stageiv_iteration_summary": pd.DataFrame(iteration_rows),
        "stageiv_volume_map_change": pd.DataFrame(map_rows),
        "stageiv_surface_shift": pd.DataFrame(shift_rows),
        "stageiv_surface_coverage": pd.DataFrame(coverage_rows),
        "stageiv_surface_components": pd.DataFrame(component_rows),
        "stageiv_surprise_by_iteration": pd.DataFrame(surprise),
        "stageiv_acquisition_channel_counts": pd.DataFrame(acq_rows),
        "stageiv_mu_domain_contact": pd.DataFrame(mu_rows),
    }
    write_tables(output_dir, tables)
    make_figures(output_dir, tables)
    write_json(output_dir / "stageiv_3d_convergence_decision.json", decision)
    write_markdown(output_dir, decision, tables)
    write_json(
        output_dir / "stageiv_3d_convergence_config.json",
        {
            "stageiv_config": cfg.to_dict(),
            "audit_cloud_size": int(audit_cloud_size),
            "support_radius": float(support_radius),
            "neighbor_k": int(neighbor_k),
            "surface_max_distance": float(surface_max_distance),
            "component_radius": float(component_radius),
            "component_min_size": int(component_min_size),
            "thresholds": {
                "volume_map_change": MAP_CHANGE_TOL,
                "surface_shift_p95": SURFACE_SHIFT_TOL,
                "coverage_p95": COVERAGE_TOL,
                "trusted_surprise": SURPRISE_TOL,
            },
        },
    )
    if build_pdf:
        write_latex_and_pdf(output_dir, decision)
    return decision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a report-only Stage IV 3D convergence audit.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--audit-cloud-size", type=int, default=20000)
    p.add_argument("--support-radius", type=float, default=0.075)
    p.add_argument("--neighbor-k", type=int, default=8)
    p.add_argument("--surface-max-distance", type=float, default=0.18)
    p.add_argument("--component-radius", type=float, default=0.035)
    p.add_argument("--component-min-size", type=int, default=12)
    p.add_argument("--no-pdf", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StageIV3DConfig.from_json(args.config) if args.config else StageIV3DConfig()
    decision = run_audit(
        args.run_dir,
        args.output_dir,
        cfg,
        audit_cloud_size=args.audit_cloud_size,
        support_radius=args.support_radius,
        neighbor_k=args.neighbor_k,
        surface_max_distance=args.surface_max_distance,
        component_radius=args.component_radius,
        component_min_size=args.component_min_size,
        build_pdf=not args.no_pdf,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
