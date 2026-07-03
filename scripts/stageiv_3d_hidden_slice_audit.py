from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import replace
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

from ml_phase.labels import PHASE_NAMES
from ml_phase.stageiv_3d import StageIV3DConfig


TOPOLOGY_LABEL_NAMES = {
    -1: "not_applicable",
    0: "trivial",
    1: "topological",
    2: "gapless_SC",
    3: "unresolved",
}

PHASE_NAME_TO_CODE = {v: k for k, v in PHASE_NAMES.items()}
PHASE_NAME_TO_CODE.update({"uniform-SC": 1, "uniform_SC": 1, "FFLO": 2, "fflo": 2, "normal": 0})
TOPO_NAME_TO_CODE = {
    "trivial": 0,
    "cFFLO": 0,
    "cfflo": 0,
    "topological": 1,
    "tFFLO": 1,
    "tfflo": 1,
    "gapless_SC": 2,
    "gapless-SC": 2,
    "unresolved": 3,
    "not_applicable": -1,
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def json_clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sorted_dataset_paths(run_dir: Path) -> list[Path]:
    def key(path: Path) -> int:
        stem = path.stem.replace("dataset_iter", "")
        try:
            return int(stem)
        except ValueError:
            return -1

    return sorted(run_dir.glob("dataset_iter*.npz"), key=key)


def iter_from_name(path: Path) -> int:
    return int(path.stem.replace("dataset_iter", ""))


def load_config(path: Path | None) -> StageIV3DConfig:
    if path is None or not path.exists():
        return StageIV3DConfig()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return StageIV3DConfig(**{k: v for k, v in payload.items() if hasattr(StageIV3DConfig, k)})


def arr_bool(z: Any, key: str, n: int, default: bool) -> np.ndarray:
    if key not in z.files:
        return np.full(n, default, dtype=bool)
    return np.asarray(z[key]).astype(bool)


def arr_int(z: Any, key: str, n: int, default: int) -> np.ndarray:
    if key not in z.files:
        return np.full(n, default, dtype=np.int64)
    return np.asarray(z[key]).astype(np.int64)


def load_npz_dataset(path: Path, cfg: StageIV3DConfig) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        x_raw = np.asarray(z["x"], dtype=np.float64)
        n = x_raw.shape[0]
        if x_raw.shape[1] >= 3:
            x3 = x_raw[:, :3]
        else:
            mu = np.asarray(z["mu"], dtype=np.float64) if "mu" in z.files else np.full(n, cfg.mu_reference)
            x3 = np.column_stack([x_raw[:, 0], x_raw[:, 1], mu])
        phase = arr_int(z, "y_phase", n, -1)
        topo = arr_int(z, "topology_label_code", n, -1)
        trusted = arr_bool(z, "trusted_exact", n, True)
        training = arr_bool(z, "training_eligible_exact", n, True)
        rerun = arr_bool(z, "rerun_required", n, False)
        q_unresolved = arr_bool(z, "q_unresolved", n, False)
        delta_unresolved = arr_bool(z, "delta_unresolved", n, False)
        topo_trusted = arr_bool(z, "topology_trusted", n, False)
        spectral = arr_int(z, "spectral_status_code", n, -1)
    trusted_phase = trusted & training & (~rerun) & (~q_unresolved) & (~delta_unresolved) & (phase >= 0)
    trusted_topo = trusted_phase & topo_trusted & np.isin(topo, [0, 1]) & np.isin(spectral, [-1, 0])
    return {
        "path": path,
        "sha256": sha256_file(path),
        "iteration": iter_from_name(path),
        "x": x3,
        "phase": phase,
        "topology": topo,
        "trusted_phase_mask": trusted_phase,
        "trusted_topology_mask": trusted_topo,
    }


def _series_to_codes(series: pd.Series, mapping: dict[str, int], default: int = -1) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(default).astype(int).to_numpy()
    return series.astype(str).map(lambda s: mapping.get(s, default)).astype(int).to_numpy()


def load_csv_reference(path: Path, cfg: StageIV3DConfig) -> dict[str, Any]:
    df = pd.read_csv(path)
    kt_col = "kBT" if "kBT" in df.columns else "kT" if "kT" in df.columns else None
    ja_col = "J_A" if "J_A" in df.columns else "JA" if "JA" in df.columns else None
    if kt_col is None or ja_col is None:
        raise ValueError(f"Reference CSV {path} must contain kBT/kT and J_A/JA columns.")
    mu = df["mu"].to_numpy(dtype=float) if "mu" in df.columns else np.full(len(df), cfg.mu_reference)
    x3 = np.column_stack([df[kt_col].to_numpy(dtype=float), df[ja_col].to_numpy(dtype=float), mu])
    if "y_phase" in df.columns:
        phase = df["y_phase"].fillna(-1).astype(int).to_numpy()
    elif "phase_code" in df.columns:
        phase = df["phase_code"].fillna(-1).astype(int).to_numpy()
    elif "phase_label" in df.columns:
        phase = _series_to_codes(df["phase_label"], PHASE_NAME_TO_CODE)
    else:
        phase = np.full(len(df), -1, dtype=np.int64)
    if "topology_label_code" in df.columns:
        topo = df["topology_label_code"].fillna(-1).astype(int).to_numpy()
    elif "topology_code" in df.columns:
        topo = df["topology_code"].fillna(-1).astype(int).to_numpy()
    elif "topology_label" in df.columns:
        topo = _series_to_codes(df["topology_label"], TOPO_NAME_TO_CODE)
    else:
        topo = np.full(len(df), -1, dtype=np.int64)
    trusted = df["trusted_exact"].astype(bool).to_numpy() if "trusted_exact" in df.columns else np.ones(len(df), dtype=bool)
    training = (
        df["training_eligible_exact"].astype(bool).to_numpy()
        if "training_eligible_exact" in df.columns
        else np.ones(len(df), dtype=bool)
    )
    rerun = df["rerun_required"].astype(bool).to_numpy() if "rerun_required" in df.columns else np.zeros(len(df), dtype=bool)
    topo_trusted = (
        df["topology_trusted"].astype(bool).to_numpy()
        if "topology_trusted" in df.columns
        else np.isin(topo, [0, 1])
    )
    trusted_phase = trusted & training & (~rerun) & (phase >= 0)
    trusted_topo = trusted_phase & topo_trusted & np.isin(topo, [0, 1])
    return {
        "path": path,
        "sha256": sha256_file(path),
        "iteration": -1,
        "x": x3,
        "phase": phase,
        "topology": topo,
        "trusted_phase_mask": trusted_phase,
        "trusted_topology_mask": trusted_topo,
    }


def load_reference(path: Path | None, cfg: StageIV3DConfig) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".npz":
        ref = load_npz_dataset(path, cfg)
        ref["iteration"] = -1
        return ref
    if path.suffix.lower() == ".csv":
        return load_csv_reference(path, cfg)
    raise ValueError(f"Unsupported reference dataset format: {path}")


def normalize_points(x: np.ndarray, cfg: StageIV3DConfig) -> np.ndarray:
    lo = np.array([cfg.kt_min, cfg.ja_min, cfg.mu_min], dtype=np.float64)
    hi = np.array([cfg.kt_max, cfg.ja_max, cfg.mu_max], dtype=np.float64)
    return (np.asarray(x, dtype=np.float64) - lo) / np.maximum(hi - lo, 1e-12)


def make_slice_grid(cfg: StageIV3DConfig, grid_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kt = np.linspace(cfg.kt_min, cfg.kt_max, grid_n)
    ja = np.linspace(cfg.ja_min, cfg.ja_max, grid_n)
    kk, jj = np.meshgrid(kt, ja, indexing="ij")
    x3 = np.column_stack([kk.ravel(), jj.ravel(), np.full(kk.size, cfg.mu_reference)])
    return kt, ja, kk, x3


def knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    k: int,
    support_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_x.shape[0] == 0:
        n = query_x.shape[0]
        return np.full(n, -1, dtype=np.int64), np.full(n, np.inf), np.zeros(n, dtype=bool)
    k_eff = max(1, min(k, train_x.shape[0]))
    labels = np.full(query_x.shape[0], -1, dtype=np.int64)
    nearest = np.full(query_x.shape[0], np.inf, dtype=np.float64)
    valid = np.zeros(query_x.shape[0], dtype=bool)
    chunk = 4096
    for start in range(0, query_x.shape[0], chunk):
        q = query_x[start : start + chunk]
        d = np.linalg.norm(q[:, None, :] - train_x[None, :, :], axis=2)
        order = np.argpartition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
        row = np.arange(q.shape[0])[:, None]
        vals = train_y[order]
        dd = d[row, order]
        nearest_chunk = np.min(dd, axis=1)
        weights = 1.0 / np.maximum(dd, 1e-12)
        pred = []
        for i in range(vals.shape[0]):
            scores: dict[int, float] = {}
            for lab, w in zip(vals[i], weights[i]):
                scores[int(lab)] = scores.get(int(lab), 0.0) + float(w)
            pred.append(max(scores.items(), key=lambda kv: (kv[1], -kv[0]))[0])
        labels[start : start + q.shape[0]] = np.asarray(pred, dtype=np.int64)
        nearest[start : start + q.shape[0]] = nearest_chunk
        valid[start : start + q.shape[0]] = nearest_chunk <= support_radius
    return labels, nearest, valid


def boundary_points(
    labels: np.ndarray,
    valid: np.ndarray,
    kt: np.ndarray,
    ja: np.ndarray,
    kind: str,
) -> np.ndarray:
    n = kt.size
    lab = labels.reshape(n, n)
    val = valid.reshape(n, n)
    pts: list[tuple[float, float]] = []

    def side(a: int) -> int:
        if kind == "normal_sc":
            return 0 if a == 0 else 1 if a in (1, 2) else -1
        if kind == "uniform_fflo":
            return 0 if a == 1 else 1 if a == 2 else -1
        if kind == "topology":
            return 0 if a == 0 else 1 if a == 1 else -1
        raise ValueError(kind)

    for i in range(n):
        for j in range(n):
            if not val[i, j]:
                continue
            s0 = side(int(lab[i, j]))
            if s0 < 0:
                continue
            if i + 1 < n and val[i + 1, j]:
                s1 = side(int(lab[i + 1, j]))
                if s1 >= 0 and s0 != s1:
                    pts.append(((kt[i] + kt[i + 1]) * 0.5, ja[j]))
            if j + 1 < n and val[i, j + 1]:
                s1 = side(int(lab[i, j + 1]))
                if s1 >= 0 and s0 != s1:
                    pts.append((kt[i], (ja[j] + ja[j + 1]) * 0.5))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def norm2(points: np.ndarray, cfg: StageIV3DConfig) -> np.ndarray:
    lo = np.array([cfg.kt_min, cfg.ja_min], dtype=float)
    hi = np.array([cfg.kt_max, cfg.ja_max], dtype=float)
    return (np.asarray(points, dtype=float) - lo) / np.maximum(hi - lo, 1e-12)


def nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.asarray([], dtype=np.float64)
    chunk = 4096
    out = np.empty(a.shape[0], dtype=np.float64)
    for start in range(0, a.shape[0], chunk):
        d = np.linalg.norm(a[start : start + chunk, None, :] - b[None, :, :], axis=2)
        out[start : start + chunk] = np.min(d, axis=1)
    return out


def boundary_metric(stage_pts: np.ndarray, ref_pts: np.ndarray, cfg: StageIV3DConfig) -> dict[str, Any]:
    if stage_pts.size == 0 or ref_pts.size == 0:
        return {
            "status": "missing_boundary",
            "stage_boundary_count": int(stage_pts.shape[0]) if stage_pts.ndim == 2 else 0,
            "reference_boundary_count": int(ref_pts.shape[0]) if ref_pts.ndim == 2 else 0,
            "symmetric_p95": math.nan,
            "hausdorff": math.nan,
        }
    a = norm2(stage_pts, cfg)
    b = norm2(ref_pts, cfg)
    ab = nearest_distances(a, b)
    ba = nearest_distances(b, a)
    both = np.concatenate([ab, ba])
    return {
        "status": "available",
        "stage_boundary_count": int(stage_pts.shape[0]),
        "reference_boundary_count": int(ref_pts.shape[0]),
        "stage_to_reference_median": float(np.median(ab)),
        "stage_to_reference_p95": float(np.percentile(ab, 95)),
        "reference_to_stage_median": float(np.median(ba)),
        "reference_to_stage_p95": float(np.percentile(ba, 95)),
        "symmetric_p95": float(np.percentile(both, 95)),
        "hausdorff": float(np.max(both)),
    }


def component_count(mask: np.ndarray, min_size: int) -> tuple[int, list[int]]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j] or seen[i, j]:
                continue
            q: deque[tuple[int, int]] = deque([(i, j)])
            seen[i, j] = True
            size = 0
            while q:
                a, b = q.popleft()
                size += 1
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    aa, bb = a + da, b + db
                    if 0 <= aa < mask.shape[0] and 0 <= bb < mask.shape[1] and mask[aa, bb] and not seen[aa, bb]:
                        seen[aa, bb] = True
                        q.append((aa, bb))
            if size >= min_size:
                sizes.append(size)
    sizes.sort(reverse=True)
    return len(sizes), sizes


def map_difference(stage: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> tuple[float, int, int]:
    denom = int(np.sum(valid))
    if denom == 0:
        return math.nan, 0, 0
    diff = int(np.sum((stage != ref) & valid))
    return float(diff / denom), diff, denom


def coverage_metric(boundary: np.ndarray, exact_x2_norm: np.ndarray, cfg: StageIV3DConfig) -> dict[str, Any]:
    if boundary.size == 0 or exact_x2_norm.size == 0:
        return {"status": "missing", "median": math.nan, "p95": math.nan, "max": math.nan}
    d = nearest_distances(norm2(boundary, cfg), exact_x2_norm)
    return {"status": "available", "median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(np.max(d))}


def evaluate_dataset(
    ds: dict[str, Any],
    ref: dict[str, Any],
    cfg: StageIV3DConfig,
    grid_n: int,
    k: int,
    support_radius: float,
    component_min_size: int,
) -> dict[str, Any]:
    kt, ja, _, grid = make_slice_grid(cfg, grid_n)
    grid_norm3 = normalize_points(grid, cfg)
    grid_norm2 = norm2(grid[:, :2], cfg)

    stage_phase_mask = ds["trusted_phase_mask"]
    ref_phase_mask = ref["trusted_phase_mask"]
    stage_phase, stage_phase_dist, stage_phase_valid = knn_predict(
        normalize_points(ds["x"][stage_phase_mask], cfg),
        ds["phase"][stage_phase_mask],
        grid_norm3,
        k,
        support_radius,
    )
    ref_phase, ref_phase_dist, ref_phase_valid = knn_predict(
        norm2(ref["x"][ref_phase_mask][:, :2], cfg),
        ref["phase"][ref_phase_mask],
        grid_norm2,
        k,
        support_radius,
    )
    phase_valid = stage_phase_valid & ref_phase_valid
    phase_map_change, phase_diff_count, phase_denom = map_difference(stage_phase, ref_phase, phase_valid)

    stage_topo_mask = ds["trusted_topology_mask"]
    ref_topo_mask = ref["trusted_topology_mask"]
    stage_topo, stage_topo_dist, stage_topo_valid = knn_predict(
        normalize_points(ds["x"][stage_topo_mask], cfg),
        ds["topology"][stage_topo_mask],
        grid_norm3,
        k,
        support_radius,
    )
    ref_topo, ref_topo_dist, ref_topo_valid = knn_predict(
        norm2(ref["x"][ref_topo_mask][:, :2], cfg),
        ref["topology"][ref_topo_mask],
        grid_norm2,
        k,
        support_radius,
    )
    topo_valid = stage_topo_valid & ref_topo_valid
    topo_map_change, topo_diff_count, topo_denom = map_difference(stage_topo, ref_topo, topo_valid)

    stage_ns = boundary_points(stage_phase, stage_phase_valid, kt, ja, "normal_sc")
    ref_ns = boundary_points(ref_phase, ref_phase_valid, kt, ja, "normal_sc")
    stage_uf = boundary_points(stage_phase, stage_phase_valid, kt, ja, "uniform_fflo")
    ref_uf = boundary_points(ref_phase, ref_phase_valid, kt, ja, "uniform_fflo")
    stage_tb = boundary_points(stage_topo, stage_topo_valid, kt, ja, "topology")
    ref_tb = boundary_points(ref_topo, ref_topo_valid, kt, ja, "topology")

    topo_stage_region = (stage_topo.reshape(grid_n, grid_n) == 1) & stage_topo_valid.reshape(grid_n, grid_n)
    topo_ref_region = (ref_topo.reshape(grid_n, grid_n) == 1) & ref_topo_valid.reshape(grid_n, grid_n)
    topo_region_valid = topo_valid.reshape(grid_n, grid_n)
    inter = int(np.sum(topo_stage_region & topo_ref_region & topo_region_valid))
    union = int(np.sum((topo_stage_region | topo_ref_region) & topo_region_valid))
    topo_jaccard = float(inter / union) if union else math.nan
    stage_comp, stage_sizes = component_count(topo_stage_region, component_min_size)
    ref_comp, ref_sizes = component_count(topo_ref_region, component_min_size)
    missed_components = max(0, ref_comp - stage_comp)

    stage_exact_x2 = norm2(ds["x"][stage_phase_mask][:, :2], cfg)
    stage_topo_x2 = norm2(ds["x"][stage_topo_mask][:, :2], cfg)

    return {
        "iteration": int(ds["iteration"]),
        "dataset_path": str(ds["path"]),
        "sample_count": int(ds["x"].shape[0]),
        "trusted_phase_count": int(np.sum(stage_phase_mask)),
        "trusted_topology_count": int(np.sum(stage_topo_mask)),
        "phase_map_change": phase_map_change,
        "phase_map_diff_count": phase_diff_count,
        "phase_map_denominator": phase_denom,
        "topology_map_change": topo_map_change,
        "topology_map_diff_count": topo_diff_count,
        "topology_map_denominator": topo_denom,
        "normal_sc_boundary": boundary_metric(stage_ns, ref_ns, cfg),
        "uniform_fflo_boundary": boundary_metric(stage_uf, ref_uf, cfg),
        "topology_boundary": boundary_metric(stage_tb, ref_tb, cfg),
        "normal_sc_coverage": coverage_metric(stage_ns, stage_exact_x2, cfg),
        "uniform_fflo_coverage": coverage_metric(stage_uf, stage_exact_x2, cfg),
        "topology_coverage": coverage_metric(stage_tb, stage_topo_x2, cfg),
        "topology_region_jaccard": topo_jaccard,
        "stage_topological_component_count": int(stage_comp),
        "reference_topological_component_count": int(ref_comp),
        "missed_topological_component_count": int(missed_components),
        "stage_largest_topological_component_cells": int(stage_sizes[0]) if stage_sizes else 0,
        "reference_largest_topological_component_cells": int(ref_sizes[0]) if ref_sizes else 0,
        "phase_valid_fraction": float(np.mean(phase_valid)),
        "topology_valid_fraction": float(np.mean(topo_valid)),
        "stage_phase_grid": stage_phase.reshape(grid_n, grid_n),
        "ref_phase_grid": ref_phase.reshape(grid_n, grid_n),
        "stage_topo_grid": stage_topo.reshape(grid_n, grid_n),
        "ref_topo_grid": ref_topo.reshape(grid_n, grid_n),
        "kt": kt,
        "ja": ja,
    }


def flatten_metrics(evaluations: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    boundary_rows = []
    component_rows = []
    for ev in evaluations:
        summary_rows.append(
            {
                "iteration": ev["iteration"],
                "sample_count": ev["sample_count"],
                "trusted_phase_count": ev["trusted_phase_count"],
                "trusted_topology_count": ev["trusted_topology_count"],
                "phase_map_change": ev["phase_map_change"],
                "phase_map_denominator": ev["phase_map_denominator"],
                "topology_map_change": ev["topology_map_change"],
                "topology_map_denominator": ev["topology_map_denominator"],
                "phase_valid_fraction": ev["phase_valid_fraction"],
                "topology_valid_fraction": ev["topology_valid_fraction"],
            }
        )
        component_rows.append(
            {
                "iteration": ev["iteration"],
                "topology_region_jaccard": ev["topology_region_jaccard"],
                "stage_topological_component_count": ev["stage_topological_component_count"],
                "reference_topological_component_count": ev["reference_topological_component_count"],
                "missed_topological_component_count": ev["missed_topological_component_count"],
                "stage_largest_topological_component_cells": ev["stage_largest_topological_component_cells"],
                "reference_largest_topological_component_cells": ev["reference_largest_topological_component_cells"],
            }
        )
        for boundary_type, key in (
            ("normal_sc", "normal_sc_boundary"),
            ("uniform_fflo", "uniform_fflo_boundary"),
            ("topology", "topology_boundary"),
        ):
            row = {"iteration": ev["iteration"], "boundary_type": boundary_type}
            row.update(ev[key])
            cov = ev[f"{boundary_type}_coverage" if boundary_type != "topology" else "topology_coverage"]
            row.update({f"coverage_{k}": v for k, v in cov.items()})
            boundary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(boundary_rows), pd.DataFrame(component_rows)


def build_decision(
    reference_path: Path | None,
    datasets: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if reference_path is None:
        return {
            "run_id": "stageiv_3d_hidden_slice_audit",
            "source_run_id": args.source_run_id,
            "hidden_slice_status": "inconclusive",
            "decision_class": "Decision D",
            "hidden_slice_passed": False,
            "reason": "reference_dataset_missing",
            "need_new_exact_calculation": False,
            "recommended_next_action": "provide_stageiii_frozen_reference_dataset_after_stageiv_return",
        }
    if len(datasets) == 0:
        return {
            "run_id": "stageiv_3d_hidden_slice_audit",
            "source_run_id": args.source_run_id,
            "hidden_slice_status": "inconclusive",
            "decision_class": "Decision D",
            "hidden_slice_passed": False,
            "reason": "stageiv_dataset_missing",
            "need_new_exact_calculation": False,
            "recommended_next_action": "return_stageiv_cumulative_datasets",
        }
    if not evaluations:
        return {
            "run_id": "stageiv_3d_hidden_slice_audit",
            "source_run_id": args.source_run_id,
            "hidden_slice_status": "inconclusive",
            "decision_class": "Decision D",
            "hidden_slice_passed": False,
            "reason": "evaluation_unavailable",
            "need_new_exact_calculation": False,
            "recommended_next_action": "inspect_hidden_slice_inputs",
        }
    final = evaluations[-1]
    topology_boundary = final["topology_boundary"]
    topology_coverage = final["topology_coverage"]
    pass_checks = {
        "phase_map": bool(np.isfinite(final["phase_map_change"]) and final["phase_map_change"] <= args.phase_map_tol),
        "topology_map": bool(np.isfinite(final["topology_map_change"]) and final["topology_map_change"] <= args.topology_map_tol),
        "topology_boundary_shift": bool(
            topology_boundary.get("status") == "available"
            and np.isfinite(topology_boundary.get("symmetric_p95", math.nan))
            and topology_boundary["symmetric_p95"] <= args.boundary_shift_tol
        ),
        "topology_coverage": bool(
            topology_coverage.get("status") == "available"
            and np.isfinite(topology_coverage.get("p95", math.nan))
            and topology_coverage["p95"] <= args.coverage_tol
        ),
        "topology_overlap": bool(
            np.isfinite(final["topology_region_jaccard"]) and final["topology_region_jaccard"] >= args.topology_overlap_min
        ),
        "missed_components": bool(final["missed_topological_component_count"] == 0),
    }
    hidden_pass = all(pass_checks.values())
    return {
        "run_id": "stageiv_3d_hidden_slice_audit",
        "source_run_id": args.source_run_id,
        "mu_reference": args.mu_reference,
        "final_iteration": int(final["iteration"]),
        "hidden_slice_status": "pass" if hidden_pass else "fail",
        "decision_class": "Decision A" if hidden_pass else "Decision C",
        "hidden_slice_passed": bool(hidden_pass),
        "pass_checks": pass_checks,
        "phase_map_change_final": final["phase_map_change"],
        "topology_map_change_final": final["topology_map_change"],
        "topology_boundary_shift_p95_final": topology_boundary.get("symmetric_p95"),
        "topology_boundary_coverage_p95_final": topology_coverage.get("p95"),
        "topology_region_jaccard_final": final["topology_region_jaccard"],
        "missed_topological_component_count_final": final["missed_topological_component_count"],
        "thresholds": {
            "phase_map_tol": args.phase_map_tol,
            "topology_map_tol": args.topology_map_tol,
            "boundary_shift_tol": args.boundary_shift_tol,
            "coverage_tol": args.coverage_tol,
            "topology_overlap_min": args.topology_overlap_min,
        },
        "need_new_exact_calculation": False,
        "recommended_next_action": "include_hidden_slice_pass_in_stageiv_final_report"
        if hidden_pass
        else "inspect_failed_hidden_slice_metrics_before_claiming_3d_capability_benchmark",
        "caveats": [
            "This report-only audit does not add Stage III samples to Stage IV training.",
            "If no saved Stage IV surrogate checkpoint is available, metrics use a fixed KNN proxy from returned exact data.",
            "Hidden-slice failure does not by itself invalidate the 3D run, but it blocks the 3D capability benchmark claim.",
        ],
    }


def make_figures(output_dir: Path, evaluations: list[dict[str, Any]], summary: pd.DataFrame, boundary: pd.DataFrame) -> None:
    figdir = output_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(summary["iteration"], summary["phase_map_change"], marker="o", label="phase map")
        ax.plot(summary["iteration"], summary["topology_map_change"], marker="o", label="topology map")
        ax.set_xlabel("Stage IV iteration")
        ax.set_ylabel("map difference vs hidden reference")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "hidden_slice_map_change.png", dpi=180)
        fig.savefig(figdir / "hidden_slice_map_change.pdf")
        plt.close(fig)
    if not boundary.empty and "symmetric_p95" in boundary.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for btype, grp in boundary.groupby("boundary_type"):
            ax.plot(grp["iteration"], grp["symmetric_p95"], marker="o", label=f"{btype} shift p95")
        ax.set_xlabel("Stage IV iteration")
        ax.set_ylabel("bidirectional p95 distance")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "hidden_slice_boundary_shift.png", dpi=180)
        fig.savefig(figdir / "hidden_slice_boundary_shift.pdf")
        plt.close(fig)
    if evaluations:
        ev = evaluations[-1]
        kt = ev["kt"]
        ja = ev["ja"]
        extent = [ja.min(), ja.max(), kt.min(), kt.max()]
        for name, stage_key, ref_key in (
            ("phase", "stage_phase_grid", "ref_phase_grid"),
            ("topology", "stage_topo_grid", "ref_topo_grid"),
        ):
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
            axes[0].imshow(ev[ref_key], origin="lower", aspect="auto", extent=extent, interpolation="nearest")
            axes[0].set_title(f"Stage III reference {name}")
            axes[1].imshow(ev[stage_key], origin="lower", aspect="auto", extent=extent, interpolation="nearest")
            axes[1].set_title(f"Stage IV slice {name}")
            for ax in axes:
                ax.set_xlabel("J_A/t")
                ax.set_ylabel("kBT/t")
            fig.tight_layout()
            fig.savefig(figdir / f"hidden_slice_{name}_comparison.png", dpi=180)
            fig.savefig(figdir / f"hidden_slice_{name}_comparison.pdf")
            plt.close(fig)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_reports(output_dir: Path, decision: dict[str, Any], summary: pd.DataFrame, boundary: pd.DataFrame) -> None:
    def df_to_md(df: pd.DataFrame) -> str:
        if df.empty:
            return "_empty_"
        slim = df.copy()
        for col in slim.columns:
            if pd.api.types.is_float_dtype(slim[col]):
                slim[col] = slim[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        headers = [str(c) for c in slim.columns]
        rows = slim.astype(str).values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    lines = [
        "# Stage IV Hidden Fixed-Mu Slice Audit",
        "",
        "This is a report-only hidden validation audit. It does not launch exact calculations, Delta-q search, or active-learning iterations.",
        "",
        "## Executive Summary",
        "",
        f"- source_run_id: `{decision.get('source_run_id')}`",
        f"- hidden_slice_status: `{decision.get('hidden_slice_status')}`",
        f"- decision_class: `{decision.get('decision_class')}`",
        f"- hidden_slice_passed: `{decision.get('hidden_slice_passed')}`",
        f"- need_new_exact_calculation: `{decision.get('need_new_exact_calculation')}`",
        f"- recommended_next_action: `{decision.get('recommended_next_action')}`",
        "",
    ]
    if decision.get("reason"):
        lines.extend(["## Reason", "", f"`{decision['reason']}`", ""])
    if not summary.empty:
        lines.extend(["## Iteration Metrics", "", df_to_md(summary.tail(8)), ""])
    if not boundary.empty:
        lines.extend(["## Boundary Metrics", "", df_to_md(boundary.tail(12)), ""])
    lines.extend(
        [
            "## Do-Not-Claim",
            "",
            "- Do not add Stage III hidden validation samples to Stage IV training.",
            "- Do not claim hidden-slice recovery if this report is inconclusive or failed.",
            "- Do not treat KNN proxy maps as a saved neural surrogate checkpoint.",
        ]
    )
    write_text(output_dir / "stageiv_3d_hidden_slice_audit.md", "\n".join(lines) + "\n")

    tex_lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\title{Stage IV Hidden Fixed-$\mu$ Slice Audit}",
        r"\author{Report-only hidden validation}",
        r"\date{\today}",
        r"\maketitle",
        r"\section*{Executive Summary}",
        f"Status: \\texttt{{{latex_escape(str(decision.get('hidden_slice_status')))}}}. "
        f"Decision: \\texttt{{{latex_escape(str(decision.get('decision_class')))}}}. "
        f"Hidden slice passed: \\texttt{{{latex_escape(str(decision.get('hidden_slice_passed')))}}}.",
        r"\section*{Figures}",
    ]
    for fig in [
        "hidden_slice_map_change.png",
        "hidden_slice_boundary_shift.png",
        "hidden_slice_phase_comparison.png",
        "hidden_slice_topology_comparison.png",
    ]:
        if (output_dir / "figures" / fig).exists():
            tex_lines.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.92\linewidth]{{figures/{fig}}}",
                    rf"\caption{{{latex_escape(fig)}}}",
                    r"\end{figure}",
                ]
            )
    tex_lines.append(r"\end{document}")
    tex_path = output_dir / "stageiv_3d_hidden_slice_audit.tex"
    write_text(tex_path, "\n".join(tex_lines) + "\n")
    pdflatex = shutil.which("pdflatex")
    status = {"status": "skipped", "reason": "pdflatex_not_found"}
    if pdflatex is not None:
        proc = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", tex_path.name],
            cwd=output_dir,
            text=True,
            capture_output=True,
            timeout=120,
        )
        status = {
            "status": "pass" if proc.returncode == 0 else "fail",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    write_json(output_dir / "pdf_build_status.json", status)


def build_audit(
    run_dir: Path,
    output_dir: Path,
    config: Path | None,
    reference_dataset: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config)
    if args.mu_reference is not None:
        cfg = replace(cfg, mu_reference=float(args.mu_reference))
    ref = load_reference(reference_dataset, cfg)
    dataset_paths = sorted_dataset_paths(run_dir)
    datasets = [load_npz_dataset(p, cfg) for p in dataset_paths]
    selected = datasets[-args.last_n :] if datasets else []
    evaluations: list[dict[str, Any]] = []
    if ref is not None:
        for ds in selected:
            evaluations.append(
                evaluate_dataset(
                    ds,
                    ref,
                    cfg,
                    int(args.grid_n),
                    int(args.knn_k),
                    float(args.support_radius),
                    int(args.component_min_size),
                )
            )
    summary, boundary, components = flatten_metrics(evaluations)
    summary.to_csv(tables_dir / "hidden_slice_metrics_by_iteration.csv", index=False)
    boundary.to_csv(tables_dir / "hidden_slice_boundary_metrics.csv", index=False)
    components.to_csv(tables_dir / "hidden_slice_component_overlap.csv", index=False)
    provenance = pd.DataFrame(
        [
            {
                "role": "stageiv_dataset",
                "path": str(ds["path"]),
                "iteration": ds["iteration"],
                "sha256": ds["sha256"],
                "sample_count": int(ds["x"].shape[0]),
            }
            for ds in datasets
        ]
        + (
            [
                {
                    "role": "stageiii_reference",
                    "path": str(ref["path"]) if ref is not None else "",
                    "iteration": -1,
                    "sha256": ref["sha256"] if ref is not None else "",
                    "sample_count": int(ref["x"].shape[0]) if ref is not None else 0,
                }
            ]
            if reference_dataset is not None
            else []
        )
    )
    provenance.to_csv(tables_dir / "hidden_slice_provenance.csv", index=False)
    decision = build_decision(reference_dataset if ref is not None else None, datasets, evaluations, args)
    decision.update(
        {
            "run_dir": str(run_dir),
            "reference_dataset": str(reference_dataset) if reference_dataset is not None else None,
            "config": str(config) if config is not None else None,
            "mu_reference": float(cfg.mu_reference),
            "grid_n": int(args.grid_n),
            "knn_k": int(args.knn_k),
            "support_radius": float(args.support_radius),
            "component_min_size": int(args.component_min_size),
            "evaluated_iterations": [int(ev["iteration"]) for ev in evaluations],
        }
    )
    write_json(output_dir / "stageiv_3d_hidden_slice_decision.json", decision)
    write_json(
        output_dir / "stageiv_3d_hidden_slice_config.json",
        {
            "stageiv_config": cfg.to_dict(),
            "audit": {
                "grid_n": int(args.grid_n),
                "knn_k": int(args.knn_k),
                "support_radius": float(args.support_radius),
                "last_n": int(args.last_n),
                "component_min_size": int(args.component_min_size),
            },
        },
    )
    make_figures(output_dir, evaluations, summary, boundary)
    write_reports(output_dir, decision, summary, boundary)
    return decision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a report-only Stage IV hidden fixed-mu slice audit.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--reference-dataset", type=Path, default=None)
    p.add_argument("--source-run-id", default="active_phase_topology_3d_t_ja_mu_from_scratch_v1")
    p.add_argument("--mu-reference", type=float, default=None)
    p.add_argument("--grid-n", type=int, default=201)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--support-radius", type=float, default=0.075)
    p.add_argument("--last-n", type=int, default=5)
    p.add_argument("--component-min-size", type=int, default=24)
    p.add_argument("--phase-map-tol", type=float, default=0.01)
    p.add_argument("--topology-map-tol", type=float, default=0.01)
    p.add_argument("--boundary-shift-tol", type=float, default=0.004167)
    p.add_argument("--coverage-tol", type=float, default=0.00625)
    p.add_argument("--topology-overlap-min", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    decision = build_audit(args.run_dir, args.output_dir, args.config, args.reference_dataset, args)
    print(json.dumps(json_clean(decision), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
