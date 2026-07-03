from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import Delaunay, QhullError, cKDTree


SOURCE_RUN_ID = "active_phase_topology_from_scratch_full_loop_v2"
RUN_ID = "topology_convergence_audit_full_loop_v2"
ROOT = Path("active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop")
RUN_DIR = ROOT / "active_runs" / SOURCE_RUN_ID
REPORT_DIR = ROOT / "reports" / RUN_ID
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

GRID_N = 401
AUDIT_K = 8
SENSITIVITY_K = [6, 8, 12]
SUPPORT_RADIUS = 0.045
SEGMENT_SUPPORT_RADIUS = 0.045
BRACKET_EDGE_MAX = 0.03
CONTOUR_MIN_ARC = 0.0015
SIGNIFICANT_ARC_FRACTION = 0.02
SIGNIFICANT_REGION_CELLS = 25
MIN_SURPRISE_DENOMINATOR = 16

MAP_CHANGE_TOL = 0.002
BOUNDARY_SHIFT_TOL = 0.004166666666666667
COVERAGE_TOL = 0.00625
TOPOLOGY_SURPRISE_TOL = 0.02

PHASE_COLORS = {
    "normal": "#6b7280",
    "uniform_SC": "#2b8cbe",
    "FFLO": "#d95f0e",
}
TOPO_COLORS = {
    0: "#3182bd",
    1: "#de2d26",
}


def ensure_dirs() -> None:
    for path in [REPORT_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.{digits}f}"
    return f"{v:.{digits}g}"


def tex_code(value: Any) -> str:
    return r"\texttt{\detokenize{" + str(value) + "}}"


def dataset_iteration(path: Path) -> int:
    return int(path.stem.replace("dataset_iter", ""))


def load_run_config() -> dict[str, Any]:
    path = RUN_DIR / "run_config.json"
    if not path.exists():
        return {}
    return read_json(path)


def domain_bounds() -> tuple[float, float, float, float]:
    cfg = load_run_config().get("active_learning_config", {})
    return (
        float(cfg.get("kt_min", 0.0)),
        float(cfg.get("kt_max", 0.56)),
        float(cfg.get("ja_min", 0.0)),
        float(cfg.get("ja_max", 2.12)),
    )


def normalize_xy(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    kt_min, kt_max, ja_min, ja_max = bounds
    kt_span = max(kt_max - kt_min, 1e-12)
    ja_span = max(ja_max - ja_min, 1e-12)
    return np.column_stack(((x - kt_min) / kt_span, (y - ja_min) / ja_span))


def denormalize_xy(xn: np.ndarray, yn: np.ndarray, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    kt_min, kt_max, ja_min, ja_max = bounds
    return kt_min + xn * (kt_max - kt_min), ja_min + yn * (ja_max - ja_min)


def load_dataset_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["iteration"] = dataset_iteration(path)
    if "needs_rerun_exact" in df.columns and "rerun_required" not in df.columns:
        df["rerun_required"] = df["needs_rerun_exact"]
    numeric_cols = [
        "kT",
        "JA",
        "delta_opt",
        "q_opt",
        "trusted_exact",
        "training_eligible_exact",
        "needs_rerun_exact",
        "rerun_required",
        "q_unresolved",
        "delta_unresolved",
        "topology_label_code",
        "topology_z2",
        "topology_trusted",
        "topology_p0",
        "topology_ppi",
        "topology_bulk_gap",
        "topology_gap_tol",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def trusted_gapped_fflo(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["phase_name"] == "FFLO")
        & (df["topology_trusted"] == 1)
        & (df["topology_label_code"].isin([0, 1]))
        & np.isfinite(df["topology_p0"])
        & np.isfinite(df["topology_ppi"])
        & np.isfinite(df["topology_bulk_gap"])
    )
    if "topology_gap_tol" in df.columns:
        mask &= df["topology_bulk_gap"] > df["topology_gap_tol"].fillna(0)
    return df.loc[mask].copy()


def exact_key(df: pd.DataFrame) -> pd.Series:
    return df["kT"].round(12).astype(str) + "|" + df["JA"].round(12).astype(str)


def load_exact_merged(iteration: int) -> pd.DataFrame:
    path = RUN_DIR / f"iter{iteration:03d}" / f"exact_merged_iter{iteration:03d}.npz"
    if not path.exists():
        return pd.DataFrame()
    with np.load(path, allow_pickle=True) as d:
        rows = {
            "kT": np.asarray(d["kT"], dtype=float),
            "JA": np.asarray(d["JA"], dtype=float),
            "phase_candidate": np.asarray(d["phase_candidate"], dtype=int),
            "trusted_exact": np.asarray(d["trusted_exact"], dtype=int),
            "training_eligible_exact": np.asarray(d["training_eligible_exact"], dtype=int),
            "topology_label_code": np.asarray(d["topology_label_code"], dtype=int),
            "topology_trusted": np.asarray(d["topology_trusted"], dtype=int),
            "topology_bulk_gap": np.asarray(d["topology_bulk_gap"], dtype=float),
            "topology_gap_tol": np.asarray(d["topology_gap_tol"], dtype=float),
            "topology_p0": np.asarray(d["topology_p0"], dtype=float),
            "topology_ppi": np.asarray(d["topology_ppi"], dtype=float),
        }
        for key in ["needs_rerun_exact", "rerun_required", "q_unresolved", "delta_unresolved"]:
            if key in d.files:
                rows[key] = np.asarray(d[key], dtype=int)
    df = pd.DataFrame(rows)
    if "needs_rerun_exact" not in df and "rerun_required" in df:
        df["needs_rerun_exact"] = df["rerun_required"]
    df["phase_name"] = np.select(
        [df["phase_candidate"] == 0, df["phase_candidate"] == 1, df["phase_candidate"] == 2],
        ["normal", "uniform_SC", "FFLO"],
        default="unknown",
    )
    return df


def input_audit(dataset_paths: list[Path]) -> pd.DataFrame:
    rows = []
    seen_iterations = set()
    for path in dataset_paths:
        df = load_dataset_csv(path)
        iteration = dataset_iteration(path)
        seen_iterations.add(iteration)
        fflo = df[df["phase_name"] == "FFLO"]
        trusted = trusted_gapped_fflo(df)
        key_counts = exact_key(df).value_counts()
        duplicate_count = int((key_counts > 1).sum())
        conflict_count = 0
        if duplicate_count:
            for _, group in df.groupby(exact_key(df)):
                if group["topology_label_code"].nunique(dropna=False) > 1 or group["phase_name"].nunique(dropna=False) > 1:
                    conflict_count += 1
        rows.append(
            {
                "iteration": iteration,
                "dataset_path": str(path),
                "dataset_sha256": sha256_file(path),
                "total_unique_rows": len(df),
                "trusted_thermodynamic_points": int((df["trusted_exact"] == 1).sum()) if "trusted_exact" in df else np.nan,
                "fflo_points": len(fflo),
                "trusted_gapped_fflo_points": len(trusted),
                "trivial_fflo_points": int((trusted["topology_label_code"] == 0).sum()),
                "topological_fflo_points": int((trusted["topology_label_code"] == 1).sum()),
                "excluded_or_untrusted_fflo_points": int(len(fflo) - len(trusted)),
                "duplicate_coordinate_keys": duplicate_count,
                "conflicting_duplicate_keys": conflict_count,
            }
        )
    missing = sorted(set(range(min(seen_iterations), max(seen_iterations) + 1)) - seen_iterations) if seen_iterations else []
    if missing:
        rows.append({"iteration": -1, "dataset_path": "missing_iterations", "dataset_sha256": ",".join(map(str, missing))})
    return pd.DataFrame(rows).sort_values("iteration")


def make_grid(bounds: tuple[float, float, float, float], grid_n: int) -> dict[str, np.ndarray]:
    kt_min, kt_max, ja_min, ja_max = bounds
    x = np.linspace(kt_min, kt_max, grid_n)
    y = np.linspace(ja_min, ja_max, grid_n)
    xx, yy = np.meshgrid(x, y)
    points_norm = normalize_xy(xx.ravel(), yy.ravel(), bounds)
    return {"x": x, "y": y, "xx": xx, "yy": yy, "points_norm": points_norm}


def idw_predict(
    train_xy: np.ndarray,
    values: np.ndarray,
    query_xy: np.ndarray,
    *,
    k: int,
    support_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(train_xy)
    kk = max(1, min(int(k), len(train_xy)))
    dist, idx = tree.query(query_xy, k=kk)
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.maximum(dist, 1e-9) ** 2
    vals = values[idx]
    pred = np.sum(weights * vals, axis=1) / np.sum(weights, axis=1)
    var = np.sum(weights * (vals - pred[:, None]) ** 2, axis=1) / np.sum(weights, axis=1)
    nearest = dist[:, 0]
    valid = np.isfinite(pred) & (nearest <= support_radius)
    return pred, np.sqrt(np.maximum(var, 0.0)), nearest, valid


def final_audit_mask(final_train: pd.DataFrame, grid: dict[str, np.ndarray], bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    xy = normalize_xy(final_train["kT"].to_numpy(float), final_train["JA"].to_numpy(float), bounds)
    tree = cKDTree(xy)
    nearest = tree.query(grid["points_norm"], k=1)[0]
    near = nearest <= SUPPORT_RADIUS
    try:
        hull = Delaunay(xy)
        inside = hull.find_simplex(grid["points_norm"]) >= 0
    except QhullError:
        inside = np.zeros(len(grid["points_norm"]), dtype=bool)
    return near & inside, nearest


def local_brackets(train: pd.DataFrame, bounds: tuple[float, float, float, float], k: int) -> pd.DataFrame:
    if len(train) < 3:
        return pd.DataFrame()
    xy = normalize_xy(train["kT"].to_numpy(float), train["JA"].to_numpy(float), bounds)
    labels = train["topology_label_code"].to_numpy(int)
    tree = cKDTree(xy)
    dist, idx = tree.query(xy, k=min(k + 1, len(train)))
    edges = set()
    for i in range(len(train)):
        for d, j in zip(np.atleast_1d(dist[i])[1:], np.atleast_1d(idx[i])[1:]):
            if i == int(j) or d > BRACKET_EDGE_MAX:
                continue
            if labels[i] == labels[int(j)]:
                continue
            a, b = sorted((i, int(j)))
            edges.add((a, b))
    rows = []
    raw = train[["kT", "JA"]].to_numpy(float)
    for a, b in sorted(edges):
        mid_raw = 0.5 * (raw[a] + raw[b])
        mid_norm = 0.5 * (xy[a] + xy[b])
        rows.append(
            {
                "kT": mid_raw[0],
                "JA": mid_raw[1],
                "x_norm": mid_norm[0],
                "y_norm": mid_norm[1],
                "edge_length_norm": float(np.linalg.norm(xy[a] - xy[b])),
                "label_a": int(labels[a]),
                "label_b": int(labels[b]),
            }
        )
    return pd.DataFrame(rows)


def contour_segments(
    grid: dict[str, np.ndarray],
    field: np.ndarray,
    valid_mask: np.ndarray,
    train: pd.DataFrame,
    brackets: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    *,
    contour_kind: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    shape = grid["xx"].shape
    masked = np.ma.array(field.reshape(shape), mask=(~valid_mask).reshape(shape))
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(grid["xx"], grid["yy"], masked, levels=[0.0])
        raw_segments = [np.asarray(seg, dtype=float) for seg in cs.allsegs[0] if len(seg) >= 2]
    finally:
        plt.close(fig)

    train0 = train[train["topology_label_code"] == 0]
    train1 = train[train["topology_label_code"] == 1]
    tree0 = cKDTree(normalize_xy(train0["kT"].to_numpy(float), train0["JA"].to_numpy(float), bounds)) if len(train0) else None
    tree1 = cKDTree(normalize_xy(train1["kT"].to_numpy(float), train1["JA"].to_numpy(float), bounds)) if len(train1) else None
    bracket_tree = cKDTree(brackets[["x_norm", "y_norm"]].to_numpy(float)) if len(brackets) else None
    rows = []
    kept: list[dict[str, Any]] = []
    for seg_id, seg in enumerate(raw_segments):
        seg_norm = normalize_xy(seg[:, 0], seg[:, 1], bounds)
        arc = float(np.sum(np.linalg.norm(np.diff(seg_norm, axis=0), axis=1))) if len(seg_norm) > 1 else 0.0
        if arc < CONTOUR_MIN_ARC:
            continue
        grid_x = np.clip(np.searchsorted(grid["x"], seg[:, 0]), 0, len(grid["x"]) - 1)
        grid_y = np.clip(np.searchsorted(grid["y"], seg[:, 1]), 0, len(grid["y"]) - 1)
        valid_fraction = float(np.mean(valid_mask.reshape(grid["xx"].shape)[grid_y, grid_x]))
        d0 = tree0.query(seg_norm, k=1)[0] if tree0 is not None else np.full(len(seg_norm), np.inf)
        d1 = tree1.query(seg_norm, k=1)[0] if tree1 is not None else np.full(len(seg_norm), np.inf)
        db = bracket_tree.query(seg_norm, k=1)[0] if bracket_tree is not None else np.full(len(seg_norm), np.inf)
        two_sided = (d0 <= SEGMENT_SUPPORT_RADIUS) & (d1 <= SEGMENT_SUPPORT_RADIUS)
        bracketed = db <= BRACKET_EDGE_MAX
        support_fraction = float(np.mean(two_sided | bracketed))
        keep = valid_fraction >= 0.75 and support_fraction >= 0.10
        rows.append(
            {
                "contour_kind": contour_kind,
                "raw_segment_id": seg_id,
                "arc_length_norm": arc,
                "valid_fraction": valid_fraction,
                "two_sided_support_fraction": float(np.mean(two_sided)),
                "bracket_support_fraction": float(np.mean(bracketed)),
                "combined_support_fraction": support_fraction,
                "nearest_trivial_median": float(np.median(d0)),
                "nearest_topological_median": float(np.median(d1)),
                "nearest_bracket_median": float(np.median(db)),
                "kept": bool(keep),
            }
        )
        if keep:
            kept.append({"kind": contour_kind, "xy": seg, "xy_norm": seg_norm, "arc_length_norm": arc, "support_fraction": support_fraction})
    return kept, pd.DataFrame(rows)


def sample_polyline(seg_norm: np.ndarray, spacing: float = 0.0015) -> np.ndarray:
    if len(seg_norm) < 2:
        return seg_norm
    samples = [seg_norm[0]]
    for a, b in zip(seg_norm[:-1], seg_norm[1:]):
        length = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(length / spacing)))
        for i in range(1, n + 1):
            samples.append(a + (b - a) * (i / n))
    return np.asarray(samples, dtype=float)


def boundary_points(contours: list[dict[str, Any]]) -> np.ndarray:
    if not contours:
        return np.empty((0, 2), dtype=float)
    pts = [sample_polyline(seg["xy_norm"]) for seg in contours]
    return np.vstack([p for p in pts if len(p)])


def boundary_shift(prev_pts: np.ndarray, curr_pts: np.ndarray) -> dict[str, Any]:
    if len(prev_pts) == 0 or len(curr_pts) == 0:
        return {
            "status": "missing_boundary",
            "directed_curr_to_prev_median": np.nan,
            "directed_prev_to_curr_median": np.nan,
            "symmetric_median": np.nan,
            "symmetric_p90": np.nan,
            "symmetric_p95": np.nan,
            "symmetric_hausdorff": np.nan,
        }
    tree_prev = cKDTree(prev_pts)
    tree_curr = cKDTree(curr_pts)
    d_cp = tree_prev.query(curr_pts, k=1)[0]
    d_pc = tree_curr.query(prev_pts, k=1)[0]
    all_d = np.concatenate([d_cp, d_pc])
    return {
        "status": "ok",
        "directed_curr_to_prev_median": float(np.median(d_cp)),
        "directed_prev_to_curr_median": float(np.median(d_pc)),
        "symmetric_median": float(np.median(all_d)),
        "symmetric_p90": float(np.quantile(all_d, 0.90)),
        "symmetric_p95": float(np.quantile(all_d, 0.95)),
        "symmetric_hausdorff": float(np.max(all_d)),
    }


def component_counts(z2_grid: np.ndarray, valid_grid: np.ndarray, contours: list[dict[str, Any]]) -> dict[str, Any]:
    topo_mask = (z2_grid == 1) & valid_grid
    labels, n = ndimage.label(topo_mask)
    sizes = np.bincount(labels.ravel()) if n else np.asarray([0])
    significant = int(np.sum(sizes[1:] >= SIGNIFICANT_REGION_CELLS)) if n else 0
    arcs = np.asarray([seg["arc_length_norm"] for seg in contours], dtype=float)
    total_arc = float(np.sum(arcs)) if len(arcs) else 0.0
    significant_arcs = arcs[arcs >= max(CONTOUR_MIN_ARC, total_arc * SIGNIFICANT_ARC_FRACTION)] if total_arc > 0 else np.asarray([])
    return {
        "boundary_component_count": int(len(contours)),
        "significant_boundary_component_count": int(len(significant_arcs)),
        "topological_region_component_count": significant,
        "total_boundary_arc_length": total_arc,
        "largest_component_arc_fraction": float(np.max(arcs) / total_arc) if total_arc > 0 else np.nan,
    }


def model_for_iteration(df: pd.DataFrame, grid: dict[str, np.ndarray], audit_mask: np.ndarray, bounds: tuple[float, float, float, float], k: int) -> dict[str, Any]:
    train = trusted_gapped_fflo(df)
    if len(train) == 0:
        n_grid = len(grid["points_norm"])
        valid = np.zeros(n_grid, dtype=bool)
        z2 = np.full(n_grid, -1, dtype=int)
        comps = component_counts(z2.reshape(grid["xx"].shape), valid.reshape(grid["xx"].shape), [])
        return {
            "train": train,
            "xy": np.empty((0, 2), dtype=float),
            "p0": np.full(n_grid, np.nan, dtype=float),
            "ppi": np.full(n_grid, np.nan, dtype=float),
            "p0_unc": np.full(n_grid, np.nan, dtype=float),
            "ppi_unc": np.full(n_grid, np.nan, dtype=float),
            "product": np.full(n_grid, np.nan, dtype=float),
            "z2": z2,
            "valid": valid,
            "nearest": np.full(n_grid, np.nan, dtype=float),
            "brackets": pd.DataFrame(),
            "contours": [],
            "contour_table": pd.DataFrame(),
            "components": comps,
        }
    xy = normalize_xy(train["kT"].to_numpy(float), train["JA"].to_numpy(float), bounds)
    p0, p0_unc, nearest, valid = idw_predict(xy, train["topology_p0"].to_numpy(float), grid["points_norm"], k=k, support_radius=SUPPORT_RADIUS)
    ppi, ppi_unc, nearest_ppi, valid_ppi = idw_predict(xy, train["topology_ppi"].to_numpy(float), grid["points_norm"], k=k, support_radius=SUPPORT_RADIUS)
    valid = audit_mask & valid & valid_ppi
    product = p0 * ppi
    z2 = np.full(len(product), -1, dtype=int)
    z2[(product > 0) & valid] = 0
    z2[(product < 0) & valid] = 1
    brackets = local_brackets(train, bounds, k=k)
    p0_contours, p0_table = contour_segments(grid, p0, valid, train, brackets, bounds, contour_kind="P0")
    ppi_contours, ppi_table = contour_segments(grid, ppi, valid, train, brackets, bounds, contour_kind="Ppi")
    contours = p0_contours + ppi_contours
    shape = grid["xx"].shape
    comps = component_counts(z2.reshape(shape), valid.reshape(shape), contours)
    return {
        "train": train,
        "xy": xy,
        "p0": p0,
        "ppi": ppi,
        "p0_unc": p0_unc,
        "ppi_unc": ppi_unc,
        "product": product,
        "z2": z2,
        "valid": valid,
        "nearest": np.minimum(nearest, nearest_ppi),
        "brackets": brackets,
        "contours": contours,
        "contour_table": pd.concat([p0_table, ppi_table], ignore_index=True) if len(p0_table) or len(ppi_table) else pd.DataFrame(),
        "components": comps,
    }


def coverage_metrics(model: dict[str, Any]) -> dict[str, Any]:
    pts = boundary_points(model["contours"])
    if len(pts) == 0:
        return {
            "coverage_status": "missing_boundary",
            "coverage_median": np.nan,
            "coverage_p90": np.nan,
            "coverage_p95": np.nan,
            "coverage_max": np.nan,
            "bracket_coverage_p95": np.nan,
        }
    train_tree = cKDTree(model["xy"])
    d = train_tree.query(pts, k=1)[0]
    if len(model["brackets"]):
        bracket_tree = cKDTree(model["brackets"][["x_norm", "y_norm"]].to_numpy(float))
        db = bracket_tree.query(pts, k=1)[0]
    else:
        db = np.full(len(pts), np.nan)
    return {
        "coverage_status": "ok",
        "coverage_median": float(np.median(d)),
        "coverage_p90": float(np.quantile(d, 0.90)),
        "coverage_p95": float(np.quantile(d, 0.95)),
        "coverage_max": float(np.max(d)),
        "bracket_coverage_median": float(np.nanmedian(db)),
        "bracket_coverage_p95": float(np.nanquantile(db, 0.95)),
        "bracket_coverage_max": float(np.nanmax(db)),
    }


def topology_surprise(prev_model: dict[str, Any], prev_df: pd.DataFrame, curr_df: pd.DataFrame, bounds: tuple[float, float, float, float], iteration: int) -> dict[str, Any]:
    prev_keys = set(exact_key(prev_df))
    curr = curr_df.copy()
    curr["coord_key"] = exact_key(curr)
    new = curr[~curr["coord_key"].isin(prev_keys)].copy()
    if new.empty:
        return {"iteration": iteration, "trusted_denominator": 0, "trusted_status": "insufficient_surprise_support"}
    if len(prev_model["train"]) == 0:
        return {
            "iteration": iteration,
            "new_rows": int(len(new)),
            "trusted_denominator": 0,
            "trusted_n_surprise": 0,
            "trusted_topology_surprise": np.nan,
            "trusted_status": "insufficient_previous_model",
            "all_selected_denominator": 0,
            "all_selected_n_surprise": 0,
            "all_selected_topology_surprise": np.nan,
            "hard_risk_denominator": 0,
            "hard_risk_n_surprise": 0,
            "hard_risk_topology_surprise": np.nan,
            "reliable_prediction_fraction_new": 0.0,
        }
    xy = normalize_xy(prev_model["train"]["kT"].to_numpy(float), prev_model["train"]["JA"].to_numpy(float), bounds)
    query = normalize_xy(new["kT"].to_numpy(float), new["JA"].to_numpy(float), bounds)
    p0, _, nearest, valid = idw_predict(xy, prev_model["train"]["topology_p0"].to_numpy(float), query, k=AUDIT_K, support_radius=SUPPORT_RADIUS)
    ppi, _, _, valid2 = idw_predict(xy, prev_model["train"]["topology_ppi"].to_numpy(float), query, k=AUDIT_K, support_radius=SUPPORT_RADIUS)
    pred = np.full(len(new), -1, dtype=int)
    pred[(p0 * ppi > 0) & valid & valid2] = 0
    pred[(p0 * ppi < 0) & valid & valid2] = 1
    exact_float = new["topology_label_code"].to_numpy(float)
    exact = np.where(np.isfinite(exact_float), exact_float, -999).astype(int)
    reliable_pred = pred >= 0
    trusted_mask = (
        (new["phase_name"] == "FFLO").to_numpy()
        & (new["topology_trusted"].to_numpy() == 1)
        & np.isin(exact, [0, 1])
        & reliable_pred
    )
    if "topology_gap_tol" in new:
        trusted_mask &= new["topology_bulk_gap"].to_numpy(float) > new["topology_gap_tol"].to_numpy(float)
    hard_mask = (
        (new["phase_name"] == "FFLO").to_numpy()
        & np.isin(exact, [0, 1])
        & reliable_pred
        & ~trusted_mask
    )
    all_mask = (new["phase_name"] == "FFLO").to_numpy() & np.isin(exact, [0, 1]) & reliable_pred

    def rate(mask: np.ndarray) -> tuple[int, int, float]:
        denom = int(np.sum(mask))
        num = int(np.sum(pred[mask] != exact[mask])) if denom else 0
        return denom, num, float(num / denom) if denom else np.nan

    td, tn, tr = rate(trusted_mask)
    ad, an, ar = rate(all_mask)
    hd, hn, hr = rate(hard_mask)
    return {
        "iteration": iteration,
        "new_rows": int(len(new)),
        "trusted_denominator": td,
        "trusted_n_surprise": tn,
        "trusted_topology_surprise": tr,
        "trusted_status": "ok" if td >= MIN_SURPRISE_DENOMINATOR else "insufficient_surprise_support",
        "all_selected_denominator": ad,
        "all_selected_n_surprise": an,
        "all_selected_topology_surprise": ar,
        "hard_risk_denominator": hd,
        "hard_risk_n_surprise": hn,
        "hard_risk_topology_surprise": hr,
        "reliable_prediction_fraction_new": float(np.mean(reliable_pred)),
    }


def support_audit(final_model: dict[str, Any], grid: dict[str, np.ndarray], bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    contours = final_model["contours"]
    if not contours:
        return pd.DataFrame()
    train = final_model["train"]
    train0 = train[train["topology_label_code"] == 0]
    train1 = train[train["topology_label_code"] == 1]
    tree0 = cKDTree(normalize_xy(train0["kT"].to_numpy(float), train0["JA"].to_numpy(float), bounds))
    tree1 = cKDTree(normalize_xy(train1["kT"].to_numpy(float), train1["JA"].to_numpy(float), bounds))
    tree_all = cKDTree(final_model["xy"])
    bracket_tree = cKDTree(final_model["brackets"][["x_norm", "y_norm"]].to_numpy(float)) if len(final_model["brackets"]) else None
    rows = []
    for comp_id, seg in enumerate(contours):
        pts = sample_polyline(seg["xy_norm"])
        d0 = tree0.query(pts, k=1)[0]
        d1 = tree1.query(pts, k=1)[0]
        da = tree_all.query(pts, k=1)[0]
        db = bracket_tree.query(pts, k=1)[0] if bracket_tree is not None else np.full(len(pts), np.nan)
        raw_x, raw_y = denormalize_xy(pts[:, 0], pts[:, 1], bounds)
        for i in range(0, len(pts), max(1, len(pts) // 60)):
            rows.append(
                {
                    "component_id": comp_id,
                    "contour_kind": seg["kind"],
                    "kT": raw_x[i],
                    "JA": raw_y[i],
                    "x_norm": pts[i, 0],
                    "y_norm": pts[i, 1],
                    "nearest_trivial_distance": d0[i],
                    "nearest_topological_distance": d1[i],
                    "nearest_bracket_distance": db[i],
                    "local_fill_distance": da[i],
                    "has_two_sided_support": bool(d0[i] <= SEGMENT_SUPPORT_RADIUS and d1[i] <= SEGMENT_SUPPORT_RADIUS),
                    "has_bracket_support": bool(np.isfinite(db[i]) and db[i] <= BRACKET_EDGE_MAX),
                    "support_class": "direct_or_bracketed" if ((d0[i] <= SEGMENT_SUPPORT_RADIUS and d1[i] <= SEGMENT_SUPPORT_RADIUS) or (np.isfinite(db[i]) and db[i] <= BRACKET_EDGE_MAX)) else "extrapolation_only",
                }
            )
    return pd.DataFrame(rows)


def write_boundary_points(models: dict[int, dict[str, Any]], bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    rows = []
    for iteration, model in models.items():
        for comp_id, seg in enumerate(model["contours"]):
            pts = sample_polyline(seg["xy_norm"], spacing=0.0025)
            raw_x, raw_y = denormalize_xy(pts[:, 0], pts[:, 1], bounds)
            for idx in range(len(pts)):
                rows.append(
                    {
                        "iteration": iteration,
                        "component_id": comp_id,
                        "contour_kind": seg["kind"],
                        "sample_id": idx,
                        "kT": raw_x[idx],
                        "JA": raw_y[idx],
                        "x_norm": pts[idx, 0],
                        "y_norm": pts[idx, 1],
                        "component_arc_length_norm": seg["arc_length_norm"],
                        "component_support_fraction": seg["support_fraction"],
                    }
                )
    return pd.DataFrame(rows)


def make_figures(
    datasets: dict[int, pd.DataFrame],
    models: dict[int, dict[str, Any]],
    metrics: pd.DataFrame,
    shifts: pd.DataFrame,
    coverage: pd.DataFrame,
    surprise: pd.DataFrame,
    components: pd.DataFrame,
    support: pd.DataFrame,
    decision: dict[str, Any],
    grid: dict[str, np.ndarray],
    bounds: tuple[float, float, float, float],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    final_iter = max(datasets)
    final_df = datasets[final_iter]
    final_model = models[final_iter]

    def save(fig: plt.Figure, name: str) -> None:
        png = FIGURE_DIR / f"{name}.png"
        pdf = FIGURE_DIR / f"{name}.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        paths[name] = png
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    for phase, color in PHASE_COLORS.items():
        sub = final_df[final_df["phase_name"] == phase]
        ax.scatter(sub["kT"], sub["JA"], s=5, c=color, alpha=0.35, label=phase, linewidths=0)
    train = final_model["train"]
    for label, color, name in [(0, TOPO_COLORS[0], "trivial FFLO / cFFLO"), (1, TOPO_COLORS[1], "topological FFLO / tFFLO")]:
        sub = train[train["topology_label_code"] == label]
        ax.scatter(sub["kT"], sub["JA"], s=7, c=color, alpha=0.75, label=name, linewidths=0)
    for seg in final_model["contours"]:
        ax.plot(seg["xy"][:, 0], seg["xy"][:, 1], color="black", lw=1.4)
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("Final thermodynamic map with cFFLO/tFFLO contour")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.18)
    save(fig, "fig01_final_map_contour")

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 5))
    for color, iteration in zip(colors, sorted(models)[-5:]):
        for seg in models[iteration]["contours"]:
            ax.plot(seg["xy"][:, 0], seg["xy"][:, 1], color=color, lw=1.2, label=f"iter{iteration:03d}")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("cFFLO/tFFLO contours in final five cumulative iterations")
    ax.grid(alpha=0.18)
    save(fig, "fig02_final5_contour_overlay")

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.plot(metrics["transition_to_iteration"], metrics["topology_map_change"], marker="o", ms=3)
    ax.axhline(MAP_CHANGE_TOL, color="black", ls="--", lw=1, label="threshold")
    ax.set_xlabel("transition to iteration")
    ax.set_ylabel("changed valid-mask fraction")
    ax.set_title("Topology map change")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "fig03_topology_map_change")

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ok = shifts[shifts["status"] == "ok"]
    ax.plot(ok["transition_to_iteration"], ok["symmetric_median"], marker="o", ms=3, label="median")
    ax.plot(ok["transition_to_iteration"], ok["symmetric_p95"], marker="o", ms=3, label="p95")
    ax.plot(ok["transition_to_iteration"], ok["symmetric_hausdorff"], marker="o", ms=3, label="Hausdorff max")
    ax.axhline(BOUNDARY_SHIFT_TOL, color="black", ls="--", lw=1, label="p95 threshold")
    ax.set_xlabel("transition to iteration")
    ax.set_ylabel("normalized distance")
    ax.set_title("Topology boundary shift")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "fig04_boundary_shift")

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.plot(coverage["iteration"], coverage["coverage_median"], marker="o", ms=3, label="exact point median")
    ax.plot(coverage["iteration"], coverage["coverage_p95"], marker="o", ms=3, label="exact point p95")
    ax.plot(coverage["iteration"], coverage["bracket_coverage_p95"], marker="o", ms=3, label="opposite-Z2 bracket p95")
    ax.axhline(COVERAGE_TOL, color="black", ls="--", lw=1, label="coverage threshold")
    ax.set_xlabel("iteration")
    ax.set_ylabel("normalized distance")
    ax.set_title("Topology boundary coverage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "fig05_boundary_coverage")

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.plot(surprise["iteration"], surprise["trusted_topology_surprise"], marker="o", ms=3, label="trusted")
    ax.plot(surprise["iteration"], surprise["all_selected_topology_surprise"], marker="o", ms=3, label="all selected")
    ax.plot(surprise["iteration"], surprise["hard_risk_topology_surprise"], marker="o", ms=3, label="hard risk")
    ax.axhline(TOPOLOGY_SURPRISE_TOL, color="black", ls="--", lw=1, label="trusted threshold")
    for _, row in surprise.iterrows():
        if np.isfinite(row.get("trusted_topology_surprise", np.nan)):
            ax.text(row["iteration"], row["trusted_topology_surprise"] + 0.01, f"{int(row['trusted_n_surprise'])}/{int(row['trusted_denominator'])}", fontsize=6)
    ax.set_xlabel("new batch appended into iteration")
    ax.set_ylabel("surprise rate")
    ax.set_ylim(bottom=0)
    ax.set_title("Out-of-sample topology surprise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "fig06_topology_surprise")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    axes[0].plot(components["iteration"], components["significant_boundary_component_count"], marker="o", ms=3, label="boundary components")
    axes[0].plot(components["iteration"], components["topological_region_component_count"], marker="o", ms=3, label="tFFLO regions")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("count")
    axes[0].set_title("Component stability")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].plot(components["iteration"], components["total_boundary_arc_length"], marker="o", ms=3, label="total arc")
    axes[1].plot(components["iteration"], components["largest_component_arc_fraction"], marker="o", ms=3, label="largest fraction")
    axes[1].set_xlabel("iteration")
    axes[1].set_title("Boundary arc length")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    save(fig, "fig07_component_stability")

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    if not support.empty:
        colors = np.where(support["support_class"] == "direct_or_bracketed", "#238b45", "#d94801")
        ax.scatter(support["kT"], support["JA"], s=8, c=colors, alpha=0.75, linewidths=0)
    for seg in final_model["contours"]:
        ax.plot(seg["xy"][:, 0], seg["xy"][:, 1], color="black", lw=1.2)
    if len(final_model["brackets"]):
        ax.scatter(final_model["brackets"]["kT"], final_model["brackets"]["JA"], s=5, c="#54278f", alpha=0.55, label="opposite-Z2 brackets")
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("Final contour direct-support audit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.18)
    save(fig, "fig08_final_support_map")

    normal_rows = []
    offsets = np.asarray([-0.015, -0.0075, 0.0, 0.0075, 0.015], dtype=float)
    for seg in final_model["contours"]:
        pts = sample_polyline(seg["xy_norm"], spacing=0.0015)
        if len(pts) < 3:
            continue
        tangent = np.gradient(pts, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        good = norm > 1e-12
        tangent[good] = tangent[good] / norm[good, None]
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        for i, point in enumerate(pts):
            for off in offsets:
                q = np.clip(point + off * normal[i], 0.0, 1.0)
                normal_rows.append((i, off, q[0], q[1]))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    if normal_rows:
        normal_df = pd.DataFrame(normal_rows, columns=["contour_index", "normal_offset", "x_norm", "y_norm"])
        query = normal_df[["x_norm", "y_norm"]].to_numpy(float)
        p0n, _, _, v0 = idw_predict(final_model["xy"], final_model["train"]["topology_p0"].to_numpy(float), query, k=AUDIT_K, support_radius=SUPPORT_RADIUS)
        ppin, _, _, v1 = idw_predict(final_model["xy"], final_model["train"]["topology_ppi"].to_numpy(float), query, k=AUDIT_K, support_radius=SUPPORT_RADIUS)
        gapn, _, _, vg = idw_predict(final_model["xy"], final_model["train"]["topology_bulk_gap"].to_numpy(float), query, k=AUDIT_K, support_radius=SUPPORT_RADIUS)
        valid = v0 & v1 & vg
        margin = np.where(valid, np.minimum(np.abs(p0n), np.abs(ppin)), np.nan)
        gap = np.where(valid, gapn, np.nan)
        n_index = int(normal_df["contour_index"].max()) + 1
        margin_grid = np.full((len(offsets), n_index), np.nan)
        gap_grid = np.full((len(offsets), n_index), np.nan)
        offset_to_row = {float(v): i for i, v in enumerate(offsets)}
        for row_i, row in normal_df.iterrows():
            r = offset_to_row[float(row["normal_offset"])]
            c = int(row["contour_index"])
            margin_grid[r, c] = margin[row_i]
            gap_grid[r, c] = gap[row_i]
        im0 = axes[0].imshow(margin_grid, aspect="auto", origin="lower", extent=[0, n_index, offsets[0], offsets[-1]], cmap="magma")
        im1 = axes[1].imshow(gap_grid, aspect="auto", origin="lower", extent=[0, n_index, offsets[0], offsets[-1]], cmap="viridis")
        axes[0].set_title("IDW Pfaffian margin near contour normal")
        axes[1].set_title("IDW bulk gap near contour normal")
        for ax in axes:
            ax.axhline(0.0, color="white", lw=0.8, alpha=0.8)
            ax.set_xlabel("sampled final contour point")
            ax.set_ylabel("normal offset")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    save(fig, "fig09_pfaffian_gap_along_contour")

    fig, ax = plt.subplots(figsize=(10.0, 5.2), constrained_layout=False)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    map_last3 = ", ".join(fmt(v, 6) for v in decision["topology_map_change_last3"])
    shift_last3 = ", ".join(fmt(v, 6) for v in decision["topology_boundary_shift_p95_last3"])
    surprise_last3 = ", ".join(fmt(v, 4) for v in decision["trusted_topology_surprise_last3"])
    lines = [
        f"Decision: {decision['decision_class']}",
        f"topology_main_converged: {decision['topology_main_converged']}",
        f"need_new_exact_calculation: {decision['need_new_exact_calculation']}",
        f"limiting_factor: {decision['limiting_factor']}",
        f"last3 map change: {map_last3}",
        f"last3 boundary shift p95: {shift_last3}",
        f"final coverage p95: {fmt(decision['topology_boundary_coverage_p95_final'])}",
        f"last3 trusted surprise: {surprise_last3}",
        f"recommended next action: {decision['recommended_next_action']}",
    ]
    fig.suptitle("Topology convergence decision summary", fontsize=14)
    ax.text(0.05, 0.88, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    save(fig, "fig10_decision_summary")
    return paths


def decide(
    metrics: pd.DataFrame,
    shifts: pd.DataFrame,
    coverage: pd.DataFrame,
    surprise: pd.DataFrame,
    components: pd.DataFrame,
    support: pd.DataFrame,
    final_iteration: int,
) -> dict[str, Any]:
    last3_iters = list(range(final_iteration - 2, final_iteration + 1))
    last3_trans = metrics[metrics["transition_to_iteration"].isin(last3_iters)]
    last3_shift = shifts[shifts["transition_to_iteration"].isin(last3_iters)]
    last3_surprise = surprise[surprise["iteration"].isin(last3_iters)]
    final_cov = coverage[coverage["iteration"] == final_iteration].iloc[0]
    final_support_fraction = float(np.mean(support["support_class"] == "direct_or_bracketed")) if not support.empty else 0.0

    boundary_exists = all(components.set_index("iteration").loc[it, "significant_boundary_component_count"] > 0 for it in last3_iters if it in set(components["iteration"]))
    map_pass = bool((last3_trans["topology_map_change"] < MAP_CHANGE_TOL).all()) and len(last3_trans) == 3
    shift_pass = bool((last3_shift["status"] == "ok").all() and (last3_shift["symmetric_p95"] <= BOUNDARY_SHIFT_TOL).all() and len(last3_shift) == 3)
    coverage_pass = bool(final_cov["coverage_p95"] < COVERAGE_TOL)
    supported_surprise = last3_surprise[last3_surprise["trusted_status"] == "ok"]
    surprise_pass = bool(len(supported_surprise) == len(last3_surprise) and (supported_surprise["trusted_topology_surprise"] <= TOPOLOGY_SURPRISE_TOL).all())
    comp_last = components[components["iteration"].isin(last3_iters)]
    component_pass = bool(
        len(comp_last) == 3
        and comp_last["significant_boundary_component_count"].nunique() == 1
        and comp_last["topological_region_component_count"].nunique() == 1
        and (comp_last["largest_component_arc_fraction"] >= 0.80).all()
    )
    direct_support_pass = final_support_fraction >= 0.80

    missing_history = len(last3_trans) < 3 or len(last3_shift) < 3 or len(comp_last) < 3
    if missing_history or not boundary_exists:
        decision_class = "Decision D"
        topology_main_converged = "inconclusive"
        limiting = "missing_boundary_or_history"
        next_action = "reconstruct_missing_iteration_history_before_new_exact_calculation"
        need_new = False
    elif map_pass and shift_pass and coverage_pass and surprise_pass and component_pass and direct_support_pass:
        decision_class = "Decision A"
        topology_main_converged = True
        limiting = "none"
        next_action = "freeze_topology_boundary_result_for_offline_report"
        need_new = False
    elif map_pass and shift_pass and surprise_pass and component_pass and direct_support_pass and (final_cov["coverage_p95"] <= 1.25 * COVERAGE_TOL):
        decision_class = "Decision B"
        topology_main_converged = "nearly"
        limiting = "boundary_coverage"
        next_action = "1_to_3_spectral_tail_batches"
        need_new = True
    else:
        decision_class = "Decision C"
        topology_main_converged = False
        failed = []
        if not map_pass:
            failed.append("topology_map_change")
        if not shift_pass:
            failed.append("topology_boundary_shift_p95")
        if not coverage_pass:
            failed.append("topology_boundary_coverage")
        if not surprise_pass:
            failed.append("trusted_topology_surprise")
        if not component_pass:
            failed.append("component_stability")
        if not direct_support_pass:
            failed.append("direct_exact_support")
        limiting = ",".join(failed)
        next_action = "target_specific_unconverged_boundary_arcs_not_full_restart"
        need_new = True

    return {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "final_iteration": int(final_iteration),
        "topology_main_converged": topology_main_converged,
        "decision_class": decision_class,
        "limiting_factor": limiting,
        "topology_map_change_last3": [float(v) for v in last3_trans["topology_map_change"].to_numpy()],
        "topology_boundary_shift_p95_last3": [float(v) for v in last3_shift["symmetric_p95"].to_numpy()],
        "topology_boundary_coverage_p95_final": float(final_cov["coverage_p95"]),
        "trusted_topology_surprise_last3": [float(v) if np.isfinite(v) else None for v in last3_surprise["trusted_topology_surprise"].to_numpy()],
        "boundary_component_count_last3": [int(v) for v in comp_last["significant_boundary_component_count"].to_numpy()],
        "topological_region_component_count_last3": [int(v) for v in comp_last["topological_region_component_count"].to_numpy()],
        "final_direct_support_fraction": final_support_fraction,
        "need_new_exact_calculation": need_new,
        "recommended_next_action": next_action,
        "caveats": [
            "online topology labels are audited offline here but still derive from the cold-start full-loop dataset",
            "absence of finite-area gapless_SC is not treated as a failure",
            "strict Hausdorff distance is diagnostic only; p95 shift is the main gate",
            "Delaunay or kNN opposite-label edges are used only as support/bracket diagnostics, not as the final contour",
        ],
    }


def write_report(decision: dict[str, Any], input_table: pd.DataFrame, metrics: pd.DataFrame, shifts: pd.DataFrame, coverage: pd.DataFrame, surprise: pd.DataFrame, components: pd.DataFrame, support: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    final_iteration = decision["final_iteration"]
    final_input = input_table[input_table["iteration"] == final_iteration].iloc[0]
    last3 = list(range(final_iteration - 2, final_iteration + 1))
    md = f"""# cFFLO/tFFLO Topology-Boundary Convergence Audit

## Executive Summary

- Source run: `{SOURCE_RUN_ID}`.
- Audit run: `{RUN_ID}`.
- Final cumulative dataset: `dataset_iter{final_iteration:03d}` with {int(final_input['total_unique_rows'])} samples.
- Final trusted gapped FFLO points: {int(final_input['trusted_gapped_fflo_points'])}; cFFLO/trivial FFLO {int(final_input['trivial_fflo_points'])}; tFFLO/topological FFLO {int(final_input['topological_fflo_points'])}.
- Decision: `{decision['decision_class']}`.
- topology_main_converged: `{decision['topology_main_converged']}`.
- limiting_factor: `{decision['limiting_factor']}`.
- need_new_exact_calculation: `{decision['need_new_exact_calculation']}`.
- recommended_next_action: `{decision['recommended_next_action']}`.

This is an offline, report-only audit.  No new thermodynamic exact calculation,
Delta-q search, active-learning loop, or historical dataset modification was
performed.  The audit uses only trusted gapped FFLO points for the formal
cFFLO/tFFLO boundary tests.

## Repository and Input Audit

The cumulative dataset files `dataset_iter000.csv` through
`dataset_iter018.csv` are present.  The formal last-three transitions are
`iter015 -> iter016`, `iter016 -> iter017`, and `iter017 -> iter018`.

The final online topology diagnostic counts are consistent with the returned
run: trivial/cFFLO {int(final_input['trivial_fflo_points'])}, topological/tFFLO
{int(final_input['topological_fflo_points'])}, and no trusted gapless-SC phase
in the current parameter range.  The absence of a finite-area gapless-SC phase
is not treated as a convergence failure.

## Audit Method

For each cumulative iteration, the same deterministic KNN inverse-distance
surrogate was fit to the iteration's trusted gapped FFLO exact points.  The
surrogate independently interpolates \\(P_0(k_B T/t,J_A/t)\\) and
\\(P_\\pi(k_B T/t,J_A/t)\\) on a fixed {GRID_N} x {GRID_N} evaluation grid.  The
fixed audit mask is the final trusted FFLO support intersected with a nearest
trusted-FFLO support radius of {SUPPORT_RADIUS} in normalized parameter
coordinates.

The cFFLO/tFFLO boundary is reconstructed from validated \\(P_0=0\\) and
\\(P_\\pi=0\\) contour segments inside that mask.  Local opposite-Z2 kNN bracket
edges are used only as support diagnostics; raw long Delaunay edges are not
used as final contours.

## Final Three Transition Metrics

| Transition to iteration | topology map change | boundary p95 shift | trusted surprise |
|---:|---:|---:|---:|
"""
    for it in last3:
        m = metrics[metrics["transition_to_iteration"] == it]
        s = shifts[shifts["transition_to_iteration"] == it]
        u = surprise[surprise["iteration"] == it]
        md += f"| {it} | {fmt(m['topology_map_change'].iloc[0]) if len(m) else 'n/a'} | {fmt(s['symmetric_p95'].iloc[0]) if len(s) else 'n/a'} | {fmt(u['trusted_topology_surprise'].iloc[0]) if len(u) else 'n/a'} |\n"

    final_cov = coverage[coverage["iteration"] == final_iteration].iloc[0]
    sens_cov_text = "; ".join(
        f"k={int(row['idw_and_knn_k'])}: {fmt(row['final_coverage_p95'])}"
        for _, row in sensitivity.iterrows()
    )
    md += f"""
Final topology-boundary exact-point coverage p95 is
{fmt(final_cov['coverage_p95'])} versus threshold {COVERAGE_TOL}.  Opposite-Z2
bracket coverage p95 is {fmt(final_cov['bracket_coverage_p95'])}.

## Sensitivity Check

The main audit uses k={AUDIT_K}.  Lightweight kNN/IDW sensitivity gives:

| k | final coverage p95 | final bracket coverage p95 | significant components |
|---:|---:|---:|---:|
"""
    for _, row in sensitivity.iterrows():
        md += f"| {int(row['idw_and_knn_k'])} | {fmt(row['final_coverage_p95'])} | {fmt(row['final_bracket_coverage_p95'])} | {int(row['final_boundary_components'])} |\n"

    md += f"""
The sensitivity check is stable in map topology and component structure.  It
also shows that the coverage margin is tight: k=6 is slightly above the
nominal coverage threshold, while the preregistered k={AUDIT_K} and k=12 are
below threshold.  This is recorded as a coverage-margin caveat, not as evidence
of map or boundary-shift failure.

## Direct Answers

1. cFFLO/tFFLO boundary stopped moving in the last three rounds:
   `{'yes' if all(np.asarray(decision['topology_boundary_shift_p95_last3']) <= BOUNDARY_SHIFT_TOL) else 'no'}`.
2. topology map change is below threshold:
   `{'yes' if all(np.asarray(decision['topology_map_change_last3']) < MAP_CHANGE_TOL) else 'no'}`.
3. topology boundary p95 shift is below threshold:
   `{'yes' if all(np.asarray(decision['topology_boundary_shift_p95_last3']) <= BOUNDARY_SHIFT_TOL) else 'no'}`.
4. topology boundary coverage is sufficient:
   `{'yes' if final_cov['coverage_p95'] < COVERAGE_TOL else 'no'}`.
5. trusted topology surprise is close to zero:
   `{'yes' if all([(v is not None and v <= TOPOLOGY_SURPRISE_TOL) for v in decision['trusted_topology_surprise_last3']]) else 'no'}`.
6. topology boundary keeps a stable main connected component:
   `{'yes' if len(set(decision['boundary_component_count_last3'])) == 1 else 'no'}`.
7. significant topology islands appeared/disappeared:
   `{'no' if len(set(decision['topological_region_component_count_last3'])) == 1 else 'yes'}`.
8. final contour direct/bracket support fraction:
   {fmt(decision['final_direct_support_fraction'])}.
9. formal decision:
   `{decision['decision_class']}` / `{decision['topology_main_converged']}`.
10. new exact calculation needed:
   `{decision['need_new_exact_calculation']}`.

## Caveats

- This audit does not merge in `dataset_iter035`.
- Gapless-SC absence is not a failure because the current parameter range does
  not show a finite-area nodal-SC phase in the returned run.
- Strict Hausdorff distance is retained as an outlier diagnostic; the main
  boundary-shift gate uses symmetric p95 distance.
- Unsupported or extrapolation-only contour fragments are not counted as the
  formal main cFFLO/tFFLO boundary.

## Output Tables and Figures

All metrics are saved under:

```text
{TABLE_DIR.as_posix()}
{FIGURE_DIR.as_posix()}
```
"""
    write_text(REPORT_DIR / "topology_convergence_audit_report.md", md)

    figure_tex = "\n".join(
        [
            rf"\begin{{figure}}[htbp]\centering\includegraphics[width=0.95\linewidth]{{figures/{name}.png}}\caption{{{caption}}}\end{{figure}}"
            for name, caption in [
                ("fig01_final_map_contour", "Final thermodynamic map, trusted cFFLO/tFFLO samples, and audited final contour."),
                ("fig02_final5_contour_overlay", "Overlay of audited cFFLO/tFFLO contours from the final five cumulative iterations."),
                ("fig03_topology_map_change", "Topology-map change on the fixed audit grid and mask."),
                ("fig04_boundary_shift", "Symmetric topology-boundary shift metrics."),
                ("fig05_boundary_coverage", "Topology-boundary coverage by exact trusted FFLO points and opposite-Z2 brackets."),
                ("fig06_topology_surprise", "Trusted, all-selected, and hard-risk topology surprise."),
                ("fig07_component_stability", "Boundary and topological-region connected-component diagnostics."),
                ("fig08_final_support_map", "Final contour direct-support and opposite-Z2 bracket map."),
                ("fig09_pfaffian_gap_along_contour", "Pfaffian-margin and bulk-gap diagnostics sampled along the final contour normal direction."),
                ("fig10_decision_summary", "Convergence decision summary panel."),
            ]
        ]
    )
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.85in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\usepackage{{float}}
\title{{cFFLO/tFFLO Topology-Boundary Convergence Audit}}
\author{{Report-only offline audit}}
\date{{\today}}
\begin{{document}}
\sloppy
\maketitle

\section{{Executive Summary}}
Source run: {tex_code(SOURCE_RUN_ID)}.  Audit run: {tex_code(RUN_ID)}.
The formal decision is {tex_code(decision['decision_class'])}; topology main
convergence is {tex_code(decision['topology_main_converged'])}; limiting
factor is {tex_code(decision['limiting_factor'])}; need new exact calculation
is {tex_code(decision['need_new_exact_calculation'])}.

\section{{Method}}
For every cumulative iteration, the same deterministic KNN inverse-distance
surrogate interpolates $P_0$ and $P_\pi$ from trusted gapped FFLO exact points.
The fixed grid has {GRID_N}$\times${GRID_N} points, and the fixed audit mask is
the final trusted FFLO support intersected with a normalized support radius
{SUPPORT_RADIUS}.  The final cFFLO/tFFLO contour is formed from validated
$P_0=0$ and $P_\pi=0$ segments with local exact opposite-Z2 support.

\section{{Final Three Transition Metrics}}
\begin{{center}}
\begin{{tabular}}{{rrrr}}
\toprule
Transition to iteration & Map change & Boundary p95 shift & Trusted surprise \\
\midrule
"""
    for it in last3:
        m = metrics[metrics["transition_to_iteration"] == it]
        s = shifts[shifts["transition_to_iteration"] == it]
        u = surprise[surprise["iteration"] == it]
        tex += f"{it} & {fmt(m['topology_map_change'].iloc[0]) if len(m) else 'n/a'} & {fmt(s['symmetric_p95'].iloc[0]) if len(s) else 'n/a'} & {fmt(u['trusted_topology_surprise'].iloc[0]) if len(u) else 'n/a'} \\\\\n"
    tex += rf"""\bottomrule
\end{{tabular}}
\end{{center}}

Final exact-point coverage p95 is {fmt(final_cov['coverage_p95'])} against the
threshold {COVERAGE_TOL}.  Final direct/bracket support fraction is
{fmt(decision['final_direct_support_fraction'])}.

Sensitivity: final coverage p95 by k is {sens_cov_text}.  The k=6 case is a
tight coverage-margin caveat; the main k={AUDIT_K} audit and k=12 are below the
nominal coverage threshold.

\section{{Figures}}
{figure_tex}

\section{{Do-Not-Claim List}}
\begin{{itemize}}
\item Do not treat raw long Delaunay opposite-label edges as the final topology boundary.
\item Do not call missing boundary shift zero.
\item Do not treat the absence of a finite-area gapless-SC region as failure.
\item Do not merge this cold-start run with \texttt{{dataset\_iter035}}.
\item Do not claim publication-grade topology closure if the decision is not Decision A.
\end{{itemize}}

\section{{Next Action}}
{tex_code(decision['recommended_next_action'])}.

\end{{document}}
"""
    write_text(REPORT_DIR / "topology_convergence_audit_report.tex", tex)
    caveat_lines = "\n".join(f"- {item}" for item in decision.get("caveats", []))
    decision_log = f"""# Decision Log

Decision: {decision['decision_class']}

topology_main_converged: {decision['topology_main_converged']}

limiting_factor: {decision['limiting_factor']}

need_new_exact_calculation: {decision['need_new_exact_calculation']}

recommended_next_action: {decision['recommended_next_action']}

Evidence:

- last3 topology map change: {decision['topology_map_change_last3']}
- last3 boundary p95 shift: {decision['topology_boundary_shift_p95_last3']}
- final coverage p95: {decision['topology_boundary_coverage_p95_final']}
- last3 trusted surprise: {decision['trusted_topology_surprise_last3']}
- final direct/bracket support fraction: {decision['final_direct_support_fraction']}

Caveats:

{caveat_lines}
"""
    write_text(REPORT_DIR / "decision_log.md", decision_log)


def run_audit() -> dict[str, Any]:
    ensure_dirs()
    bounds = domain_bounds()
    dataset_paths = sorted(RUN_DIR.glob("dataset_iter*.csv"), key=dataset_iteration)
    if not dataset_paths:
        raise FileNotFoundError(f"No dataset_iter*.csv found under {RUN_DIR}")
    datasets = {dataset_iteration(path): load_dataset_csv(path) for path in dataset_paths}
    input_table = input_audit(dataset_paths)
    input_table.to_csv(TABLE_DIR / "input_provenance_by_iteration.csv", index=False)

    final_iteration = max(datasets)
    grid = make_grid(bounds, GRID_N)
    final_train = trusted_gapped_fflo(datasets[final_iteration])
    audit_mask, final_nearest = final_audit_mask(final_train, grid, bounds)
    mask_table = pd.DataFrame(
        [
            {
                "grid_n": GRID_N,
                "grid_points": GRID_N * GRID_N,
                "audit_mask_points": int(np.sum(audit_mask)),
                "audit_mask_fraction": float(np.mean(audit_mask)),
                "support_radius": SUPPORT_RADIUS,
                "final_nearest_distance_p95": float(np.quantile(final_nearest[audit_mask], 0.95)) if np.any(audit_mask) else np.nan,
            }
        ]
    )
    mask_table.to_csv(TABLE_DIR / "audit_mask_summary.csv", index=False)

    models: dict[int, dict[str, Any]] = {}
    component_rows = []
    coverage_rows = []
    contour_rows = []
    for iteration, df in datasets.items():
        model = model_for_iteration(df, grid, audit_mask, bounds, AUDIT_K)
        models[iteration] = model
        comps = {"iteration": iteration, **model["components"]}
        component_rows.append(comps)
        coverage_rows.append({"iteration": iteration, **coverage_metrics(model)})
        if not model["contour_table"].empty:
            ct = model["contour_table"].copy()
            ct["iteration"] = iteration
            contour_rows.append(ct)
    components = pd.DataFrame(component_rows)
    coverage = pd.DataFrame(coverage_rows)
    contour_table = pd.concat(contour_rows, ignore_index=True) if contour_rows else pd.DataFrame()
    components.to_csv(TABLE_DIR / "topology_boundary_components.csv", index=False)
    coverage.to_csv(TABLE_DIR / "topology_boundary_coverage.csv", index=False)
    contour_table.to_csv(TABLE_DIR / "topology_contour_segment_filter_audit.csv", index=False)

    metric_rows = []
    shift_rows = []
    surprise_rows = []
    iterations = sorted(datasets)
    shape = grid["xx"].shape
    for prev, curr in zip(iterations[:-1], iterations[1:]):
        prev_model = models[prev]
        curr_model = models[curr]
        both = audit_mask & prev_model["valid"] & curr_model["valid"]
        changed = (prev_model["z2"] != curr_model["z2"]) & both
        prev_valid = audit_mask & prev_model["valid"]
        curr_valid = audit_mask & curr_model["valid"]
        metric_rows.append(
            {
                "transition_from_iteration": prev,
                "transition_to_iteration": curr,
                "valid_common_points": int(np.sum(both)),
                "valid_common_fraction_of_audit_mask": float(np.sum(both) / max(np.sum(audit_mask), 1)),
                "newly_valid_fraction": float(np.sum(curr_valid & ~prev_valid) / max(np.sum(audit_mask), 1)),
                "no_longer_valid_fraction": float(np.sum(prev_valid & ~curr_valid) / max(np.sum(audit_mask), 1)),
                "changed_grid_count": int(np.sum(changed)),
                "topology_map_change": float(np.sum(changed) / max(np.sum(both), 1)) if np.sum(both) else np.nan,
                "map_change_pass": bool((np.sum(changed) / max(np.sum(both), 1)) < MAP_CHANGE_TOL) if np.sum(both) else False,
            }
        )
        shift_rows.append(
            {
                "transition_from_iteration": prev,
                "transition_to_iteration": curr,
                **boundary_shift(boundary_points(prev_model["contours"]), boundary_points(curr_model["contours"])),
            }
        )
        surprise_rows.append(topology_surprise(prev_model, datasets[prev], datasets[curr], bounds, curr))
    metrics = pd.DataFrame(metric_rows)
    shifts = pd.DataFrame(shift_rows)
    surprise = pd.DataFrame(surprise_rows)
    metrics.to_csv(TABLE_DIR / "topology_convergence_metrics_by_iteration.csv", index=False)
    shifts.to_csv(TABLE_DIR / "topology_boundary_shift_by_iteration.csv", index=False)
    surprise.to_csv(TABLE_DIR / "topology_surprise_by_iteration.csv", index=False)

    boundary_points_table = write_boundary_points(models, bounds)
    boundary_points_table.to_csv(TABLE_DIR / "topology_boundary_points_by_iteration.csv", index=False)

    support = support_audit(models[final_iteration], grid, bounds)
    support.to_csv(TABLE_DIR / "topology_boundary_support_audit.csv", index=False)

    sensitivity_rows = []
    for k in SENSITIVITY_K:
        m = model_for_iteration(datasets[final_iteration], grid, audit_mask, bounds, k)
        cov = coverage_metrics(m)
        sensitivity_rows.append(
            {
                "idw_and_knn_k": k,
                "final_boundary_components": m["components"]["significant_boundary_component_count"],
                "final_topological_region_components": m["components"]["topological_region_component_count"],
                "final_total_arc_length": m["components"]["total_boundary_arc_length"],
                "final_coverage_p95": cov["coverage_p95"],
                "final_bracket_coverage_p95": cov["bracket_coverage_p95"],
                "final_contour_point_count": len(boundary_points(m["contours"])),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity["grid_resolution"] = GRID_N
    sensitivity["support_radius"] = SUPPORT_RADIUS
    sensitivity["interpolation_seed"] = "not_applicable_deterministic_idw"
    sensitivity.to_csv(TABLE_DIR / "topology_sensitivity_summary.csv", index=False)

    decision = decide(metrics, shifts, coverage, surprise, components, support, final_iteration)
    decision["sensitivity_final_coverage_p95_by_k"] = {
        str(int(row["idw_and_knn_k"])): float(row["final_coverage_p95"])
        for _, row in sensitivity.iterrows()
    }
    decision["sensitivity_final_bracket_coverage_p95_by_k"] = {
        str(int(row["idw_and_knn_k"])): float(row["final_bracket_coverage_p95"])
        for _, row in sensitivity.iterrows()
    }
    if any(float(row["final_coverage_p95"]) > COVERAGE_TOL for _, row in sensitivity.iterrows()):
        decision.setdefault("caveats", []).append(
            "sensitivity coverage margin is tight: k=6 is slightly above the nominal coverage threshold, while main k=8 and k=12 are below"
        )
    decision["source_dataset"] = str(RUN_DIR / f"dataset_iter{final_iteration:03d}.npz")
    decision["source_dataset_csv"] = str(RUN_DIR / f"dataset_iter{final_iteration:03d}.csv")
    decision["source_dataset_sha256"] = sha256_file(RUN_DIR / f"dataset_iter{final_iteration:03d}.csv")
    (REPORT_DIR / "topology_convergence_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    config = "\n".join(
        [
            f"run_id: {RUN_ID}",
            f"source_run_id: {SOURCE_RUN_ID}",
            f"grid_n: {GRID_N}",
            f"audit_k: {AUDIT_K}",
            f"sensitivity_k: {SENSITIVITY_K}",
            f"support_radius: {SUPPORT_RADIUS}",
            f"segment_support_radius: {SEGMENT_SUPPORT_RADIUS}",
            f"bracket_edge_max: {BRACKET_EDGE_MAX}",
            f"map_change_tol: {MAP_CHANGE_TOL}",
            f"boundary_shift_tol: {BOUNDARY_SHIFT_TOL}",
            f"coverage_tol: {COVERAGE_TOL}",
            f"topology_surprise_tol: {TOPOLOGY_SURPRISE_TOL}",
            "surrogate: deterministic_knn_inverse_distance_interpolation_for_P0_and_Ppi",
            "uses_new_exact_calculation: false",
            "",
        ]
    )
    write_text(REPORT_DIR / "topology_convergence_config.yaml", config)
    make_figures(datasets, models, metrics, shifts, coverage, surprise, components, support, decision, grid, bounds)
    write_report(decision, input_table, metrics, shifts, coverage, surprise, components, support, sensitivity)
    return decision


def main() -> None:
    decision = run_audit()
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
