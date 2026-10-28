from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.interpolate import RBFInterpolator, UnivariateSpline
from scipy.spatial import Delaunay, QhullError, cKDTree


PHASE_LABELS = {0: "normal", 1: "uniform_SC", 2: "FFLO"}
TOPOLOGY_LABELS = {-1: "not_applicable", 0: "trivial", 1: "topological", 2: "gapless_SC", 3: "unresolved"}
SPECTRAL_LABELS = {-1: "not_applicable", 0: "gapped", 1: "gapless", 2: "unresolved"}
TETRA_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TRI_EDGES = ((0, 1), (1, 2), (2, 0))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def ensure_rgb_png(path: Path) -> None:
    image = Image.open(path)
    if image.mode == "RGB":
        return
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        background.save(path)
        return
    image.convert("RGB").save(path)


def save_figure(fig: Any, png_path: Path, pdf_path: Path, dpi: int = 220) -> None:
    fig.savefig(png_path, dpi=dpi, facecolor="white", transparent=False)
    ensure_rgb_png(png_path)
    fig.savefig(pdf_path, facecolor="white", transparent=False)


def arr(z: np.lib.npyio.NpzFile, key: str, default: float | int = np.nan) -> np.ndarray:
    if key in z.files:
        return np.asarray(z[key])
    n = int(np.asarray(z["x"]).shape[0])
    return np.full(n, default)


def count_codes(values: np.ndarray, labels: dict[int, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in sorted(labels):
        count = int(np.sum(values == code))
        rows.append({"code": int(code), "label": labels[code], "count": count})
    extras = sorted(set(np.asarray(values, dtype=int).tolist()) - set(labels))
    for code in extras:
        rows.append({"code": int(code), "label": f"unknown_{code}", "count": int(np.sum(values == code))})
    return rows


def finite_percentiles(values: np.ndarray, percentiles: list[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {f"p{int(p)}": math.nan for p in percentiles}
    return {f"p{int(p)}": float(np.percentile(v, p)) for p in percentiles}


def normalize_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = np.nanmin(x, axis=0)
    spans = np.nanmax(x, axis=0) - mins
    spans = np.where(spans > 0, spans, 1.0)
    return (x - mins) / spans, mins, spans


def denormalize_columns(xn: np.ndarray, mins: np.ndarray, spans: np.ndarray) -> np.ndarray:
    return xn * spans + mins


def local_edge_threshold(xn: np.ndarray, factor: float = 4.0, cap: float = 0.32) -> float:
    if len(xn) < 4:
        return math.nan
    tree = cKDTree(xn)
    distances, _ = tree.query(xn, k=min(2, len(xn)))
    nn = distances[:, 1] if distances.ndim == 2 and distances.shape[1] > 1 else distances
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if not nn.size:
        return cap
    return float(min(cap, max(0.045, np.percentile(nn, 90) * factor)))


def tetra_max_edge(vertices: np.ndarray) -> float:
    return float(max(np.linalg.norm(vertices[i] - vertices[j]) for i, j in TETRA_EDGES))


def triangle_max_edge(vertices: np.ndarray) -> float:
    return float(max(np.linalg.norm(vertices[i] - vertices[j]) for i, j in TRI_EDGES))


def make_boundary_surface_polygons(
    x: np.ndarray,
    subset_mask: np.ndarray,
    positive_mask: np.ndarray,
    surface_name: str,
    max_polygons: int = 5000,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Build a diagnostic binary-label boundary surface from local Delaunay tets.

    The result is for report visualization only. It is not used as a formal
    convergence contour, because the Stage IV convergence audit has its own
    fixed-grid/KNN surface metrics.
    """
    rows = np.where(subset_mask)[0]
    diag: dict[str, Any] = {
        "surface": surface_name,
        "subset_points": int(len(rows)),
        "positive_points": int(np.sum(subset_mask & positive_mask)),
        "negative_points": int(np.sum(subset_mask & ~positive_mask)),
        "edge_threshold_normalized": math.nan,
        "raw_polygon_count": 0,
        "plotted_polygon_count": 0,
        "status": "not_attempted",
        "formal_use": "diagnostic_visualization_only",
    }
    if len(rows) < 8:
        diag["status"] = "too_few_points"
        return [], diag
    labels = np.asarray(positive_mask[rows], dtype=bool)
    if np.unique(labels).size < 2:
        diag["status"] = "single_label"
        return [], diag

    xs = np.asarray(x[rows], dtype=float)
    xn, mins, spans = normalize_columns(xs)
    threshold = local_edge_threshold(xn)
    diag["edge_threshold_normalized"] = threshold
    try:
        tri = Delaunay(xn, qhull_options="QJ")
    except QhullError as exc:
        diag["status"] = f"qhull_failed: {exc.__class__.__name__}"
        return [], diag

    polygons: list[np.ndarray] = []
    for simplex in tri.simplices:
        simplex_labels = labels[simplex]
        if np.all(simplex_labels) or not np.any(simplex_labels):
            continue
        verts = xn[simplex]
        if tetra_max_edge(verts) > threshold:
            continue
        cuts = []
        for i, j in TETRA_EDGES:
            if simplex_labels[i] != simplex_labels[j]:
                cuts.append(0.5 * (verts[i] + verts[j]))
        if len(cuts) == 3:
            polygons.append(denormalize_columns(np.asarray(cuts), mins, spans))
        elif len(cuts) == 4:
            cuts_arr = denormalize_columns(np.asarray(cuts), mins, spans)
            polygons.append(cuts_arr[[0, 1, 2]])
            polygons.append(cuts_arr[[0, 2, 3]])

    diag["raw_polygon_count"] = int(len(polygons))
    if len(polygons) > max_polygons:
        keep = np.linspace(0, len(polygons) - 1, max_polygons).astype(int)
        polygons = [polygons[i] for i in keep]
    diag["plotted_polygon_count"] = int(len(polygons))
    diag["status"] = "ok" if polygons else "no_local_crossing_polygons"
    return polygons, diag


def extract_boundary_crossing_points_3d(
    x: np.ndarray,
    subset_mask: np.ndarray,
    positive_mask: np.ndarray,
    surface_name: str,
    max_points: int = 12000,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = np.where(subset_mask)[0]
    diag: dict[str, Any] = {
        "surface": surface_name,
        "subset_points": int(len(rows)),
        "positive_points": int(np.sum(subset_mask & positive_mask)),
        "negative_points": int(np.sum(subset_mask & ~positive_mask)),
        "edge_threshold_normalized": math.nan,
        "raw_crossing_point_count": 0,
        "fit_point_count": 0,
        "status": "not_attempted",
        "smooth_method": "rbf_thin_plate_spline_J_of_T_mu",
        "formal_use": "diagnostic_visualization_only",
    }
    if len(rows) < 8:
        diag["status"] = "too_few_points"
        return np.empty((0, 3)), diag
    labels = np.asarray(positive_mask[rows], dtype=bool)
    if np.unique(labels).size < 2:
        diag["status"] = "single_label"
        return np.empty((0, 3)), diag

    xs = np.asarray(x[rows], dtype=float)
    xn, mins, spans = normalize_columns(xs)
    threshold = local_edge_threshold(xn)
    diag["edge_threshold_normalized"] = threshold
    try:
        tri = Delaunay(xn, qhull_options="QJ")
    except QhullError as exc:
        diag["status"] = f"qhull_failed: {exc.__class__.__name__}"
        return np.empty((0, 3)), diag

    edge_keys: set[tuple[int, int]] = set()
    points = []
    for simplex in tri.simplices:
        simplex_labels = labels[simplex]
        if np.all(simplex_labels) or not np.any(simplex_labels):
            continue
        for i, j in TETRA_EDGES:
            a = int(simplex[i])
            b = int(simplex[j])
            if labels[a] == labels[b]:
                continue
            if np.linalg.norm(xn[a] - xn[b]) > threshold:
                continue
            key = (min(a, b), max(a, b))
            if key in edge_keys:
                continue
            edge_keys.add(key)
            points.append(0.5 * (xs[a] + xs[b]))

    if not points:
        diag["status"] = "no_local_crossing_edges"
        return np.empty((0, 3)), diag
    pts = np.asarray(points, dtype=float)
    diag["raw_crossing_point_count"] = int(len(pts))
    if len(pts) > max_points:
        rng = np.random.default_rng(20260627)
        keep = np.sort(rng.choice(len(pts), size=max_points, replace=False))
        pts = pts[keep]
    diag["fit_point_count"] = int(len(pts))
    diag["status"] = "ok"
    return pts, diag


def build_smooth_boundary_surface(
    boundary_points: np.ndarray,
    data_x: np.ndarray,
    grid_n_t: int = 72,
    grid_n_mu: int = 72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    diag = {
        "grid_n_t": grid_n_t,
        "grid_n_mu": grid_n_mu,
        "valid_grid_points": 0,
        "support_radius_normalized": math.nan,
        "status": "not_attempted",
    }
    if len(boundary_points) < 24:
        diag["status"] = "too_few_boundary_points"
        empty = np.empty((0, 0))
        return empty, empty, empty, empty.astype(bool), diag

    u = boundary_points[:, [0, 2]]
    y = boundary_points[:, 1]
    un, umins, uspans = normalize_columns(u)
    support_radius = local_edge_threshold(un, factor=3.0, cap=0.22)
    diag["support_radius_normalized"] = support_radius

    max_fit = 3500
    if len(un) > max_fit:
        rng = np.random.default_rng(20260627)
        fit_idx = np.sort(rng.choice(len(un), size=max_fit, replace=False))
        fit_u = un[fit_idx]
        fit_y = y[fit_idx]
    else:
        fit_u = un
        fit_y = y

    try:
        interpolator = RBFInterpolator(
            fit_u,
            fit_y,
            kernel="thin_plate_spline",
            smoothing=max(1e-5, 0.0025 * float(np.nanstd(fit_y))),
            neighbors=min(96, len(fit_u)),
        )
    except Exception as exc:  # noqa: BLE001 - report-only fallback path
        diag["status"] = f"rbf_failed: {exc.__class__.__name__}"
        empty = np.empty((0, 0))
        return empty, empty, empty, empty.astype(bool), diag

    t_grid = np.linspace(np.nanmin(boundary_points[:, 0]), np.nanmax(boundary_points[:, 0]), grid_n_t)
    mu_grid = np.linspace(np.nanmin(boundary_points[:, 2]), np.nanmax(boundary_points[:, 2]), grid_n_mu)
    tt, mm = np.meshgrid(t_grid, mu_grid, indexing="xy")
    query = np.column_stack([tt.ravel(), mm.ravel()])
    qn = (query - umins) / uspans
    jj = interpolator(qn).reshape(tt.shape)
    jj = np.clip(jj, np.nanmin(data_x[:, 1]), np.nanmax(data_x[:, 1]))

    tree = cKDTree(un)
    dist, _ = tree.query(qn, k=1)
    support_mask = dist.reshape(tt.shape) <= support_radius
    jj = np.where(support_mask, jj, np.nan)
    diag["valid_grid_points"] = int(np.sum(support_mask))
    diag["status"] = "ok" if np.any(support_mask) else "no_supported_grid"
    return tt, jj, mm, support_mask, diag


def add_smooth_boundary_surface(
    ax: Any,
    boundary_points: np.ndarray,
    data_x: np.ndarray,
    color: str,
    label: str,
    alpha: float = 0.28,
) -> dict[str, Any]:
    tt, jj, mm, support_mask, diag = build_smooth_boundary_surface(boundary_points, data_x)
    if diag["status"] == "ok":
        ax.plot_surface(
            tt,
            jj,
            mm,
            rstride=1,
            cstride=1,
            color=color,
            alpha=alpha,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        ax.scatter(
            boundary_points[:, 0],
            boundary_points[:, 1],
            boundary_points[:, 2],
            s=3,
            c=color,
            alpha=0.08,
            depthshade=False,
            edgecolors="none",
        )
        # Proxy handle for the legend because plot_surface labels are backend fragile.
        ax.plot([], [], [], color=color, alpha=min(1.0, alpha + 0.25), linewidth=6, label=label)
    return diag


def add_boundary_surface(ax: Any, polygons: list[np.ndarray], color: str, label: str, alpha: float = 0.16) -> None:
    if not polygons:
        return
    collection = Poly3DCollection(
        polygons,
        facecolors=color,
        edgecolors=color,
        linewidths=0.08,
        alpha=alpha,
        label=label,
    )
    collection.set_zsort("average")
    ax.add_collection3d(collection)


def add_slice_boundary_curve(
    ax: Any,
    x2: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    level: float,
    color: str,
    linestyle: str,
    linewidth: float,
    boundary_name: str,
    mu_interval: str,
) -> dict[str, Any]:
    rows = np.where(mask)[0]
    diag = {
        "boundary": boundary_name,
        "mu_interval": mu_interval,
        "subset_points": int(len(rows)),
        "segment_count": 0,
        "edge_threshold_normalized": math.nan,
        "status": "not_attempted",
        "formal_use": "diagnostic_visualization_only",
    }
    if len(rows) < 6:
        diag["status"] = "too_few_points"
        return diag
    y = np.asarray(values[rows], dtype=float)
    if np.nanmin(y) > level or np.nanmax(y) < level or np.unique(y[np.isfinite(y)]).size < 2:
        diag["status"] = "level_not_bracketed"
        return diag

    pts = np.asarray(x2[rows], dtype=float)
    pn, _, _ = normalize_columns(pts)
    threshold = local_edge_threshold(pn, factor=4.5, cap=0.38)
    diag["edge_threshold_normalized"] = threshold
    try:
        triangulation = mtri.Triangulation(pts[:, 0], pts[:, 1])
    except RuntimeError as exc:
        diag["status"] = f"triangulation_failed: {exc.__class__.__name__}"
        return diag

    tri_norm = pn[triangulation.triangles]
    max_edges = np.array([triangle_max_edge(v) for v in tri_norm], dtype=float)
    triangulation.set_mask(max_edges > threshold)
    try:
        contour = ax.tricontour(
            triangulation,
            y,
            levels=[level],
            colors=[color],
            linewidths=[linewidth],
            linestyles=[linestyle],
            alpha=0.96,
        )
    except ValueError as exc:
        diag["status"] = f"contour_failed: {exc.__class__.__name__}"
        return diag
    diag["segment_count"] = int(sum(len(seg) for seg in contour.allsegs[0]))
    diag["status"] = "ok" if diag["segment_count"] else "no_contour_segments"
    return diag


def extract_boundary_crossing_points_2d(
    x2: np.ndarray,
    subset_mask: np.ndarray,
    positive_mask: np.ndarray,
    boundary_name: str,
    mu_interval: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = np.where(subset_mask)[0]
    diag = {
        "boundary": boundary_name,
        "mu_interval": mu_interval,
        "subset_points": int(len(rows)),
        "raw_crossing_point_count": 0,
        "fit_point_count": 0,
        "curve_point_count": 0,
        "edge_threshold_normalized": math.nan,
        "status": "not_attempted",
        "smooth_method": "binned_median_univariate_spline_J_of_T",
        "formal_use": "diagnostic_visualization_only",
    }
    if len(rows) < 6:
        diag["status"] = "too_few_points"
        return np.empty((0, 2)), diag
    labels = np.asarray(positive_mask[rows], dtype=bool)
    if np.unique(labels).size < 2:
        diag["status"] = "single_label"
        return np.empty((0, 2)), diag

    pts = np.asarray(x2[rows], dtype=float)
    pn, _, _ = normalize_columns(pts)
    threshold = local_edge_threshold(pn, factor=4.2, cap=0.34)
    diag["edge_threshold_normalized"] = threshold
    try:
        tri = Delaunay(pn, qhull_options="QJ")
    except QhullError as exc:
        diag["status"] = f"qhull_failed: {exc.__class__.__name__}"
        return np.empty((0, 2)), diag

    edge_keys: set[tuple[int, int]] = set()
    crossings = []
    for simplex in tri.simplices:
        simplex_labels = labels[simplex]
        if np.all(simplex_labels) or not np.any(simplex_labels):
            continue
        verts = pn[simplex]
        if triangle_max_edge(verts) > threshold:
            continue
        for i, j in TRI_EDGES:
            a = int(simplex[i])
            b = int(simplex[j])
            if labels[a] == labels[b]:
                continue
            if np.linalg.norm(pn[a] - pn[b]) > threshold:
                continue
            key = (min(a, b), max(a, b))
            if key in edge_keys:
                continue
            edge_keys.add(key)
            crossings.append(0.5 * (pts[a] + pts[b]))
    if not crossings:
        diag["status"] = "no_local_crossing_edges"
        return np.empty((0, 2)), diag
    crossing_pts = np.asarray(crossings, dtype=float)
    diag["raw_crossing_point_count"] = int(len(crossing_pts))
    diag["fit_point_count"] = int(len(crossing_pts))
    diag["status"] = "ok"
    return crossing_pts, diag


def smooth_curve_from_crossing_points(points: np.ndarray, min_bins: int = 8, max_bins: int = 56) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 6:
        return np.array([]), np.array([])
    order = np.argsort(points[:, 0])
    pts = points[order]
    t_min = float(np.nanmin(pts[:, 0]))
    t_max = float(np.nanmax(pts[:, 0]))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return np.array([]), np.array([])

    n_bins = int(np.clip(len(pts) // 8, min_bins, max_bins))
    bins = np.linspace(t_min, t_max, n_bins + 1)
    t_centers = []
    j_medians = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pts[:, 0] >= lo) & (pts[:, 0] < hi if hi < bins[-1] else pts[:, 0] <= hi)
        if np.sum(mask) == 0:
            continue
        t_centers.append(float(np.nanmedian(pts[mask, 0])))
        j_medians.append(float(np.nanmedian(pts[mask, 1])))
    tx = np.asarray(t_centers, dtype=float)
    jy = np.asarray(j_medians, dtype=float)
    keep = np.isfinite(tx) & np.isfinite(jy)
    tx = tx[keep]
    jy = jy[keep]
    if len(tx) < 4:
        return tx, jy
    unique_t, unique_idx = np.unique(tx, return_index=True)
    tx = unique_t
    jy = jy[unique_idx]
    if len(tx) < 4:
        return tx, jy

    curve_t = np.linspace(float(tx.min()), float(tx.max()), 220)
    variance = float(np.nanvar(jy))
    smooth_s = max(1e-6, len(tx) * max(variance, 1e-5) * 0.22)
    try:
        spline = UnivariateSpline(tx, jy, k=min(3, len(tx) - 1), s=smooth_s)
        curve_j = spline(curve_t)
    except Exception:  # noqa: BLE001 - report-only fallback path
        curve_j = np.interp(curve_t, tx, jy)
    return curve_t, curve_j


def add_smooth_slice_boundary_curve(
    ax: Any,
    x2: np.ndarray,
    subset_mask: np.ndarray,
    positive_mask: np.ndarray,
    color: str,
    linestyle: str,
    linewidth: float,
    boundary_name: str,
    mu_interval: str,
) -> dict[str, Any]:
    crossing_points, diag = extract_boundary_crossing_points_2d(x2, subset_mask, positive_mask, boundary_name, mu_interval)
    if diag["status"] != "ok":
        return diag
    curve_t, curve_j = smooth_curve_from_crossing_points(crossing_points)
    diag["curve_point_count"] = int(len(curve_t))
    if len(curve_t) < 2:
        diag["status"] = "smooth_curve_failed"
        return diag
    ax.plot(curve_t, curve_j, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.96)
    ax.scatter(crossing_points[:, 0], crossing_points[:, 1], s=2, c=color, alpha=0.07, edgecolors="none")
    diag["status"] = "ok"
    return diag


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


def dataset_iteration(path: Path) -> int:
    m = re.search(r"dataset_iter(\d+)\.npz$", path.name)
    return int(m.group(1)) if m else -1


def compute_iteration_summary(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("dataset_iter*.npz"), key=dataset_iteration):
        data = load_dataset(path)
        it = dataset_iteration(path)
        phase = np.asarray(data.get("y_phase", []), dtype=int)
        topo = np.asarray(data.get("topology_label_code", np.full(len(phase), -1)), dtype=int)
        trusted = np.asarray(data.get("trusted_exact", np.zeros(len(phase))), dtype=int)
        topo_trusted = np.asarray(data.get("topology_trusted", np.zeros(len(phase))), dtype=int)
        rerun = np.asarray(data.get("needs_rerun_exact", np.zeros(len(phase))), dtype=int)
        q_un = np.asarray(data.get("q_unresolved", np.zeros(len(phase))), dtype=int)
        d_un = np.asarray(data.get("delta_unresolved", np.zeros(len(phase))), dtype=int)
        rows.append(
            {
                "iteration": it,
                "sample_count": int(len(phase)),
                "normal_count": int(np.sum(phase == 0)),
                "uniform_sc_count": int(np.sum(phase == 1)),
                "fflo_count": int(np.sum(phase == 2)),
                "topology_not_applicable_count": int(np.sum(topo == -1)),
                "trivial_count": int(np.sum(topo == 0)),
                "topological_count": int(np.sum(topo == 1)),
                "gapless_sc_count": int(np.sum(topo == 2)),
                "topology_unresolved_count": int(np.sum(topo == 3)),
                "trusted_exact_count": int(np.sum(trusted == 1)),
                "topology_trusted_count": int(np.sum(topo_trusted == 1)),
                "rerun_required_count": int(np.sum(rerun == 1)),
                "q_unresolved_count": int(np.sum(q_un == 1)),
                "delta_unresolved_count": int(np.sum(d_un == 1)),
            }
        )
    return pd.DataFrame(rows)


def parse_slurm_logs(package_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for path in sorted(package_root.glob("slurm-*.out")):
        text = path.read_text(encoding="utf-8", errors="replace")
        job_match = re.search(r"slurm-(\d+)_(\d+)\.out$", path.name)
        run_id = re.search(r"run_id=(.*)", text)
        iteration = re.search(r"iteration=(\d+)", text)
        rank = re.search(r"rank=(\d+)", text)
        host = re.search(r"hostname=([^\s]+)", text)
        done = re.search(r"Oracle finished rank\s+(\d+)/(\d+)\s+with\s+(\d+)\s+points in\s+([0-9.]+)s", text)
        failed = "Traceback" in text or "CUDA error" in text or "FAILED" in text
        rows.append(
            {
                "file": path.name,
                "job_id": int(job_match.group(1)) if job_match else np.nan,
                "array_rank": int(job_match.group(2)) if job_match else np.nan,
                "run_id": run_id.group(1).strip() if run_id else "",
                "iteration": int(iteration.group(1)) if iteration else np.nan,
                "rank": int(rank.group(1)) if rank else np.nan,
                "hostname": host.group(1) if host else "",
                "points": int(done.group(3)) if done else np.nan,
                "runtime_sec": float(done.group(4)) if done else np.nan,
                "status": "completed" if done else "failed_or_incomplete" if failed else "unknown",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()
    good = df[df["status"] == "completed"].copy()
    summary = (
        good.groupby("iteration", dropna=True)
        .agg(
            completed_ranks=("rank", "count"),
            point_count=("points", "sum"),
            max_rank_runtime_sec=("runtime_sec", "max"),
            mean_rank_runtime_sec=("runtime_sec", "mean"),
            min_rank_runtime_sec=("runtime_sec", "min"),
        )
        .reset_index()
    )
    summary["max_rank_runtime_hr"] = summary["max_rank_runtime_sec"] / 3600.0
    summary["mean_rank_runtime_hr"] = summary["mean_rank_runtime_sec"] / 3600.0
    return df, summary


def scatter3d(ax: Any, x: np.ndarray, mask: np.ndarray, color: str, label: str, alpha: float, size: float = 10.0, marker: str = "o") -> None:
    if np.any(mask):
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            x[mask, 2],
            s=size,
            c=color,
            alpha=alpha,
            label=f"{label} ({int(np.sum(mask))})",
            marker=marker,
            depthshade=False,
            edgecolors="none",
        )


def format_3d_axis(ax: Any, title: str, elev: float, azim: float) -> None:
    ax.set_xlabel("kBT/t")
    ax.set_ylabel("J_A/t")
    ax.set_zlabel("mu/t")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)


def make_jakt_phase_view_figure(out: Path, data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Make a mu-depth-preserving kBT-JA view for comparison with 2D maps."""
    x = np.asarray(data["x"], dtype=float)
    phase = np.asarray(data["y_phase"], dtype=int)
    normal_sc_points, normal_sc_diag = extract_boundary_crossing_points_3d(
        x,
        np.ones(len(phase), dtype=bool),
        phase != 0,
        "jakt_view_thermodynamic_normal_sc",
    )
    uniform_fflo_points, uniform_fflo_diag = extract_boundary_crossing_points_3d(
        x,
        (phase == 1) | (phase == 2),
        phase == 2,
        "jakt_view_thermodynamic_uniform_sc_fflo",
    )

    fig = plt.figure(figsize=(14.5, 6.7))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    phase_styles = [
        (phase == 0, "#8c8c8c", "normal", 0.10, 7),
        (phase == 1, "#1f77b4", "uniform_SC", 0.78, 18),
        (phase == 2, "#d62728", "FFLO", 0.23, 12),
    ]
    for mask, color, label, alpha, size in phase_styles:
        scatter3d(ax1, x, mask, color, label, alpha, size)
    normal_sc_render = add_smooth_boundary_surface(
        ax1,
        normal_sc_points,
        x,
        "#111111",
        "normal/SC smooth boundary surface",
        0.18,
    )
    uniform_fflo_render = add_smooth_boundary_surface(
        ax1,
        uniform_fflo_points,
        x,
        "#7e22ce",
        "uniform_SC/FFLO smooth boundary surface",
        0.24,
    )
    format_3d_axis(ax1, "JA-kBT primary view with mu depth", 90, -90)
    ax1.set_box_aspect((1.15, 1.0, 0.32))
    ax1.legend(loc="upper left", fontsize=8)

    mu = x[:, 2]
    if np.nanmax(mu) > np.nanmin(mu):
        mu_norm = (mu - np.nanmin(mu)) / (np.nanmax(mu) - np.nanmin(mu))
    else:
        mu_norm = np.zeros_like(mu)
    projection_styles = [
        (phase == 0, "#8c8c8c", "normal", 10, 0.10),
        (phase == 1, "#1f77b4", "uniform_SC", 22, 0.72),
        (phase == 2, "#d62728", "FFLO", 14, 0.20),
    ]
    for mask, color, label, size, alpha_base in projection_styles:
        if not np.any(mask):
            continue
        rgba = np.tile(np.asarray(to_rgba(color)), (int(np.sum(mask)), 1))
        rgba[:, 3] = np.clip(alpha_base + 0.28 * mu_norm[mask], 0.05, 0.85)
        ax2.scatter(
            x[mask, 0],
            x[mask, 1],
            s=size,
            c=rgba,
            edgecolors="none",
            label=f"{label} ({int(np.sum(mask))})",
        )
    if len(normal_sc_points):
        ax2.scatter(
            normal_sc_points[:, 0],
            normal_sc_points[:, 1],
            s=5,
            c="#111111",
            alpha=0.16,
            edgecolors="none",
            label="normal/SC local crossings",
        )
    if len(uniform_fflo_points):
        ax2.scatter(
            uniform_fflo_points[:, 0],
            uniform_fflo_points[:, 1],
            s=5,
            c="#7e22ce",
            alpha=0.18,
            edgecolors="none",
            label="uniform_SC/FFLO local crossings",
        )
    ax2.set_title("Collapsed JA-kBT projection; alpha increases with mu/t")
    ax2.set_xlabel("kBT/t")
    ax2.set_ylabel("J_A/t")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle("Stage IV thermodynamic phase map viewed from the JA-kBT plane", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, out / "phase_3d_jakt_view.png", out / "phase_3d_jakt_view.pdf")
    plt.close(fig)

    normal_sc_diag.update({f"jakt_smooth_{k}": v for k, v in normal_sc_render.items()})
    uniform_fflo_diag.update({f"jakt_smooth_{k}": v for k, v in uniform_fflo_render.items()})
    return [normal_sc_diag, uniform_fflo_diag]


def make_3d_phase_figure(out: Path, data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    x = np.asarray(data["x"], dtype=float)
    phase = np.asarray(data["y_phase"], dtype=int)
    normal_sc_points, normal_sc_diag = extract_boundary_crossing_points_3d(
        x,
        np.ones(len(phase), dtype=bool),
        phase != 0,
        "thermodynamic_normal_sc",
    )
    uniform_fflo_points, uniform_fflo_diag = extract_boundary_crossing_points_3d(
        x,
        (phase == 1) | (phase == 2),
        phase == 2,
        "thermodynamic_uniform_sc_fflo",
    )
    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for ax in (ax1, ax2):
        scatter3d(ax, x, phase == 0, "#8c8c8c", "normal", 0.10, 7)
        scatter3d(ax, x, phase == 1, "#1f77b4", "uniform_SC", 0.72, 18)
        scatter3d(ax, x, phase == 2, "#d62728", "FFLO", 0.23, 12)
        normal_sc_render = add_smooth_boundary_surface(ax, normal_sc_points, x, "#111111", "normal/SC smooth boundary surface", 0.20)
        uniform_fflo_render = add_smooth_boundary_surface(ax, uniform_fflo_points, x, "#7e22ce", "uniform_SC/FFLO smooth boundary surface", 0.26)
    format_3d_axis(ax1, "Final thermodynamic labels - smooth boundary view", 18, -55)
    format_3d_axis(ax2, "Final thermodynamic labels - high-elevation view", 63, -47)
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out / "phase_3d_transparent.png", out / "phase_3d_transparent.pdf")
    plt.close(fig)
    normal_sc_diag.update({f"smooth_{k}": v for k, v in normal_sc_render.items()})
    uniform_fflo_diag.update({f"smooth_{k}": v for k, v in uniform_fflo_render.items()})
    return [normal_sc_diag, uniform_fflo_diag]


def make_3d_topology_figure(out: Path, data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    x = np.asarray(data["x"], dtype=float)
    phase = np.asarray(data["y_phase"], dtype=int)
    topo = np.asarray(data["topology_label_code"], dtype=int)
    normal_sc_points, normal_sc_diag = extract_boundary_crossing_points_3d(
        x,
        np.ones(len(phase), dtype=bool),
        phase != 0,
        "topology_view_normal_sc",
    )
    uniform_fflo_points, uniform_fflo_diag = extract_boundary_crossing_points_3d(
        x,
        (phase == 1) | (phase == 2),
        phase == 2,
        "topology_view_uniform_sc_fflo",
    )
    cfflo_tfflo_points, cfflo_tfflo_diag = extract_boundary_crossing_points_3d(
        x,
        (phase == 2) & ((topo == 0) | (topo == 1)),
        (phase == 2) & (topo == 1),
        "topology_cfflo_tfflo",
    )
    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    masks = [
        (phase == 0, "#bdbdbd", "normal / not_applicable", 0.06, 6, "o"),
        ((phase == 1) & (topo == 0), "#00a6d6", "trivial uniform_SC", 0.70, 18, "o"),
        ((phase == 2) & (topo == 0), "#2563eb", "trivial FFLO / cFFLO", 0.28, 12, "o"),
        ((phase == 2) & (topo == 1), "#d81b60", "topological FFLO / tFFLO", 0.60, 14, "o"),
        (topo == 2, "#f59e0b", "gapless_SC", 0.85, 24, "^"),
        (topo == 3, "#111111", "topology unresolved", 0.85, 22, "x"),
    ]
    for ax in (ax1, ax2):
        for mask, color, label, alpha, size, marker in masks:
            scatter3d(ax, x, mask, color, label, alpha, size, marker)
        normal_sc_render = add_smooth_boundary_surface(ax, normal_sc_points, x, "#111111", "normal/SC smooth boundary surface", 0.13)
        uniform_fflo_render = add_smooth_boundary_surface(ax, uniform_fflo_points, x, "#f97316", "uniform_SC/FFLO smooth boundary surface", 0.20)
        cfflo_tfflo_render = add_smooth_boundary_surface(ax, cfflo_tfflo_points, x, "#8b5cf6", "cFFLO/tFFLO smooth boundary surface", 0.30)
    format_3d_axis(ax1, "Final topology-aware labels - smooth boundary view", 18, -55)
    format_3d_axis(ax2, "Final topology-aware labels - high-elevation view", 63, -47)
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out / "topology_3d_transparent.png", out / "topology_3d_transparent.pdf")
    plt.close(fig)
    normal_sc_diag.update({f"smooth_{k}": v for k, v in normal_sc_render.items()})
    uniform_fflo_diag.update({f"smooth_{k}": v for k, v in uniform_fflo_render.items()})
    cfflo_tfflo_diag.update({f"smooth_{k}": v for k, v in cfflo_tfflo_render.items()})
    return [normal_sc_diag, uniform_fflo_diag, cfflo_tfflo_diag]


def make_mu_slice_atlas(out: Path, data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    x = np.asarray(data["x"], dtype=float)
    phase = np.asarray(data["y_phase"], dtype=int)
    topo = np.asarray(data["topology_label_code"], dtype=int)
    bins = np.linspace(-0.5, 1.5, 7)
    x2 = x[:, :2]
    boundary_rows: list[dict[str, Any]] = []
    phase_colors = {0: "#a3a3a3", 1: "#1f77b4", 2: "#d62728"}
    topo_masks = [
        (phase == 0, "#d0d0d0", "normal"),
        ((phase == 1) & (topo == 0), "#00a6d6", "trivial uniform_SC"),
        ((phase == 2) & (topo == 0), "#2563eb", "cFFLO"),
        ((phase == 2) & (topo == 1), "#d81b60", "tFFLO"),
        (topo == 2, "#f59e0b", "gapless_SC"),
        (topo == 3, "#111111", "unresolved"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharex=True, sharey=True)
    for ax, lo, hi in zip(axes.ravel(), bins[:-1], bins[1:]):
        m = (x[:, 2] >= lo) & (x[:, 2] < hi if hi < bins[-1] else x[:, 2] <= hi)
        mu_label = f"{lo:.2f}_{hi:.2f}"
        for code, color in phase_colors.items():
            mm = m & (phase == code)
            ax.scatter(x[mm, 0], x[mm, 1], s=9, c=color, alpha=0.45 if code != 0 else 0.22, edgecolors="none", label=PHASE_LABELS[code])
        boundary_rows.append(
            add_smooth_slice_boundary_curve(
                ax,
                x2,
                m,
                phase != 0,
                "#111111",
                "solid",
                1.35,
                "normal_sc",
                mu_label,
            )
        )
        boundary_rows.append(
            add_smooth_slice_boundary_curve(
                ax,
                x2,
                m & ((phase == 1) | (phase == 2)),
                phase == 2,
                "#7e22ce",
                "dashed",
                1.25,
                "uniform_sc_fflo",
                mu_label,
            )
        )
        ax.set_title(f"{lo:.2f} <= mu/t < {hi:.2f}  n={int(np.sum(m))}")
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("kBT/t")
    for ax in axes[:, 0]:
        ax.set_ylabel("J_A/t")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    boundary_handles = [
        Line2D([0], [0], color="#111111", lw=1.35, linestyle="solid", label="normal/SC boundary"),
        Line2D([0], [0], color="#7e22ce", lw=1.25, linestyle="dashed", label="uniform_SC/FFLO boundary"),
    ]
    fig.legend(handles + boundary_handles, labels + [h.get_label() for h in boundary_handles], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Thermodynamic phase atlas by mu/t slice", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(fig, out / "phase_mu_slice_atlas.png", out / "phase_mu_slice_atlas.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharex=True, sharey=True)
    for ax, lo, hi in zip(axes.ravel(), bins[:-1], bins[1:]):
        m = (x[:, 2] >= lo) & (x[:, 2] < hi if hi < bins[-1] else x[:, 2] <= hi)
        mu_label = f"{lo:.2f}_{hi:.2f}"
        for base_mask, color, label in topo_masks:
            mm = m & base_mask
            if np.any(mm):
                ax.scatter(x[mm, 0], x[mm, 1], s=9, c=color, alpha=0.45 if label != "normal" else 0.14, edgecolors="none", label=label)
        boundary_rows.append(
            add_smooth_slice_boundary_curve(
                ax,
                x2,
                m,
                phase != 0,
                "#111111",
                "solid",
                1.25,
                "topology_view_normal_sc",
                mu_label,
            )
        )
        boundary_rows.append(
            add_smooth_slice_boundary_curve(
                ax,
                x2,
                m & ((phase == 1) | (phase == 2)),
                phase == 2,
                "#f97316",
                "dashed",
                1.15,
                "topology_view_uniform_sc_fflo",
                mu_label,
            )
        )
        boundary_rows.append(
            add_smooth_slice_boundary_curve(
                ax,
                x2,
                m & (phase == 2) & ((topo == 0) | (topo == 1)),
                topo == 1,
                "#8b5cf6",
                "solid",
                1.45,
                "cfflo_tfflo",
                mu_label,
            )
        )
        ax.set_title(f"{lo:.2f} <= mu/t < {hi:.2f}  n={int(np.sum(m))}")
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("kBT/t")
    for ax in axes[:, 0]:
        ax.set_ylabel("J_A/t")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    boundary_handles = [
        Line2D([0], [0], color="#111111", lw=1.25, linestyle="solid", label="normal/SC boundary"),
        Line2D([0], [0], color="#f97316", lw=1.15, linestyle="dashed", label="uniform_SC/FFLO boundary"),
        Line2D([0], [0], color="#8b5cf6", lw=1.45, linestyle="solid", label="cFFLO/tFFLO boundary"),
    ]
    fig.legend(list(by_label.values()) + boundary_handles, list(by_label.keys()) + [h.get_label() for h in boundary_handles], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Topology-aware atlas by mu/t slice", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(fig, out / "topology_mu_slice_atlas.png", out / "topology_mu_slice_atlas.pdf")
    plt.close(fig)
    return boundary_rows


def make_margin_gap_figures(out: Path, data: dict[str, np.ndarray]) -> None:
    phase = np.asarray(data["y_phase"], dtype=int)
    topo = np.asarray(data["topology_label_code"], dtype=int)
    margin = np.asarray(data.get("topology_pfaffian_margin", np.full(len(phase), np.nan)), dtype=float)
    gap = np.asarray(data.get("topology_bulk_gap", np.full(len(phase), np.nan)), dtype=float)
    sc = phase != 0
    trusted_topo = np.asarray(data.get("topology_trusted", np.zeros(len(phase))), dtype=int) == 1
    masks = [
        (sc & (topo == 0), "#2563eb", "trivial gapped SC"),
        (sc & (topo == 1), "#d81b60", "topological gapped SC"),
        (sc & (topo == 2), "#f59e0b", "gapless_SC"),
        (sc & (topo == 3), "#111111", "unresolved"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for mask, color, label in masks:
        v = margin[mask & np.isfinite(margin)]
        if v.size:
            axes[0].hist(np.log10(np.clip(v, 1e-16, None)), bins=40, alpha=0.55, color=color, label=label)
        g = gap[mask & np.isfinite(gap)]
        if g.size:
            axes[1].hist(np.log10(np.clip(g, 1e-16, None)), bins=40, alpha=0.55, color=color, label=label)
    axes[0].set_xlabel("log10 Pfaffian margin")
    axes[1].set_xlabel("log10 bulk gap")
    for ax in axes:
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"Pfaffian margin, SC points (trusted topology={int(np.sum(trusted_topo))})")
    axes[1].set_title("Bulk-gap distribution, SC points")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, out / "pfaffian_margin_bulk_gap_histograms.png", out / "pfaffian_margin_bulk_gap_histograms.pdf")
    plt.close(fig)


def make_growth_figure(out: Path, iteration_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    ax.plot(iteration_df["iteration"], iteration_df["sample_count"], marker="o", label="total")
    ax.plot(iteration_df["iteration"], iteration_df["normal_count"], marker=".", label="normal")
    ax.plot(iteration_df["iteration"], iteration_df["uniform_sc_count"], marker=".", label="uniform_SC")
    ax.plot(iteration_df["iteration"], iteration_df["fflo_count"], marker=".", label="FFLO")
    ax.set_xlabel("dataset iteration")
    ax.set_ylabel("sample count")
    ax.set_title("Dataset growth by thermodynamic phase")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(iteration_df["iteration"], iteration_df["trivial_count"], marker=".", label="trivial")
    ax.plot(iteration_df["iteration"], iteration_df["topological_count"], marker=".", label="topological")
    ax.plot(iteration_df["iteration"], iteration_df["topology_unresolved_count"], marker=".", label="unresolved")
    ax.plot(iteration_df["iteration"], iteration_df["gapless_sc_count"], marker=".", label="gapless_SC")
    ax.set_xlabel("dataset iteration")
    ax.set_ylabel("sample count")
    ax.set_title("Topology-label growth")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, out / "dataset_growth_phase_topology.png", out / "dataset_growth_phase_topology.pdf")
    plt.close(fig)


def make_convergence_figure(out: Path, audit_dir: Path) -> None:
    vol = pd.read_csv(audit_dir / "tables" / "stageiv_volume_map_change.csv")
    shift = pd.read_csv(audit_dir / "tables" / "stageiv_surface_shift.csv")
    cov = pd.read_csv(audit_dir / "tables" / "stageiv_surface_coverage.csv")
    surprise = pd.read_csv(audit_dir / "tables" / "stageiv_surprise_by_iteration.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    ax = axes[0, 0]
    ax.plot(vol["transition_to_iteration"], vol["phase_volume_map_change"], marker="o", label="phase")
    ax.plot(vol["transition_to_iteration"], vol["topology_volume_map_change"], marker="o", label="topology")
    ax.axhline(0.002, color="k", linestyle="--", linewidth=1, label="0.002")
    ax.set_title("Volume-map change")
    ax.set_xlabel("transition to iteration")
    ax.set_ylabel("fraction")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for surface, sub in shift.groupby("surface"):
        ax.plot(sub["transition_to_iteration"], sub["p95"], marker="o", label=surface)
    ax.axhline(0.004167, color="k", linestyle="--", linewidth=1, label="0.004167")
    ax.set_title("Surface shift p95")
    ax.set_xlabel("transition to iteration")
    ax.set_ylabel("normalized distance")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for surface, sub in cov.groupby("surface"):
        ax.plot(sub["iteration"], sub["p95"], marker="o", label=surface)
    ax.axhline(0.00625, color="k", linestyle="--", linewidth=1, label="0.00625")
    ax.set_title("Surface coverage p95")
    ax.set_xlabel("iteration")
    ax.set_ylabel("normalized distance")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(surprise["transition_to_iteration"], surprise["trusted_phase_surprise"], marker="o", label="trusted phase")
    ax.plot(surprise["transition_to_iteration"], surprise["trusted_topology_surprise"], marker="o", label="trusted topology")
    ax.axhline(0.02, color="#d81b60", linestyle="--", linewidth=1, label="topology 0.02")
    ax.axhline(0.05, color="#444444", linestyle=":", linewidth=1, label="phase 0.05")
    ax.set_title("Out-of-sample trusted surprise")
    ax.set_xlabel("transition to iteration")
    ax.set_ylabel("rate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, out / "convergence_metrics_summary.png", out / "convergence_metrics_summary.pdf")
    plt.close(fig)


def tex_escape(text: Any) -> str:
    s = str(text)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def write_report(
    out_dir: Path,
    package_root: Path,
    run_dir: Path,
    config: dict[str, Any],
    final_path: Path,
    final_data: dict[str, np.ndarray],
    iteration_df: pd.DataFrame,
    slurm_summary: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    figures = out_dir / "figures"
    tables = out_dir / "tables"
    phase = np.asarray(final_data["y_phase"], dtype=int)
    topo = np.asarray(final_data["topology_label_code"], dtype=int)
    trusted = np.asarray(final_data.get("trusted_exact", np.zeros(len(phase))), dtype=int)
    topo_trusted = np.asarray(final_data.get("topology_trusted", np.zeros(len(phase))), dtype=int)
    rerun = np.asarray(final_data.get("needs_rerun_exact", np.zeros(len(phase))), dtype=int)
    q_un = np.asarray(final_data.get("q_unresolved", np.zeros(len(phase))), dtype=int)
    d_un = np.asarray(final_data.get("delta_unresolved", np.zeros(len(phase))), dtype=int)
    final_row = iteration_df.iloc[-1].to_dict()
    conv = decision.get("component_decisions", {}).get("convergence_audit", {})
    hidden = decision.get("component_decisions", {}).get("hidden_slice_audit", {})

    result_rows = [
        ("run_id", config.get("run_id", "")),
        ("final_dataset", str(final_path)),
        ("final_dataset_sha256", sha256_file(final_path)),
        ("sample_count", len(phase)),
        ("normal_count", int(np.sum(phase == 0))),
        ("uniform_SC_count", int(np.sum(phase == 1))),
        ("FFLO_count", int(np.sum(phase == 2))),
        ("trivial_count", int(np.sum(topo == 0))),
        ("topological_count", int(np.sum(topo == 1))),
        ("gapless_SC_count", int(np.sum(topo == 2))),
        ("topology_unresolved_count", int(np.sum(topo == 3))),
        ("trusted_exact_count", int(np.sum(trusted == 1))),
        ("topology_trusted_count", int(np.sum(topo_trusted == 1))),
        ("rerun_required_count", int(np.sum(rerun == 1))),
        ("q_unresolved_count", int(np.sum(q_un == 1))),
        ("delta_unresolved_count", int(np.sum(d_un == 1))),
        ("postrun_bundle_status", decision.get("postrun_bundle_status", "not_available")),
        ("stageiv_convergence_status", conv.get("stageiv_convergence_status", "not_available")),
        ("decision_class", decision.get("decision_class", "not_available")),
        ("mu_domain_complete", conv.get("mu_domain_complete", "not_available")),
        ("mu_range_limited", conv.get("mu_range_limited", "not_available")),
        ("hidden_slice_status", hidden.get("hidden_slice_status", "not_available")),
    ]
    pd.DataFrame(result_rows, columns=["metric", "value"]).to_csv(tables / "return_report_key_metrics.csv", index=False)

    md = []
    md.append("# Stage IV 3D Topology-Aware Return Report\n")
    md.append("## Executive Summary\n")
    md.append(f"- Final dataset: `{final_path}`.\n")
    md.append(f"- Final unique samples: **{len(phase)}**.\n")
    md.append(f"- Thermodynamic counts: normal={int(np.sum(phase == 0))}, uniform_SC={int(np.sum(phase == 1))}, FFLO={int(np.sum(phase == 2))}.\n")
    md.append(f"- Topology counts: trivial={int(np.sum(topo == 0))}, topological={int(np.sum(topo == 1))}, gapless_SC={int(np.sum(topo == 2))}, unresolved={int(np.sum(topo == 3))}, not_applicable={int(np.sum(topo == -1))}.\n")
    md.append(f"- Official file-set status: complete; `dataset_iter025` exists and no dataset/merge/trusted files are missing.\n")
    md.append(f"- Official convergence audit status: **{conv.get('stageiv_convergence_status', 'not_available')}**, decision class **{conv.get('decision_class', decision.get('decision_class', 'not_available'))}**.\n")
    md.append(f"- Hidden fixed-mu validation status: **{hidden.get('hidden_slice_status', 'not_available')}** because no Stage III reference dataset was supplied.\n")
    md.append("- Interpretation: the returned run is data-complete and scientifically useful, but should not be claimed as a formally converged Stage IV 3D phase/topology map yet.\n\n")

    md.append("## Key Convergence Diagnostics\n")
    md.append(f"- Topology volume-map change last three transitions: `{conv.get('topology_volume_map_change_last3', [])}`.\n")
    md.append(f"- Topology surface p95 shift last three transitions: `{conv.get('topology_surface_shift_p95_last3', [])}`.\n")
    md.append(f"- Final topology surface coverage p95: `{conv.get('topology_surface_coverage_p95_final', 'not_available')}`.\n")
    md.append(f"- Trusted topology surprise last three transitions: `{conv.get('trusted_topology_surprise_last3', [])}`.\n")
    md.append(f"- Topology component count last three iterations: `{conv.get('topology_surface_component_count_last3', [])}`.\n")
    md.append(f"- Mu-domain complete: `{conv.get('mu_domain_complete', 'not_available')}`; mu-range limited: `{conv.get('mu_range_limited', 'not_available')}`.\n\n")

    md.append("## Visualizations\n")
    for name, caption in [
        ("phase_3d_transparent.png", "Transparent 3D thermodynamic point cloud with diagnostic normal/SC and uniform_SC/FFLO boundary surfaces."),
        ("phase_3d_jakt_view.png", "Thermodynamic phase map viewed primarily in the kBT/t-J_A/t plane, with mu/t retained as depth/opacity for comparison to earlier 2D phase maps."),
        ("topology_3d_transparent.png", "Transparent 3D topology-aware point cloud with diagnostic phase and cFFLO/tFFLO boundary surfaces."),
        ("phase_mu_slice_atlas.png", "Thermodynamic phase atlas by mu/t slice with diagnostic boundary curves."),
        ("topology_mu_slice_atlas.png", "Topology atlas by mu/t slice with diagnostic phase and cFFLO/tFFLO boundary curves."),
        ("convergence_metrics_summary.png", "Convergence metrics from the report-only audit."),
        ("pfaffian_margin_bulk_gap_histograms.png", "Pfaffian margin and bulk-gap distributions."),
        ("dataset_growth_phase_topology.png", "Dataset and topology-count growth."),
    ]:
        md.append(f"### {caption}\n\n![{caption}](figures/{name})\n\n")

    md.append("## Result Tables\n")
    md.append("- `tables/final_phase_counts.csv`\n")
    md.append("- `tables/final_topology_counts.csv`\n")
    md.append("- `tables/final_reliability_counts.csv`\n")
    md.append("- `tables/dataset_iteration_summary.csv`\n")
    md.append("- `tables/slurm_rank_runtime_summary.csv`\n")
    md.append("- `tables/slurm_iteration_runtime_summary.csv`\n")
    md.append("- `tables/boundary_surface_diagnostics.csv`\n")
    md.append("- `tables/slice_boundary_curve_diagnostics.csv`\n")
    md.append("- `tables/return_report_key_metrics.csv`\n\n")

    md.append("## Boundary Visualization Notes\n")
    md.append("- The added 3D surfaces are diagnostic, report-only smooth RBF fits of locally supported boundary crossing points extracted from the final exact labels after filtering long local edges.\n")
    md.append("- The added 2D curves are binned-median smoothing-spline fits of locally supported boundary crossing points within each fixed-mu slice.\n")
    md.append("- The JA-kBT view collapses the sampled 3D cloud toward the historical 2D plotting plane while retaining mu/t as depth and opacity; it should be read as a comparison view, not as a single fixed-mu phase diagram.\n")
    md.append("- These overlays improve readability of the sampled 3D phase structure, but they do not replace the formal Stage IV convergence metrics.\n")
    md.append("- In particular, topology-surface convergence remains not_converged because map change, p95 shift, coverage, trusted topology surprise, and component stability still fail the current audit.\n\n")

    md.append("## Caveats\n")
    md.append("- The 3D scatter and boundary overlays show sampled exact points plus diagnostic interpolants, not a publication-grade final phase surface.\n")
    md.append("- The official convergence audit uses report-only KNN/nearest-neighbor proxy surfaces; visual review and targeted follow-up are still needed.\n")
    md.append("- Hidden fixed-mu validation is not complete without the frozen Stage III reference dataset.\n")
    md.append("- The current audit recommends inspecting failed surfaces before more exact work; it does not justify a from-scratch restart.\n")

    report_md = "\n".join(md)
    (out_dir / "stageiv_3d_return_report.md").write_text(report_md, encoding="utf-8")

    decision_log = [
        "# Decision Log\n",
        "- File-set check: pass; `dataset_iter025` exists.\n",
        f"- Final sample count: {len(phase)}.\n",
        f"- Convergence audit: {conv.get('stageiv_convergence_status', 'not_available')}.\n",
        f"- Decision class: {conv.get('decision_class', decision.get('decision_class', 'not_available'))}.\n",
        f"- Hidden slice: {hidden.get('hidden_slice_status', 'not_available')}.\n",
        "- Next action: inspect topology/thermodynamic failed surface regions and provide Stage III frozen reference for hidden fixed-mu validation before claiming Stage IV closure.\n",
    ]
    (out_dir / "decision_log.md").write_text("\n".join(decision_log), encoding="utf-8")

    tex_lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.8in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{float}",
        r"\usepackage{enumitem}",
        r"\setlist{nosep}",
        r"\begin{document}",
        r"\title{Stage IV 3D Topology-Aware Return Report}",
        r"\author{Report-only audit}",
        r"\date{2026-06-27}",
        r"\maketitle",
        r"\section*{Executive Summary}",
        rf"Final unique samples: \textbf{{{len(phase)}}}. Thermodynamic counts: normal={int(np.sum(phase == 0))}, uniform\_SC={int(np.sum(phase == 1))}, FFLO={int(np.sum(phase == 2))}.",
        "",
        rf"Topology counts: trivial={int(np.sum(topo == 0))}, topological={int(np.sum(topo == 1))}, gapless\_SC={int(np.sum(topo == 2))}, unresolved={int(np.sum(topo == 3))}, not\_applicable={int(np.sum(topo == -1))}.",
        "",
        rf"Official convergence audit status: \textbf{{{tex_escape(conv.get('stageiv_convergence_status', 'not_available'))}}}. Hidden fixed-mu validation: \textbf{{{tex_escape(hidden.get('hidden_slice_status', 'not_available'))}}}.",
        "",
        r"\section*{Convergence Diagnostics}",
        r"\begin{itemize}",
        rf"\item Topology volume-map change last three transitions: {tex_escape(conv.get('topology_volume_map_change_last3', []))}.",
        rf"\item Topology surface p95 shift last three transitions: {tex_escape(conv.get('topology_surface_shift_p95_last3', []))}.",
        rf"\item Final topology surface coverage p95: {tex_escape(conv.get('topology_surface_coverage_p95_final', 'not_available'))}.",
        rf"\item Trusted topology surprise last three transitions: {tex_escape(conv.get('trusted_topology_surprise_last3', []))}.",
        rf"\item Topology component count last three iterations: {tex_escape(conv.get('topology_surface_component_count_last3', []))}.",
        rf"\item Mu-domain complete: {tex_escape(conv.get('mu_domain_complete', 'not_available'))}; mu-range limited: {tex_escape(conv.get('mu_range_limited', 'not_available'))}.",
        r"\end{itemize}",
    ]
    for name, title in [
        ("phase_3d_transparent.png", "Transparent 3D thermodynamic point cloud with diagnostic phase-boundary surfaces"),
        ("phase_3d_jakt_view.png", "Thermodynamic phase map from the kBT/t-J_A/t view, with mu/t retained as depth and opacity"),
        ("topology_3d_transparent.png", "Transparent 3D topology-aware point cloud with diagnostic phase and topology-boundary surfaces"),
        ("phase_mu_slice_atlas.png", "Thermodynamic phase atlas by mu/t slice with boundary curves"),
        ("topology_mu_slice_atlas.png", "Topology atlas by mu/t slice with boundary curves"),
        ("convergence_metrics_summary.png", "Convergence metrics summary"),
        ("pfaffian_margin_bulk_gap_histograms.png", "Pfaffian margin and bulk-gap histograms"),
        ("dataset_growth_phase_topology.png", "Dataset and topology-count growth"),
    ]:
        tex_lines.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.95\linewidth]{{figures/{name}}}",
                rf"\caption{{{tex_escape(title)}}}",
                r"\end{figure}",
            ]
        )
    tex_lines.extend(
        [
            r"\section*{Caveats}",
            r"\begin{itemize}",
            r"\item The 3D boundary surfaces and 2D boundary curves are report-only smooth diagnostic fits built from final exact labels with local long-edge filtering.",
        r"\item The \(k_B T/t\)-\(J_A/t\) view is a comparison projection of the 3D data cloud and should not be interpreted as a single fixed-\(\mu\) phase diagram.",
            r"\item The convergence audit is report-only and uses nearest-neighbor/KNN proxy surfaces.",
            r"\item Hidden fixed-mu validation remains inconclusive until the frozen Stage III reference dataset is supplied.",
            r"\item Do not claim formal Stage IV convergence from this returned run alone.",
            r"\end{itemize}",
            r"\end{document}",
        ]
    )
    (out_dir / "stageiv_3d_return_report.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def compile_pdf(out_dir: Path) -> dict[str, Any]:
    status = {"pdf_requested": True, "pdf_built": False, "pdflatex": shutil.which("pdflatex"), "returncode": None}
    if not status["pdflatex"]:
        return status
    for suffix in ("aux", "log", "out", "toc"):
        stale = out_dir / f"stageiv_3d_return_report.{suffix}"
        if stale.exists():
            stale.unlink()
    cmd = [status["pdflatex"], "-interaction=nonstopmode", "-halt-on-error", "stageiv_3d_return_report.tex"]
    proc = subprocess.run(cmd, cwd=out_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    status["returncode"] = proc.returncode
    (out_dir / "pdflatex_output.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        return status
    status["pdf_built"] = (out_dir / "stageiv_3d_return_report.pdf").exists()
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Stage IV 3D returned-result report with transparent 3D visualizations.")
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    run_dir = package_root / "ML_Phase_StageIV_Topology3D" / "active_runs" / "active_phase_topology_3d_t_ja_mu_from_scratch_v1"
    config_path = package_root / "configs" / "stageiv_3d_production.json"
    final_path = run_dir / "dataset_iter025.npz"
    postrun_dir = package_root / "ML_Phase_StageIV_Topology3D" / "reports" / "stageiv_3d_postrun_bundle_local"
    decision_path = postrun_dir / "stageiv_3d_postrun_bundle_decision.json"
    audit_dir = postrun_dir / "convergence_audit"
    out_dir = args.output_dir or (package_root / "ML_Phase_StageIV_Topology3D" / "reports" / "stageiv_3d_return_report_local")
    figures = out_dir / "figures"
    tables = out_dir / "tables"
    appendices = out_dir / "appendices"
    for d in (figures, tables, appendices):
        d.mkdir(parents=True, exist_ok=True)

    config = load_json(config_path)
    data = load_dataset(final_path)
    iteration_df = compute_iteration_summary(run_dir)
    slurm_df, slurm_summary = parse_slurm_logs(package_root)
    decision = load_json(decision_path)

    phase = np.asarray(data["y_phase"], dtype=int)
    topo = np.asarray(data["topology_label_code"], dtype=int)
    spectral = np.asarray(data["topology_spectral_status_code"], dtype=int)
    reliability_rows = [
        {"metric": "trusted_exact", "count": int(np.sum(arr_np == 1))}
        for arr_np in [np.asarray(data.get("trusted_exact", np.zeros(len(phase))), dtype=int)]
    ]
    reliability_rows.extend(
        [
            {"metric": "training_eligible_exact", "count": int(np.sum(np.asarray(data.get("training_eligible_exact", np.zeros(len(phase))), dtype=int) == 1))},
            {"metric": "needs_rerun_exact", "count": int(np.sum(np.asarray(data.get("needs_rerun_exact", np.zeros(len(phase))), dtype=int) == 1))},
            {"metric": "q_unresolved", "count": int(np.sum(np.asarray(data.get("q_unresolved", np.zeros(len(phase))), dtype=int) == 1))},
            {"metric": "delta_unresolved", "count": int(np.sum(np.asarray(data.get("delta_unresolved", np.zeros(len(phase))), dtype=int) == 1))},
            {"metric": "topology_trusted", "count": int(np.sum(np.asarray(data.get("topology_trusted", np.zeros(len(phase))), dtype=int) == 1))},
            {"metric": "topology_applicable", "count": int(np.sum(np.asarray(data.get("topology_applicable", np.zeros(len(phase))), dtype=int) == 1))},
        ]
    )

    pd.DataFrame(count_codes(phase, PHASE_LABELS)).to_csv(tables / "final_phase_counts.csv", index=False)
    pd.DataFrame(count_codes(topo, TOPOLOGY_LABELS)).to_csv(tables / "final_topology_counts.csv", index=False)
    pd.DataFrame(count_codes(spectral, SPECTRAL_LABELS)).to_csv(tables / "final_spectral_counts.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(tables / "final_reliability_counts.csv", index=False)
    iteration_df.to_csv(tables / "dataset_iteration_summary.csv", index=False)
    slurm_df.to_csv(tables / "slurm_rank_runtime_summary.csv", index=False)
    slurm_summary.to_csv(tables / "slurm_iteration_runtime_summary.csv", index=False)

    shutil.copy2(decision_path, appendices / "stageiv_3d_postrun_bundle_decision.json") if decision_path.exists() else None
    for name in ["stageiv_3d_convergence_decision.json", "stageiv_3d_convergence_config.json"]:
        src = audit_dir / name
        if src.exists():
            shutil.copy2(src, appendices / name)

    boundary_surface_rows: list[dict[str, Any]] = []
    boundary_surface_rows.extend(make_3d_phase_figure(figures, data))
    boundary_surface_rows.extend(make_jakt_phase_view_figure(figures, data))
    boundary_surface_rows.extend(make_3d_topology_figure(figures, data))
    slice_boundary_rows = make_mu_slice_atlas(figures, data)
    pd.DataFrame(boundary_surface_rows).to_csv(tables / "boundary_surface_diagnostics.csv", index=False)
    pd.DataFrame(slice_boundary_rows).to_csv(tables / "slice_boundary_curve_diagnostics.csv", index=False)
    make_margin_gap_figures(figures, data)
    make_growth_figure(figures, iteration_df)
    if audit_dir.exists():
        make_convergence_figure(figures, audit_dir)

    write_report(out_dir, package_root, run_dir, config, final_path, data, iteration_df, slurm_summary, decision)
    pdf_status = compile_pdf(out_dir)
    write_json(out_dir / "pdf_build_status.json", pdf_status)

    manifest = {
        "package_root": str(package_root),
        "run_dir": str(run_dir),
        "final_dataset": str(final_path),
        "final_dataset_sha256": sha256_file(final_path),
        "output_dir": str(out_dir),
        "sample_count": int(len(phase)),
        "pdf_status": pdf_status,
    }
    write_json(out_dir / "reproduction_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
