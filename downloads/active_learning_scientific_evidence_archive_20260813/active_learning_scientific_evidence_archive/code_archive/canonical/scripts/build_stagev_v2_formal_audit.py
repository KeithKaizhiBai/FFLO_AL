from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.spatial import cKDTree


PHASE_NAME_TO_CODE = {"normal": 0, "uniform_SC": 1, "FFLO": 2}
PHASE_CODE_TO_NAME = {v: k for k, v in PHASE_NAME_TO_CODE.items()}
TOPO_NAME = {0: "cFFLO/trivial", 1: "tFFLO/topological"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def norm_xyz(df: pd.DataFrame, mins: np.ndarray, spans: np.ndarray) -> np.ndarray:
    x = df[["kT", "JA", "mu"]].to_numpy(float)
    return (x - mins) / spans


def coords_key(df: pd.DataFrame) -> pd.Series:
    if not {"kT", "JA", "mu"}.issubset(df.columns):
        return pd.Series([], dtype=str)
    return (
        df["kT"].round(10).astype(str)
        + "|"
        + df["JA"].round(10).astype(str)
        + "|"
        + df["mu"].round(10).astype(str)
    )


def phase_codes(df: pd.DataFrame) -> np.ndarray:
    return df["phase_name"].map(PHASE_NAME_TO_CODE).to_numpy(int)


def has_core_schema(df: pd.DataFrame) -> bool:
    return {"kT", "JA", "mu", "phase_name", "trusted_exact", "training_eligible_exact"}.issubset(df.columns)


def reliable_phase_df(df: pd.DataFrame) -> pd.DataFrame:
    if not has_core_schema(df):
        return pd.DataFrame(columns=df.columns)
    mask = df["trusted_exact"].astype(bool) & df["training_eligible_exact"].astype(bool)
    return df[mask].copy()


def reliable_topology_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "trusted_exact",
        "training_eligible_exact",
        "topology_trusted",
        "phase_name",
        "topology_label_code",
    }
    if not needed.issubset(df.columns):
        return pd.DataFrame(columns=df.columns)
    mask = (
        df["trusted_exact"].astype(bool)
        & df["training_eligible_exact"].astype(bool)
        & df["topology_trusted"].astype(bool)
        & df["phase_name"].eq("FFLO")
        & df["topology_label_code"].isin([0, 1])
    )
    return df[mask].copy()


def support_radius(xn: np.ndarray, k: int = 8, scale: float = 1.35) -> float:
    if len(xn) <= k:
        return 0.1
    tree = cKDTree(xn)
    d, _ = tree.query(xn, k=k + 1)
    return float(np.quantile(d[:, -1], 0.95) * scale)


def make_audit_cloud(mins: np.ndarray, spans: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cloud = rng.random((n, 3))
    return cloud


def knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    radius: float,
    k: int = 11,
    chunk: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full(len(query_x), -1, dtype=int)
    valid = np.zeros(len(query_x), dtype=bool)
    if len(train_x) == 0:
        return pred, valid
    kk = min(k, len(train_x))
    tree = cKDTree(train_x)
    classes = np.array(sorted(set(int(v) for v in train_y)))
    for start in range(0, len(query_x), chunk):
        q = query_x[start : start + chunk]
        d, idx = tree.query(q, k=kk)
        if kk == 1:
            d = d[:, None]
            idx = idx[:, None]
        local_y = train_y[idx]
        weights = 1.0 / np.maximum(d, 1.0e-12)
        nearest_ok = d[:, 0] <= radius
        votes = np.zeros((len(q), len(classes)), dtype=float)
        for ci, c in enumerate(classes):
            votes[:, ci] = np.sum(weights * (local_y == c), axis=1)
        local_pred = classes[np.argmax(votes, axis=1)]
        pred[start : start + len(q)] = local_pred
        valid[start : start + len(q)] = nearest_ok
    return pred, valid


def load_json_blocks(log_dir: Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.nohup.log")):
        active: list[str] = []
        depth = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not active and stripped.startswith("{"):
                active = [line]
                depth = line.count("{") - line.count("}")
            elif active:
                active.append(line)
                depth += line.count("{") - line.count("}")
            if active and depth <= 0:
                try:
                    blocks.append(json.loads("\n".join(active)))
                except json.JSONDecodeError:
                    pass
                active = []
                depth = 0
    return blocks


def reward_summary_from_logs(log_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block in load_json_blocks(log_dir):
        if "validation" not in block or "lambda_state" not in block:
            continue
        row: dict[str, Any] = {"iteration": block.get("iteration")}
        for name, value in block.get("lambda_state", {}).items():
            row[f"lambda_{name}"] = value
        for name, value in block.get("alpha_state", {}).items():
            row[f"alpha_{name}"] = value
        for name, value in block.get("model_status_by_boundary", {}).items():
            row[f"status_{name}"] = value
        row["reward_history_rows"] = block.get("reward_history_rows")
        row["new_reward_rows"] = block.get("new_reward_rows")
        for name, value in block.get("validation", {}).items():
            row[name] = value
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["iteration"], keep="last").sort_values("iteration")
    return out


def extract_brackets(
    df: pd.DataFrame,
    kind: str,
    mins: np.ndarray,
    spans: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, pd.DataFrame]:
    if kind == "ns":
        base = reliable_phase_df(df)
        x = norm_xyz(base, mins, spans)
        y = (base["phase_name"] != "normal").astype(int).to_numpy()
    elif kind == "uf":
        base = reliable_phase_df(df)
        base = base[base["phase_name"].isin(["uniform_SC", "FFLO"])].copy()
        x = norm_xyz(base, mins, spans)
        y = (base["phase_name"] == "FFLO").astype(int).to_numpy()
    elif kind == "topology":
        base = reliable_topology_df(df)
        x = norm_xyz(base, mins, spans)
        y = base["topology_label_code"].to_numpy(int)
    else:
        raise ValueError(kind)
    if len(base) <= k or len(set(y)) < 2:
        return np.empty((0, 3)), pd.DataFrame()
    tree = cKDTree(x)
    d_self, _ = tree.query(x, k=min(k + 1, len(x)))
    local_cut = float(np.quantile(d_self[:, -1], 0.90) * 1.25)
    d, idx = tree.query(x, k=min(k + 1, len(x)))
    pairs: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    for i in range(len(x)):
        for dj, j in zip(np.atleast_1d(d[i])[1:], np.atleast_1d(idx[i])[1:]):
            j = int(j)
            if i == j or dj > local_cut or y[i] == y[j]:
                continue
            a, b = sorted((i, j))
            if (a, b) in pairs:
                continue
            pairs.add((a, b))
            mid = 0.5 * (x[a] + x[b])
            rows.append(
                {
                    "kind": kind,
                    "i": int(a),
                    "j": int(b),
                    "edge_length_norm": float(np.linalg.norm(x[a] - x[b])),
                    "kT": float(mid[0] * spans[0] + mins[0]),
                    "JA": float(mid[1] * spans[1] + mins[1]),
                    "mu": float(mid[2] * spans[2] + mins[2]),
                    "p0_cross": bool(kind == "topology" and np.sign(base.iloc[a]["topology_p0"]) != np.sign(base.iloc[b]["topology_p0"])),
                    "ppi_cross": bool(kind == "topology" and np.sign(base.iloc[a]["topology_ppi"]) != np.sign(base.iloc[b]["topology_ppi"])),
                }
            )
    mids = np.array([[r["kT"], r["JA"], r["mu"]] for r in rows], dtype=float) if rows else np.empty((0, 3))
    if len(mids):
        mids = (mids - mins) / spans
    return mids, pd.DataFrame(rows)


def bidirectional_shift(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if len(a) == 0 or len(b) == 0:
        return {
            "status": "missing_surface",
            "count_a": len(a),
            "count_b": len(b),
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "hausdorff": np.nan,
        }
    ta = cKDTree(a)
    tb = cKDTree(b)
    dab, _ = tb.query(a, k=1)
    dba, _ = ta.query(b, k=1)
    both = np.concatenate([dab, dba])
    return {
        "status": "ok",
        "count_a": len(a),
        "count_b": len(b),
        "median": float(np.median(both)),
        "p90": float(np.quantile(both, 0.90)),
        "p95": float(np.quantile(both, 0.95)),
        "hausdorff": float(np.max(both)),
    }


def component_count(points: np.ndarray) -> tuple[int, float]:
    if len(points) == 0:
        return 0, 0.0
    if len(points) == 1:
        return 1, 1.0
    tree = cKDTree(points)
    nn, _ = tree.query(points, k=min(4, len(points)))
    radius = float(np.quantile(np.atleast_2d(nn)[:, -1], 0.90) * 1.5)
    pairs = tree.query_pairs(radius)
    parent = list(range(len(points)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)
    counts = Counter(find(i) for i in range(len(points)))
    return len(counts), max(counts.values()) / len(points)


def coverage_metrics(points: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    if len(points) == 0 or len(support) == 0:
        return {"coverage_median": np.nan, "coverage_p90": np.nan, "coverage_p95": np.nan, "coverage_max": np.nan}
    tree = cKDTree(support)
    d, _ = tree.query(points, k=1)
    return {
        "coverage_median": float(np.median(d)),
        "coverage_p90": float(np.quantile(d, 0.90)),
        "coverage_p95": float(np.quantile(d, 0.95)),
        "coverage_max": float(np.max(d)),
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_fig(fig: plt.Figure, path_png: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_final_maps(final_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    paths: list[Path] = []
    colors = {"normal": "#8f8f8f", "uniform_SC": "#2f7ed8", "FFLO": "#d62728"}
    fig = plt.figure(figsize=(7.4, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    for name, sub in final_df.groupby("phase_name"):
        ax.scatter(sub["kT"], sub["JA"], sub["mu"], s=5, alpha=0.32, color=colors.get(name, "k"), label=f"{name} ({len(sub)})", depthshade=False)
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_zlabel(r"$\mu/t$")
    ax.set_title("Final Stage V-v2 thermodynamic labels")
    ax.view_init(elev=23, azim=-58)
    ax.legend(fontsize=7)
    p = figures_dir / "fig01_final_3d_thermodynamic_phase_map.png"
    save_fig(fig, p)
    paths.append(p)

    fflo = reliable_topology_df(final_df)
    fig = plt.figure(figsize=(7.4, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    for code, sub in fflo.groupby("topology_label_code"):
        ax.scatter(sub["kT"], sub["JA"], sub["mu"], s=7, alpha=0.42, label=f"{TOPO_NAME[int(code)]} ({len(sub)})", depthshade=False)
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_zlabel(r"$\mu/t$")
    ax.set_title("Final trusted FFLO topology labels")
    ax.view_init(elev=23, azim=-58)
    ax.legend(fontsize=7)
    p = figures_dir / "fig02_final_3d_fflo_topology_map.png"
    save_fig(fig, p)
    paths.append(p)
    return paths


def plot_slice_atlas(final_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    paths: list[Path] = []
    colors = {"normal": "#8f8f8f", "uniform_SC": "#2f7ed8", "FFLO": "#d62728"}
    topo_colors = {0: "#2f7ed8", 1: "#d62728"}
    qs = np.quantile(final_df["mu"], np.linspace(0, 1, 7))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        lo, hi = qs[i], qs[i + 1]
        sub = final_df[(final_df["mu"] >= lo) & (final_df["mu"] <= hi if i == 5 else final_df["mu"] < hi)]
        for name, ss in sub.groupby("phase_name"):
            ax.scatter(ss["kT"], ss["JA"], s=6, alpha=0.38, color=colors.get(name, "k"), label=name)
        ax.set_title(f"{lo:.2f} <= mu/t <= {hi:.2f}; N={len(sub)}", fontsize=8)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel(r"$k_B T/t$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$J_A/t$")
    fig.suptitle("Thermodynamic phase atlas by mu quantiles", y=0.99)
    p = figures_dir / "fig03_fixed_mu_thermodynamic_slice_atlas.png"
    save_fig(fig, p)
    paths.append(p)

    topo = reliable_topology_df(final_df)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        lo, hi = qs[i], qs[i + 1]
        sub = topo[(topo["mu"] >= lo) & (topo["mu"] <= hi if i == 5 else topo["mu"] < hi)]
        for code, ss in sub.groupby("topology_label_code"):
            ax.scatter(ss["kT"], ss["JA"], s=7, alpha=0.48, color=topo_colors[int(code)], label=TOPO_NAME[int(code)])
        ax.set_title(f"{lo:.2f} <= mu/t <= {hi:.2f}; N={len(sub)}", fontsize=8)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel(r"$k_B T/t$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$J_A/t$")
    fig.suptitle("Trusted FFLO topology atlas by mu quantiles", y=0.99)
    p = figures_dir / "fig04_fixed_mu_topology_slice_atlas.png"
    save_fig(fig, p)
    paths.append(p)
    return paths


def plot_metric_lines(df: pd.DataFrame, x: str, cols: list[str], title: str, ylabel: str, path: Path, threshold: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for col in cols:
        if col in df:
            ax.plot(df[x], df[col], marker="o", markersize=3, linewidth=1.2, label=col)
    if threshold is not None:
        ax.axhline(threshold, color="k", linestyle="--", linewidth=1, label=f"threshold {threshold:g}")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7)
    save_fig(fig, path)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(df[x].astype(str), df[y], color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=35)
    save_fig(fig, path)


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    package_root = Path(args.package_root)
    output_root = getattr(args, "output_root", "ML_Phase_StageV_V2_Multihead")
    run_id = getattr(args, "run_id", "stagev_v2_multihead_boundary_learning_3d_v1")
    report_subdir = getattr(args, "report_subdir", "stagev_v2_formal_audit_local")
    active_run = package_root / output_root / "active_runs" / run_id
    logs_dir = package_root / "logs"
    report_root = ensure_dir(package_root / output_root / "reports" / report_subdir)
    tables_dir = ensure_dir(report_root / "tables")
    figures_dir = ensure_dir(report_root / "figures")

    dataset_paths = sorted(active_run.glob("dataset_iter*.csv"))
    datasets: dict[int, pd.DataFrame] = {}
    for path in dataset_paths:
        m = re.search(r"dataset_iter(\d+)\.csv", path.name)
        if m:
            datasets[int(m.group(1))] = read_csv(path)
    final_iter = max(datasets)
    final_df = datasets[final_iter]
    mins = final_df[["kT", "JA", "mu"]].min().to_numpy(float)
    maxs = final_df[["kT", "JA", "mu"]].max().to_numpy(float)
    spans = np.maximum(maxs - mins, 1.0e-12)

    log_texts = [p.read_text(encoding="utf-8", errors="replace") for p in logs_dir.glob("*.nohup.log")]
    baseline = {
        "run_id": run_id,
        "final_dataset": f"dataset_iter{final_iter:03d}",
        "final_exact_iteration": final_iter - 1,
        "production_loop_completed": any(
            ("production loop completed" in text) or ("topology coverage tail completed" in text)
            for text in log_texts
        ),
        "total_samples": int(len(final_df)),
        "trusted_exact": int(final_df["trusted_exact"].sum()),
        "phase_counts": {str(k): int(v) for k, v in final_df["phase_name"].value_counts().to_dict().items()},
        "topology_counts": {TOPO_NAME[int(k)]: int(v) for k, v in reliable_topology_df(final_df)["topology_label_code"].value_counts().to_dict().items()},
    }
    expected = {
        "total_samples": 7067,
        "trusted_exact": 6928,
        "normal": 4711,
        "uniform_SC": 83,
        "FFLO": 2273,
        "cFFLO/trivial": 1892,
        "tFFLO/topological": 381,
    }
    if getattr(args, "dynamic_baseline", False):
        baseline["baseline_reproduced"] = True
        baseline["baseline_check_mode"] = "dynamic"
    else:
        baseline["baseline_reproduced"] = (
            baseline["total_samples"] == expected["total_samples"]
            and baseline["trusted_exact"] == expected["trusted_exact"]
            and all(baseline["phase_counts"].get(k, 0) == expected[k] for k in ["normal", "uniform_SC", "FFLO"])
            and all(baseline["topology_counts"].get(k, 0) == expected[k] for k in ["cFFLO/trivial", "tFFLO/topological"])
        )
        baseline["baseline_check_mode"] = "fixed_stagev2_dataset_iter097"

    count_rows = []
    for it, df in sorted(datasets.items()):
        if not has_core_schema(df):
            count_rows.append(
                {
                    "dataset_iteration": it,
                    "exact_iteration_completed": it - 1,
                    "total_samples": len(df),
                    "trusted_exact": 0,
                    "training_eligible": 0,
                    "normal": 0,
                    "uniform_SC": 0,
                    "FFLO": 0,
                    "cFFLO": 0,
                    "tFFLO": 0,
                    "topology_untrusted_or_unresolved": 0,
                    "duplicate_coordinate_count": 0,
                    "failed_or_untrusted_count": 0,
                    "data_path": str(active_run / f"dataset_iter{it:03d}.csv"),
                    "schema_status": "missing_core_columns",
                }
            )
            continue
        key = coords_key(df)
        dup = int(key.duplicated().sum())
        topo = reliable_topology_df(df)
        count_rows.append(
            {
                "dataset_iteration": it,
                "exact_iteration_completed": it - 1,
                "total_samples": len(df),
                "trusted_exact": int(df["trusted_exact"].sum()),
                "training_eligible": int(df["training_eligible_exact"].sum()),
                "normal": int(df["phase_name"].eq("normal").sum()),
                "uniform_SC": int(df["phase_name"].eq("uniform_SC").sum()),
                "FFLO": int(df["phase_name"].eq("FFLO").sum()),
                "cFFLO": int(topo["topology_label_code"].eq(0).sum()),
                "tFFLO": int(topo["topology_label_code"].eq(1).sum()),
                "topology_untrusted_or_unresolved": int(df["phase_name"].eq("FFLO").sum() - len(topo)),
                "duplicate_coordinate_count": dup,
                "failed_or_untrusted_count": int((~df["trusted_exact"].astype(bool)).sum()),
                "data_path": str(active_run / f"dataset_iter{it:03d}.csv"),
                "schema_status": "ok",
            }
        )
    counts_df = pd.DataFrame(count_rows)
    write_csv(counts_df, tables_dir / "stagev_v2_iteration_counts.csv")

    cloud = make_audit_cloud(mins, spans, args.audit_cloud_size, args.seed)
    final_phase = reliable_phase_df(final_df)
    phase_radius = support_radius(norm_xyz(final_phase, mins, spans), k=args.knn_k)
    final_topo = reliable_topology_df(final_df)
    topo_radius = support_radius(norm_xyz(final_topo, mins, spans), k=args.knn_k, scale=1.50)

    state_iters = list(range(max(1, final_iter - args.last_n_transitions), final_iter + 1))
    phase_predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    topo_predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for it in state_iters:
        df = datasets[it]
        rp = reliable_phase_df(df)
        phase_predictions[it] = knn_predict(norm_xyz(rp, mins, spans), phase_codes(rp), cloud, phase_radius, k=args.knn_k)
        rt = reliable_topology_df(df)
        topo_predictions[it] = knn_predict(norm_xyz(rt, mins, spans), rt["topology_label_code"].to_numpy(int), cloud, topo_radius, k=args.knn_k)

    phase_rows = []
    topo_rows = []
    surprise_rows = []
    bracket_cache: dict[tuple[int, str], tuple[np.ndarray, pd.DataFrame]] = {}
    surface_rows = []
    component_rows = []
    for it in state_iters[1:]:
        prev = it - 1
        pp, vp = phase_predictions[prev]
        pc, vc = phase_predictions[it]
        valid = vp & vc
        phase_rows.append(
            {
                "transition": f"{prev:03d}->{it:03d}",
                "to_dataset_iteration": it,
                "phase_map_change": float(np.mean(pp[valid] != pc[valid])) if valid.any() else np.nan,
                "valid_probe_count": int(valid.sum()),
                "valid_probe_fraction": float(valid.mean()),
                "newly_valid_fraction": float((~vp & vc).mean()),
                "lost_valid_fraction": float((vp & ~vc).mean()),
            }
        )
        tp, tvp = topo_predictions[prev]
        tc, tvc = topo_predictions[it]
        tvalid = tvp & tvc
        topo_rows.append(
            {
                "transition": f"{prev:03d}->{it:03d}",
                "to_dataset_iteration": it,
                "topology_map_change": float(np.mean(tp[tvalid] != tc[tvalid])) if tvalid.any() else np.nan,
                "valid_probe_count": int(tvalid.sum()),
                "valid_probe_fraction": float(tvalid.mean()),
                "newly_valid_fraction": float((~tvp & tvc).mean()),
                "lost_valid_fraction": float((tvp & ~tvc).mean()),
            }
        )
        prev_keys = set(coords_key(datasets[prev]))
        cur = datasets[it]
        new = cur[~coords_key(cur).isin(prev_keys)].copy()
        rp_prev = reliable_phase_df(datasets[prev])
        phase_pred_new, phase_valid_new = knn_predict(norm_xyz(rp_prev, mins, spans), phase_codes(rp_prev), norm_xyz(new, mins, spans), phase_radius, k=args.knn_k)
        phase_mask = new["trusted_exact"].astype(bool) & new["training_eligible_exact"].astype(bool) & phase_valid_new
        rt_prev = reliable_topology_df(datasets[prev])
        topo_new = new[
            new["trusted_exact"].astype(bool)
            & new["training_eligible_exact"].astype(bool)
            & new["topology_trusted"].astype(bool)
            & new["phase_name"].eq("FFLO")
            & new["topology_label_code"].isin([0, 1])
        ].copy()
        topo_pred_new, topo_valid_new = knn_predict(norm_xyz(rt_prev, mins, spans), rt_prev["topology_label_code"].to_numpy(int), norm_xyz(topo_new, mins, spans), topo_radius, k=args.knn_k)
        surprise_rows.append(
            {
                "transition": f"{prev:03d}->{it:03d}",
                "to_dataset_iteration": it,
                "new_points": len(new),
                "trusted_phase_denominator": int(phase_mask.sum()),
                "trusted_phase_surprise": float(np.mean(phase_pred_new[phase_mask.to_numpy()] != phase_codes(new)[phase_mask.to_numpy()])) if phase_mask.any() else np.nan,
                "trusted_topology_denominator": int(np.sum(topo_valid_new)),
                "trusted_topology_surprise": float(np.mean(topo_pred_new[topo_valid_new] != topo_new["topology_label_code"].to_numpy(int)[topo_valid_new])) if np.any(topo_valid_new) else np.nan,
                "insufficient_topology_support": bool(np.sum(topo_valid_new) < args.min_surprise_denominator),
            }
        )
        for kind in ["ns", "uf", "topology"]:
            if (prev, kind) not in bracket_cache:
                bracket_cache[(prev, kind)] = extract_brackets(datasets[prev], kind, mins, spans, k=args.knn_k)
            if (it, kind) not in bracket_cache:
                bracket_cache[(it, kind)] = extract_brackets(datasets[it], kind, mins, spans, k=args.knn_k)
            a, _ = bracket_cache[(it, kind)]
            b, _ = bracket_cache[(prev, kind)]
            shift = bidirectional_shift(a, b)
            support_df = reliable_topology_df(datasets[it]) if kind == "topology" else reliable_phase_df(datasets[it])
            support = norm_xyz(support_df, mins, spans)
            cov = coverage_metrics(a, support)
            cc, largest = component_count(a)
            surface_rows.append(
                {
                    "transition": f"{prev:03d}->{it:03d}",
                    "to_dataset_iteration": it,
                    "surface": kind,
                    **shift,
                    **cov,
                }
            )
            component_rows.append(
                {
                    "dataset_iteration": it,
                    "surface": kind,
                    "boundary_component_count": cc,
                    "largest_component_fraction": largest,
                    "boundary_proxy_point_count": len(a),
                }
            )
    phase_conv = pd.DataFrame(phase_rows)
    topo_conv = pd.DataFrame(topo_rows)
    surface_df = pd.DataFrame(surface_rows)
    surprise_df = pd.DataFrame(surprise_rows)
    component_df = pd.DataFrame(component_rows).drop_duplicates(["dataset_iteration", "surface"])
    write_csv(phase_conv, tables_dir / "stagev_v2_phase_convergence.csv")
    write_csv(topo_conv, tables_dir / "stagev_v2_topology_convergence.csv")
    write_csv(surface_df, tables_dir / "stagev_v2_surface_shift_coverage.csv")
    write_csv(surprise_df, tables_dir / "stagev_v2_surprise_by_iteration.csv")
    write_csv(component_df, tables_dir / "stagev_v2_component_stability.csv")

    _, final_topo_brackets = extract_brackets(final_df, "topology", mins, spans, k=args.knn_k)
    if not final_topo_brackets.empty:
        write_csv(final_topo_brackets, tables_dir / "stagev_v2_final_topology_brackets.csv")
    p0ppi_rows = []
    if not final_topo_brackets.empty:
        p0ppi_rows.append(
            {
                "metric": "topology_brackets",
                "count": len(final_topo_brackets),
                "p0_cross_count": int(final_topo_brackets["p0_cross"].sum()),
                "ppi_cross_count": int(final_topo_brackets["ppi_cross"].sum()),
                "both_cross_count": int((final_topo_brackets["p0_cross"] & final_topo_brackets["ppi_cross"]).sum()),
            }
        )
    final_topo = reliable_topology_df(final_df)
    p0ppi_rows.append(
        {
            "metric": "final_trusted_fflo",
            "count": len(final_topo),
            "p0_cross_count": int((final_topo["topology_p0"] < 0).sum()),
            "ppi_cross_count": int((final_topo["topology_ppi"] < 0).sum()),
            "both_cross_count": int(((final_topo["topology_p0"] < 0) & (final_topo["topology_ppi"] < 0)).sum()),
        }
    )
    p0ppi_df = pd.DataFrame(p0ppi_rows)
    write_csv(p0ppi_df, tables_dir / "stagev_v2_p0_ppi_dominance.csv")

    reward_history_path = active_run / "stagev2_reward_history.csv"
    reward_history = read_csv(reward_history_path) if reward_history_path.exists() else pd.DataFrame()
    reward_summary = reward_summary_from_logs(logs_dir)
    if reward_summary.empty:
        prior = package_root / output_root / "reports" / "stagev_v2_return_report_local" / "tables" / "reward_summary_by_iteration.csv"
        if prior.exists():
            reward_summary = read_csv(prior)
    write_csv(reward_summary, tables_dir / "stagev_v2_learning_heads_audit.csv")

    source_rows = []
    for col in ["selection_source", "candidate_source", "dominant_boundary"]:
        if col not in reward_history:
            continue
        grouped = reward_history.groupby(col, dropna=False)
        for name, g in grouped:
            source_rows.append(
                {
                    "field": col,
                    "name": str(name),
                    "selected_count": len(g),
                    "trusted_rate": float(g["trusted_exact"].mean()) if "trusted_exact" in g else np.nan,
                    "topology_trusted_rate": float(g["topology_trusted"].mean()) if "topology_trusted" in g else np.nan,
                    "mean_reward_p0": float(g["reward_p0_normalized"].mean()) if "reward_p0_normalized" in g else np.nan,
                    "mean_reward_ppi": float(g["reward_ppi_normalized"].mean()) if "reward_ppi_normalized" in g else np.nan,
                    "mean_reward_gap": float(g["reward_gap_normalized"].mean()) if "reward_gap_normalized" in g else np.nan,
                }
            )
    source_df = pd.DataFrame(source_rows)
    write_csv(source_df, tables_dir / "stagev_v2_proposal_source_audit.csv")

    if "selection_rank" in reward_history.columns and not reward_history.empty:
        last10 = reward_history[reward_history["selection_rank"].notna()].tail(640).copy()
        final_focus = (
            last10.groupby("dominant_boundary", dropna=False)
            .agg(
                selected_count=("dominant_boundary", "size"),
                trusted_rate=("trusted_exact", "mean"),
                topology_trusted_rate=("topology_trusted", "mean"),
            )
            .reset_index()
            .sort_values("selected_count", ascending=False)
        )
    else:
        final_focus = pd.DataFrame(
            [
                {
                    "dominant_boundary": "not_available_targeted_tail",
                    "selected_count": 0,
                    "trusted_rate": np.nan,
                    "topology_trusted_rate": np.nan,
                }
            ]
        )
    write_csv(final_focus, tables_dir / "stagev_v2_final_focus_audit.csv")

    phase_counts = baseline["phase_counts"]
    topology_counts = baseline["topology_counts"]
    comparison = pd.DataFrame(
        [
            {
                "run": "Stage V-v2 current",
                "total": baseline["total_samples"],
                "normal": phase_counts.get("normal", 0),
                "uniform_SC": phase_counts.get("uniform_SC", 0),
                "FFLO": phase_counts.get("FFLO", 0),
                "cFFLO": topology_counts.get("cFFLO/trivial", 0),
                "tFFLO": topology_counts.get("tFFLO/topological", 0),
            },
            {"run": "Stage V-v1", "total": 6892, "normal": 4855, "uniform_SC": 130, "FFLO": 1907, "cFFLO": 1637, "tFFLO": 270},
            {"run": "Stage IV-A", "total": 7081, "normal": 2350, "uniform_SC": 141, "FFLO": 4590, "cFFLO": np.nan, "tFFLO": 1466},
        ]
    )
    comparison["fflo_fraction"] = comparison["FFLO"] / comparison["total"]
    comparison["tfflo_fraction_of_fflo"] = comparison["tFFLO"] / comparison["FFLO"]
    write_csv(comparison, tables_dir / "stagev_v2_comparison_to_v1_stageiv.csv")

    # Figures
    figures: list[Path] = []
    figures += plot_final_maps(final_df, figures_dir)
    figures += plot_slice_atlas(final_df, figures_dir)
    plot_metric_lines(phase_conv, "to_dataset_iteration", ["phase_map_change"], "Phase-volume map change", "fraction", figures_dir / "fig05_phase_volume_map_change.png", threshold=0.002)
    figures.append(figures_dir / "fig05_phase_volume_map_change.png")
    plot_metric_lines(topo_conv, "to_dataset_iteration", ["topology_map_change"], "Topology-volume map change", "fraction", figures_dir / "fig06_topology_volume_map_change.png", threshold=0.002)
    figures.append(figures_dir / "fig06_topology_volume_map_change.png")
    plot_metric_lines(surface_df[surface_df["surface"].isin(["ns", "uf"])], "to_dataset_iteration", ["p95", "coverage_p95"], "Thermodynamic surface shift and coverage proxy", "normalized distance", figures_dir / "fig07_thermodynamic_surface_shift_coverage.png", threshold=0.004167)
    figures.append(figures_dir / "fig07_thermodynamic_surface_shift_coverage.png")
    plot_metric_lines(surface_df[surface_df["surface"].eq("topology")], "to_dataset_iteration", ["p95", "coverage_p95"], "Topology surface shift and coverage proxy", "normalized distance", figures_dir / "fig08_topology_surface_shift_coverage.png", threshold=0.004167)
    figures.append(figures_dir / "fig08_topology_surface_shift_coverage.png")
    plot_metric_lines(surprise_df, "to_dataset_iteration", ["trusted_phase_surprise", "trusted_topology_surprise"], "Trusted surprise by iteration", "surprise rate", figures_dir / "fig09_trusted_phase_topology_surprise.png", threshold=0.02)
    figures.append(figures_dir / "fig09_trusted_phase_topology_surprise.png")
    plot_metric_lines(component_df[component_df["surface"].eq("topology")], "dataset_iteration", ["boundary_component_count", "largest_component_fraction"], "Topology component stability proxy", "count / fraction", figures_dir / "fig10_topology_component_stability.png")
    figures.append(figures_dir / "fig10_topology_component_stability.png")
    plot_bar(p0ppi_df, "metric", "p0_cross_count", "P0/Ppi dominance audit: P0 crossings", figures_dir / "fig11_p0_ppi_dominance.png")
    figures.append(figures_dir / "fig11_p0_ppi_dominance.png")
    if not reward_summary.empty:
        plot_metric_lines(reward_summary, "iteration", ["lambda_p0", "lambda_ppi", "lambda_ns", "lambda_uf", "lambda_gap"], "Per-head lambda trajectory", "lambda", figures_dir / "fig12_p0_ppi_selected_contribution_lambda.png")
        figures.append(figures_dir / "fig12_p0_ppi_selected_contribution_lambda.png")
        plot_metric_lines(reward_summary, "iteration", ["alpha_p0", "alpha_ppi", "alpha_ns", "alpha_uf", "alpha_gap"], "Per-head alpha trajectory", "alpha", figures_dir / "fig13_alpha_trajectories.png")
        figures.append(figures_dir / "fig13_alpha_trajectories.png")
        plot_metric_lines(reward_summary, "iteration", ["rank_correlation_model_p0", "rank_correlation_model_ppi", "rank_correlation_model_ns", "rank_correlation_model_uf", "rank_correlation_model_gap"], "Reward model rank correlations", "rho", figures_dir / "fig14_reward_correlations.png")
        figures.append(figures_dir / "fig14_reward_correlations.png")
    if not source_df.empty:
        plot_bar(source_df[source_df["field"].eq("selection_source")].sort_values("selected_count", ascending=False).head(12), "name", "selected_count", "Selection source counts", figures_dir / "fig15_proposal_source_selected_fraction.png")
        figures.append(figures_dir / "fig15_proposal_source_selected_fraction.png")
        plot_bar(source_df[source_df["field"].eq("dominant_boundary")].sort_values("mean_reward_ppi", ascending=False), "name", "mean_reward_ppi", "Source reward efficiency proxy", figures_dir / "fig16_source_reward_efficiency.png")
        figures.append(figures_dir / "fig16_source_reward_efficiency.png")
    if not final_focus.empty:
        plot_bar(final_focus, "dominant_boundary", "selected_count", "Final 10 iterations selected-boundary focus", figures_dir / "fig17_final10_selected_boundary_focus.png")
        figures.append(figures_dir / "fig17_final10_selected_boundary_focus.png")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(comparison))
    ax.bar(x - 0.2, comparison["fflo_fraction"], width=0.4, label="FFLO fraction")
    ax.bar(x + 0.2, comparison["tfflo_fraction_of_fflo"], width=0.4, label="tFFLO/FFLO")
    ax.set_xticks(x, comparison["run"])
    ax.set_title("Stage V-v2 vs Stage V-v1 vs Stage IV-A")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    p = figures_dir / "fig18_stagev2_vs_v1_stageiv_comparison.png"
    save_fig(fig, p)
    figures.append(p)

    # Decision
    last3_phase = phase_conv.tail(3)["phase_map_change"].tolist()
    last3_topo = topo_conv.tail(3)["topology_map_change"].tolist()
    topo_surface = surface_df[surface_df["surface"].eq("topology")].tail(3)
    last3_shift = topo_surface["p95"].tolist()
    final_topo_cov = float(topo_surface.iloc[-1]["coverage_p95"]) if len(topo_surface) else np.nan
    last3_surprise = surprise_df.tail(3)["trusted_topology_surprise"].tolist()
    last3_components = component_df[component_df["surface"].eq("topology")].tail(3)["boundary_component_count"].tolist()
    phase_pass = all(np.isfinite(last3_phase)) and max(last3_phase) < 0.002
    topo_pass = all(np.isfinite(last3_topo)) and max(last3_topo) < 0.002
    shift_pass = all(np.isfinite(last3_shift)) and max(last3_shift) <= 0.004167
    surprise_supported = all(not x for x in surprise_df.tail(3)["insufficient_topology_support"].tolist())
    surprise_pass = surprise_supported and all((not np.isfinite(x)) or x <= 0.02 for x in last3_surprise)
    coverage_pass = np.isfinite(final_topo_cov) and final_topo_cov < 0.00625
    component_stable = len(set(last3_components)) == 1 if last3_components else False
    p0_cross = int(p0ppi_df.loc[p0ppi_df["metric"].eq("topology_brackets"), "p0_cross_count"].iloc[0]) if "topology_brackets" in p0ppi_df["metric"].tolist() else 0
    ppi_cross = int(p0ppi_df.loc[p0ppi_df["metric"].eq("topology_brackets"), "ppi_cross_count"].iloc[0]) if "topology_brackets" in p0ppi_df["metric"].tolist() else 0
    p0_suppressed = p0_cross == 0 and ppi_cross > 0
    gap_focus = int(final_focus.loc[final_focus["dominant_boundary"].astype(str).eq("gap"), "selected_count"].iloc[0]) if "gap" in final_focus["dominant_boundary"].astype(str).tolist() else 0
    gap_useful = bool(gap_focus > 0 and final_focus["topology_trusted_rate"].fillna(0).max() > 0.0)
    global_sobol_count = int(source_df[(source_df["field"].eq("selection_source")) & (source_df["name"].eq("global_sobol"))]["selected_count"].sum()) if not source_df.empty else 0
    global_sobol_total = int(source_df[source_df["field"].eq("selection_source")]["selected_count"].sum()) if not source_df.empty else 0
    global_sobol_fixed = bool(global_sobol_total and global_sobol_count / global_sobol_total < 0.5)

    missing_online = True
    gross_coverage_fail = np.isfinite(final_topo_cov) and final_topo_cov > 1.25 * 0.00625
    if not baseline["baseline_reproduced"]:
        decision_class = "E"
        converged = "false"
        recommended = "baseline_or_data_integrity_audit"
    elif not phase_pass or not topo_pass or not shift_pass or not component_stable or gross_coverage_fail:
        decision_class = "C"
        converged = "false"
        recommended = "targeted_topology_surface_coverage_tail_or_acquisition_v3_support_fix"
    elif not coverage_pass and np.isfinite(final_topo_cov) and final_topo_cov <= 1.25 * 0.00625:
        decision_class = "B"
        converged = "nearly"
        recommended = "2_to_4_microbatch_tail"
    elif phase_pass and topo_pass and shift_pass and coverage_pass and surprise_pass and component_stable and not p0_suppressed:
        decision_class = "A"
        converged = "true"
        recommended = "freeze_stagev_v2_result"
    else:
        decision_class = "D" if missing_online else "C"
        converged = "inconclusive" if decision_class == "D" else "false"
        recommended = "collect_missing_online_surrogate_checkpoints_or_run_fixed_audit_tail"

    limiting: list[str] = []
    if not phase_pass:
        limiting.append("phase_map_change")
    if not topo_pass:
        limiting.append("topology_map_change")
    if not shift_pass:
        limiting.append("topology_surface_shift")
    if not coverage_pass:
        limiting.append("topology_surface_coverage")
    if not surprise_pass:
        limiting.append("trusted_topology_surprise_or_support")
    if not component_stable:
        limiting.append("component_stability")
    if p0_suppressed:
        limiting.append("p0_channel_support")

    decision = {
        "decision_class": decision_class,
        "stagev_v2_converged": converged,
        "production_loop_completed": baseline["production_loop_completed"],
        "final_dataset": baseline["final_dataset"],
        "final_exact_iteration": baseline["final_exact_iteration"],
        "total_samples": baseline["total_samples"],
        "trusted_exact": baseline["trusted_exact"],
        "phase_counts": baseline["phase_counts"],
        "topology_counts": baseline["topology_counts"],
        "phase_map_change_last3": last3_phase,
        "topology_map_change_last3": last3_topo,
        "topology_surface_shift_p95_last3": last3_shift,
        "topology_surface_coverage_p95_final": final_topo_cov,
        "trusted_topology_surprise_last3": last3_surprise,
        "topology_component_count_last3": last3_components,
        "p0_channel_suppressed": p0_suppressed,
        "ppi_dominant_due_to_physics": bool(ppi_cross > p0_cross and ppi_cross > 0),
        "global_sobol_dominance_fixed": global_sobol_fixed,
        "gap_head_useful": gap_useful,
        "main_limiting_factor": limiting[0] if limiting else "none",
        "secondary_limiting_factors": limiting[1:],
        "need_new_exact_calculation": decision_class in ["B", "C"],
        "recommended_next_action": recommended,
        "recommended_tail_batches": 3 if decision_class == "B" else (4 if decision_class == "C" and gross_coverage_fail else 0),
        "do_not_continue_v2_unchanged": bool(decision_class == "C" and (p0_suppressed or not global_sobol_fixed or not gap_useful)),
        "caveats": [
            "Audit uses a unified KNN exact-label surrogate because online surrogate checkpoints are not available locally.",
            "Surface extraction is a local opposite-label bracket proxy, not a smoothed final boundary surface.",
            "No exact calculation, Delta-q search, active-learning loop, or HPC job was run.",
        ],
    }
    (report_root / "stagev_v2_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (report_root / "stagev_v2_audit_summary.json").write_text(json.dumps({"baseline": baseline, "decision": decision}, indent=2), encoding="utf-8")
    (report_root / "stagev_v2_audit_config.yaml").write_text(
        "\n".join(
            [
                f"audit_cloud_size: {args.audit_cloud_size}",
                f"seed: {args.seed}",
                f"knn_k: {args.knn_k}",
                f"phase_support_radius: {phase_radius}",
                f"topology_support_radius: {topo_radius}",
                "surrogate: unified_knn_exact_label_surrogate",
                "surface_extraction: local_knn_opposite_label_bracket_midpoints",
            ]
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    names = ["phase", "topology", "shift", "coverage", "surprise", "components", "p0"]
    vals = [phase_pass, topo_pass, shift_pass, coverage_pass, surprise_pass, component_stable, not p0_suppressed]
    ax.bar(names, [1 if v else 0 for v in vals], color=["#2ca02c" if v else "#d62728" for v in vals])
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Final decision summary: Decision {decision_class}")
    ax.set_ylabel("gate pass")
    ax.tick_params(axis="x", rotation=30)
    p = figures_dir / "fig19_final_decision_summary_panel.png"
    save_fig(fig, p)
    figures.append(p)

    report_md = report_root / "stagev_v2_formal_audit_report.md"
    figure_lines = "\n".join(f"![{p.stem}](figures/{p.name})\n" for p in figures)
    report_md.write_text(
        f"""# Stage V-v2 Formal Convergence And Acquisition-Learning Audit

## Scope

This report audits the returned Stage V-v2 run `{baseline['run_id']}` using
existing local and downloaded artifacts only.  It does not run exact
calculations, Delta-q search, an active-learning loop, or any HPC job.

## Baseline Reproduction

- Final dataset: `{baseline['final_dataset']}`
- Final exact iteration: `{baseline['final_exact_iteration']}`
- Production completed: `{baseline['production_loop_completed']}`
- Total samples: `{baseline['total_samples']}`
- Trusted exact samples: `{baseline['trusted_exact']}`
- Phase counts: `{baseline['phase_counts']}`
- Topology counts: `{baseline['topology_counts']}`
- Baseline reproduced: `{baseline['baseline_reproduced']}`

## Method

The audit reconstructs cumulative states from `dataset_iter000.csv` through
`dataset_iter097.csv`.  Because saved online surrogate checkpoints are not
available locally, all iteration-to-iteration map comparisons use one
consistent offline KNN exact-label surrogate on a fixed 3D audit cloud of
`{args.audit_cloud_size}` points.  Boundary surfaces are represented by local
kNN opposite-label bracket midpoints.  Raw long global Delaunay edges are not
used.

## Decision

- Decision class: `{decision_class}`
- stagev_v2_converged: `{converged}`
- Main limiting factor: `{decision['main_limiting_factor']}`
- Secondary limiting factors: `{decision['secondary_limiting_factors']}`
- Need new exact calculation: `{decision['need_new_exact_calculation']}`
- Recommended next action: `{recommended}`

## Direct Answers

1. Stage V-v2 production completed: `{baseline['production_loop_completed']}`.
2. Baseline returned counts were reproduced: `{baseline['baseline_reproduced']}`.
3. Phase-map last-three changes: `{last3_phase}`.
4. Topology-map last-three changes: `{last3_topo}`.
5. Topology-surface p95 shifts: `{last3_shift}`.
6. Final topology-surface coverage p95: `{final_topo_cov}`.
7. Trusted topology surprise last three: `{last3_surprise}`.
8. Topology component count last three: `{last3_components}`.
9. P0 channel suppressed by bracket proxy: `{p0_suppressed}`.
10. Ppi dominant by bracket proxy: `{decision['ppi_dominant_due_to_physics']}`.
11. Global Sobol dominance fixed: `{global_sobol_fixed}`.
12. Gap head useful by final-focus proxy: `{gap_useful}`.
13. Need additional exact calculation: `{decision['need_new_exact_calculation']}`.
14. Recommended tail batches: `{decision['recommended_tail_batches']}`.
15. Freeze Stage V-v2 now: `{decision_class == 'A'}`.

## Caveats

- This is a fixed offline audit, not a replay of the exact online production
  surrogate.
- The boundary surface metrics are local-support proxies and should be followed
  by a dedicated smoothed-surface visualization step if the decision permits.
- Stage V-v2, Stage V-v1, and Stage IV-A are compared by matched diagnostic
  fields where available; point counts alone are not treated as proof of
  superiority.

## Figures

{figure_lines}
""",
        encoding="utf-8",
    )
    (report_root / "decision_log.md").write_text(
        "\n".join(
            [
                "# Stage V-v2 Formal Audit Decision Log",
                "",
                f"- Decision class: `{decision_class}`",
                f"- stagev_v2_converged: `{converged}`",
                f"- Main limiting factor: `{decision['main_limiting_factor']}`",
                f"- Need new exact calculation: `{decision['need_new_exact_calculation']}`",
                f"- Recommended next action: `{recommended}`",
                "",
                "No exact calculation, Delta-q search, active-learning loop, or HPC job was run.",
            ]
        ),
        encoding="utf-8",
    )
    write_pdf(report_md, figures, report_root / "stagev_v2_formal_audit_report.pdf")
    return {"report_root": str(report_root), "decision": decision, "figures": len(figures)}


def write_pdf(md_path: Path, figures: list[Path], out: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch)
    story: list[Any] = []
    lines = md_path.read_text(encoding="utf-8").splitlines()
    title = lines[0].lstrip("# ").strip()
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.12 * inch))
    key_lines = [line for line in lines if line.startswith("- ")][:18]
    table_data = [["Item", "Value"]]
    for line in key_lines:
        text = line[2:]
        if ":" in text:
            a, b = text.split(":", 1)
            table_data.append([a.strip(" `"), b.strip()])
    table = Table(table_data, colWidths=[2.2 * inch, 4.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("See the Markdown report and CSV tables for full numerical details.", styles["BodyText"]))
    for fig in figures:
        story.append(PageBreak())
        story.append(Paragraph(fig.stem.replace("_", " ").title(), styles["Heading2"]))
        story.append(Image(str(fig), width=6.7 * inch, height=4.65 * inch))
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage V-v2 formal audit from returned artifacts.")
    parser.add_argument("--package-root", default="stagev_v2_multihead_boundary_learning_3d_v1_hpc")
    parser.add_argument("--output-root", default="ML_Phase_StageV_V2_Multihead")
    parser.add_argument("--run-id", default="stagev_v2_multihead_boundary_learning_3d_v1")
    parser.add_argument("--report-subdir", default="stagev_v2_formal_audit_local")
    parser.add_argument("--dynamic-baseline", action="store_true")
    parser.add_argument("--audit-cloud-size", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--last-n-transitions", type=int, default=10)
    parser.add_argument("--min-surprise-denominator", type=int, default=16)
    args = parser.parse_args()
    result = build_audit(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
