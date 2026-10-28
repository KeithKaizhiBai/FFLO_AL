from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback is for minimal Python installs.
    cKDTree = None


RUN_ID = "stagev_acqv2_boundary_support_learned_residual_3d_v1"
OUTPUT_ROOT = "ML_Phase_StageV_AcqV2"
REPORT_NAME = "stagev_acqv2_return_report"

PHASE_NAMES = {0: "normal", 1: "uniform_SC", 2: "FFLO"}
PHASE_COLORS = {0: "#bdbdbd", 1: "#1f77b4", 2: "#d62728"}
TOPOLOGY_NAMES = {-1: "not_applicable", 0: "trivial", 1: "topological", 2: "gapless_SC", 3: "unresolved"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def tex_escape(text: object) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def save_figure(fig: plt.Figure, png: Path, pdf: Path) -> None:
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def numeric_dataset_iters(run_dir: Path) -> list[int]:
    out: list[int] = []
    for p in run_dir.glob("dataset_iter*.npz"):
        stem = p.stem
        try:
            out.append(int(stem.replace("dataset_iter", "")))
        except ValueError:
            pass
    return sorted(out)


def numeric_exact_iters(run_dir: Path) -> list[int]:
    out: list[int] = []
    for p in run_dir.glob("iter???"):
        try:
            out.append(int(p.name.replace("iter", "")))
        except ValueError:
            pass
    return sorted(out)


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def phase_counts(data: dict[str, np.ndarray]) -> dict[str, int]:
    y = np.asarray(data["y_phase"], dtype=int)
    return {name: int(np.sum(y == code)) for code, name in PHASE_NAMES.items()}


def topology_counts(data: dict[str, np.ndarray]) -> dict[str, int]:
    y_phase = np.asarray(data["y_phase"], dtype=int)
    topo = np.asarray(data.get("topology_label_code", np.full(y_phase.shape, -1)), dtype=int)
    out = {
        "sc_trivial": int(np.sum((y_phase > 0) & (topo == 0))),
        "sc_topological": int(np.sum((y_phase > 0) & (topo == 1))),
        "sc_gapless": int(np.sum((y_phase > 0) & (topo == 2))),
        "sc_unresolved": int(np.sum((y_phase > 0) & (topo == 3))),
        "fflo_trivial": int(np.sum((y_phase == 2) & (topo == 0))),
        "fflo_topological": int(np.sum((y_phase == 2) & (topo == 1))),
        "uniform_trivial": int(np.sum((y_phase == 1) & (topo == 0))),
        "uniform_topological": int(np.sum((y_phase == 1) & (topo == 1))),
    }
    return out


def collect_dataset_tables(run_dir: Path, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_phase: list[dict[str, Any]] = []
    rows_topo: list[dict[str, Any]] = []
    rows_integrity: list[dict[str, Any]] = []
    for it in numeric_dataset_iters(run_dir):
        path = run_dir / f"dataset_iter{it:03d}.npz"
        data = load_npz_dict(path)
        n = int(np.asarray(data["x"]).shape[0])
        pc = phase_counts(data)
        tc = topology_counts(data)
        rows_phase.append({"dataset_iteration": it, "samples": n, **pc})
        rows_topo.append({"dataset_iteration": it, **tc})
        rows_integrity.append(
            {
                "dataset_iteration": it,
                "dataset_path": str(path),
                "samples": n,
                "sha256": sha256_file(path),
            }
        )
    phase_df = pd.DataFrame(rows_phase)
    topo_df = pd.DataFrame(rows_topo)
    integrity_df = pd.DataFrame(rows_integrity)
    phase_df.to_csv(tables / "phase_counts_by_iteration.csv", index=False)
    topo_df.to_csv(tables / "topology_counts_by_iteration.csv", index=False)
    integrity_df.to_csv(tables / "dataset_integrity.csv", index=False)
    return phase_df, topo_df, integrity_df


def collect_iteration_tables(run_dir: Path, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reward_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for it in numeric_exact_iters(run_dir):
        iter_dir = run_dir / f"iter{it:03d}"
        reward = read_json(iter_dir / "stagev_reward_update_summary.json")
        if reward:
            reward_rows.append(reward)
        summary = read_json(iter_dir / "stagev_selection_summary.json")
        if summary:
            row = {
                "iteration": it,
                "mode": summary.get("mode"),
                "selected_batch_size": summary.get("selected_batch_size"),
                "dataset_samples": summary.get("dataset_samples"),
                "candidate_count": summary.get("candidate_count"),
                "a0_max": summary.get("a0_max"),
                "lambda_t": summary.get("lambda_t"),
                "mean_selection_probability": summary.get("mean_selection_probability"),
            }
            for key, val in (summary.get("support_counts") or {}).items():
                row[f"support_count_{key}"] = val
            for key, val in (summary.get("candidate_source_counts") or {}).items():
                row[f"candidate_source_{key}"] = val
            selection_rows.append(row)
        meta_path = iter_dir / "selected_points_metadata.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            row = {"iteration": it, "selected_rows": int(len(meta))}
            for col in [
                "A0",
                "final_A",
                "g_theta",
                "lambda_t",
                "selection_probability",
                "nearest_exact_distance",
                "exact_repulsion",
                "A_normal_sc",
                "A_uniform_fflo",
                "A_p0_topology",
                "A_ppi_topology",
                "A_gap_nodal",
                "m_NS",
                "m_UF",
                "m_P0",
                "m_Ppi",
                "m_gap",
                "pred_delta",
                "pred_q",
            ]:
                if col in meta.columns:
                    values = pd.to_numeric(meta[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
                    row[f"{col}_median"] = float(values.median()) if values.notna().any() else np.nan
                    if col.startswith("m_"):
                        row[f"abs_{col}_median"] = float(np.nanmedian(np.abs(values.to_numpy(float)))) if values.notna().any() else np.nan
            if "candidate_source" in meta.columns:
                counts = meta["candidate_source"].astype(str).value_counts()
                for key, val in counts.items():
                    row[f"selected_source_{key}"] = int(val)
            feature_rows.append(row)
    reward_df = pd.DataFrame(reward_rows).sort_values("iteration") if reward_rows else pd.DataFrame()
    selection_df = pd.DataFrame(selection_rows).sort_values("iteration") if selection_rows else pd.DataFrame()
    feature_df = pd.DataFrame(feature_rows).sort_values("iteration") if feature_rows else pd.DataFrame()
    reward_df.to_csv(tables / "reward_learning_by_iteration.csv", index=False)
    selection_df.to_csv(tables / "selection_summary_by_iteration.csv", index=False)
    feature_df.to_csv(tables / "selection_feature_summary_by_iteration.csv", index=False)
    return reward_df, selection_df, feature_df


def normalize_points(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    scale = np.maximum(hi - lo, 1e-12)
    return (points - lo) / scale


def majority_knn_labels(
    train_points_norm: np.ndarray,
    train_labels: np.ndarray,
    probe_points_norm: np.ndarray,
    k: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    k_eff = int(max(1, min(k, len(train_labels))))
    if cKDTree is None:
        labels = np.empty(probe_points_norm.shape[0], dtype=int)
        dists = np.empty(probe_points_norm.shape[0], dtype=float)
        chunk = 512
        for start in range(0, len(labels), chunk):
            stop = min(start + chunk, len(labels))
            d = ((probe_points_norm[start:stop, None, :] - train_points_norm[None, :, :]) ** 2).sum(axis=2)
            idx = np.argpartition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
            dsel = np.take_along_axis(d, idx, axis=1)
            dists[start:stop] = np.sqrt(np.min(dsel, axis=1))
            lab = train_labels[idx]
            for i, row in enumerate(lab):
                labels[start + i] = int(np.bincount(row.astype(int), minlength=4).argmax())
        return labels, dists
    tree = cKDTree(train_points_norm)
    dist, idx = tree.query(probe_points_norm, k=k_eff)
    if k_eff == 1:
        return train_labels[idx].astype(int), np.asarray(dist, dtype=float)
    labels = np.empty(probe_points_norm.shape[0], dtype=int)
    for i, row in enumerate(train_labels[idx]):
        labels[i] = int(np.bincount(row.astype(int), minlength=4).argmax())
    return labels, np.asarray(dist[:, 0], dtype=float)


def compute_phase_map_proxy(run_dir: Path, tables: Path, final_data: dict[str, np.ndarray]) -> pd.DataFrame:
    iters = numeric_dataset_iters(run_dir)
    if len(iters) < 2:
        df = pd.DataFrame()
        df.to_csv(tables / "phase_map_convergence_proxy.csv", index=False)
        return df
    x_final = np.asarray(final_data["x"], dtype=float)
    lo = np.array([0.0, 0.0, -0.5], dtype=float)
    hi = np.array([0.56, 2.12, 1.5], dtype=float)
    rng = np.random.default_rng(20260701)
    n_probe = 24000
    probes = lo + rng.random((n_probe, 3)) * (hi - lo)
    probes_norm = normalize_points(probes, lo, hi)
    rows: list[dict[str, Any]] = []
    prev_labels: np.ndarray | None = None
    prev_dist: np.ndarray | None = None
    support_radius = 0.075
    for it in iters:
        data = load_npz_dict(run_dir / f"dataset_iter{it:03d}.npz")
        x = normalize_points(np.asarray(data["x"], dtype=float), lo, hi)
        y = np.asarray(data["y_phase"], dtype=int)
        if len(y) == 0:
            labels = np.full(n_probe, -1, dtype=int)
            dist = np.full(n_probe, np.inf, dtype=float)
        else:
            labels, dist = majority_knn_labels(x, y, probes_norm, k=7)
        valid = dist <= support_radius
        if prev_labels is None or prev_dist is None:
            rows.append(
                {
                    "dataset_iteration": it,
                    "map_change_common_supported": np.nan,
                    "valid_fraction": float(np.mean(valid)),
                    "newly_valid_fraction": float(np.mean(valid)),
                    "no_longer_valid_fraction": np.nan,
                    "support_radius_normalized": support_radius,
                    "probe_count": n_probe,
                }
            )
        else:
            prev_valid = prev_dist <= support_radius
            common = valid & prev_valid
            changed = float(np.mean(labels[common] != prev_labels[common])) if np.any(common) else np.nan
            rows.append(
                {
                    "dataset_iteration": it,
                    "map_change_common_supported": changed,
                    "valid_fraction": float(np.mean(valid)),
                    "newly_valid_fraction": float(np.mean(valid & ~prev_valid)),
                    "no_longer_valid_fraction": float(np.mean(prev_valid & ~valid)),
                    "support_radius_normalized": support_radius,
                    "probe_count": n_probe,
                }
            )
        prev_labels = labels
        prev_dist = dist
    df = pd.DataFrame(rows)
    df.to_csv(tables / "phase_map_convergence_proxy.csv", index=False)
    return df


def extract_boundary_midpoints(
    data: dict[str, np.ndarray],
    boundary: str,
    max_norm_dist: float = 0.09,
    k: int = 12,
) -> np.ndarray:
    x = np.asarray(data["x"], dtype=float)
    phase = np.asarray(data["y_phase"], dtype=int)
    lo = np.array([0.0, 0.0, -0.5], dtype=float)
    hi = np.array([0.56, 2.12, 1.5], dtype=float)
    xn = normalize_points(x, lo, hi)
    if len(x) < 2 or cKDTree is None:
        return np.empty((0, 3), dtype=float)
    tree = cKDTree(xn)
    dist, idx = tree.query(xn, k=min(k + 1, len(x)))
    pairs: set[tuple[int, int]] = set()
    for i in range(len(x)):
        for d, j in zip(np.atleast_1d(dist[i])[1:], np.atleast_1d(idx[i])[1:]):
            if not np.isfinite(d) or d > max_norm_dist:
                continue
            a, b = int(phase[i]), int(phase[j])
            ok = False
            if boundary == "normal_sc":
                ok = (a == 0 and b > 0) or (b == 0 and a > 0)
            elif boundary == "uniform_fflo":
                ok = {a, b} == {1, 2}
            elif boundary == "topology_p0":
                topo = np.asarray(data.get("topology_label_code", np.full(phase.shape, -1)), dtype=int)
                ok = (a == 2 and b == 2) and ({int(topo[i]), int(topo[j])} == {0, 1})
            if ok:
                pairs.add(tuple(sorted((i, int(j)))))
    if not pairs:
        return np.empty((0, 3), dtype=float)
    mids = np.array([(x[i] + x[j]) * 0.5 for i, j in sorted(pairs)], dtype=float)
    return mids


def make_phase_3d_figure(figures: Path, data: dict[str, np.ndarray]) -> dict[str, int]:
    x = np.asarray(data["x"], dtype=float)
    phase = np.asarray(data["y_phase"], dtype=int)
    ns = extract_boundary_midpoints(data, "normal_sc")
    uf = extract_boundary_midpoints(data, "uniform_fflo")
    fig = plt.figure(figsize=(12.8, 5.6))
    views = [(23, -58, "oblique 3D view"), (88, -90, "JA-kBT primary view")]
    for n, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, n, projection="3d")
        for code, name in PHASE_NAMES.items():
            mask = phase == code
            ax.scatter(x[mask, 0], x[mask, 1], x[mask, 2], s=7, alpha=0.28 if code == 0 else 0.5, c=PHASE_COLORS[code], label=f"{name} ({int(mask.sum())})")
        if len(ns):
            ax.scatter(ns[:, 0], ns[:, 1], ns[:, 2], s=8, c="black", alpha=0.45, label="normal/SC local crossings")
        if len(uf):
            ax.scatter(uf[:, 0], uf[:, 1], uf[:, 2], s=8, c="#7b3294", alpha=0.6, label="uniform/FFLO local crossings")
        ax.set_xlabel("kBT/t")
        ax.set_ylabel("J_A/t")
        ax.set_zlabel("mu/t")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_box_aspect((1.2, 1.1, 0.8))
        if n == 2:
            ax.set_zticks([])
        if n == 1:
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("Stage V thermodynamic phase map from exact points")
    save_figure(fig, figures / "stagev_phase_3d.png", figures / "stagev_phase_3d.pdf")
    return {"normal_sc_crossings": int(len(ns)), "uniform_fflo_crossings": int(len(uf))}


def make_topology_3d_figure(figures: Path, data: dict[str, np.ndarray]) -> dict[str, int]:
    x = np.asarray(data["x"], dtype=float)
    phase = np.asarray(data["y_phase"], dtype=int)
    topo = np.asarray(data.get("topology_label_code", np.full(phase.shape, -1)), dtype=int)
    labels = [
        ("normal", phase == 0, "#cccccc", 0.16),
        ("uniform SC", phase == 1, "#1f77b4", 0.55),
        ("cFFLO / trivial FFLO", (phase == 2) & (topo == 0), "#f28e2b", 0.55),
        ("tFFLO / topological FFLO", (phase == 2) & (topo == 1), "#c51b7d", 0.75),
        ("gapless/unresolved SC", (phase > 0) & ((topo == 2) | (topo == 3)), "#000000", 0.8),
    ]
    tb = extract_boundary_midpoints(data, "topology_p0")
    fig = plt.figure(figsize=(12.8, 5.6))
    views = [(22, -58, "oblique 3D view"), (88, -90, "JA-kBT primary view")]
    for n, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, n, projection="3d")
        for label, mask, color, alpha in labels:
            if np.any(mask):
                ax.scatter(x[mask, 0], x[mask, 1], x[mask, 2], s=8, alpha=alpha, c=color, label=f"{label} ({int(mask.sum())})")
        if len(tb):
            ax.scatter(tb[:, 0], tb[:, 1], tb[:, 2], s=9, c="#4b0082", alpha=0.75, label="local cFFLO/tFFLO crossings")
        ax.set_xlabel("kBT/t")
        ax.set_ylabel("J_A/t")
        ax.set_zlabel("mu/t")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_box_aspect((1.2, 1.1, 0.8))
        if n == 2:
            ax.set_zticks([])
        if n == 1:
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("Stage V topology-aware SC labels from exact points")
    save_figure(fig, figures / "stagev_topology_3d.png", figures / "stagev_topology_3d.pdf")
    return {"topology_crossings": int(len(tb))}


def choose_mu_levels(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(x[:, 2])), float(np.nanmax(x[:, 2]))
    return np.linspace(lo + 0.12 * (hi - lo), hi - 0.08 * (hi - lo), 6)


def make_phase_slice_figure(figures: Path, data: dict[str, np.ndarray]) -> None:
    x = np.asarray(data["x"], dtype=float)
    phase = np.asarray(data["y_phase"], dtype=int)
    ns = extract_boundary_midpoints(data, "normal_sc")
    uf = extract_boundary_midpoints(data, "uniform_fflo")
    levels = choose_mu_levels(x)
    half_width = 0.08
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), sharex=True, sharey=True)
    for ax, mu0 in zip(axes.flat, levels):
        slab = np.abs(x[:, 2] - mu0) <= half_width
        if np.sum(slab) < 70:
            idx = np.argsort(np.abs(x[:, 2] - mu0))[: min(240, len(x))]
            slab = np.zeros(len(x), dtype=bool)
            slab[idx] = True
        for code, name in PHASE_NAMES.items():
            mask = slab & (phase == code)
            ax.scatter(x[mask, 0], x[mask, 1], s=13, alpha=0.58 if code else 0.32, c=PHASE_COLORS[code], label=name)
        if len(ns):
            m = np.abs(ns[:, 2] - mu0) <= half_width
            ax.scatter(ns[m, 0], ns[m, 1], s=9, c="black", alpha=0.55, label="normal/SC")
        if len(uf):
            m = np.abs(uf[:, 2] - mu0) <= half_width
            ax.scatter(uf[m, 0], uf[m, 1], s=9, c="#7b3294", alpha=0.65, label="uniform/FFLO")
        ax.set_title(f"mu/t ~= {mu0:.2f}; n={int(slab.sum())}")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("kBT/t")
        ax.set_ylabel("J_A/t")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Thermodynamic phase slices in mu/t slabs")
    fig.subplots_adjust(bottom=0.12)
    save_figure(fig, figures / "stagev_phase_mu_slices.png", figures / "stagev_phase_mu_slices.pdf")


def make_topology_slice_figure(figures: Path, data: dict[str, np.ndarray]) -> None:
    x = np.asarray(data["x"], dtype=float)
    phase = np.asarray(data["y_phase"], dtype=int)
    topo = np.asarray(data.get("topology_label_code", np.full(phase.shape, -1)), dtype=int)
    tb = extract_boundary_midpoints(data, "topology_p0")
    levels = choose_mu_levels(x)
    half_width = 0.08
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), sharex=True, sharey=True)
    groups = [
        ("normal", phase == 0, "#cccccc", 0.24),
        ("uniform SC", phase == 1, "#1f77b4", 0.65),
        ("cFFLO", (phase == 2) & (topo == 0), "#f28e2b", 0.62),
        ("tFFLO", (phase == 2) & (topo == 1), "#c51b7d", 0.78),
    ]
    for ax, mu0 in zip(axes.flat, levels):
        slab = np.abs(x[:, 2] - mu0) <= half_width
        if np.sum(slab) < 70:
            idx = np.argsort(np.abs(x[:, 2] - mu0))[: min(240, len(x))]
            slab = np.zeros(len(x), dtype=bool)
            slab[idx] = True
        for label, mask0, color, alpha in groups:
            mask = slab & mask0
            if np.any(mask):
                ax.scatter(x[mask, 0], x[mask, 1], s=13, alpha=alpha, c=color, label=label)
        if len(tb):
            m = np.abs(tb[:, 2] - mu0) <= half_width
            ax.scatter(tb[m, 0], tb[m, 1], s=9, c="#4b0082", alpha=0.75, label="local c/t crossing")
        ax.set_title(f"mu/t ~= {mu0:.2f}; n={int(slab.sum())}")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("kBT/t")
        ax.set_ylabel("J_A/t")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Topology-aware SC slices in mu/t slabs")
    fig.subplots_adjust(bottom=0.12)
    save_figure(fig, figures / "stagev_topology_mu_slices.png", figures / "stagev_topology_mu_slices.pdf")


def make_eta_slice_figure(figures: Path, data: dict[str, np.ndarray]) -> None:
    x = np.asarray(data["x"], dtype=float)
    eta = np.asarray(data["y_reg"], dtype=float)[:, 2]
    phase = np.asarray(data["y_phase"], dtype=int)
    levels = choose_mu_levels(x)
    half_width = 0.08
    finite = np.isfinite(eta)
    vmax = float(np.nanpercentile(np.abs(eta[finite]), 98)) if np.any(finite) else 1.0
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), sharex=True, sharey=True)
    for ax, mu0 in zip(axes.flat, levels):
        slab = (np.abs(x[:, 2] - mu0) <= half_width) & (phase > 0) & finite
        if np.sum(slab) < 50:
            idx = np.argsort(np.abs(x[:, 2] - mu0) + 0.1 * (phase == 0))[: min(220, len(x))]
            slab = np.zeros(len(x), dtype=bool)
            slab[idx] = True
            slab &= finite
        sc = ax.scatter(x[slab, 0], x[slab, 1], s=15, c=eta[slab], cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.75)
        ax.set_title(f"mu/t ~= {mu0:.2f}; SC/nearby n={int(slab.sum())}")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("kBT/t")
        ax.set_ylabel("J_A/t")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.82, label="eta")
    fig.suptitle("Exact diode-efficiency eta on mu/t slices")
    save_figure(fig, figures / "stagev_eta_mu_slices.png", figures / "stagev_eta_mu_slices.pdf")


def make_growth_figures(figures: Path, phase_df: pd.DataFrame, topo_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for col, color in [("normal", PHASE_COLORS[0]), ("uniform_SC", PHASE_COLORS[1]), ("FFLO", PHASE_COLORS[2])]:
        ax.plot(phase_df["dataset_iteration"], phase_df[col], label=col, lw=2, color=color)
    ax.set_xlabel("dataset iteration")
    ax.set_ylabel("cumulative exact samples")
    ax.set_title("Stage V thermodynamic phase-count growth")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "stagev_phase_count_growth.png", figures / "stagev_phase_count_growth.pdf")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for col, label, color in [
        ("fflo_trivial", "cFFLO / trivial FFLO", "#f28e2b"),
        ("fflo_topological", "tFFLO / topological FFLO", "#c51b7d"),
        ("uniform_trivial", "uniform trivial", "#1f77b4"),
        ("uniform_topological", "uniform topological", "#4c78a8"),
    ]:
        if col in topo_df:
            ax.plot(topo_df["dataset_iteration"], topo_df[col], label=label, lw=2, color=color)
    ax.set_xlabel("dataset iteration")
    ax.set_ylabel("cumulative exact samples")
    ax.set_title("Stage V topology-label growth")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "stagev_topology_count_growth.png", figures / "stagev_topology_count_growth.pdf")


def make_convergence_figure(figures: Path, proxy_df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8.8, 5.0))
    if not proxy_df.empty:
        ax1.plot(proxy_df["dataset_iteration"], proxy_df["map_change_common_supported"], color="#1f77b4", lw=2, label="kNN phase-map change proxy")
        ax1.axhline(0.002, color="black", ls="--", lw=1.2, label="0.002 reference")
        ax1.set_ylabel("fraction changed")
        ax2 = ax1.twinx()
        ax2.plot(proxy_df["dataset_iteration"], proxy_df["valid_fraction"], color="#d95f02", lw=1.6, alpha=0.85, label="supported probe fraction")
        ax2.set_ylabel("supported probe fraction")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    ax1.set_xlabel("dataset iteration")
    ax1.set_title("Report-only phase-map convergence proxy on fixed probes")
    ax1.grid(True, alpha=0.25)
    save_figure(fig, figures / "stagev_phase_map_change_proxy.png", figures / "stagev_phase_map_change_proxy.pdf")


def make_reward_figure(figures: Path, reward_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    if not reward_df.empty:
        x = reward_df["iteration"]
        axes[0].plot(x, reward_df["lambda_t"], color="#54278f", lw=2, label="lambda_t")
        axes[0].set_ylabel("lambda_t")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.25)
        for col, label, color in [
            ("rank_correlation_a0", "A0 rank corr.", "#1f77b4"),
            ("rank_correlation_model", "learned model rank corr.", "#d62728"),
            ("rank_correlation_delta_vs_a0", "model-A0 rank-corr gain", "#2ca02c"),
        ]:
            if col in reward_df:
                axes[1].plot(x, reward_df[col], lw=2, label=label, color=color)
        axes[1].set_ylabel("rank correlation")
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.25)
    axes[1].set_xlabel("acquisition iteration")
    fig.suptitle("Stage V learned residual acquisition diagnostics")
    save_figure(fig, figures / "stagev_reward_learning.png", figures / "stagev_reward_learning.pdf")


def make_selection_figures(figures: Path, feature_df: pd.DataFrame) -> None:
    if feature_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for col, label, color in [
        ("A_normal_sc_mean", "normal/SC", "black"),
        ("A_uniform_fflo_mean", "uniform/FFLO", "#7b3294"),
        ("A_p0_topology_mean", "P0 topology", "#c51b7d"),
        ("A_ppi_topology_mean", "Ppi topology", "#4c78a8"),
        ("A_gap_nodal_mean", "gap/nodal", "#2ca02c"),
    ]:
        if col in feature_df:
            ax.plot(feature_df["iteration"], feature_df[col], lw=2, label=label, color=color)
    ax.set_xlabel("acquisition iteration")
    ax.set_ylabel("mean selected channel score")
    ax.set_title("Selected-point acquisition channel focus")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "stagev_selection_channel_focus.png", figures / "stagev_selection_channel_focus.pdf")

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for col, label in [
        ("abs_m_NS_median", "|m_NS|"),
        ("abs_m_UF_median", "|m_UF|"),
        ("abs_m_P0_median", "|m_P0|"),
        ("abs_m_Ppi_median", "|m_Ppi|"),
    ]:
        if col in feature_df:
            ax.plot(feature_df["iteration"], feature_df[col], lw=2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("acquisition iteration")
    ax.set_ylabel("median absolute selected margin")
    ax.set_title("Selected feature margins; lower means closer to a learned boundary")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "stagev_selected_margin_features.png", figures / "stagev_selected_margin_features.pdf")


def make_selection_source_figure(figures: Path, feature_df: pd.DataFrame) -> None:
    if feature_df.empty:
        return
    source_cols = [c for c in feature_df.columns if c.startswith("selected_source_")]
    if not source_cols:
        return
    plot_df = feature_df[["iteration", *source_cols]].fillna(0.0).copy()
    totals = plot_df[source_cols].sum(axis=1).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bottom = np.zeros(len(plot_df), dtype=float)
    for col in source_cols:
        frac = (plot_df[col] / totals).fillna(0.0).to_numpy(float)
        ax.fill_between(plot_df["iteration"], bottom, bottom + frac, step="mid", alpha=0.7, label=col.replace("selected_source_", ""))
        bottom += frac
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("acquisition iteration")
    ax.set_ylabel("selected fraction")
    ax.set_title("Selected candidate-source mixture")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(True, alpha=0.25)
    save_figure(fig, figures / "stagev_selection_source_mix.png", figures / "stagev_selection_source_mix.pdf")


def write_summary_tables(
    tables: Path,
    final_iter: int,
    final_data: dict[str, np.ndarray],
    phase_df: pd.DataFrame,
    topo_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    boundary_counts: dict[str, int],
    complete: bool,
) -> pd.DataFrame:
    pc = phase_counts(final_data)
    tc = topology_counts(final_data)
    final_reward = reward_df.iloc[-1].to_dict() if not reward_df.empty else {}
    last_proxy = proxy_df.iloc[-1].to_dict() if not proxy_df.empty else {}
    row = {
        "latest_complete_dataset_iteration": final_iter,
        "completed_exact_iterations_in_dataset": max(0, final_iter),
        "data_complete_to_requested_100": bool(complete),
        "total_samples": int(np.asarray(final_data["x"]).shape[0]),
        **pc,
        **tc,
        **boundary_counts,
        "latest_map_change_proxy": last_proxy.get("map_change_common_supported", np.nan),
        "latest_valid_probe_fraction": last_proxy.get("valid_fraction", np.nan),
        "latest_lambda_t": final_reward.get("lambda_t", np.nan),
        "latest_rank_corr_a0": final_reward.get("rank_correlation_a0", np.nan),
        "latest_rank_corr_model": final_reward.get("rank_correlation_model", np.nan),
        "latest_reward_history_rows": final_reward.get("reward_history_rows", np.nan),
    }
    df = pd.DataFrame([row])
    df.to_csv(tables / "final_summary.csv", index=False)
    return df


def fmt_float(x: Any, ndigits: int = 4) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "NA"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "NA"


def write_markdown_report(
    out: Path,
    final_path: Path,
    final_sha: str,
    final_summary: pd.DataFrame,
    phase_df: pd.DataFrame,
    topo_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    latest_exact_iter: int,
    latest_dataset_iter: int,
    unmerged_exact_iter: int | None,
) -> None:
    row = final_summary.iloc[0].to_dict()
    complete = bool(row["data_complete_to_requested_100"])
    md: list[str] = []
    md.append("# Stage V Acquisition-v2 3D Return Report\n\n")
    md.append("## Executive Summary\n\n")
    md.append(f"- Local available cumulative dataset: `dataset_iter{latest_dataset_iter:03d}.npz` with {int(row['total_samples'])} samples.\n")
    md.append(f"- Completed exact iterations represented in the cumulative dataset: `iter000` through `iter{latest_dataset_iter-1:03d}`.\n")
    if complete:
        md.append("- The local return appears complete through the requested 100-dataset endpoint.\n")
    else:
        md.append("- The local return is not complete to `dataset_iter100`; conclusions below are based only on the available local files.\n")
    if unmerged_exact_iter is not None:
        md.append(f"- `iter{unmerged_exact_iter:03d}` exact shards exist but no merged/trusted append output was found locally, so they are not included in the cumulative dataset.\n")
    md.append(f"- Final available thermodynamic counts: normal={int(row['normal'])}, uniform_SC={int(row['uniform_SC'])}, FFLO={int(row['FFLO'])}.\n")
    md.append(f"- Final available FFLO topology counts: cFFLO/trivial={int(row['fflo_trivial'])}, tFFLO/topological={int(row['fflo_topological'])}; gapless/unresolved SC={int(row['sc_gapless']) + int(row['sc_unresolved'])}.\n")
    md.append(f"- Latest report-only fixed-probe phase-map change proxy: {fmt_float(row['latest_map_change_proxy'], 5)} with supported probe fraction {fmt_float(row['latest_valid_probe_fraction'], 3)}.\n")
    md.append(f"- Learned residual state: lambda_t={fmt_float(row['latest_lambda_t'], 3)}, model rank correlation={fmt_float(row['latest_rank_corr_model'], 3)}, A0 rank correlation={fmt_float(row['latest_rank_corr_a0'], 3)}.\n\n")

    md.append("## Scope and Caveats\n\n")
    md.append("- This report uses only existing Stage V output files; no new exact calculation or Delta-q search was run.\n")
    md.append("- Phase-map convergence is evaluated with exact-data and fixed-probe kNN proxies, not with a formal StopController.\n")
    md.append("- Local crossing markers are diagnostic nearest-neighbor brackets; they are not a final smooth thermodynamic boundary surface.\n")
    md.append("- The current local copy is missing cumulative datasets beyond `dataset_iter093`, despite the expectation of a 100-iteration run. Re-download or collect the final archive before making a final convergence claim.\n\n")

    md.append("## Key Tables\n\n")
    md.append("- `tables/final_summary.csv`\n")
    md.append("- `tables/phase_counts_by_iteration.csv`\n")
    md.append("- `tables/topology_counts_by_iteration.csv`\n")
    md.append("- `tables/phase_map_convergence_proxy.csv`\n")
    md.append("- `tables/reward_learning_by_iteration.csv`\n")
    md.append("- `tables/selection_feature_summary_by_iteration.csv`\n\n")

    figures = [
        ("stagev_phase_3d.png", "3D thermodynamic phase map with local crossing markers"),
        ("stagev_topology_3d.png", "3D topology-aware SC labels"),
        ("stagev_phase_mu_slices.png", "Thermodynamic phase slices in fixed mu/t slabs"),
        ("stagev_topology_mu_slices.png", "Topology-aware SC slices in fixed mu/t slabs"),
        ("stagev_eta_mu_slices.png", "Exact eta feature on mu/t slices"),
        ("stagev_phase_count_growth.png", "Cumulative thermodynamic phase-count growth"),
        ("stagev_topology_count_growth.png", "Cumulative topology-label growth"),
        ("stagev_phase_map_change_proxy.png", "Report-only fixed-probe phase-map convergence proxy"),
        ("stagev_reward_learning.png", "Learned residual acquisition diagnostics"),
        ("stagev_selection_channel_focus.png", "Selected-point acquisition channel focus"),
        ("stagev_selected_margin_features.png", "Selected ML feature margins"),
        ("stagev_selection_source_mix.png", "Selected candidate-source mixture"),
    ]
    md.append("## Figures\n\n")
    for fname, caption in figures:
        md.append(f"### {caption}\n\n")
        md.append(f"![{caption}](figures/{fname})\n\n")

    md.append("## Data Provenance\n\n")
    md.append(f"- Final available dataset file: `{final_path.name}`; full absolute path is recorded in `reproduction_manifest.json`.\n")
    md.append(f"- Final available dataset SHA256: `{final_sha}`\n")
    md.append(f"- Latest exact iteration directory present locally: `iter{latest_exact_iter:03d}`\n")
    md.append(f"- Latest cumulative dataset present locally: `dataset_iter{latest_dataset_iter:03d}`\n")
    md.append("\n")
    (out / f"{REPORT_NAME}.md").write_text("".join(md), encoding="utf-8")


def write_latex_report(out: Path) -> Path:
    md_path = out / f"{REPORT_NAME}.md"
    text = md_path.read_text(encoding="utf-8")
    lines: list[str] = []
    lines.append(r"\documentclass[11pt]{article}" "\n")
    lines.append(r"\usepackage[margin=0.72in]{geometry}" "\n")
    lines.append(r"\usepackage{graphicx}" "\n")
    lines.append(r"\usepackage{hyperref}" "\n")
    lines.append(r"\usepackage{float}" "\n")
    lines.append(r"\usepackage{enumitem}" "\n")
    lines.append(r"\setlist{nosep}" "\n")
    lines.append(r"\begin{document}" "\n")
    lines.append(r"\title{Stage V Acquisition-v2 3D Return Report}" "\n")
    lines.append(r"\author{Report-only local audit}" "\n")
    lines.append(r"\date{2026-07-01}" "\n")
    lines.append(r"\maketitle" "\n")
    in_list = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            if in_list:
                lines.append(r"\end{itemize}" "\n")
                in_list = False
            lines.append(r"\section*{" + tex_escape(line[2:]) + "}\n")
        elif line.startswith("## "):
            if in_list:
                lines.append(r"\end{itemize}" "\n")
                in_list = False
            lines.append(r"\section{" + tex_escape(line[3:]) + "}\n")
        elif line.startswith("### "):
            if in_list:
                lines.append(r"\end{itemize}" "\n")
                in_list = False
            lines.append(r"\subsection{" + tex_escape(line[4:]) + "}\n")
        elif line.startswith("- "):
            if not in_list:
                lines.append(r"\begin{itemize}" "\n")
                in_list = True
            lines.append(r"\item " + tex_escape(line[2:]) + "\n")
        elif line.startswith("!["):
            if in_list:
                lines.append(r"\end{itemize}" "\n")
                in_list = False
            alt = line.split("](", 1)[0][2:]
            path = line.split("](", 1)[1].rstrip(")")
            width = "0.96\\linewidth"
            lines.append(r"\begin{figure}[H]\centering" "\n")
            lines.append(r"\includegraphics[width=" + width + r"]{" + tex_escape(path) + "}\n")
            lines.append(r"\caption{" + tex_escape(alt) + "}\n")
            lines.append(r"\end{figure}" "\n")
        elif line.strip():
            if in_list:
                lines.append(r"\end{itemize}" "\n")
                in_list = False
            lines.append(tex_escape(line) + "\n\n")
    if in_list:
        lines.append(r"\end{itemize}" "\n")
    lines.append(r"\end{document}" "\n")
    tex_path = out / f"{REPORT_NAME}.tex"
    tex_path.write_text("".join(lines), encoding="utf-8")
    return tex_path


def build_pdf(out: Path, tex_path: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=out,
            text=True,
            capture_output=True,
            timeout=240,
        )
    except FileNotFoundError:
        status = {"pdf_requested": True, "pdf_built": False, "error": "pdflatex not found"}
    else:
        (out / "pdflatex_output.log").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
        status = {
            "pdf_requested": True,
            "pdf_built": proc.returncode == 0 and (out / f"{REPORT_NAME}.pdf").exists(),
            "returncode": proc.returncode,
        }
    save_json(out / "pdf_build_status.json", status)
    return status


def write_decision_log(out: Path, final_summary: pd.DataFrame, latest_dataset_iter: int, unmerged_exact_iter: int | None) -> None:
    row = final_summary.iloc[0].to_dict()
    lines = [
        "# Stage V Return Decision Log\n\n",
        f"- Local cumulative data are available through `dataset_iter{latest_dataset_iter:03d}` ({int(row['total_samples'])} samples).\n",
        f"- Final available phase counts: normal={int(row['normal'])}, uniform_SC={int(row['uniform_SC'])}, FFLO={int(row['FFLO'])}.\n",
        f"- Final available FFLO topology counts: cFFLO={int(row['fflo_trivial'])}, tFFLO={int(row['fflo_topological'])}.\n",
        f"- Latest learned-residual diagnostics: lambda_t={fmt_float(row['latest_lambda_t'], 3)}, model rank corr={fmt_float(row['latest_rank_corr_model'], 3)}, A0 rank corr={fmt_float(row['latest_rank_corr_a0'], 3)}.\n",
        f"- Latest fixed-probe phase-map change proxy: {fmt_float(row['latest_map_change_proxy'], 5)}.\n",
    ]
    if unmerged_exact_iter is not None:
        lines.append(f"- Caveat: `iter{unmerged_exact_iter:03d}` exact shards are present but were not merged/appended locally.\n")
    lines.append("- Decision: do not claim formal 100-iteration convergence from this local copy until the missing final cumulative datasets are downloaded or reconstructed.\n")
    (out / "decision_log.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a report-only Stage V acquisition-v2 return report.")
    parser.add_argument("--package-root", type=Path, default=Path("stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc"))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    output_root = package_root / args.output_root
    run_dir = output_root / "active_runs" / args.run_id
    out = output_root / "reports" / "stagev_acqv2_return_report_local"
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    dataset_iters = numeric_dataset_iters(run_dir)
    exact_iters = numeric_exact_iters(run_dir)
    if not dataset_iters:
        raise SystemExit(f"No dataset_iter*.npz files found under {run_dir}")
    latest_dataset_iter = max(dataset_iters)
    latest_exact_iter = max(exact_iters) if exact_iters else -1
    final_path = run_dir / f"dataset_iter{latest_dataset_iter:03d}.npz"
    final_data = load_npz_dict(final_path)
    final_sha = sha256_file(final_path)

    phase_df, topo_df, integrity_df = collect_dataset_tables(run_dir, tables)
    reward_df, selection_df, feature_df = collect_iteration_tables(run_dir, tables)
    proxy_df = compute_phase_map_proxy(run_dir, tables, final_data)
    boundary_counts = {}
    boundary_counts.update(make_phase_3d_figure(figures, final_data))
    boundary_counts.update(make_topology_3d_figure(figures, final_data))
    make_phase_slice_figure(figures, final_data)
    make_topology_slice_figure(figures, final_data)
    make_eta_slice_figure(figures, final_data)
    make_growth_figures(figures, phase_df, topo_df)
    make_convergence_figure(figures, proxy_df)
    make_reward_figure(figures, reward_df)
    make_selection_figures(figures, feature_df)
    make_selection_source_figure(figures, feature_df)

    iter_latest_dir = run_dir / f"iter{latest_dataset_iter:03d}"
    unmerged_exact_iter = None
    if iter_latest_dir.exists() and not (iter_latest_dir / f"exact_merged_iter{latest_dataset_iter:03d}.npz").exists():
        shard_count = len(list(iter_latest_dir.glob("exact_shard_rank*_of*.npz")))
        if shard_count > 0:
            unmerged_exact_iter = latest_dataset_iter

    complete_to_100 = latest_dataset_iter >= 100
    final_summary = write_summary_tables(
        tables,
        latest_dataset_iter,
        final_data,
        phase_df,
        topo_df,
        proxy_df,
        reward_df,
        boundary_counts,
        complete=complete_to_100,
    )
    write_markdown_report(
        out,
        final_path,
        final_sha,
        final_summary,
        phase_df,
        topo_df,
        proxy_df,
        reward_df,
        latest_exact_iter,
        latest_dataset_iter,
        unmerged_exact_iter,
    )
    tex_path = write_latex_report(out)
    pdf_status = build_pdf(out, tex_path)
    write_decision_log(out, final_summary, latest_dataset_iter, unmerged_exact_iter)
    manifest = {
        "package_root": str(package_root),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "run_id": args.run_id,
        "latest_dataset_iteration": latest_dataset_iter,
        "latest_exact_iteration_dir": latest_exact_iter,
        "unmerged_exact_iteration": unmerged_exact_iter,
        "final_dataset": str(final_path),
        "final_dataset_sha256": final_sha,
        "report_dir": str(out),
        "pdf_status": pdf_status,
        "notes": [
            "Report-only analysis; no new exact calculation was run.",
            "Phase-map convergence uses a fixed-probe kNN proxy, not a formal StopController.",
        ],
    }
    save_json(out / "reproduction_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
