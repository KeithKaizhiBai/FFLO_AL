from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import ActiveLearningConfig
from .dataset_builder import load_flat_dataset
from .labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract exact phase-boundary brackets from an active-learning dataset.")
    p.add_argument("--dataset", type=Path, required=True, help="dataset_iterXXX.npz from active_runs/<run_id>.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for boundary CSV/JSON/figures.")
    p.add_argument("--kt-bin-width", type=float, default=0.005, help="kT bin width used to form local JA cuts.")
    p.add_argument("--max-local-spacing", type=float, default=0.035, help="Normalized spacing above which a boundary is low confidence.")
    p.add_argument(
        "--max-refinement-points",
        type=int,
        default=512,
        help="Legacy argument retained for compatibility; midpoint target generation is disabled.",
    )
    p.add_argument("--output-root", type=Path, default=Path("ML_Phase"), help="Only used for config defaults.")
    return p.parse_args()


def _dataset_frame(dataset_path: Path, cfg: ActiveLearningConfig) -> pd.DataFrame:
    ds = load_flat_dataset(dataset_path)
    records = ds.records
    df = pd.DataFrame(
        {
            "kT": ds.x[:, 0],
            "JA": ds.x[:, 1],
            "delta_opt": ds.y_reg[:, 0],
            "q_opt": ds.y_reg[:, 1],
            "eta": ds.y_reg[:, 2],
            "ic_plus": ds.y_reg[:, 3],
            "ic_minus": ds.y_reg[:, 4],
            "phase_label": ds.y_phase,
            "is_sc": ds.y_phase != PHASE_NORMAL,
            "is_uniform_sc": ds.y_phase == PHASE_UNIFORM_SC,
            "is_fflo": ds.y_phase == PHASE_FFLO,
            "trusted_exact": np.asarray(records.get("trusted_exact", np.ones(ds.x.shape[0])), dtype=bool),
            "training_eligible_exact": np.asarray(
                records.get("training_eligible_exact", np.ones(ds.x.shape[0])), dtype=bool
            ),
            "needs_rerun_exact": np.asarray(records.get("needs_rerun_exact", np.zeros(ds.x.shape[0])), dtype=bool),
            "q_expanded": np.asarray(records.get("q_expanded", np.zeros(ds.x.shape[0])), dtype=bool),
            "q_unresolved": np.asarray(records.get("q_unresolved", np.zeros(ds.x.shape[0])), dtype=bool),
            "delta_refined": np.asarray(records.get("delta_refined", np.zeros(ds.x.shape[0])), dtype=bool),
            "delta_unresolved": np.asarray(records.get("delta_unresolved", np.zeros(ds.x.shape[0])), dtype=bool),
            "delta_boundary_band_normal": np.asarray(
                records.get("delta_boundary_band_normal", np.zeros(ds.x.shape[0])), dtype=bool
            ),
            "exact_status_code": np.asarray(records.get("exact_status_code", np.zeros(ds.x.shape[0])), dtype=np.int64),
            "positive_delta_gap": np.asarray(records.get("positive_delta_gap", np.full(ds.x.shape[0], np.nan))),
        }
    )
    df = df[np.isfinite(df["kT"]) & np.isfinite(df["JA"])].copy()
    df["sc_value"] = df["delta_opt"] - float(cfg.delta_eps)
    df["fflo_value"] = np.abs(df["q_opt"]) - float(cfg.q_eps)
    df["eta_value"] = df["eta"]
    df["strong_diode_value"] = np.abs(df["eta"]) - float(cfg.eta_strong)
    return df


def _normalized_distance(a: pd.Series, b: pd.Series, cfg: ActiveLearningConfig) -> float:
    dk = (float(a["kT"]) - float(b["kT"])) / max(float(cfg.kt_max - cfg.kt_min), 1e-12)
    dj = (float(a["JA"]) - float(b["JA"])) / max(float(cfg.ja_max - cfg.ja_min), 1e-12)
    return float(np.sqrt(dk * dk + dj * dj))


def _crossing_fraction(v0: float, v1: float) -> float:
    if not np.isfinite(v0) or not np.isfinite(v1):
        return 0.5
    denom = abs(v0) + abs(v1)
    if denom <= 0.0:
        return 0.5
    return float(abs(v0) / denom)


def _boundary_confidence(a: pd.Series, b: pd.Series, spacing: float, cfg: ActiveLearningConfig, max_spacing: float) -> tuple[str, str]:
    reasons: list[str] = []
    if bool(a["needs_rerun_exact"]) or bool(b["needs_rerun_exact"]):
        reasons.append("needs_rerun_exact")
    if bool(a["q_unresolved"]) or bool(b["q_unresolved"]):
        reasons.append("q_unresolved")
    if bool(a["delta_unresolved"]) or bool(b["delta_unresolved"]):
        reasons.append("delta_unresolved")
    if bool(a["delta_boundary_band_normal"]) or bool(b["delta_boundary_band_normal"]):
        reasons.append("boundary_band_normal")
    if bool(a["q_expanded"]) or bool(b["q_expanded"]):
        reasons.append("q_expanded_confirmed")
    if bool(a["delta_refined"]) or bool(b["delta_refined"]):
        reasons.append("delta_refined")
    if spacing > max_spacing:
        reasons.append("large_local_spacing")
    if not bool(a["training_eligible_exact"]) or not bool(b["training_eligible_exact"]):
        reasons.append("not_training_eligible")

    severe = {"needs_rerun_exact", "q_unresolved", "not_training_eligible", "large_local_spacing"}
    if any(r in severe for r in reasons):
        return "low", ";".join(reasons)
    if reasons:
        return "medium", ";".join(reasons)
    return "high", "clean_bracket"


def _record_boundary(
    boundary_type: str,
    a: pd.Series,
    b: pd.Series,
    value_key: str,
    cfg: ActiveLearningConfig,
    max_spacing: float,
) -> dict:
    v0 = float(a[value_key])
    v1 = float(b[value_key])
    frac = _crossing_fraction(v0, v1)
    kt_boundary = float(a["kT"]) + frac * (float(b["kT"]) - float(a["kT"]))
    ja_boundary = float(a["JA"]) + frac * (float(b["JA"]) - float(a["JA"]))
    spacing = _normalized_distance(a, b, cfg)
    confidence, reasons = _boundary_confidence(a, b, spacing, cfg, max_spacing)
    return {
        "boundary_type": boundary_type,
        "kT_boundary": kt_boundary,
        "JA_boundary": ja_boundary,
        "kT_left": float(a["kT"]),
        "JA_left": float(a["JA"]),
        "kT_right": float(b["kT"]),
        "JA_right": float(b["JA"]),
        "phase_left": int(a["phase_label"]),
        "phase_right": int(b["phase_label"]),
        "delta_left": float(a["delta_opt"]),
        "delta_right": float(b["delta_opt"]),
        "q_left": float(a["q_opt"]),
        "q_right": float(b["q_opt"]),
        "eta_left": float(a["eta"]),
        "eta_right": float(b["eta"]),
        "value_left": v0,
        "value_right": v1,
        "local_spacing_normalized": spacing,
        "confidence": confidence,
        "risk_reason": reasons,
        "left_needs_rerun": bool(a["needs_rerun_exact"]),
        "right_needs_rerun": bool(b["needs_rerun_exact"]),
        "left_boundary_band": bool(a["delta_boundary_band_normal"]),
        "right_boundary_band": bool(b["delta_boundary_band_normal"]),
        "left_q_expanded": bool(a["q_expanded"]),
        "right_q_expanded": bool(b["q_expanded"]),
        "left_delta_refined": bool(a["delta_refined"]),
        "right_delta_refined": bool(b["delta_refined"]),
    }


def _sign_change(v0: float, v1: float) -> bool:
    if not np.isfinite(v0) or not np.isfinite(v1):
        return False
    return (v0 <= 0.0 <= v1) or (v1 <= 0.0 <= v0)


def _extract_boundaries(df: pd.DataFrame, cfg: ActiveLearningConfig, kt_bin_width: float, max_spacing: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    kt0 = float(df["kT"].min())
    bins = np.floor((df["kT"] - kt0) / max(kt_bin_width, 1e-12)).astype(int)
    work = df.copy()
    work["kt_bin"] = bins
    rows: list[dict] = []

    tests: list[tuple[str, str, Callable[[pd.Series, pd.Series], bool]]] = [
        ("normal_sc", "sc_value", lambda a, b: bool(a["is_sc"]) != bool(b["is_sc"])),
        (
            "uniform_fflo",
            "fflo_value",
            lambda a, b: bool(a["is_sc"]) and bool(b["is_sc"]) and bool(a["is_fflo"]) != bool(b["is_fflo"]),
        ),
        ("eta_zero", "eta_value", lambda a, b: _sign_change(float(a["eta_value"]), float(b["eta_value"]))),
        (
            "strong_diode",
            "strong_diode_value",
            lambda a, b: _sign_change(float(a["strong_diode_value"]), float(b["strong_diode_value"])),
        ),
    ]

    for _, group in work.groupby("kt_bin", sort=True):
        g = group.sort_values(["JA", "kT"]).reset_index(drop=True)
        if g.shape[0] < 2:
            continue
        for i in range(g.shape[0] - 1):
            a = g.iloc[i]
            b = g.iloc[i + 1]
            if float(a["JA"]) == float(b["JA"]) and float(a["kT"]) == float(b["kT"]):
                continue
            for boundary_type, value_key, predicate in tests:
                if predicate(a, b):
                    rows.append(_record_boundary(boundary_type, a, b, value_key, cfg, max_spacing))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(["boundary_type", "kT_boundary", "JA_boundary"]).reset_index(drop=True)
    return out


def _targeted_refinement_points(boundaries: pd.DataFrame, max_points: int) -> pd.DataFrame:
    del boundaries, max_points
    return pd.DataFrame(
        columns=[
            "kT",
            "JA",
            "boundary_type",
            "source_confidence",
            "source_reason",
            "selection_disabled",
        ]
    )


def _write_per_type_csv(boundaries: pd.DataFrame, output_dir: Path) -> None:
    for boundary_type, group in boundaries.groupby("boundary_type"):
        group.to_csv(output_dir / f"{boundary_type}_boundary_segments.csv", index=False)


def _plot_boundaries(df: pd.DataFrame, boundaries: pd.DataFrame, targets: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.4), constrained_layout=True)
    colors = np.where(df["phase_label"].to_numpy() == PHASE_NORMAL, "tab:blue", "tab:cyan")
    colors = np.where(df["phase_label"].to_numpy() == PHASE_FFLO, "tab:orange", colors)
    ax.scatter(df["kT"], df["JA"], s=2, c=colors, alpha=0.18, linewidths=0)
    marker_map = {
        "normal_sc": ("black", "o"),
        "uniform_fflo": ("tab:purple", "s"),
        "eta_zero": ("tab:green", "^"),
        "strong_diode": ("tab:red", "x"),
    }
    for boundary_type, group in boundaries.groupby("boundary_type"):
        color, marker = marker_map.get(boundary_type, ("black", "."))
        ax.scatter(group["kT_boundary"], group["JA_boundary"], s=18, c=color, marker=marker, label=boundary_type)
    if not targets.empty:
        ax.scatter(targets["kT"], targets["JA"], s=36, facecolors="none", edgecolors="black", linewidths=1.0, label="targeted")
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("Extracted Exact Boundary Brackets")
    ax.set_xlim(0.0, max(float(df["kT"].max()), 0.56))
    ax.set_ylim(0.0, max(float(df["JA"].max()), 2.12))
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _summary(df: pd.DataFrame, boundaries: pd.DataFrame, targets: pd.DataFrame, args: argparse.Namespace) -> dict:
    by_type = {}
    by_conf = {}
    if not boundaries.empty:
        by_type = {str(k): int(v) for k, v in boundaries["boundary_type"].value_counts().sort_index().items()}
        by_conf = {str(k): int(v) for k, v in boundaries["confidence"].value_counts().sort_index().items()}
    return {
        "dataset": str(args.dataset),
        "n_exact_points": int(df.shape[0]),
        "phase_counts": {str(int(k)): int(v) for k, v in df["phase_label"].value_counts().sort_index().items()},
        "kt_bin_width": float(args.kt_bin_width),
        "max_local_spacing": float(args.max_local_spacing),
        "n_boundary_segments": int(boundaries.shape[0]),
        "boundary_segments_by_type": by_type,
        "boundary_segments_by_confidence": by_conf,
        "n_targeted_refinement_points": int(targets.shape[0]),
        "target_generation": "disabled",
        "target_generation_reason": "Boundary extraction is diagnostic-only; selected exact calls come from ML acquisition.",
        "outputs": {
            "all_boundary_segments": "all_boundary_segments.csv",
            "targeted_refinement_points": "targeted_refinement_points.csv",
            "boundary_diagnostics": "boundary_diagnostics.png",
        },
    }


def extract_phase_boundaries(args: argparse.Namespace) -> dict:
    cfg = ActiveLearningConfig(output_root=str(args.output_root))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _dataset_frame(args.dataset, cfg)
    boundaries = _extract_boundaries(df, cfg, kt_bin_width=float(args.kt_bin_width), max_spacing=float(args.max_local_spacing))
    targets = _targeted_refinement_points(boundaries, max_points=int(args.max_refinement_points))

    df.to_csv(output_dir / "exact_points_for_boundary_extraction.csv", index=False)
    boundaries.to_csv(output_dir / "all_boundary_segments.csv", index=False)
    targets.to_csv(output_dir / "targeted_refinement_points.csv", index=False)
    if not boundaries.empty:
        _write_per_type_csv(boundaries, output_dir)
    _plot_boundaries(df, boundaries, targets, output_dir / "boundary_diagnostics.png")
    summary = _summary(df, boundaries, targets, args)
    (output_dir / "boundary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    summary = extract_phase_boundaries(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
