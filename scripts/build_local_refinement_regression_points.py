from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_RUN_GLOBS = [
    "hpc_upload_robust_incremental_qwindow_20260602_v3/ML_Phase_512_Speed_20260602/active_runs/active_boundary_discovery_robust_incremental_*_acq_v1",
    "project_history/06_incremental_qwindow/hpc_upload_robust_incremental_qwindow_20260602_v3/ML_Phase_512_Speed_20260602/active_runs/active_boundary_discovery_robust_incremental_*_acq_v1",
]


def _latest_merged_npz(run_dir: Path) -> Path | None:
    merged = sorted(run_dir.glob("iter*/exact_merged_iter*.npz"))
    return merged[-1] if merged else None


def _safe_array(data: dict[str, np.ndarray], key: str, default: float | int | str, n: int) -> np.ndarray:
    if key in data:
        return np.asarray(data[key])
    return np.full(n, default)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _rows_for_run(run_dir: Path, q_eps: float) -> list[dict[str, object]]:
    merged_path = _latest_merged_npz(run_dir)
    if merged_path is None:
        return []
    data = _load_npz(merged_path)
    n = int(np.asarray(data["kT"]).shape[0])
    kT = np.asarray(data["kT"], dtype=float)
    JA = np.asarray(data["JA"], dtype=float)
    phase = _safe_array(data, "phase_candidate", -1, n).astype(int)
    q = _safe_array(data, "q_opt", np.nan, n).astype(float)
    delta = _safe_array(data, "delta_opt", np.nan, n).astype(float)
    deltaf = _safe_array(data, "free_energy_gap_to_normal", np.nan, n).astype(float)
    trusted = _safe_array(data, "trusted_exact", 0, n).astype(bool)
    training = _safe_array(data, "training_eligible_exact", 0, n).astype(bool)
    rerun = _safe_array(data, "rerun_required", 0, n).astype(bool)
    q_expanded = _safe_array(data, "q_expanded", 0, n).astype(bool)
    q_unresolved = _safe_array(data, "q_unresolved", 0, n).astype(bool)
    q_status = _safe_array(data, "q_status", -1, n).astype(int)
    q_edge_initial = _safe_array(data, "qopt_edge_hit_initial", 0, n).astype(bool)
    delta_amb = _safe_array(data, "delta_boundary_ambiguous", 0, n).astype(bool)
    band_normal = _safe_array(data, "delta_boundary_band_normal", 0, n).astype(bool)
    phase_changed = _safe_array(data, "phase_changed_after_q_expansion", 0, n).astype(bool)
    near_deg = _safe_array(data, "near_degenerate_branch_count", 0, n).astype(int)
    boxes = _safe_array(data, "local_boxes_refined_count", -1, n).astype(int)
    candidates = _safe_array(data, "local_minima_detected_count", -1, n).astype(int)
    source_iter = merged_path.parent.name

    out: list[dict[str, object]] = []
    for i in range(n):
        categories: list[str] = []
        if phase[i] == 0 and training[i] and not rerun[i] and q_status[i] == 0:
            categories.append("stable_normal_interior")
        if band_normal[i]:
            categories.append("boundary_band_normal")
        if phase[i] == 1 and trusted[i] and not delta_amb[i] and abs(q[i]) < q_eps:
            categories.append("clean_uniform_sc")
        if phase[i] == 1 and trusted[i] and not delta_amb[i] and abs(q[i]) >= q_eps:
            categories.append("clean_fflo")
        if phase_changed[i] and phase[i] == 1:
            categories.append("previous_normal_to_fflo_correction")
        if q_expanded[i] or q_edge_initial[i]:
            categories.append("q_edge_risk")
        if rerun[i]:
            categories.append("rerun_required")
        if near_deg[i] > 0 or delta_amb[i]:
            categories.append("near_degenerate_or_delta_ambiguous")
        for category in categories:
            out.append(
                {
                    "category": category,
                    "source_run": run_dir.name,
                    "source_iter": source_iter,
                    "source_index": i,
                    "kT": float(kT[i]),
                    "JA": float(JA[i]),
                    "phase_candidate": int(phase[i]),
                    "q_opt": float(q[i]),
                    "delta_opt": float(delta[i]),
                    "DeltaF": float(deltaf[i]),
                    "trusted_exact": int(trusted[i]),
                    "training_eligible_exact": int(training[i]),
                    "rerun_required": int(rerun[i]),
                    "q_expanded": int(q_expanded[i]),
                    "q_unresolved": int(q_unresolved[i]),
                    "delta_boundary_ambiguous": int(delta_amb[i]),
                    "delta_boundary_band_normal": int(band_normal[i]),
                    "phase_changed_after_q_expansion": int(phase_changed[i]),
                    "near_degenerate_branch_count": int(near_deg[i]),
                    "local_boxes_refined_count": int(boxes[i]),
                    "local_minima_detected_count": int(candidates[i]),
                }
            )
    return out


def discover_run_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in DEFAULT_RUN_GLOBS:
        out.extend(sorted(root.glob(pattern)))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in out:
        resolved = path.resolve()
        if resolved not in seen and path.is_dir():
            unique.append(path)
            seen.add(resolved)
    return unique


def build_points(run_dirs: list[Path], output: Path, max_per_category: int, q_eps: float) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for run_dir in run_dirs:
        candidates.extend(_rows_for_run(run_dir, q_eps=q_eps))
    candidates.sort(key=lambda r: (str(r["category"]), str(r["source_run"]), float(r["kT"]), float(r["JA"])))
    counts: dict[str, int] = {}
    seen_coords: set[tuple[float, float, str]] = set()
    selected: list[dict[str, object]] = []
    for row in candidates:
        category = str(row["category"])
        if counts.get(category, 0) >= max_per_category:
            continue
        key = (round(float(row["kT"]), 10), round(float(row["JA"]), 10), category)
        if key in seen_coords:
            continue
        selected.append(row)
        seen_coords.add(key)
        counts[category] = counts.get(category, 0) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "source_run",
        "source_iter",
        "source_index",
        "kT",
        "JA",
        "phase_candidate",
        "q_opt",
        "delta_opt",
        "DeltaF",
        "trusted_exact",
        "training_eligible_exact",
        "rerun_required",
        "q_expanded",
        "q_unresolved",
        "delta_boundary_ambiguous",
        "delta_boundary_band_normal",
        "phase_changed_after_q_expansion",
        "near_degenerate_branch_count",
        "local_boxes_refined_count",
        "local_minima_detected_count",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed-point local-refinement regression point set.")
    parser.add_argument("--run-dir", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/local_refinement_refactor/stage_00_baseline/fixed_point_regression_points.csv"))
    parser.add_argument("--max-per-category", type=int, default=8)
    parser.add_argument("--q-eps", type=float, default=1e-2)
    args = parser.parse_args()
    root = Path.cwd()
    run_dirs = args.run_dir if args.run_dir else discover_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError("No robust-incremental run directories found; pass --run-dir explicitly.")
    rows = build_points(run_dirs, args.output, max_per_category=args.max_per_category, q_eps=args.q_eps)
    print(f"Wrote {len(rows)} regression points: {args.output}")


if __name__ == "__main__":
    main()
