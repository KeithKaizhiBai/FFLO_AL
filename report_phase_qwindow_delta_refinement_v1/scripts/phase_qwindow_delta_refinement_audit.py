from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
for path_candidate in [Path.cwd().resolve(), REPO_ROOT]:
    path_text = str(path_candidate)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


RUN_ID_DEFAULT = "active_boundary_discovery_512seed_256x50"
REPORT_NAME_DEFAULT = "report_phase_qwindow_delta_refinement_v1"
PHASE_NORMAL = 0
PHASE_UNIFORM_SC = 1
PHASE_FFLO = 2


def _write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _bool_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(False, index=df.index)
    raw = df[name]
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin(["1", "true", "yes"])


def _coord_key(kbt: float, ja: float) -> str:
    return f"{float(kbt):.8f},{float(ja):.8f}"


def _rank_slice(n: int, rank: int, world_size: int) -> np.ndarray:
    if int(world_size) <= 0:
        raise ValueError("world_size must be positive")
    return np.arange(n, dtype=int)[int(rank) :: int(world_size)]


def _phase_name_from_delta_q(delta: float, q: float, delta_eps: float, q_eps: float, ambiguous: bool = False) -> str:
    if ambiguous:
        return "boundary_ambiguous"
    if not np.isfinite(delta) or not np.isfinite(q):
        return "unknown"
    if float(delta) < float(delta_eps):
        return "normal"
    if abs(float(q)) <= float(q_eps):
        return "uniform_SC"
    return "FFLO"


def _phase_name_from_label(value: object, delta: object = np.nan, q: object = np.nan, delta_eps: float = 1e-3, q_eps: float = 1e-2) -> str:
    if isinstance(value, str) and value:
        return value
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return _phase_name_from_delta_q(float(delta), float(q), delta_eps, q_eps)
    if ivalue == PHASE_NORMAL:
        return "normal"
    if ivalue == PHASE_UNIFORM_SC:
        return "uniform_SC"
    if ivalue == PHASE_FFLO:
        return "FFLO"
    return "unknown"


def _safe_float(row: pd.Series, names: list[str], default: float = float("nan")) -> float:
    for name in names:
        if name in row.index:
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return default


def _local_minima(q: np.ndarray, delta_q: np.ndarray, deltaf_q: np.ndarray, topn: int = 8, energy_window: float = 1e-4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if q.size == 0:
        return pd.DataFrame(rows)
    candidates: list[int] = []
    for i in range(q.size):
        left_ok = i == 0 or deltaf_q[i] <= deltaf_q[i - 1]
        right_ok = i == q.size - 1 or deltaf_q[i] <= deltaf_q[i + 1]
        if left_ok and right_ok and np.isfinite(deltaf_q[i]):
            candidates.append(i)
    if not candidates and np.isfinite(deltaf_q).any():
        candidates = [int(np.nanargmin(deltaf_q))]
    candidates = sorted(set(candidates), key=lambda i: float(deltaf_q[i]))
    global_min = float(np.nanmin(deltaf_q)) if np.isfinite(deltaf_q).any() else float("nan")
    for rank, i in enumerate(candidates[:topn], start=1):
        rows.append(
            {
                "minimum_rank": int(rank),
                "q_local_min": float(q[i]),
                "Delta_local_min": float(delta_q[i]),
                "DeltaF_local_min": float(deltaf_q[i]),
                "energy_above_global": float(deltaf_q[i] - global_min) if np.isfinite(global_min) else float("nan"),
                "within_low_energy_window": bool(np.isfinite(global_min) and (deltaf_q[i] - global_min) <= float(energy_window)),
                "grid_index": int(i),
            }
        )
    return pd.DataFrame(rows)


def _eval_phase_q_scan(kbt: float, ja: float, q_min: float, q_max: float, n_q: int, device: str, delta_eps: float, q_eps: float) -> dict[str, object]:
    import torch
    from eta_phase_diagram_cuda import EtaPhaseConfig, build_q_vec, compute_omega_min_q_batch, maybe_set_linalg_backend

    cfg = EtaPhaseConfig(q_min=float(q_min), q_max=float(q_max), n_q=int(n_q))
    cfg_scaled = cfg.scaled()
    maybe_set_linalg_backend(cfg_scaled)
    dev = torch.device(device)
    q_vec = build_q_vec(cfg_scaled)
    q_vec_t = torch.as_tensor(q_vec, device=dev, dtype=cfg_scaled.dtype)
    k_vec = torch.linspace(-math.pi, math.pi, cfg_scaled.n_k, dtype=cfg_scaled.dtype, device=dev)
    kt_batch = torch.as_tensor([float(kbt)], dtype=cfg_scaled.dtype, device=dev)
    ja_batch = torch.as_tensor([float(ja)], dtype=cfg_scaled.dtype, device=dev)

    omega_t, delta_q_t, q_opt_t, delta_opt_t = compute_omega_min_q_batch(kt_batch, ja_batch, cfg_scaled, k_vec, q_vec_t)
    omega = omega_t.detach().cpu().numpy()[0, 0]
    delta_q = delta_q_t.detach().cpu().numpy()[0, 0]
    q_opt = float(q_opt_t.detach().cpu().numpy()[0, 0])
    delta_opt = float(delta_opt_t.detach().cpu().numpy()[0, 0])

    normal_cfg = EtaPhaseConfig(q_min=float(q_min), q_max=float(q_max), n_q=int(n_q), delta_min=0.0, delta_max=0.0, n_delta=1)
    normal_scaled = normal_cfg.scaled()
    normal_q = build_q_vec(normal_scaled)
    normal_q_t = torch.as_tensor(normal_q, device=dev, dtype=normal_scaled.dtype)
    omega_n_t, _, _, _ = compute_omega_min_q_batch(kt_batch, ja_batch, normal_scaled, k_vec, normal_q_t)
    omega_normal = omega_n_t.detach().cpu().numpy()[0, 0]
    deltaf_q = omega - omega_normal
    idx_q_opt = int(np.nanargmin(np.abs(q_vec - q_opt)))
    deltaf_min = float(np.nanmin(deltaf_q)) if np.isfinite(deltaf_q).any() else float("nan")
    dq = float(np.min(np.diff(q_vec))) if q_vec.size > 1 else float("nan")
    q_edge_margin = float(min(q_opt - float(q_vec[0]), float(q_vec[-1]) - q_opt)) if np.isfinite(q_opt) else float("nan")
    minima = _local_minima(q_vec, delta_q, deltaf_q)
    return {
        "summary": {
            "q_min_new": float(q_vec[0]),
            "q_max_new": float(q_vec[-1]),
            "n_q_new": int(q_vec.size),
            "dq_new": dq,
            "q_opt_new": q_opt,
            "Delta_opt_new": delta_opt,
            "DeltaF_min_new": deltaf_min,
            "phase_label_new": _phase_name_from_delta_q(delta_opt, q_opt, delta_eps, q_eps),
            "q_opt_edge_margin": q_edge_margin,
            "q_opt_edge_hit": bool(np.isfinite(q_edge_margin) and np.isfinite(dq) and q_edge_margin <= 3.0 * dq),
            "idx_q_opt": idx_q_opt,
            "low_energy_local_minima_count": int(minima.shape[0]),
        },
        "branch": {
            "q_grid": q_vec,
            "F_min_q": omega,
            "F_normal_q": omega_normal,
            "DeltaF_q": deltaf_q,
            "Delta_star_q": delta_q,
        },
        "minima": minima,
    }


def _default_config(ml_phase_root: Path, run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "ml_phase_root": str(ml_phase_root),
        "final_dataset": str(ml_phase_root / "active_runs" / run_id / "dataset_iter020.npz"),
        "source_audit_tables": str(ml_phase_root / "reports" / "audit_tables"),
        "qwindow_audit": {
            "q_expansion_levels": [
                {"name": "expand_0p5_width", "side_pad_width_factor": 0.5, "n_q_multiplier_min": 1.5},
                {"name": "expand_1p0_width", "side_pad_width_factor": 1.0, "n_q_multiplier_min": 2.0},
            ],
            "q_max_abs": math.pi,
            "delta_eps": 1.0e-3,
            "q_eps": 1.0e-2,
            "phase_deltaf_tol": 1.0e-8,
            "local_minima_energy_window": 1.0e-4,
        },
        "delta_refinement": {
            "delta_eps_strict": 1.0e-3,
            "q_eps": 1.0e-2,
            "free_energy_ambiguity_tol_strict": 1.0e-8,
            "positive_delta_gap_tol_strict": 1.0e-10,
            "delta_refine_half_width": 0.01,
            "n_delta_refined": 1000,
            "max_delta_refinements": 3,
            "enable_q_expansion": True,
            "q_expand_factor": 2.0,
            "q_expand_pad_steps": 80,
            "max_q_refinements": 4,
            "q_max_abs": math.pi,
        },
        "selection_rules": {
            "high_JA_low_T_normal_sc_boundary": "JA > 1.1 and distance_to_normal_sc_boundary <= 0.025",
            "delta_sensitive": "delta_ambiguous or delta_unresolved or boundary_band_normal or needs_rerun_exact",
            "control_points": "up to 20 trusted high-JA boundary points without ambiguity flags",
        },
        "safety": {
            "audit_only": True,
            "do_not_modify_active_learning": True,
            "do_not_append_training_data": True,
            "do_not_redefine_phase_criterion": True,
        },
    }


def _standardize_input(df: pd.DataFrame, role: str) -> pd.DataFrame:
    out = df.copy()
    out["audit_role"] = role
    if "kBT" not in out.columns and "kT" in out.columns:
        out["kBT"] = out["kT"]
    for col in ["q_min", "q_max", "n_q", "q_opt", "Delta_opt", "DeltaF", "free_energy_gap_to_normal", "phase", "phase_label"]:
        if col not in out.columns:
            out[col] = np.nan
    out["coord_key"] = [_coord_key(k, j) for k, j in zip(out["kBT"], out["JA"])]
    return out


def _select_inputs(ml_phase_root: Path) -> dict[str, pd.DataFrame]:
    tables = ml_phase_root / "reports" / "audit_tables"
    high = _read_csv(tables / "high_JA_boundary_kink_points.csv")
    delta = _read_csv(tables / "delta_ambiguous_points.csv")
    rerun = _read_csv(tables / "rerun_required_points.csv")

    high_boundary = high[(high["JA"] > 1.1) & (high["distance_to_normal_sc_boundary"] <= 0.025)].copy()
    delta_flags = _bool_series(high_boundary, "delta_boundary_ambiguous") | _bool_series(high_boundary, "delta_unresolved") | _bool_series(high_boundary, "delta_boundary_band_normal") | _bool_series(high_boundary, "needs_rerun_exact")
    delta_sensitive = high_boundary[delta_flags].copy()
    qwindow_sensitive = high_boundary.copy()

    delta_high = delta[(delta["JA"] > 1.1) & (delta["distance_to_normal_sc_boundary"] <= 0.035)].copy()
    rerun_high = rerun[(rerun["JA"] > 0.9) & (rerun["distance_to_normal_sc_boundary"] <= 0.04)].copy()
    clean_mask = (
        (high_boundary["JA"] > 1.1)
        & _bool_series(high_boundary, "trusted_exact")
        & ~_bool_series(high_boundary, "delta_boundary_ambiguous")
        & ~_bool_series(high_boundary, "delta_unresolved")
        & ~_bool_series(high_boundary, "delta_boundary_band_normal")
        & ~_bool_series(high_boundary, "needs_rerun_exact")
    )
    clean = high_boundary[clean_mask].sort_values(["distance_to_normal_sc_boundary", "JA"], ascending=[True, False]).copy()
    if clean.shape[0] > 20:
        clean = clean.iloc[np.linspace(0, clean.shape[0] - 1, 20, dtype=int)].copy()

    return {
        "qwindow_sensitive_points": _standardize_input(qwindow_sensitive, "phase_qwindow_boundary_sensitive"),
        "delta_sensitive_points": _standardize_input(pd.concat([delta_sensitive, delta_high, rerun_high], ignore_index=True, sort=False), "near_zero_delta_refinement"),
        "clean_control_points": _standardize_input(clean, "clean_high_JA_boundary_control"),
    }


def _combine_inputs(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(parts.values(), ignore_index=True, sort=False)
    roles = combined.groupby("coord_key")["audit_role"].apply(lambda s: ";".join(sorted(set(map(str, s))))).rename("audit_roles")
    combined = combined.drop_duplicates("coord_key", keep="first").drop(columns=["audit_role"]).merge(roles, on="coord_key", how="left")
    return combined.sort_values(["JA", "kBT"], ascending=[False, True]).reset_index(drop=True)


def setup_audit(ml_phase_root: Path, report_root: Path, run_id: str) -> None:
    for sub in ["config", "input_points", "raw_outputs/qwindow", "raw_outputs/delta_refinement", "tables", "figures", "reports", "scripts", "branch_curves"]:
        (report_root / sub).mkdir(parents=True, exist_ok=True)
    cfg = _default_config(ml_phase_root, run_id)
    parts = _select_inputs(ml_phase_root)
    for name, df in parts.items():
        _write_csv(df, report_root / "input_points" / f"{name}.csv")
    combined = _combine_inputs(parts)
    _write_csv(combined, report_root / "input_points" / "combined_phase_audit_points.csv")
    cfg["input_counts"] = {name: int(df.shape[0]) for name, df in parts.items()}
    cfg["input_counts"]["combined_unique_points"] = int(combined.shape[0])
    _write_text_lf(report_root / "config" / "phase_qwindow_delta_refinement_config.json", json.dumps(cfg, indent=2) + "\n")
    _write_helpers(report_root)
    _write_pending_report(report_root, cfg)


def _write_helpers(report_root: Path) -> None:
    scripts_dir = report_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    source_script = Path(__file__).resolve()
    script_copy = scripts_dir / "phase_qwindow_delta_refinement_audit.py"
    if source_script != script_copy.resolve():
        script_copy.write_bytes(source_script.read_bytes())
    prelude = """set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPORT_ROOT="${REPORT_ROOT:-${SUBMIT_DIR}}"
if [ "$(basename "${REPORT_ROOT}")" = "scripts" ]; then
  REPORT_ROOT="$(cd "${REPORT_ROOT}/.." && pwd)"
else
  REPORT_ROOT="$(cd "${REPORT_ROOT}" && pwd)"
fi
PROJECT_DIR="${PROJECT_DIR:-$(cd "${REPORT_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PROJECT_DIR}"
"""
    qwindow = """#!/bin/bash
#SBATCH --job-name=phase_qwin
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --exclude=gpuh01

""" + prelude + """
"${PYTHON_BIN}" "${REPORT_ROOT}/scripts/phase_qwindow_delta_refinement_audit.py" run-qwindow \
  --report-root "${REPORT_ROOT}" \
  --rank "${SLURM_ARRAY_TASK_ID}" \
  --world-size "${SLURM_ARRAY_TASK_COUNT}" \
  --device cuda:0
"""
    delta = """#!/bin/bash
#SBATCH --job-name=phase_delta
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --exclude=gpuh01

""" + prelude + """
"${PYTHON_BIN}" "${REPORT_ROOT}/scripts/phase_qwindow_delta_refinement_audit.py" run-delta \
  --report-root "${REPORT_ROOT}" \
  --rank "${SLURM_ARRAY_TASK_ID}" \
  --world-size "${SLURM_ARRAY_TASK_COUNT}" \
  --device cuda:0
"""
    collect = """#!/bin/bash
""" + prelude + """
"${PYTHON_BIN}" "${REPORT_ROOT}/scripts/phase_qwindow_delta_refinement_audit.py" collect --report-root "${REPORT_ROOT}"
"""
    _write_text_lf(scripts_dir / "submit_phase_qwindow_array.sh", qwindow)
    _write_text_lf(scripts_dir / "submit_delta_refinement_array.sh", delta)
    _write_text_lf(scripts_dir / "collect_phase_audit_results.sh", collect)


def _qwindow_points(report_root: Path) -> pd.DataFrame:
    df = pd.read_csv(report_root / "input_points" / "combined_phase_audit_points.csv")
    mask = df["audit_roles"].astype(str).str.contains("phase_qwindow_boundary_sensitive|near_zero_delta_refinement", regex=True)
    return df[mask].reset_index(drop=True)


def run_qwindow(report_root: Path, rank: int, world_size: int, device: str) -> None:
    cfg = json.loads((report_root / "config" / "phase_qwindow_delta_refinement_config.json").read_text(encoding="utf-8"))
    qcfg = cfg["qwindow_audit"]
    df = _qwindow_points(report_root)
    idxs = _rank_slice(df.shape[0], rank, world_size)
    out_dir = report_root / "raw_outputs" / "qwindow"
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = report_root / "branch_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    minima_rows: list[pd.DataFrame] = []
    for i in idxs:
        row = df.iloc[int(i)]
        q_min_old = _safe_float(row, ["q_min"], -1.0)
        q_max_old = _safe_float(row, ["q_max"], 0.5)
        n_q_old = int(_safe_float(row, ["n_q"], 400))
        old_width = q_max_old - q_min_old
        old_dq = old_width / max(n_q_old - 1, 1)
        old_phase = _phase_name_from_label(row.get("phase", row.get("phase_label")), row.get("Delta_opt"), row.get("q_opt"))
        old_deltaf = _safe_float(row, ["DeltaF", "free_energy_gap_to_normal"], float("nan"))
        old_delta = _safe_float(row, ["Delta_opt"], float("nan"))
        old_q = _safe_float(row, ["q_opt"], float("nan"))
        for level in qcfg["q_expansion_levels"]:
            pad = float(level["side_pad_width_factor"]) * old_width
            q_min_new = max(-float(qcfg["q_max_abs"]), q_min_old - pad)
            q_max_new = min(float(qcfg["q_max_abs"]), q_max_old + pad)
            new_width = q_max_new - q_min_new
            density_preserving_nq = int(math.ceil(new_width / max(abs(old_dq), 1e-12))) + 1
            min_multiplier_nq = int(math.ceil(n_q_old * float(level["n_q_multiplier_min"])))
            n_q_new = max(3, density_preserving_nq, min_multiplier_nq)
            t0 = time.perf_counter()
            payload: dict[str, object] = {
                "point_index": int(i),
                "coord_key": row["coord_key"],
                "level": str(level["name"]),
                "kBT": float(row["kBT"]),
                "JA": float(row["JA"]),
                "audit_roles": str(row["audit_roles"]),
                "old_phase": old_phase,
                "old_q_opt": old_q,
                "old_Delta_opt": old_delta,
                "old_DeltaF_min": old_deltaf,
                "q_min_old": q_min_old,
                "q_max_old": q_max_old,
                "n_q_old": n_q_old,
                "status": "ok",
                "failure_reason": "N/A",
                "rank": int(rank),
                "world_size": int(world_size),
                "hostname": socket.gethostname(),
            }
            try:
                result = _eval_phase_q_scan(
                    float(row["kBT"]),
                    float(row["JA"]),
                    q_min_new,
                    q_max_new,
                    n_q_new,
                    device=device,
                    delta_eps=float(qcfg["delta_eps"]),
                    q_eps=float(qcfg["q_eps"]),
                )
                payload.update(result["summary"])
                payload["phase_q_window_expanded_checked"] = True
                payload["phase_q_window_valid"] = bool(not result["summary"]["q_opt_edge_hit"])
                payload["expanded_window_found_lower_branch"] = bool(
                    np.isfinite(old_deltaf)
                    and np.isfinite(float(result["summary"]["DeltaF_min_new"]))
                    and float(result["summary"]["DeltaF_min_new"]) < old_deltaf - float(qcfg["phase_deltaf_tol"])
                )
                payload["phase_changed_by_q_expansion"] = bool(str(payload["old_phase"]) != str(payload["phase_label_new"]))
                payload["uniform_fflo_changed_by_q_expansion"] = bool(
                    str(payload["old_phase"]) in ["uniform_SC", "FFLO"]
                    and str(payload["phase_label_new"]) in ["uniform_SC", "FFLO"]
                    and str(payload["old_phase"]) != str(payload["phase_label_new"])
                )
                branch_path = curve_dir / f"phase_qwindow_point{i:04d}_{level['name']}.npz"
                np.savez(
                    branch_path,
                    **result["branch"],
                    **{k: np.asarray([v]) for k, v in payload.items() if isinstance(v, (int, float, bool, str))},
                )
                payload["branch_curve_path"] = str(branch_path)
                minima = result["minima"].copy()
                if not minima.empty:
                    minima.insert(0, "level", str(level["name"]))
                    minima.insert(0, "point_index", int(i))
                    minima.insert(0, "coord_key", row["coord_key"])
                    minima.insert(0, "JA", float(row["JA"]))
                    minima.insert(0, "kBT", float(row["kBT"]))
                    minima_rows.append(minima)
            except Exception as exc:
                payload["status"] = "failed"
                payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
            payload["elapsed_sec"] = time.perf_counter() - t0
            summary_rows.append(payload)
    pd.DataFrame(summary_rows).to_csv(out_dir / f"phase_qwindow_rank{rank:03d}_of{world_size:03d}.csv", index=False)
    if minima_rows:
        pd.concat(minima_rows, ignore_index=True).to_csv(out_dir / f"local_minima_rank{rank:03d}_of{world_size:03d}.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / f"local_minima_rank{rank:03d}_of{world_size:03d}.csv", index=False)


def _delta_points(report_root: Path) -> pd.DataFrame:
    df = pd.read_csv(report_root / "input_points" / "combined_phase_audit_points.csv")
    mask = df["audit_roles"].astype(str).str.contains("near_zero_delta_refinement|clean_high_JA_boundary_control", regex=True)
    return df[mask].reset_index(drop=True)


def run_delta(report_root: Path, rank: int, world_size: int, device: str) -> None:
    from eta_phase_diagram_cuda import EtaPhaseConfig
    from ml_phase.exact_oracle import evaluate_points

    cfg = json.loads((report_root / "config" / "phase_qwindow_delta_refinement_config.json").read_text(encoding="utf-8"))
    dcfg = cfg["delta_refinement"]
    df = _delta_points(report_root)
    idxs = _rank_slice(df.shape[0], rank, world_size)
    pts = df.iloc[idxs][["kBT", "JA"]].to_numpy(dtype=np.float64)
    out_dir = report_root / "raw_outputs" / "delta_refinement"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"delta_refinement_rank{rank:03d}_of{world_size:03d}.npz"
    if pts.shape[0] == 0:
        np.savez(out_path, kT=np.empty(0), JA=np.empty(0))
        return
    result = evaluate_points(
        points=pts,
        cfg=EtaPhaseConfig(),
        device=device,
        save_every=1,
        output_file=out_path,
        delta_eps=float(dcfg["delta_eps_strict"]),
        delta_boundary_margin=0.005,
        free_energy_ambiguity_tol=float(dcfg["free_energy_ambiguity_tol_strict"]),
        positive_delta_gap_tol=float(dcfg["positive_delta_gap_tol_strict"]),
        enable_q_expansion=bool(dcfg["enable_q_expansion"]),
        q_expand_factor=float(dcfg["q_expand_factor"]),
        q_expand_pad_steps=int(dcfg["q_expand_pad_steps"]),
        q_max_abs=float(dcfg["q_max_abs"]),
        max_q_refinements=int(dcfg["max_q_refinements"]),
        enable_delta_refinement=True,
        delta_refine_half_width=float(dcfg["delta_refine_half_width"]),
        n_delta_refined=int(dcfg["n_delta_refined"]),
        max_delta_refinements=int(dcfg["max_delta_refinements"]),
        allow_ambiguous_output=True,
    )
    payload = result.to_dict()
    payload["rank"] = np.asarray([rank], dtype=np.int64)
    payload["world_size"] = np.asarray([world_size], dtype=np.int64)
    payload["hostname"] = np.asarray([socket.gethostname()])
    np.savez(out_path, **payload)


def _load_delta_outputs(report_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((report_root / "raw_outputs" / "delta_refinement").glob("delta_refinement_rank*.npz")):
        with np.load(path, allow_pickle=True) as z:
            if "kT" not in z.files or z["kT"].size == 0:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "kBT": z["kT"],
                        "JA": z["JA"],
                        "refined_eta": z["eta"],
                        "refined_q_opt": z["q_opt"],
                        "refined_Delta_opt": z["delta_opt"],
                        "refined_DeltaF": z["free_energy_gap_to_normal"],
                        "refined_phase_candidate": z["phase_candidate"],
                        "refined_delta_status": z["delta_status"],
                        "refined_delta_ambiguous": z["delta_boundary_ambiguous"],
                        "refined_delta_unresolved": z["delta_unresolved"],
                        "refined_positive_delta_gap": z["positive_delta_gap"],
                        "refined_positive_delta_checked": z["positive_delta_checked"],
                        "refined_q_status": z["q_status"],
                        "refined_q_expanded": z["q_expanded"],
                        "refined_q_unresolved": z["q_unresolved"],
                        "refined_trusted_exact": z["trusted_exact"],
                        "refined_exact_status_code": z["exact_status_code"],
                        "refined_exact_status_name": z["exact_status_name"],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_results(report_root: Path) -> None:
    tables = report_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    q_files = sorted((report_root / "raw_outputs" / "qwindow").glob("phase_qwindow_rank*.csv"))
    minima_files = sorted((report_root / "raw_outputs" / "qwindow").glob("local_minima_rank*.csv"))
    qdf = pd.concat([pd.read_csv(p) for p in q_files], ignore_index=True) if q_files else pd.DataFrame()
    minima = pd.concat([pd.read_csv(p) for p in minima_files if p.stat().st_size > 1], ignore_index=True) if minima_files else pd.DataFrame()
    if not qdf.empty:
        qdf.to_csv(tables / "phase_qwindow_comparison.csv", index=False)
    else:
        pd.DataFrame().to_csv(tables / "phase_qwindow_comparison.csv", index=False)
    if not minima.empty:
        minima.to_csv(tables / "low_energy_local_minima.csv", index=False)
    else:
        pd.DataFrame().to_csv(tables / "low_energy_local_minima.csv", index=False)

    old_delta = _delta_points(report_root)
    new_delta = _load_delta_outputs(report_root)
    if not new_delta.empty:
        old_delta["coord_key"] = [_coord_key(k, j) for k, j in zip(old_delta["kBT"], old_delta["JA"])]
        new_delta["coord_key"] = [_coord_key(k, j) for k, j in zip(new_delta["kBT"], new_delta["JA"])]
        comp = old_delta.merge(new_delta, on="coord_key", how="left", suffixes=("_old", ""))
        comp["old_phase"] = [
            _phase_name_from_label(v, d, q)
            for v, d, q in zip(comp.get("phase", comp.get("phase_label")), comp["Delta_opt"], comp["q_opt"])
        ]
        comp["refined_phase"] = [
            _phase_name_from_delta_q(d, q, 1e-3, 1e-2, bool(a) or bool(u))
            for d, q, a, u in zip(comp["refined_Delta_opt"], comp["refined_q_opt"], comp["refined_delta_ambiguous"], comp["refined_delta_unresolved"])
        ]
        comp["old_DeltaF"] = comp.apply(lambda r: _safe_float(r, ["DeltaF", "free_energy_gap_to_normal"], float("nan")), axis=1)
        comp["delta_refinement_triggered"] = comp["refined_positive_delta_checked"].astype(float).fillna(0).astype(int).astype(bool) | comp["refined_delta_ambiguous"].astype(float).fillna(0).astype(int).astype(bool)
        comp["delta_refinement_valid"] = ~(comp["refined_delta_unresolved"].astype(float).fillna(0).astype(int).astype(bool))
        comp["boundary_ambiguous"] = comp["refined_phase"].eq("boundary_ambiguous")
        comp["changed_after_delta_refinement"] = comp["old_phase"].astype(str) != comp["refined_phase"].astype(str)
        comp.to_csv(tables / "delta_refinement_comparison.csv", index=False)
    else:
        comp = pd.DataFrame()
        pd.DataFrame().to_csv(tables / "delta_refinement_comparison.csv", index=False)

    combined = _combined_summary(qdf, comp, minima)
    combined.to_csv(tables / "combined_phase_robustness_summary.csv", index=False)
    _write_figures(report_root, qdf, comp, minima)
    _write_final_reports(report_root, qdf, comp, minima, combined)


def _combined_summary(qdf: pd.DataFrame, comp: pd.DataFrame, minima: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = set()
    for df in [qdf, comp]:
        if not df.empty and "coord_key" in df.columns:
            keys.update(df["coord_key"].dropna().astype(str).tolist())
    for key in sorted(keys):
        qsub = qdf[qdf["coord_key"].astype(str) == key] if not qdf.empty and "coord_key" in qdf.columns else pd.DataFrame()
        dsub = comp[comp["coord_key"].astype(str) == key] if not comp.empty and "coord_key" in comp.columns else pd.DataFrame()
        row: dict[str, object] = {"coord_key": key}
        source = qsub.iloc[0] if not qsub.empty else (dsub.iloc[0] if not dsub.empty else None)
        if source is not None:
            row["kBT"] = float(source["kBT"])
            row["JA"] = float(source["JA"])
            row["audit_roles"] = str(source.get("audit_roles", "N/A"))
        row["phase_q_window_expanded_checked"] = bool(not qsub.empty)
        row["phase_q_window_valid"] = bool(qsub["phase_q_window_valid"].all()) if not qsub.empty and "phase_q_window_valid" in qsub else None
        row["expanded_window_found_lower_branch"] = bool(qsub["expanded_window_found_lower_branch"].any()) if not qsub.empty and "expanded_window_found_lower_branch" in qsub else None
        row["phase_changed_by_q_expansion"] = bool(qsub["phase_changed_by_q_expansion"].any()) if not qsub.empty and "phase_changed_by_q_expansion" in qsub else None
        row["low_energy_local_minima_count"] = int(minima[minima["coord_key"].astype(str) == key].shape[0]) if not minima.empty and "coord_key" in minima else 0
        row["delta_refinement_triggered"] = bool(dsub["delta_refinement_triggered"].any()) if not dsub.empty and "delta_refinement_triggered" in dsub else None
        row["delta_refinement_valid"] = bool(dsub["delta_refinement_valid"].all()) if not dsub.empty and "delta_refinement_valid" in dsub else None
        row["boundary_ambiguous"] = bool(dsub["boundary_ambiguous"].any()) if not dsub.empty and "boundary_ambiguous" in dsub else None
        row["changed_after_delta_refinement"] = bool(dsub["changed_after_delta_refinement"].any()) if not dsub.empty and "changed_after_delta_refinement" in dsub else None
        row["eta_response_valid"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _write_figures(report_root: Path, qdf: pd.DataFrame, comp: pd.DataFrame, minima: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not qdf.empty:
        latest = qdf.sort_values("level").groupby("coord_key", as_index=False).tail(1)
        fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        colors = np.where(latest["expanded_window_found_lower_branch"].astype(bool), "tab:red", "tab:blue")
        ax.scatter(latest["kBT"], latest["JA"], c=colors, s=28, alpha=0.85)
        ax.set_xlabel(r"$k_B T/t$")
        ax.set_ylabel(r"$J_A/t$")
        ax.set_title("Expanded q-window lower-branch check")
        fig.savefig(fig_dir / "phase_change_map_qwindow_delta.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        ax.scatter(latest["old_q_opt"], latest["q_opt_new"], s=20, alpha=0.8)
        lo = float(np.nanmin([latest["old_q_opt"].min(), latest["q_opt_new"].min()]))
        hi = float(np.nanmax([latest["old_q_opt"].max(), latest["q_opt_new"].max()]))
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.8)
        ax.set_xlabel(r"old $q_{\rm opt}$")
        ax.set_ylabel(r"expanded-window $q_{\rm opt}$")
        ax.set_title(r"$q_{\rm opt}$ shift after q-window expansion")
        fig.savefig(fig_dir / "qopt_shift_after_expansion.png", dpi=220)
        plt.close(fig)

    if not comp.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        ax.scatter(comp["old_DeltaF"], comp["refined_DeltaF"], s=22, alpha=0.85)
        ax.axhline(0.0, color="0.25", lw=0.8)
        ax.axvline(0.0, color="0.25", lw=0.8)
        ax.set_xlabel(r"old $\Delta F$")
        ax.set_ylabel(r"refined $\Delta F$")
        ax.set_title(r"$\Delta F$ before/after near-zero-$\Delta$ refinement")
        fig.savefig(fig_dir / "DeltaF_before_after_refinement.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        colors = np.where(comp["changed_after_delta_refinement"].astype(bool), "tab:red", "tab:green")
        ax.scatter(comp["kBT"], comp["JA"], c=colors, s=28, alpha=0.85)
        ax.set_xlabel(r"$k_B T/t$")
        ax.set_ylabel(r"$J_A/t$")
        ax.set_title("High-JA boundary points before/after Delta refinement")
        fig.savefig(fig_dir / "high_JA_boundary_points_before_after.png", dpi=220)
        plt.close(fig)

    if not minima.empty:
        fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
        ax.scatter(minima["q_local_min"], minima["DeltaF_local_min"], c=minima["JA"], s=18, alpha=0.8, cmap="viridis")
        ax.axhline(0.0, color="0.25", lw=0.8)
        ax.set_xlabel(r"local-minimum $q$")
        ax.set_ylabel(r"$\Delta F_{\min}(q)$")
        ax.set_title("Saved low-energy local-minimum candidates")
        fig.savefig(fig_dir / "local_minima_branch_candidates.png", dpi=220)
        plt.close(fig)


def _counts(qdf: pd.DataFrame, comp: pd.DataFrame, combined: pd.DataFrame) -> dict[str, object]:
    return {
        "qwindow_rows": int(qdf.shape[0]),
        "qwindow_points": int(qdf["coord_key"].nunique()) if not qdf.empty and "coord_key" in qdf else 0,
        "expanded_window_found_lower_branch_points": int(combined["expanded_window_found_lower_branch"].fillna(False).astype(bool).sum()) if "expanded_window_found_lower_branch" in combined else 0,
        "phase_changed_by_q_expansion_points": int(combined["phase_changed_by_q_expansion"].fillna(False).astype(bool).sum()) if "phase_changed_by_q_expansion" in combined else 0,
        "delta_rows": int(comp.shape[0]),
        "delta_changed_points": int(comp["changed_after_delta_refinement"].sum()) if "changed_after_delta_refinement" in comp else 0,
        "boundary_ambiguous_points": int(comp["boundary_ambiguous"].sum()) if "boundary_ambiguous" in comp else 0,
    }


def _write_final_reports(report_root: Path, qdf: pd.DataFrame, comp: pd.DataFrame, minima: pd.DataFrame, combined: pd.DataFrame) -> None:
    counts = _counts(qdf, comp, combined)
    md = _report_markdown(counts, qdf, comp, minima)
    _write_text_lf(report_root / "phase_qwindow_delta_refinement.md", md)
    _write_text_lf(report_root / "reports" / "phase_qwindow_delta_refinement.md", md)
    _write_text_lf(report_root / "decision_log.md", _decision_log(counts))
    tex = _markdown_to_basic_tex(md)
    _write_text_lf(report_root / "phase_qwindow_delta_refinement.tex", tex)
    _try_compile_pdf(report_root / "phase_qwindow_delta_refinement.tex")


def _write_pending_report(report_root: Path, cfg: dict[str, object]) -> None:
    counts = cfg.get("input_counts", {})
    md = f"""# Phase q-window and Delta-refinement audit

Status: input points and HPC helper scripts prepared; exact reruns are pending.

## Input Counts

```json
{json.dumps(counts, indent=2)}
```

## Scope

This is an audit-only production-oriented numerical robustness update. It does
not modify acquisition, StopController, NN training, or the active-learning
dataset.

## Required interpretation rules

- q-window expansion is not required to prove superconductivity once a
  lower-free-energy positive-Delta superconducting state is already found.
- q-window expansion is required to check branch identity, q_opt stability,
  boundary robustness, and future topology classification.
- eta response has already been downgraded to response-extraction pathology
  unless `eta_response_valid=True`.
"""
    _write_text_lf(report_root / "phase_qwindow_delta_refinement.md", md)
    _write_text_lf(report_root / "reports" / "phase_qwindow_delta_refinement.md", md)
    _write_text_lf(report_root / "decision_log.md", _pending_decision_log(counts))
    _write_text_lf(report_root / "phase_qwindow_delta_refinement.tex", _markdown_to_basic_tex(md))


def _pending_decision_log(counts: dict[str, object]) -> str:
    return f"""# Decision Log: Phase q-window and Delta-refinement audit

Status: setup complete; exact q-window and Delta-refinement reruns are pending.

## Prepared Inputs

```json
{json.dumps(counts, indent=2)}
```

## Decisions Already Fixed

- Do not modify active-learning acquisition, NN training, StopController, or
  the existing active-learning dataset.
- Do not redefine the free-energy phase criterion.
- Use q-window expansion to test branch identity, q_opt stability, boundary
  robustness, and topology readiness.
- Use near-zero Delta refinement to resolve tolerance-sensitive normal/SC
  boundary points.
- Do not treat eta response as robust positive physics unless
  `eta_response_valid=True`.

## Pending Numerical Checks

- Run `scripts/submit_phase_qwindow_array.sh`.
- Run `scripts/submit_delta_refinement_array.sh`.
- After both jobs complete, run `scripts/collect_phase_audit_results.sh`.

## Expected Final Outputs

- `tables/phase_qwindow_comparison.csv`
- `tables/delta_refinement_comparison.csv`
- `tables/low_energy_local_minima.csv`
- `tables/combined_phase_robustness_summary.csv`
- final `phase_qwindow_delta_refinement.md`
- final `phase_qwindow_delta_refinement.pdf`
"""


def _report_markdown(counts: dict[str, object], qdf: pd.DataFrame, comp: pd.DataFrame, minima: pd.DataFrame) -> str:
    lines = [
        "# Phase q-window and Delta-refinement audit",
        "",
        "## Purpose and Scope",
        "",
        "This report is a production-oriented numerical robustness update for the discovery active-learning phase diagram. It strengthens phase-side q-window and near-zero-Delta diagnostics without changing the active-learning acquisition function, neural-network surrogate, StopController, or active-learning dataset.",
        "",
        "## Phase Criterion Retained",
        "",
        "The basic thermodynamic criterion is unchanged:",
        "",
        r"\[",
        r"\Delta F_{\min}=\min_{\Delta>0,q}\left[F(\Delta,q)-F_N\right].",
        r"\]",
        "",
        r"Finding one positive-\(\Delta\) superconducting state with \(F_{\rm SC}<F_N\) is sufficient for the basic SC classification. Complete branch enumeration is not required to prove superconductivity.",
        "",
        "## Why Expanded q-window Scans Are Still Needed",
        "",
        r"q-window expansion is not required to prove superconductivity once a lower-free-energy SC state is already found; it is required to check branch identity, \(q_{\rm opt}\) stability, boundary robustness, and future topology classification.",
        "",
        r"The expanded scan saves \(F_{\min}(q)=\min_\Delta F(\Delta,q)\) and multiple low-energy local minima so later topology work can evaluate branch-resolved invariants.",
        "",
        "## Summary Counts",
        "",
        "```json",
        json.dumps(counts, indent=2),
        "```",
        "",
        "## q-window Expansion Outputs",
        "",
    ]
    if qdf.empty:
        lines.append("No q-window rerun output was found.")
    else:
        latest = qdf.sort_values("level").groupby("coord_key", as_index=False).tail(1)
        lines.append(f"Compared {latest.shape[0]} unique points after expanded q-window scans.")
        lines.append(f"Expanded-window lower-branch points: {int(latest['expanded_window_found_lower_branch'].fillna(False).astype(bool).sum())}.")
        lines.append(f"Phase changes after q expansion: {int(latest['phase_changed_by_q_expansion'].fillna(False).astype(bool).sum())}.")
        lines.append("")
        lines.append("![qopt shift after expansion](figures/qopt_shift_after_expansion.png)")
        lines.append("")
        lines.append("![phase change map](figures/phase_change_map_qwindow_delta.png)")
    lines.extend(["", "## near-zero Delta Refinement Outputs", ""])
    if comp.empty:
        lines.append("No Delta-refinement rerun output was found.")
    else:
        lines.append(f"Compared {comp.shape[0]} Delta-refinement points.")
        lines.append(f"Changed after Delta refinement: {int(comp['changed_after_delta_refinement'].sum())}.")
        lines.append(f"Boundary ambiguous after refinement: {int(comp['boundary_ambiguous'].sum())}.")
        lines.append("")
        lines.append("![DeltaF before after refinement](figures/DeltaF_before_after_refinement.png)")
        lines.append("")
        lines.append("![high JA boundary before after](figures/high_JA_boundary_points_before_after.png)")
    lines.extend(["", "## Branch-resolved Readiness", ""])
    if minima.empty:
        lines.append("No low-energy local-minimum table was found.")
    else:
        lines.append(f"Saved {minima.shape[0]} local-minimum rows for later branch-resolved/topology checks.")
        lines.append("")
        lines.append("![local minima branch candidates](figures/local_minima_branch_candidates.png)")
    lines.extend(
        [
            "",
            "## Eta Response Caveat",
            "",
            "The eta response has already been downgraded to response-extraction pathology unless `eta_response_valid=True`. This report does not re-promote high-JA positive eta as robust physics.",
            "",
            "## Interpretation Boundaries",
            "",
            "- Basic SC classification: one lower-free-energy positive-Delta state is enough.",
            r"- Branch-resolved completeness: expanded q-window scans test whether the selected branch and \(q_{\rm opt}\) are stable.",
            "- Topology readiness: saved low-energy FFLO local minima are candidates for later invariant calculations.",
        ]
    )
    return "\n".join(lines) + "\n"


def _decision_log(counts: dict[str, object]) -> str:
    return f"""# Decision Log: Phase q-window and Delta-refinement audit

## Scope

Audit-only numerical robustness update for high-JA, low-T phase-boundary
points. No active-learning dataset is modified.

## Current Counts

```json
{json.dumps(counts, indent=2)}
```

## Decisions

- Keep the original free-energy phase criterion.
- Use expanded q-window scans to test branch identity, q_opt stability, and
  boundary robustness, not to redefine superconductivity.
- Use near-zero Delta refinement only for tolerance-sensitive normal/SC
  boundary points.
- Treat eta as invalid for robust positive-response claims unless
  eta_response_valid=True.

## Next Step

Inspect `tables/combined_phase_robustness_summary.csv` and
`tables/low_energy_local_minima.csv`. Points with lower branches outside the
old q-window or multiple near-degenerate FFLO minima should be prioritized for
branch-resolved topology calculations.
"""


def _markdown_to_basic_tex(md: str) -> str:
    body = md
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "# ": r"\section*{",
        "## ": r"\subsection*{",
    }
    lines: list[str] = []
    in_code = False
    in_math = False
    for raw in body.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            lines.append(r"\begin{verbatim}" if not in_code else r"\end{verbatim}")
            in_code = not in_code
            continue
        if in_code:
            lines.append(line)
            continue
        if line.startswith(r"\["):
            in_math = True
            lines.append(line)
            continue
        if line.startswith(r"\]"):
            in_math = False
            lines.append(line)
            continue
        if in_math:
            lines.append(line)
            continue
        if line.startswith("!["):
            start = line.rfind("(")
            end = line.rfind(")")
            if start >= 0 and end > start:
                path = line[start + 1 : end]
                if path.lower().endswith(".png"):
                    lines.append(r"\begin{figure}[H]\centering")
                    lines.append(rf"\includegraphics[width=0.78\linewidth]{{{path}}}")
                    lines.append(r"\end{figure}")
            continue
        if line.startswith("# "):
            lines.append(r"\section*{" + _tex_escape(line[2:]) + "}")
        elif line.startswith("## "):
            lines.append(r"\subsection*{" + _tex_escape(line[3:]) + "}")
        elif line.startswith("- "):
            lines.append(r"\noindent $\bullet$ " + _tex_escape(line[2:]) + r"\\")
        elif line.startswith("|"):
            lines.append(r"\begin{verbatim}" + line + r"\end{verbatim}")
        else:
            lines.append(_tex_escape(line))
    return "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{graphicx}",
            r"\usepackage{float}",
            r"\usepackage{amsmath}",
            r"\usepackage{hyperref}",
            r"\begin{document}",
            *lines,
            r"\end{document}",
            "",
        ]
    )


def _tex_escape(text: str) -> str:
    if text.startswith(r"\[") or text.startswith(r"\]"):
        return text
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def _try_compile_pdf(tex_path: Path) -> None:
    if shutil.which("pdflatex") is None:
        return
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def self_test() -> None:
    q = np.asarray([0.0, 1.0, 2.0, 3.0])
    delta = np.asarray([0.1, 0.2, 0.1, 0.2])
    deltaf = np.asarray([0.0, -2.0, -1.0, -3.0])
    minima = _local_minima(q, delta, deltaf, topn=3)
    assert minima.shape[0] >= 2
    assert float(minima.iloc[0]["DeltaF_local_min"]) == -3.0
    assert _phase_name_from_delta_q(0.0, 0.0, 1e-3, 1e-2) == "normal"
    assert _phase_name_from_delta_q(0.01, 0.0, 1e-3, 1e-2) == "uniform_SC"
    assert _phase_name_from_delta_q(0.01, 0.2, 1e-3, 1e-2) == "FFLO"
    md = _report_markdown({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    required = [
        "q-window expansion is not required to prove superconductivity",
        "eta response has already been downgraded",
        "Basic SC classification",
    ]
    for phrase in required:
        assert phrase in md


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit-only phase q-window and Delta-refinement workflow.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    setup = sub.add_parser("setup")
    setup.add_argument("--ml-phase-root", type=Path, required=True)
    setup.add_argument("--report-root", type=Path, default=Path(REPORT_NAME_DEFAULT))
    setup.add_argument("--run-id", type=str, default=RUN_ID_DEFAULT)
    for name in ["run-qwindow", "run-delta", "collect"]:
        sp = sub.add_parser(name)
        sp.add_argument("--report-root", type=Path, required=True)
        if name != "collect":
            sp.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
            sp.add_argument("--world-size", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
            sp.add_argument("--device", type=str, default="cuda:0")
    sub.add_parser("self-test")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "setup":
        setup_audit(args.ml_phase_root, args.report_root, args.run_id)
        print(f"Prepared report/audit folder: {args.report_root}")
    elif args.cmd == "run-qwindow":
        run_qwindow(args.report_root, args.rank, args.world_size, args.device)
    elif args.cmd == "run-delta":
        run_delta(args.report_root, args.rank, args.world_size, args.device)
    elif args.cmd == "collect":
        collect_results(args.report_root)
    elif args.cmd == "self-test":
        self_test()
        print("self-test passed")


if __name__ == "__main__":
    main()
