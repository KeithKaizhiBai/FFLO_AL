from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
for path_candidate in [Path.cwd().resolve(), REPO_ROOT]:
    path_text = str(path_candidate)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


RUN_ID_DEFAULT = "active_boundary_discovery_512seed_256x50"
AUDIT_NAME_DEFAULT = "numerical_audit_qwindow_delta_v1"


def _bool_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(False, index=df.index)
    raw = df[name]
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin(["1", "true", "yes"])


def _coord_key(kbt: float, ja: float) -> str:
    return f"{float(kbt):.8f},{float(ja):.8f}"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _phase_from_delta_q(delta: float, q: float, delta_eps: float, q_eps: float) -> str:
    if not np.isfinite(delta):
        return "unknown"
    if float(delta) < float(delta_eps):
        return "normal"
    if abs(float(q)) < float(q_eps):
        return "uniform_SC"
    return "FFLO"


def _config_payload(ml_phase_root: Path, run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "ml_phase_root": str(ml_phase_root),
        "final_dataset": str(ml_phase_root / "active_runs" / run_id / "dataset_iter020.npz"),
        "source_audit_tables": str(ml_phase_root / "reports" / "audit_tables"),
        "qwindow_audit": {
            "q_expansion_levels": [
                {"name": "expand_0p5_width", "side_pad_width_factor": 0.5, "n_q_multiplier": 2},
                {"name": "expand_1p0_width", "side_pad_width_factor": 1.0, "n_q_multiplier": 4},
            ],
            "q_max_abs": math.pi,
            "n_edge": 3,
            "delta_eps": 1.0e-3,
            "endpoint_deltaf_tol": 1.0e-8,
            "save_branch_arrays": True,
        },
        "delta_audit": {
            "delta_eps_strict": 1.0e-3,
            "q_eps": 1.0e-2,
            "positive_delta_gap_tol_strict": 1.0e-10,
            "free_energy_ambiguity_tol_strict": 1.0e-8,
            "delta_refine_half_width": 0.01,
            "n_delta_refined": 1000,
            "max_delta_refinements": 3,
            "enable_q_expansion": True,
            "q_expand_factor": 2.0,
            "q_expand_pad_steps": 80,
            "max_q_refinements": 4,
            "q_max_abs": math.pi,
        },
        "subset_rules": {
            "eta_positive_high_JA": "JA > 1.25 and eta > 0",
            "high_JA_kink_delta": "JA > 1.1 and distance_to_normal_sc_boundary <= 0.025 and (delta_ambiguous or delta_unresolved or boundary_band_normal)",
            "rerun_required": "all rows from rerun_required_points.csv",
            "clean_control": "up to 20 trusted high-JA kink points without Delta ambiguity, boundary-band, or rerun flags",
        },
        "safety": {
            "do_not_modify_active_runs": True,
            "do_not_overwrite_final_dataset": True,
            "do_not_append_training_data": True,
            "audit_only": True,
        },
    }


def _write_readme(audit_root: Path, config: dict[str, object]) -> None:
    package_root = Path(str(config["ml_phase_root"])).parent
    try:
        audit_path_for_hpc = audit_root.relative_to(package_root).as_posix()
    except ValueError:
        audit_path_for_hpc = audit_root.as_posix()
    readme = f"""# Numerical Audit: q-window and Delta refinement

This folder is an independent numerical audit for `{config['run_id']}`.
It does not modify the original active-learning run, does not overwrite
`dataset_iter020.npz`, and does not append rerun results to training data.

## Inputs

- `input_points/eta_positive_high_JA_selected.csv`
- `input_points/high_JA_kink_delta_selected.csv`
- `input_points/rerun_required_selected.csv`
- `input_points/clean_control_selected.csv`
- `input_points/combined_audit_points.csv`

## Rerun commands

On the HPC login node, from the uploaded package directory:

```bash
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
export PROJECT_DIR=$PWD
cd "{audit_path_for_hpc}"

sbatch scripts/submit_qwindow_array.sh
sbatch scripts/submit_delta_array.sh
```

After both jobs finish:

```bash
bash scripts/collect_results.sh
```

## Interpretation limits

The old cFFLO/tFFLO curves, if used in later analysis, are reference curves
only. This audit does not compute a pointwise topology oracle.
"""
    _write_text_lf(audit_root / "README.md", readme)


def _write_slurm_helpers(audit_root: Path) -> None:
    scripts_dir = audit_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    slurm_prelude = """set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
AUDIT_ROOT="${AUDIT_ROOT:-${SUBMIT_DIR}}"
if [ "$(basename "${AUDIT_ROOT}")" = "scripts" ]; then
  AUDIT_ROOT="$(cd "${AUDIT_ROOT}/.." && pwd)"
else
  AUDIT_ROOT="$(cd "${AUDIT_ROOT}" && pwd)"
fi
PROJECT_DIR="${PROJECT_DIR:-$(cd "${AUDIT_ROOT}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PROJECT_DIR}"
"""
    qwindow = """#!/bin/bash
#SBATCH --job-name=aud_qwin
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --array=0-7

""" + slurm_prelude + """
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/numerical_audit_qwindow_delta.py" run-qwindow \
  --audit-root "${AUDIT_ROOT}" \
  --rank "${SLURM_ARRAY_TASK_ID}" \
  --world-size "${SLURM_ARRAY_TASK_COUNT}" \
  --device cuda:0
"""
    delta = """#!/bin/bash
#SBATCH --job-name=aud_delta
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --array=0-7

""" + slurm_prelude + """
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/numerical_audit_qwindow_delta.py" run-delta \
  --audit-root "${AUDIT_ROOT}" \
  --rank "${SLURM_ARRAY_TASK_ID}" \
  --world-size "${SLURM_ARRAY_TASK_COUNT}" \
  --device cuda:0
"""
    collect = """#!/bin/bash
""" + slurm_prelude + """
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/numerical_audit_qwindow_delta.py" collect --audit-root "${AUDIT_ROOT}"
"""
    _write_text_lf(scripts_dir / "submit_qwindow_array.sh", qwindow)
    _write_text_lf(scripts_dir / "submit_delta_array.sh", delta)
    _write_text_lf(scripts_dir / "collect_results.sh", collect)
    source_script = Path(__file__).resolve()
    script_copy = scripts_dir / "numerical_audit_qwindow_delta.py"
    if source_script != script_copy.resolve():
        script_copy.write_bytes(source_script.read_bytes())


def setup_audit(ml_phase_root: Path, run_id: str, audit_name: str) -> Path:
    audit_root = ml_phase_root / audit_name
    for sub in [
        "config",
        "input_points",
        "raw_outputs/qwindow_rerun",
        "raw_outputs/delta_refine_rerun",
        "tables",
        "figures",
        "reports",
    ]:
        (audit_root / sub).mkdir(parents=True, exist_ok=True)

    tables = ml_phase_root / "reports" / "audit_tables"
    eta = _read_csv(tables / "eta_positive_high_JA_points.csv")
    high = _read_csv(tables / "high_JA_boundary_kink_points.csv")
    rerun = _read_csv(tables / "rerun_required_points.csv")

    eta_sel = eta[(eta["JA"] > 1.25) & (eta["eta"] > 0)].copy()
    kink_flags = _bool_series(high, "delta_boundary_ambiguous") | _bool_series(high, "delta_unresolved") | _bool_series(high, "delta_boundary_band_normal")
    kink_delta = high[(high["JA"] > 1.1) & (high["distance_to_normal_sc_boundary"] <= 0.025) & kink_flags].copy()
    rerun_sel = rerun.copy()
    clean_mask = (
        (high["JA"] > 1.1)
        & (high["distance_to_normal_sc_boundary"] <= 0.025)
        & _bool_series(high, "trusted_exact")
        & ~_bool_series(high, "delta_boundary_ambiguous")
        & ~_bool_series(high, "delta_unresolved")
        & ~_bool_series(high, "delta_boundary_band_normal")
        & ~_bool_series(high, "needs_rerun_exact")
    )
    clean = high[clean_mask].sort_values(["distance_to_normal_sc_boundary", "JA"], ascending=[True, False]).copy()
    if clean.shape[0] > 20:
        idx = np.linspace(0, clean.shape[0] - 1, 20, dtype=int)
        clean = clean.iloc[idx].copy()

    eta_sel["audit_role"] = "eta_positive_high_JA_qwindow"
    kink_delta["audit_role"] = "high_JA_kink_delta"
    rerun_sel["audit_role"] = "rerun_required_delta"
    clean["audit_role"] = "clean_high_JA_kink_control"

    _write_csv(eta_sel, audit_root / "input_points" / "eta_positive_high_JA_selected.csv")
    _write_csv(kink_delta, audit_root / "input_points" / "high_JA_kink_delta_selected.csv")
    _write_csv(rerun_sel, audit_root / "input_points" / "rerun_required_selected.csv")
    _write_csv(clean, audit_root / "input_points" / "clean_control_selected.csv")

    combined = pd.concat([eta_sel, kink_delta, rerun_sel, clean], ignore_index=True, sort=False)
    combined["coord_key"] = [_coord_key(k, j) for k, j in zip(combined["kBT"], combined["JA"])]
    roles = combined.groupby("coord_key")["audit_role"].apply(lambda s: ";".join(sorted(set(map(str, s))))).rename("audit_roles")
    combined = combined.drop_duplicates("coord_key", keep="first").drop(columns=["audit_role"]).merge(roles, on="coord_key", how="left")
    _write_csv(combined, audit_root / "input_points" / "combined_audit_points.csv")

    config = _config_payload(ml_phase_root, run_id)
    config["subset_counts"] = {
        "eta_positive_high_JA_selected": int(eta_sel.shape[0]),
        "high_JA_kink_delta_selected": int(kink_delta.shape[0]),
        "rerun_required_selected": int(rerun_sel.shape[0]),
        "clean_control_selected": int(clean.shape[0]),
        "combined_unique_audit_points": int(combined.shape[0]),
    }
    _write_text_lf(audit_root / "config" / "audit_config.json", json.dumps(config, indent=2) + "\n")
    _write_readme(audit_root, config)
    _write_slurm_helpers(audit_root)
    _write_pending_report(audit_root, config)
    return audit_root


def _rank_slice(n: int, rank: int, world_size: int) -> np.ndarray:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return np.arange(n, dtype=int)[int(rank) :: int(world_size)]


def _ic_points(j_q: np.ndarray, q_vec: np.ndarray, idx_q_opt: int) -> tuple[float, float, float, float, int, int]:
    n_q = j_q.shape[0]
    ic_plus = 0.0
    ic_minus = 0.0
    q_plus = np.nan
    q_minus = np.nan
    idx_plus = -1
    idx_minus = -1
    for iq in range(idx_q_opt, n_q - 1):
        if j_q[iq] > 0 and j_q[iq + 1] <= j_q[iq]:
            ic_plus = float(j_q[iq])
            q_plus = float(q_vec[iq])
            idx_plus = int(iq)
            break
    if np.isnan(q_plus) and n_q >= 2 and j_q[-1] > 0 and j_q[-1] > j_q[-2]:
        ic_plus = float(j_q[-1])
        q_plus = float(q_vec[-1])
        idx_plus = int(n_q - 1)
    for iq in range(idx_q_opt, 0, -1):
        if j_q[iq] < 0 and j_q[iq - 1] >= j_q[iq]:
            ic_minus = float(j_q[iq])
            q_minus = float(q_vec[iq])
            idx_minus = int(iq)
            break
    if np.isnan(q_minus) and n_q >= 2 and j_q[0] < 0 and j_q[0] < j_q[1]:
        ic_minus = float(j_q[0])
        q_minus = float(q_vec[0])
        idx_minus = 0
    return ic_plus, ic_minus, q_plus, q_minus, idx_plus, idx_minus


def _eta_from_ic(ic_plus: float, ic_minus: float) -> float:
    if ic_plus == 0 and ic_minus == 0:
        return 0.0
    if ic_plus == 0:
        return -1.0
    if ic_minus == 0:
        return 1.0
    return float((abs(ic_plus) - abs(ic_minus)) / (abs(ic_plus) + abs(ic_minus)))


def _endpoint_indices(delta_q: np.ndarray, deltaf_q: np.ndarray, idx_q_opt: int, delta_eps: float, deltaf_tol: float) -> tuple[int, int]:
    inactive = (delta_q < float(delta_eps)) | (deltaf_q >= -abs(float(deltaf_tol)))
    left_idx = -1
    right_idx = -1
    for i in range(idx_q_opt, -1, -1):
        if inactive[i]:
            left_idx = int(i)
            break
    for i in range(idx_q_opt, inactive.size):
        if inactive[i]:
            right_idx = int(i)
            break
    return left_idx, right_idx


def _eval_branch(kbt: float, ja: float, q_min: float, q_max: float, n_q: int, device: str, delta_eps: float, deltaf_tol: float) -> dict[str, object]:
    import torch
    from eta_phase_diagram_cuda import EtaPhaseConfig, build_q_vec, compute_current_from_omega, compute_omega_min_q_batch, maybe_set_linalg_backend

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

    j_q = compute_current_from_omega(omega.reshape(1, 1, -1), q_vec)[0, 0]
    idx_q_opt = int(np.argmin(np.abs(q_vec - q_opt)))
    ic_plus, ic_minus, q_plus, q_minus, idx_plus, idx_minus = _ic_points(j_q, q_vec, idx_q_opt)
    eta = _eta_from_ic(ic_plus, ic_minus)
    left_idx, right_idx = _endpoint_indices(delta_q, deltaf_q, idx_q_opt, delta_eps, deltaf_tol)
    dq = float(np.min(np.diff(q_vec))) if q_vec.size > 1 else float("nan")
    endpoint_left_found = left_idx >= 0
    endpoint_right_found = right_idx >= 0
    q_end_left = float(q_vec[left_idx]) if endpoint_left_found else float("nan")
    q_end_right = float(q_vec[right_idx]) if endpoint_right_found else float("nan")
    margins = [
        q_opt - float(q_vec[0]),
        float(q_vec[-1]) - q_opt,
        q_plus - float(q_vec[0]) if np.isfinite(q_plus) else float("nan"),
        float(q_vec[-1]) - q_plus if np.isfinite(q_plus) else float("nan"),
        q_minus - float(q_vec[0]) if np.isfinite(q_minus) else float("nan"),
        float(q_vec[-1]) - q_minus if np.isfinite(q_minus) else float("nan"),
    ]
    endpoint_margins = [
        q_end_left - float(q_vec[0]) if endpoint_left_found else float("nan"),
        float(q_vec[-1]) - q_end_right if endpoint_right_found else float("nan"),
    ]
    return {
        "summary": {
            "q_min_new": float(q_vec[0]),
            "q_max_new": float(q_vec[-1]),
            "n_q_new": int(q_vec.size),
            "dq_new": dq,
            "q_opt_new": q_opt,
            "delta_opt_new": delta_opt,
            "eta_new": eta,
            "Ic_plus_new": ic_plus,
            "Ic_minus_new": ic_minus,
            "q_Ic_plus_new": q_plus,
            "q_Ic_minus_new": q_minus,
            "idx_q_opt": idx_q_opt,
            "idx_Ic_plus": idx_plus,
            "idx_Ic_minus": idx_minus,
            "left_endpoint_found": bool(endpoint_left_found),
            "right_endpoint_found": bool(endpoint_right_found),
            "q_end_left": q_end_left,
            "q_end_right": q_end_right,
            "q_edge_margin_response": float(np.nanmin(np.asarray(margins + endpoint_margins, dtype=np.float64))),
        },
        "branch": {
            "q": q_vec,
            "Delta_star_q": delta_q,
            "F_min_q": omega,
            "F_normal_q": omega_normal,
            "DeltaF_q": deltaf_q,
            "I_q": j_q,
        },
    }


def run_qwindow(audit_root: Path, rank: int, world_size: int, device: str) -> None:
    config = json.loads((audit_root / "config" / "audit_config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(audit_root / "input_points" / "eta_positive_high_JA_selected.csv")
    idxs = _rank_slice(df.shape[0], rank, world_size)
    out_dir = audit_root / "raw_outputs" / "qwindow_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)
    qcfg = config["qwindow_audit"]
    rows: list[dict[str, object]] = []
    for i in idxs:
        row = df.iloc[int(i)]
        q_min_old = float(row.get("q_min", -1.0))
        q_max_old = float(row.get("q_max", 0.5))
        n_q_old = int(row.get("n_q", 400)) if np.isfinite(float(row.get("n_q", 400))) else 400
        width = q_max_old - q_min_old
        for level in qcfg["q_expansion_levels"]:
            pad = float(level["side_pad_width_factor"]) * width
            q_min_new = max(-float(qcfg["q_max_abs"]), q_min_old - pad)
            q_max_new = min(float(qcfg["q_max_abs"]), q_max_old + pad)
            n_q_new = max(3, int(round(n_q_old * int(level["n_q_multiplier"]))))
            t0 = time.perf_counter()
            result = _eval_branch(
                float(row["kBT"]),
                float(row["JA"]),
                q_min_new,
                q_max_new,
                n_q_new,
                device=device,
                delta_eps=float(qcfg["delta_eps"]),
                deltaf_tol=float(qcfg["endpoint_deltaf_tol"]),
            )
            elapsed = time.perf_counter() - t0
            summary = dict(result["summary"])
            branch = result["branch"]
            n_edge = int(qcfg["n_edge"])
            response_valid = (
                bool(summary["left_endpoint_found"])
                and bool(summary["right_endpoint_found"])
                and np.isfinite(float(summary["q_edge_margin_response"]))
                and float(summary["q_edge_margin_response"]) > n_edge * float(summary["dq_new"])
            )
            eta_old = float(row["eta"])
            eta_new = float(summary["eta_new"])
            eta_sign_stable = bool(np.sign(eta_old) == np.sign(eta_new))
            if eta_new > 0 and response_valid and eta_sign_stable:
                classification = "response_stable_positive"
            elif (not eta_sign_stable) or (abs(eta_new) < 0.25 * abs(eta_old)) or not response_valid:
                classification = "q_window_artifact" if not response_valid or not eta_sign_stable else "response_inconclusive"
            else:
                classification = "response_inconclusive"
            payload = {
                **summary,
                "point_index": int(i),
                "level": str(level["name"]),
                "kBT": float(row["kBT"]),
                "JA": float(row["JA"]),
                "eta_old": eta_old,
                "Ic_plus_old": float(row["Ic_plus"]),
                "Ic_minus_old": float(row["Ic_minus"]),
                "q_opt_old": float(row["q_opt"]),
                "Delta_opt_old": float(row["Delta_opt"]),
                "response_q_window_valid": bool(response_valid),
                "eta_sign_stable": eta_sign_stable,
                "eta_abs_change": abs(eta_new - eta_old),
                "qwindow_classification": classification,
                "elapsed_sec": elapsed,
                "rank": int(rank),
                "world_size": int(world_size),
                "hostname": socket.gethostname(),
            }
            rows.append(payload)
            branch_path = out_dir / f"qwindow_point{i:04d}_{level['name']}.npz"
            np.savez(branch_path, **branch, **{k: np.asarray([v]) for k, v in payload.items() if isinstance(v, (int, float, bool, str))})
    out = out_dir / f"qwindow_summary_rank{rank:03d}_of{world_size:03d}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


def _points_for_delta(audit_root: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(audit_root / "input_points" / "high_JA_kink_delta_selected.csv"),
        pd.read_csv(audit_root / "input_points" / "rerun_required_selected.csv"),
        pd.read_csv(audit_root / "input_points" / "clean_control_selected.csv"),
    ]
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["coord_key"] = [_coord_key(k, j) for k, j in zip(df["kBT"], df["JA"])]
    return df.drop_duplicates("coord_key", keep="first").copy()


def run_delta(audit_root: Path, rank: int, world_size: int, device: str) -> None:
    from eta_phase_diagram_cuda import EtaPhaseConfig
    from ml_phase.exact_oracle import evaluate_points

    config = json.loads((audit_root / "config" / "audit_config.json").read_text(encoding="utf-8"))
    dcfg = config["delta_audit"]
    df = _points_for_delta(audit_root)
    idxs = _rank_slice(df.shape[0], rank, world_size)
    pts = df.iloc[idxs][["kBT", "JA"]].to_numpy(dtype=np.float64)
    out_dir = audit_root / "raw_outputs" / "delta_refine_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"delta_refine_rank{rank:03d}_of{world_size:03d}.npz"
    if pts.shape[0] == 0:
        np.savez(out_npz, kT=np.empty(0), JA=np.empty(0))
        return
    result = evaluate_points(
        points=pts,
        cfg=EtaPhaseConfig(),
        device=device,
        save_every=1,
        output_file=out_npz,
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
    np.savez(out_npz, **payload)


def _load_delta_results(audit_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((audit_root / "raw_outputs" / "delta_refine_rerun").glob("delta_refine_rank*.npz")):
        with np.load(path, allow_pickle=True) as z:
            if "kT" not in z.files or z["kT"].size == 0:
                continue
            df = pd.DataFrame(
                {
                    "kBT": z["kT"],
                    "JA": z["JA"],
                    "eta_new": z["eta"],
                    "q_opt_new": z["q_opt"],
                    "Delta_opt_new": z["delta_opt"],
                    "Ic_plus_new": z["ic_plus"],
                    "Ic_minus_new": z["ic_minus"],
                    "phase_candidate_new": z["phase_candidate"],
                    "delta_status_new": z["delta_status"],
                    "delta_ambiguous_new": z["delta_boundary_ambiguous"],
                    "delta_unresolved_new": z["delta_unresolved"],
                    "free_energy_gap_to_normal_new": z["free_energy_gap_to_normal"],
                    "positive_delta_gap_new": z["positive_delta_gap"],
                    "trusted_exact_new": z["trusted_exact"],
                    "q_status_new": z["q_status"],
                    "q_expanded_new": z["q_expanded"],
                    "q_unresolved_new": z["q_unresolved"],
                    "exact_status_code_new": z["exact_status_code"],
                }
            )
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_results(audit_root: Path) -> None:
    tables = audit_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    q_summaries = sorted((audit_root / "raw_outputs" / "qwindow_rerun").glob("qwindow_summary_rank*.csv"))
    if q_summaries:
        qdf = pd.concat([pd.read_csv(p) for p in q_summaries], ignore_index=True)
        qdf.to_csv(tables / "qwindow_comparison.csv", index=False)
    else:
        qdf = pd.DataFrame()
        pd.DataFrame().to_csv(tables / "qwindow_comparison.csv", index=False)

    old_delta = _points_for_delta(audit_root)
    new_delta = _load_delta_results(audit_root)
    if not new_delta.empty:
        old_delta["coord_key"] = [_coord_key(k, j) for k, j in zip(old_delta["kBT"], old_delta["JA"])]
        new_delta["coord_key"] = [_coord_key(k, j) for k, j in zip(new_delta["kBT"], new_delta["JA"])]
        comp = old_delta.merge(new_delta, on="coord_key", how="left", suffixes=("_old", ""))
        comp["phase_new_strict"] = [
            _phase_from_delta_q(d, q, 1e-3, 1e-2)
            if (not bool(amb) and not bool(unres))
            else "boundary_ambiguous"
            for d, q, amb, unres in zip(comp["Delta_opt_new"], comp["q_opt_new"], comp["delta_ambiguous_new"], comp["delta_unresolved_new"])
        ]
        comp["phase_changed"] = comp["phase"].astype(str) != comp["phase_new_strict"].astype(str)
        comp.to_csv(tables / "delta_refine_comparison.csv", index=False)
    else:
        comp = pd.DataFrame()
        pd.DataFrame().to_csv(tables / "delta_refine_comparison.csv", index=False)

    summary = {
        "qwindow_rows": int(qdf.shape[0]),
        "qwindow_response_stable_positive": int((qdf.get("qwindow_classification", pd.Series(dtype=str)) == "response_stable_positive").sum()) if not qdf.empty else 0,
        "qwindow_artifact_or_inconclusive": int((qdf.get("qwindow_classification", pd.Series(dtype=str)) != "response_stable_positive").sum()) if not qdf.empty else 0,
        "delta_rows": int(comp.shape[0]),
        "delta_phase_changed": int(comp["phase_changed"].sum()) if "phase_changed" in comp else 0,
        "delta_still_ambiguous": int((comp.get("phase_new_strict", pd.Series(dtype=str)) == "boundary_ambiguous").sum()) if not comp.empty else 0,
    }
    _write_text_lf(tables / "combined_before_after_summary.csv", pd.DataFrame([summary]).to_csv(index=False))
    _write_final_report(audit_root, summary, qdf, comp)
    _write_figures(audit_root, qdf, comp)


def _write_pending_report(audit_root: Path, config: dict[str, object]) -> None:
    report = audit_root / "reports" / "numerical_audit_qwindow_delta_report.md"
    counts = config.get("subset_counts", {})
    text = f"""# Numerical Audit q-window/Delta Report

Status: input subsets prepared; exact rerun results are pending.

## Input subset counts

```json
{json.dumps(counts, indent=2)}
```

Run the q-window and Delta audit jobs, then run `scripts/collect_results.sh`.
"""
    _write_text_lf(report, text)


def _write_final_report(audit_root: Path, summary: dict[str, object], qdf: pd.DataFrame, comp: pd.DataFrame) -> None:
    lines = ["# Numerical Audit q-window/Delta Report", "", "Status: comparison tables generated.", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## q-window response audit")
    lines.append("")
    if qdf.empty:
        lines.append("No q-window rerun outputs were found.")
    else:
        lines.append(f"Rows: {qdf.shape[0]}. Class counts: `{qdf['qwindow_classification'].value_counts().to_dict()}`.")
        lines.append("A point is only called response-stable positive when eta remains positive and the response-level q-window validity test passes.")
    lines.append("")
    lines.append("## Delta refinement audit")
    lines.append("")
    if comp.empty:
        lines.append("No Delta rerun outputs were found.")
    else:
        lines.append(f"Rows: {comp.shape[0]}. Phase changed: {int(comp['phase_changed'].sum())}. New strict phase counts: `{comp['phase_new_strict'].value_counts().to_dict()}`.")
    lines.append("")
    lines.append("## Safety note")
    lines.append("")
    lines.append("These rerun results are not appended to active-learning datasets. They are audit-only outputs.")
    _write_text_lf(audit_root / "reports" / "numerical_audit_qwindow_delta_report.md", "\n".join(lines) + "\n")


def _write_figures(audit_root: Path, qdf: pd.DataFrame, comp: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig_dir = audit_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not qdf.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        for label, sub in qdf.groupby("qwindow_classification"):
            ax.scatter(sub["eta_old"], sub["eta_new"], s=24, label=label, alpha=0.8)
        ax.axhline(0.0, color="0.2", lw=0.8)
        ax.axvline(0.0, color="0.2", lw=0.8)
        ax.set_xlabel("old eta")
        ax.set_ylabel("expanded-window eta")
        ax.set_title("q-window eta sign stability")
        ax.legend(fontsize=8)
        fig.savefig(fig_dir / "qwindow_eta_sign_stability.png", dpi=220)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
        for label, sub in qdf.groupby("qwindow_classification"):
            ax.scatter(sub["kBT"], sub["JA"], s=28, label=label, alpha=0.8)
        ax.set_xlabel(r"$k_B T/t$")
        ax.set_ylabel(r"$J_A/t$")
        ax.set_title("High-JA eta before/after q-window audit")
        ax.legend(fontsize=8)
        fig.savefig(fig_dir / "eta_positive_high_JA_before_after.png", dpi=220)
        plt.close(fig)
    if not comp.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        ax.scatter(comp["Delta_opt"], comp["Delta_opt_new"], s=18, alpha=0.8)
        ax.axline((0, 0), slope=1, color="0.3", lw=0.8)
        ax.set_xlabel("old Delta")
        ax.set_ylabel("strict-rerun Delta")
        ax.set_title("Delta boundary shift check")
        fig.savefig(fig_dir / "delta_boundary_shift_check.png", dpi=220)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
        for label, sub in comp.groupby("phase_new_strict"):
            ax.scatter(sub["kBT"], sub["JA"], s=26, label=label, alpha=0.8)
        ax.set_xlabel(r"$k_B T/t$")
        ax.set_ylabel(r"$J_A/t$")
        ax.set_title("High-JA kink before/after Delta audit")
        ax.legend(fontsize=8)
        fig.savefig(fig_dir / "high_JA_kink_before_after.png", dpi=220)
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independent q-window/Delta numerical audit harness.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--ml-phase-root", type=Path, required=True)
    s.add_argument("--run-id", type=str, default=RUN_ID_DEFAULT)
    s.add_argument("--audit-name", type=str, default=AUDIT_NAME_DEFAULT)
    for name in ["run-qwindow", "run-delta", "collect"]:
        sp = sub.add_parser(name)
        sp.add_argument("--audit-root", type=Path, required=True)
        if name != "collect":
            sp.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
            sp.add_argument("--world-size", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
            sp.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "setup":
        out = setup_audit(args.ml_phase_root, args.run_id, args.audit_name)
        print(f"Prepared numerical audit folder: {out}")
    elif args.cmd == "run-qwindow":
        run_qwindow(args.audit_root, args.rank, args.world_size, args.device)
    elif args.cmd == "run-delta":
        run_delta(args.audit_root, args.rank, args.world_size, args.device)
    elif args.cmd == "collect":
        collect_results(args.audit_root)


if __name__ == "__main__":
    main()
