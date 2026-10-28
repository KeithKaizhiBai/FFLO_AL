from __future__ import annotations

import argparse
import json
import math
import socket
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


POINT_IDS = [11, 13, 17, 20, 21, 25]
NQ_LEVELS = [3200, 6400, 12800]
CURVE_SAVE_POINTS = [11, 21, 25]
CLASS_ORDER = [
    "density-converged large positive eta",
    "weak near-zero positive eta",
    "sign-changing artifact",
    "q-extremum-location unstable",
    "unresolved",
]
AUDIT_NAME_DEFAULT = "numerical_audit_qdensity_v1"


def _write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _rank_slice(n: int, rank: int, world_size: int) -> np.ndarray:
    return np.arange(n, dtype=int)[int(rank) :: int(world_size)]


def _eta_from_ic(ic_plus: float, ic_minus: float) -> float:
    if ic_plus == 0 and ic_minus == 0:
        return 0.0
    if ic_plus == 0:
        return -1.0
    if ic_minus == 0:
        return 1.0
    return float((abs(ic_plus) - abs(ic_minus)) / (abs(ic_plus) + abs(ic_minus)))


def _ic_points(j_q: np.ndarray, q_vec: np.ndarray, idx_q_opt: int) -> tuple[float, float, float, float, int, int]:
    left = j_q[: idx_q_opt + 1]
    right = j_q[idx_q_opt:]
    idx_plus = int(np.argmax(left))
    idx_minus = int(idx_q_opt + np.argmin(right))
    return (
        float(j_q[idx_plus]),
        float(j_q[idx_minus]),
        float(q_vec[idx_plus]),
        float(q_vec[idx_minus]),
        idx_plus,
        idx_minus,
    )


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


def _local_extrema(q: np.ndarray, current: np.ndarray, mask: np.ndarray, topn: int = 5) -> dict[str, np.ndarray]:
    maxima: list[int] = []
    minima: list[int] = []
    for i in range(1, len(current) - 1):
        if not bool(mask[i]):
            continue
        if current[i] >= current[i - 1] and current[i] >= current[i + 1]:
            maxima.append(i)
        if current[i] <= current[i - 1] and current[i] <= current[i + 1]:
            minima.append(i)
    maxima = sorted(maxima, key=lambda i: current[i], reverse=True)[:topn]
    minima = sorted(minima, key=lambda i: current[i])[:topn]
    return {
        "top_max_indices": np.asarray(maxima, dtype=np.int64),
        "top_max_q": np.asarray([q[i] for i in maxima], dtype=np.float64),
        "top_max_I": np.asarray([current[i] for i in maxima], dtype=np.float64),
        "top_min_indices": np.asarray(minima, dtype=np.int64),
        "top_min_q": np.asarray([q[i] for i in minima], dtype=np.float64),
        "top_min_I": np.asarray([current[i] for i in minima], dtype=np.float64),
    }


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
    current = compute_current_from_omega(omega.reshape(1, 1, -1), q_vec)[0, 0]

    idx_q_opt = int(np.argmin(np.abs(q_vec - q_opt)))
    ic_plus, ic_minus, q_plus, q_minus, idx_plus, idx_minus = _ic_points(current, q_vec, idx_q_opt)
    eta = _eta_from_ic(ic_plus, ic_minus)
    left_idx, right_idx = _endpoint_indices(delta_q, deltaf_q, idx_q_opt, delta_eps, deltaf_tol)
    dq = float(np.min(np.diff(q_vec))) if q_vec.size > 1 else float("nan")
    left_found = left_idx >= 0
    right_found = right_idx >= 0
    q_left = float(q_vec[left_idx]) if left_found else float("nan")
    q_right = float(q_vec[right_idx]) if right_found else float("nan")
    margins = [
        q_opt - float(q_vec[0]),
        float(q_vec[-1]) - q_opt,
        q_plus - float(q_vec[0]) if np.isfinite(q_plus) else float("nan"),
        float(q_vec[-1]) - q_plus if np.isfinite(q_plus) else float("nan"),
        q_minus - float(q_vec[0]) if np.isfinite(q_minus) else float("nan"),
        float(q_vec[-1]) - q_minus if np.isfinite(q_minus) else float("nan"),
        q_left - float(q_vec[0]) if left_found else float("nan"),
        float(q_vec[-1]) - q_right if right_found else float("nan"),
    ]
    branch_mask = (delta_q >= float(delta_eps)) & (deltaf_q < -abs(float(deltaf_tol)))
    return {
        "summary": {
            "q_min": float(q_vec[0]),
            "q_max": float(q_vec[-1]),
            "nq": int(q_vec.size),
            "dq": dq,
            "q_opt": q_opt,
            "Delta_opt": delta_opt,
            "q_Ic_plus": q_plus,
            "q_Ic_minus": q_minus,
            "Ic_plus": ic_plus,
            "Ic_minus": ic_minus,
            "eta": eta,
            "idx_q_opt": idx_q_opt,
            "idx_Ic_plus": idx_plus,
            "idx_Ic_minus": idx_minus,
            "left_endpoint_found": bool(left_found),
            "right_endpoint_found": bool(right_found),
            "q_left_endpoint": q_left,
            "q_right_endpoint": q_right,
            "endpoint_margin_left": q_left - float(q_vec[0]) if left_found else float("nan"),
            "endpoint_margin_right": float(q_vec[-1]) - q_right if right_found else float("nan"),
            "q_edge_margin_response": float(np.nanmin(np.asarray(margins, dtype=np.float64))),
        },
        "branch": {
            "q_grid": q_vec,
            "F_q": omega,
            "Delta_q": delta_q,
            "I_q": current,
            "F_normal_q": omega_normal,
            "DeltaF_q": deltaf_q,
            "branch_valid_mask": branch_mask.astype(np.int8),
        },
        "extrema": _local_extrema(q_vec, current, branch_mask),
    }


def _default_config(source_root: Path) -> dict[str, object]:
    return {
        "source_root": str(source_root),
        "source_qwindow_table": str(source_root / "tables" / "qwindow_comparison_complete_of004.csv"),
        "points": POINT_IDS,
        "fixed_window_level": "expand_1p0_width",
        "nq_levels": NQ_LEVELS,
        "curve_save_points": CURVE_SAVE_POINTS,
        "delta_eps": 1.0e-3,
        "endpoint_deltaf_tol": 1.0e-8,
        "n_edge": 5,
        "convergence": {
            "eta_abs_tol": 0.02,
            "eta_rel_tol": 0.10,
            "ic_rel_tol": 0.10,
            "q_shift_tol_dq": 2.0,
        },
        "safety": {
            "audit_only": True,
            "do_not_append_training_data": True,
            "do_not_modify_active_learning": True,
            "do_not_modify_production_oracle": True,
        },
    }


def setup_audit(source_root: Path, audit_root: Path) -> None:
    cfg = _default_config(source_root)
    table = pd.read_csv(cfg["source_qwindow_table"])
    rows = table[(table["point_index"].isin(POINT_IDS)) & (table["level"] == cfg["fixed_window_level"])].copy()
    if rows.shape[0] != len(POINT_IDS):
        raise ValueError(f"Expected {len(POINT_IDS)} fixed-window rows, found {rows.shape[0]}")
    rows = rows.sort_values("point_index")
    out = pd.DataFrame(
        {
            "point_id": rows["point_index"].astype(int),
            "kBT": rows["kBT"].astype(float),
            "JA": rows["JA"].astype(float),
            "q_min": rows["q_min_new"].astype(float),
            "q_max": rows["q_max_new"].astype(float),
            "source_eta_old": rows["eta_old"].astype(float),
            "source_eta_expand_1p0": rows["eta_new"].astype(float),
        }
    )
    for sub in ["config", "input_points", "raw_outputs/qdensity_rerun", "tables", "curves", "figures", "reports", "scripts"]:
        (audit_root / sub).mkdir(parents=True, exist_ok=True)
    out.to_csv(audit_root / "input_points" / "qdensity_positive_eta_points.csv", index=False)
    _write_text_lf(audit_root / "config" / "qdensity_config.json", json.dumps(cfg, indent=2) + "\n")
    _write_helpers(audit_root)
    _write_pending_report(audit_root, out, cfg)


def _write_helpers(audit_root: Path) -> None:
    scripts_dir = audit_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    prelude = """set -euo pipefail
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
    submit = """#!/bin/bash
#SBATCH --job-name=aud_qdens
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --exclude=gpuh01

""" + prelude + """
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/numerical_audit_qdensity.py" run \\
  --audit-root "${AUDIT_ROOT}" \\
  --rank "${SLURM_ARRAY_TASK_ID}" \\
  --world-size "${SLURM_ARRAY_TASK_COUNT}" \\
  --device cuda:0
"""
    collect = """#!/bin/bash
""" + prelude + """
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/numerical_audit_qdensity.py" collect --audit-root "${AUDIT_ROOT}"
"""
    _write_text_lf(scripts_dir / "submit_qdensity_array.sh", submit)
    _write_text_lf(scripts_dir / "collect_qdensity_results.sh", collect)
    source = Path(__file__).resolve()
    target = scripts_dir / "numerical_audit_qdensity.py"
    if source != target.resolve():
        target.write_bytes(source.read_bytes())


def _write_pending_report(audit_root: Path, points: pd.DataFrame, cfg: dict[str, object]) -> None:
    text = f"""# q-density Convergence Audit

Status: input points prepared; fixed-window q-density reruns are pending.

Points:

```text
{points.to_string(index=False)}
```

Nq levels: `{cfg['nq_levels']}`.
"""
    _write_text_lf(audit_root / "reports" / "qdensity_convergence_report.md", text)


def _task_table(audit_root: Path) -> pd.DataFrame:
    cfg = json.loads((audit_root / "config" / "qdensity_config.json").read_text(encoding="utf-8"))
    points = pd.read_csv(audit_root / "input_points" / "qdensity_positive_eta_points.csv")
    rows: list[dict[str, object]] = []
    for _, row in points.iterrows():
        for nq in cfg["nq_levels"]:
            rows.append({**row.to_dict(), "nq": int(nq)})
    return pd.DataFrame(rows)


def run_tasks(audit_root: Path, rank: int, world_size: int, device: str) -> None:
    cfg = json.loads((audit_root / "config" / "qdensity_config.json").read_text(encoding="utf-8"))
    tasks = _task_table(audit_root)
    idxs = _rank_slice(tasks.shape[0], rank, world_size)
    out_dir = audit_root / "raw_outputs" / "qdensity_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = audit_root / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for idx in idxs:
        task = tasks.iloc[int(idx)]
        t0 = time.perf_counter()
        payload: dict[str, object] = {
            "point_id": int(task["point_id"]),
            "kBT": float(task["kBT"]),
            "JA": float(task["JA"]),
            "q_min": float(task["q_min"]),
            "q_max": float(task["q_max"]),
            "nq": int(task["nq"]),
            "rank": int(rank),
            "world_size": int(world_size),
            "hostname": socket.gethostname(),
            "status": "ok",
            "failure_reason": "N/A",
        }
        try:
            result = _eval_branch(
                float(task["kBT"]),
                float(task["JA"]),
                float(task["q_min"]),
                float(task["q_max"]),
                int(task["nq"]),
                device=device,
                delta_eps=float(cfg["delta_eps"]),
                deltaf_tol=float(cfg["endpoint_deltaf_tol"]),
            )
            summary = dict(result["summary"])
            summary["response_window_valid"] = (
                bool(summary["left_endpoint_found"])
                and bool(summary["right_endpoint_found"])
                and np.isfinite(float(summary["q_edge_margin_response"]))
                and float(summary["q_edge_margin_response"]) > float(cfg["n_edge"]) * float(summary["dq"])
            )
            payload.update(summary)
            payload["elapsed_sec"] = time.perf_counter() - t0
            if int(task["point_id"]) in [int(x) for x in cfg["curve_save_points"]]:
                branch = result["branch"]
                extrema = result["extrema"]
                np.savez(
                    curve_dir / f"point{int(task['point_id']):04d}_nq{int(task['nq'])}_response.npz",
                    **branch,
                    **extrema,
                    **{k: np.asarray([v]) for k, v in payload.items() if isinstance(v, (int, float, bool, str))},
                )
                pd.DataFrame(
                    [
                        {"kind": "max", "rank": i + 1, "index": int(idx), "q": float(q), "I": float(cur)}
                        for i, (idx, q, cur) in enumerate(zip(extrema["top_max_indices"], extrema["top_max_q"], extrema["top_max_I"]))
                    ]
                    + [
                        {"kind": "min", "rank": i + 1, "index": int(idx), "q": float(q), "I": float(cur)}
                        for i, (idx, q, cur) in enumerate(zip(extrema["top_min_indices"], extrema["top_min_q"], extrema["top_min_I"]))
                    ]
                ).to_csv(curve_dir / f"point{int(task['point_id']):04d}_nq{int(task['nq'])}_top_extrema.csv", index=False)
        except Exception as exc:
            payload["status"] = "failed"
            payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
            payload["elapsed_sec"] = time.perf_counter() - t0
        rows.append(payload)
    pd.DataFrame(rows).to_csv(out_dir / f"qdensity_summary_rank{rank:03d}_of{world_size:03d}.csv", index=False)


def _rel_change(new: float, old: float) -> float:
    denom = max(abs(float(old)), 1.0e-30)
    return float(abs(float(new) - float(old)) / denom)


def collect_results(audit_root: Path) -> None:
    cfg = json.loads((audit_root / "config" / "qdensity_config.json").read_text(encoding="utf-8"))
    out_dir = audit_root / "raw_outputs" / "qdensity_rerun"
    tables = audit_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summaries = sorted(out_dir.glob("qdensity_summary_rank*_of*.csv"))
    if not summaries:
        raise FileNotFoundError(f"No qdensity summary CSVs found in {out_dir}")
    df = pd.concat([pd.read_csv(p) for p in summaries], ignore_index=True)
    df = df.sort_values(["point_id", "nq"]).reset_index(drop=True)
    conv_rows: list[dict[str, object]] = []
    for point_id, sub in df.groupby("point_id", sort=True):
        sub = sub.sort_values("nq")
        prev = None
        for _, row in sub.iterrows():
            payload = row.to_dict()
            if prev is None or row.get("status") != "ok" or prev.get("status") != "ok":
                payload.update(
                    {
                        "eta_sign_stable": "N/A",
                        "eta_abs_change_from_previous_nq": "N/A",
                        "eta_rel_change_from_previous_nq": "N/A",
                        "Ic_plus_rel_change_from_previous_nq": "N/A",
                        "Ic_minus_rel_change_from_previous_nq": "N/A",
                        "q_Ic_plus_shift_in_fine_dq": "N/A",
                        "q_Ic_minus_shift_in_fine_dq": "N/A",
                        "q_density_valid": False,
                    }
                )
            else:
                eta_abs = abs(float(row["eta"]) - float(prev["eta"]))
                eta_rel = _rel_change(float(row["eta"]), float(prev["eta"]))
                icp_rel = _rel_change(float(row["Ic_plus"]), float(prev["Ic_plus"]))
                icm_rel = _rel_change(float(row["Ic_minus"]), float(prev["Ic_minus"]))
                qp_shift = abs(float(row["q_Ic_plus"]) - float(prev["q_Ic_plus"])) / float(row["dq"])
                qm_shift = abs(float(row["q_Ic_minus"]) - float(prev["q_Ic_minus"])) / float(row["dq"])
                eta_sign = bool(np.sign(float(row["eta"])) == np.sign(float(prev["eta"])))
                q_valid = (
                    eta_sign
                    and (eta_abs < float(cfg["convergence"]["eta_abs_tol"]) or eta_rel < float(cfg["convergence"]["eta_rel_tol"]))
                    and icp_rel < float(cfg["convergence"]["ic_rel_tol"])
                    and icm_rel < float(cfg["convergence"]["ic_rel_tol"])
                    and qp_shift < float(cfg["convergence"]["q_shift_tol_dq"])
                    and qm_shift < float(cfg["convergence"]["q_shift_tol_dq"])
                    and bool(row["response_window_valid"])
                    and float(row["endpoint_margin_left"]) > float(cfg["n_edge"]) * float(row["dq"])
                    and float(row["endpoint_margin_right"]) > float(cfg["n_edge"]) * float(row["dq"])
                )
                payload.update(
                    {
                        "eta_sign_stable": eta_sign,
                        "eta_abs_change_from_previous_nq": eta_abs,
                        "eta_rel_change_from_previous_nq": eta_rel,
                        "Ic_plus_rel_change_from_previous_nq": icp_rel,
                        "Ic_minus_rel_change_from_previous_nq": icm_rel,
                        "q_Ic_plus_shift_in_fine_dq": qp_shift,
                        "q_Ic_minus_shift_in_fine_dq": qm_shift,
                        "q_density_valid": bool(q_valid),
                    }
                )
            conv_rows.append(payload)
            prev = row
    conv = pd.DataFrame(conv_rows)
    conv["eta_change_from_previous_nq"] = conv["eta_abs_change_from_previous_nq"]
    summary = _summary_by_point(conv, cfg)
    class_map = {int(row["point_id"]): row["classification"] for _, row in summary.iterrows()}
    conv["classification"] = conv["point_id"].map(lambda x: class_map.get(int(x), "unresolved"))
    conv.to_csv(tables / "qdensity_convergence.csv", index=False)
    summary.to_csv(tables / "qdensity_summary_by_point.csv", index=False)
    _write_check_summary(audit_root, conv, summary)
    _write_figures(audit_root, conv)
    _write_report(audit_root, conv, summary)


def _summary_by_point(conv: pd.DataFrame, cfg: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for point_id, sub in conv.groupby("point_id", sort=True):
        sub = sub.sort_values("nq")
        row6400 = sub[sub["nq"] == 6400]
        row12800 = sub[sub["nq"] == 12800]
        if row6400.empty or row12800.empty:
            cls = "unresolved"
            reason = "missing 6400 or 12800 row"
        else:
            r = row12800.iloc[0]
            eta128 = float(r["eta"]) if r["status"] == "ok" else float("nan")
            eta640 = float(row6400.iloc[0]["eta"]) if row6400.iloc[0]["status"] == "ok" else float("nan")
            if r["status"] != "ok":
                cls = "unresolved"
                reason = str(r.get("failure_reason", "failed"))
            elif np.sign(eta128) != np.sign(eta640):
                cls = "sign-changing artifact"
                reason = "eta sign changes between nq=6400 and nq=12800"
            elif str(r.get("q_Ic_plus_shift_in_fine_dq", "N/A")) != "N/A" and (
                float(r["q_Ic_plus_shift_in_fine_dq"]) >= float(cfg["convergence"]["q_shift_tol_dq"])
                or float(r["q_Ic_minus_shift_in_fine_dq"]) >= float(cfg["convergence"]["q_shift_tol_dq"])
            ):
                cls = "q-extremum-location unstable"
                reason = "critical-current extremum shifts exceed tolerance"
            elif eta128 > 0 and bool(r["q_density_valid"]):
                cls = "density-converged large positive eta"
                reason = "passes current q-density convergence screen with positive eta"
            elif eta128 > 0 and abs(eta128) < float(cfg["convergence"]["eta_abs_tol"]):
                cls = "weak near-zero positive eta"
                reason = "positive eta_12800 is below the absolute eta tolerance"
            else:
                cls = "unresolved"
                reason = "fails one or more convergence tolerances"
        last = sub.iloc[-1]
        rows.append(
            {
                "point_id": int(point_id),
                "kBT": last.get("kBT", "N/A"),
                "JA": last.get("JA", "N/A"),
                "eta_3200": _value_at(sub, 3200, "eta"),
                "eta_6400": _value_at(sub, 6400, "eta"),
                "eta_12800": _value_at(sub, 12800, "eta"),
                "classification": cls,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _write_check_summary(audit_root: Path, conv: pd.DataFrame, summary: pd.DataFrame) -> None:
    tables = audit_root / "tables"
    expected_points = POINT_IDS
    expected_nq = NQ_LEVELS
    expected_curve_points = CURVE_SAVE_POINTS
    expected_rows = len(expected_points) * len(expected_nq)
    actual_pairs = {(int(row["point_id"]), int(row["nq"])) for _, row in conv.iterrows()}
    missing_pairs = [
        {"point_id": point_id, "nq": nq}
        for point_id in expected_points
        for nq in expected_nq
        if (point_id, nq) not in actual_pairs
    ]
    curve_dir = audit_root / "curves"
    missing_curve_files: list[str] = []
    for point_id in expected_curve_points:
        for nq in expected_nq:
            for suffix in ["response.npz", "top_extrema.csv"]:
                name = f"point{point_id:04d}_nq{nq}_{suffix}"
                if not (curve_dir / name).exists():
                    missing_curve_files.append(name)
    payload = {
        "expected_rows": expected_rows,
        "actual_rows": int(conv.shape[0]),
        "status_counts": conv["status"].value_counts(dropna=False).to_dict() if "status" in conv else {},
        "missing_point_nq_pairs": missing_pairs,
        "classification_counts": {label: int((summary["classification"] == label).sum()) for label in CLASS_ORDER},
        "expected_curve_points": expected_curve_points,
        "missing_curve_files": missing_curve_files,
    }
    _write_text_lf(tables / "qdensity_check_summary.json", json.dumps(payload, indent=2) + "\n")


def _value_at(df: pd.DataFrame, nq: int, col: str) -> object:
    row = df[df["nq"] == nq]
    if row.empty or col not in row:
        return "N/A"
    value = row.iloc[0][col]
    if pd.isna(value):
        return "N/A"
    return value


def _write_figures(audit_root: Path, conv: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig_dir = audit_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ok = conv[conv["status"] == "ok"].copy()
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for point_id, sub in ok.groupby("point_id"):
        ax.plot(sub["nq"], sub["eta"], marker="o", label=f"p{int(point_id)}")
    ax.axhline(0, color="0.25", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nq")
    ax.set_ylabel(r"$\eta$")
    ax.set_title("q-density eta convergence")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(fig_dir / "eta_vs_nq.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for point_id, sub in ok.groupby("point_id"):
        ax.plot(sub["nq"], sub["Ic_plus"].abs(), marker="o", label=f"|Ic+| p{int(point_id)}")
        ax.plot(sub["nq"], sub["Ic_minus"].abs(), marker="x", ls="--", label=f"|Ic-| p{int(point_id)}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nq")
    ax.set_ylabel(r"$|I_c|$")
    ax.set_title("critical-current magnitude convergence")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(fig_dir / "Ic_vs_nq.png", dpi=240)
    plt.close(fig)

    shifts = ok[ok["q_Ic_plus_shift_in_fine_dq"].astype(str) != "N/A"].copy()
    if not shifts.empty:
        shifts["q_Ic_plus_shift_in_fine_dq"] = shifts["q_Ic_plus_shift_in_fine_dq"].astype(float)
        shifts["q_Ic_minus_shift_in_fine_dq"] = shifts["q_Ic_minus_shift_in_fine_dq"].astype(float)
        fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
        for point_id, sub in shifts.groupby("point_id"):
            ax.plot(sub["nq"], sub["q_Ic_plus_shift_in_fine_dq"], marker="o", label=f"Ic+ p{int(point_id)}")
            ax.plot(sub["nq"], sub["q_Ic_minus_shift_in_fine_dq"], marker="x", ls="--", label=f"Ic- p{int(point_id)}")
        ax.axhline(2.0, color="0.3", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("nq")
        ax.set_ylabel("shift / fine dq")
        ax.set_title("q-Ic location shift")
        ax.legend(fontsize=6, ncol=2)
        fig.savefig(fig_dir / "qIc_shift_vs_nq.png", dpi=240)
        plt.close(fig)


def _write_report(audit_root: Path, conv: pd.DataFrame, summary: pd.DataFrame) -> None:
    check = json.loads((audit_root / "tables" / "qdensity_check_summary.json").read_text(encoding="utf-8"))
    lines = ["# Fixed-window q-density convergence report", ""]
    lines.append("Status: fixed-window q-density reruns collected and classified.")
    lines.append("")
    lines.append("This is an audit-only report. No active-learning dataset is modified.")
    lines.append("")
    lines.append("The response-level q-window is fixed to the previous `expand_1.0` window. The audit changes only the q-grid density: `nq = 3200, 6400, 12800`.")
    lines.append("")
    lines.append("## Data completeness")
    lines.append("")
    lines.append(f"- Expected point/nq rows: `{check['expected_rows']}`")
    lines.append(f"- Collected point/nq rows: `{check['actual_rows']}`")
    lines.append(f"- Status counts: `{check['status_counts']}`")
    lines.append(f"- Missing point/nq pairs: `{check['missing_point_nq_pairs']}`")
    lines.append("")
    if check["missing_curve_files"]:
        lines.append("Curve-output warning: the returned run does not include all requested curve files. In particular, point 25 curves are absent because the submitted config saved curves only for points 11 and 21.")
        lines.append("")
        lines.append("Missing curve files:")
        lines.append("")
        for name in check["missing_curve_files"]:
            lines.append(f"- `{name}`")
        lines.append("")
    lines.append("## Classification summary")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|---|---:|")
    for label in CLASS_ORDER:
        lines.append(f"| {label} | {int((summary['classification'] == label).sum())} |")
    lines.append("")
    lines.append("## Grouped results")
    lines.append("")
    cols = ["point_id", "kBT", "JA", "eta_3200", "eta_6400", "eta_12800", "classification", "reason"]
    for label in CLASS_ORDER:
        sub = summary[summary["classification"] == label]
        lines.append(f"### {label}")
        lines.append("")
        if sub.empty:
            lines.append("No points in this group.")
            lines.append("")
            continue
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in sub[cols].iterrows():
            lines.append("| " + " | ".join(_format_md_value(row[col]) for col in cols) + " |")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("A positive eta candidate is considered stable only if the `nq=6400` and `nq=12800` rows keep the same sign, the eta change is small, both critical-current magnitudes are stable, the critical-current extrema move by less than two fine-grid spacings, and the response endpoints stay more than five fine-grid spacings from the fixed q-window edges.")
    lines.append("")
    lines.append("Several rows have saturated eta values of `+1` or `-1` because one critical-current branch is numerically zero on that grid. These cases should be interpreted as response-extraction instability unless the q-density convergence checks pass.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `tables/qdensity_convergence.csv`")
    lines.append("- `tables/qdensity_summary_by_point.csv`")
    lines.append("- `tables/qdensity_check_summary.json`")
    lines.append("- `figures/eta_vs_nq.png`")
    lines.append("- `figures/Ic_vs_nq.png`")
    lines.append("- `figures/qIc_shift_vs_nq.png`")
    _write_text_lf(audit_root / "reports" / "qdensity_convergence_report.md", "\n".join(lines) + "\n")
    _write_latex_report(audit_root, summary, check)


def _format_md_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _latex_escape(text: object) -> str:
    value = str(text)
    replacements = {
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
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _write_latex_report(audit_root: Path, summary: pd.DataFrame, check: dict[str, object]) -> None:
    tex: list[str] = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{float}",
        r"\title{Fixed-window q-density convergence audit}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Scope}",
        "This audit keeps the response-level q-window fixed to the previous expand\\_1.0 window and changes only the q-grid density. It does not modify the active-learning dataset, acquisition function, StopController, neural network, or production exact-oracle workflow.",
        r"\section*{Data completeness}",
        r"\begin{itemize}",
        f"\\item Expected point/nq rows: {int(check['expected_rows'])}",
        f"\\item Collected point/nq rows: {int(check['actual_rows'])}",
        f"\\item Missing point/nq pairs: {len(check['missing_point_nq_pairs'])}",
        f"\\item Missing requested curve files: {len(check['missing_curve_files'])}",
        r"\end{itemize}",
    ]
    if check["missing_curve_files"]:
        tex.extend(
            [
                r"\noindent The returned run does not include all requested curve files. Point 25 curves are missing because the submitted configuration saved full response curves only for points 11 and 21.",
                "",
            ]
        )
    tex.extend(
        [
            r"\section*{Classification summary}",
            r"\begin{center}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Class & Count \\",
            r"\midrule",
        ]
    )
    for label in CLASS_ORDER:
        tex.append(f"{_latex_escape(label)} & {int((summary['classification'] == label).sum())} \\\\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    tex.extend(
        [
            r"\section*{Per-point results}",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{rrrrlp{0.40\linewidth}}",
            r"\toprule",
            r"Point & $k_BT/t$ & $J_A/t$ & $\eta_{6400}$ & $\eta_{12800}$ & Classification \\",
            r"\midrule",
        ]
    )
    for _, row in summary.iterrows():
        tex.append(
            f"{int(row['point_id'])} & {float(row['kBT']):.6g} & {float(row['JA']):.6g} & "
            f"{float(row['eta_6400']):.6g} & {float(row['eta_12800']):.6g} & "
            f"{_latex_escape(row['classification'])} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    for fig_name, caption in [
        ("eta_vs_nq.png", r"$\eta$ as a function of q-grid density."),
        ("Ic_vs_nq.png", r"Critical-current magnitudes as a function of q-grid density."),
        ("qIc_shift_vs_nq.png", r"Critical-current extremum location shifts in units of the finer grid spacing."),
    ]:
        if (audit_root / "figures" / fig_name).exists():
            tex.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.78\linewidth]{{../figures/{fig_name}}}",
                    rf"\caption{{{caption}}}",
                    r"\end{figure}",
                ]
            )
    tex.extend(
        [
            r"\section*{Interpretation}",
            r"A residual positive-\(\eta\) point is accepted as density-converged only if the \(n_q=6400\) and \(n_q=12800\) rows keep the same sign, \(\eta\), \(I_c^+\), and \(I_c^-\) are stable within tolerance, the extrema positions shift by less than two fine-grid spacings, and the superconducting endpoints remain more than five fine-grid spacings away from the fixed q-window edges.",
            r"Rows with \(\eta=\pm 1\) often indicate that one critical-current branch is numerically zero on that grid. These saturated values are not by themselves evidence for a robust diode response.",
            r"\end{document}",
        ]
    )
    _write_text_lf(audit_root / "reports" / "qdensity_convergence_report.tex", "\n".join(tex) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit-only fixed-window q-density convergence test.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_setup = sub.add_parser("setup")
    p_setup.add_argument("--source-root", type=Path, required=True)
    p_setup.add_argument("--audit-root", type=Path, default=None)
    p_run = sub.add_parser("run")
    p_run.add_argument("--audit-root", type=Path, required=True)
    p_run.add_argument("--rank", type=int, required=True)
    p_run.add_argument("--world-size", type=int, required=True)
    p_run.add_argument("--device", type=str, default="cuda:0")
    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "setup":
        audit_root = args.audit_root or (args.source_root.parent / AUDIT_NAME_DEFAULT)
        setup_audit(args.source_root, audit_root)
        print(f"Prepared q-density audit folder: {audit_root}")
    elif args.cmd == "run":
        run_tasks(args.audit_root, args.rank, args.world_size, args.device)
    elif args.cmd == "collect":
        collect_results(args.audit_root)


if __name__ == "__main__":
    main()
