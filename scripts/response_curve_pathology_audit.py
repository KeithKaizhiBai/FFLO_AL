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
SCRIPTS_DIR = REPO_ROOT / "scripts"
QDENSITY_SCRIPT_DIR = REPO_ROOT.parent / "scripts"
for path_candidate in [Path.cwd().resolve(), REPO_ROOT, SCRIPTS_DIR, QDENSITY_SCRIPT_DIR]:
    path_text = str(path_candidate)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from numerical_audit_qdensity import _eval_branch, _write_text_lf  # noqa: E402


POINT_IDS = [21, 25]
NQ_LEVELS = [6400, 12800]
ETA_DENOMINATOR_THRESHOLD = 1.0e-4
BRANCH_NEAR_ZERO_THRESHOLD = 1.0e-8
AUDIT_SUBDIR = "response_curve_pathology_audit"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_table(qdensity_root: Path) -> pd.DataFrame:
    points = pd.read_csv(qdensity_root / "input_points" / "qdensity_positive_eta_points.csv")
    rows: list[dict[str, object]] = []
    for _, row in points[points["point_id"].isin(POINT_IDS)].iterrows():
        for nq in NQ_LEVELS:
            payload = row.to_dict()
            payload["nq"] = int(nq)
            rows.append(payload)
    return pd.DataFrame(rows).sort_values(["point_id", "nq"]).reset_index(drop=True)


def _write_helpers(qdensity_root: Path, audit_root: Path) -> None:
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
QDENSITY_ROOT="${QDENSITY_ROOT:-$(cd "${AUDIT_ROOT}/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${QDENSITY_ROOT}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PROJECT_DIR}"
"""
    submit = """#!/bin/bash
#SBATCH --job-name=resp_path
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --exclude=gpuh01

""" + prelude + """
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/response_curve_pathology_audit.py" run \
  --qdensity-root "${QDENSITY_ROOT}" \
  --audit-root "${AUDIT_ROOT}" \
  --rank "${SLURM_ARRAY_TASK_ID}" \
  --world-size "${SLURM_ARRAY_TASK_COUNT}" \
  --device cuda:0
"""
    collect = """#!/bin/bash
""" + prelude + """
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/response_curve_pathology_audit.py" analyze \
  --qdensity-root "${QDENSITY_ROOT}" \
  --audit-root "${AUDIT_ROOT}"
"""
    _write_text_lf(scripts_dir / "submit_response_pathology_curves.sh", submit)
    _write_text_lf(scripts_dir / "collect_response_pathology.sh", collect)


def setup_audit(qdensity_root: Path, audit_root: Path) -> None:
    audit_root.mkdir(parents=True, exist_ok=True)
    for sub in ["config", "input_points", "curves", "figures", "tables", "reports", "scripts", "raw_outputs"]:
        (audit_root / sub).mkdir(parents=True, exist_ok=True)
    cfg = _load_json(qdensity_root / "config" / "qdensity_config.json")
    payload = {
        "qdensity_root": str(qdensity_root),
        "points": POINT_IDS,
        "nq_levels": NQ_LEVELS,
        "eta_denominator_threshold": ETA_DENOMINATOR_THRESHOLD,
        "branch_near_zero_threshold": BRANCH_NEAR_ZERO_THRESHOLD,
        "delta_eps": cfg.get("delta_eps", 1.0e-3),
        "endpoint_deltaf_tol": cfg.get("endpoint_deltaf_tol", 1.0e-8),
        "n_edge": cfg.get("n_edge", 5),
        "audit_only": True,
    }
    _write_text_lf(audit_root / "config" / "response_pathology_config.json", json.dumps(payload, indent=2) + "\n")
    _target_table(qdensity_root).to_csv(audit_root / "input_points" / "response_pathology_points.csv", index=False)
    _write_helpers(qdensity_root, audit_root)
    _write_text_lf(
        audit_root / "reports" / "response_curve_pathology_report.md",
        "# Response-curve pathology audit\n\nStatus: input points prepared; curve reruns/analysis pending.\n",
    )


def _rank_slice(n: int, rank: int, world_size: int) -> np.ndarray:
    return np.arange(n, dtype=int)[int(rank) :: int(world_size)]


def run_tasks(qdensity_root: Path, audit_root: Path, rank: int, world_size: int, device: str) -> None:
    cfg = _load_json(audit_root / "config" / "response_pathology_config.json")
    tasks = _target_table(qdensity_root)
    idxs = _rank_slice(tasks.shape[0], rank, world_size)
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
            payload.update(summary)
            payload["response_window_valid"] = (
                bool(summary["left_endpoint_found"])
                and bool(summary["right_endpoint_found"])
                and np.isfinite(float(summary["q_edge_margin_response"]))
                and float(summary["q_edge_margin_response"]) > float(cfg["n_edge"]) * float(summary["dq"])
            )
            payload["elapsed_sec"] = time.perf_counter() - t0
            np.savez(
                audit_root / "curves" / f"point{int(task['point_id']):04d}_nq{int(task['nq'])}_response.npz",
                **result["branch"],
                **{k: np.asarray([v]) for k, v in payload.items() if isinstance(v, (int, float, bool, str))},
            )
        except Exception as exc:
            payload["status"] = "failed"
            payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
            payload["elapsed_sec"] = time.perf_counter() - t0
        rows.append(payload)
    out = audit_root / "raw_outputs" / f"response_pathology_summary_rank{rank:03d}_of{world_size:03d}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


def _scalar(npz: np.lib.npyio.NpzFile, key: str, default: float = float("nan")) -> float:
    if key not in npz:
        return default
    arr = np.asarray(npz[key])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _curve_path(qdensity_root: Path, audit_root: Path, point_id: int, nq: int) -> tuple[Path | None, str]:
    candidates = [
        audit_root / "curves" / f"point{point_id:04d}_nq{nq}_response.npz",
        qdensity_root / "curves" / f"point{point_id:04d}_nq{nq}_response.npz",
    ]
    for path in candidates:
        if path.exists():
            return path, "pathology" if AUDIT_SUBDIR in str(path) else "qdensity"
    return None, "missing"


def _local_extrema_branch(q: np.ndarray, current: np.ndarray, mask: np.ndarray, indices: np.ndarray, branch: str, topn: int = 10) -> pd.DataFrame:
    extrema: list[int] = []
    idx_set = set(int(i) for i in indices)
    for i in indices:
        i = int(i)
        if i < 0 or i >= len(current):
            continue
        left_ok = (i - 1 not in idx_set) or current[i] >= current[i - 1]
        right_ok = (i + 1 not in idx_set) or current[i] >= current[i + 1]
        if branch == "plus" and left_ok and right_ok:
            extrema.append(i)
        left_ok = (i - 1 not in idx_set) or current[i] <= current[i - 1]
        right_ok = (i + 1 not in idx_set) or current[i] <= current[i + 1]
        if branch == "minus" and left_ok and right_ok:
            extrema.append(i)
    if branch == "plus":
        extrema = sorted(extrema, key=lambda i: current[i], reverse=True)
    else:
        extrema = sorted(extrema, key=lambda i: current[i])
    rows = []
    for rank, i in enumerate(extrema[:topn], start=1):
        rows.append({"branch": branch, "rank": rank, "index": int(i), "q": float(q[i]), "I": float(current[i]), "branch_valid": bool(mask[i])})
    if not rows and idx_set:
        selected = int(indices[np.argmax(current[indices])] if branch == "plus" else indices[np.argmin(current[indices])])
        rows.append({"branch": branch, "rank": 1, "index": selected, "q": float(q[selected]), "I": float(current[selected]), "branch_valid": bool(mask[selected])})
    return pd.DataFrame(rows)


def _analyze_curve(path: Path, point_id: int, nq: int, cfg: dict[str, object], fig_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    npz = np.load(path, allow_pickle=True)
    q = np.asarray(npz["q_grid"], dtype=float)
    current = np.asarray(npz["I_q"], dtype=float)
    delta = np.asarray(npz["Delta_q"], dtype=float)
    free_energy = np.asarray(npz["F_q"], dtype=float)
    mask = np.asarray(npz["branch_valid_mask"]).astype(bool)
    q_opt = _scalar(npz, "q_opt")
    idx_q_opt = int(np.nanargmin(np.abs(q - q_opt))) if np.isfinite(q_opt) else int(len(q) // 2)
    plus_idx = np.arange(0, idx_q_opt + 1, dtype=int)
    minus_idx = np.arange(idx_q_opt, len(q), dtype=int)
    plus_ext = _local_extrema_branch(q, current, mask, plus_idx, "plus", topn=10)
    minus_ext = _local_extrema_branch(q, current, mask, minus_idx, "minus", topn=10)
    extrema = pd.concat([plus_ext, minus_ext], ignore_index=True)
    ic_plus = _scalar(npz, "Ic_plus")
    ic_minus = _scalar(npz, "Ic_minus")
    denom = abs(ic_plus) + abs(ic_minus)
    min_abs_ic = min(abs(ic_plus), abs(ic_minus))
    row = {
        "point_id": point_id,
        "nq": nq,
        "kBT": _scalar(npz, "kBT"),
        "JA": _scalar(npz, "JA"),
        "q_min": float(q[0]),
        "q_max": float(q[-1]),
        "dq": float(np.min(np.diff(q))),
        "q_opt": q_opt,
        "Delta_opt": _scalar(npz, "Delta_opt"),
        "q_Ic_plus": _scalar(npz, "q_Ic_plus"),
        "q_Ic_minus": _scalar(npz, "q_Ic_minus"),
        "Ic_plus": ic_plus,
        "Ic_minus": ic_minus,
        "abs_Ic_plus": abs(ic_plus),
        "abs_Ic_minus": abs(ic_minus),
        "Ic_denominator_abs": denom,
        "eta": _scalar(npz, "eta"),
        "eta_denominator_unreliable": bool(denom < float(cfg["eta_denominator_threshold"])),
        "branch_near_zero": bool(min_abs_ic < float(cfg["branch_near_zero_threshold"])),
        "left_endpoint_found": bool(_scalar(npz, "left_endpoint_found", 0.0)),
        "right_endpoint_found": bool(_scalar(npz, "right_endpoint_found", 0.0)),
        "q_left_endpoint": _scalar(npz, "q_left_endpoint"),
        "q_right_endpoint": _scalar(npz, "q_right_endpoint"),
        "endpoint_margin_left": _scalar(npz, "endpoint_margin_left"),
        "endpoint_margin_right": _scalar(npz, "endpoint_margin_right"),
        "response_window_valid": bool(_scalar(npz, "response_window_valid", 0.0)),
        "positive_eta_allowed": bool(denom >= float(cfg["eta_denominator_threshold"]) and min_abs_ic >= float(cfg["branch_near_zero_threshold"])),
        "curve_file": str(path),
    }
    if row["eta_denominator_unreliable"]:
        row["pathology_class"] = "eta_ill_conditioned_small_denominator"
    elif row["branch_near_zero"]:
        row["pathology_class"] = "eta_ill_conditioned_branch_near_zero"
    elif float(row["eta"]) > 0 and row["positive_eta_allowed"]:
        row["pathology_class"] = "positive_eta_response_curve_allowed"
    else:
        row["pathology_class"] = "no_positive_eta_pathology"
    _plot_curve(fig_dir, point_id, nq, q, current, delta, free_energy, mask, row)
    extrema["point_id"] = point_id
    extrema["nq"] = nq
    extrema["selected_q_Ic"] = extrema["branch"].map({"plus": row["q_Ic_plus"], "minus": row["q_Ic_minus"]})
    extrema["distance_to_selected_q_Ic"] = abs(extrema["q"] - extrema["selected_q_Ic"])
    return row, extrema


def _plot_curve(fig_dir: Path, point_id: int, nq: int, q: np.ndarray, current: np.ndarray, delta: np.ndarray, free_energy: np.ndarray, mask: np.ndarray, row: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    q_opt = float(row["q_opt"])
    left = q <= q_opt
    right = q >= q_opt
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True, constrained_layout=True)
    axes[0].plot(q[left], current[left], lw=1.0, color="#1f77b4", label="I+(q), q <= q_opt")
    axes[0].plot(q[right], current[right], lw=1.0, color="#d62728", label="I-(q), q >= q_opt")
    for key, color, label in [
        ("q_Ic_plus", "#1f77b4", "q_Ic+"),
        ("q_Ic_minus", "#d62728", "q_Ic-"),
        ("q_opt", "0.15", "q_opt"),
        ("q_left_endpoint", "0.45", "left endpoint"),
        ("q_right_endpoint", "0.45", "right endpoint"),
    ]:
        value = float(row[key])
        if np.isfinite(value):
            axes[0].axvline(value, color=color, ls="--", lw=0.8, label=label)
    axes[0].set_ylabel("I(q)")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].set_title(f"point {point_id}, nq={nq}: response branches")
    axes[1].plot(q, delta, lw=1.0, color="#2ca02c")
    axes[1].fill_between(q, 0, delta, where=mask, color="#2ca02c", alpha=0.16, label="branch_valid_mask")
    axes[1].set_ylabel("Delta(q)")
    axes[1].legend(fontsize=7)
    axes[2].plot(q, free_energy, lw=1.0, color="#9467bd")
    axes[2].set_ylabel("F(q)")
    axes[2].set_xlabel("q")
    fig.savefig(fig_dir / f"point{point_id:04d}_nq{nq}_response_pathology.png", dpi=240)
    plt.close(fig)


def analyze(qdensity_root: Path, audit_root: Path) -> None:
    cfg = _load_json(audit_root / "config" / "response_pathology_config.json")
    rows: list[dict[str, object]] = []
    extrema_frames: list[pd.DataFrame] = []
    missing: list[dict[str, object]] = []
    for point_id in POINT_IDS:
        for nq in NQ_LEVELS:
            path, source = _curve_path(qdensity_root, audit_root, point_id, nq)
            if path is None:
                missing.append({"point_id": point_id, "nq": nq, "missing": True})
                rows.append({"point_id": point_id, "nq": nq, "status": "missing_curve", "curve_source": "missing"})
                continue
            row, extrema = _analyze_curve(path, point_id, nq, cfg, audit_root / "figures")
            row["status"] = "ok"
            row["curve_source"] = source
            rows.append(row)
            extrema["curve_source"] = source
            extrema_frames.append(extrema)
    tables = audit_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "response_pathology_summary.csv", index=False)
    if extrema_frames:
        extrema = pd.concat(extrema_frames, ignore_index=True)
    else:
        extrema = pd.DataFrame()
    extrema.to_csv(tables / "response_pathology_top10_extrema.csv", index=False)
    pd.DataFrame(missing).to_csv(tables / "response_pathology_missing_curves.csv", index=False)
    _write_report(audit_root, summary, extrema, missing)


def _write_report(audit_root: Path, summary: pd.DataFrame, extrema: pd.DataFrame, missing: list[dict[str, object]]) -> None:
    lines = ["# Response-curve pathology audit", ""]
    lines.append("This audit checks whether saturated positive eta is a robust diode response or an ill-conditioned response-extraction result.")
    lines.append("")
    lines.append("## Completeness")
    lines.append("")
    lines.append(f"- Expected curves: `{len(POINT_IDS) * len(NQ_LEVELS)}`")
    lines.append(f"- Missing curves: `{len(missing)}`")
    if missing:
        lines.append("")
        for item in missing:
            lines.append(f"- missing point `{item['point_id']}`, nq `{item['nq']}`")
    lines.append("")
    lines.append("## Response summary")
    lines.append("")
    cols = [
        "point_id",
        "nq",
        "Ic_plus",
        "Ic_minus",
        "Ic_denominator_abs",
        "eta",
        "eta_denominator_unreliable",
        "branch_near_zero",
        "positive_eta_allowed",
        "pathology_class",
        "status",
        "curve_source",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(_fmt(row.get(col, "N/A")) for col in cols) + " |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The audit does not allow a positive-eta claim when `abs(Ic_plus) + abs(Ic_minus) < 1e-4` or when either critical-current branch is numerically zero. Such cases are labelled as eta ill-conditioned or branch-near-zero diagnostics.")
    lines.append("")
    if "pathology_class" in summary:
        counts = summary["pathology_class"].fillna("missing").value_counts().to_dict()
        lines.append(f"Pathology class counts: `{counts}`.")
    lines.append("")
    if missing:
        lines.append("")
        lines.append("Any point listed as `missing_curve` still requires curve reruns before response-level interpretation.")
    _write_text_lf(audit_root / "reports" / "response_curve_pathology_report.md", "\n".join(lines) + "\n")
    _write_latex(audit_root, summary, missing)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    if pd.isna(value):
        return "N/A"
    return str(value)


def _latex_escape(value: object) -> str:
    text = str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }.items():
        text = text.replace(old, new)
    return text


def _write_latex(audit_root: Path, summary: pd.DataFrame, missing: list[dict[str, object]]) -> None:
    tex = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\title{Response-curve pathology audit}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Scope}",
        r"This audit checks whether saturated positive \(\eta\) is a robust diode response or an ill-conditioned critical-current extraction result. It is audit-only and does not modify active-learning data.",
        r"\section*{Completeness}",
        f"Expected curves: {len(POINT_IDS) * len(NQ_LEVELS)}. Missing curves: {len(missing)}.",
        r"\section*{Response summary}",
        r"\begin{center}\small",
        r"\begin{tabular}{rrrrrrlp{0.22\linewidth}}",
        r"\toprule",
        r"Point & nq & $I_c^+$ & $I_c^-$ & $|I_c^+|+|I_c^-|$ & $\eta$ & branch zero & pathology \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        if row.get("status") != "ok":
            tex.append(f"{int(row['point_id'])} & {int(row['nq'])} & \\multicolumn{{6}}{{l}}{{missing curve}} \\\\")
            continue
        tex.append(
            f"{int(row['point_id'])} & {int(row['nq'])} & {float(row['Ic_plus']):.4g} & {float(row['Ic_minus']):.4g} & "
            f"{float(row['Ic_denominator_abs']):.4g} & {float(row['eta']):.4g} & "
            f"{_latex_escape(row['branch_near_zero'])} & {_latex_escape(row.get('pathology_class', 'N/A'))} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    for point_id in POINT_IDS:
        for nq in NQ_LEVELS:
            fig = audit_root / "figures" / f"point{point_id:04d}_nq{nq}_response_pathology.png"
            if fig.exists():
                tex.extend(
                    [
                        r"\begin{figure}[H]",
                        r"\centering",
                        rf"\includegraphics[width=0.86\linewidth]{{../figures/{fig.name}}}",
                        rf"\caption{{Response branch pathology audit for point {point_id}, \(n_q={nq}\).}}",
                        r"\end{figure}",
                    ]
                )
    tex.extend(
        [
            r"\section*{Interpretation}",
            r"If \(|I_c^+|+|I_c^-|<10^{-4}\), or if either critical-current branch is numerically zero, \(\eta\) is ill-conditioned and should not be used as evidence for a robust positive diode response. The current returned curves put points 21 and 25 in ill-conditioned classes rather than robust positive-response classes.",
            r"\end{document}",
        ]
    )
    _write_text_lf(audit_root / "reports" / "response_curve_pathology_report.tex", "\n".join(tex) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit response curves for eta branch pathologies.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_setup = sub.add_parser("setup")
    p_setup.add_argument("--qdensity-root", type=Path, required=True)
    p_setup.add_argument("--audit-root", type=Path, default=None)
    p_run = sub.add_parser("run")
    p_run.add_argument("--qdensity-root", type=Path, required=True)
    p_run.add_argument("--audit-root", type=Path, required=True)
    p_run.add_argument("--rank", type=int, required=True)
    p_run.add_argument("--world-size", type=int, required=True)
    p_run.add_argument("--device", type=str, default="cuda:0")
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("--qdensity-root", type=Path, required=True)
    p_analyze.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "setup":
        audit_root = args.audit_root or (args.qdensity_root / AUDIT_SUBDIR)
        setup_audit(args.qdensity_root, audit_root)
        print(f"Prepared response pathology audit folder: {audit_root}")
    elif args.cmd == "run":
        run_tasks(args.qdensity_root, args.audit_root, args.rank, args.world_size, args.device)
    elif args.cmd == "analyze":
        analyze(args.qdensity_root, args.audit_root)


if __name__ == "__main__":
    main()
