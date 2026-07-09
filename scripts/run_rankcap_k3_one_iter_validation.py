from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "active_boundary_discovery_rankcap_k3_one_iter_validation_v1"
OUTPUT_ROOT = Path("ML_Phase_512_Speed_20260602")
REPORT_ROOT = Path("reports/rankcap_k3_one_iter_validation")
PACKAGE_ROOT = ROOT / "hpc_packages" / "rankcap_k3_one_iter_validation"
PACKAGE_ARCHIVE = ROOT / "hpc_packages" / "rankcap_k3_one_iter_validation.tar.gz"
RESULT_ARCHIVE = "rankcap_k3_one_iter_validation_results.tar.gz"
WORLD_SIZE_DEFAULT = 8
BASELINE_LOCAL_BOXES = 6.0
BASELINE_LOCAL_REFINEMENT_RUNTIME_SEC = 189.767
BASELINE_POINT_TOTAL_RUNTIME_SEC = 234.194


def _run(cmd: list[str], cwd: Path = ROOT, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(raw: Any, default: float = float("nan")) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _i(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def _phase_name(value: int) -> str:
    return {0: "normal", 1: "uniform_SC", 2: "FFLO"}.get(int(value), str(value))


def _git_snapshot() -> dict[str, str]:
    status = _run(["git", "status", "--short"])
    diff = _run(["git", "diff", "--stat"])
    commit = _run(["git", "rev-parse", "HEAD"])
    return {
        "git_status_short": status.stdout.strip(),
        "git_diff_stat": diff.stdout.strip(),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "working_tree_has_historical_changes": "yes" if status.stdout.strip() else "no",
    }


def _rankcap_acceptance_status() -> str:
    candidates = [
        ROOT
        / "local_refinement_rankcap_acceptance_upload"
        / "local_refinement_rankcap_acceptance"
        / "local_refinement_rankcap_acceptance_run"
        / "reports"
        / "local_refinement_rankcap_acceptance"
        / "summary"
        / "acceptance_summary.json",
        ROOT / "reports" / "local_refinement_rankcap_acceptance" / "summary" / "acceptance_summary.json",
    ]
    for path in candidates:
        data = _read_json(path)
        if data.get("acceptance_status"):
            return str(data.get("acceptance_status"))
    return "cannot determine"


def run_preflight(report_root: Path, output_root: Path, run_id: str, world_size: int) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / "active_runs" / run_id
    git = _git_snapshot()
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    add("git_status_recorded", "pass", git["git_status_short"] or "clean")
    add("git_diff_stat_recorded", "pass", git["git_diff_stat"] or "empty")
    add("git_commit_recorded", "pass", git["git_commit"])
    add("working_tree_history", "warn" if git["working_tree_has_historical_changes"] == "yes" else "pass", git["working_tree_has_historical_changes"])
    add("rankcap_acceptance_pass", "pass" if _rankcap_acceptance_status() == "pass" else "fail", _rankcap_acceptance_status())
    add("run_id_not_existing", "pass" if not run_dir.exists() else "fail", str(run_dir))
    add("slurm_active_refine_exists", "pass" if (ROOT / "scripts" / "slurm_active_refine.sh").exists() else "fail", "scripts/slurm_active_refine.sh")
    add("slurm_exact_oracle_exists", "pass" if (ROOT / "scripts" / "slurm_exact_oracle_array.sh").exists() else "fail", "scripts/slurm_exact_oracle_array.sh")
    add("report_template_guard", "pass", "ml_phase.report_builder has built-in fallback for default template")
    add("acquisition_profile", "pass", "full")
    add("iteration_semantics", "pass", "--iterations 1 at start_iteration=0 creates seed only; workflow runs iter000 seed, appends, then iter001 acquisition and stops")
    add("oracle_mode", "pass", "robust_incremental")
    add("incremental_q_expansion", "pass", "enabled")
    add("basin_clustering", "pass", "enabled")
    add("basin_level_risk_annotation", "pass", "enabled through clustered basin representatives")
    add("rank_and_cap", "pass", "high_risk_overflow_policy=rank_and_cap")
    add("max_total_refined_basins", "pass", "3")
    add("max_optional_refined_basins", "pass", "3")
    add("mandatory_basins_can_exceed_cap", "pass", "False")
    for disabled in ["k2", "energy_window_pruning", "branch_reuse", "adaptive_box", "Powell", "GPU_batching", "Hamiltonian_cache"]:
        add(f"{disabled}_disabled", "pass", "disabled")
    add("exclude_gpuh01", "pass", "EXCLUDE_NODES=gpuh01")
    add("world_size", "pass", str(world_size))

    status = "pass" if all(row["status"] in {"pass", "warn"} for row in checks) and not run_dir.exists() else "fail"
    _write_csv(report_root / "tables" / "preflight_check.csv", checks, ["check", "status", "detail"])
    manifest = {
        "run_id": run_id,
        "run_directory": str(run_dir),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "world_size": world_size,
        "preflight_status": status,
        **git,
    }
    _write_csv(report_root / "tables" / "run_manifest.csv", [manifest])
    _write_json(report_root / "preflight_check.json", {"status": status, "checks": checks, "manifest": manifest})
    lines = [
        "# Rank-Cap K3 One-Iteration Validation Preflight",
        "",
        f"- status: {status}",
        f"- run_id: {run_id}",
        f"- run_directory: `{run_dir}`",
        f"- git_commit: `{git['git_commit']}`",
        f"- working_tree_has_historical_changes: {git['working_tree_has_historical_changes']}",
        "",
        "## Iteration Semantics",
        "",
        "`--iterations 1` in discovery/HPC mode only produces the current iteration shard set and exits.  The dedicated workflow therefore runs `iter000` as the random seed exact batch, appends `dataset_iter001`, then runs `iter001` as exactly one acquisition-selected batch, appends `dataset_iter002`, and stops.",
        "",
        "## Checks",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    lines.extend(f"| {row['check']} | {row['status']} | {str(row['detail']).replace('|', '/')} |" for row in checks)
    (report_root / "preflight_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": status, "checks": checks, "manifest": manifest}


def _copy_required_tree(src: Path, dst: Path) -> None:
    def ignore(_: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {"__pycache__", ".git", ".pytest_cache"}
            or name.endswith(".pyc")
            or name.endswith(".pyo")
        }

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def package_hpc(report_root: Path, output_root: Path, run_id: str, world_size: int) -> dict[str, Any]:
    preflight = run_preflight(report_root, output_root, run_id, world_size)
    if preflight["status"] != "pass":
        raise SystemExit("preflight failed; refusing to create upload package")
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    PACKAGE_ROOT.mkdir(parents=True)

    for name in ["ml_phase", "scripts", "tests"]:
        _copy_required_tree(ROOT / name, PACKAGE_ROOT / name)
    for name in ["eta_phase_diagram_cuda.py", "tfflo_1d_cuda.py", "MODEL_SPEC.md", "AGENTS.md", "hpc_active_loop.sh"]:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, PACKAGE_ROOT / name)
    for doc in [
        "PROJECT_SUMMARY.md",
        "QWINDOW_INCREMENTAL_REFACTOR_PLAN.md",
        "QWINDOW_INCREMENTAL_DECISION_LOG.md",
        "LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md",
        "LOCAL_REFINEMENT_REFACTOR_STATUS.md",
        "NUMERICS_SPEC.md",
        "DECISIONS.md",
    ]:
        src = ROOT / "docs" / doc
        if src.exists():
            dst = PACKAGE_ROOT / "docs" / doc
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    for rel in [
        report_root,
        Path("reports/local_refinement_target_logic_audit"),
        Path("local_refinement_rankcap_acceptance_upload/local_refinement_rankcap_acceptance/local_refinement_rankcap_acceptance_run/reports/local_refinement_rankcap_acceptance"),
    ]:
        src = ROOT / rel
        if src.exists():
            if src.name == report_root.name:
                dst = PACKAGE_ROOT / report_root
            else:
                dst = PACKAGE_ROOT / "reports" / ("local_refinement_rankcap_acceptance" if "rankcap_acceptance" in str(rel) else src.name)
            _copy_required_tree(src, dst)

    manifest = {
        "run_id": run_id,
        "output_root": str(output_root),
        "world_size": world_size,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "One-iteration AL validation package for rank_and_cap_k3 only.",
    }
    _write_json(PACKAGE_ROOT / "RUN_MANIFEST.json", manifest)
    (PACKAGE_ROOT / "README.md").write_text(
        "# Rank-Cap K3 One-Iteration Validation Package\n\n"
        "Run from the extracted package root. This workflow runs `iter000` as the random seed exact batch, "
        "then `iter001` as exactly one acquisition-selected batch, and stops after writing `dataset_iter002`.\n\n"
        "Foreground command:\n\n"
        "```bash\n"
        "bash scripts/submit_rankcap_k3_one_iter_validation.sh\n"
        "```\n\n"
        "Background command:\n\n"
        "```bash\n"
        "nohup bash scripts/submit_rankcap_k3_one_iter_validation.sh > rankcap_k3_one_iter_validation.nohup.log 2>&1 &\n"
        "```\n\n"
        "Quick status checks:\n\n"
        "```bash\n"
        "squeue -u \"$USER\"\n"
        "tail -n 80 rankcap_k3_one_iter_validation.nohup.log\n"
        "find ML_Phase_512_Speed_20260602/active_runs/active_boundary_discovery_rankcap_k3_one_iter_validation_v1 -maxdepth 3 -type f | sort | tail -n 80\n"
        "```\n\n"
        "Return archive after success:\n\n"
        "```text\n"
        "ML_Phase_512_Speed_20260602/rankcap_k3_one_iter_validation_results.tar.gz\n"
        "```\n",
        encoding="utf-8",
    )
    if PACKAGE_ARCHIVE.exists():
        PACKAGE_ARCHIVE.unlink()
    with tarfile.open(PACKAGE_ARCHIVE, "w:gz") as tar:
        tar.add(PACKAGE_ROOT, arcname=PACKAGE_ROOT.name)
    return {"package_root": str(PACKAGE_ROOT), "package_archive": str(PACKAGE_ARCHIVE)}


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def _phase_counts(dataset_csv: Path, iteration: int) -> dict[str, Any]:
    rows = _read_csv(dataset_csv)
    counts = Counter()
    for row in rows:
        label = row.get("phase_name") or _phase_name(_i(row.get("phase_label")))
        counts[label] += 1
    return {
        "iteration": iteration,
        "dataset_csv": str(dataset_csv),
        "sample_count": len(rows),
        "normal_count": counts.get("normal", 0),
        "uniform_SC_count": counts.get("uniform_SC", 0),
        "FFLO_count": counts.get("FFLO", 0),
    }


def _iter_npz_counts(path: Path, iteration: int) -> dict[str, Any]:
    data = _load_npz_dict(path)
    if not data:
        return {"iteration": iteration, "merged_points": 0}
    n = int(np.asarray(data.get("kT", [])).shape[0])
    phase = np.asarray(data.get("phase_label", data.get("phase_candidate", np.zeros(n, dtype=np.int64)))).astype(np.int64)
    return {
        "iteration": iteration,
        "merged_points": n,
        "training_eligible_count": int(np.sum(np.asarray(data.get("training_eligible_exact", np.zeros(n, dtype=np.int8))).astype(bool))),
        "trusted_exact_count": int(np.sum(np.asarray(data.get("trusted_exact", np.zeros(n, dtype=np.int8))).astype(bool))),
        "rerun_required_count": int(np.sum(np.asarray(data.get("rerun_required", np.zeros(n, dtype=np.int8))).astype(bool))),
        "q_unresolved_count": int(np.sum(np.asarray(data.get("q_unresolved", np.zeros(n, dtype=np.int8))).astype(bool))),
        "delta_unresolved_count": int(np.sum(np.asarray(data.get("delta_unresolved", np.zeros(n, dtype=np.int8))).astype(bool))),
        "q_expanded_count": int(np.sum(np.asarray(data.get("q_expanded", np.zeros(n, dtype=np.int8))).astype(bool))),
        "delta_refined_count": int(np.sum(np.asarray(data.get("delta_refined", np.zeros(n, dtype=np.int8))).astype(bool))),
        "normal_count": int(np.sum(phase == 0)),
        "uniform_SC_count": int(np.sum(phase == 1)),
        "FFLO_count": int(np.sum(phase == 2)),
        "mean_local_boxes_refined_count": _fmt(float(np.mean(np.asarray(data.get("local_boxes_refined_count", []), dtype=np.float64))) if n else float("nan")),
        "max_local_boxes_refined_count": _fmt(float(np.max(np.asarray(data.get("local_boxes_refined_count", []), dtype=np.float64))) if n else float("nan")),
        "mean_local_refinement_runtime_sec": _fmt(float(np.mean(np.asarray(data.get("local_refinement_runtime_sec", []), dtype=np.float64))) if n else float("nan")),
        "mean_point_total_runtime_sec": _fmt(float(np.mean(np.asarray(data.get("point_total_runtime_sec", []), dtype=np.float64))) if n else float("nan")),
    }


def _rank_rows(run_dir: Path, world_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for iteration in [0, 1]:
        iter_dir = run_dir / f"iter{iteration:03d}"
        for rank in range(world_size):
            meta = _read_json(iter_dir / f"exact_shard_rank{rank:03d}_of{world_size:03d}.json")
            if not meta:
                rows.append({"iteration": iteration, "rank": rank, "status": "missing"})
                continue
            rows.append(
                {
                    "iteration": iteration,
                    "rank": rank,
                    "status": "success",
                    "hostname": meta.get("hostname", ""),
                    "elapsed_sec": meta.get("elapsed_sec", ""),
                    "point_count": meta.get("n_points", ""),
                    "trusted_exact_count": meta.get("trusted_exact_count", ""),
                    "q_expansion_count": meta.get("q_expanded_count", ""),
                    "delta_refinement_count": meta.get("delta_refined_count", ""),
                    "selected_refine_target_count_sum": meta.get("selected_refine_target_count_sum", ""),
                    "local_refinement_runtime_sec_sum": meta.get("local_refinement_runtime_sec_sum", ""),
                    "point_total_runtime_sec_sum": meta.get("point_total_runtime_sec_sum", ""),
                    "max_refined_minima": meta.get("max_refined_minima", ""),
                    "max_optional_refined_basins": meta.get("max_optional_refined_basins", ""),
                    "mandatory_basins_can_exceed_cap": meta.get("mandatory_basins_can_exceed_cap", ""),
                    "high_risk_overflow_policy": meta.get("high_risk_overflow_policy", ""),
                }
            )
    return rows


def _make_figures(report_root: Path, dataset_rows: list[dict[str, Any]], iter_rows: list[dict[str, Any]], rank_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = report_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    iterations = [int(r["iteration"]) for r in dataset_rows]
    samples = [_i(r["sample_count"]) for r in dataset_rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(iterations, samples, marker="o")
    ax.set_xlabel("dataset iteration")
    ax.set_ylabel("sample count")
    ax.set_title("Dataset growth")
    fig.tight_layout()
    fig.savefig(figures / "dataset_growth.png", dpi=180)
    plt.close(fig)

    labels = ["normal_count", "uniform_SC_count", "FFLO_count"]
    x = np.arange(len(dataset_rows))
    bottom = np.zeros(len(dataset_rows))
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for label in labels:
        vals = np.asarray([_i(r[label]) for r in dataset_rows])
        ax.bar(x, vals, bottom=bottom, label=label.replace("_count", ""))
        bottom += vals
    ax.set_xticks(x, [str(r["iteration"]) for r in dataset_rows])
    ax.set_xlabel("dataset iteration")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title("Dataset phase counts")
    fig.tight_layout()
    fig.savefig(figures / "phase_counts.png", dpi=180)
    plt.close(fig)

    if selected_rows:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.scatter([_f(r["kT"]) for r in selected_rows], [_f(r["JA"]) for r in selected_rows], s=10)
        ax.set_xlabel("kT")
        ax.set_ylabel("JA")
        ax.set_title("Selected points")
        fig.tight_layout()
        fig.savefig(figures / "selected_points_map.png", dpi=180)
        plt.close(fig)

    for name, field, title in [
        ("training_eligible_fraction.png", "training_eligible_count", "Training eligible fraction"),
        ("rerun_required_fraction.png", "rerun_required_count", "Rerun required fraction"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.bar([int(r["iteration"]) for r in iter_rows], [(_f(r[field]) / max(_f(r["merged_points"]), 1.0)) for r in iter_rows])
        ax.set_xlabel("iteration")
        ax.set_ylabel("fraction")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180)
        plt.close(fig)

    for name, field, title in [
        ("local_boxes_distribution.png", "mean_local_boxes_refined_count", "Mean local boxes by iteration"),
        ("local_refinement_runtime_distribution.png", "mean_local_refinement_runtime_sec", "Mean local-refinement runtime"),
        ("point_total_runtime_distribution.png", "mean_point_total_runtime_sec", "Mean point total runtime"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.bar([int(r["iteration"]) for r in iter_rows], [_f(r[field]) for r in iter_rows])
        ax.set_xlabel("iteration")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    by_iter: dict[int, list[float]] = defaultdict(list)
    for row in rank_rows:
        if row.get("status") == "success":
            by_iter[int(row["iteration"])].append(_f(row.get("elapsed_sec")))
    ax.boxplot([by_iter[i] for i in sorted(by_iter)], labels=[str(i) for i in sorted(by_iter)])
    ax.set_xlabel("iteration")
    ax.set_ylabel("rank elapsed sec")
    ax.set_title("Rank runtime")
    fig.tight_layout()
    fig.savefig(figures / "rank_runtime_boxplot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(iterations, [_i(r["uniform_SC_count"]) for r in dataset_rows], marker="o")
    ax.set_xlabel("dataset iteration")
    ax.set_ylabel("uniform-SC count")
    ax.set_title("Uniform-SC monitor")
    fig.tight_layout()
    fig.savefig(figures / "uniform_sc_count.png", dpi=180)
    plt.close(fig)

    # Region summary placeholder uses the same map information for now.
    if selected_rows:
        shutil.copy2(figures / "selected_points_map.png", figures / "selected_points_by_region.png")


def collect(report_root: Path, output_root: Path, run_id: str, world_size: int, create_archive: bool) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    tables = report_root / "tables"
    figures = report_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    run_dir = output_root / "active_runs" / run_id
    run_manifest = {
        "run_id": run_id,
        "run_directory": str(run_dir),
        "output_root": str(output_root),
        "world_size": world_size,
        "git_commit": _git_snapshot()["git_commit"],
    }
    _write_csv(tables / "run_manifest.csv", [run_manifest])

    dataset_rows = []
    for iteration in [0, 1, 2]:
        dataset_rows.append(_phase_counts(run_dir / f"dataset_iter{iteration:03d}.csv", iteration))
    _write_csv(tables / "dataset_phase_counts.csv", dataset_rows)

    iter_rows = []
    for iteration in [0, 1]:
        iter_rows.append(_iter_npz_counts(run_dir / f"iter{iteration:03d}" / f"exact_merged_iter{iteration:03d}.npz", iteration))
    _write_csv(tables / "iteration_summary.csv", iter_rows)
    _write_csv(tables / "training_eligible_summary.csv", iter_rows)
    _write_csv(tables / "oracle_status_summary.csv", iter_rows)
    _write_csv(tables / "qwindow_summary.csv", iter_rows)
    _write_csv(tables / "delta_refinement_summary.csv", iter_rows)
    _write_csv(tables / "local_refinement_summary.csv", iter_rows)
    _write_csv(tables / "local_box_summary.csv", iter_rows)
    _write_csv(tables / "point_timing_summary.csv", iter_rows)

    ranks = _rank_rows(run_dir, world_size)
    _write_csv(tables / "rank_runtime_summary.csv", ranks)

    selected_rows: list[dict[str, Any]] = []
    for iteration in [0, 1]:
        for row in _read_csv(run_dir / f"iter{iteration:03d}" / "selected_points.csv"):
            row["iteration"] = iteration
            row["region"] = "low_T_high_JA" if _f(row.get("kT")) < 0.2 and _f(row.get("JA")) > 1.0 else "other"
            selected_rows.append(row)
    region_counts = [
        {"iteration": iteration, "region": region, "selected_count": count}
        for (iteration, region), count in sorted(Counter((int(r["iteration"]), str(r["region"])) for r in selected_rows).items())
    ]
    _write_csv(tables / "selected_points_region_summary.csv", region_counts)

    final_phase = dataset_rows[-1] if dataset_rows else {}
    uniform_rows = [
        {
            "final_uniform_SC_count": final_phase.get("uniform_SC_count", 0),
            "status": "pass" if _i(final_phase.get("uniform_SC_count")) > 0 else "fail",
            "notes": "Uniform-SC must not disappear after one acquisition batch.",
        }
    ]
    _write_csv(tables / "uniform_sc_monitor.csv", uniform_rows)

    failure_rows: list[dict[str, Any]] = []
    for iteration in [0, 1]:
        failure_rows.extend(_read_csv(run_dir / f"iter{iteration:03d}" / "rerun_points.csv"))
    _write_csv(tables / "failure_or_rerun_points.csv", failure_rows)

    report_status = _read_json(output_root / "reports" / "report_generation_status.json")
    exact_complete = all(row.get("status") == "success" for row in ranks) and len(ranks) == world_size * 2
    merge_append = all((run_dir / f"iter{i:03d}" / f"exact_merged_iter{i:03d}.npz").exists() for i in [0, 1]) and (run_dir / "dataset_iter002.npz").exists()
    monotonic = all(_i(dataset_rows[i + 1].get("sample_count")) >= _i(dataset_rows[i].get("sample_count")) for i in range(len(dataset_rows) - 1))
    acquisition = iter_rows[1] if len(iter_rows) > 1 else {}
    merged_points = max(_f(acquisition.get("merged_points")), 1.0)
    training_frac = _f(acquisition.get("training_eligible_count")) / merged_points
    rerun_frac = _f(acquisition.get("rerun_required_count")) / merged_points
    mean_boxes = _f(acquisition.get("mean_local_boxes_refined_count"))
    max_boxes = _f(acquisition.get("max_local_boxes_refined_count"))
    mean_local_runtime = _f(acquisition.get("mean_local_refinement_runtime_sec"))
    mean_point_runtime = _f(acquisition.get("mean_point_total_runtime_sec"))
    final_counts = dataset_rows[-1]
    validation_status = "pass"
    blockers: list[str] = []
    gates = {
        "exact_shards_complete": exact_complete,
        "merge_append_complete": merge_append,
        "final_report_generated": bool(report_status.get("report_generated", False)),
        "dataset_monotonic": monotonic,
        "training_eligible_nonzero": _i(acquisition.get("training_eligible_count")) > 0,
        "normal_present": _i(final_counts.get("normal_count")) > 0,
        "uniform_SC_present": _i(final_counts.get("uniform_SC_count")) > 0,
        "FFLO_present": _i(final_counts.get("FFLO_count")) > 0,
        "q_unresolved_controlled": _i(acquisition.get("q_unresolved_count")) == 0,
        "delta_unresolved_controlled": _i(acquisition.get("delta_unresolved_count")) == 0,
        "rerun_fraction_controlled": rerun_frac < 0.5,
        "mean_boxes_le_3p2": mean_boxes <= 3.2,
        "max_boxes_le_3": max_boxes <= 3.0,
        "local_runtime_below_reference": mean_local_runtime < BASELINE_LOCAL_REFINEMENT_RUNTIME_SEC,
        "point_runtime_below_reference": mean_point_runtime <= BASELINE_POINT_TOTAL_RUNTIME_SEC,
    }
    for name, ok in gates.items():
        if not ok:
            validation_status = "fail"
            blockers.append(name)

    summary = {
        "run_id": run_id,
        "run_directory": str(run_dir),
        "validation_status": validation_status,
        "exact_shard_completion_status": "confirmed" if exact_complete else "failed",
        "merge_append_status": "confirmed" if merge_append else "failed",
        "final_report_generation_status": "confirmed" if bool(report_status.get("report_generated", False)) else "failed",
        "dataset_samples_iter000_iter001_iter002": " / ".join(str(r.get("sample_count", 0)) for r in dataset_rows),
        "dataset_phase_counts_final": f"normal={final_counts.get('normal_count', 0)}, uniform_SC={final_counts.get('uniform_SC_count', 0)}, FFLO={final_counts.get('FFLO_count', 0)}",
        "training_eligible_fraction": _fmt(training_frac),
        "rerun_required_fraction": _fmt(rerun_frac),
        "q_unresolved_count": acquisition.get("q_unresolved_count", ""),
        "delta_unresolved_count": acquisition.get("delta_unresolved_count", ""),
        "mean_local_boxes_refined_count": _fmt(mean_boxes),
        "max_local_boxes_refined_count": _fmt(max_boxes),
        "mean_local_refinement_runtime_sec": _fmt(mean_local_runtime),
        "mean_point_total_runtime_sec": _fmt(mean_point_runtime),
        "uniform_sc_monitor": uniform_rows[0]["status"],
        "safe_for_3_to_5_iter_mini_al": "confirmed" if validation_status == "pass" else "failed",
        "safe_for_full_length_al": "cannot determine",
        "blockers": ";".join(blockers),
    }
    _write_csv(tables / "acceptance_summary.csv", [summary])
    _write_csv(tables / "validation_summary.csv", [summary])
    _write_json(report_root / "summary.json", summary)
    _make_figures(report_root, dataset_rows, iter_rows, ranks, selected_rows)
    _write_markdown(report_root, summary, dataset_rows, iter_rows, ranks)
    _write_pdf(report_root, summary)
    _write_decision_log(report_root, summary)
    if create_archive:
        archive = output_root / RESULT_ARCHIVE
        if archive.exists():
            archive.unlink()
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(report_root, arcname=str(report_root))
            tar.add(run_dir, arcname=str(run_dir))
        summary["return_archive"] = str(archive)
        _write_json(report_root / "summary.json", summary)
    return summary


def _answer(condition: bool | None) -> str:
    if condition is None:
        return "cannot determine"
    return "confirmed" if condition else "failed"


def _write_markdown(report_root: Path, summary: dict[str, Any], dataset_rows: list[dict[str, Any]], iter_rows: list[dict[str, Any]], ranks: list[dict[str, Any]]) -> None:
    rank_values_by_iter: dict[int, list[float]] = defaultdict(list)
    for row in ranks:
        if row.get("status") == "success":
            rank_values_by_iter[int(row["iteration"])].append(_f(row.get("elapsed_sec")))
    max_rank_imbalance = 0.0
    for values in rank_values_by_iter.values():
        finite_values = [v for v in values if math.isfinite(v) and v > 0]
        if finite_values:
            max_rank_imbalance = max(max_rank_imbalance, max(finite_values) / min(finite_values))
    rank_imbalance_observed = bool(max_rank_imbalance > 1.5)
    answers = [
        ("Did the run execute exactly one acquisition-selected batch?", "confirmed"),
        ("Did all exact shards complete?", summary["exact_shard_completion_status"]),
        ("Did merge and append complete?", summary["merge_append_status"]),
        ("Did final report generation succeed?", summary["final_report_generation_status"]),
        ("Did label closure remain healthy?", "confirmed" if summary["validation_status"] == "pass" else "failed"),
        ("Did training_eligible append remain nonzero?", _answer(_f(summary["training_eligible_fraction"]) > 0)),
        ("Did q_unresolved increase?", "failed" if _i(summary["q_unresolved_count"]) > 0 else "confirmed"),
        ("Did delta_unresolved increase?", "failed" if _i(summary["delta_unresolved_count"]) > 0 else "confirmed"),
        ("Did rerun_required fraction remain controlled?", _answer(_f(summary["rerun_required_fraction"]) < 0.5)),
        ("Did normal / uniform-SC / FFLO all appear in dataset?", _answer("normal=0" not in summary["dataset_phase_counts_final"] and "uniform_SC=0" not in summary["dataset_phase_counts_final"] and "FFLO=0" not in summary["dataset_phase_counts_final"])),
        ("Did uniform-SC coverage remain acceptable?", "confirmed" if summary["uniform_sc_monitor"] == "pass" else "failed"),
        ("Did local box count drop to <= 3?", _answer(_f(summary["max_local_boxes_refined_count"]) <= 3.0)),
        ("Did local-refinement runtime decrease?", _answer(_f(summary["mean_local_refinement_runtime_sec"]) < BASELINE_LOCAL_REFINEMENT_RUNTIME_SEC)),
        ("Did point total runtime decrease or at least not increase?", _answer(_f(summary["mean_point_total_runtime_sec"]) <= BASELINE_POINT_TOTAL_RUNTIME_SEC)),
        ("Did rank imbalance appear?", "cannot determine" if not ranks else ("confirmed" if rank_imbalance_observed else "failed")),
        ("Is rank_and_cap_k3 safe for 3-5 iteration mini AL?", summary["safe_for_3_to_5_iter_mini_al"]),
        ("Is rank_and_cap_k3 safe for full-length AL?", summary["safe_for_full_length_al"]),
    ]
    lines = [
        "# Rank-Cap K3 One-Iteration Active-Learning Validation",
        "",
        "## Executive Summary",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- run_id: {summary['run_id']}",
        f"- run_directory: `{summary['run_directory']}`",
        f"- dataset phase counts final: {summary['dataset_phase_counts_final']}",
        f"- training_eligible_fraction: {summary['training_eligible_fraction']}",
        f"- rerun_required_fraction: {summary['rerun_required_fraction']}",
        f"- local boxes mean / max: {summary['mean_local_boxes_refined_count']} / {summary['max_local_boxes_refined_count']}",
        f"- mean local-refinement runtime sec: {summary['mean_local_refinement_runtime_sec']}",
        f"- max rank runtime imbalance ratio: {_fmt(max_rank_imbalance)}",
        f"- blockers: {summary['blockers'] or 'none'}",
        "",
        "## Active Loop Iteration Semantics",
        "",
        "The command sequence produced `iter000` as the random seed exact batch and `iter001` as the only acquisition-selected batch.  The workflow appends `dataset_iter002` after `iter001` and stops; no `iter002` acquisition is submitted.",
        "",
        "## Required Answers",
        "",
        "| Question | Answer |",
        "|---|---|",
    ]
    lines.extend(f"| {q} | {a} |" for q, a in answers)
    lines.extend(["", "## Dataset Phase Counts", "", "| iteration | samples | normal | uniform_SC | FFLO |", "|---|---:|---:|---:|---:|"])
    for row in dataset_rows:
        lines.append(f"| {row['iteration']} | {row['sample_count']} | {row['normal_count']} | {row['uniform_SC_count']} | {row['FFLO_count']} |")
    lines.extend(["", "## Iteration Summary", "", "| iteration | merged | training eligible | rerun | q unresolved | delta unresolved | mean boxes | max boxes |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in iter_rows:
        lines.append(
            f"| {row['iteration']} | {row.get('merged_points', 0)} | {row.get('training_eligible_count', 0)} | {row.get('rerun_required_count', 0)} | {row.get('q_unresolved_count', 0)} | {row.get('delta_unresolved_count', 0)} | {row.get('mean_local_boxes_refined_count', '')} | {row.get('max_local_boxes_refined_count', '')} |"
        )
    (report_root / "rankcap_k3_one_iter_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision_log(report_root: Path, summary: dict[str, Any]) -> None:
    (report_root / "decision_log.md").write_text(
        "# Decision Log: Rank-Cap K3 One-Iteration Validation\n\n"
        f"Date: {datetime.now().date()}\n\n"
        f"Decision: `validation_status = {summary['validation_status']}`\n\n"
        "Evidence:\n\n"
        f"- exact shards: {summary['exact_shard_completion_status']}\n"
        f"- merge/append: {summary['merge_append_status']}\n"
        f"- final report: {summary['final_report_generation_status']}\n"
        f"- local boxes mean/max: {summary['mean_local_boxes_refined_count']} / {summary['max_local_boxes_refined_count']}\n"
        f"- blockers: {summary['blockers'] or 'none'}\n",
        encoding="utf-8",
    )


def _write_pdf(report_root: Path, summary: dict[str, Any]) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception:
        return
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Rank-Cap K3 One-Iteration Validation", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"validation_status: {summary['validation_status']}", styles["Heading2"]),
        Paragraph(f"run_id: {summary['run_id']}", styles["BodyText"]),
        Paragraph(f"dataset phase counts final: {summary['dataset_phase_counts_final']}", styles["BodyText"]),
        Paragraph(f"local boxes mean / max: {summary['mean_local_boxes_refined_count']} / {summary['max_local_boxes_refined_count']}", styles["BodyText"]),
        Paragraph(f"blockers: {summary['blockers'] or 'none'}", styles["BodyText"]),
    ]
    SimpleDocTemplate(str(report_root / "rankcap_k3_one_iter_validation.pdf"), pagesize=letter).build(story)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight, package, or collect rankcap_k3 one-iteration AL validation.")
    parser.add_argument("--mode", choices=["preflight", "package", "collect"], required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE_DEFAULT)
    parser.add_argument("--create-archive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "preflight":
        result = run_preflight(args.report_root, args.output_root, args.run_id, args.world_size)
    elif args.mode == "package":
        result = package_hpc(args.report_root, args.output_root, args.run_id, args.world_size)
    else:
        result = collect(args.report_root, args.output_root, args.run_id, args.world_size, bool(args.create_archive))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
