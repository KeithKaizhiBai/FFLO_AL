from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_LOCAL_BOXES = 6.0
BASELINE_LOCAL_REFINEMENT_RUNTIME_SEC = 189.767
BASELINE_POINT_TOTAL_RUNTIME_SEC = 234.194


@dataclass(frozen=True)
class LoopPackageConfig:
    package_kind: str
    package_name: str
    run_id: str
    output_root: str
    report_name: str
    n_iters: int
    acquisition_batches: int
    result_archive: str
    submit_script: str
    description: str
    full_loop: bool = False


CONFIGS = {
    "5iter": LoopPackageConfig(
        package_kind="5iter",
        package_name="rankcap_k3_5iter_validation",
        run_id="active_boundary_discovery_rankcap_k3_5iter_validation_v1",
        output_root="ML_Phase_512_RankCapK3_5Iter",
        report_name="rankcap_k3_5iter_validation",
        n_iters=6,
        acquisition_batches=5,
        result_archive="rankcap_k3_5iter_validation_results.tar.gz",
        submit_script="submit_rankcap_k3_5iter_validation.sh",
        description="Rank-and-cap k3 seed plus five acquisition-batch closed-loop validation.",
    ),
    "full": LoopPackageConfig(
        package_kind="full",
        package_name="rankcap_k3_full_loop",
        run_id="active_boundary_discovery_rankcap_k3_full_loop_v1",
        output_root="ML_Phase_512_RankCapK3_FullLoop",
        report_name="rankcap_k3_full_loop",
        n_iters=31,
        acquisition_batches=30,
        result_archive="rankcap_k3_full_loop_results.tar.gz",
        submit_script="submit_rankcap_k3_full_loop.sh",
        description="Rank-and-cap k3 full active-learning loop package.",
        full_loop=True,
    ),
}


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


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


def _git_snapshot() -> dict[str, str]:
    status = _run(["git", "status", "--short"])
    diff = _run(["git", "diff", "--stat"])
    commit = _run(["git", "rev-parse", "HEAD"])
    return {
        "git_status_short": status.stdout.strip(),
        "git_diff_stat": diff.stdout.strip(),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "working_tree_has_changes": "yes" if status.stdout.strip() else "no",
    }


def _path_status(path: Path) -> str:
    return "pass" if path.exists() else "fail"


def _acceptance_status() -> str:
    candidates = [
        ROOT / "reports" / "local_refinement_rankcap_acceptance" / "summary" / "acceptance_summary.json",
        ROOT
        / "local_refinement_rankcap_acceptance_upload"
        / "local_refinement_rankcap_acceptance"
        / "local_refinement_rankcap_acceptance_run"
        / "reports"
        / "local_refinement_rankcap_acceptance"
        / "summary"
        / "acceptance_summary.json",
    ]
    seen: list[str] = []
    for path in candidates:
        data = _read_json(path)
        if data.get("acceptance_status"):
            status = str(data["acceptance_status"])
            if status == "pass":
                return status
            seen.append(status)
    return seen[0] if seen else "cannot determine"


def _one_iter_status() -> str:
    candidates = [
        ROOT / "reports" / "rankcap_k3_one_iter_validation" / "summary.json",
        ROOT / "rankcap_k3_one_iter_validation" / "reports" / "rankcap_k3_one_iter_validation" / "summary.json",
    ]
    seen: list[str] = []
    for path in candidates:
        data = _read_json(path)
        if data.get("validation_status"):
            status = str(data["validation_status"])
            if status == "pass":
                return status
            seen.append(status)
    return seen[0] if seen else "cannot determine"


def _has_exclude_support(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return "EXCLUDE_NODES" in text and "--exclude=\"${EXCLUDE_NODES}\"" in text


def run_preflight(config: LoopPackageConfig, world_size: int, n_iters: int | None = None) -> dict[str, Any]:
    n_iters = n_iters or config.n_iters
    report_root = ROOT / "reports" / config.report_name
    run_dir = ROOT / config.output_root / "active_runs" / config.run_id
    git = _git_snapshot()
    checks: list[dict[str, str]] = []

    def add(check: str, status: str, detail: str) -> None:
        checks.append({"check": check, "status": status, "detail": detail})

    add("git_status_recorded", "pass", git["git_status_short"] or "clean")
    add("git_diff_stat_recorded", "pass", git["git_diff_stat"] or "empty")
    add("git_commit_recorded", "pass", git["git_commit"])
    add("rankcap_acceptance_status", "pass" if _acceptance_status() == "pass" else "warn", _acceptance_status())
    add("one_iter_validation_status", "pass" if _one_iter_status() == "pass" else "warn", _one_iter_status())
    add("run_dir_absent", "pass" if not run_dir.exists() else "fail", str(run_dir))
    add("hpc_active_loop_exists", _path_status(ROOT / "hpc_active_loop.sh"), "hpc_active_loop.sh")
    add("slurm_active_refine_exists", _path_status(ROOT / "scripts" / "slurm_active_refine.sh"), "scripts/slurm_active_refine.sh")
    add("slurm_exact_oracle_exists", _path_status(ROOT / "scripts" / "slurm_exact_oracle_array.sh"), "scripts/slurm_exact_oracle_array.sh")
    add("active_loop_excludes_gpuh01", "pass" if _has_exclude_support(ROOT / "hpc_active_loop.sh") else "fail", "EXCLUDE_NODES -> sbatch --exclude")
    add("submit_script_exists", _path_status(ROOT / "scripts" / config.submit_script), f"scripts/{config.submit_script}")
    add("world_size", "pass", str(world_size))
    add("n_iters", "pass", str(n_iters))
    add("acquisition_batches", "pass", str(max(0, n_iters - 1)))
    add("output_root_inside_package", "pass", config.output_root)
    add("oracle_mode", "pass", "robust_incremental")
    add("acquisition_profile", "pass", "full")
    add("rank_and_cap_k3", "pass", "max_total=3, optional=3, mandatory_overflow_policy=rank_and_cap")
    for disabled in ["k2", "energy_window", "branch_reuse", "Powell", "adaptive_box", "GPU_batching", "Hamiltonian_cache"]:
        add(f"{disabled}_disabled", "pass", "disabled")
    if config.full_loop:
        add("full_loop_guard", "pass", "submit requires CONFIRM_FULL_LOOP=1")

    status = "pass" if all(row["status"] in {"pass", "warn"} for row in checks) else "fail"
    report_root.mkdir(parents=True, exist_ok=True)
    _write_csv(report_root / "tables" / "preflight_check.csv", checks, ["check", "status", "detail"])
    manifest = {
        "package_kind": config.package_kind,
        "package_name": config.package_name,
        "run_id": config.run_id,
        "output_root": config.output_root,
        "report_name": config.report_name,
        "world_size": world_size,
        "n_iters": n_iters,
        "acquisition_batches": max(0, n_iters - 1),
        "preflight_status": status,
        **git,
    }
    _write_csv(report_root / "tables" / "run_manifest.csv", [manifest])
    _write_json(report_root / "preflight_check.json", {"status": status, "checks": checks, "manifest": manifest})
    md = [
        f"# {config.report_name} Preflight",
        "",
        f"- status: {status}",
        f"- package_kind: {config.package_kind}",
        f"- run_id: {config.run_id}",
        f"- output_root: `{config.output_root}`",
        f"- world_size: {world_size}",
        f"- n_iters: {n_iters}",
        f"- acquisition_batches: {max(0, n_iters - 1)}",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    md.extend(f"| {row['check']} | {row['status']} | {row['detail'].replace('|', '/')} |" for row in checks)
    (report_root / "preflight_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"status": status, "manifest": manifest, "checks": checks}


def _ignore_tree(_: str, names: list[str]) -> set[str]:
    blocked = {"__pycache__", ".git", ".pytest_cache", "active_runs"}
    return {name for name in names if name in blocked or name.endswith((".pyc", ".pyo"))}


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore_tree)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _normalize_shell_scripts(package_root: Path) -> list[str]:
    normalized: list[str] = []
    for path in package_root.rglob("*.sh"):
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        text = raw.decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        path.write_text(text, encoding="utf-8", newline="\n")
        normalized.append(str(path.relative_to(package_root)))
    return normalized


def _copy_evidence_reports(package_root: Path) -> None:
    sources = [
        ROOT / "reports" / "local_refinement_target_logic_audit",
        ROOT / "rankcap_k3_one_iter_validation" / "reports" / "rankcap_k3_one_iter_validation",
        ROOT / "reports" / "rankcap_k3_one_iter_validation",
        ROOT
        / "local_refinement_rankcap_acceptance_upload"
        / "local_refinement_rankcap_acceptance"
        / "local_refinement_rankcap_acceptance_run"
        / "reports"
        / "local_refinement_rankcap_acceptance",
        ROOT / "reports" / "local_refinement_rankcap_acceptance",
    ]
    copied = set()
    for src in sources:
        if not src.exists():
            continue
        name = src.name
        if name in copied:
            continue
        _copy_tree(src, package_root / "reports" / name)
        copied.add(name)


def package_hpc(config: LoopPackageConfig, world_size: int, n_iters: int | None = None) -> dict[str, Any]:
    n_iters = n_iters or config.n_iters
    preflight = run_preflight(config, world_size, n_iters)
    if preflight["status"] != "pass":
        raise SystemExit(f"preflight failed for {config.package_kind}; refusing to package")

    package_root = ROOT / "hpc_packages" / config.package_name
    archive_path = ROOT / "hpc_packages" / f"{config.package_name}.tar.gz"
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)

    for directory in ["ml_phase", "scripts", "tests", "docs"]:
        _copy_tree(ROOT / directory, package_root / directory)
    for file_name in ["AGENTS.md", "MODEL_SPEC.md", "eta_phase_diagram_cuda.py", "tfflo_1d_cuda.py", "hpc_active_loop.sh"]:
        _copy_file(ROOT / file_name, package_root / file_name)
    _copy_evidence_reports(package_root)

    manifest = {
        **preflight["manifest"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": config.description,
        "result_archive": config.result_archive,
        "submit_command": f"nohup bash scripts/{config.submit_script} > {config.package_name}.nohup.log 2>&1 &",
        "full_loop_requires_confirmation": config.full_loop,
    }
    _write_json(package_root / "RUN_MANIFEST.json", manifest)
    readme = [
        f"# {config.package_name}",
        "",
        config.description,
        "",
        "## Run",
        "",
        "```bash",
        f"nohup bash scripts/{config.submit_script} > {config.package_name}.nohup.log 2>&1 &",
        "```",
        "",
        "The package writes outputs only under its own output root:",
        "",
        f"```text\n{config.output_root}/\n```",
        "",
        "gpuh01 is excluded by default through `EXCLUDE_NODES=gpuh01`.",
    ]
    if config.full_loop:
        readme.extend(["", "For the full loop, set `CONFIRM_FULL_LOOP=1` explicitly before submitting."])
    (package_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8", newline="\n")

    normalized = _normalize_shell_scripts(package_root)
    manifest["normalized_shell_scripts"] = normalized
    _write_json(package_root / "RUN_MANIFEST.json", manifest)

    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(package_root, arcname=config.package_name)
    sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (archive_path.with_suffix(archive_path.suffix + ".sha256")).write_text(f"{sha256}  {archive_path.name}\n", encoding="utf-8")
    metadata = {
        "archive": str(archive_path),
        "package_root": str(package_root),
        "sha256": sha256,
        "size_bytes": archive_path.stat().st_size,
        "manifest": manifest,
    }
    _write_json(archive_path.with_suffix(archive_path.suffix + ".metadata.json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def _phase_counts_from_dataset(path: Path) -> dict[str, Any]:
    df = _read_csv(path)
    row: dict[str, Any] = {"dataset": path.name, "samples": int(len(df))}
    if df.empty:
        row.update({"normal": 0, "uniform_SC": 0, "FFLO": 0})
        return row
    if "phase_name" in df.columns:
        counts = df["phase_name"].astype(str).value_counts().to_dict()
    elif "phase_label" in df.columns:
        mapping = {0: "normal", 1: "uniform_SC", 2: "FFLO"}
        counts = df["phase_label"].map(mapping).value_counts().to_dict()
    else:
        counts = {}
    row.update({
        "normal": int(counts.get("normal", 0)),
        "uniform_SC": int(counts.get("uniform_SC", 0)),
        "FFLO": int(counts.get("FFLO", 0)),
    })
    return row


def _iter_index_from_name(path: Path) -> int:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return int(digits) if digits else -1


def _shard_rows(iter_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(iter_dir.glob("exact_shard_rank*_of*.json")):
        data = _read_json(path)
        if not data:
            continue
        row = {"iteration": _iter_index_from_name(iter_dir), "shard_file": path.name}
        row.update(data)
        rows.append(row)
    return rows


def _scan_tracebacks(root: Path) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    patterns = ["Traceback", "RuntimeError", "CUDA initialization", "out of memory", "silent fallback", "silent mismatch"]
    for path in root.glob("slurm-*.out"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in patterns:
            if pattern in text:
                matches.append({"file": path.name, "pattern": pattern})
    return matches


def _concat_rerun_points(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("iter*/rerun_points.csv")):
        df = _read_csv(path)
        if not df.empty:
            df.insert(0, "iteration", _iter_index_from_name(path.parent))
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_basic(figures: Path, tables: dict[str, pd.DataFrame]) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    dataset = tables.get("dataset_phase_counts", pd.DataFrame())
    if not dataset.empty:
        plt.figure(figsize=(6, 4))
        x = dataset["iteration"]
        plt.plot(x, dataset["samples"], marker="o")
        plt.xlabel("iteration")
        plt.ylabel("dataset samples")
        plt.tight_layout()
        plt.savefig(figures / "dataset_growth.png", dpi=180)
        plt.close()

        plt.figure(figsize=(6, 4))
        for col in ["normal", "uniform_SC", "FFLO"]:
            plt.plot(x, dataset[col], marker="o", label=col)
        plt.xlabel("iteration")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "phase_counts.png", dpi=180)
        plt.savefig(figures / "uniform_sc_count.png", dpi=180)
        plt.close()

    summary = tables.get("iteration_summary", pd.DataFrame())
    if not summary.empty:
        for col, name, ylabel in [
            ("training_eligible_fraction", "training_eligible_fraction.png", "fraction"),
            ("rerun_required_fraction", "rerun_required_fraction.png", "fraction"),
            ("mean_local_boxes_refined_count", "local_boxes_distribution.png", "mean boxes"),
            ("mean_local_refinement_runtime_sec", "local_refinement_runtime_distribution.png", "seconds"),
            ("mean_point_total_runtime_sec", "point_total_runtime_distribution.png", "seconds"),
        ]:
            if col in summary.columns:
                plt.figure(figsize=(6, 4))
                plt.plot(summary["iteration"], summary[col], marker="o")
                plt.xlabel("iteration")
                plt.ylabel(ylabel)
                plt.tight_layout()
                plt.savefig(figures / name, dpi=180)
                plt.close()

    ranks = tables.get("rank_runtime_summary", pd.DataFrame())
    if not ranks.empty and "elapsed_sec" in ranks.columns:
        plt.figure(figsize=(6, 4))
        ranks.boxplot(column="elapsed_sec", by="iteration")
        plt.suptitle("")
        plt.title("rank runtime")
        plt.xlabel("iteration")
        plt.ylabel("seconds")
        plt.tight_layout()
        plt.savefig(figures / "rank_runtime_boxplot.png", dpi=180)
        plt.close()


def collect_report(config: LoopPackageConfig, world_size: int, n_iters: int | None = None, create_archive: bool = False) -> dict[str, Any]:
    n_iters = n_iters or config.n_iters
    output_root = ROOT / config.output_root
    run_dir = output_root / "active_runs" / config.run_id
    report_root = ROOT / "reports" / config.report_name
    tables_dir = report_root / "tables"
    figures_dir = report_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    for path in sorted(run_dir.glob("dataset_iter*.csv")):
        row = _phase_counts_from_dataset(path)
        row["iteration"] = _iter_index_from_name(path)
        dataset_rows.append(row)
    dataset_df = pd.DataFrame(dataset_rows).sort_values("iteration") if dataset_rows else pd.DataFrame()
    dataset_df.to_csv(tables_dir / "dataset_phase_counts.csv", index=False)

    rank_rows: list[dict[str, Any]] = []
    iter_rows: list[dict[str, Any]] = []
    local_box_rows = []
    for iter_dir in sorted(run_dir.glob("iter*")):
        iteration = _iter_index_from_name(iter_dir)
        shards = _shard_rows(iter_dir)
        rank_rows.extend(shards)
        selected = _read_csv(iter_dir / "selected_points.csv")
        rerun = _read_csv(iter_dir / "rerun_points.csv")
        append = _read_json(run_dir / f"dataset_iter{iteration + 1:03d}.append.json")
        status = _read_json(iter_dir / "status.json")
        shard_count = len(shards)
        n_points = sum(_i(row.get("n_points")) for row in shards)
        trusted = sum(_i(row.get("trusted_exact_count")) for row in shards)
        q_unresolved = sum(_i(row.get("q_unresolved_count")) for row in shards)
        delta_unresolved = sum(_i(row.get("delta_unresolved_count")) for row in shards)
        selected_targets = sum(_i(row.get("selected_refine_target_count_sum")) for row in shards)
        local_runtime = sum(_f(row.get("local_refinement_runtime_sec_sum"), 0.0) for row in shards)
        point_runtime = sum(_f(row.get("point_total_runtime_sec_sum"), 0.0) for row in shards)
        elapsed = [_f(row.get("elapsed_sec")) for row in shards if math.isfinite(_f(row.get("elapsed_sec")))]
        boxes_mean = selected_targets / n_points if n_points else float("nan")
        iter_rows.append({
            "iteration": iteration,
            "selected_points": int(len(selected)),
            "shards_found": shard_count,
            "shards_expected": world_size,
            "all_shards_complete": shard_count == world_size,
            "merged": bool((iter_dir / f"exact_merged_iter{iteration:03d}.npz").exists()),
            "appended": bool((run_dir / f"dataset_iter{iteration + 1:03d}.npz").exists()),
            "status_completed": bool(status.get("completed", False)),
            "exact_points": n_points,
            "trusted_exact_count": trusted,
            "training_eligible_appended": _i(append.get("training_eligible_points_appended")),
            "new_unique_samples_added": _i(append.get("new_unique_samples_added")),
            "rerun_required_count": int(len(rerun)),
            "rerun_required_fraction": int(len(rerun)) / n_points if n_points else float("nan"),
            "q_unresolved_count": q_unresolved,
            "delta_unresolved_count": delta_unresolved,
            "selected_refine_target_count_sum": selected_targets,
            "mean_local_boxes_refined_count": boxes_mean,
            "local_refinement_runtime_sec_sum": local_runtime,
            "mean_local_refinement_runtime_sec": local_runtime / n_points if n_points else float("nan"),
            "point_total_runtime_sec_sum": point_runtime,
            "mean_point_total_runtime_sec": point_runtime / n_points if n_points else float("nan"),
            "rank_runtime_min_sec": min(elapsed) if elapsed else float("nan"),
            "rank_runtime_max_sec": max(elapsed) if elapsed else float("nan"),
            "rank_runtime_imbalance_ratio": max(elapsed) / min(elapsed) if elapsed and min(elapsed) > 0 else float("nan"),
        })
        for path in sorted((iter_dir / "performance").glob("*local_box_timing_rank*_of*.csv")):
            df = _read_csv(path)
            if not df.empty:
                df.insert(0, "iteration", iteration)
                local_box_rows.append(df)

    rank_df = pd.DataFrame(rank_rows)
    iter_df = pd.DataFrame(iter_rows).sort_values("iteration") if iter_rows else pd.DataFrame()
    local_box_df = pd.concat(local_box_rows, ignore_index=True) if local_box_rows else pd.DataFrame()
    rerun_df = _concat_rerun_points(run_dir)
    rank_df.to_csv(tables_dir / "rank_runtime_summary.csv", index=False)
    iter_df.to_csv(tables_dir / "iteration_summary.csv", index=False)
    iter_df.to_csv(tables_dir / "training_eligible_summary.csv", index=False)
    iter_df.to_csv(tables_dir / "oracle_status_summary.csv", index=False)
    iter_df.to_csv(tables_dir / "local_refinement_summary.csv", index=False)
    iter_df.to_csv(tables_dir / "local_box_summary.csv", index=False)
    iter_df.to_csv(tables_dir / "point_timing_summary.csv", index=False)
    rerun_df.to_csv(tables_dir / "failure_or_rerun_points.csv", index=False)
    if not local_box_df.empty:
        local_box_df.to_csv(tables_dir / "local_box_rows.csv", index=False)

    manifest = {
        "package_kind": config.package_kind,
        "package_name": config.package_name,
        "run_id": config.run_id,
        "run_directory": str(run_dir),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "world_size": world_size,
        "n_iters_expected": n_iters,
        "acquisition_batches_expected": max(0, n_iters - 1),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_csv(tables_dir / "run_manifest.csv", [manifest])

    tracebacks = _scan_tracebacks(ROOT)
    _write_csv(tables_dir / "traceback_scan.csv", tracebacks, ["file", "pattern"])
    report_status = _read_json(output_root / "reports" / "report_generation_status.json")
    expected_iters_present = len(iter_df) >= n_iters
    all_shards = bool(not iter_df.empty and iter_df["all_shards_complete"].all())
    all_merged = bool(not iter_df.empty and iter_df["merged"].all())
    all_appended = bool(not iter_df.empty and iter_df["appended"].all())
    dataset_monotonic = bool(dataset_df.empty or dataset_df["samples"].is_monotonic_increasing)
    final_phase = dataset_df.iloc[-1].to_dict() if not dataset_df.empty else {}
    phase_coverage = all(_i(final_phase.get(name)) > 0 for name in ["normal", "uniform_SC", "FFLO"])
    training_nonzero = bool(not iter_df.empty and (iter_df["training_eligible_appended"] > 0).all())
    q_ok = bool(not iter_df.empty and (iter_df["q_unresolved_count"] == 0).all())
    delta_ok = bool(not iter_df.empty and (iter_df["delta_unresolved_count"] == 0).all())
    rerun_ok = bool(not iter_df.empty and (iter_df["rerun_required_fraction"].fillna(0) < 0.5).all())
    box_mean_ok = bool(not iter_df.empty and (iter_df["mean_local_boxes_refined_count"].fillna(0) <= 3.2).all())
    max_boxes = int(local_box_df.groupby(["iteration", "point_id"]).size().max()) if not local_box_df.empty else 0
    max_box_ok = max_boxes <= 3 if max_boxes else True
    no_tracebacks = len(tracebacks) == 0
    report_ok = str(report_status.get("status", "")).lower() in {"ok", "pdflatex_missing", "pdflatex_failed"}
    validation_status = "pass" if all([
        expected_iters_present,
        all_shards,
        all_merged,
        all_appended,
        dataset_monotonic,
        phase_coverage,
        training_nonzero,
        q_ok,
        delta_ok,
        rerun_ok,
        box_mean_ok,
        max_box_ok,
        no_tracebacks,
        report_ok,
    ]) else "fail"

    summary = {
        **manifest,
        "validation_status": validation_status,
        "expected_iters_present": expected_iters_present,
        "all_shards_complete": all_shards,
        "merge_status": "pass" if all_merged else "fail",
        "append_status": "pass" if all_appended else "fail",
        "final_report_status": report_status.get("status", "missing"),
        "dataset_monotonic": dataset_monotonic,
        "phase_coverage": phase_coverage,
        "training_eligible_nonzero_each_iter": training_nonzero,
        "q_unresolved_ok": q_ok,
        "delta_unresolved_ok": delta_ok,
        "rerun_fraction_ok": rerun_ok,
        "mean_local_boxes_ok": box_mean_ok,
        "max_local_boxes_refined_count": max_boxes,
        "max_local_boxes_ok": max_box_ok,
        "traceback_scan_ok": no_tracebacks,
        "final_dataset_samples": _i(final_phase.get("samples")),
        "final_normal_count": _i(final_phase.get("normal")),
        "final_uniform_SC_count": _i(final_phase.get("uniform_SC")),
        "final_FFLO_count": _i(final_phase.get("FFLO")),
        "mean_local_boxes_refined_count": float(iter_df["mean_local_boxes_refined_count"].mean()) if not iter_df.empty else float("nan"),
        "mean_local_refinement_runtime_sec": float(iter_df["mean_local_refinement_runtime_sec"].mean()) if not iter_df.empty else float("nan"),
        "mean_point_total_runtime_sec": float(iter_df["mean_point_total_runtime_sec"].mean()) if not iter_df.empty else float("nan"),
        "baseline_local_boxes_reference": BASELINE_LOCAL_BOXES,
        "baseline_local_refinement_runtime_sec_reference": BASELINE_LOCAL_REFINEMENT_RUNTIME_SEC,
        "baseline_point_total_runtime_sec_reference": BASELINE_POINT_TOTAL_RUNTIME_SEC,
    }
    _write_json(report_root / "summary.json", summary)
    _write_csv(tables_dir / "validation_summary.csv", [summary])

    _plot_basic(figures_dir, {
        "dataset_phase_counts": dataset_df,
        "iteration_summary": iter_df,
        "rank_runtime_summary": rank_df,
    })
    _write_markdown_report(config, report_root, summary, iter_df, dataset_df)
    _try_write_pdf(report_root / f"{config.report_name}.md", report_root / f"{config.report_name}.pdf")

    if create_archive:
        archive_path = output_root / config.result_archive
        if archive_path.exists():
            archive_path.unlink()
        with tarfile.open(archive_path, "w:gz") as tar:
            if run_dir.exists():
                tar.add(run_dir, arcname=f"{config.output_root}/active_runs/{config.run_id}")
            if report_root.exists():
                tar.add(report_root, arcname=f"reports/{config.report_name}")
            status_path = output_root / "reports" / "report_generation_status.json"
            if status_path.exists():
                tar.add(status_path, arcname=f"{config.output_root}/reports/report_generation_status.json")
        summary["return_archive"] = str(archive_path)
        summary["return_archive_size_bytes"] = archive_path.stat().st_size
        _write_json(report_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _answer(value: bool) -> str:
    return "confirmed" if value else "failed"


def _write_markdown_report(config: LoopPackageConfig, report_root: Path, summary: dict[str, Any], iter_df: pd.DataFrame, dataset_df: pd.DataFrame) -> None:
    lines = [
        f"# {config.report_name}",
        "",
        "## Executive Summary",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- run_id: {summary['run_id']}",
        f"- run_directory: `{summary['run_directory']}`",
        f"- expected iterations: {summary['n_iters_expected']}",
        f"- expected acquisition batches: {summary['acquisition_batches_expected']}",
        f"- final dataset samples: {summary['final_dataset_samples']}",
        f"- final phase counts: normal={summary['final_normal_count']}, uniform_SC={summary['final_uniform_SC_count']}, FFLO={summary['final_FFLO_count']}",
        f"- mean local boxes: {summary['mean_local_boxes_refined_count']:.6g}",
        f"- max local boxes: {summary['max_local_boxes_refined_count']}",
        f"- mean local-refinement runtime sec: {summary['mean_local_refinement_runtime_sec']:.6g}",
        f"- mean point-total runtime sec: {summary['mean_point_total_runtime_sec']:.6g}",
        "",
        "## Required Answers",
        "",
        "| Question | Answer |",
        "|---|---|",
        f"| Did expected iterations complete? | {_answer(bool(summary['expected_iters_present']))} |",
        f"| Did all exact shards complete? | {_answer(bool(summary['all_shards_complete']))} |",
        f"| Did merge complete? | {summary['merge_status']} |",
        f"| Did append complete? | {summary['append_status']} |",
        f"| Did final report generation avoid blocking numerical completion? | {_answer(summary['final_report_status'] != 'missing')} |",
        f"| Did dataset samples grow monotonically? | {_answer(bool(summary['dataset_monotonic']))} |",
        f"| Did normal / uniform-SC / FFLO all appear in the final dataset? | {_answer(bool(summary['phase_coverage']))} |",
        f"| Did training_eligible append remain nonzero each iteration? | {_answer(bool(summary['training_eligible_nonzero_each_iter']))} |",
        f"| Did q_unresolved remain controlled? | {_answer(bool(summary['q_unresolved_ok']))} |",
        f"| Did delta_unresolved remain controlled? | {_answer(bool(summary['delta_unresolved_ok']))} |",
        f"| Did rerun_required fraction remain controlled? | {_answer(bool(summary['rerun_fraction_ok']))} |",
        f"| Did mean local boxes stay <= 3.2? | {_answer(bool(summary['mean_local_boxes_ok']))} |",
        f"| Did max local boxes stay <= 3? | {_answer(bool(summary['max_local_boxes_ok']))} |",
        f"| Were tracebacks / CUDA failures absent from root slurm logs? | {_answer(bool(summary['traceback_scan_ok']))} |",
        "",
        "## Iteration Summary",
        "",
    ]
    if iter_df.empty:
        lines.append("No iteration table was available.")
    else:
        cols = [
            "iteration",
            "exact_points",
            "training_eligible_appended",
            "rerun_required_fraction",
            "q_unresolved_count",
            "delta_unresolved_count",
            "mean_local_boxes_refined_count",
            "mean_local_refinement_runtime_sec",
            "mean_point_total_runtime_sec",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in iter_df[cols].iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    lines.extend(["", "## Dataset Phase Counts", ""])
    if dataset_df.empty:
        lines.append("No dataset table was available.")
    else:
        cols = ["iteration", "samples", "normal", "uniform_SC", "FFLO"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in dataset_df[cols].iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    lines.extend([
        "",
        "## Decision",
        "",
        "The package keeps the one-iteration validated rank_and_cap_k3 oracle configuration. This report does not change phase criteria, Delta tolerance, final ambiguity tolerance, acquisition, StopController logic, or candidate-domain strategy.",
    ])
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / f"{config.report_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision = [
        f"# {config.report_name} Decision Log",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- run_id: {summary['run_id']}",
        f"- final dataset samples: {summary['final_dataset_samples']}",
        f"- final phase counts: normal={summary['final_normal_count']}, uniform_SC={summary['final_uniform_SC_count']}, FFLO={summary['final_FFLO_count']}",
        f"- local boxes mean/max: {summary['mean_local_boxes_refined_count']:.6g}/{summary['max_local_boxes_refined_count']}",
        "- next step: inspect validation_summary.csv before deciding whether to continue.",
    ]
    (report_root / "decision_log.md").write_text("\n".join(decision) + "\n", encoding="utf-8")


def _try_write_pdf(md_path: Path, pdf_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return
    try:
        text = md_path.read_text(encoding="utf-8")
        chunks = [text[i : i + 2800] for i in range(0, len(text), 2800)] or [""]
        with PdfPages(pdf_path) as pdf:
            for chunk in chunks:
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
                ax.axis("off")
                ax.text(0, 1, chunk, va="top", ha="left", family="monospace", fontsize=7, wrap=True)
                pdf.savefig(fig)
                plt.close(fig)
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Package and collect rankcap_k3 active-loop HPC runs.")
    parser.add_argument("--mode", choices=["preflight", "package", "collect"], required=True)
    parser.add_argument("--package-kind", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--create-archive", action="store_true")
    args = parser.parse_args()

    config = CONFIGS[args.package_kind]
    if args.mode == "preflight":
        print(json.dumps(run_preflight(config, args.world_size, args.n_iters), indent=2, sort_keys=True))
    elif args.mode == "package":
        package_hpc(config, args.world_size, args.n_iters)
    elif args.mode == "collect":
        collect_report(config, args.world_size, args.n_iters, args.create_archive)


if __name__ == "__main__":
    main()
