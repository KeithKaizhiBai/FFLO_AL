from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_RE = re.compile(r"dataset_iter(\d{3})\.(npz|csv)$")
ITER_RE = re.compile(r"iter(\d{3})$")
SHARD_RE = re.compile(r"exact_shard_rank(\d{3})_of(\d{3})\.npz$")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_command(cmd: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return {
            "command": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except FileNotFoundError as exc:
        return {"command": cmd, "returncode": None, "stdout": "", "stderr": str(exc)}


def parse_dataset_files(run_dir: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("dataset_iter*.*")):
        match = DATASET_RE.match(path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        suffix = match.group(2)
        row = out.setdefault(iteration, {"iteration": iteration})
        row[f"{suffix}_exists"] = True
        row[f"{suffix}_path"] = str(path)
        row[f"{suffix}_size_bytes"] = path.stat().st_size
    return out


def parse_iteration_dirs(run_dir: Path) -> list[int]:
    iterations: list[int] = []
    for path in sorted(run_dir.glob("iter*")):
        if not path.is_dir():
            continue
        match = ITER_RE.match(path.name)
        if match:
            iterations.append(int(match.group(1)))
    return iterations


def count_selected_points(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return None


def inspect_iteration(run_dir: Path, iteration: int, expected_world_size: int | None) -> dict[str, Any]:
    tag = f"{iteration:03d}"
    iter_dir = run_dir / f"iter{tag}"
    shards = sorted(iter_dir.glob("exact_shard_rank*_of*.npz"))
    shard_ranks: list[int] = []
    shard_world_sizes: set[int] = set()
    for shard in shards:
        match = SHARD_RE.match(shard.name)
        if match:
            shard_ranks.append(int(match.group(1)))
            shard_world_sizes.add(int(match.group(2)))

    expected = expected_world_size
    if expected is None and len(shard_world_sizes) == 1:
        expected = next(iter(shard_world_sizes))
    missing_ranks = ""
    if expected is not None:
        present = set(shard_ranks)
        missing_ranks = ",".join(f"{rank:03d}" for rank in range(expected) if rank not in present)

    selected = iter_dir / "selected_points.csv"
    selected_by_pool = iter_dir / "selected_points_by_pool.csv"
    exact_merged = iter_dir / f"exact_merged_iter{tag}.npz"
    exact_trusted = iter_dir / f"exact_trusted_iter{tag}.npz"
    merge_summary = iter_dir / f"merge_summary_iter{tag}.json"
    stop_metrics = iter_dir / f"stop_metrics_iter{tag}.json"
    selection_summary = iter_dir / "stageiv_selection_summary.json"
    partition_metadata = iter_dir / "partition_metadata.json"
    rerun_points = iter_dir / "rerun_points.csv"

    merge = read_json(merge_summary) if merge_summary.exists() else {}
    stop = read_json(stop_metrics) if stop_metrics.exists() else {}
    selection = read_json(selection_summary) if selection_summary.exists() else {}

    return {
        "iteration": iteration,
        "iter_dir_exists": iter_dir.exists(),
        "selected_points_exists": selected.exists(),
        "selected_points_count": count_selected_points(selected),
        "selected_points_by_pool_exists": selected_by_pool.exists(),
        "selected_points_by_pool_count": count_selected_points(selected_by_pool),
        "partition_metadata_exists": partition_metadata.exists(),
        "selection_summary_exists": selection_summary.exists(),
        "selection_selected_batch_size": selection.get("selected_batch_size", ""),
        "exact_shard_count": len(shards),
        "exact_shard_world_sizes": ",".join(str(v) for v in sorted(shard_world_sizes)),
        "missing_shard_ranks": missing_ranks,
        "exact_merged_exists": exact_merged.exists(),
        "exact_trusted_exists": exact_trusted.exists(),
        "merge_summary_exists": merge_summary.exists(),
        "merge_exact_points": merge.get("exact_points", merge.get("merged_points", "")),
        "merge_training_eligible_points": merge.get("training_eligible_points", ""),
        "merge_rerun_required_points": merge.get("rerun_required_points", ""),
        "stop_metrics_exists": stop_metrics.exists(),
        "stop_reason": stop.get("stop_reason", ""),
        "stop": stop.get("stop", ""),
        "convergence_pass": stop.get("convergence_pass", ""),
        "rerun_points_exists": rerun_points.exists(),
        "rerun_points_count": count_selected_points(rerun_points),
    }


def load_expected_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    return read_json(config_path)


def summarize_status(
    *,
    run_dir: Path,
    output_root: Path,
    run_id: str,
    config: dict[str, Any],
    expected_world_size: int | None,
    job_id: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = parse_dataset_files(run_dir)
    iteration_dirs = parse_iteration_dirs(run_dir)
    max_batches = config.get("max_acquisition_batches")
    if max_batches is not None:
        expected_exact_iterations = list(range(int(max_batches) + 1))
        expected_dataset_iterations = list(range(1, int(max_batches) + 2))
    else:
        max_seen = max(iteration_dirs) if iteration_dirs else -1
        expected_exact_iterations = list(range(max_seen + 1))
        max_dataset = max(datasets) if datasets else 0
        expected_dataset_iterations = list(range(1, max_dataset + 1))

    rows: list[dict[str, Any]] = []
    for iteration in expected_exact_iterations:
        row = inspect_iteration(run_dir, iteration, expected_world_size)
        dataset_next = datasets.get(iteration + 1, {})
        row["expected_next_dataset_iteration"] = iteration + 1
        row["next_dataset_npz_exists"] = bool(dataset_next.get("npz_exists", False))
        row["next_dataset_csv_exists"] = bool(dataset_next.get("csv_exists", False))
        rows.append(row)

    dataset_rows: list[dict[str, Any]] = []
    for iteration in expected_dataset_iterations:
        row = datasets.get(iteration, {"iteration": iteration})
        dataset_rows.append(
            {
                "iteration": iteration,
                "npz_exists": bool(row.get("npz_exists", False)),
                "csv_exists": bool(row.get("csv_exists", False)),
                "npz_size_bytes": row.get("npz_size_bytes", ""),
                "csv_size_bytes": row.get("csv_size_bytes", ""),
                "npz_path": row.get("npz_path", ""),
                "csv_path": row.get("csv_path", ""),
            }
        )

    latest_dataset = max(datasets) if datasets else None
    latest_iteration_dir = max(iteration_dirs) if iteration_dirs else None
    missing_dataset_iterations = [r["iteration"] for r in dataset_rows if not r["npz_exists"]]
    missing_merge_iterations = [r["iteration"] for r in rows if r["iter_dir_exists"] and not r["exact_merged_exists"]]
    missing_trusted_iterations = [r["iteration"] for r in rows if r["iter_dir_exists"] and not r["exact_trusted_exists"]]
    incomplete_shard_iterations = [
        r["iteration"]
        for r in rows
        if r["iter_dir_exists"] and expected_world_size is not None and int(r["exact_shard_count"]) < expected_world_size
    ]

    archive = output_root / "stageiv_3d_topology_full_loop_results.tar.gz"
    postrun_bundle = output_root / "reports" / "stageiv_3d_postrun_bundle" / "stageiv_3d_postrun_bundle_decision.json"
    hpc_commands: dict[str, Any] = {}
    if job_id:
        hpc_commands["squeue"] = run_command(["squeue", "-j", job_id, "-o", "%i|%T|%M|%D|%R"])
        hpc_commands["sacct"] = run_command(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Reason",
                "--parsable2",
            ]
        )

    status = "run_dir_missing"
    if run_dir.exists():
        status = "no_datasets_yet"
        if latest_dataset is not None:
            expected_final_dataset = int(max_batches) + 1 if max_batches is not None else latest_dataset
            status = "complete_file_set_detected" if latest_dataset >= expected_final_dataset and not missing_dataset_iterations else "partial_or_running"
        if postrun_bundle.exists():
            status = "postrun_bundle_detected"

    summary = {
        "run_id": run_id,
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "run_dir_exists": run_dir.exists(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_world_size": expected_world_size,
        "expected_exact_iterations": len(expected_exact_iterations),
        "expected_dataset_iterations": len(expected_dataset_iterations),
        "latest_dataset_iteration": latest_dataset,
        "latest_iteration_dir": latest_iteration_dir,
        "missing_dataset_iterations": missing_dataset_iterations,
        "missing_merge_iterations": missing_merge_iterations,
        "missing_trusted_iterations": missing_trusted_iterations,
        "incomplete_shard_iterations": incomplete_shard_iterations,
        "result_archive_exists": archive.exists(),
        "result_archive_path": str(archive),
        "result_archive_size_bytes": archive.stat().st_size if archive.exists() else None,
        "postrun_bundle_decision_exists": postrun_bundle.exists(),
        "postrun_bundle_decision_path": str(postrun_bundle),
        "hpc_job_id": job_id,
        "hpc_commands": hpc_commands,
        "hpc_status": status,
        "next_action": next_action_for_status(status),
    }
    return summary, rows, dataset_rows


def next_action_for_status(status: str) -> str:
    if status == "run_dir_missing":
        return "check_upload_path_or_submit_stageiv_full_loop"
    if status == "no_datasets_yet":
        return "monitor_slurm_exact_array_and_submit_log"
    if status == "partial_or_running":
        return "continue_monitoring_or_inspect_missing_iteration"
    if status == "complete_file_set_detected":
        return "collect_results_and_run_postrun_bundle"
    if status == "postrun_bundle_detected":
        return "return_archive_and_review_stageiv_postrun_bundle"
    return "inspect_status_outputs"


def write_markdown(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage IV 3D HPC Status",
        "",
        "This report is read-only. It inspects files and optional Slurm status; it does not submit jobs, merge shards, append datasets, or run exact calculations.",
        "",
        "## Summary",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- output_root: `{summary['output_root']}`",
        f"- run_dir_exists: `{summary['run_dir_exists']}`",
        f"- hpc_status: `{summary['hpc_status']}`",
        f"- latest_dataset_iteration: `{summary['latest_dataset_iteration']}`",
        f"- latest_iteration_dir: `{summary['latest_iteration_dir']}`",
        f"- result_archive_exists: `{summary['result_archive_exists']}`",
        f"- postrun_bundle_decision_exists: `{summary['postrun_bundle_decision_exists']}`",
        f"- next_action: `{summary['next_action']}`",
        "",
        "## Missing Items",
        "",
        f"- missing_dataset_iterations: `{summary['missing_dataset_iterations']}`",
        f"- missing_merge_iterations: `{summary['missing_merge_iterations']}`",
        f"- missing_trusted_iterations: `{summary['missing_trusted_iterations']}`",
        f"- incomplete_shard_iterations: `{summary['incomplete_shard_iterations']}`",
        "",
        "## Tables",
        "",
        "- `tables/stageiv_iteration_file_status.csv`",
        "- `tables/stageiv_dataset_file_status.csv`",
        "",
    ]
    if summary.get("hpc_job_id"):
        lines.extend(
            [
                "## Slurm Query",
                "",
                f"- job_id: `{summary['hpc_job_id']}`",
                "- raw `squeue` and `sacct` output are stored in `stageiv_3d_hpc_status.json`.",
                "",
            ]
        )
    write_text(output_dir / "stageiv_3d_hpc_status.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Stage IV 3D HPC run status checker.")
    parser.add_argument("--output-root", type=Path, default=Path("ML_Phase_StageIV_Topology3D"))
    parser.add_argument("--run-id", default="active_phase_topology_3d_t_ja_mu_from_scratch_v1")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/stageiv_3d_production.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--job-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or args.output_root / "active_runs" / args.run_id
    output_dir = args.output_dir or args.output_root / "reports" / "stageiv_3d_hpc_status"
    config = load_expected_config(args.config)
    summary, iteration_rows, dataset_rows = summarize_status(
        run_dir=run_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        config=config,
        expected_world_size=args.world_size,
        job_id=args.job_id or None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    write_json(output_dir / "stageiv_3d_hpc_status.json", summary)
    write_csv(
        tables_dir / "stageiv_iteration_file_status.csv",
        iteration_rows,
        [
            "iteration",
            "iter_dir_exists",
            "selected_points_exists",
            "selected_points_count",
            "selected_points_by_pool_exists",
            "selected_points_by_pool_count",
            "partition_metadata_exists",
            "selection_summary_exists",
            "selection_selected_batch_size",
            "exact_shard_count",
            "exact_shard_world_sizes",
            "missing_shard_ranks",
            "exact_merged_exists",
            "exact_trusted_exists",
            "merge_summary_exists",
            "merge_exact_points",
            "merge_training_eligible_points",
            "merge_rerun_required_points",
            "stop_metrics_exists",
            "stop_reason",
            "stop",
            "convergence_pass",
            "rerun_points_exists",
            "rerun_points_count",
            "expected_next_dataset_iteration",
            "next_dataset_npz_exists",
            "next_dataset_csv_exists",
        ],
    )
    write_csv(
        tables_dir / "stageiv_dataset_file_status.csv",
        dataset_rows,
        ["iteration", "npz_exists", "csv_exists", "npz_size_bytes", "csv_size_bytes", "npz_path", "csv_path"],
    )
    write_markdown(output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
