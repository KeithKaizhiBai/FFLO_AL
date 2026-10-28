from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.stageiv_3d import STAGEIV_OUTPUT_ROOT, STAGEIV_RUN_ID
from scripts.stageiv_3d_hpc_status import load_expected_config, summarize_status


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_final_dataset_iteration(config: dict[str, Any]) -> int | None:
    batches = config.get("max_acquisition_batches")
    if batches is None:
        return None
    return int(batches) + 1


def find_output_root(return_path: Path, output_root_name: str, run_id: str) -> Path:
    if return_path.name == output_root_name:
        return return_path
    if (return_path / "active_runs" / run_id).exists():
        return return_path
    child = return_path / output_root_name
    if child.exists():
        return child
    return return_path


def tar_member_rows(path: Path, run_id: str, final_dataset_iter: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    key_patterns = [
        f"active_runs/{run_id}/dataset_iter",
        f"active_runs/{run_id}/iter",
        "stageiv_3d_postrun_bundle_decision.json",
        "stageiv_3d_topology_full_loop_results.tar.gz",
    ]
    final_npz = f"dataset_iter{final_dataset_iter:03d}.npz" if final_dataset_iter is not None else ""
    final_csv = f"dataset_iter{final_dataset_iter:03d}.csv" if final_dataset_iter is not None else ""
    rows: list[dict[str, Any]] = []
    summary = {
        "tar_readable": False,
        "tar_member_count": 0,
        "contains_run_dir_members": False,
        "contains_final_dataset_npz": False,
        "contains_final_dataset_csv": False,
        "contains_postrun_bundle_decision": False,
    }
    try:
        with tarfile.open(path, "r:*") as tar:
            members = tar.getmembers()
            summary["tar_readable"] = True
            summary["tar_member_count"] = len(members)
            for member in members:
                normalized = member.name.replace("\\", "/")
                is_key = any(pattern in normalized for pattern in key_patterns)
                if final_npz and normalized.endswith(final_npz):
                    is_key = True
                    summary["contains_final_dataset_npz"] = True
                if final_csv and normalized.endswith(final_csv):
                    is_key = True
                    summary["contains_final_dataset_csv"] = True
                if f"active_runs/{run_id}/" in normalized:
                    summary["contains_run_dir_members"] = True
                if normalized.endswith("stageiv_3d_postrun_bundle_decision.json"):
                    summary["contains_postrun_bundle_decision"] = True
                if is_key:
                    rows.append(
                        {
                            "member": normalized,
                            "size_bytes": member.size,
                            "type": "dir" if member.isdir() else "file",
                        }
                    )
    except Exception as exc:
        summary["tar_error"] = str(exc)
    return summary, rows


def inspect_directory(
    *,
    return_path: Path,
    output_root_name: str,
    run_id: str,
    config: dict[str, Any],
    world_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_root = find_output_root(return_path, output_root_name, run_id)
    run_dir = output_root / "active_runs" / run_id
    summary, iteration_rows, dataset_rows = summarize_status(
        run_dir=run_dir,
        output_root=output_root,
        run_id=run_id,
        config=config,
        expected_world_size=world_size,
        job_id=None,
    )
    final_iter = expected_final_dataset_iteration(config)
    final_npz = run_dir / f"dataset_iter{final_iter:03d}.npz" if final_iter is not None else None
    final_csv = run_dir / f"dataset_iter{final_iter:03d}.csv" if final_iter is not None else None
    postrun = output_root / "reports" / "stageiv_3d_postrun_bundle" / "stageiv_3d_postrun_bundle_decision.json"
    artifacts = [
        {
            "artifact": "output_root",
            "path": str(output_root),
            "exists": output_root.exists(),
            "size_bytes": "",
        },
        {
            "artifact": "run_dir",
            "path": str(run_dir),
            "exists": run_dir.exists(),
            "size_bytes": "",
        },
        {
            "artifact": "final_dataset_npz",
            "path": str(final_npz) if final_npz else "",
            "exists": final_npz.exists() if final_npz else False,
            "size_bytes": final_npz.stat().st_size if final_npz and final_npz.exists() else "",
        },
        {
            "artifact": "final_dataset_csv",
            "path": str(final_csv) if final_csv else "",
            "exists": final_csv.exists() if final_csv else False,
            "size_bytes": final_csv.stat().st_size if final_csv and final_csv.exists() else "",
        },
        {
            "artifact": "postrun_bundle_decision",
            "path": str(postrun),
            "exists": postrun.exists(),
            "size_bytes": postrun.stat().st_size if postrun.exists() else "",
        },
    ]
    summary["return_path_type"] = "directory"
    summary["return_path"] = str(return_path)
    summary["resolved_output_root"] = str(output_root)
    summary["expected_final_dataset_iteration"] = final_iter
    summary["final_dataset_npz_exists"] = artifacts[2]["exists"]
    summary["final_dataset_csv_exists"] = artifacts[3]["exists"]
    summary["postrun_bundle_decision_exists"] = artifacts[4]["exists"]
    summary["return_check_status"] = return_status(summary)
    summary["return_next_action"] = return_next_action(summary)
    return summary, iteration_rows, dataset_rows, artifacts


def return_status(summary: dict[str, Any]) -> str:
    if not summary.get("run_dir_exists", False):
        return "run_dir_missing"
    if not summary.get("final_dataset_npz_exists", False):
        return "final_dataset_missing"
    if not summary.get("postrun_bundle_decision_exists", False):
        return "final_dataset_present_postrun_missing"
    return "postrun_bundle_present"


def return_next_action(summary: dict[str, Any]) -> str:
    status = str(summary.get("return_check_status", ""))
    if status == "run_dir_missing":
        return "verify_download_or_extract_result_archive"
    if status == "final_dataset_missing":
        return "inspect_hpc_status_and_continue_or_resume_stageiv_loop"
    if status == "final_dataset_present_postrun_missing":
        return "run_scripts_build_stageiv_3d_all_postrun_reports"
    if status == "postrun_bundle_present":
        return "review_postrun_bundle_decision_and_stageiv_final_report_inputs"
    if status == "tar_only_final_dataset_present":
        return "extract_archive_then_run_postrun_bundle"
    if status == "tar_only_incomplete":
        return "extract_archive_or_recollect_results_then_check_directory"
    return "inspect_return_check_outputs"


def inspect_tar(
    *,
    return_path: Path,
    run_id: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    final_iter = expected_final_dataset_iteration(config)
    tar_summary, key_rows = tar_member_rows(return_path, run_id, final_iter)
    status = "tar_only_final_dataset_present" if tar_summary.get("contains_final_dataset_npz") else "tar_only_incomplete"
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "return_path": str(return_path),
        "return_path_type": "tar_archive",
        "return_archive_sha256": sha256_file(return_path) if return_path.exists() else "",
        "expected_final_dataset_iteration": final_iter,
        "hpc_status": "tar_archive_inspected",
        "return_check_status": status,
        "return_next_action": return_next_action({"return_check_status": status}),
        **tar_summary,
    }
    artifacts = [
        {
            "artifact": "return_archive",
            "path": str(return_path),
            "exists": return_path.exists(),
            "size_bytes": return_path.stat().st_size if return_path.exists() else "",
        }
    ]
    return summary, key_rows, [], artifacts


def write_markdown(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage IV 3D Return Check",
        "",
        "This report is read-only. It validates a returned Stage IV directory or archive without extracting archives, submitting jobs, merging shards, appending datasets, or running exact calculations.",
        "",
        "## Summary",
        "",
        f"- run_id: `{summary.get('run_id')}`",
        f"- return_path: `{summary.get('return_path')}`",
        f"- return_path_type: `{summary.get('return_path_type')}`",
        f"- expected_final_dataset_iteration: `{summary.get('expected_final_dataset_iteration')}`",
        f"- return_check_status: `{summary.get('return_check_status')}`",
        f"- hpc_status: `{summary.get('hpc_status')}`",
        f"- return_next_action: `{summary.get('return_next_action')}`",
        "",
        "## Key Evidence",
        "",
        f"- final_dataset_npz_exists: `{summary.get('final_dataset_npz_exists', summary.get('contains_final_dataset_npz'))}`",
        f"- final_dataset_csv_exists: `{summary.get('final_dataset_csv_exists', summary.get('contains_final_dataset_csv'))}`",
        f"- postrun_bundle_decision_exists: `{summary.get('postrun_bundle_decision_exists', summary.get('contains_postrun_bundle_decision'))}`",
        "",
        "## Tables",
        "",
        "- `tables/stageiv_return_artifacts.csv`",
        "- `tables/stageiv_return_dataset_status.csv` when a directory was inspected",
        "- `tables/stageiv_return_iteration_status.csv` when a directory was inspected",
        "- `tables/stageiv_return_tar_key_members.csv` when an archive was inspected",
        "",
        "## Next Command Hints",
        "",
    ]
    if summary.get("return_check_status") == "final_dataset_present_postrun_missing":
        lines.extend(
            [
                "Run the report-only post-run bundle from the package or repository root:",
                "",
                "```bash",
                "REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_all_postrun_reports.sh",
                "```",
                "",
            ]
        )
    elif summary.get("return_check_status") == "tar_only_final_dataset_present":
        lines.extend(
            [
                "Extract the archive, then rerun this checker on the extracted `ML_Phase_StageIV_Topology3D` directory before the post-run bundle.",
                "",
            ]
        )
    write_text(output_dir / "stageiv_3d_return_check.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only checker for returned Stage IV 3D HPC outputs.")
    parser.add_argument("--return-path", type=Path, default=Path(STAGEIV_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=STAGEIV_RUN_ID)
    parser.add_argument("--output-root-name", default=STAGEIV_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/stageiv_3d_production.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/stageiv_3d_return_check"))
    parser.add_argument("--world-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_expected_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.return_path.is_file() and args.return_path.suffixes[-2:] in [[".tar", ".gz"], [".tgz"]]:
        summary, first_table, dataset_rows, artifact_rows = inspect_tar(
            return_path=args.return_path,
            run_id=args.run_id,
            config=config,
        )
        write_csv(
            args.output_dir / "tables" / "stageiv_return_tar_key_members.csv",
            first_table,
            ["member", "size_bytes", "type"],
        )
    elif args.return_path.exists() and args.return_path.is_dir():
        summary, first_table, dataset_rows, artifact_rows = inspect_directory(
            return_path=args.return_path,
            output_root_name=args.output_root_name,
            run_id=args.run_id,
            config=config,
            world_size=args.world_size,
        )
        write_csv(
            args.output_dir / "tables" / "stageiv_return_iteration_status.csv",
            first_table,
            [
                "iteration",
                "iter_dir_exists",
                "selected_points_exists",
                "selected_points_count",
                "exact_shard_count",
                "missing_shard_ranks",
                "exact_merged_exists",
                "exact_trusted_exists",
                "merge_summary_exists",
                "stop_metrics_exists",
                "rerun_points_exists",
                "expected_next_dataset_iteration",
                "next_dataset_npz_exists",
                "next_dataset_csv_exists",
            ],
        )
        write_csv(
            args.output_dir / "tables" / "stageiv_return_dataset_status.csv",
            dataset_rows,
            ["iteration", "npz_exists", "csv_exists", "npz_size_bytes", "csv_size_bytes", "npz_path", "csv_path"],
        )
    else:
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "return_path": str(args.return_path),
            "return_path_type": "missing",
            "return_check_status": "return_path_missing",
            "return_next_action": "download_or_extract_stageiv_results",
        }
        artifact_rows = [
            {
                "artifact": "return_path",
                "path": str(args.return_path),
                "exists": False,
                "size_bytes": "",
            }
        ]
    write_csv(
        args.output_dir / "tables" / "stageiv_return_artifacts.csv",
        artifact_rows,
        ["artifact", "path", "exists", "size_bytes"],
    )
    write_json(args.output_dir / "stageiv_3d_return_check.json", summary)
    write_markdown(args.output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
