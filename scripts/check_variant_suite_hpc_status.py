from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


RESULT_ARCHIVE = "local_refinement_refactor_variant_suite_results.tar.gz"
ARRAY_SUITE_STATUS = Path("reports/local_refinement_refactor/variant_regression/summary/array_suite_status.json")
JOBID_SUFFIX = ".jobid"
LOG_SUFFIXES = {".log", ".out", ".err", ".txt"}
FATAL_PATTERNS = {
    "old_nvidia_driver": re.compile(r"NVIDIA driver.*too old|driver on your system is too old", re.IGNORECASE),
    "cuda_runtime_probe_fail": re.compile(r"cuda_runtime_probe=fail", re.IGNORECASE),
    "cuda_runtime_error": re.compile(r"RuntimeError:.*CUDA|torch\._C\._cuda_init", re.IGNORECASE),
    "slurm_failed": re.compile(r"\b(CANCELLED|FAILED|OUT_OF_MEMORY|TIMEOUT)\b", re.IGNORECASE),
}


def _display(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def default_run_root(package_root: Path) -> Path:
    return package_root / "local_refinement_refactor_variant_suite_run"


def read_jobids(run_root: Path) -> dict[str, str]:
    logs = run_root / "logs"
    jobids: dict[str, str] = {}
    if not logs.exists():
        return jobids
    for path in sorted(logs.glob(f"*{JOBID_SUFFIX}")):
        value = path.read_text(encoding="utf-8", errors="ignore").strip()
        jobids[path.stem] = value
    return jobids


def scan_logs(run_root: Path) -> dict[str, Any]:
    logs = run_root / "logs"
    matches: list[dict[str, Any]] = []
    files_scanned = 0
    if logs.exists():
        for path in sorted(p for p in logs.rglob("*") if p.is_file() and p.suffix.lower() in LOG_SUFFIXES):
            files_scanned += 1
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line_number, line in enumerate(lines, start=1):
                for kind, pattern in FATAL_PATTERNS.items():
                    if pattern.search(line):
                        matches.append(
                            {
                                "kind": kind,
                                "file": _display(path, run_root),
                                "line": line_number,
                                "text": line.strip()[:300],
                            }
                        )
    return {
        "status": "pass" if not matches else "fail",
        "files_scanned": files_scanned,
        "matches": matches,
    }


def read_array_suite_summary(run_root: Path) -> dict[str, Any]:
    path = run_root / ARRAY_SUITE_STATUS
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "status": "not_found",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "status": "unreadable",
            "error": repr(exc),
        }
    return {
        "path": str(path),
        "exists": True,
        "status": payload.get("status", "unknown"),
        "expected_tasks": payload.get("expected_tasks"),
        "successful_tasks": payload.get("successful_tasks"),
        "failed_or_missing_tasks": payload.get("failed_or_missing_tasks"),
        "task_status": payload.get("task_status"),
        "comparison_status": payload.get("comparison_status"),
        "performance_status": payload.get("performance_status"),
        "failures": payload.get("failures", []),
    }


def query_scheduler(jobids: dict[str, str]) -> dict[str, Any]:
    tools = {
        "squeue": shutil.which("squeue"),
        "sacct": shutil.which("sacct"),
    }
    results: dict[str, Any] = {"tools": tools, "jobs": {}}
    for label, jobid in jobids.items():
        job_result: dict[str, Any] = {}
        if tools["squeue"]:
            completed = subprocess.run(
                [tools["squeue"], "-j", jobid, "-h", "-o", "%i|%T|%N|%R"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            job_result["squeue_returncode"] = completed.returncode
            job_result["squeue_stdout"] = completed.stdout.strip()
            job_result["squeue_stderr"] = completed.stderr.strip()
        if tools["sacct"]:
            completed = subprocess.run(
                [tools["sacct"], "-j", jobid, "--format=JobID,State,ExitCode,Elapsed", "--parsable2", "-n"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            job_result["sacct_returncode"] = completed.returncode
            job_result["sacct_stdout"] = completed.stdout.strip()
            job_result["sacct_stderr"] = completed.stderr.strip()
        results["jobs"][label] = job_result
    return results


def check_hpc_status(package_root: Path, run_root: Path | None = None, *, query_scheduler_enabled: bool = False) -> dict[str, Any]:
    package_root = package_root.resolve()
    run_root = (run_root if run_root is not None else default_run_root(package_root)).resolve()
    archive = run_root / RESULT_ARCHIVE
    jobids = read_jobids(run_root)
    log_scan = scan_logs(run_root)
    array_summary = read_array_suite_summary(run_root)
    scheduler = query_scheduler(jobids) if query_scheduler_enabled else {"queried": False}

    archive_exists = archive.exists()
    failures: list[str] = []
    if not archive_exists:
        failures.append(f"missing return archive: {archive}")
    if log_scan["status"] != "pass":
        failures.append("fatal CUDA/Slurm patterns were found in logs")
    if array_summary["exists"] and array_summary["status"] != "pass":
        failures.append(f"array suite summary status is {array_summary['status']}")

    if archive_exists and log_scan["status"] == "pass" and (
        not array_summary["exists"] or array_summary["status"] == "pass"
    ):
        status = "ready_to_return"
    elif archive_exists:
        status = "ready_to_return_with_validation_failures"
    elif log_scan["status"] != "pass":
        status = "failed_or_needs_log_review"
    else:
        status = "pending_or_missing_return_archive"

    return {
        "status": status,
        "package_root": str(package_root),
        "run_root": str(run_root),
        "return_archive": {
            "path": str(archive),
            "exists": archive_exists,
            "size_bytes": archive.stat().st_size if archive_exists else 0,
        },
        "jobids": jobids,
        "log_scan": log_scan,
        "array_suite_summary": array_summary,
        "scheduler": scheduler,
        "failures": failures,
        "next_action": (
            f"download {archive}"
            if status == "ready_to_return"
            else (
                f"download {archive} and inspect summary/missing_or_failed_tasks.csv"
                if status == "ready_to_return_with_validation_failures"
                else "inspect logs and scheduler state, then rerun failed jobs or postprocess"
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect variant-suite HPC job state, logs, and return archive.")
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--query-scheduler", action="store_true")
    parser.add_argument("--fail-if-not-ready", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = check_hpc_status(args.package_root, args.run_root, query_scheduler_enabled=args.query_scheduler)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.fail_if_not_ready and result["status"] != "ready_to_return":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
