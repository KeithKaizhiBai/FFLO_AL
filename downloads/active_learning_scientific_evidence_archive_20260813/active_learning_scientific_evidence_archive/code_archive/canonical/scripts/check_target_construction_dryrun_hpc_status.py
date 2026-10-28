from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


RESULT_ARCHIVE = "local_refinement_target_construction_dryrun_results.tar.gz"
DEFAULT_RUN_ROOT_NAME = "local_refinement_target_construction_dryrun_run"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"json_error": str(exc), "path": str(path)}


def _read_jobids(log_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not log_dir.exists():
        return out
    for path in sorted(log_dir.glob("*.jobid")):
        out[path.stem] = path.read_text(encoding="utf-8", errors="ignore").strip()
    return out


def _query_job(jobid: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    squeue = shutil.which("squeue")
    sacct = shutil.which("sacct")
    if squeue:
        proc = subprocess.run(
            [squeue, "-j", jobid, "-h", "-o", "%A|%T|%M|%R"],
            text=True,
            capture_output=True,
            check=False,
        )
        result["squeue_returncode"] = proc.returncode
        result["squeue_stdout"] = proc.stdout.strip()
        result["squeue_stderr"] = proc.stderr.strip()
    if sacct:
        proc = subprocess.run(
            [sacct, "-j", jobid, "--format=JobID,State,ExitCode,Elapsed", "--parsable2", "--noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        result["sacct_returncode"] = proc.returncode
        result["sacct_stdout"] = proc.stdout.strip()
        result["sacct_stderr"] = proc.stderr.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check target-construction dry-run HPC status.")
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--query-scheduler", action="store_true")
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    run_root = (args.run_root or package_root / DEFAULT_RUN_ROOT_NAME).resolve()
    archive = run_root / RESULT_ARCHIVE
    status_json = (
        run_root
        / "reports"
        / "local_refinement_refactor"
        / "target_construction_dryrun"
        / "summary"
        / "target_construction_gate_status.json"
    )
    jobids = _read_jobids(run_root / "logs")
    summary = {
        "status": "complete" if archive.exists() else "pending_or_missing_return_archive",
        "package_root": str(package_root),
        "run_root": str(run_root),
        "return_archive": {
            "path": str(archive),
            "exists": archive.exists(),
            "size_bytes": archive.stat().st_size if archive.exists() else 0,
        },
        "gate_status": _read_json(status_json),
        "jobids": jobids,
    }
    if args.query_scheduler:
        summary["scheduler"] = {name: _query_job(jobid) for name, jobid in jobids.items() if jobid}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
