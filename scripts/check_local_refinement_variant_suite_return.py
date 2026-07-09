from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path
from typing import Any


RETURN_ARCHIVE_NAME = "local_refinement_refactor_variant_suite_results.tar.gz"
REQUIRED_ARCHIVE_PREFIXES = (
    "RUN_MANIFEST.json",
    "fixed_points/fixed_point_regression_points.csv",
    "logs/",
    "reports/local_refinement_refactor/variant_regression/",
)
FATAL_LOG_PATTERNS = {
    "old_nvidia_driver": re.compile(r"NVIDIA driver.*too old|driver on your system is too old", re.IGNORECASE),
    "cuda_runtime_probe_fail": re.compile(r"cuda_runtime_probe=fail", re.IGNORECASE),
    "cuda_runtime_error": re.compile(r"RuntimeError:.*CUDA|torch\._C\._cuda_init", re.IGNORECASE),
    "slurm_failed": re.compile(r"\b(CANCELLED|FAILED|OUT_OF_MEMORY|TIMEOUT)\b", re.IGNORECASE),
    "excluded_node_seen": re.compile(r"\bgpuh01\b", re.IGNORECASE),
}
LOG_SUFFIXES = {".log", ".out", ".err", ".txt"}


def _display(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def find_return_archive(path: Path) -> Path | None:
    if path.is_file():
        if path.name == RETURN_ARCHIVE_NAME:
            return path
        return None
    if not path.exists():
        return None
    exact = sorted(p for p in path.rglob(RETURN_ARCHIVE_NAME) if p.is_file())
    return exact[0] if exact else None


def inspect_return_archive(archive: Path) -> dict[str, Any]:
    check: dict[str, Any] = {
        "path": str(archive),
        "exists": archive.exists(),
        "size_bytes": archive.stat().st_size if archive.exists() else 0,
        "status": "missing",
        "missing_required_prefixes": list(REQUIRED_ARCHIVE_PREFIXES),
    }
    if not archive.exists():
        return check

    try:
        with tarfile.open(archive, "r:*") as tar:
            names = set(tar.getnames())
    except tarfile.TarError as exc:
        check["status"] = "fail"
        check["tar_error"] = repr(exc)
        return check

    missing = [
        prefix
        for prefix in REQUIRED_ARCHIVE_PREFIXES
        if prefix not in names and not any(name.startswith(prefix) for name in names)
    ]
    check["member_count"] = len(names)
    check["missing_required_prefixes"] = missing
    check["status"] = "pass" if not missing else "fail"
    return check


def scan_logs(path: Path) -> dict[str, Any]:
    root = path if path.is_dir() else path.parent
    matches: list[dict[str, Any]] = []
    files_scanned = 0
    if root.exists():
        for candidate in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in LOG_SUFFIXES):
            files_scanned += 1
            try:
                lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                for key, pattern in FATAL_LOG_PATTERNS.items():
                    if pattern.search(line):
                        matches.append(
                            {
                                "kind": key,
                                "file": _display(candidate, root),
                                "line": line_number,
                                "text": line.strip()[:300],
                            }
                        )
    return {
        "status": "pass" if not matches else "fail",
        "root": str(root),
        "files_scanned": files_scanned,
        "matches": matches,
    }


def check_return_readiness(path: Path) -> dict[str, Any]:
    target = path.resolve()
    archive = find_return_archive(target)
    archive_check = inspect_return_archive(archive) if archive is not None else {
        "path": "",
        "exists": False,
        "size_bytes": 0,
        "status": "missing",
        "missing_required_prefixes": list(REQUIRED_ARCHIVE_PREFIXES),
    }
    log_check = scan_logs(target)
    failures: list[str] = []
    if archive is None:
        failures.append(f"missing {RETURN_ARCHIVE_NAME}")
    elif archive_check.get("status") != "pass":
        failures.append("return archive is present but failed structural inspection")
    if log_check.get("status") != "pass":
        failures.append("fatal or policy-relevant log patterns were found")

    status = "ready_to_import" if not failures else "not_ready"
    import_command = (
        f"python scripts/import_local_refinement_variant_suite_results.py {archive}"
        if archive is not None
        else f"python scripts/import_local_refinement_variant_suite_results.py {RETURN_ARCHIVE_NAME}"
    )
    return {
        "status": status,
        "checked_path": str(target),
        "return_archive": str(archive) if archive is not None else "",
        "archive_check": archive_check,
        "log_check": log_check,
        "failures": failures,
        "next_import_command": import_command if status == "ready_to_import" else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a downloaded variant-suite HPC result directory contains a return archive ready for import."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Downloaded return archive, extracted variant-suite package, run root, or parent directory to inspect.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = check_return_readiness(args.path)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "ready_to_import" else 1


if __name__ == "__main__":
    raise SystemExit(main())
