from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


PACKAGE_NAME = "local_refinement_refactor_stage01_instrumentation"
REQUIRED_PATHS = [
    "README.md",
    "RUN_MANIFEST.json",
    "fixed_points/fixed_point_regression_points.csv",
    "eta_phase_diagram_cuda.py",
    "tfflo_1d_cuda.py",
    "ml_phase/exact_oracle.py",
    "scripts/run_local_refinement_fixed_point_regression.py",
    "scripts/compare_local_refinement_variants.py",
    "scripts/verify_local_refinement_stage1_gate.py",
    "scripts/collect_local_refinement_stage1_outputs.py",
    "scripts/preflight_local_refinement_stage1_hpc.py",
    "scripts/submit_stage1_regression_workflow.sh",
    "scripts/submit_local_refinement_fixed_point_regression.sh",
    "scripts/submit_local_refinement_instrumented_benchmark.sh",
    "scripts/slurm_stage1_postprocess.sh",
    "scripts/collect_stage1_regression_outputs.sh",
    "scripts/verify_stage1_gate.sh",
]
FORBIDDEN_PARTS = {
    "__pycache__",
    "active_runs",
    "datasets",
    "figures",
    "hpc_jobs",
    "models",
}
FORBIDDEN_SUFFIXES = (
    "local_refinement_refactor_stage1_regression_results.tar.gz",
    "local_refinement_refactor_stage1_regression_results_test.tar.gz",
)
REQUIRED_MANIFEST_KEYS = [
    "package_name",
    "purpose",
    "active_learning",
    "fixed_points",
    "baseline_output",
    "instrumented_output",
    "comparison_output",
    "instrumentation_flag",
    "expected_local_box_csv",
    "workflow_submit",
    "workflow_aliases",
    "preflight",
    "postprocess_job",
    "return_archive",
    "expected_fixed_points",
]
README_COMMANDS = [
    "python scripts/preflight_local_refinement_stage1_hpc.py --package-root .",
    "bash scripts/submit_stage1_regression_workflow.sh",
    "bash scripts/submit_local_refinement_fixed_point_regression.sh",
    "bash scripts/submit_local_refinement_instrumented_benchmark.sh",
    "bash scripts/compare_stage1_regression.sh",
    "bash scripts/verify_stage1_gate.sh",
    "bash scripts/collect_stage1_regression_outputs.sh",
]
GPU_SLURM_SCRIPTS = [
    "scripts/slurm_stage0_baseline_regression.sh",
    "scripts/slurm_stage1_instrumented_regression.sh",
]


def _rel_names(root: Path) -> set[str]:
    return {str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file()}


def _archive_names(archive: Path) -> set[str]:
    with tarfile.open(archive, "r:gz") as tar:
        names: set[str] = set()
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            prefix = PACKAGE_NAME + "/"
            if name.startswith(prefix):
                name = name[len(prefix) :]
            names.add(name)
        return names


def _read_text_from_archive(archive: Path, rel_path: str) -> str:
    with tarfile.open(archive, "r:gz") as tar:
        member_name = f"{PACKAGE_NAME}/{rel_path}"
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(member_name)
        return extracted.read().decode("utf-8")


def _read_text_from_dir(root: Path, rel_path: str) -> str:
    return (root / rel_path).read_text(encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_names(names: set[str]) -> list[str]:
    failures: list[str] = []
    for rel_path in REQUIRED_PATHS:
        if rel_path not in names:
            failures.append(f"missing required path: {rel_path}")
    for name in sorted(names):
        parts = set(Path(name).parts)
        forbidden = sorted(parts & FORBIDDEN_PARTS)
        if forbidden and name.startswith(("ml_phase/", "code_snapshot/ml_phase/")):
            failures.append(f"forbidden ml_phase output/cache path: {name}")
        if name.endswith(FORBIDDEN_SUFFIXES):
            failures.append(f"forbidden return-bundle artifact: {name}")
        if "return_bundle_metadata" in parts or "imported_results" in parts:
            failures.append(f"forbidden transient metadata path: {name}")
    return failures


def _validate_texts(read_text, names: set[str]) -> list[str]:
    failures: list[str] = []
    if "RUN_MANIFEST.json" not in names:
        failures.append("RUN_MANIFEST.json missing")
        manifest = {}
    else:
        manifest = json.loads(read_text("RUN_MANIFEST.json"))
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            failures.append(f"RUN_MANIFEST.json missing key: {key}")
    if manifest.get("package_name") != PACKAGE_NAME:
        failures.append(f"manifest package_name mismatch: {manifest.get('package_name')}")
    if manifest.get("active_learning") != "not_run":
        failures.append(f"manifest active_learning should be not_run, got {manifest.get('active_learning')}")
    if manifest.get("preflight") != "scripts/preflight_local_refinement_stage1_hpc.py":
        failures.append(f"manifest preflight mismatch: {manifest.get('preflight')}")
    if manifest.get("expected_fixed_points") != 32:
        failures.append(f"manifest expected_fixed_points should be 32, got {manifest.get('expected_fixed_points')}")

    if "README.md" not in names:
        failures.append("README.md missing")
    else:
        readme = read_text("README.md")
        for command in README_COMMANDS:
            if command not in readme:
                failures.append(f"README missing command: {command}")

    for name in sorted(n for n in names if n.endswith(".sh")):
        text = read_text(name)
        if "\r" in text:
            failures.append(f"shell script has CR line endings: {name}")
        if not text.startswith("#!/bin/bash"):
            failures.append(f"shell script missing bash shebang: {name}")
        if name in GPU_SLURM_SCRIPTS:
            if "#SBATCH --exclude=gpuh01" not in text:
                failures.append(f"GPU Slurm script missing gpuh01 exclusion: {name}")
            if 'torch.empty(1, device="cuda")' not in text:
                failures.append(f"GPU Slurm script missing CUDA runtime probe: {name}")
    return failures


def _validate_archive_sidecars(archive: Path, archive_sha256: str) -> list[str]:
    failures: list[str] = []
    archive_size = archive.stat().st_size
    sha256_sidecar = archive.with_suffix(archive.suffix + ".sha256")
    metadata_sidecar = archive.with_suffix(archive.suffix + ".metadata.json")

    if not sha256_sidecar.exists():
        failures.append(f"missing archive sha256 sidecar: {sha256_sidecar}")
    else:
        line = sha256_sidecar.read_text(encoding="utf-8").strip()
        parts = line.split()
        if len(parts) != 2:
            failures.append(f"sha256 sidecar should contain digest and archive name: {sha256_sidecar}")
        else:
            sidecar_digest, sidecar_name = parts
            if sidecar_digest != archive_sha256:
                failures.append(
                    f"sha256 sidecar digest mismatch: {sidecar_digest} != {archive_sha256}"
                )
            if sidecar_name != archive.name:
                failures.append(f"sha256 sidecar archive name mismatch: {sidecar_name} != {archive.name}")

    if not metadata_sidecar.exists():
        failures.append(f"missing archive metadata sidecar: {metadata_sidecar}")
    else:
        metadata = json.loads(metadata_sidecar.read_text(encoding="utf-8"))
        if metadata.get("package_name") != PACKAGE_NAME:
            failures.append(f"metadata package_name mismatch: {metadata.get('package_name')}")
        if metadata.get("archive_name") != archive.name:
            failures.append(f"metadata archive_name mismatch: {metadata.get('archive_name')}")
        if metadata.get("archive_size_bytes") != archive_size:
            failures.append(
                f"metadata archive_size_bytes mismatch: {metadata.get('archive_size_bytes')} != {archive_size}"
            )
        if metadata.get("archive_sha256") != archive_sha256:
            failures.append(
                f"metadata archive_sha256 mismatch: {metadata.get('archive_sha256')} != {archive_sha256}"
            )
    return failures


def validate_package(package_dir: Path | None, archive: Path | None) -> dict[str, Any]:
    failures: list[str] = []
    checked: dict[str, Any] = {}
    if package_dir is not None:
        names = _rel_names(package_dir)
        failures.extend(f"directory: {failure}" for failure in _validate_names(names))
        failures.extend(f"directory: {failure}" for failure in _validate_texts(lambda p: _read_text_from_dir(package_dir, p), names))
        checked["directory_file_count"] = len(names)
    if archive is not None:
        names = _archive_names(archive)
        failures.extend(f"archive: {failure}" for failure in _validate_names(names))
        failures.extend(f"archive: {failure}" for failure in _validate_texts(lambda p: _read_text_from_archive(archive, p), names))
        archive_sha256 = _sha256_file(archive)
        failures.extend(f"archive: {failure}" for failure in _validate_archive_sidecars(archive, archive_sha256))
        checked["archive_file_count"] = len(names)
        checked["archive_size_bytes"] = archive.stat().st_size if archive.exists() else 0
        checked["archive_sha256"] = archive_sha256
    return {
        "status": "pass" if not failures else "fail",
        "package_dir": str(package_dir) if package_dir is not None else None,
        "archive": str(archive) if archive is not None else None,
        "checked": checked,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Stage 1 local-refinement HPC package contents.")
    parser.add_argument("--package-dir", type=Path, default=Path("hpc_packages") / PACKAGE_NAME)
    parser.add_argument("--archive", type=Path, default=Path("hpc_packages") / f"{PACKAGE_NAME}.tar.gz")
    parser.add_argument("--output-json", type=Path, default=Path("reports/local_refinement_refactor/stage_01_instrumentation/package_validation.json"))
    args = parser.parse_args()
    package_dir = args.package_dir if args.package_dir.exists() else None
    archive = args.archive if args.archive.exists() else None
    summary = validate_package(package_dir, archive)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
