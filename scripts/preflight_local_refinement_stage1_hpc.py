from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FIXED_POINTS = 32
PACKAGE_NAME = "local_refinement_refactor_stage01_instrumentation"
REPORT_DIR = ROOT / "reports" / "local_refinement_refactor" / "stage_01_instrumentation"
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
    "scripts/submit_stage1_regression_workflow.sh",
    "scripts/submit_local_refinement_fixed_point_regression.sh",
    "scripts/submit_local_refinement_instrumented_benchmark.sh",
    "scripts/slurm_stage0_baseline_regression.sh",
    "scripts/slurm_stage1_instrumented_regression.sh",
    "scripts/slurm_stage1_postprocess.sh",
]
PY_COMPILE_PATHS = [
    "ml_phase/exact_oracle.py",
    "scripts/run_local_refinement_fixed_point_regression.py",
    "scripts/compare_local_refinement_variants.py",
    "scripts/verify_local_refinement_stage1_gate.py",
    "scripts/collect_local_refinement_stage1_outputs.py",
    "scripts/preflight_local_refinement_stage1_hpc.py",
]
IMPORT_CHECK_MODULES = [
    "eta_phase_diagram_cuda",
    "ml_phase.exact_oracle",
]


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _torch_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "importable": False,
        "cuda_available": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "error": None,
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on HPC environment.
        snapshot["error"] = repr(exc)
        return snapshot
    snapshot["importable"] = True
    snapshot["torch_version"] = str(torch.__version__)
    snapshot["cuda_available"] = bool(torch.cuda.is_available())
    snapshot["torch_cuda_version"] = str(torch.version.cuda)
    return snapshot


def _default_run_root(package_root: Path) -> Path:
    env_run_root = os.environ.get("RUN_ROOT")
    if env_run_root:
        return Path(env_run_root)
    if os.access(package_root, os.W_OK):
        return package_root / "local_refinement_refactor_stage1_run"
    fallback = os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or os.environ.get("HOME")
    if fallback:
        return Path(fallback) / "local_refinement_refactor_stage1_run"
    return package_root / "local_refinement_refactor_stage1_run"


def _check_imports(package_root: Path, module_names: list[str]) -> tuple[list[str], list[str]]:
    package_root_text = str(package_root)
    inserted = False
    if package_root_text not in sys.path:
        sys.path.insert(0, package_root_text)
        inserted = True
    module_keys = set(module_names)
    module_keys.update(module_name.split(".", 1)[0] for module_name in module_names)
    previous_modules = {module_key: sys.modules.get(module_key) for module_key in module_keys}
    previous_dont_write_bytecode = sys.dont_write_bytecode
    for module_key in module_keys:
        sys.modules.pop(module_key, None)
    imported: list[str] = []
    failures: list[str] = []
    try:
        sys.dont_write_bytecode = True
        for module_name in module_names:
            try:
                importlib.import_module(module_name)
                imported.append(module_name)
            except Exception as exc:  # pragma: no cover - depends on HPC Python environment.
                failures.append(f"import check failed for {module_name}: {exc!r}")
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        for module_key in module_keys:
            sys.modules.pop(module_key, None)
        for module_key, module in previous_modules.items():
            if module is not None:
                sys.modules[module_key] = module
        if inserted:
            try:
                sys.path.remove(package_root_text)
            except ValueError:
                pass
    return imported, failures


def run_preflight(
    package_root: Path,
    output_json: Path | None = None,
    *,
    expected_fixed_points: int = EXPECTED_FIXED_POINTS,
    require_cuda: bool = False,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    failures: list[str] = []
    checked: dict[str, Any] = {
        "package_root": str(package_root),
        "expected_fixed_points": expected_fixed_points,
    }

    for rel_path in REQUIRED_PATHS:
        path = package_root / rel_path
        if not path.exists():
            failures.append(f"missing required path: {rel_path}")

    manifest_path = package_root / "RUN_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked["package_name"] = manifest.get("package_name")
        checked["active_learning"] = manifest.get("active_learning")
        if manifest.get("package_name") != PACKAGE_NAME:
            failures.append(f"manifest package_name mismatch: {manifest.get('package_name')}")
        if manifest.get("active_learning") != "not_run":
            failures.append(f"manifest active_learning should be not_run, got {manifest.get('active_learning')}")
        if manifest.get("fixed_points") != "fixed_points/fixed_point_regression_points.csv":
            failures.append(f"manifest fixed_points path mismatch: {manifest.get('fixed_points')}")
        if manifest.get("instrumentation_flag") != "--enable-local-box-instrumentation":
            failures.append(f"manifest instrumentation_flag mismatch: {manifest.get('instrumentation_flag')}")

    fixed_points = package_root / "fixed_points" / "fixed_point_regression_points.csv"
    if fixed_points.exists():
        fixed_point_count = _count_csv_rows(fixed_points)
        checked["fixed_point_count"] = fixed_point_count
        if fixed_point_count != expected_fixed_points:
            failures.append(f"fixed-point row count mismatch: {fixed_point_count} != {expected_fixed_points}")

    syntax_checked: list[str] = []
    for rel_path in PY_COMPILE_PATHS:
        path = package_root / rel_path
        if not path.exists():
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            syntax_checked.append(rel_path)
        except SyntaxError as exc:
            failures.append(f"syntax check failed for {rel_path}: {exc}")
    checked["syntax_check_count"] = len(syntax_checked)
    checked["syntax_check_paths"] = syntax_checked

    imported_modules, import_failures = _check_imports(package_root, IMPORT_CHECK_MODULES)
    checked["import_check_count"] = len(imported_modules)
    checked["import_check_modules"] = imported_modules
    failures.extend(import_failures)

    torch_info = _torch_snapshot()
    checked["torch"] = torch_info
    if require_cuda and not torch_info.get("cuda_available"):
        failures.append("torch CUDA is not available")

    summary = {
        "status": "pass" if not failures else "fail",
        "checked": checked,
        "failures": failures,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the Stage 1 fixed-point HPC package before Slurm submit.")
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=REPORT_DIR / "stage1_runtime_preflight.json")
    parser.add_argument("--expected-fixed-points", type=int, default=EXPECTED_FIXED_POINTS)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    if args.package_root is None:
        package_root = ROOT
    else:
        package_root = args.package_root if args.package_root.is_absolute() else Path.cwd() / args.package_root
    run_root = args.run_root or _default_run_root(package_root)
    output_json = args.output_json if args.output_json.is_absolute() else run_root / args.output_json
    summary = run_preflight(
        package_root,
        output_json,
        expected_fixed_points=args.expected_fixed_points,
        require_cuda=args.require_cuda,
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
