from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

try:
    from scripts.verify_local_refinement_stage1_gate import verify_gate, write_gate_report
except ModuleNotFoundError:
    from verify_local_refinement_stage1_gate import verify_gate, write_gate_report


ROOT = Path(__file__).resolve().parents[1]


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


DEFAULT_INCLUDE_PATHS = [
    "README.md",
    "RUN_MANIFEST.json",
    "logs",
    "reports/local_refinement_refactor/stage_00_baseline/regression_gpu_baseline",
    "reports/local_refinement_refactor/stage_01_instrumentation/regression_gpu_instrumented",
    "reports/local_refinement_refactor/stage_01_instrumentation/baseline_vs_instrumented",
    "reports/local_refinement_refactor/stage_01_instrumentation/return_bundle_metadata",
    "fixed_points/fixed_point_regression_points.csv",
]


def _iter_existing_paths(root: Path, rel_path: str) -> list[Path]:
    path = root / rel_path
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return [p for p in path.rglob("*") if p.is_file()]


def collect_outputs(package_root: Path, archive_path: Path, run_root: Path | None = None) -> dict[str, object]:
    run_root = run_root or _default_run_root(package_root)
    metadata_dir = run_root / "reports" / "local_refinement_refactor" / "stage_01_instrumentation" / "return_bundle_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    gate_summary = verify_gate(
        baseline_dir=run_root / "reports" / "local_refinement_refactor" / "stage_00_baseline" / "regression_gpu_baseline",
        instrumented_dir=run_root / "reports" / "local_refinement_refactor" / "stage_01_instrumentation" / "regression_gpu_instrumented",
        comparison_dir=run_root / "reports" / "local_refinement_refactor" / "stage_01_instrumentation" / "baseline_vs_instrumented",
        local_box_csv=run_root
        / "reports"
        / "local_refinement_refactor"
        / "stage_01_instrumentation"
        / "regression_gpu_instrumented"
        / "baseline_local_box_timing.csv",
        expected_points=32,
        max_q_opt_diff=1.0e-12,
        max_delta_opt_diff=1.0e-12,
        max_deltaf_diff=1.0e-10,
    )
    write_gate_report(gate_summary, run_root / "reports" / "local_refinement_refactor" / "stage_01_instrumentation" / "baseline_vs_instrumented")

    existing: list[str] = []
    missing: list[str] = []
    for rel_path in DEFAULT_INCLUDE_PATHS:
        source_root = package_root if rel_path in {"README.md", "RUN_MANIFEST.json", "fixed_points/fixed_point_regression_points.csv"} else run_root
        if (source_root / rel_path).exists():
            existing.append(rel_path)
        else:
            missing.append(rel_path)

    manifest = {
        "archive": str(archive_path),
        "gate_status": gate_summary.get("status"),
        "include_paths_existing": existing,
        "include_paths_missing": missing,
    }
    (metadata_dir / "return_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (metadata_dir / "missing_paths.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    archive_path = archive_path if archive_path.is_absolute() else run_root / archive_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        added: set[Path] = set()
        for rel_path in DEFAULT_INCLUDE_PATHS:
            source_root = package_root if rel_path in {"README.md", "RUN_MANIFEST.json", "fixed_points/fixed_point_regression_points.csv"} else run_root
            for file_path in _iter_existing_paths(source_root, rel_path):
                resolved = file_path.resolve()
                if resolved in added or resolved == archive_path.resolve():
                    continue
                added.add(resolved)
                tar.add(file_path, arcname=file_path.relative_to(source_root))

    manifest["archive"] = str(archive_path)
    manifest["archive_size_bytes"] = archive_path.stat().st_size
    (metadata_dir / "return_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect local-refinement Stage 1 regression outputs for return.")
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None, help="Backward-compatible alias for --package-root and default --run-root.")
    parser.add_argument("--archive", type=Path, default=Path("local_refinement_refactor_stage1_regression_results.tar.gz"))
    args = parser.parse_args()
    package_root = args.root or args.package_root
    package_root = package_root if package_root.is_absolute() else ROOT / package_root
    run_root = args.run_root
    if args.root is not None and run_root is None:
        run_root = package_root
    run_root = run_root or _default_run_root(package_root)
    run_root = run_root if run_root.is_absolute() else package_root / run_root
    manifest = collect_outputs(package_root.resolve(), args.archive, run_root=run_root.resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
