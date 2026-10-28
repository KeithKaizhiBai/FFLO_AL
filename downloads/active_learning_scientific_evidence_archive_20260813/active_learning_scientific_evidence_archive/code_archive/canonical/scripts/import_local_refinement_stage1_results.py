from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_local_refinement_stage1_gate import verify_gate, write_gate_report


def _default_import_root(archive: Path) -> Path:
    return archive.parent / "imported_results"


def _safe_members(tar: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    dest = destination.resolve()
    members: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"Unsafe archive path: {member.name}")
        members.append(member)
    return members


def _default_extract_dir(archive: Path, import_root: Path) -> Path:
    stem = archive.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return import_root / stem


def import_and_verify(
    archive: Path,
    import_root: Path,
    extract_dir: Path | None = None,
    expected_points: int = 32,
    max_q_opt_diff: float = 1.0e-12,
    max_delta_opt_diff: float = 1.0e-12,
    max_deltaf_diff: float = 1.0e-10,
) -> dict[str, Any]:
    if not archive.exists():
        raise FileNotFoundError(f"Missing return archive: {archive}")
    target_dir = extract_dir if extract_dir is not None else _default_extract_dir(archive, import_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(target_dir, members=_safe_members(tar, target_dir))

    comparison_dir = target_dir / "reports" / "local_refinement_refactor" / "stage_01_instrumentation" / "baseline_vs_instrumented"
    summary = verify_gate(
        baseline_dir=target_dir / "reports" / "local_refinement_refactor" / "stage_00_baseline" / "regression_gpu_baseline",
        instrumented_dir=target_dir
        / "reports"
        / "local_refinement_refactor"
        / "stage_01_instrumentation"
        / "regression_gpu_instrumented",
        comparison_dir=comparison_dir,
        local_box_csv=target_dir
        / "reports"
        / "local_refinement_refactor"
        / "stage_01_instrumentation"
        / "regression_gpu_instrumented"
        / "baseline_local_box_timing.csv",
        expected_points=expected_points,
        max_q_opt_diff=max_q_opt_diff,
        max_delta_opt_diff=max_delta_opt_diff,
        max_deltaf_diff=max_deltaf_diff,
    )
    write_gate_report(summary, comparison_dir)

    manifest = {
        "archive": str(archive),
        "extract_dir": str(target_dir),
        "gate_status": summary["status"],
        "gate_status_json": str(comparison_dir / "stage1_gate_status.json"),
        "gate_status_md": str(comparison_dir / "stage1_gate_status.md"),
        "missing_files": summary.get("missing_files", []),
        "failures": summary.get("failures", []),
    }
    import_root.mkdir(parents=True, exist_ok=True)
    (target_dir / "import_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (import_root / "latest_import_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and verify a returned local-refinement Stage 1 HPC result bundle.")
    parser.add_argument("archive", type=Path, help="Returned local_refinement_refactor_stage1_regression_results.tar.gz")
    parser.add_argument(
        "--import-root",
        type=Path,
        default=None,
        help="Import destination. Defaults to imported_results/ next to the returned archive.",
    )
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--expected-points", type=int, default=32)
    parser.add_argument("--max-q-opt-diff", type=float, default=1.0e-12)
    parser.add_argument("--max-delta-opt-diff", type=float, default=1.0e-12)
    parser.add_argument("--max-deltaf-diff", type=float, default=1.0e-10)
    args = parser.parse_args()
    import_root = args.import_root if args.import_root is not None else _default_import_root(args.archive)

    manifest = import_and_verify(
        args.archive,
        import_root,
        args.extract_dir,
        expected_points=args.expected_points,
        max_q_opt_diff=args.max_q_opt_diff,
        max_delta_opt_diff=args.max_delta_opt_diff,
        max_deltaf_diff=args.max_deltaf_diff,
    )
    print(json.dumps(manifest, indent=2))
    if manifest["gate_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
