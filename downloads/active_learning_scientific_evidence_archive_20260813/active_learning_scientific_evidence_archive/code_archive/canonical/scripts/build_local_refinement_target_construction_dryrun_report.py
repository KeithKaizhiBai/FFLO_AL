from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path(
    "local_refinement_refactor_hpc_upload_set/"
    "local_refinement_refactor_variant_suite/"
    "local_refinement_refactor_variant_suite_run"
)
DEFAULT_OUTPUT_DIR = Path("reports/local_refinement_target_construction_dryrun")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepend_dryrun_header(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8") if src.exists() else ""
    header = """# Local-Refinement Target-Construction Dry-Run Gate

This report is the Stage 0 dry-run gate required by
`project_history/plans_and_runbooks/TwoPhase_Optimization_Middle.md`.

It is report-only: it reads returned variant-array metadata and source-code
line evidence, but it does not call `evaluate_points`, does not enter local-box
scans, does not submit Slurm jobs, and does not modify production exact-oracle
logic.

"""
    dst.write_text(header + text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Stage 0 local-refinement target-construction dry-run gate report."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    helper = ROOT / "scripts" / "debug_local_refinement_target_construction.py"
    cmd = [
        sys.executable,
        str(helper),
        "--run-root",
        str(args.run_root),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    target_md = output_dir / "target_logic_audit.md"
    target_pdf = output_dir / "target_logic_audit.pdf"
    target_decision = output_dir / "decision_log.md"
    dryrun_md = output_dir / "dryrun_report.md"
    dryrun_pdf = output_dir / "dryrun_report.pdf"
    dryrun_decision = output_dir / "tables" / "dryrun_decision_log.md"

    prepend_dryrun_header(target_md, dryrun_md)
    copy_if_exists(target_pdf, dryrun_pdf)
    copy_if_exists(target_decision, dryrun_decision)

    table_dir = output_dir / "tables"
    copy_if_exists(table_dir / "target_construction_summary.csv", table_dir / "target_construction_by_variant.csv")
    copy_if_exists(table_dir / "refine_target_reason_counts.csv", table_dir / "selection_reason_counts.csv")

    print(f"dryrun_report_md={dryrun_md.resolve()}")
    print(f"dryrun_report_pdf={dryrun_pdf.resolve()}")
    print(f"tables={table_dir.resolve()}")
    print(f"figures={(output_dir / 'figures').resolve()}")


if __name__ == "__main__":
    main()
