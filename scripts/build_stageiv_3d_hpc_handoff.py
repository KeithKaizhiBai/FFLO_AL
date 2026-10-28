from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "active_phase_topology_3d_t_ja_mu_from_scratch_v1"
PACKAGE_NAME = f"{RUN_ID}_hpc"
OUTPUT_ROOT = "ML_Phase_StageIV_Topology3D"
PACKAGE = ROOT / "hpc_packages" / f"{PACKAGE_NAME}.tar.gz"
PACKAGE_ROOT = ROOT / "hpc_packages" / PACKAGE_NAME
OUT_DIR = ROOT / "reports" / "stageiv_3d_hpc_handoff"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def commands() -> list[dict[str, str]]:
    return [
        {
            "step": "upload",
            "purpose": "Upload the package archive to the cluster.",
            "command": f"scp hpc_packages/{PACKAGE_NAME}.tar.gz sci_bfu@login02:~/bkz/Fu_FFLO/",
            "notes": "Run from local or use the user's preferred file-transfer method.",
        },
        {
            "step": "extract",
            "purpose": "Extract the self-contained package on the login node.",
            "command": f"cd ~/bkz/Fu_FFLO && tar -xzf {PACKAGE_NAME}.tar.gz && cd {PACKAGE_NAME}",
            "notes": "If tar reports an unexpected EOF, re-upload the archive; do not run a partial extraction.",
        },
        {
            "step": "verify_archive",
            "purpose": "Verify archive integrity after upload.",
            "command": f"sha256sum ../{PACKAGE_NAME}.tar.gz",
            "notes": f"Expected SHA256: {sha256_file(PACKAGE) if PACKAGE.exists() else 'missing'}",
        },
        {
            "step": "pre_submit_check",
            "purpose": "Run the read-only submit-ready checker.",
            "command": "export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python\nbash scripts/check_stageiv_3d_submit_ready.sh",
            "notes": "This does not submit jobs or run exact calculations.",
        },
        {
            "step": "submit",
            "purpose": "Submit the production Stage IV full loop.",
            "command": f"export CONFIRM_STAGEIV_FULL_LOOP=1\nnohup bash scripts/submit_stageiv_3d_full_loop.sh > {RUN_ID}.nohup.log 2>&1 &",
            "notes": "The Slurm scripts exclude gpuh01 and include a runtime hostname guard.",
        },
        {
            "step": "monitor",
            "purpose": "Monitor the active run and checkpoint state.",
            "command": f"tail -n 120 {RUN_ID}.nohup.log\nbash scripts/check_stageiv_3d_hpc_status.sh",
            "notes": "The status checker is read-only.",
        },
        {
            "step": "single_rank_recovery",
            "purpose": "Recover transient failed exact-array ranks without restarting from scratch.",
            "command": "bash scripts/inspect_stageiv_failed_task.sh <job_id>\nITER=<failed_iteration> FAILED_RANKS=<rank_list> bash scripts/recover_stageiv_failed_exact_iter.sh\nSTART_ITER=<next_iteration> bash scripts/resume_stageiv_3d_full_loop.sh",
            "notes": "Use only when selected-points partition exists and only specified ranks failed.",
        },
        {
            "step": "collect",
            "purpose": "Collect the result archive after the full loop completes.",
            "command": "bash scripts/collect_stageiv_3d_results.sh",
            "notes": f"Expected archive: {OUTPUT_ROOT}/stageiv_3d_topology_full_loop_results.tar.gz",
        },
        {
            "step": "return_check",
            "purpose": "Check downloaded/extracted results before interpretation.",
            "command": f"RETURN_PATH={OUTPUT_ROOT} bash scripts/check_stageiv_3d_return_bundle.sh",
            "notes": "This does not extract archives or run exact calculations.",
        },
        {
            "step": "postrun_bundle",
            "purpose": "Run report-only convergence and hidden fixed-mu validation audits.",
            "command": "REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_all_postrun_reports.sh",
            "notes": "REFERENCE_DATASET is validation-only and must not be merged into training.",
        },
    ]


def write_commands_csv(rows: list[dict[str, str]]) -> None:
    path = OUT_DIR / "tables" / "stageiv_handoff_commands.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "purpose", "command", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_status_figure(summary: dict[str, Any]) -> None:
    fig_path = OUT_DIR / "figures" / "stageiv_handoff_status.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        labels = ["package", "returned data", "post-run audit"]
        values = [
            1 if summary.get("package_validation_status") == "pass" else 0,
            1 if summary.get("production_run_returned") else 0,
            1 if summary.get("postrun_bundle_exists") else 0,
        ]
        colors = ["#2ca25f" if value else "#f0ad4e" for value in values]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Gate complete")
        ax.set_title("Stage IV Handoff Gate State")
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.04, "pass" if value else "pending", ha="center")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        write_text(fig_path.with_suffix(".txt"), f"Figure generation failed: {exc}\n")


def latex_escape(text: Any) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def write_reports(summary: dict[str, Any], command_rows: list[dict[str, str]]) -> None:
    md = [
        "# Stage IV 3D HPC Handoff",
        "",
        "This handoff is an operational guide for the current self-contained Stage IV 3D package. It does not submit jobs, run exact calculations, merge shards, append datasets, or modify historical data.",
        "",
        "## Current State",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- package: `{rel(PACKAGE)}`",
        f"- package_sha256: `{summary['package_sha256']}`",
        f"- package_size_bytes: `{summary['package_size_bytes']}`",
        f"- package_validation_status: `{summary['package_validation_status']}`",
        f"- production_run_returned: `{summary['production_run_returned']}`",
        f"- final_dataset_exists: `{summary['final_dataset_exists']}`",
        f"- postrun_bundle_exists: `{summary['postrun_bundle_exists']}`",
        "",
        "## Command Sequence",
        "",
    ]
    for row in command_rows:
        md.extend(
            [
                f"### {row['step']}",
                "",
                row["purpose"],
                "",
                "```bash",
                row["command"],
                "```",
                "",
                f"Notes: {row['notes']}",
                "",
            ]
        )
    md.extend(
        [
            "## Do-Not-Claim",
            "",
            "- Package validation is not Stage IV convergence.",
            "- Do not merge Stage III or Phase-II datasets into the cold-start Stage IV training run.",
            "- Do not mark missing surfaces or boundaries as zero shift.",
            "- Do not change production physics definitions, acquisition, exact oracle, StopController, or tolerances through this handoff.",
            "",
            "## Next Required External Action",
            "",
            "Upload the package, verify the SHA256 hash on the login node, run the submit-ready checker, then submit the production full loop.",
        ]
    )
    write_text(OUT_DIR / "stageiv_3d_hpc_handoff.md", "\n".join(md) + "\n")

    tex = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\begin{document}",
        r"\title{Stage IV 3D HPC Handoff}",
        r"\maketitle",
        r"\section*{Current State}",
        r"\begin{itemize}",
        rf"\item run\_id: \texttt{{{latex_escape(RUN_ID)}}}",
        rf"\item package: \texttt{{{latex_escape(rel(PACKAGE))}}}",
        rf"\item package\_sha256: \texttt{{{latex_escape(summary['package_sha256'])}}}",
        rf"\item package\_validation\_status: \texttt{{{latex_escape(summary['package_validation_status'])}}}",
        rf"\item production\_run\_returned: \texttt{{{latex_escape(summary['production_run_returned'])}}}",
        r"\end{itemize}",
        r"\section*{Command Sequence}",
        r"\scriptsize",
        r"\begin{longtable}{p{0.16\linewidth}p{0.29\linewidth}p{0.45\linewidth}}",
        r"\textbf{Step} & \textbf{Purpose} & \textbf{Notes}\\\hline",
    ]
    for row in command_rows:
        tex.append(f"{latex_escape(row['step'])} & {latex_escape(row['purpose'])} & {latex_escape(row['notes'])}\\\\")
    tex.extend(
        [
            r"\end{longtable}",
            r"\normalsize",
            r"\section*{Decision}",
            "The package is ready for HPC submission, but Stage IV scientific completion remains pending until returned production datasets and post-run audits are available.",
            r"\end{document}",
        ]
    )
    write_text(OUT_DIR / "stageiv_3d_hpc_handoff.tex", "\n".join(tex) + "\n")

    decision = [
        "# Stage IV HPC Handoff Decision Log",
        "",
        "Decision:",
        "",
        "```text",
        "handoff_status = ready_for_hpc_submit",
        "stageiv_scientific_status = package_ready_hpc_pending",
        "next_required_action = upload_verify_and_submit",
        "```",
        "",
        "The current archive is validated locally. Scientific convergence remains unproven until HPC output is returned and the post-run bundle passes.",
    ]
    write_text(OUT_DIR / "decision_log.md", "\n".join(decision) + "\n")


def main() -> None:
    goal = read_json(ROOT / "reports" / "stageiv_3d_goal_status" / "stageiv_3d_goal_status.json")
    manifest = read_json(PACKAGE_ROOT / "RUN_MANIFEST.json")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "package": rel(PACKAGE),
        "package_exists": PACKAGE.exists(),
        "package_size_bytes": PACKAGE.stat().st_size if PACKAGE.exists() else None,
        "package_sha256": sha256_file(PACKAGE) if PACKAGE.exists() else "",
        "package_validation_status": manifest.get("package_validation_status"),
        "production_run_returned": bool(goal.get("production_run_returned")),
        "final_dataset_exists": bool(goal.get("final_dataset_exists")),
        "postrun_bundle_exists": bool(goal.get("postrun_bundle_exists")),
        "handoff_status": "ready_for_hpc_submit" if manifest.get("package_validation_status") == "pass" else "package_not_ready",
    }
    rows = commands()
    write_commands_csv(rows)
    write_json(OUT_DIR / "stageiv_3d_hpc_handoff.json", summary)
    write_status_figure(summary)
    write_reports(summary, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
