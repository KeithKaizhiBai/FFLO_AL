from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CONFIG = {
    "run_id": "active_phase_topology_3d_t_ja_mu_from_scratch_v1",
    "output_root": "ML_Phase_StageIV_Topology3D",
    "mu_min": -0.5,
    "mu_max": 1.5,
    "guard_mu_min": -1.0,
    "guard_mu_max": 2.0,
    "initial_seed_size": 1024,
    "batch_size": 256,
    "max_acquisition_batches": 24,
}

REQUIRED_FILES = [
    "AGENTS.md",
    "MODEL_SPEC.md",
    "README_STAGEIV_3D_HPC.md",
    "RUN_MANIFEST.json",
    "configs/stageiv_3d_production.json",
    "eta_phase_diagram_cuda.py",
    "tfflo_1d_cuda.py",
    "ml_phase/__init__.py",
    "ml_phase/stageiv_3d.py",
    "ml_phase/exact_oracle.py",
    "ml_phase/topology_oracle.py",
    "ml_phase/hpc.py",
    "ml_phase/active_refine.py",
    "scripts/slurm_stageiv_exact_array.sh",
    "scripts/submit_stageiv_3d_full_loop.sh",
    "scripts/run_stageiv_3d_preflight.sh",
    "scripts/check_stageiv_3d_hpc_status.sh",
    "scripts/stageiv_3d_select.py",
    "scripts/stageiv_3d_preflight.py",
    "scripts/stageiv_3d_submit_check.py",
]

PY_COMPILE_FILES = [
    "eta_phase_diagram_cuda.py",
    "tfflo_1d_cuda.py",
    "ml_phase/stageiv_3d.py",
    "ml_phase/exact_oracle.py",
    "ml_phase/topology_oracle.py",
    "ml_phase/hpc.py",
    "ml_phase/active_refine.py",
    "scripts/stageiv_3d_select.py",
    "scripts/stageiv_3d_preflight.py",
    "scripts/stageiv_3d_submit_check.py",
    "scripts/stageiv_3d_hpc_status.py",
    "scripts/stageiv_3d_postrun_report.py",
    "scripts/stageiv_3d_convergence_audit.py",
    "scripts/stageiv_3d_hidden_slice_audit.py",
    "scripts/stageiv_3d_postrun_bundle.py",
]

IMPORT_MODULES = [
    "numpy",
    "pandas",
    "torch",
    "ml_phase.stageiv_3d",
    "ml_phase.topology_oracle",
    "ml_phase.hpc",
    "ml_phase.active_refine",
    "ml_phase.exact_oracle",
]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
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


def command_result(cmd: list[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return {
            "command": " ".join(cmd),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"command": " ".join(cmd), "returncode": None, "stdout": "", "stderr": str(exc)}


def add_row(rows: list[dict[str, Any]], check: str, status: str, details: str = "", path: str = "") -> None:
    rows.append({"check": check, "status": status, "path": path, "details": details})


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def check_required_files(root: Path, rows: list[dict[str, Any]], *, allow_missing_manifest: bool) -> None:
    for rel in REQUIRED_FILES:
        path = root / rel
        if rel == "RUN_MANIFEST.json" and allow_missing_manifest and not path.exists():
            add_row(rows, "required_file", "warning", "missing during package-build smoke only", rel)
            continue
        add_row(rows, "required_file", "pass" if path.exists() else "fail", "exists" if path.exists() else "missing", rel)


def check_config(root: Path, config_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    config = read_json(config_path)
    if not config:
        add_row(rows, "config_load", "fail", "could not parse JSON", str(config_path.relative_to(root)))
        return {}
    add_row(rows, "config_load", "pass", "parsed JSON", str(config_path.relative_to(root)))
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        ok = actual == expected
        add_row(rows, "config_value", "pass" if ok else "fail", f"{key}: actual={actual!r}; expected={expected!r}", str(config_path.relative_to(root)))
    if config.get("mu_reference") is not None:
        mu_ref = float(config["mu_reference"])
        ok = float(config.get("mu_min", 0.0)) <= mu_ref <= float(config.get("mu_max", 0.0))
        add_row(rows, "config_mu_reference", "pass" if ok else "fail", f"mu_reference={mu_ref}", str(config_path.relative_to(root)))
    return config


def check_shell_encoding(root: Path, rows: list[dict[str, Any]]) -> None:
    shell_files = sorted(root.glob("scripts/*.sh"))
    if not shell_files:
        add_row(rows, "shell_encoding", "fail", "no shell scripts found", "scripts/*.sh")
        return
    for path in shell_files:
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        problems: list[str] = []
        if data.startswith(b"\xef\xbb\xbf"):
            problems.append("utf8_bom")
        if b"\r\n" in data or b"\r" in data:
            problems.append("crlf_or_cr")
        try:
            data.decode("ascii")
        except UnicodeDecodeError:
            problems.append("non_ascii")
        add_row(rows, "shell_encoding", "pass" if not problems else "fail", ";".join(problems) if problems else "ASCII LF no BOM", rel)


def check_slurm_guards(root: Path, rows: list[dict[str, Any]]) -> None:
    slurm = root / "scripts" / "slurm_stageiv_exact_array.sh"
    submit = root / "scripts" / "submit_stageiv_3d_full_loop.sh"
    slurm_text = slurm.read_text(encoding="utf-8", errors="replace") if slurm.exists() else ""
    submit_text = submit.read_text(encoding="utf-8", errors="replace") if submit.exists() else ""
    add_row(rows, "slurm_exclude_directive", "pass" if "#SBATCH --exclude=gpuh01" in slurm_text else "fail", "SBATCH excludes gpuh01", slurm.relative_to(root).as_posix())
    add_row(rows, "slurm_runtime_hostname_guard", "pass" if "gpuh01" in slurm_text and "refusing to run" in slurm_text else "fail", "runtime gpuh01 guard present", slurm.relative_to(root).as_posix())
    add_row(rows, "submit_exclude_nodes_default", "pass" if "EXCLUDE_NODES=\"${EXCLUDE_NODES:-gpuh01}\"" in submit_text else "fail", "submit default excludes gpuh01", submit.relative_to(root).as_posix())
    add_row(rows, "submit_sbatch_exclude", "pass" if "sbatch --parsable --exclude=\"${EXCLUDE_NODES}\"" in submit_text else "fail", "sbatch uses exclude nodes", submit.relative_to(root).as_posix())


def check_bash_syntax(root: Path, rows: list[dict[str, Any]]) -> None:
    for path in sorted(root.glob("scripts/*.sh")):
        rel = path.relative_to(root).as_posix()
        result = command_result(["bash", "-n", rel], root)
        text = (str(result.get("stdout", "")) + "\n" + str(result.get("stderr", ""))).replace("\x00", "").strip()
        if result["returncode"] == 0:
            status = "pass"
            details = "bash -n passed"
        elif result["returncode"] is None or "Windows Subsystem for Linux has no installed distributions" in text:
            status = "warning"
            details = "bash unavailable on this host; run this check on the Linux login node"
        else:
            status = "fail"
            details = text[:500]
        add_row(rows, "bash_syntax", status, details, rel)


def check_py_compile(root: Path, python_bin: str, rows: list[dict[str, Any]]) -> None:
    existing = [rel for rel in PY_COMPILE_FILES if (root / rel).exists()]
    if not existing:
        add_row(rows, "py_compile", "fail", "no files available for py_compile")
        return
    result = command_result([python_bin, "-m", "py_compile", *existing], root)
    details = str(result.get("stderr") or result.get("stdout") or "py_compile passed")[:800]
    add_row(rows, "py_compile", "pass" if result["returncode"] == 0 else "fail", details, ",".join(existing))


def check_imports(root: Path, rows: list[dict[str, Any]]) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for module in IMPORT_MODULES:
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "")
            add_row(rows, "python_import", "pass", str(version), module)
        except Exception as exc:
            add_row(rows, "python_import", "fail", f"{type(exc).__name__}: {exc}", module)


def check_torch_cuda(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        details = {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_available": available,
            "device_count": int(torch.cuda.device_count()) if available else 0,
        }
        status = "pass" if available else "warning"
        add_row(rows, "torch_cuda_login_node_probe", status, json.dumps(details, sort_keys=True), "torch.cuda")
        return details
    except Exception as exc:
        add_row(rows, "torch_cuda_login_node_probe", "warning", f"{type(exc).__name__}: {exc}", "torch.cuda")
        return {"error": str(exc)}


def check_nvidia_smi(root: Path, rows: list[dict[str, Any]]) -> None:
    result = command_result(["nvidia-smi", "-L"], root)
    if result["returncode"] == 0:
        add_row(rows, "nvidia_smi_login_node_probe", "pass", str(result.get("stdout", ""))[:500], "nvidia-smi")
    else:
        add_row(rows, "nvidia_smi_login_node_probe", "warning", str(result.get("stderr", ""))[:500], "nvidia-smi")


def check_run_dir_collision(root: Path, config: dict[str, Any], rows: list[dict[str, Any]], allow_existing_run: bool) -> None:
    output_root = root / str(config.get("output_root", EXPECTED_CONFIG["output_root"]))
    run_id = str(config.get("run_id", EXPECTED_CONFIG["run_id"]))
    run_dir = output_root / "active_runs" / run_id
    start_iter = int(os.environ.get("START_ITER", "0"))
    if run_dir.exists() and start_iter == 0 and not allow_existing_run:
        add_row(rows, "run_dir_collision", "fail", "run directory already exists and START_ITER=0", str(run_dir.relative_to(root)))
    elif run_dir.exists():
        add_row(rows, "run_dir_collision", "warning", f"run directory exists; START_ITER={start_iter}", str(run_dir.relative_to(root)))
    else:
        add_row(rows, "run_dir_collision", "pass", "no existing run directory", str(run_dir.relative_to(root)))


def check_no_generated_payload(root: Path, rows: list[dict[str, Any]]) -> None:
    blocked = [
        root / "active_runs",
        root / "datasets",
        root / "figures",
        root / "reports" / "active_runs",
        root / "ml_phase" / "active_runs",
        root / "ml_phase" / "datasets",
        root / "ml_phase" / "figures",
        root / "ml_phase" / "reports",
    ]
    for path in blocked:
        status = "fail" if path.exists() else "pass"
        add_row(rows, "generated_payload_absent", status, "absent" if status == "pass" else "unexpected generated payload", str(path.relative_to(root)))


def write_markdown(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    failed = [row for row in rows if row["status"] == "fail"]
    warnings = [row for row in rows if row["status"] == "warning"]
    lines = [
        "# Stage IV 3D Submit-Ready Check",
        "",
        f"- status: `{summary['submit_ready_status']}`",
        f"- checked_at_utc: `{summary['checked_at_utc']}`",
        f"- package_root: `{summary['package_root']}`",
        f"- python_executable: `{summary['python_executable']}`",
        f"- failed_checks: `{len(failed)}`",
        f"- warning_checks: `{len(warnings)}`",
        "",
        "## Interpretation",
        "",
        "This is a read-only pre-submit package/environment check. It does not submit Slurm jobs, merge shards, append datasets, run exact calculations, or continue active learning.",
        "",
        "CUDA and `nvidia-smi` may be unavailable on a login node; those are warnings here because the exact Slurm array prints and checks CUDA inside the allocated GPU job.",
        "",
        "## Failed Checks",
        "",
    ]
    if failed:
        for row in failed:
            lines.append(f"- `{row['check']}` `{row.get('path', '')}`: {row.get('details', '')}")
    else:
        lines.append("- none")
    lines += ["", "## Warning Checks", ""]
    if warnings:
        for row in warnings:
            lines.append(f"- `{row['check']}` `{row.get('path', '')}`: {row.get('details', '')}")
    else:
        lines.append("- none")
    write_text(output_dir / "stageiv_3d_submit_check.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Stage IV 3D HPC submit-ready package check.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stageiv_3d_production.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("ML_Phase_StageIV_Topology3D/reports/stageiv_3d_submit_check"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--allow-existing-run", action="store_true")
    parser.add_argument("--allow-missing-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    rows: list[dict[str, Any]] = []

    check_required_files(root, rows, allow_missing_manifest=bool(args.allow_missing_manifest))
    config = check_config(root, config_path, rows)
    check_shell_encoding(root, rows)
    check_slurm_guards(root, rows)
    check_bash_syntax(root, rows)
    check_py_compile(root, str(args.python_bin), rows)
    check_imports(root, rows)
    torch_cuda = check_torch_cuda(rows)
    check_nvidia_smi(root, rows)
    check_run_dir_collision(root, config, rows, bool(args.allow_existing_run))
    check_no_generated_payload(root, rows)

    failed = [row for row in rows if row["status"] == "fail"]
    warnings = [row for row in rows if row["status"] == "warning"]
    summary = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(root),
        "python_executable": str(sys.executable),
        "python_bin_for_compile": str(args.python_bin),
        "config_path": str(config_path),
        "submit_ready_status": "pass" if not failed else "fail",
        "failed_check_count": len(failed),
        "warning_check_count": len(warnings),
        "torch_cuda_probe": torch_cuda,
        "run_id": config.get("run_id", ""),
        "output_root": config.get("output_root", ""),
        "root_sha256_manifest": sha256_file(root / "RUN_MANIFEST.json") if (root / "RUN_MANIFEST.json").exists() else "missing",
    }
    tables_dir = output_dir / "tables"
    write_json(output_dir / "stageiv_3d_submit_check.json", summary)
    write_csv(tables_dir / "stageiv_submit_check_items.csv", rows)
    write_markdown(output_dir, summary, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
