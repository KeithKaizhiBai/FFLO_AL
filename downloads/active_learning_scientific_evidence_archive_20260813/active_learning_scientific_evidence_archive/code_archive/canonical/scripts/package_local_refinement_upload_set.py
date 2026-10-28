from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "hpc_packages"
UPLOAD_SET_NAME = "local_refinement_refactor_hpc_upload_set"
PACKAGE_ARCHIVES = [
    {
        "package_name": "local_refinement_refactor_stage01_instrumentation",
        "archive": "local_refinement_refactor_stage01_instrumentation.tar.gz",
        "role": "completed_reference",
        "upload_priority": "optional_reference",
        "hpc_command": "Already completed and locally imported; rerun only if Stage 1 instrumentation gate needs reproduction.",
        "return_archive": "local_refinement_refactor_stage1_regression_results.tar.gz",
        "local_return_check": "python scripts/import_local_refinement_stage1_results.py local_refinement_refactor_stage1_regression_results.tar.gz",
        "expected_run_root_suffix": "local_refinement_refactor_stage1_run",
        "expected_workflow_aliases": {
            "scripts/submit_local_refinement_fixed_point_regression.sh": "scripts/submit_stage1_regression_workflow.sh",
            "scripts/submit_local_refinement_instrumented_benchmark.sh": "scripts/submit_stage1_regression_workflow.sh",
        },
        "expected_support_scripts": [],
    },
    {
        "package_name": "local_refinement_refactor_variant_suite",
        "archive": "local_refinement_refactor_variant_suite.tar.gz",
        "role": "pending_gpu_validation",
        "upload_priority": "required_next",
        "hpc_command": "bash scripts/submit_local_refinement_fixed_point_regression.sh",
        "return_archive": "local_refinement_refactor_variant_suite_results.tar.gz",
        "local_return_check": "python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz",
        "expected_run_root_suffix": "local_refinement_refactor_variant_suite_run",
        "expected_workflow_aliases": {
            "scripts/submit_local_refinement_fixed_point_regression.sh": "scripts/submit_variant_suite_regression_workflow.sh",
            "scripts/submit_local_refinement_instrumented_benchmark.sh": "scripts/submit_variant_suite_regression_workflow.sh",
        },
        "expected_support_scripts": [
            "scripts/check_variant_suite_hpc_status.py",
        ],
    },
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_package_entry(spec: dict[str, str]) -> dict[str, Any]:
    archive = OUT_ROOT / spec["archive"]
    if not archive.exists():
        raise FileNotFoundError(f"Missing package archive: {archive}")
    sha256_sidecar = archive.with_suffix(archive.suffix + ".sha256")
    metadata_sidecar = archive.with_suffix(archive.suffix + ".metadata.json")
    if not sha256_sidecar.exists():
        raise FileNotFoundError(f"Missing package sha256 sidecar: {sha256_sidecar}")
    if not metadata_sidecar.exists():
        raise FileNotFoundError(f"Missing package metadata sidecar: {metadata_sidecar}")

    actual_sha256 = _sha256_file(archive)
    sidecar_parts = sha256_sidecar.read_text(encoding="utf-8").strip().split()
    sidecar_sha256 = sidecar_parts[0] if sidecar_parts else ""
    if sidecar_sha256 != actual_sha256:
        raise ValueError(f"SHA256 sidecar mismatch for {archive.name}: {sidecar_sha256} != {actual_sha256}")
    metadata = _read_json(metadata_sidecar)
    if metadata.get("archive_sha256") != actual_sha256:
        raise ValueError(f"Metadata SHA256 mismatch for {archive.name}")
    if metadata.get("archive_size_bytes") != archive.stat().st_size:
        raise ValueError(f"Metadata size mismatch for {archive.name}")

    return {
        "package_name": spec["package_name"],
        "archive": spec["archive"],
        "archive_relpath": f"archives/{archive.name}",
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": actual_sha256,
        "role": spec["role"],
        "upload_priority": spec["upload_priority"],
        "hpc_command": spec["hpc_command"],
        "return_archive": spec["return_archive"],
        "local_return_check": spec["local_return_check"],
        "expected_run_root_suffix": spec["expected_run_root_suffix"],
        "expected_workflow_aliases": spec["expected_workflow_aliases"],
        "expected_support_scripts": spec["expected_support_scripts"],
        "metadata": metadata,
    }


def _readme(manifest: dict[str, Any]) -> str:
    required = next(pkg for pkg in manifest["packages"] if pkg["upload_priority"] == "required_next")
    reference = [pkg for pkg in manifest["packages"] if pkg["upload_priority"] != "required_next"]
    reference_lines = "\n".join(f"- `{pkg['archive_relpath']}`: {pkg['role']}" for pkg in reference)
    return f"""# Local Refinement Refactor HPC Upload Set

This set groups the current local-refinement HPC packages and their checksums.

Before extracting or uploading nested archives, verify this handoff bundle:

```bash
python verify_upload_set.py
```

The package that should be uploaded and run next is:

```text
{required['archive_relpath']}
```

The Stage 1 instrumentation package is included only as a reproducibility
reference because its returned GPU/CUDA gate has already passed locally.

Reference packages:

{reference_lines}

## Run Next HPC Validation

After extracting this upload-set archive on the cluster, the least error-prone
entry point is:

```bash
bash run_required_variant_suite.sh
```

Equivalently, upload or copy `archives/{required['archive']}` to the cluster,
then:

```bash
tar -xzf {required['archive_relpath']}
cd {required['package_name']}
{required['hpc_command']}
```

The variant-suite scripts write runtime outputs under `RUN_ROOT`, defaulting
to the extracted package's own run directory when writable.

## Return Check

After the job completes, download:

```text
{required['return_archive']}
```

Then run locally:

```bash
{required['local_return_check']}
```
"""


def _run_order(manifest: dict[str, Any]) -> str:
    lines = [
        "# Local Refinement Refactor HPC Run Order",
        "",
        "## Verify Handoff Bundle",
        "",
        "```bash",
        "python verify_upload_set.py",
        "```",
        "",
        "## One-Command Required Run",
        "",
        "```bash",
        "bash run_required_variant_suite.sh",
        "```",
        "",
        "## Already Completed",
        "",
        "- Stage 1 instrumentation fixed-point regression: returned GPU/CUDA gate passed locally.",
        "",
        "## Required Next",
        "",
    ]
    for package in manifest["packages"]:
        if package["upload_priority"] == "required_next":
            lines.extend(
                [
                    f"- Package: `{package['archive_relpath']}`",
                    f"- Extracted directory: `{package['package_name']}`",
                    f"- Command: `{package['hpc_command']}`",
                    f"- Return archive: `{package['return_archive']}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Not Submitted In This Set",
            "",
            "- Stage 5 branch reuse production variant: not integrated yet.",
            "- Stage 6 adaptive-box production variant: diagnostic skeleton only.",
            "- GPU batching / Hamiltonian cache: planning only.",
            "",
        ]
    )
    return "\n".join(lines)


def _cluster_run_helper(manifest: dict[str, Any]) -> str:
    required = next(pkg for pkg in manifest["packages"] if pkg["upload_priority"] == "required_next")
    archive_relpath = required["archive_relpath"]
    package_name = required["package_name"]
    hpc_command = required["hpc_command"]
    return_archive = required["return_archive"]
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${{PYTHON_BIN:-python}}"

echo "[run_required_variant_suite] verifying upload set under $SCRIPT_DIR"
"$PYTHON_BIN" verify_upload_set.py --upload-root "$SCRIPT_DIR"

if [ ! -f "{archive_relpath}" ]; then
    echo "[run_required_variant_suite] missing archive: {archive_relpath}" >&2
    exit 2
fi

if [ ! -d "{package_name}" ]; then
    echo "[run_required_variant_suite] extracting {archive_relpath}"
    tar -xzf "{archive_relpath}"
else
    echo "[run_required_variant_suite] using existing {package_name} directory"
fi

cd "{package_name}"
echo "[run_required_variant_suite] submitting required variant-suite workflow"
{hpc_command}

cat <<'EOF'

[run_required_variant_suite] Submitted. If squeue has no visible jobs but the
return archive is unclear, run from the extracted variant-suite package:

python scripts/check_variant_suite_hpc_status.py --package-root . --run-root "${{RUN_ROOT:-{required['expected_run_root_suffix']}}}" --query-scheduler

Expected return archive:

{return_archive}
EOF
"""


def _return_checklist(manifest: dict[str, Any]) -> str:
    lines = [
        "# Return Checklist",
        "",
        "1. Confirm Slurm jobs finished without CUDA driver mismatch.",
        "2. Confirm `gpuh01` stayed excluded unless explicitly overridden.",
        "3. Download the return archive from `RUN_ROOT`.",
        "4. If a whole run directory was downloaded, locate/check the return archive locally.",
        "5. Run the local importer/checker.",
        "6. Inspect `variant_suite_gate_status.json` before interpreting performance gains.",
        "",
        "## Local Commands",
        "",
    ]
    for package in manifest["packages"]:
        if package["upload_priority"] == "required_next":
            lines.extend(
                [
                    "```bash",
                    "python scripts/check_local_refinement_variant_suite_return.py <downloaded-return-directory-or-archive>",
                    "```",
                    "",
                    "```bash",
                    package["local_return_check"],
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def _verify_upload_set_script() -> str:
    return r'''from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tar_text(tar: tarfile.TarFile, member_name: str) -> str:
    try:
        member = tar.getmember(member_name)
    except KeyError:
        return ""
    extracted = tar.extractfile(member)
    if extracted is None:
        return ""
    return extracted.read().decode("utf-8")


def verify_shell_output_policy(tar: tarfile.TarFile, names: set[str], root: str) -> dict:
    violations: list[dict[str, object]] = []
    script_prefix = f"{root}/scripts/"
    shell_scripts = sorted(name for name in names if name.startswith(script_prefix) and name.endswith(".sh"))
    for script_name in shell_scripts:
        script_text = read_tar_text(tar, script_name)
        for line_number, line in enumerate(script_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "logs" not in stripped and "reports" not in stripped:
                continue
            if "$RUN_ROOT" in stripped or "${RUN_ROOT}" in stripped:
                continue
            violations.append(
                {
                    "script": script_name,
                    "line": line_number,
                    "text": stripped,
                }
            )
    return {
        "status": "fail" if violations else "pass",
        "shell_script_count": len(shell_scripts),
        "violations": violations,
    }


def verify_gpu_slurm_policy(tar: tarfile.TarFile, names: set[str], root: str) -> dict:
    violations: list[dict[str, object]] = []
    script_prefix = f"{root}/scripts/"
    shell_scripts = sorted(name for name in names if name.startswith(script_prefix) and name.endswith(".sh"))
    gpu_scripts: list[str] = []
    required_tokens = {
        "exclude_gpuh01": "#SBATCH --exclude=gpuh01",
        "cuda_tensor_probe": "torch.empty(1, device=\"cuda\")",
        "cuda_runtime_probe_pass": "cuda_runtime_probe=pass",
    }
    for script_name in shell_scripts:
        script_text = read_tar_text(tar, script_name)
        is_gpu_script = "#SBATCH --gres=gpu:1" in script_text or "--device cuda" in script_text
        if not is_gpu_script:
            continue
        gpu_scripts.append(script_name)
        missing = [key for key, token in required_tokens.items() if token not in script_text]
        if missing:
            violations.append(
                {
                    "script": script_name,
                    "missing": missing,
                }
            )
    return {
        "status": "fail" if violations else "pass",
        "gpu_script_count": len(gpu_scripts),
        "violations": violations,
    }


def verify_nested_package_archive(archive: Path, package: dict) -> dict:
    failures: list[str] = []
    package_name = str(package.get("package_name", ""))
    root = package_name.rstrip("/")
    expected_aliases = package.get("expected_workflow_aliases", {})
    expected_support_scripts = package.get("expected_support_scripts", [])
    expected_run_root_suffix = str(package.get("expected_run_root_suffix", ""))
    hpc_command = str(package.get("hpc_command", ""))
    return_archive = str(package.get("return_archive", ""))

    checked: dict[str, object] = {
        "package_root": root,
        "expected_run_root_suffix": expected_run_root_suffix,
        "alias_count": len(expected_aliases) if isinstance(expected_aliases, dict) else 0,
    }

    try:
        with tarfile.open(archive, "r:gz") as tar:
            names = set(tar.getnames())
            required_paths = [
                f"{root}/README.md",
                f"{root}/RUN_MANIFEST.json",
            ]
            if isinstance(expected_aliases, dict):
                required_paths.extend(f"{root}/{alias}" for alias in expected_aliases)
                required_paths.extend(f"{root}/{target}" for target in expected_aliases.values())
            if isinstance(expected_support_scripts, list):
                required_paths.extend(f"{root}/{path}" for path in expected_support_scripts)

            missing = [path for path in required_paths if path not in names]
            checked["missing_required_paths"] = missing
            failures.extend(f"missing nested package path {path}" for path in missing)

            manifest_text = read_tar_text(tar, f"{root}/RUN_MANIFEST.json")
            readme_text = read_tar_text(tar, f"{root}/README.md")
            checked["has_run_manifest"] = bool(manifest_text)
            checked["has_readme"] = bool(readme_text)

            run_manifest = json.loads(manifest_text) if manifest_text else {}
            checked["run_manifest_package_name"] = run_manifest.get("package_name", "")
            checked["run_manifest_writable_run_root_env"] = run_manifest.get("writable_run_root_env", "")
            if run_manifest.get("package_name") != package_name:
                failures.append(f"nested RUN_MANIFEST package_name mismatch for {package_name}")
            if run_manifest.get("writable_run_root_env") != "RUN_ROOT":
                failures.append(f"nested RUN_MANIFEST missing writable RUN_ROOT env for {package_name}")

            manifest_aliases = run_manifest.get("workflow_aliases", {})
            checked["workflow_aliases"] = manifest_aliases
            if isinstance(expected_aliases, dict):
                for alias, target in expected_aliases.items():
                    if manifest_aliases.get(alias) != target:
                        failures.append(f"workflow alias mismatch for {package_name}: {alias} -> {manifest_aliases.get(alias)}")
                    alias_text = read_tar_text(tar, f"{root}/{alias}")
                    target_basename = Path(target).name
                    if target_basename not in alias_text or "exec bash" not in alias_text:
                        failures.append(f"alias script does not exec expected workflow for {package_name}: {alias}")

            if expected_run_root_suffix:
                if expected_run_root_suffix not in readme_text:
                    failures.append(f"README missing expected RUN_ROOT suffix for {package_name}: {expected_run_root_suffix}")
                workflow_text = ""
                if isinstance(expected_aliases, dict):
                    for target in sorted(set(expected_aliases.values())):
                        workflow_text += read_tar_text(tar, f"{root}/{target}")
                if expected_run_root_suffix not in workflow_text:
                    failures.append(f"workflow script missing expected RUN_ROOT suffix for {package_name}: {expected_run_root_suffix}")

            if hpc_command.startswith("bash ") and hpc_command not in readme_text:
                failures.append(f"README missing required HPC command for {package_name}: {hpc_command}")
            if return_archive and return_archive not in readme_text and return_archive not in json.dumps(run_manifest):
                failures.append(f"return archive not documented in nested package for {package_name}: {return_archive}")

            shell_output_policy = verify_shell_output_policy(tar, names, root)
            checked["shell_output_policy"] = shell_output_policy
            if shell_output_policy.get("status") != "pass":
                for violation in shell_output_policy.get("violations", []):
                    failures.append(
                        "shell script writes logs/reports without RUN_ROOT for "
                        f"{package_name}: {violation.get('script')}:{violation.get('line')}"
                    )

            gpu_slurm_policy = verify_gpu_slurm_policy(tar, names, root)
            checked["gpu_slurm_policy"] = gpu_slurm_policy
            if gpu_slurm_policy.get("status") != "pass":
                for violation in gpu_slurm_policy.get("violations", []):
                    failures.append(
                        "GPU Slurm script missing gpuh01 exclusion or CUDA runtime probe for "
                        f"{package_name}: {violation.get('script')} missing={violation.get('missing')}"
                    )
    except (tarfile.TarError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        failures.append(f"could not inspect nested package archive {archive.name}: {exc}")

    checked["status"] = "fail" if failures else "pass"
    checked["failures"] = failures
    return checked


def verify_upload_set(upload_root: Path) -> dict:
    failures: list[str] = []
    checked: dict[str, object] = {"upload_root": str(upload_root)}

    manifest_path = upload_root / "UPLOAD_MANIFEST.json"
    if not manifest_path.exists():
        return {"status": "fail", "checked": checked, "failures": [f"missing manifest: {manifest_path}"]}

    manifest = read_json(manifest_path)
    packages = manifest.get("packages", [])
    checked["package_count"] = len(packages)
    checked["required_next_package"] = manifest.get("required_next_package", "")

    top_level_check = {
        "readme_mentions_run_helper": False,
        "run_order_mentions_run_helper": False,
        "run_helper_present": False,
        "run_helper_verifies_upload_set": False,
        "run_helper_extracts_required_archive": False,
        "run_helper_submits_required_alias": False,
        "run_helper_has_destructive_delete": False,
    }
    readme_text = (upload_root / "README.md").read_text(encoding="utf-8") if (upload_root / "README.md").exists() else ""
    run_order_text = (upload_root / "RUN_ORDER.md").read_text(encoding="utf-8") if (upload_root / "RUN_ORDER.md").exists() else ""
    helper_path = upload_root / "run_required_variant_suite.sh"
    helper_text = helper_path.read_text(encoding="utf-8") if helper_path.exists() else ""
    top_level_check["readme_mentions_run_helper"] = "bash run_required_variant_suite.sh" in readme_text
    top_level_check["run_order_mentions_run_helper"] = "bash run_required_variant_suite.sh" in run_order_text
    top_level_check["run_helper_present"] = helper_path.exists()
    top_level_check["run_helper_verifies_upload_set"] = "verify_upload_set.py --upload-root" in helper_text
    top_level_check["run_helper_extracts_required_archive"] = (
        'tar -xzf "archives/local_refinement_refactor_variant_suite.tar.gz"' in helper_text
    )
    top_level_check["run_helper_submits_required_alias"] = (
        "bash scripts/submit_local_refinement_fixed_point_regression.sh" in helper_text
    )
    top_level_check["run_helper_has_destructive_delete"] = "rm -" in helper_text or "Remove-Item" in helper_text
    checked["top_level_handoff"] = top_level_check
    for key, value in top_level_check.items():
        if key == "run_helper_has_destructive_delete":
            if value:
                failures.append("run_required_variant_suite.sh contains a destructive delete command")
        elif not value:
            failures.append(f"top-level handoff check failed: {key}")

    required_next = [
        pkg.get("package_name", "")
        for pkg in packages
        if pkg.get("upload_priority") == "required_next"
    ]
    if required_next != [manifest.get("required_next_package")]:
        failures.append(
            "required_next_package does not match exactly one required_next package: "
            + repr(required_next)
        )

    package_checks = {}
    for package in packages:
        name = str(package.get("package_name", ""))
        archive_relpath = str(package.get("archive_relpath", ""))
        archive = upload_root / archive_relpath
        package_check: dict[str, object] = {"archive": str(archive)}

        if not archive.exists():
            failures.append(f"missing package archive: {archive_relpath}")
            package_checks[name] = package_check
            continue

        actual_sha256 = sha256_file(archive)
        actual_size = archive.stat().st_size
        package_check["archive_sha256"] = actual_sha256
        package_check["archive_size_bytes"] = actual_size

        sha_sidecar = archive.with_name(archive.name + ".sha256")
        metadata_sidecar = archive.with_name(archive.name + ".metadata.json")
        if not sha_sidecar.exists():
            failures.append(f"missing package sha256 sidecar: {sha_sidecar.relative_to(upload_root)}")
        else:
            sidecar_parts = sha_sidecar.read_text(encoding="utf-8").strip().split()
            sidecar_sha256 = sidecar_parts[0] if sidecar_parts else ""
            package_check["sha256_sidecar"] = sidecar_sha256
            if sidecar_sha256 != actual_sha256:
                failures.append(f"sha256 sidecar mismatch for {name}: {sidecar_sha256} != {actual_sha256}")

        if not metadata_sidecar.exists():
            failures.append(f"missing package metadata sidecar: {metadata_sidecar.relative_to(upload_root)}")
        else:
            metadata = read_json(metadata_sidecar)
            package_check["metadata_archive_sha256"] = metadata.get("archive_sha256", "")
            package_check["metadata_archive_size_bytes"] = metadata.get("archive_size_bytes", None)
            if metadata.get("archive_sha256") != actual_sha256:
                failures.append(f"metadata sha256 mismatch for {name}")
            if metadata.get("archive_size_bytes") != actual_size:
                failures.append(f"metadata size mismatch for {name}")

        if package.get("archive_sha256") != actual_sha256:
            failures.append(f"manifest sha256 mismatch for {name}")
        if package.get("archive_size_bytes") != actual_size:
            failures.append(f"manifest size mismatch for {name}")

        nested_check = verify_nested_package_archive(archive, package)
        package_check["nested_package"] = nested_check
        failures.extend(nested_check.get("failures", []))

        package_checks[name] = package_check

    checked["packages"] = package_checks
    return {"status": "fail" if failures else "pass", "checked": checked, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local-refinement upload-set archive sidecars.")
    parser.add_argument("--upload-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = verify_upload_set(args.upload_root.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def build_upload_set() -> Path:
    entries = [_load_package_entry(spec) for spec in PACKAGE_ARCHIVES]
    upload_root = OUT_ROOT / UPLOAD_SET_NAME
    if upload_root.exists():
        shutil.rmtree(upload_root)
    upload_root.mkdir(parents=True)

    for entry in entries:
        archive = OUT_ROOT / entry["archive"]
        _copy_file(archive, upload_root / "archives" / archive.name)
        _copy_file(archive.with_suffix(archive.suffix + ".sha256"), upload_root / "archives" / archive.with_suffix(archive.suffix + ".sha256").name)
        _copy_file(
            archive.with_suffix(archive.suffix + ".metadata.json"),
            upload_root / "archives" / archive.with_suffix(archive.suffix + ".metadata.json").name,
        )

    manifest = {
        "upload_set_name": UPLOAD_SET_NAME,
        "purpose": "Current local-refinement refactor HPC upload set and run handoff",
        "active_learning": "not_run",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "packages": entries,
        "required_next_package": "local_refinement_refactor_variant_suite",
        "stage1_status": "completed_reference",
        "stage2_3_4_status": "pending_gpu_validation",
        "stage5_status": "prototype_not_submitted",
        "stage6_status": "skeleton_not_submitted",
        "stage7_status": "package_handoff_ready",
    }
    _write_text(upload_root / "UPLOAD_MANIFEST.json", json.dumps(manifest, indent=2))
    _write_text(upload_root / "README.md", _readme(manifest))
    _write_text(upload_root / "RUN_ORDER.md", _run_order(manifest))
    _write_text(upload_root / "RETURN_CHECKLIST.md", _return_checklist(manifest))
    _write_text(upload_root / "run_required_variant_suite.sh", _cluster_run_helper(manifest))
    (upload_root / "run_required_variant_suite.sh").chmod(0o755)
    _write_text(upload_root / "verify_upload_set.py", _verify_upload_set_script())

    archive = OUT_ROOT / f"{UPLOAD_SET_NAME}.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(upload_root, arcname=UPLOAD_SET_NAME)
    archive_sha256 = _sha256_file(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{archive_sha256}  {archive.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    archive.with_suffix(archive.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "upload_set_name": UPLOAD_SET_NAME,
                "archive_name": archive.name,
                "archive_size_bytes": archive.stat().st_size,
                "archive_sha256": archive_sha256,
                "package_count": len(entries),
                "generated_utc": manifest["generated_utc"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return archive


def main() -> None:
    archive = build_upload_set()
    print(f"Wrote upload set: {archive}")


if __name__ == "__main__":
    main()
