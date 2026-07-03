from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tarfile

from scripts import package_local_refinement_upload_set as upload_set


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_upload_set_builds_handoff_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_set, "OUT_ROOT", upload_set.ROOT / "hpc_packages")

    archive = upload_set.build_upload_set()
    upload_root = upload_set.OUT_ROOT / upload_set.UPLOAD_SET_NAME

    assert archive.exists()
    assert archive.with_suffix(archive.suffix + ".sha256").exists()
    assert archive.with_suffix(archive.suffix + ".metadata.json").exists()
    assert (upload_root / "README.md").exists()
    assert (upload_root / "RUN_ORDER.md").exists()
    assert (upload_root / "RETURN_CHECKLIST.md").exists()
    assert (upload_root / "run_required_variant_suite.sh").exists()
    assert (upload_root / "verify_upload_set.py").exists()
    run_helper = (upload_root / "run_required_variant_suite.sh").read_text(encoding="utf-8")
    assert "python" in run_helper
    assert "verify_upload_set.py --upload-root" in run_helper
    assert "tar -xzf \"archives/local_refinement_refactor_variant_suite.tar.gz\"" in run_helper
    assert "bash scripts/submit_local_refinement_fixed_point_regression.sh" in run_helper
    assert "check_variant_suite_hpc_status.py" in run_helper
    return_checklist = (upload_root / "RETURN_CHECKLIST.md").read_text(encoding="utf-8")
    assert "python scripts/check_local_refinement_variant_suite_return.py" in return_checklist
    assert "python scripts/import_local_refinement_variant_suite_results.py" in return_checklist

    manifest = json.loads((upload_root / "UPLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["upload_set_name"] == upload_set.UPLOAD_SET_NAME
    assert manifest["required_next_package"] == "local_refinement_refactor_variant_suite"
    assert manifest["stage1_status"] == "completed_reference"
    assert manifest["stage2_3_4_status"] == "pending_gpu_validation"

    packages = {pkg["package_name"]: pkg for pkg in manifest["packages"]}
    assert packages["local_refinement_refactor_stage01_instrumentation"]["upload_priority"] == "optional_reference"
    assert packages["local_refinement_refactor_variant_suite"]["upload_priority"] == "required_next"
    assert packages["local_refinement_refactor_variant_suite"]["local_return_check"].endswith(
        "local_refinement_refactor_variant_suite_results.tar.gz"
    )
    assert packages["local_refinement_refactor_stage01_instrumentation"]["expected_run_root_suffix"] == (
        "local_refinement_refactor_stage1_run"
    )
    assert packages["local_refinement_refactor_variant_suite"]["expected_run_root_suffix"] == (
        "local_refinement_refactor_variant_suite_run"
    )
    assert packages["local_refinement_refactor_stage01_instrumentation"]["expected_workflow_aliases"] == {
        "scripts/submit_local_refinement_fixed_point_regression.sh": "scripts/submit_stage1_regression_workflow.sh",
        "scripts/submit_local_refinement_instrumented_benchmark.sh": "scripts/submit_stage1_regression_workflow.sh",
    }
    assert packages["local_refinement_refactor_variant_suite"]["expected_workflow_aliases"] == {
        "scripts/submit_local_refinement_fixed_point_regression.sh": "scripts/submit_variant_suite_regression_workflow.sh",
        "scripts/submit_local_refinement_instrumented_benchmark.sh": "scripts/submit_variant_suite_regression_workflow.sh",
    }
    assert packages["local_refinement_refactor_variant_suite"]["expected_support_scripts"] == [
        "scripts/check_variant_suite_hpc_status.py",
    ]

    archive_sha256 = _sha256_file(archive)
    archive_size = archive.stat().st_size
    archive_sha_sidecar = archive.with_suffix(archive.suffix + ".sha256").read_text(encoding="utf-8").split()[0]
    archive_metadata = json.loads(archive.with_suffix(archive.suffix + ".metadata.json").read_text(encoding="utf-8"))
    assert archive_sha_sidecar == archive_sha256
    assert archive_metadata["archive_sha256"] == archive_sha256
    assert archive_metadata["archive_size_bytes"] == archive_size

    for package in packages.values():
        copied_archive = upload_root / package["archive_relpath"]
        copied_metadata = json.loads(
            (upload_root / "archives" / (package["archive"] + ".metadata.json")).read_text(encoding="utf-8")
        )
        copied_sha_sidecar = (upload_root / "archives" / (package["archive"] + ".sha256")).read_text(
            encoding="utf-8"
        ).split()[0]
        copied_sha256 = _sha256_file(copied_archive)
        assert copied_sha_sidecar == copied_sha256
        assert copied_metadata["archive_sha256"] == copied_sha256
        assert copied_metadata["archive_size_bytes"] == copied_archive.stat().st_size
        assert package["archive_sha256"] == copied_sha256
        assert package["archive_size_bytes"] == copied_archive.stat().st_size

    readme = (upload_root / "README.md").read_text(encoding="utf-8")
    assert "archives/local_refinement_refactor_variant_suite.tar.gz" in readme
    assert "python verify_upload_set.py" in readme
    assert "bash run_required_variant_suite.sh" in readme
    assert "bash scripts/submit_local_refinement_fixed_point_regression.sh" in readme
    assert "python scripts/import_local_refinement_variant_suite_results.py" in readme
    run_order = (upload_root / "RUN_ORDER.md").read_text(encoding="utf-8")
    assert "bash run_required_variant_suite.sh" in run_order

    verify = subprocess.run(
        [sys.executable, str(upload_root / "verify_upload_set.py"), "--upload-root", str(upload_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    verify_result = json.loads(verify.stdout)
    assert verify_result["status"] == "pass"
    assert verify_result["checked"]["required_next_package"] == "local_refinement_refactor_variant_suite"
    top_level_handoff = verify_result["checked"]["top_level_handoff"]
    assert top_level_handoff["readme_mentions_run_helper"] is True
    assert top_level_handoff["run_order_mentions_run_helper"] is True
    assert top_level_handoff["run_helper_present"] is True
    assert top_level_handoff["run_helper_verifies_upload_set"] is True
    assert top_level_handoff["run_helper_extracts_required_archive"] is True
    assert top_level_handoff["run_helper_submits_required_alias"] is True
    assert top_level_handoff["run_helper_has_destructive_delete"] is False
    nested_stage1 = verify_result["checked"]["packages"]["local_refinement_refactor_stage01_instrumentation"][
        "nested_package"
    ]
    nested_variant = verify_result["checked"]["packages"]["local_refinement_refactor_variant_suite"]["nested_package"]
    assert nested_stage1["status"] == "pass"
    assert nested_stage1["shell_output_policy"]["status"] == "pass"
    assert nested_stage1["shell_output_policy"]["violations"] == []
    assert nested_stage1["gpu_slurm_policy"]["status"] == "pass"
    assert nested_stage1["gpu_slurm_policy"]["gpu_script_count"] == 2
    assert nested_stage1["gpu_slurm_policy"]["violations"] == []
    assert nested_stage1["workflow_aliases"] == packages["local_refinement_refactor_stage01_instrumentation"][
        "expected_workflow_aliases"
    ]
    assert nested_variant["status"] == "pass"
    assert nested_variant["shell_output_policy"]["status"] == "pass"
    assert nested_variant["shell_output_policy"]["violations"] == []
    assert nested_variant["gpu_slurm_policy"]["status"] == "pass"
    assert nested_variant["gpu_slurm_policy"]["gpu_script_count"] == 5
    assert nested_variant["gpu_slurm_policy"]["violations"] == []
    assert nested_variant["workflow_aliases"] == packages["local_refinement_refactor_variant_suite"][
        "expected_workflow_aliases"
    ]
    assert nested_variant["run_manifest_writable_run_root_env"] == "RUN_ROOT"
    assert nested_variant["missing_required_paths"] == []

    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    prefix = upload_set.UPLOAD_SET_NAME
    assert f"{prefix}/UPLOAD_MANIFEST.json" in names
    assert f"{prefix}/run_required_variant_suite.sh" in names
    assert f"{prefix}/verify_upload_set.py" in names
    assert f"{prefix}/archives/local_refinement_refactor_stage01_instrumentation.tar.gz" in names
    assert f"{prefix}/archives/local_refinement_refactor_variant_suite.tar.gz" in names
