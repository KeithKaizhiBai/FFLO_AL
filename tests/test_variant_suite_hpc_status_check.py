from __future__ import annotations

from scripts.check_variant_suite_hpc_status import RESULT_ARCHIVE, check_hpc_status


def test_variant_suite_hpc_status_ready_when_archive_exists_and_logs_clean(tmp_path):
    package_root = tmp_path / "local_refinement_refactor_variant_suite"
    run_root = package_root / "local_refinement_refactor_variant_suite_run"
    logs = run_root / "logs"
    logs.mkdir(parents=True)
    (logs / "variant_baseline.jobid").write_text("12345\n", encoding="utf-8")
    (logs / "variant_baseline_env_snapshot.txt").write_text("cuda_runtime_probe=pass\n", encoding="utf-8")
    (run_root / RESULT_ARCHIVE).write_text("synthetic archive placeholder\n", encoding="utf-8")

    result = check_hpc_status(package_root)

    assert result["status"] == "ready_to_return"
    assert result["return_archive"]["exists"] is True
    assert result["jobids"]["variant_baseline"] == "12345"
    assert result["log_scan"]["status"] == "pass"
    assert result["failures"] == []


def test_variant_suite_hpc_status_flags_missing_archive_and_old_driver_log(tmp_path):
    package_root = tmp_path / "local_refinement_refactor_variant_suite"
    run_root = package_root / "local_refinement_refactor_variant_suite_run"
    logs = run_root / "logs"
    logs.mkdir(parents=True)
    (logs / "variant_baseline.jobid").write_text("67890\n", encoding="utf-8")
    (logs / "variant_baseline_env_snapshot.txt").write_text(
        "RuntimeError: The NVIDIA driver on your system is too old\n",
        encoding="utf-8",
    )

    result = check_hpc_status(package_root)

    assert result["status"] == "failed_or_needs_log_review"
    assert result["return_archive"]["exists"] is False
    assert "missing return archive" in result["failures"][0]
    assert result["log_scan"]["status"] == "fail"
    assert result["log_scan"]["matches"][0]["kind"] == "old_nvidia_driver"
