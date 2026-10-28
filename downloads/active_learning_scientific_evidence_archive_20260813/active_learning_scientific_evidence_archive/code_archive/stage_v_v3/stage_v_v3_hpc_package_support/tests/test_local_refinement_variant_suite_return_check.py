from __future__ import annotations

import tarfile

from scripts.check_local_refinement_variant_suite_return import (
    RETURN_ARCHIVE_NAME,
    check_return_readiness,
)


def _add_text(tar: tarfile.TarFile, tmp_path, name: str, text: str = "x\n") -> None:
    path = tmp_path / name.replace("/", "_")
    path.write_text(text, encoding="utf-8")
    tar.add(path, arcname=name)


def test_variant_suite_return_check_accepts_structurally_complete_archive(tmp_path):
    run_root = tmp_path / "local_refinement_refactor_variant_suite_run"
    run_root.mkdir()
    archive = run_root / RETURN_ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as tar:
        _add_text(tar, tmp_path, "RUN_MANIFEST.json", "{}\n")
        _add_text(tar, tmp_path, "fixed_points/fixed_point_regression_points.csv", "point_id,kT,JA\n")
        _add_text(tar, tmp_path, "logs/variant_baseline.out", "cuda_runtime_probe=pass\n")
        _add_text(
            tar,
            tmp_path,
            "reports/local_refinement_refactor/variant_regression/baseline/baseline_pointwise.csv",
            "point_id,q_opt,delta_opt,DeltaF\n",
        )

    result = check_return_readiness(tmp_path)

    assert result["status"] == "ready_to_import"
    assert result["archive_check"]["status"] == "pass"
    assert result["log_check"]["status"] == "pass"
    assert result["failures"] == []
    assert RETURN_ARCHIVE_NAME in result["next_import_command"]


def test_variant_suite_return_check_reports_missing_archive_and_cuda_log_failure(tmp_path):
    logs = tmp_path / "local_refinement_refactor_variant_suite_run" / "logs"
    logs.mkdir(parents=True)
    (logs / "variant_baseline.out").write_text(
        "RuntimeError: The NVIDIA driver on your system is too old\n",
        encoding="utf-8",
    )

    result = check_return_readiness(tmp_path)

    assert result["status"] == "not_ready"
    assert result["archive_check"]["status"] == "missing"
    assert "missing local_refinement_refactor_variant_suite_results.tar.gz" in result["failures"]
    assert result["log_check"]["status"] == "fail"
    assert result["log_check"]["matches"][0]["kind"] == "old_nvidia_driver"
