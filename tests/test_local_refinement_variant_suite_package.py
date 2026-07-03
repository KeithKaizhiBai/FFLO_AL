from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path

from scripts.compare_local_refinement_variants import compare
from scripts.aggregate_local_refinement_variant_array_suite import aggregate_suite
from scripts import import_local_refinement_variant_suite_results as import_script
from scripts.import_local_refinement_variant_suite_results import import_and_verify, verify_variant_suite
from scripts import package_local_refinement_variant_suite_hpc as package_script


def test_variant_suite_package_builds_expected_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(package_script, "OUT_ROOT", tmp_path / "hpc_packages")

    archive = package_script.build_package()
    package_root = tmp_path / "hpc_packages" / package_script.PACKAGE_NAME

    assert archive.exists()
    assert package_root.exists()
    assert archive.with_suffix(archive.suffix + ".sha256").exists()
    assert archive.with_suffix(archive.suffix + ".metadata.json").exists()

    manifest = json.loads((package_root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["package_name"] == package_script.PACKAGE_NAME
    assert manifest["active_learning"] == "not_run"
    assert manifest["variants"] == package_script.RUNNABLE_VARIANTS
    assert manifest["scheduler"] == "slurm_array"
    assert manifest["array_dimension"] == "variant x fixed_point_id"
    assert manifest["expected_tasks"] == manifest["expected_fixed_points"] * len(package_script.RUNNABLE_VARIANTS)
    assert manifest["task_matrix"] == "config/task_matrix.csv"
    assert manifest["variant_configs"]["cluster_optional_k2"]["max_optional_refined_basins"] == 2
    assert manifest["variant_configs"]["cluster_energy_window"]["energy_window_pruning_enabled"] is True
    assert "code_snapshot/" in manifest["package_layout"]
    assert manifest["workflow_aliases"]["scripts/submit_local_refinement_fixed_point_regression.sh"] == (
        "scripts/submit_variant_suite_regression_workflow.sh"
    )
    assert manifest["workflow_aliases"]["scripts/submit_local_refinement_instrumented_benchmark.sh"] == (
        "scripts/submit_variant_suite_regression_workflow.sh"
    )
    assert manifest["hpc_status_check"] == "scripts/check_variant_suite_hpc_status.py"
    assert "Stage 5 branch reuse is not integrated into production loop." in manifest["notes"]

    workflow = (package_root / "scripts" / "submit_variant_suite_regression_workflow.sh").read_text(encoding="utf-8")
    assert "local_refinement_refactor_variant_suite_run" in workflow
    assert "preflight_local_refinement_variant_suite_hpc.py" in workflow
    assert '--output-json "${RUN_ROOT}/reports/local_refinement_refactor/variant_regression/preflight.json"' in workflow
    assert '--array="0-${last_task}%${MAX_CONCURRENT}"' in workflow
    assert "--dependency=afterany:" in workflow
    assert "--dependency=afterok:" not in workflow
    assert "check_variant_suite_hpc_status.py" in workflow

    fixed_point_alias = (
        package_root / "scripts" / "submit_local_refinement_fixed_point_regression.sh"
    ).read_text(encoding="utf-8")
    benchmark_alias = (
        package_root / "scripts" / "submit_local_refinement_instrumented_benchmark.sh"
    ).read_text(encoding="utf-8")
    assert 'exec bash "${SCRIPT_DIR}/submit_variant_suite_regression_workflow.sh" "$@"' in fixed_point_alias
    assert 'exec bash "${SCRIPT_DIR}/submit_variant_suite_regression_workflow.sh" "$@"' in benchmark_alias

    slurm = (package_root / "scripts" / "slurm_variant_point_array.sh").read_text(encoding="utf-8")
    assert "#SBATCH --exclude=gpuh01" in slurm
    assert "SLURM_ARRAY_TASK_ID" in slurm
    assert "run_local_refinement_variant_point.py" in slurm
    assert "--enable-local-box-instrumentation" in slurm
    assert "torch.empty(1, device=\"cuda\")" in slurm

    collect = (package_root / "scripts" / "collect_variant_suite_outputs.sh").read_text(encoding="utf-8")
    assert "local_refinement_refactor_variant_suite_results.tar.gz" in collect
    assert 'RETURN_METADATA_DIR="${RUN_ROOT}/reports/local_refinement_refactor/variant_regression/return_bundle_metadata"' in collect
    assert 'mkdir -p "${RETURN_METADATA_DIR}"' in collect
    assert '} > reports/local_refinement_refactor/variant_regression/return_bundle_metadata/return_manifest.txt' not in collect
    assert '-C "${PACKAGE_ROOT}" README.md RUN_MANIFEST.json config fixed_points/fixed_point_regression_points.csv' in collect
    assert "-C \"${RUN_ROOT}\" logs reports/local_refinement_refactor/variant_regression" in collect

    aggregator = (package_root / "scripts" / "aggregate_local_refinement_variant_array_suite.py").read_text(encoding="utf-8")
    assert "def aggregate_suite" in aggregator

    performance_collect = (package_root / "scripts" / "collect_local_refinement_performance_report.sh").read_text(
        encoding="utf-8"
    )
    assert "build_local_refinement_performance_report.py" in performance_collect
    assert "performance_report" in performance_collect

    postprocess = (package_root / "scripts" / "slurm_variant_suite_postprocess.sh").read_text(encoding="utf-8")
    assert "export PYTHON_BIN" in postprocess
    assert "aggregate_local_refinement_variant_array_suite.py" in postprocess
    assert "bash scripts/collect_variant_suite_outputs.sh" in postprocess

    readme = (package_root / "README.md").read_text(encoding="utf-8")
    assert "point-wise Slurm array" in readme
    assert "afterany" in readme
    assert "bash scripts/submit_local_refinement_fixed_point_regression.sh" in readme
    assert "bash scripts/submit_local_refinement_instrumented_benchmark.sh" in readme
    assert "python scripts/check_variant_suite_hpc_status.py" in readme

    status_checker = (package_root / "scripts" / "check_variant_suite_hpc_status.py").read_text(encoding="utf-8")
    assert "def check_hpc_status" in status_checker

    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert f"{package_script.PACKAGE_NAME}/RUN_MANIFEST.json" in names
    assert f"{package_script.PACKAGE_NAME}/config/variants.json" in names
    assert f"{package_script.PACKAGE_NAME}/config/task_matrix.csv" in names
    assert f"{package_script.PACKAGE_NAME}/code_snapshot/ml_phase/exact_oracle.py" in names
    assert f"{package_script.PACKAGE_NAME}/code_snapshot/scripts/run_local_refinement_fixed_point_regression.py" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/slurm_variant_point_array.sh" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/run_local_refinement_variant_point.py" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/aggregate_local_refinement_variant_array_suite.py" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/build_local_refinement_performance_report.py" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/collect_local_refinement_performance_report.sh" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/submit_local_refinement_fixed_point_regression.sh" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/submit_local_refinement_instrumented_benchmark.sh" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/import_local_refinement_variant_suite_results.py" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/check_variant_suite_hpc_status.py" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/audit_local_refinement_stage_reports.py" in names
    assert f"{package_script.PACKAGE_NAME}/scripts/audit_local_refinement_runbook_tests.py" in names
    assert f"{package_script.PACKAGE_NAME}/fixed_points/fixed_point_regression_points.csv" in names


def _write_pointwise(path: Path, variant: str, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rows.append(
            {
                "point_id": i,
                "kT": 0.1 + 0.1 * i,
                "JA": 0.2 + 0.1 * i,
                "phase_candidate": 0 if i == 0 else 1,
                "q_opt": 0.0 if i == 0 else 0.1 * i,
                "delta_opt": 0.0 if i == 0 else 0.2,
                "DeltaF": 1.0e-5 if i == 0 else -1.0e-3,
                "trusted_exact": 1,
                "training_eligible_exact": 1,
                "q_unresolved": 0,
                "delta_unresolved": 0,
                "rerun_required": 0,
                "source_category": "synthetic",
                "source_run": "test",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_variant_manifest(path: Path, variant: str, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mode": "exact",
                "variant_name": variant,
                "n_points": n,
                "variant_config": package_script.resolve_variant_config(variant),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_synthetic_variant_suite(root: Path, n: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "RUN_MANIFEST.json").write_text(
        json.dumps(
            {
                "package_name": package_script.PACKAGE_NAME,
                "active_learning": "not_run",
                "expected_fixed_points": n,
                "variants": package_script.RUNNABLE_VARIANTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fixed_points = root / "fixed_points" / "fixed_point_regression_points.csv"
    fixed_points.parent.mkdir(parents=True, exist_ok=True)
    with fixed_points.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["point_id", "category", "source_run", "kT", "JA"])
        writer.writeheader()
        for i in range(n):
            writer.writerow({"point_id": i, "category": "synthetic", "source_run": "test", "kT": 0.1, "JA": 0.2})

    for variant in package_script.RUNNABLE_VARIANTS:
        variant_root = root / "reports" / "local_refinement_refactor" / "variant_regression" / variant
        _write_pointwise(variant_root / f"{variant}_pointwise.csv", variant, n=n)
        _write_variant_manifest(variant_root / "regression_manifest.json", variant, n=n)

    baseline_csv = root / "reports" / "local_refinement_refactor" / "variant_regression" / "baseline" / "baseline_pointwise.csv"
    for variant in package_script.COMPARISON_VARIANTS:
        candidate_csv = root / "reports" / "local_refinement_refactor" / "variant_regression" / variant / f"{variant}_pointwise.csv"
        output_dir = root / "reports" / "local_refinement_refactor" / "variant_regression" / "comparisons" / f"baseline_vs_{variant}"
        compare(baseline_csv, candidate_csv, output_dir)


def _write_synthetic_array_inputs(root: Path, n: int = 3) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "RUN_MANIFEST.json").write_text(
        json.dumps(
            {
                "package_name": package_script.PACKAGE_NAME,
                "active_learning": "not_run",
                "expected_fixed_points": n,
                "expected_tasks": n * len(package_script.RUNNABLE_VARIANTS),
                "variants": package_script.RUNNABLE_VARIANTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fixed_points = root / "fixed_points" / "fixed_point_regression_points.csv"
    fixed_points.parent.mkdir(parents=True, exist_ok=True)
    task_matrix = root / "config" / "task_matrix.csv"
    task_matrix.parent.mkdir(parents=True, exist_ok=True)

    fixed_rows = []
    task_rows = []
    task_id = 0
    for point_id in range(n):
        fixed_rows.append(
            {
                "point_id": point_id,
                "category": "synthetic",
                "source_run": "test",
                "source_iter": "iter000",
                "source_index": point_id,
                "kT": 0.1 + 0.01 * point_id,
                "JA": 0.2 + 0.02 * point_id,
            }
        )
    with fixed_points.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fixed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fixed_rows)

    output_root = root / "reports" / "local_refinement_refactor" / "variant_regression"
    for variant in package_script.RUNNABLE_VARIANTS:
        for point in fixed_rows:
            row = {"task_id": task_id, "variant": variant, **point}
            task_rows.append(row)
            point_dir = output_root / "point_tasks" / variant
            point_dir.mkdir(parents=True, exist_ok=True)
            point_id = int(point["point_id"])
            point_csv = point_dir / f"point_{point_id:03d}.csv"
            point_status = point_dir / f"point_{point_id:03d}.json"
            point_row = {
                "task_id": task_id,
                "point_id": point_id,
                "kT": point["kT"],
                "JA": point["JA"],
                "phase_candidate": 0 if point_id == 0 else 1,
                "q_opt": 0.0 if point_id == 0 else 0.1 * point_id,
                "delta_opt": 0.0 if point_id == 0 else 0.2,
                "DeltaF": 1.0e-5 if point_id == 0 else -1.0e-3,
                "trusted_exact": 1,
                "training_eligible_exact": 1,
                "q_unresolved": 0,
                "delta_unresolved": 0,
                "rerun_required": 0,
                "local_minima_detected_count": 1,
                "clustered_basin_count": 1,
                "selected_refine_target_count": 1,
                "basin_clustering_enabled": 1 if variant != "baseline" else 0,
                "basin_clustering_merged_count": 0,
                "energy_window_pruning_enabled": 1 if variant == "cluster_energy_window" else 0,
                "energy_window_pruned_count": 0,
                "local_boxes_refined_count": 1,
                "local_refinement_reused_count": 0,
                "point_total_runtime_sec": 1.0,
                "local_refinement_runtime_sec": 0.5,
                "source_category": "synthetic",
                "source_run": "test",
            }
            with point_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(point_row.keys()))
                writer.writeheader()
                writer.writerow(point_row)
            point_status.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "task_id": task_id,
                        "variant_name": variant,
                        "point_id": point_id,
                        "wall_runtime_sec": 1.0,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            task_id += 1

    with task_matrix.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
        writer.writeheader()
        writer.writerows(task_rows)
    return task_matrix


def test_variant_array_aggregator_builds_legacy_suite_layout(tmp_path):
    task_matrix = _write_synthetic_array_inputs(tmp_path, n=3)
    output_root = tmp_path / "reports" / "local_refinement_refactor" / "variant_regression"

    summary = aggregate_suite(
        package_root=tmp_path,
        run_root=tmp_path,
        task_matrix=task_matrix,
        output_root=output_root,
        tolerances={
            "max_q_opt_abs_diff": 1.0e-10,
            "max_delta_opt_abs_diff": 1.0e-10,
            "max_deltaf_abs_diff": 1.0e-8,
        },
    )

    assert summary["status"] == "pass"
    assert summary["successful_tasks"] == 3 * len(package_script.RUNNABLE_VARIANTS)
    assert (output_root / "summary" / "array_suite_status.json").exists()
    assert (output_root / "summary" / "task_status.csv").exists()
    assert (output_root / "summary" / "equivalence_matrix.csv").exists()
    assert (output_root / "performance_report" / "performance_summary.json").exists()
    assert (output_root / "baseline" / "baseline_pointwise.csv").exists()
    assert (output_root / "cluster_optional_k2" / "cluster_optional_k2_pointwise.csv").exists()

    gate = verify_variant_suite(tmp_path, expected_points=3)
    assert gate["status"] == "pass"


def test_variant_suite_result_verifier_passes_synthetic_return_bundle(tmp_path):
    source_root = tmp_path / "source_return"
    _write_synthetic_variant_suite(source_root, n=3)

    summary = verify_variant_suite(source_root, expected_points=3)
    assert summary["status"] == "pass"
    assert summary["comparison_checks"]["cluster_energy_window"]["flag_mismatch_count"] == 0

    archive = tmp_path / "local_refinement_refactor_variant_suite_results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in source_root.rglob("*"):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(source_root))

    manifest = import_and_verify(archive, import_root=tmp_path / "imported", expected_points=3)
    assert manifest["import_status"] == "pass"
    assert manifest["gate_status"] == "pass"
    assert manifest["performance_report_status"] == "pass"
    assert Path(manifest["gate_status_json"]).exists()
    assert Path(manifest["performance_summary_json"]).exists()
    assert Path(manifest["performance_report_md"]).exists()
    assert Path(manifest["runtime_summary_csv"]).exists()
    assert Path(manifest["local_box_summary_csv"]).exists()
    assert (tmp_path / "imported" / "latest_variant_suite_import_manifest.json").exists()


def test_variant_suite_result_verifier_fails_missing_variant_output(tmp_path):
    source_root = tmp_path / "source_return"
    _write_synthetic_variant_suite(source_root, n=3)
    (source_root / "reports" / "local_refinement_refactor" / "variant_regression" / "cluster_optional_k2" / "cluster_optional_k2_pointwise.csv").unlink()

    summary = verify_variant_suite(source_root, expected_points=3)

    assert summary["status"] == "fail"
    assert "cluster_optional_k2_pointwise" in summary["missing_files"]


def test_variant_suite_import_fails_when_performance_report_fails(tmp_path, monkeypatch):
    source_root = tmp_path / "source_return"
    _write_synthetic_variant_suite(source_root, n=3)
    archive = tmp_path / "local_refinement_refactor_variant_suite_results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in source_root.rglob("*"):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(source_root))

    def fail_performance_report(*args, **kwargs):
        raise RuntimeError("synthetic performance failure")

    monkeypatch.setattr(import_script, "build_performance_report", fail_performance_report)

    manifest = import_script.import_and_verify(archive, import_root=tmp_path / "imported", expected_points=3)

    assert manifest["gate_status"] == "pass"
    assert manifest["import_status"] == "fail"
    assert manifest["performance_report_status"] == "not_built"
    assert any("synthetic performance failure" in failure for failure in manifest["failures"])
