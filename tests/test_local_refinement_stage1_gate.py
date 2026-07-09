from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from scripts.collect_local_refinement_stage1_outputs import collect_outputs
from scripts.compare_local_refinement_variants import compare
from scripts.import_local_refinement_stage1_results import import_and_verify
from scripts.package_local_refinement_refactor_hpc import (
    _readme,
    _shell_roots_preamble,
    _workflow_alias_script,
    _workflow_submit_script,
)
from scripts.preflight_local_refinement_stage1_hpc import run_preflight
from scripts.validate_local_refinement_hpc_package import validate_package
from scripts.verify_local_refinement_stage1_gate import verify_gate, write_gate_report


ROOT = Path(__file__).resolve().parents[1]


def _write_pointwise(path: Path, deltaf_offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "point_id": 0,
            "kT": 0.1,
            "JA": 0.2,
            "phase_candidate": 0,
            "q_opt": 0.0,
            "delta_opt": 0.0,
            "DeltaF": 1e-5 + deltaf_offset,
            "trusted_exact": 1,
            "training_eligible_exact": 1,
            "q_unresolved": 0,
            "delta_unresolved": 0,
            "rerun_required": 0,
        },
        {
            "point_id": 1,
            "kT": 0.3,
            "JA": 0.4,
            "phase_candidate": 2,
            "q_opt": 0.25,
            "delta_opt": 0.2,
            "DeltaF": -1e-3 + deltaf_offset,
            "trusted_exact": 1,
            "training_eligible_exact": 1,
            "q_unresolved": 0,
            "delta_unresolved": 0,
            "rerun_required": 0,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, instrumented: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mode": "exact",
                "variant_name": "baseline",
                "n_points": 2,
                "enable_local_box_instrumentation": bool(instrumented),
            }
        ),
        encoding="utf-8",
    )


def _write_local_box_csv(path: Path) -> None:
    rows = [
        {
            "point_id": 0,
            "branch_id": 0,
            "selection_reason": "global_best",
            "box_q_min": -0.1,
            "box_q_max": 0.1,
            "box_Delta_min": 0.0,
            "box_Delta_max": 0.2,
            "box_runtime_sec": 0.5,
            "refined_q": 0.0,
            "refined_Delta": 0.0,
            "refined_DeltaF": 1e-5,
            "pruned_reason": "not_pruned_stage_1",
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_stage1_gate_verifier_passes_matching_artifacts(tmp_path):
    baseline_dir = tmp_path / "stage_00_baseline" / "regression_gpu_baseline"
    instrumented_dir = tmp_path / "stage_01_instrumentation" / "regression_gpu_instrumented"
    comparison_dir = tmp_path / "stage_01_instrumentation" / "baseline_vs_instrumented"
    local_box_csv = instrumented_dir / "baseline_local_box_timing.csv"

    _write_pointwise(baseline_dir / "baseline_pointwise.csv")
    _write_pointwise(instrumented_dir / "baseline_pointwise.csv")
    _write_manifest(baseline_dir / "regression_manifest.json", instrumented=False)
    _write_manifest(instrumented_dir / "regression_manifest.json", instrumented=True)
    _write_local_box_csv(local_box_csv)
    compare(baseline_dir / "baseline_pointwise.csv", instrumented_dir / "baseline_pointwise.csv", comparison_dir)

    summary = verify_gate(
        baseline_dir=baseline_dir,
        instrumented_dir=instrumented_dir,
        comparison_dir=comparison_dir,
        local_box_csv=local_box_csv,
        expected_points=2,
        max_q_opt_diff=0.0,
        max_delta_opt_diff=0.0,
        max_deltaf_diff=0.0,
    )
    write_gate_report(summary, comparison_dir)

    assert summary["status"] == "pass"
    assert (comparison_dir / "stage1_gate_status.json").exists()
    assert (comparison_dir / "stage1_gate_status.md").exists()


def test_stage1_gate_verifier_fails_when_artifacts_missing(tmp_path):
    baseline_dir = tmp_path / "missing_baseline"
    instrumented_dir = tmp_path / "missing_instrumented"
    comparison_dir = tmp_path / "missing_comparison"
    local_box_csv = instrumented_dir / "baseline_local_box_timing.csv"

    summary = verify_gate(
        baseline_dir=baseline_dir,
        instrumented_dir=instrumented_dir,
        comparison_dir=comparison_dir,
        local_box_csv=local_box_csv,
        expected_points=2,
        max_q_opt_diff=0.0,
        max_delta_opt_diff=0.0,
        max_deltaf_diff=0.0,
    )

    assert summary["status"] == "fail"
    assert "baseline_csv" in summary["missing_files"]
    assert "local_box_csv" in summary["missing_files"]


def test_stage1_output_collector_writes_archive_even_when_gate_fails(tmp_path):
    (tmp_path / "README.md").write_text("test package\n", encoding="utf-8")
    (tmp_path / "RUN_MANIFEST.json").write_text("{}", encoding="utf-8")
    fixed_points = tmp_path / "fixed_points" / "fixed_point_regression_points.csv"
    fixed_points.parent.mkdir(parents=True)
    fixed_points.write_text("point_id,kT,JA\n0,0.1,0.2\n", encoding="utf-8")

    archive = tmp_path / "stage1_results.tar.gz"
    manifest = collect_outputs(tmp_path, archive)

    assert archive.exists()
    assert manifest["gate_status"] == "fail"
    default_run_root = tmp_path / "local_refinement_refactor_stage1_run"
    metadata = default_run_root / "reports" / "local_refinement_refactor" / "stage_01_instrumentation" / "return_bundle_metadata"
    assert (metadata / "return_manifest.json").exists()
    assert (metadata / "missing_paths.txt").exists()
    assert not (tmp_path / "reports").exists()
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert "README.md" in names
    assert "RUN_MANIFEST.json" in names
    assert "fixed_points/fixed_point_regression_points.csv" in names


def test_stage1_return_importer_verifies_extracted_bundle(tmp_path):
    source_root = tmp_path / "source_bundle"
    baseline_dir = source_root / "reports" / "local_refinement_refactor" / "stage_00_baseline" / "regression_gpu_baseline"
    instrumented_dir = (
        source_root
        / "reports"
        / "local_refinement_refactor"
        / "stage_01_instrumentation"
        / "regression_gpu_instrumented"
    )
    comparison_dir = (
        source_root
        / "reports"
        / "local_refinement_refactor"
        / "stage_01_instrumentation"
        / "baseline_vs_instrumented"
    )
    local_box_csv = instrumented_dir / "baseline_local_box_timing.csv"

    _write_pointwise(baseline_dir / "baseline_pointwise.csv")
    _write_pointwise(instrumented_dir / "baseline_pointwise.csv")
    _write_manifest(baseline_dir / "regression_manifest.json", instrumented=False)
    _write_manifest(instrumented_dir / "regression_manifest.json", instrumented=True)
    _write_local_box_csv(local_box_csv)
    compare(baseline_dir / "baseline_pointwise.csv", instrumented_dir / "baseline_pointwise.csv", comparison_dir)

    archive = tmp_path / "returned_stage1_results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in source_root.rglob("*"):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(source_root))

    manifest = import_and_verify(
        archive,
        import_root=tmp_path / "imported",
        expected_points=2,
        max_q_opt_diff=0.0,
        max_delta_opt_diff=0.0,
        max_deltaf_diff=0.0,
    )

    assert manifest["gate_status"] == "pass"
    assert Path(manifest["gate_status_json"]).exists()
    assert Path(manifest["extract_dir"]).exists()

    cli_cwd = tmp_path / "external_cli_cwd"
    cli_cwd.mkdir()
    cli_import_root = tmp_path / "imported_cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_local_refinement_stage1_results.py"),
            str(archive),
            "--import-root",
            str(cli_import_root),
            "--expected-points",
            "2",
            "--max-q-opt-diff",
            "0.0",
            "--max-delta-opt-diff",
            "0.0",
            "--max-deltaf-diff",
            "0.0",
        ],
        cwd=cli_cwd,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"gate_status": "pass"' in completed.stdout
    assert (cli_import_root / "latest_import_manifest.json").exists()
    assert not (cli_cwd / "imported_results").exists()


def test_stage1_return_importer_cli_defaults_import_next_to_archive(tmp_path):
    source_root = tmp_path / "source_bundle"
    baseline_dir = source_root / "reports" / "local_refinement_refactor" / "stage_00_baseline" / "regression_gpu_baseline"
    instrumented_dir = (
        source_root
        / "reports"
        / "local_refinement_refactor"
        / "stage_01_instrumentation"
        / "regression_gpu_instrumented"
    )
    comparison_dir = (
        source_root
        / "reports"
        / "local_refinement_refactor"
        / "stage_01_instrumentation"
        / "baseline_vs_instrumented"
    )
    local_box_csv = instrumented_dir / "baseline_local_box_timing.csv"

    _write_pointwise(baseline_dir / "baseline_pointwise.csv")
    _write_pointwise(instrumented_dir / "baseline_pointwise.csv")
    _write_manifest(baseline_dir / "regression_manifest.json", instrumented=False)
    _write_manifest(instrumented_dir / "regression_manifest.json", instrumented=True)
    _write_local_box_csv(local_box_csv)
    compare(baseline_dir / "baseline_pointwise.csv", instrumented_dir / "baseline_pointwise.csv", comparison_dir)

    archive_dir = tmp_path / "downloaded_package" / "local_refinement_refactor_stage1_run"
    archive_dir.mkdir(parents=True)
    archive = archive_dir / "returned_stage1_results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in source_root.rglob("*"):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(source_root))

    cli_cwd = tmp_path / "external_cli_cwd"
    cli_cwd.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_local_refinement_stage1_results.py"),
            str(archive),
            "--expected-points",
            "2",
            "--max-q-opt-diff",
            "0.0",
            "--max-delta-opt-diff",
            "0.0",
            "--max-deltaf-diff",
            "0.0",
        ],
        cwd=cli_cwd,
        text=True,
        capture_output=True,
        check=True,
    )

    default_import_root = archive_dir / "imported_results"
    assert '"gate_status": "pass"' in completed.stdout
    assert (default_import_root / "latest_import_manifest.json").exists()
    assert not (cli_cwd / "imported_results").exists()


def test_hpc_package_validator_fails_missing_required_paths(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "README.md").write_text("incomplete\n", encoding="utf-8")

    summary = validate_package(package_dir, archive=None)

    assert summary["status"] == "fail"
    assert any("missing required path" in failure for failure in summary["failures"])


def test_stage1_hpc_preflight_passes_minimal_package(tmp_path):
    required_files = [
        "README.md",
        "RUN_MANIFEST.json",
        "eta_phase_diagram_cuda.py",
        "tfflo_1d_cuda.py",
        "ml_phase/__init__.py",
        "ml_phase/exact_oracle.py",
        "scripts/run_local_refinement_fixed_point_regression.py",
        "scripts/compare_local_refinement_variants.py",
        "scripts/verify_local_refinement_stage1_gate.py",
        "scripts/collect_local_refinement_stage1_outputs.py",
        "scripts/preflight_local_refinement_stage1_hpc.py",
    ]
    for rel_path in required_files:
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel_path.endswith(".py"):
            path.write_text("x = 1\n", encoding="utf-8")
        elif rel_path == "RUN_MANIFEST.json":
            path.write_text(
                json.dumps(
                    {
                        "package_name": "local_refinement_refactor_stage01_instrumentation",
                        "active_learning": "not_run",
                        "fixed_points": "fixed_points/fixed_point_regression_points.csv",
                        "instrumentation_flag": "--enable-local-box-instrumentation",
                    }
                ),
                encoding="utf-8",
            )
        else:
            path.write_text("test\n", encoding="utf-8")
    for rel_path in [
        "scripts/submit_stage1_regression_workflow.sh",
        "scripts/submit_local_refinement_fixed_point_regression.sh",
        "scripts/submit_local_refinement_instrumented_benchmark.sh",
        "scripts/slurm_stage0_baseline_regression.sh",
        "scripts/slurm_stage1_instrumented_regression.sh",
        "scripts/slurm_stage1_postprocess.sh",
    ]:
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\n", encoding="utf-8")

    fixed_points = tmp_path / "fixed_points" / "fixed_point_regression_points.csv"
    fixed_points.parent.mkdir(parents=True, exist_ok=True)
    with fixed_points.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["point_id", "kT", "JA"])
        writer.writeheader()
        writer.writerows({"point_id": i, "kT": 0.1, "JA": 0.2} for i in range(2))

    summary = run_preflight(tmp_path, tmp_path / "preflight.json", expected_fixed_points=2)

    assert summary["status"] == "pass"
    assert summary["checked"]["fixed_point_count"] == 2
    assert summary["checked"]["import_check_count"] == 2
    assert (tmp_path / "preflight.json").exists()


def test_fixed_point_runner_imports_package_root_modules_from_scripts_entry(tmp_path):
    package_root = tmp_path / "package"
    external_cwd = tmp_path / "external_cwd"
    external_cwd.mkdir()
    scripts_dir = package_root / "scripts"
    scripts_dir.mkdir(parents=True)
    (package_root / "ml_phase").mkdir()
    (package_root / "ml_phase" / "__init__.py").write_text("", encoding="utf-8")
    (scripts_dir / "run_local_refinement_fixed_point_regression.py").write_text(
        (ROOT / "scripts" / "run_local_refinement_fixed_point_regression.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (package_root / "eta_phase_diagram_cuda.py").write_text(
        "class EtaPhaseConfig:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package_root / "ml_phase" / "exact_oracle.py").write_text(
        "import numpy as np\n"
        "from eta_phase_diagram_cuda import EtaPhaseConfig\n\n"
        "class Result:\n"
        "    def __init__(self, points):\n"
        "        n = int(points.shape[0])\n"
        "        self.kT = points[:, 0]\n"
        "        self.JA = points[:, 1]\n"
        "        self.phase_candidate = np.zeros(n, dtype=int)\n"
        "        self.q_opt = np.zeros(n)\n"
        "        self.delta_opt = np.zeros(n)\n"
        "        self.free_energy_gap_to_normal = np.zeros(n)\n"
        "        self.trusted_exact = np.ones(n, dtype=int)\n"
        "        self.training_eligible_exact = np.ones(n, dtype=int)\n"
        "        self.q_unresolved = np.zeros(n, dtype=int)\n"
        "        self.delta_unresolved = np.zeros(n, dtype=int)\n"
        "        self.rerun_required = np.zeros(n, dtype=int)\n"
        "        self.local_minima_detected_count = np.zeros(n, dtype=int)\n"
        "        self.local_boxes_refined_count = np.zeros(n, dtype=int)\n"
        "        self.local_refinement_reused_count = np.zeros(n, dtype=int)\n"
        "        self.point_total_runtime_sec = np.zeros(n)\n"
        "        self.local_refinement_runtime_sec = np.zeros(n)\n\n"
        "def evaluate_points(points, cfg, **_kwargs):\n"
        "    if not isinstance(cfg, EtaPhaseConfig):\n"
        "        raise TypeError(type(cfg).__name__)\n"
        "    return Result(points)\n",
        encoding="utf-8",
    )

    points_file = package_root / "fixed_points" / "fixed_point_regression_points.csv"
    points_file.parent.mkdir()
    with points_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "source_run", "kT", "JA"])
        writer.writeheader()
        writer.writerow({"category": "stub", "source_run": "test", "kT": "0.1", "JA": "0.2"})

    completed = subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "run_local_refinement_fixed_point_regression.py"),
            "--points-file",
            "fixed_points/fixed_point_regression_points.csv",
            "--output-dir",
            "reports/regression_stub",
            "--device",
            "cpu",
        ],
        cwd=external_cwd,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Wrote local-refinement regression outputs" in completed.stdout
    default_run_root = package_root / "local_refinement_refactor_stage1_run"
    assert (default_run_root / "reports" / "regression_stub" / "baseline_pointwise.csv").exists()
    assert (default_run_root / "reports" / "regression_stub" / "regression_manifest.json").exists()
    assert not (package_root / "reports").exists()
    assert not (external_cwd / "reports").exists()


def test_fixed_point_runner_writes_relative_outputs_to_run_root(tmp_path):
    package_root = tmp_path / "package"
    external_cwd = tmp_path / "external_cwd"
    run_root = tmp_path / "writable_run_root"
    external_cwd.mkdir()
    run_root.mkdir()
    scripts_dir = package_root / "scripts"
    scripts_dir.mkdir(parents=True)
    (package_root / "ml_phase").mkdir()
    (package_root / "ml_phase" / "__init__.py").write_text("", encoding="utf-8")
    (scripts_dir / "run_local_refinement_fixed_point_regression.py").write_text(
        (ROOT / "scripts" / "run_local_refinement_fixed_point_regression.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (package_root / "eta_phase_diagram_cuda.py").write_text(
        "class EtaPhaseConfig:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package_root / "ml_phase" / "exact_oracle.py").write_text(
        "import numpy as np\n"
        "from eta_phase_diagram_cuda import EtaPhaseConfig\n\n"
        "class Result:\n"
        "    def __init__(self, points):\n"
        "        n = int(points.shape[0])\n"
        "        self.kT = points[:, 0]\n"
        "        self.JA = points[:, 1]\n"
        "        self.phase_candidate = np.zeros(n, dtype=int)\n"
        "        self.q_opt = np.zeros(n)\n"
        "        self.delta_opt = np.zeros(n)\n"
        "        self.free_energy_gap_to_normal = np.zeros(n)\n"
        "        self.trusted_exact = np.ones(n, dtype=int)\n"
        "        self.training_eligible_exact = np.ones(n, dtype=int)\n"
        "        self.q_unresolved = np.zeros(n, dtype=int)\n"
        "        self.delta_unresolved = np.zeros(n, dtype=int)\n"
        "        self.rerun_required = np.zeros(n, dtype=int)\n"
        "        self.local_minima_detected_count = np.zeros(n, dtype=int)\n"
        "        self.local_boxes_refined_count = np.zeros(n, dtype=int)\n"
        "        self.local_refinement_reused_count = np.zeros(n, dtype=int)\n"
        "        self.point_total_runtime_sec = np.zeros(n)\n"
        "        self.local_refinement_runtime_sec = np.zeros(n)\n\n"
        "def evaluate_points(points, cfg, **_kwargs):\n"
        "    if not isinstance(cfg, EtaPhaseConfig):\n"
        "        raise TypeError(type(cfg).__name__)\n"
        "    return Result(points)\n",
        encoding="utf-8",
    )
    points_file = package_root / "fixed_points" / "fixed_point_regression_points.csv"
    points_file.parent.mkdir()
    with points_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "source_run", "kT", "JA"])
        writer.writeheader()
        writer.writerow({"category": "stub", "source_run": "test", "kT": "0.1", "JA": "0.2"})

    completed = subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "run_local_refinement_fixed_point_regression.py"),
            "--points-file",
            "fixed_points/fixed_point_regression_points.csv",
            "--output-dir",
            "reports/regression_stub",
            "--device",
            "cpu",
        ],
        cwd=external_cwd,
        env={**os.environ, "RUN_ROOT": str(run_root)},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Wrote local-refinement regression outputs" in completed.stdout
    assert (run_root / "reports" / "regression_stub" / "baseline_pointwise.csv").exists()
    assert (run_root / "reports" / "regression_stub" / "regression_manifest.json").exists()
    assert not (package_root / "reports").exists()
    assert not (external_cwd / "reports").exists()


def test_stage1_package_defaults_runtime_outputs_under_package_run_dir():
    preamble = _shell_roots_preamble()
    readme = _readme()
    workflow = _workflow_submit_script()
    alias = _workflow_alias_script("submit_stage1_regression_workflow.sh")

    assert 'RUN_ROOT="${PACKAGE_ROOT}/local_refinement_refactor_stage1_run"' in preamble
    assert 'RUN_ROOT="${PACKAGE_ROOT}"' not in preamble
    assert "$PACKAGE_ROOT/local_refinement_refactor_stage1_run" in readme
    assert "submit_local_refinement_fixed_point_regression.sh" in readme
    assert "submit_local_refinement_instrumented_benchmark.sh" in readme
    assert "scripts write under the package root when it is\nwritable" not in readme
    assert '--output-json "${RUN_ROOT}/reports/local_refinement_refactor/stage_01_instrumentation/stage1_runtime_preflight.json"' in workflow
    assert 'exec bash "${SCRIPT_DIR}/submit_stage1_regression_workflow.sh" "$@"' in alias
