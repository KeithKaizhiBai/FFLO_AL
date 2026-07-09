from __future__ import annotations

import csv
import json

from scripts import audit_twophase_optimization_completion as audit_script


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_twophase_completion_audit_reports_pending_hpc_gate(tmp_path):
    json_path = tmp_path / "twophase_completion_audit.json"
    csv_path = tmp_path / "twophase_completion_requirements.csv"
    md_path = tmp_path / "twophase_completion_audit.md"

    summary = audit_script.audit_twophase_completion(json_path=json_path, csv_path=csv_path, md_path=md_path)

    assert summary["status"] == "pending_hpc"
    assert summary["status_counts"]["pending_hpc"] == 1
    assert summary["status_counts"]["pass"] == 11
    assert summary["requirement_count"] == 15
    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()

    rows = {row["requirement"]: row for row in _read_csv(csv_path)}
    assert rows["all Stage 0-7 report files exist"]["status"] == "pass"
    assert rows["runbook-named unit/synthetic tests exist"]["status"] == "pass"
    assert rows["goal-run report protocol validates"]["status"] == "pass"
    assert rows["HPC upload-set verifier passes"]["status"] == "pass"
    assert rows["nested package shell outputs stay under RUN_ROOT"]["status"] == "pass"
    assert rows["GPU Slurm scripts exclude gpuh01 and probe CUDA runtime"]["status"] == "pass"
    assert "gpu_script_count=7" in rows["GPU Slurm scripts exclude gpuh01 and probe CUDA runtime"]["evidence"]
    assert rows["variant-suite return readiness checker is available"]["status"] == "pass"
    assert "checklist_mentions_checker=1" in rows["variant-suite return readiness checker is available"]["evidence"]
    assert rows["variant-suite HPC status checker is packaged"]["status"] == "pass"
    assert "package_present=1" in rows["variant-suite HPC status checker is packaged"]["evidence"]
    assert rows["Stage 2-4 GPU variant-suite return is imported and passed"]["status"] == "pending_hpc"
    assert "gate_status=pending" in rows["Stage 2-4 GPU variant-suite return is imported and passed"]["evidence"]

    json_summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_summary["status"] == "pending_hpc"
    assert "Stage 2-4 variant-suite GPU return archive" in md_path.read_text(encoding="utf-8")
