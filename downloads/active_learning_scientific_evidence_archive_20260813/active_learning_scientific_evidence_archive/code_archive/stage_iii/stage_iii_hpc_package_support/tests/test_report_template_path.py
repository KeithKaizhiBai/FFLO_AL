from __future__ import annotations

import json
from pathlib import Path

from ml_phase.report_builder import build_report


def test_default_report_template_resolves_or_falls_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_root = tmp_path / "ML_Phase" / "active_runs"
    run_dir = run_root / "run001"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": {"run_id": "run001"},
                "active_learning_config": {
                    "run_mode": "discovery",
                    "candidate_domain_mode": "full",
                    "selection_mode": "stochastic",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics_history.json").write_text("[]", encoding="utf-8")

    out = build_report(
        run_dir=run_dir,
        template_path=Path("report/active_learning_phase_boundary_report.tex"),
        output_tex=tmp_path / "ML_Phase" / "reports" / "active_learning_phase_boundary_report.tex",
    )
    text = out.read_text(encoding="utf-8")
    assert "Template source:" in text
    assert "run001" in text


def test_explicit_missing_report_template_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "ML_Phase" / "active_runs" / "run001"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({"args": {"run_id": "run001"}, "active_learning_config": {}}),
        encoding="utf-8",
    )
    (run_dir / "metrics_history.json").write_text("[]", encoding="utf-8")

    try:
        build_report(
            run_dir=run_dir,
            template_path=Path("missing/custom_template.tex"),
            output_tex=tmp_path / "out.tex",
        )
    except FileNotFoundError as exc:
        assert "LaTeX report template not found" in str(exc)
        assert "Pass --template" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for explicit missing template")
