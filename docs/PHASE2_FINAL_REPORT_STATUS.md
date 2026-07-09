# Phase-II Final Report Status

Status: final_report_completed

Date: 2026-06-20

## Current State

```text
audit_planning = complete
audit_running = complete
audit_passed = True
audit_failed = False
final_report_building = complete
final_report_completed = True
final_report_status = complete
```


## Audit Result

```text
publication_boundary_audit = pass
audit_decision = Decision A
need_new_exact_calculation = False
targeted_rerun_count = 0
hard_risk_total = 129
boundary_near_hard_risk_count = 88
max_local_p95_shift = 0.0
strict_local_hausdorff_diagnostic = 0.8125
significant_local_hausdorff_gate_value = 0.0
significant_arc_fraction_threshold = 0.05
```

## Outputs

```text
project_history/reports/report_active_learning_r0015_note/
project_history/reports/report_phase2_robust_al_final_202606/
project_history/reports/_supporting_reports/
reports/_phase2_supporting_reports/
```

## LaTeX Build

```text
latex_report_build = complete
latex_builder = scripts/build_phase2_latex_reports.py
audit_tex = reports/_phase2_supporting_reports/hard_risk_boundary_impact_audit_v2/hard_risk_boundary_impact_audit_v2.tex
audit_pdf = reports/_phase2_supporting_reports/hard_risk_boundary_impact_audit_v2/hard_risk_boundary_impact_audit_v2.pdf
phase2_tex = project_history/reports/report_phase2_robust_al_final_202606/phase2_robust_al_final_report.tex
phase2_pdf = project_history/reports/report_phase2_robust_al_final_202606/phase2_robust_al_final_report.pdf
compiler = pdflatex
audit_pages = 6
phase2_pages = 14
render_check = pass
```

The LaTeX build is report-only.  It reuses the existing audit/report CSV,
JSON, Markdown, and PNG artifacts and does not modify numerical data,
production oracle code, acquisition logic, StopController logic, phase
criteria, or tolerances.

## Report Consolidation

```text
project_history/reports first-level entries:
    _supporting_reports
    report_active_learning_r0015_note
    report_phase2_robust_al_final_202606

reports first-level entries:
    _phase2_supporting_reports
```

The Phase-II final report was rewritten as the primary discussion report.  It
now includes the random-seed active-learning loop, acquisition/oracle
responsibility split, optimization timeline, validation ladder, final phase
map, learning curves, hard-risk uncertainty layer, and supporting-report map.
