# Phase-II LaTeX Report Build

Date: 2026-06-20

## Summary

The Phase-II final report and hard-risk boundary-impact audit now have
LaTeX source files and PDF outputs compiled with `pdflatex`.

## Files

```text
scripts/build_phase2_latex_reports.py
reports/hard_risk_boundary_impact_audit_v2/hard_risk_boundary_impact_audit_v2.tex
reports/hard_risk_boundary_impact_audit_v2/hard_risk_boundary_impact_audit_v2.pdf
report_phase2_robust_al_final_202606/phase2_robust_al_final_report.tex
report_phase2_robust_al_final_202606/phase2_robust_al_final_report.pdf
```

## Verification

```text
python -m py_compile scripts/build_phase2_latex_reports.py
python scripts/build_phase2_latex_reports.py
pdflatex hard_risk_boundary_impact_audit_v2.tex
pdflatex phase2_robust_al_final_report.tex
pdftoppm render spot checks
```

The compiled PDFs have 6 pages for the audit and 11 pages for the Phase-II
final report.  Spot-rendered pages show readable tables and embedded figures.

## Scope

This was a report-generation update only.  No numerical results were
recomputed, and no production physics or active-learning code was changed.
