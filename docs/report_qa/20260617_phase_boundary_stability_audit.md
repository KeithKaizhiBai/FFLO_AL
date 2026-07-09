# Phase Boundary Stability Audit Note

Date: 2026-06-17

This note summarizes the report-only audit of how StopController computes
phase-map stability and boundary-shift stability for the
`active_boundary_discovery_rankcap_k3_full_loop_v1` full-loop run.

Generated report:

```text
reports/phase_boundary_stability_audit/phase_boundary_stability_audit.md
reports/phase_boundary_stability_audit/phase_boundary_stability_audit.pdf
reports/phase_boundary_stability_audit/tables/
reports/phase_boundary_stability_audit/figures/
```

Main findings:

```text
phase_map_change:
    Defined as mean(current phase_pred != previous phase_pred) on the saved
    monitor dense-grid surrogate predictions.
    It is not an exact-dataset phase-map metric.

boundary_shift_normal_sc / boundary_shift_uniform_fflo:
    Boundaries are extracted from predicted phase-label crossings on the same
    dense monitor grid.
    The metric is the p95 of the bidirectional normalized nearest-neighbor
    distances between current and previous boundary point sets.

boundary_coverage_p95:
    Different from boundary_shift.
    It measures current predicted boundary points to exact dataset samples.
```

Final StopController state:

```text
stop_reason = max_iterations
convergence_pass = false
passed_condition_count = 3
required_pass_count = 4
patience_counter = 0

phase_map_change = 0.0006204676775119246 < 0.002
boundary_shift_normal_sc = 0.002604166666666674 < 0.004166666666666667
boundary_shift_uniform_fflo = 0.0 < 0.004166666666666667
label_surprise_rate = 0.18359375 > 0.05
boundary_coverage_p95 = 0.006588078458684216 > 0.00625
```

Important interpretation:

```text
The main predicted phase map and the main predicted thermodynamic boundaries
are stable by the StopController stability metrics.  Formal convergence did not
pass because label surprise and boundary coverage failed.  These are different
statements and should not be collapsed into "the phase map did not converge".
```

Validation:

```text
python -m py_compile scripts/build_phase_boundary_stability_audit.py
python scripts/build_phase_boundary_stability_audit.py
pdflatex -interaction=nonstopmode phase_boundary_stability_audit.tex
pdftoppm -png -r 130 reports/phase_boundary_stability_audit/phase_boundary_stability_audit.pdf tmp/pdfs/phase_boundary_stability_audit/page
```

The audit recomputed `phase_map_change`, both `boundary_shift` metrics, and
`boundary_coverage_p95` from saved artifacts.  The maximum absolute discrepancy
against stored StopController metrics was zero for all recomputed metrics.

Do not claim:

```text
1. Do not claim formal StopController convergence passed.
2. Do not treat high label surprise as proof of large boundary drift.
3. Do not conflate boundary coverage failure with boundary shift failure.
4. Do not modify StopController or thresholds based on this report.
5. Do not interpret eta response as a thermodynamic phase boundary.
```

Recommended next step:

```text
If formal convergence is scientifically required, plan a separate cleanup
acquisition or boundary-coverage validation targeting label surprise and
boundary coverage.  Do not change thresholds as part of this audit.
```
