# Second-Stage Discussion Report

Date: 2026-06-17

## Question

Can we assemble a discussion-ready second-stage report that covers the route
from the first exact grid phase diagram, through active learning, numerical
audits, local-refinement optimization, and the latest rankcap_k3 full-loop
result?

## Output

Generated report directory:

```text
project_history/reports/report_second_stage_discussion/
```

Main files:

```text
second_stage_discussion_report.md
second_stage_discussion_report.tex
second_stage_discussion_report.pdf
decision_log.md
tables/milestone_summary.csv
tables/validation_summary.csv
tables/speedup_summary.csv
tables/final_state_summary.csv
tables/figure_manifest.csv
figures/*.png
```

## Report Position

This report is intended as the current high-level discussion narrative.  It
supersedes the older local-refinement refactor note for current status, but it
does not replace the detailed technical reports and CSV evidence.

The structure is milestone-oriented:

```text
exact warm-up phase diagram
active learning from exact data
from-scratch active learning
q-window / Delta / response numerical audits
target-construction failure diagnosis
rank_and_cap_k3 design and validation
five-iteration closed-loop validation
full-loop validation
formal convergence status and remaining issue
```

## Main Scientific/Engineering Message

The project has moved from phase-map discovery to a validated cost-controlled
exact oracle.  The machine-learning model remains a scheduler for exact calls;
physical labels still come from exact BdG free-energy minimization.

The rank_and_cap_k3 local-refinement optimization should be described as
accepted for oracle-cost reduction:

```text
fixed-point acceptance: pass
one-iteration validation: pass
five-iteration validation: pass after rank-level recheck
full-loop validation: pass after rank-level recheck
```

The latest full loop:

```text
31 exact iterations = one seed iteration plus 30 acquisition batches
final dataset samples = 6880
normal = 1777
uniform_SC = 715
FFLO = 4388
mean local boxes = 2.79297
corrected max local boxes = 3
```

Compared with the robust-incremental reference:

```text
mean local boxes: 6.0 -> 2.79297
mean local-refinement runtime: 189.767 -> 88.2856 sec/point
mean point-total runtime: 234.194 -> 117.285 sec/point
```

## Important Caveat

The full loop is not a formal StopController convergence result:

```text
stop_reason = max_iterations
convergence_pass = false
label_surprise_rate = 0.18359375 > 0.05
boundary_coverage_p95 = 0.006588078458684216 > 0.00625
```

The last-five audit supports that this is a late-stage selection / stopping
metric issue, not evidence that rankcap_k3 broke exact labels.

## Verification

Commands run:

```text
python -m py_compile scripts/build_second_stage_discussion_report.py
python scripts/build_second_stage_discussion_report.py
pdflatex -interaction=nonstopmode second_stage_discussion_report.tex
pdftoppm -png -r 130 project_history/reports/report_second_stage_discussion/second_stage_discussion_report.pdf tmp/pdfs/second_stage_discussion_page
```

Rendered pages visually checked:

```text
page 1: title, abstract, contents
page 2: executive summary and roadmap flow
page 10: target-explosion and validation-funnel figures
page 11: validation table and fixed-point local-box figure
page 13: five-iteration phase growth and full-loop final phase diagram
page 17: last-five stop-failure diagnostics
```

## Do Not Overclaim

Do not claim formal active-learning convergence.

Do not claim raw per-epoch training loss curves exist in the returned full-loop
package; the report uses surrogate validation metrics.

Do not claim rankcap_k3 changes the thermodynamic phase criterion.

Do not use old uncorrected local-box aggregation as an authoritative validation
gate.
