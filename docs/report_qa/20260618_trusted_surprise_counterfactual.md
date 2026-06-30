# 2026-06-18 Trusted Surprise Counterfactual

## Context

This note records a report-only counterfactual audit of the rankcap_k3 full-loop
StopController label-surprise blocker.  No StopController, acquisition, exact
oracle, rankcap_k3, tolerance, Slurm state, or active-learning artifact was
modified.

## StopController Reproduction

The current `label_surprise_rate` definition was exactly reproduced for all
saved acquisition iterations.  It matches selected rows to exact merged labels
by rounded `(kT, JA)` and computes the denominator from all matched selected
exact points.  It does not filter by `trusted_exact`,
`training_eligible_exact`, `rerun_required`, `q_unresolved`, or
`delta_unresolved`.

Final current all-selected surprise:

```text
47 / 256 = 0.183594
```

All final selected-batch surprises are in the hard-risk group.

## Trusted Surprise Counterfactual

The trusted mask used in the report was:

```text
trusted_exact == True
training_eligible_exact == True
rerun_required == False
q_unresolved == False
delta_unresolved == False
```

`q_expanded == True` was not excluded by itself.

Final counterfactual metrics:

```text
surprise_nonrerun          = 0 / 163 = 0.000000
surprise_training_eligible = 0 / 163 = 0.000000
surprise_trusted           = 0 / 137 = 0.000000
surprise_hard_risk         = 47 / 119 = 0.394958
```

For the strict last-five scope:

```text
all surprise count = 205
rerun-required surprise count = 205
fraction from rerun-required = 1.000000
trusted surprise count = 0
```

## Counterfactual Stop Result

Keeping all other StopController inputs fixed and replacing only the surprise
input with `surprise_trusted`, the first iteration meeting the required pass
count is iteration 11 and the patience rule would trigger formal convergence at
iteration 17.

At the final iteration, the remaining failed condition under trusted surprise
is:

```text
boundary_coverage_p95
```

The trusted-surprise counterfactual still reaches the required pass count
because phase-map change, normal/SC boundary shift, uniform-SC/FFLO boundary
shift, and trusted surprise all pass.

## Decision

Recommended decision:

```text
Decision B: split formal surprise into trusted and hard-risk layers.
```

Evidence:

```text
trusted denominator is large enough in the final batch
trusted surprise is below tolerance
last-five surprise is entirely rerun-required
all-selected surprise is dominated by hard-risk numerical frontier points
```

Suggested future code design, not implemented here:

```text
keep current all-selected surprise as acquisition difficulty diagnostic
add surprise_trusted and surprise_hard_risk
gate formal convergence on trusted surprise with a configured denominator floor
record hard-risk surprise separately and route those points to a numerical
reliability queue
```

## Generated Report

```text
reports/trusted_surprise_counterfactual/trusted_surprise_counterfactual.md
reports/trusted_surprise_counterfactual/trusted_surprise_counterfactual.pdf
reports/trusted_surprise_counterfactual/decision_log.md
reports/trusted_surprise_counterfactual/tables/
reports/trusted_surprise_counterfactual/figures/
```
