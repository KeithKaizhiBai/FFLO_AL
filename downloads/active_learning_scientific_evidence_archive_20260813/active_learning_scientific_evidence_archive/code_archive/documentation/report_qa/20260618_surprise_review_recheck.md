# 2026-06-18 Surprise Review Recheck

## Question

Does the late-stage `label_surprise_rate` indicate broad phase-map instability,
or is it a selected-batch artifact concentrated in a specific hard-risk class?

## Answer

The recheck supports the latter.  The final-iteration and strict last-five
statistics show that selected-batch surprise is concentrated in
`predicted normal -> exact FFLO` points close to the predicted normal/SC
boundary.

Final iteration:

```text
selected = 256
matched exact = 256
surprise = 47
surprise_rate = 0.18359375
all final surprises = predicted normal -> exact FFLO
```

Strict last-five iterations:

```text
selected = 1280
matched exact = 1280
surprise = 205
surprise_rate = 0.16015625
all last-five surprises = predicted normal -> exact FFLO
```

The StopController denominator was confirmed from
`ml_phase/stop_controller.py::label_surprise_rate`: it counts every selected
point that matches `exact_merged_iterXXX.npz` by rounded `(kT, JA)`.  It does
not filter by `trusted_exact`, `training_eligible_exact`, `rerun_required`, or
q-expansion status.

Layered last-five surprise:

```text
StopController all matched selected: 205 / 1280 = 0.16015625
trusted nonrerun:                    0 / 849 = 0
trusted nonrerun, no q-edge:         0 / 489 = 0
trusted qexpanded nonrerun:          0 / 360 = 0
rerun required:                    205 / 341 = 0.6011730205278593
qexpanded or q-edge:               205 / 791 = 0.25916561314791403
```

Thus the surprise that blocks formal convergence is dominated by
rerun-required hard-risk selected points.  The trusted/nonrerun subset in the
last five iterations has zero surprise under this audit.

## Consequence

Do not treat selected-batch surprise as a global phase-map error proxy.  A fair
convergence report should separate:

```text
all-selected surprise
trusted-nonrerun surprise
trusted-qexpanded surprise
rerun-required surprise
fixed-probe or random-control surprise
```

The current artifacts do not contain an independent fixed-probe or random
control batch with paired pre-exact predictions, so that metric is currently
`cannot determine`.

## Evidence Files

```text
reports/surprise_review_recheck/surprise_review_recheck.md
reports/surprise_review_recheck/surprise_review_recheck.pdf
reports/surprise_review_recheck/tables/stopcontroller_surprise_denominator_audit.csv
reports/surprise_review_recheck/tables/scope_summary.csv
reports/surprise_review_recheck/tables/layered_surprise_metrics.csv
reports/surprise_review_recheck/tables/normal_to_fflo_diagnostics_by_scope.csv
```

## Recommended Next Step

Before changing StopController or continuing full discovery, add a held-out
fixed-probe or random-control exact batch.  Use it alongside the selected-batch
stratified surprise metrics to decide whether formal convergence should depend
on all-selected surprise or a trusted/control surprise criterion.
