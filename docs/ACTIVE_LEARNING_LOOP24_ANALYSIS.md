# Active Learning Loop24 Analysis

Date: 2026-05-11

Analyzed output:

```text
hpc_upload_qdelta_20260510_143031/ML_Phase_128_loop24
```

Run ID:

```text
active_boundary_loop_v1
```

## 1. What Was Completed

The 128-point automatic active-learning loop completed 25 recorded iterations:

```text
iter000 through iter024
```

The automatic workflow repeatedly performed:

```text
train/select candidates
run H100 exact oracle
merge shards
filter trusted exact points
append trusted points
continue to the next iteration
build report
```

This validates the active-learning loop as a working exact-BdG scheduler rather
than a manually stitched workflow.

## 2. Aggregate Results

```text
selected exact calls: 3200
merged exact points: 3200
trusted exact points: 3053
rerun / boundary-band points: 147

initial warm-start samples: 21528
latest dataset_iter025 samples: 23214
net sample increase: 1686

q expanded and confirmed: 21
q unresolved: 0
q-edge-hit after confirmation: 0

Delta refined: 2574
Delta unresolved before boundary-band interpretation: 147
```

The run generated figures for all iterations and a compiled report:

```text
ML_Phase_128_loop24/reports/active_learning_phase_boundary_report.pdf
```

The report compiled successfully. A known report issue remains: `Current exact
samples` reads the previous iteration input dataset rather than the final
appended dataset. For this run, the report shows 23181 while
`dataset_iter025.npz` contains 23214 samples.

## 3. Scientific Interpretation

The active learner repeatedly selected points near the intended high-value
regions:

```text
high-J_A, low-temperature boundary
main normal/SC boundary around intermediate J_A
low-J_A, low-temperature eta/Delta-sensitive region
high-temperature tail where the boundary bends downward
```

Across all selected points:

```text
JA > 1.2 selected points: 1270
kT < 0.1 selected points: 1394
```

This supports the core active-learning objective: exact BdG calls are being
concentrated near phase-boundary and high-risk regions rather than being spent
uniformly over the full phase diagram.

## 4. q-Window Status

The q-window logic is not the current bottleneck.

```text
q expanded and confirmed: 21
q unresolved: 0
q-edge-hit after confirmation: 0
```

This supports the design decision that q should be adaptive for high \(J_A\).
The workflow found and corrected q-window risk cases without leaving unresolved
q-truncated labels in the trusted dataset.

## 5. Delta Boundary-Band Status

All 147 non-trusted points had the same reason:

```text
delta_boundary_unresolved;max_delta_refinement_reached
```

Most of them had:

```text
Delta_opt = 0
positive_delta_gap ~ 1e-9 to 1e-8
```

This means the best strictly positive-\(\Delta\) state was not lower than the
normal state by a resolvable condensation-energy scale. At the adopted
precision, these points should be interpreted as normal-side or normal/SC
boundary-band data, not as generic exact-oracle failures.

Recommended interpretation:

```text
robust SC:
    superconducting free-energy gain < -tolerance

normal-side boundary band:
    Delta_opt = 0
    positive_delta_gap >= 0
    positive_delta_gap <= positive_delta_gap_tol

stable normal:
    Delta_opt = 0
    positive_delta_gap > positive_delta_gap_tol
```

The finite-resolution boundary band should be preserved in metadata and in the
report. It should not trigger unlimited repeated Delta refinement unless a
future study specifically requires resolving a much smaller energy scale.

## 6. Other Pain Points

### Repeated Boundary-Band Coordinates

The 147 boundary-band/rerun records correspond to only 73 rounded coordinates.
Some coordinates reappeared five or six times across iterations. This means the
acquisition function identifies them as important but the loop currently has no
cooldown or boundary-band suppression policy.

Impact:

```text
active learning spends repeated budget on points that are already known to be
inside the chosen finite-resolution boundary band
```

### Report Lag

The report's total sample count lags by one dataset append. Latest-iteration
exact statistics are correct, but `Current exact samples` should report the
latest available `dataset_iterXXX.npz`, not the input dataset for the latest
iteration.

### Missing Boundary-Shift Diagnostic

The current report has RMSE, phase accuracy, and learning curves, but it does
not directly quantify whether the inferred boundary is still moving. A useful
active-learning stopping criterion requires boundary displacement or selected
point overlap metrics.

### Boundary F1 Is Often N/A

`Boundary F1` becomes unavailable after the initial warm-start validation. This
likely reflects validation-label sparsity or the current boundary-positive
definition. It should be replaced or supplemented with a direct boundary-shift
metric.

## 7. Next Direction

Immediate next steps:

```text
1. Change reporting semantics from "Delta unresolved failure" to
   "finite-resolution normal/SC boundary band" when the positive-Delta state is
   not lower than the normal state by the adopted tolerance.

2. Add a cooldown or suppression rule for repeated boundary-band coordinates so
   the active learner does not repeatedly spend exact calls on the same
   sub-tolerance points.

3. Add per-iteration diagnostics:
   trusted / selected
   boundary-band count
   q-expanded count
   selected-point overlap with earlier iterations
   boundary displacement between iterations
   latest dataset size after append

4. Fix report_builder.py so the total sample count uses the latest appended
   dataset.

5. Extract an explicit phase-boundary curve with an uncertainty band. This is
   the output most directly aligned with the scientific goal.
```

After these changes, run another moderate continuation test, for example 5 to
10 iterations at 128 or 256 points per iteration, before increasing to 512+
points. The purpose should be to test boundary convergence, not merely to
increase exact-call count.

## 8. Implemented Follow-Up

Implemented on 2026-05-11:

```text
ml_phase.hpc:
    derives delta_boundary_band_normal, training_eligible_exact, and
    needs_rerun_exact during shard merge;
    writes merge_summary_iterXXX.json;
    writes exact_training_iterXXX.npz alongside exact_trusted_iterXXX.npz.

ml_phase.append_trusted:
    appends training_eligible_exact points;
    reports clean_trusted_points and boundary_band_points_appended separately.

ml_phase.active_refine:
    suppresses previously identified boundary-band coordinates during future
    candidate selection;
    writes selection_diagnostics.json for selected-point reuse diagnostics.

ml_phase.report_builder:
    reports the latest appended dataset size;
    separates clean trusted, boundary-band normal, training-eligible, and true
    rerun-required counts.
```

Recommended validation calculation:

```text
Continue from dataset_iter025 for 3 to 5 iterations at 128 points per
iteration. The expected behavior is that boundary-band normal points are
included in training, true rerun-required points drop, and repeated exact
coordinates from the old boundary band are suppressed.
```

## 9. Tail10 Validation Result

Analyzed output:

```text
hpc_upload_qdelta_20260511_115659/ML_Phase_128_looptail10
```

This continuation contains:

```text
iter025 through iter034
dataset_iter025 through dataset_iter035
```

Aggregate result:

```text
selected exact calls: 1280
clean trusted points: 1215
boundary-band normal points: 63
training-eligible points: 1278
true rerun-required points: 2
q expanded and confirmed: 6
q unresolved: 0
Delta refined: 1142
Delta unresolved: 65
Delta unresolved requiring rerun: 2

dataset_iter025 samples: 23214
dataset_iter035 samples: 23587
net unique sample increase: 373
```

Interpretation:

```text
The boundary-band semantic change works. Most points that would previously
have appeared as Delta-unresolved failures are now training-eligible
normal/SC boundary-band normal points.
```

The two remaining true rerun-required points are positive-Delta boundary cases
near:

```text
kT/t ~= 0.074667
J_A/t ~= 1.19
```

They are not the common Delta=0 positive-gap boundary-band normal case and
should remain marked for stricter inspection if this small region matters.

New bottleneck:

```text
candidate efficiency
```

Although 1278 of 1280 exact records are training-eligible, the net unique
dataset increase is only 373 points. `selection_diagnostics.json` shows that
roughly 82 to 97 of the 128 selected points in each iteration were already in
the dataset. This means the current acquisition/diversity logic still spends
many calls on existing exact coordinates.

Next implementation target:

```text
Hard-exclude or strongly downweight existing exact coordinates and recent
selected coordinates during candidate selection. The boundary-band cooldown is
working for its intended class of repeated points, but it is not enough to
prevent general exact-coordinate repetition.
```

## 10. Existing-Data Exclusion Implementation

Implemented on 2026-05-11:

```text
ml_phase.config:
    exclude_existing_exact = True
    existing_exclusion_decimals = 4
    recent_selection_cooldown_iters = 5
    recent_selection_cooldown_decimals = 4
    boundary_band_cooldown_decimals = 4

ml_phase.active_refine:
    keeps score_raw before post-processing;
    hard-excludes candidate coordinates already present in the exact dataset;
    cools down recently selected and boundary-band coordinates;
    writes candidate-pool exclusion counts to selection_diagnostics.json.
```

The 4-decimal rounding was chosen because an 8-decimal coordinate match is too
strict for this active-learning grid: the available physical area is still
large, and the goal is to prevent repeated exact calls at effectively identical
points rather than to impose a nonzero exclusion radius.

Next validation calculation:

```text
Continue from dataset_iter035 for 3 to 5 iterations at 128 points per
iteration. Check that selection_diagnostics.json reports
already_in_dataset_rounded = 0 or close to 0 and that the net dataset increase
is much closer to the number of training-eligible exact points.
```

## 11. Exclusion Validation Result

Analyzed output:

```text
hpc_upload_qdelta_20260511_162837/ML_Phase_128_verify_delta
```

This validation continued from `dataset_iter035` through three new iterations:

```text
iter035 through iter037
dataset_iter035 through dataset_iter038
```

Aggregate result:

```text
selected exact calls: 384
merged exact points: 384
training-eligible exact points: 378
clean trusted exact points: 350
boundary-band normal points: 28
rerun-required points after boundary-band interpretation: 6

dataset_iter035 samples: 23587
dataset_iter038 samples: 23957
net unique sample increase: 370

q expanded and confirmed: 3
q unresolved: 0
Delta refined: 322
Delta unresolved: 34
Delta unresolved requiring rerun: 6
```

The candidate-efficiency fix worked:

```text
selection_diagnostics.json reported:
    iter035 already_in_dataset_rounded: 0
    iter036 already_in_dataset_rounded: 0
    iter037 already_in_dataset_rounded: 0

independent rounded/exact comparison against the input dataset found:
    iter035 already-existing selected coordinates: 4
    iter036 already-existing selected coordinates: 3
    iter037 already-existing selected coordinates: 1
```

Before the exclusion change, the continuation run selected roughly 82 to 97
already-known dataset coordinates per 128-point iteration. After the change,
existing-dataset repeats dropped to 1-4 per iteration and the net dataset
increase became close to the selected-point budget.

Remaining issue:

```text
Six true rerun-required points remain after boundary-band interpretation. Some
are positive-Delta ambiguous points near the high-JA low-temperature boundary,
for example kT/t ~= 0.074667 and JA/t ~= 1.17925. Future acquisition should
track and cool down true rerun-required coordinates separately from
boundary-band normal points.
```

Implementation follow-up:

```text
Fix the remaining high-JA existing-coordinate leak and make
selection_diagnostics.json agree with an independent rounded-coordinate
comparison against dataset_iterXXX. Also add true-rerun cooldown using
rerun_points.csv or needs_rerun_exact so positive-Delta ambiguous coordinates
are not repeatedly selected by the standard active-learning loop.
```

## 12. Existing-Radius Exclusion Follow-Up

Implemented on 2026-05-12:

```text
ml_phase.config:
    existing_min_dist = 0.015

ml_phase.active_refine:
    computes normalized min distance from every candidate to the existing exact
    dataset;
    hard-excludes candidates with d_existing < existing_min_dist;
    records existing_min_distance, existing_distance_exclusion, and
    selected_to_existing_normalized_* diagnostics.
```

The normalized distance uses the same coordinate scaling as the existing
selected-selected diversity rule:

```text
d = sqrt((Delta kT / 0.56)^2 + (Delta JA / 2.12)^2)
```

An offline check with `dataset_iter038` found:

```text
finite candidates before exclusion: 30130
finite candidates after existing-distance exclusion: 1366
```

This remains large enough for a 128-point validation iteration, but the rule is
strong and should be validated by checking that selected points still follow the
normal/SC and high-JA boundary regions rather than drifting to irrelevant
candidate-space edges.
