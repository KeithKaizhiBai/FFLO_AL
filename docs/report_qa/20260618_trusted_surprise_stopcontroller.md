# Trusted Surprise StopController Implementation

## Question

How should the active-learning StopController treat late-stage selected-batch
surprise when the remaining surprise is concentrated in rerun-required
normal-to-FFLO boundary points?

## Answer

The implementation now separates three quantities:

```text
label_surprise_all_selected
label_surprise_trusted
label_surprise_hard_risk
```

The old `label_surprise_rate` field remains a backward-compatible alias for
`label_surprise_all_selected`.  This preserves the previous StopController
behavior when `stop_surprise_mode=all_selected`, which remains the default.

The opt-in trusted gate uses:

```text
trusted_exact == true
training_eligible_exact == true
rerun_required == false
q_unresolved == false
delta_unresolved == false
```

It does not exclude a point only because `q_expanded == true`.  If a q-expanded
point is trusted, training eligible, non-rerun, q-resolved, and delta-resolved,
it remains part of trusted surprise.

Hard-risk surprise uses:

```text
rerun_required == true
or trusted_exact == false
or training_eligible_exact == false
or q_unresolved == true
or delta_unresolved == true
```

This is a numerical-frontier diagnostic, not a clean phase-label prediction
error.

## Evidence

Replay of the saved rankcap_k3 full-loop StopController history gave:

```text
all_selected reconstruction = exact pass
all_selected final surprise = 47 / 256 = 0.18359375
trusted final surprise = 0 / 137 = 0.0
trusted earliest counterfactual stop iteration = 17
remaining blocker under trusted mode = boundary_coverage_p95
```

The replay did not rerun exact BdG calculations.

## Implementation Notes

New StopConfig fields:

```text
stop_surprise_mode = all_selected | trusted
trusted_surprise_min_denominator = 64
trusted_surprise_min_fraction = 0.25
```

The trusted C4 gate requires both:

```text
label_surprise_trusted < surprise_tol
trusted_surprise_denominator_valid == true
```

The existing `surprise_tol`, `required_pass_count`, `patience`, phase-map
tolerance, boundary-shift tolerance, and boundary-coverage tolerance were not
changed.

## HPC Packages

Generated packages:

```text
hpc_packages/stopcontroller_all_selected_baseline_snapshot.tar.gz
hpc_packages/stopcontroller_trusted_surprise_v1.tar.gz
```

Both packages exclude `gpuh01` by default and include LF bash scripts.  The
trusted package uses:

```text
STOP_SURPRISE_MODE=trusted
TRUSTED_SURPRISE_MIN_DENOMINATOR=64
TRUSTED_SURPRISE_MIN_FRACTION=0.25
```

## Caveats

This change does not solve the hard-risk numerical frontier.  It separates
main-phase convergence from provisional or unresolved exact-label diagnostics.
The trusted-surprise short validation still needs to run on HPC before using
trusted mode as a production default.
