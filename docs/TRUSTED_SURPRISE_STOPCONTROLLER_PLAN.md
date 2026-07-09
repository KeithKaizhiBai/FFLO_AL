# Trusted Surprise StopController Plan

## Goal

Split main-phase convergence from the numerical hard-risk frontier without
changing physical phase definitions, exact-oracle logic, acquisition weights,
rankcap_k3 local refinement, or any existing convergence tolerance.

## Evidence

The trusted-surprise counterfactual report shows:

```text
final all-selected surprise = 47 / 256 = 0.183594
final trusted surprise      = 0 / 137 = 0.000000
final hard-risk surprise    = 47 / 119 = 0.394958
last-five surprise from rerun-required points = 205 / 205
trusted-only counterfactual stop iteration = 17
```

The current all-selected StopController reconstruction was exactly reproduced.
Therefore the code change must preserve `all_selected` behavior as the default
and make trusted surprise an explicit opt-in mode.

## Scope

Allowed changes:

```text
StopController surprise metric selection
StopController metadata schema
report and replay diagnostics
unit and replay tests
HPC packaging scripts
```

Forbidden changes:

```text
thermodynamic phase criterion
Delta refinement trigger tolerance
final ambiguity tolerance
stable-normal admission logic
rankcap_k3 local-refinement logic
q-window incremental expansion
acquisition weights
candidate domain
random seed
batch size
phase-map-change tolerance
boundary-shift tolerance
label-surprise tolerance
boundary-coverage tolerance
required_pass_count
patience requirement
max-iteration rule
eta response logic
topology logic
```

## Surprise Layers

`label_surprise_all_selected`:

```text
All matched selected points with predicted_phase_before_exact and exact phase.
Role: backward-compatible acquisition-selected difficulty diagnostic.
```

`label_surprise_trusted`:

```text
trusted_exact == true
training_eligible_exact == true
rerun_required == false
q_unresolved == false
delta_unresolved == false
```

`q_expanded == true` is not excluded if the point is trusted, training eligible,
non-rerun, q-resolved, and delta-resolved.

`label_surprise_hard_risk`:

```text
rerun_required == true
or trusted_exact == false
or training_eligible_exact == false
or q_unresolved == true
or delta_unresolved == true
```

This is a numerical-frontier diagnostic, not a clean main-phase prediction
error.

## Stop Mode

Add:

```text
stop_surprise_mode = all_selected | trusted
trusted_surprise_min_denominator = 64
trusted_surprise_min_fraction = 0.25
```

Default:

```text
stop_surprise_mode = all_selected
```

Trusted gate:

```text
C4_label_surprise_rate =
    label_surprise_trusted < surprise_tol
    and trusted_surprise_denominator_valid
```

The `surprise_tol`, `required_pass_count`, and `patience` values remain
unchanged.

## Metadata

Each stop metrics JSON must retain:

```text
label_surprise_rate
```

as the all-selected backward-compatible field.

It must also record:

```text
label_surprise_all_selected
label_surprise_trusted
label_surprise_hard_risk
label_surprise_selected_for_gate
n_surprise / n_denominator for each surprise layer
trusted_surprise_denominator_valid
trusted_fraction_selected
hard_risk_fraction_selected
main_phase_converged
numerical_frontier_status
publication_ready
publication_ready_reason
```

## Replay Gate

Before packaging:

```text
scripts/replay_stopcontroller_surprise_modes.py
```

must prove:

```text
all_selected mode exactly reproduces saved label_surprise_rate
all_selected mode exactly reproduces passed_condition_count
all_selected mode exactly reproduces patience_counter
all_selected mode exactly reproduces stop state
trusted mode final trusted surprise = 0 / 137
trusted mode earliest stop iteration = 17
```

## HPC Validation Package

Create:

```text
hpc_packages/stopcontroller_all_selected_baseline_snapshot/
hpc_packages/stopcontroller_trusted_surprise_v1/
```

The baseline package replays the existing full loop and should not submit exact
BdG work by default.  The trusted package runs the same robust_incremental
rankcap_k3 configuration with:

```text
STOP_SURPRISE_MODE=trusted
TRUSTED_SURPRISE_MIN_DENOMINATOR=64
TRUSTED_SURPRISE_MIN_FRACTION=0.25
```

The package must exclude `gpuh01` in Slurm submission scripts.
