# Phase q-window and Delta-refinement audit

Status: input points and HPC helper scripts prepared; exact reruns are pending.

## Input Counts

```json
{
  "qwindow_sensitive_points": 342,
  "delta_sensitive_points": 96,
  "clean_control_points": 20,
  "combined_unique_points": 345
}
```

## Scope

This is an audit-only production-oriented numerical robustness update. It does
not modify acquisition, StopController, NN training, or the active-learning
dataset.

## Required interpretation rules

- q-window expansion is not required to prove superconductivity once a
  lower-free-energy positive-Delta superconducting state is already found.
- q-window expansion is required to check branch identity, q_opt stability,
  boundary robustness, and future topology classification.
- eta response has already been downgraded to response-extraction pathology
  unless `eta_response_valid=True`.
