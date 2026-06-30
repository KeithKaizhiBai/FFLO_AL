# Trusted Surprise StopController Decision Log

## 2026-06-18: Add Trusted Surprise as an Explicit StopController Mode

Decision:

```text
Keep the existing all-selected label-surprise calculation as the default
StopController mode and add an opt-in trusted surprise mode.  Record hard-risk
surprise as a numerical-frontier diagnostic rather than using it directly as a
main-phase convergence veto.
```

Reason:

```text
The full-loop counterfactual replay shows that the final all-selected surprise
is 47/256, but trusted surprise is 0/137 and all strict last-five surprise
points are rerun-required.  The current all-selected metric is useful as an
acquisition-selected difficulty diagnostic, but it mixes clean exact-label
errors with unresolved numerical-frontier points.
```

Consequences:

```text
The default `all_selected` mode preserves backward compatibility and must
replay the existing full-loop StopController exactly.  The opt-in `trusted`
mode can be used for validation packages and future production only after its
denominator floor is satisfied.  Boundary coverage remains a separate stop
condition, and the hard-risk frontier must still be reported and audited.
```

No changes are made to:

```text
phase criterion
Delta tolerance
final ambiguity tolerance
rankcap_k3 local-refinement target construction
q-window expansion
acquisition weights
candidate domain
surprise tolerance
required_pass_count
patience
boundary thresholds
```
