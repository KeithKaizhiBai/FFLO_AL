# Tail Surprise Continuation Return

Question:

```text
The tail-surprise continuation result has been downloaded and unpacked.  Does
it positively support convergence, and should the project run another full loop
or move to the next major stage?
```

Answer:

```text
The return positively supports main phase-map and main-boundary convergence
under the trusted-surprise StopController gate.  The final evaluated iteration
is iter034, producing dataset_iter035 with 7434 total samples.  StopController
reports stop=true, stop_reason=converged_main_phase_boundaries,
convergence_pass=true, passed_condition_count=5/5, and patience_counter=4/4.
```

Key numbers:

```text
phase_map_change = 0.0016287 < 0.002
boundary_shift_normal_sc = 0.0041667 <= 0.0041667
boundary_shift_uniform_fflo = 0
boundary_coverage_p95 = 0.0046875 < 0.00625
trusted surprise = 0/127 = 0
all-selected surprise = 75/256 = 0.29296875
hard-risk surprise = 75/129 = 0.581395
rerun_required_count = 110
```

Interpretation:

```text
The clean/trusted exact-label layer no longer blocks main-boundary convergence.
The old all-selected surprise remains high because the acquisition-selected
batch still contains many hard-risk/rerun frontier points.  Those points should
not be treated as clean phase-label errors, but they also cannot be ignored for
publication-ready numerical reliability.
```

Decision:

```text
Do not run another full main active-learning loop just to lower the historical
all-selected surprise metric.  The next step is a targeted hard-risk
boundary-impact audit/cleanup, followed by the next major physics/report stage
using the trusted-gate converged rankcap_k3 phase map.
```

Report artifacts:

```text
reports/rankcap_k3_tail_surprise_continuation_return/
    rankcap_k3_tail_surprise_continuation_return.md
    rankcap_k3_tail_surprise_continuation_return.pdf
    decision_log.md
    tables/
    figures/
```
