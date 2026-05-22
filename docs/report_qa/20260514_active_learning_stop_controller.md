# Active-Learning Stop Controller

## Question

Why is candidate exhaustion no longer a good active-learning stopping
criterion after switching to soft repulsion?

## Short Answer

Soft repulsion deliberately avoids deleting every nearby candidate. It lowers
the score of redundant candidates while keeping the acquisition landscape
continuous. Therefore the loop may always be able to choose another batch even
after the main phase-boundary information has saturated. The stop decision
should instead be based on exact-oracle feedback and boundary stability.

## Technical Notes

The implemented stop controller evaluates convergence after each exact BdG
merge and trusted append. It does not modify acquisition scores and it does not
change the selected candidates. It only reads the iteration outputs and decides
whether the loop should continue.

The current metrics are:

```text
phase_map_change
boundary_shift_normal_sc
boundary_shift_uniform_fflo
label_surprise_rate
selected_A0_ratio
q_edge_trigger_rate
rerun_required_rate
boundary_coverage_p95
```

The main boundary targets are normal/SC and uniform_SC/FFLO. The \(\eta=0\)
response sign boundary is not a thermodynamic phase boundary and is not used
as a main stop condition. Topology is also excluded until the pointwise exact
oracle produces topological labels.

The convergence rule is:

```text
after min_iterations:
    at least 5 of C1..C7 must pass
    C6 q-edge/rerun rates must pass
    C7 main-boundary coverage must pass
    the combined pass must persist for patience consecutive iterations
```

The q-edge/rerun and boundary-coverage conditions are mandatory gates because
stable-looking ML maps are not sufficient if exact calculations still hit
q-window failures or if the main boundaries remain poorly covered by exact
data.

## Report Use

Use this note to explain why the acquisition-only active-learning loop stops
based on exact-data convergence and numerical-risk saturation, not on whether
the soft acquisition score can still produce candidate points.
