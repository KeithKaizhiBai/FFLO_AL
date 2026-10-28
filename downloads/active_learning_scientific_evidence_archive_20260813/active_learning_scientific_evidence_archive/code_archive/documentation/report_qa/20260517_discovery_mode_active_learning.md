# Discovery-Mode Active Learning

## Question

Why do we need a discovery mode in addition to warm-start refinement?

## Short Answer

Warm-start refinement asks how efficiently the ML scheduler improves an already
known exact phase diagram. Discovery mode asks a different question: can the
active-learning loop discover the main phase structure starting only from a
small random set of exact BdG calculations?

## Technical Notes

Discovery mode uses

```text
run_mode = discovery
candidate_domain_mode = full
initialization = random_grid
selection_mode = stochastic
finite_t_band_width = disabled
```

The initial exact-call batch is sampled randomly from the full rectangular
candidate grid. It does not use the large warm-start exact dataset, the
finite-temperature prior boundary, or any fixed regional quota.

After the random seed exact calls return, the loop trains the usual regression
ensemble and phase classifier. The acquisition score remains

$$
A_{0,\mathrm{main}}
=A_{\mathrm{phase}}+A_{\mathrm{numerical}}+A_{\mathrm{explore}}.
$$

The selected batch is drawn stochastically with probability proportional to the
corrected acquisition score after unseen-point masking, observation repulsion,
and dynamic batch repulsion:

$$
p_i \propto
\left[
A_{0,\mathrm{main}}(x_i)
R_{\mathrm{obs}}(x_i)
R_{\mathrm{batch}}(x_i)
\right]^\gamma .
$$

This keeps ML/acquisition responsible for deciding where the next exact BdG
calls should go, while avoiding a deterministic top-\(k\) collapse onto one
narrow region.

The main stop rule now uses only the main phase-map and thermodynamic-boundary
conditions:

```text
phase_map_change
normal/SC boundary shift
uniform_SC/FFLO boundary shift
label_surprise_rate
boundary_coverage_p95
```

Selected acquisition-score saturation, q-edge rate, and rerun-required rate are
diagnostics or cleanup warnings. They should not prevent the main
phase-boundary loop from stopping once the main phase map has converged.

## Report Use

Use this note to explain the difference between a discovery benchmark and
warm-start boundary refinement. The hidden dense exact grid, when used, is an
offline evaluator only; it must not enter training, acquisition, or online stop
decisions.
