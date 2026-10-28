# 2026-05-20 discovery run analysis

Run checked:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter
```

## Run status

- Run mode: discovery.
- Candidate domain: full rectangular dense grid.
- Selection mode: stochastic.
- Initial dataset: random grid seed, 512 exact points.
- Latest completed iteration: 49.
- Final dataset: `dataset_iter050.npz`.
- Final exact samples: 12970.
- Stop reason: `max_iterations`.
- The run did not satisfy the main convergence rule before the hard stop.

Latest StopController metrics at iteration 49:

```text
phase_map_change = 0.00341257
boundary_shift_normal_sc = 0.00416667
boundary_shift_uniform_fflo = 0.003125
label_surprise_rate = 0
boundary_coverage_p95 = 0.00658808
```

Only label surprise and uniform/FFLO boundary shift passed.  The normal/SC
boundary shift is effectively at the tolerance but fails because the test is a
strict `<` comparison, while phase-map change and boundary coverage remain above
their configured tolerances.

## Dataset composition

Final exact phase counts:

```text
normal: 11004
uniform_SC: 307
FFLO: 1659
```

The dataset is strongly normal-dominated.  This is not necessarily a bug in the
oracle; it reflects the full-domain discovery sampling and the physical size of
the normal region in the current candidate rectangle.

## Numerical oracle quality

Latest exact batch, iteration 49:

```text
selected points = 256
trusted exact = 250 / 256
training eligible = 256 / 256
q_edge_hit = 0 / 256
q_expanded = 0 / 256
q_unresolved = 0 / 256
needs_rerun_exact = 0 / 256
delta_boundary_ambiguous = 6 / 256
delta_refined = 220 / 256
delta_unresolved = 6 / 256
```

Thus the late-iteration numerical q-window and rerun diagnostics look clean.
The remaining ambiguity is mainly finite-resolution Delta boundary-band
metadata, not q-window failure.

## Selection behavior

The updated workflow does use acquisition-only stochastic selection, not
midpoint selection and not prior-band refinement.  However, the active pool is
still very large.

Latest iteration:

```text
candidate_pool_total = 77361
candidate_pool_finite_after_exclusion = 49061
active_pool_size = 49061
active_pool_quantile_used = 0.9
active_pool_rel_to_p95 = 0.3
A0 p90 = 1.05287
A0 p95 = 1.07891
relative p95 threshold = 0.323673
N_eff / active_pool_size = 0.384324
selected_A0_mean / unseen_A0_mean = 1.27945
selected_boundary_band_fraction = 0.0703125
random_baseline_boundary_band_fraction = 0.046875
```

The selected points have higher acquisition value than the unseen average, so
the acquisition score is influencing the sampler.  But the active pool remains
about 49k points, and only about 7% of the latest selected points lie in the
configured boundary band.  The random baseline is about 4.7%, so there is some
boundary enrichment, but it is weak.

The likely reason is the current active-pool rule:

```text
A0 >= quantile_threshold OR A0 >= active_pool_rel_to_p95 * p95(A0)
```

With `active_pool_rel_to_p95 = 0.3`, the relative threshold is much lower than
the p90 threshold.  Because the rule uses OR, many candidates enter the active
pool even though they are not in the top 10% by A0.  Sampling over this large
pool can visually resemble broad stochastic coverage.

## Report status

The report compiled successfully:

```text
active_learning_phase_boundary_report.pdf, 11 pages
```

The report correctly identifies:

- discovery mode,
- stochastic selection,
- full candidate domain,
- no warm-start samples,
- no finite-T band mask,
- `dataset_iter050.npz`,
- stop reason `max_iterations`,
- Boundary F1 as unavailable because hidden-ground-truth boundary evaluation is
  not implemented.

The q-window and Delta refinement explanation is present in the report.

## Interpretation

This run is a valid discovery-mode run, but it has not demonstrated strong
boundary-focused active learning yet.  The data suggest that the stochastic
sampler is acquisition-biased but not selective enough.  The next algorithmic
change should tighten high-information pool construction or make the
relative-to-p95 condition less permissive.  Options include:

- use quantile gating alone for the first test;
- change the OR condition to a stricter condition;
- raise `active_pool_rel_to_p95`, for example from 0.3 to 0.7 or 0.8;
- raise `active_pool_quantile` to 0.95;
- increase `sampling_power` after checking that the pool is already narrow
  enough.

The numerical oracle does not appear to be the bottleneck in the latest batch.
