# Discovery Selection Diagnostics

Date: 2026-05-19

Context:

The latest discovery-mode active-learning run is stored under

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed
```

The intended run is

```text
active_runs/active_boundary_discovery_512seed_256x50
```

The directory also contains an older mistaken default-run directory

```text
active_runs/active_boundary_loop_512x50_acquisition_only
```

which should not be mixed with the discovery analysis.

Main findings:

1. The discovery run stopped at `iter065`, not at `iter099`.
2. The final exact dataset is `dataset_iter066.npz` with 17061 samples.
3. The stop reason is `converged_main_phase_boundaries`.
4. StopController passed 4 of 5 main convergence conditions for 4 consecutive
   iterations.
5. `boundary_coverage_p95` was slightly above threshold:

```text
boundary_coverage_p95 = 0.00627166
coverage_tol = 0.00625
```

Last-iteration selection:

```text
selected points = 256
selection_source = acquisition_stochastic
selection_pool = acquisition
midpoint selection = disabled
candidate_domain_mode = full
```

The last selected batch does not show clear main-boundary focusing.  Measured
against the extracted normal/SC and uniform-SC/FFLO boundaries:

```text
fraction within normalized distance 0.05 of either main boundary = 44 / 256
fraction within normalized distance 0.10 of either main boundary = 69 / 256
fraction within normalized distance 0.20 of either main boundary = 130 / 256
median distance to either main boundary = 0.1980
mean distance to either main boundary = 0.2415
```

A random sample of 256 points from the full grid would have roughly

```text
expected count within distance 0.05 = 48
mean distance = 0.2169
median distance = 0.1547
```

Thus the last selected batch is not more boundary-focused than a random full
grid sample; by mean and median distance it is slightly farther from the main
boundaries.

Interpretation:

The issue is not that stochastic selection ignored a strong boundary-focused
score.  The selected trace shows that the acquisition score itself remains high
over broad non-boundary regions, especially high-\(J_A\) normal regions.

For `iter065`, selected-point correlations with distance to the main
thermodynamic boundaries were approximately

```text
corr(A0_main, boundary distance) = +0.36
corr(delta_boundary_score, boundary distance) = +0.51
corr(extrapolation_risk_score, boundary distance) = +0.41
corr(U_reg_phase, boundary distance) = -0.51
corr(P_SC, boundary distance) = -0.46
```

This means that the current `A0_main` is partly driven by broad
`delta_boundary_score` and exploration terms away from the actual extracted
main boundaries.  The last selected batch visually resembles random sampling
because the score landscape is broad and noisy rather than sharply localized on
the normal/SC and uniform-SC/FFLO boundaries.

Practical consequence:

For future discovery-mode active learning, improving boundary focus requires
changing the acquisition or sampling policy, not merely increasing the number
of iterations.  Candidate options include:

```text
1. add an explicit ML-predicted phase-boundary distance term;
2. sharpen the stochastic sampler, for example sampling_power > 1;
3. downweight broad delta_boundary_score far inside high-confidence normal
   regions;
4. make exploration quota/score adaptive and decay it after the phase map
   stabilizes;
5. report selected-point distance-to-main-boundary diagnostics every iteration.
```

This note is diagnostic and does not change the canonical physical definitions
or numerical rules.
