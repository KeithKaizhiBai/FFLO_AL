# Phase-II Robust Active-Learning Convergence, Numerical Reliability, and Optimization Report

## Executive Summary

Phase-II converts a sequence of small validation reports into one reproducible
story: starting from exact seed labels, the active-learning loop repeatedly
trained a surrogate, selected a full-acquisition batch, called the robust exact
oracle, appended trusted labels, and stopped only after the main phase map and
main thermodynamic boundaries stabilized.

Final frozen state:

```text
main_phase_converged = True
publication_boundary_audit = pass
publication_ready_for_main_phase_map = True
frozen dataset = dataset_iter035
total samples = 7434
normal / uniform_SC / FFLO = 1867 / 715 / 4852
trusted surprise = 0/127
hard-risk surprise = 75/129
targeted rerun count = 0
```

The hard-risk frontier remains active and must be shown as an uncertainty
layer.  It does not invalidate the main thermodynamic phase map under the
publication-grade boundary-impact audit.

## Random-Seed Active-Learning Loop

![RankCapK3 active-learning loop](figures/phase2_from_seed_loop.png)

The loop is intentionally asymmetric: acquisition proposes points, but only the
exact BdG oracle assigns labels.  The surrogate is a scheduler rather than a
physics replacement.

## Stage Timeline

![Optimization timeline](figures/phase2_optimization_timeline.png)

| Stage | Objective | Main result |
|---|---|---|
| Initial exact grid | Reference phase structure | Coarse exact map available; active learning need is boundary refinement |
| Acquisition development | Avoid deep-normal oversampling while retaining discovery | Full acquisition retained for FFLO/uniform-SC coverage; simple-phase not used for final discovery |
| Robust exact oracle | Separate phase labels from numerical reliability | Trusted, training-eligible, rerun-required, q/Delta unresolved layers recorded |
| Incremental q-window | Reduce repeated q-window work | Incremental strip computation retained with fallback and reliability audit |
| Local-refinement audit | Find why optimized variants timed out | Mandatory overflow produced approximately 85 boxes in failed variants |
| RankCapK3 | Bound selected refinement targets | Fixed-point acceptance passed; selected targets <= 3 with physics-equivalent labels |
| One- and five-iteration validation | Check real AL behavior before full run | RankCapK3 remained stable and reduced local boxes in active-loop context |
| Full loop | Run production discovery with optimized oracle | 30 acquisition batches completed; main phase nearly stable but all-selected surprise remained high |
| Trusted-surprise tail | Use trusted labels as formal main-phase surprise gate | Stopped at iter034; trusted surprise 0/127; dataset_iter035 frozen |
| Hard-risk publication audit | Test whether hard-risk frontier can move main boundaries | Decision A; targeted rerun count 0; publication boundary audit pass |


## Acquisition and Oracle Split

![Acquisition and oracle responsibility split](figures/phase2_acquisition_oracle_flow.png)

Acquisition evolved through B-delta gating, active-pool narrowing, stochastic
sampling, observation repulsion, and batch repulsion.  The final run used full
acquisition because it better retained uniform-SC and FFLO discovery coverage
than a simple phase-only boundary profile.

The robust oracle records phase labels separately from reliability metadata:
`trusted_exact`, `training_eligible_exact`, `rerun_required`, `q_unresolved`,
and `delta_unresolved`.  This separation is what allowed the final
trusted-surprise StopController gate to distinguish main phase-map convergence
from the numerical hard-risk frontier.

## Local-Refinement Optimization

The failed optimized variants were not physically disproven; they suffered a
target-construction bug where mandatory branches could overflow to roughly
85 local boxes.  The corrected `rankcap_k3` path performs basin-level risk
annotation and hard caps total refined basins at three.

```text
baseline local boxes = 6.0
rankcap mean local boxes = 2.79296875
rankcap max local boxes = 3
local-refinement runtime reduction = 53.48%
point-total runtime reduction = 49.92%
```

![Local-box distribution](figures/local_box_distribution.png)

## Full Loop and Tail Continuation

```text
phase_map_change = 0.001628727653468802
normal/SC boundary shift = 0.004166666666666666
uniform/FFLO boundary shift = 0.0
boundary_coverage_p95 = 0.0046875000000001586
```

![Dataset growth and phase counts](figures/dataset_growth_and_phase_counts.png)

![Surrogate metric curves](figures/surrogate_metric_curves.png)

![Runtime curve](figures/runtime_curve.png)

## Final Phase Map

![Clean final phase map](figures/final_phase_map_clean.png)

![Final phase map with uncertainty](figures/final_phase_map_with_uncertainty.png)

![Hard-risk uncertainty band](figures/hard_risk_uncertainty_band.png)

## Hard-Risk Boundary-Impact Audit

The publication audit reconstructs 129 hard-risk points.  Of these, 110 are
rerun-required and 19 are non-rerun hard-risk through untrusted or
training-ineligible metadata.  Counterfactual local flips, continuous clusters,
phase-constrained flips, Hausdorff diagnostics, affected arc length, and
topology checks do not identify a cluster that moves the main thermodynamic
boundaries beyond tolerance.  The targeted rerun list is empty.

![Tail surprise layers](figures/tail_surprise_layers.png)

## Final Decision

Freeze `dataset_iter035`, the `rankcap_k3` production oracle settings, and the
trusted-surprise StopController configuration for the Phase-II main phase-map
result.  Do not launch another full active-learning loop for this result.

Recommended next physics stage: branch-resolved topology classification,
hidden-ground-truth evaluation, multi-seed benchmarking, and final publication
figure polishing.

## Do-Not-Claim List

1. Do not claim the hard-risk frontier has disappeared.
2. Do not treat provisional hard-risk labels as definitive phase labels.
3. Do not claim eta anomalies have all become physical effects.
4. Do not treat topology reference curves as pointwise topology labels.
5. Do not present this single-seed run as a complete multi-seed benchmark.
6. Do not claim hidden-ground-truth benchmarking is complete.
7. Do not treat synchronized global stress-test islands as real boundary instability.
