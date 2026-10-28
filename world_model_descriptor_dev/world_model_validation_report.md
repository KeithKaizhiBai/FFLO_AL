# Frozen V3/V4 descriptor and probabilistic world-model validation

**Main decision: Decision C - descriptor semantics incomplete or inconsistent.**

## Scope

Strictly offline local development on frozen Stage V-v3/V-v4 cumulative CSV artifacts. No BdG solver, Delta-q search, exact calculation, active-learning continuation, HPC job, or frozen-data modification was performed. Development models, scalers, optimizer state, selected coordinates, and all learned state are explicitly forbidden from future strict cold-start production.

## Hardware and device

Apple M2 MacBook Air, 8 CPU cores, 16 GB RAM, macOS 15.7.7, PyTorch 2.13.0. Selected device: `cpu`. MPS available: False; MPS stable on the fixed smoke split: False.

## Descriptor semantics

The label-free output space uses delta, abs_q, deltaf, log_gap, k_gap, p0, ppi, pf_margin with fixed physical transforms. Phase, topology class, proposal source, action family, reward, iteration, and graph/candidate identifiers never enter the model input or continuous training targets. Normal-state q is masked as not applicable rather than filled with zero. Second-basin fields were not sufficiently available in the final common raw schema and were excluded.

## Experiments

The pipeline evaluates constant, local kNN, deterministic MLP, and masked heteroscedastic deep-ensemble models. Tests cover 3D spatial blocks, fixed-mu slices, a parameter corner, future rows, bidirectional V3/V4 transfer, interval calibration, epistemic-error association, fixed Sobol probe uncertainty, reduced sequential replay, and retrospective realized uncertainty reduction. Full numerical values are authoritative in the companion CSV files.

## Interpretation

Mean prediction is learnable on future/slice tests for several core observables, but strict spatial-block extrapolation is weak and topology-valid holdout counts are only 8 (V3) and 4 (V4) in the selected block. Calibration and epistemic-error ranking are not uniformly reliable. Probe uncertainty is non-monotonic with budget. Retrospective historical reductions are compared with random, spatially redundant, and retraining-noise controls, and are not uniformly distinguishable.

## Decision

Decision C - descriptor semantics incomplete or inconsistent is the only main decision. Revise the observable/mask schema and spatial coverage diagnostics before improving likelihood calibration or attempting strict cold-start EIG. No historical learned state may be transferred.
