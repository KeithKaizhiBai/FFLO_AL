# Stage VI-0.2: Hierarchical World Model v2

**Single main decision: Decision D - insufficient frozen artifacts.**

## Artifact gate

The frozen V3/V4 final CSVs and exact merged NPZ shards were inspected directly. The 131-field exact schema contains gap, k-at-gap, Pfaffian outputs, and thermodynamic/oracle metadata, but no fixed-k spectrum, eigenvalue curve, DOS proxy, near-zero spectral-minimum count, or spectral curvature. Free-energy basin counters are not spectral fingerprints and were not repurposed. Therefore PCA, autoencoder, and low-energy-moment fingerprint tests cannot be performed from the allowed frozen inputs.

## Executed smoke

All weights and split scalers were initialized from scratch. Level 1, Level 2, and conditional Level 4 were revalidated on spatial blocks, mu slices, corners, future holdout, and bidirectional cross-run transfer. Level 3 compared the old joint `[log_gap,k_gap]` descriptor with the only legal available-v2 ablation, `[log_gap]`. Uncertainty-error Spearman, risk-coverage AUC, interval coverage, 10k level-wise probe maps, sequential replay, and post-hoc hierarchy reconstruction were computed.

## Interpretation

Removing k-gap avoids forcing a non-smooth argmin output, but log-gap-only is not a spectral fingerprint and cannot satisfy the requested redesign. Mean spatial SC AUROC is 0.977; median spatial R2 is 0.432 for Level 1, 0.799 for Level 2, 0.086 for the log-gap-only Level 3 ablation, and 0.908 for Level 4. Cross-run q and topology transfer remains strong, but log-gap-only does not consistently improve on the old log-gap field. Pfaffian-margin and uncertainty calibration are mixed. Level-wise replay uncertainty is not uniformly monotonic. The available evidence cannot establish that the model knows what it does not know.

## Decision

Decision D - insufficient frozen artifacts is the only decision. The blocker is missing frozen spectral artifacts, not a request for new exact calculations within this task. Do not run the five-member full-local configuration or EIG. First export already-computed fixed-grid spectra or physically defined low-energy spectral moments, if they exist outside the current package; otherwise redesign future exact-output logging. No development learned state is transferable.
