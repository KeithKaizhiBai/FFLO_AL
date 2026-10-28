# Decision log

Decision C - descriptor semantics incomplete or inconsistent.

The frozen inputs and field meanings are reproducible, but the proposed
descriptor/mask system does not yet support reliable spatial extrapolation:
topology-valid spatial holdouts are too sparse, label-free phase recovery is
poor, epistemic uncertainty is not uniformly error-aligned, sequential probe
uncertainty is non-monotonic, and historical uncertainty reductions are not
uniformly distinguishable from random/redundant controls and retraining noise.

Do not transfer any learned development state or start a strict cold-start EIG
pilot. Revise the descriptor/mask schema and spatial coverage diagnostics first.
