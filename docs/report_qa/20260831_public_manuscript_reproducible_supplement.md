# Public manuscript and reproducible Supplement

## What changed?

The manuscript continuation now presents the scientific problem and results in
public physics language. Internal gate, registry, hash, and audit vocabulary is
kept in the machine-readable workflow record instead of the scientific prose.
Implementation detail and dense numerical evidence are reconstructed in a
self-contained Supplement.

## What efficiency can be claimed?

The frozen candidate grid contains (241\times321=77{,}361) locations. The
endpoint contains 6,880 unique exact coordinates, so the sampled fraction is
(6{,}880/77{,}361=0.0889337), or 8.893%. The corresponding exact-location
reduction is (77{,}361/6{,}880=11.244). This is a count-based reduction, not a
wall-time speedup. Boundary enrichment is not recoverable because the export
does not contain a candidate-wide boundary denominator.

## What method details were recovered?

The production surrogate uses separate five-member PyTorch regression and
classification ensembles. Each member is a two-hidden-layer, 64-unit ReLU MLP.
Training uses Adam with learning rate (10^{-3}), weight decay (10^{-5}),
batch size 512, 240 epochs, and a 15% stratified validation split. Regression
uses mean-squared error and classification uses cross entropy. The Supplement
also records the exact implemented acquisition decomposition, microbatch
selection, reliability hierarchy, numerical stopping diagnostics, topology
criterion, sensitivity checks, and Stage-IV limitations.

## What remains a human judgment?

The reviewer must approve the public wording of the exact-location efficiency
claim, the Stage-III reconstruction caveat, the Stage-IV finite-domain scope,
and whether any text-only cleanup of internal labels in existing figure artwork
should be authorized later. No figure artwork was modified in this run.

```text
GATE_PUBLIC_MANUSCRIPT_AND_SUPPLEMENT_HUMAN_REVIEW = HUMAN_WAIT
PHYSICS_RECOMPUTED = false
EXTERNAL_RETRIEVAL = false
```
