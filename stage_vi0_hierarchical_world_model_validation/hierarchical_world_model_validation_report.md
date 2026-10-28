# Stage VI-0: Hierarchical Physical World-Model Validation

**Single main decision: Decision C - descriptor schema still incomplete.**

## Scope

Strictly offline local smoke validation on frozen Stage V-v3 and V-v4 cumulative CSV datasets. No exact solver, Delta-q search, active-learning loop, HPC job, or frozen-data edit was performed. All models and split-specific scalers were initialized from scratch. No historical weight, scaler, optimizer, latent, cluster, or coordinate state is transferable.

## Hierarchy

Level 1 learns physically derived SC existence together with \(\Delta\) and \(F_{SC}^{min}-F_{normal}\). Level 2 is trained only on trusted SC and learns \(|q|/\pi\). The requested \(F_{FFLO}^{min}-F_{uniform}^{min}\) field is absent from the common frozen schema and is not fabricated. Level 3 is trained only where spectral outputs are applicable and reliable. Level 4 uses only trusted gapped SC points and learns continuous Pfaffian quantities; Z2 remains evaluation-only.

## Main results

The tiled 3D spatial-block split supplies all physical branches without putting invalid normal points into topology training. The SC AUROC is approximately 0.979; median Level-1 regression R2 is 0.441, and Level-2 q R2 is 0.797. Level-3 median R2 is only 0.051: log-gap is weak and k-gap remains non-predictive. Level-4 median R2 is 0.912, driven by learnable P0/Ppi, while Pfaffian margin and epistemic-error alignment remain weak. Sequential layer-wise probe uncertainty is not uniformly monotonic.

## Critical questions

1. Thermodynamics can be learned independently at moderate spatial accuracy, with strong SC classification.
2. Conditioning on trusted SC makes q learnable, but branch-energy improvement cannot be tested because deltaF_UF is absent.
3. Removing invalid topology points makes P0/Ppi regression meaningful, but topology uncertainty is not consistently error-aligned.
4. Hierarchical masking improves Level-3 NLL/calibration, while Level-4 improvement is run-dependent rather than universal.
5. The poorly learnable observables are k-gap, log-gap, and Pfaffian margin.
6. The missing deltaF_UF is a descriptor/logging problem; the remaining spectral failures persist with 639-694 valid spatial test points and are not explained by mask sparsity alone.

## Decision

Decision C - descriptor schema still incomplete is the only decision. The hierarchical separation fixes the principal not-applicable/missing-value mixing problem, but the full schema is not validated because `deltaF_UF` is unavailable and level-wise uncertainty/replay gates are not uniformly passed. Do not enter an EIG pilot. Preserve only schema, masks, architecture, loss design, source code, and tests; reinitialize and refit everything in any future strict cold-start study.
