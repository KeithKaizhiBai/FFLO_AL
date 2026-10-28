# Frozen descriptor/world-model validation Q&A

## Is the fixed descriptor ready for strict cold-start EIG?

No. Decision C applies. Frozen field semantics are reproducible, but strict
spatial topology holdouts are too sparse and several important continuous
outputs extrapolate poorly. Ensemble uncertainty is not consistently aligned
with error or monotonically reduced by additional budget.

## Did V3/V4 learned state enter the proposed production state?

No. This was development-only. Data, scalers, weights, checkpoints, optimizer
state, latent/cluster state, coordinates, reward history, and replay buffers
are explicitly denied. Only schemas, fixed transforms, architecture/loss
source, and unit tests are allowlisted.

## Was retrospective information value distinguishable?

Only the V4 1024-to-3000 interval clearly exceeded the tested controls. The
remaining historical intervals were not distinguishable from random or
spatially redundant controls and retraining noise, so EIG viability is not
established.

## Where is the report?

`world_model_descriptor_dev/world_model_validation_report.pdf` and the
synchronized package `output/pdf/world_model_descriptor_validation/`.
