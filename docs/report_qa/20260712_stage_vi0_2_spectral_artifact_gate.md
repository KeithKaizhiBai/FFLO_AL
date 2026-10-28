# Stage VI-0.2 spectral artifact gate Q&A

## Why is the decision D rather than C?

The proposed v2 schema can be written, but its defining new Level-3 information
is absent from the allowed frozen artifacts. This prevents the requested
fingerprint/PCA/autoencoder or low-energy-moment validation, matching the
task's Decision D rule.

## Can free-energy local-minimum counters be used as spectral moments?

No. They describe Delta-q free-energy search structure, not quasiparticle
spectral minima, DOS, or spectral curvature.

## Did the log-gap-only ablation help?

Not consistently. Its spatial R2 is about 0.098 (V3) and 0.074 (V4), and it
does not robustly improve the old log-gap field. It is not a fingerprint.

## What is allowed next?

Locate/export already-computed spectral artifacts if they exist. Otherwise a
future logging change requires separate authorization. No full-local, EIG, or
new exact calculation is authorized by this audit.
