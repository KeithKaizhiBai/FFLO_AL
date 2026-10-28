# Stage VI-0 hierarchical world-model Q&A

## Did hierarchical masking solve the unified-model problem?

Partly. It removes not-applicable normal q and invalid topology values from
their losses. SC classification, q, and P0/Ppi become meaningfully learnable.
It does not make spectral quantities, Pfaffian margin, or ensemble uncertainty
uniformly reliable.

## What are the key spatial results?

Mean SC AUROC is 0.9786. Median R2 is 0.4407 (Level 1), 0.7969 (Level 2),
0.0506 (Level 3), and 0.9124 (Level 4). The strong Level-4 median is driven by
P0/Ppi; Pfaffian margin remains much weaker.

## Why is the decision still C?

The frozen common schema lacks deltaF_UF, Level-3 spectral outputs remain weak,
topology uncertainty calibration does not improve in both runs, and no
run/level replay curve shows monotonic probe uncertainty.

## Can Stage VI strict cold start begin?

No EIG pilot is authorized. Only schema, masks, architecture/loss source,
validation source, and tests may be reused. All learned state must be rebuilt.
