# Target-Construction HPC Package Notes

Date: 2026-06-08

## Question

What does the 32 fixed-point target-construction-only package compute?

## Answer

It computes the exact-oracle front half needed for target construction:

```text
coarse scan
q-window expansion
candidate detection
basin clustering
risk annotation
ordinary energy-window marking
final target selection
```

It then stops.  It does not run local refinement boxes, does not run active
learning, and does not write training data.

## Package Design

The Slurm array dimension is fixed point, not variant.  Each task computes one
fixed point once and applies these variants to the same candidate set:

```text
baseline
cluster_only
rank_and_cap_k3
rank_and_cap_k2
rank_and_cap_energy_window
```

This avoids repeating the expensive coarse/q-expansion scan five times per
point.

## HPC Safety

The generated Slurm scripts and submit workflow exclude:

```text
gpuh01
```

All generated shell scripts are normalized to LF line endings.

## Report Use

This package is the gate before any full local-box GPU regression.  If any
`rank_and_cap_*` selected target count exceeds the cap, the next step is code
review, not rerunning missing full-refinement tasks.
