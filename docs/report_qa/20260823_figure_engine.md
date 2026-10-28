# Reusable scientific figure engine

## Why was a project-wide figure engine needed?

The Figure 1 iterations showed that individual assets could be scientifically
correct while the master composition still lost hierarchy, spacing rhythm, or
typographic consistency. The reusable engine separates the manuscript claim
and evidence contract from shared visual rules, generation mechanics, and
validation. Physics remains primary; machine learning is represented as a
scheduler and not as a replacement for the exact BdG oracle.

## How does it work?

Before drawing, a figure builder loads the canonical constitution and validates
a figure contract that identifies the primary physical object, supporting
evidence, auxiliary information, frozen sources, semantic colors, forbidden
claims, and human gate. The preflight record stores SHA-256 hashes of the rules.
Assets and geometry are then registered, the editable SVG is assembled, and a
generic lint writes JSON and Markdown reports. Hard failures stop the build;
human scientific approval remains separate.

## What was learned from the Figure 1 integration?

The existing v5 artwork already satisfied its scientific, geometry, hierarchy,
typography, output, and legacy-specific checks. Its only baseline hard failure
was governance: the legacy generator had not loaded a project-wide contract
before drawing. Connecting the existing build entry points to preflight removed
that failure without changing the figure. This demonstrates why compliance and
visual redesign should be treated as separate operations.

## How does this extend to Figures 2--4?

- Figure 2 contracts make a 2D phase map primary and register convergence plots
  and active-learning trajectories as supporting evidence.
- Figure 3 contracts distinguish gapped cFFLO, gap closing, and reopened gapped
  tFFLO, while treating extracted boundaries as supported diagnostics rather
  than independent evidence.
- Figure 4 contracts require a higher-dimensional overview plus shared-scale
  slices and preserve the explicit non-closure caveat for the 3D topology
  surface.

The engine does not create quantitative claims or recompute physics. Frozen
data and the manuscript claim registry remain authoritative.
