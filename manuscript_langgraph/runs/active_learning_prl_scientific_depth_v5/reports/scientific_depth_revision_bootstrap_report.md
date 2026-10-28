# Scientific-depth revision bootstrap report

## Current graph state

The verified parent is `active_learning_public_reproducible_v4` at
`GATE_PUBLIC_MANUSCRIPT_AND_SUPPLEMENT_HUMAN_REVIEW = HUMAN_WAIT`.  This child
run preserves that gate record and all parent outputs.

## Current section word counts

abstract: 133
introduction: 472
results: 675
discussion: 194
total: 1477

## Current figure placement

The parent source places each full-width figure after one compact result
paragraph.  The compiled manuscript then devotes pages 3--6 primarily to the
four figures, leaving the scientific discussion detached from the visual
evidence.

## Current method kernel and depth diagnosis

The parent main text states the oracle/scheduler separation and qualitative
phase logic, but omits the compact model, the explicit `(Delta,q)` minimization,
and the Pfaffian relation.  Fig. 2 lacks a direct description of the curved
Normal--superconducting support and the narrow uniform-SC strip; Fig. 3 lacks a
bulk-state interpretation of the cFFLO/tFFLO ordering; Fig. 4 does not say how
the four fixed-mu slices differ.  The Discussion combines contribution and
outlook but does not synthesize physical contribution, method, efficiency, and
limitations separately.

## Planned minimal patch

The existing publication layer remains authoritative.  This continuation adds
physical-interpretation packets, scientific-depth and minimum-method-kernel
gates, a Discussion synthesis gate, and a float-interleaving comparison.  It
also performs two mandatory consistency audits before prose synthesis.

## Read-only scientific sources

The Stage-I and Stage-II endpoint arrays, strict-data Fig. 2--4 tables, source
code for the BdG objective, and frozen figure manifests are read only.  No
boundary is fitted and no physical label is recalculated.

## Protected files

The parent run, parent PDFs, Figures 1--4, figure freeze manifests, production
code, datasets, labels, and human gate records are not modified.  The export is
not a Git worktree, so SHA-256 records replace a commit identifier.
