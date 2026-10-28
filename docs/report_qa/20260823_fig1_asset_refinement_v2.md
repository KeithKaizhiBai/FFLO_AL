# Figure 1 engine-v2 asset-refinement Q&A

## What scientific story does the refined figure tell?

The exact BdG oracle generates a free-energy landscape
$F(\Delta,q)$ and audited physical observations. Reliability-aware active
learning uses those observations to schedule the next exact evaluations; it
does not replace the BdG solver. The central visual therefore prioritizes the
thermodynamic and topology-resolved physical state space over the surrounding
computational loop.

## How should the phase markers be read?

Normal, uniform superconducting, cFFLO, and tFFLO markers are schematic
representative minima encountered across parameter space. They do not assert
that all minima coexist at one parameter point. Green cFFLO and purple tFFLO
markers connect thermodynamic branch identification to a separate topology
audit. Topology is determined through the trusted gap and Pfaffian workflow,
not from free-energy depth.

## What does the tFFLO panel claim?

It shows a reopened gapped bulk with $\nu=1$ and a red in-gap boundary-state
branch. The red branch is not a gapless bulk band. The gap-closing transition
is visually separate from the final tFFLO phase.

## What was changed relative to engine v1?

Only four scientific assets were refined: the central landscape/branch
markers, the internal BdG-oracle hierarchy, the physical-observation readout,
and the layered appearance of canonical workflow ribbons. The card
architecture, canvas, claims, definitions, and engine rules were unchanged.
No physics was recomputed.

## What evidence supports readiness for review?

The generic engine lint passes 45/45 checks, the Fig.1-specific audit passes
19/19, and all 18 engine unit tests pass. The central card ranks first in the
saliency regression, browser geometry checks pass, and the PDF is exactly
178.00 × 120.15 mm. These checks establish technical readiness only;
`GATE_FIG1_ASSET_REFINEMENT_APPROVAL` remains a human decision.
