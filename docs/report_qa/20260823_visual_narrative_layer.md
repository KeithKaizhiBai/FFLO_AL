# Visual Narrative Layer report note

## Why is the existing Figure Engine insufficient by itself?

The engine correctly enforces geometry, typography, spacing, semantic color,
asset provenance, and scientific labels. Those checks establish that a figure
is valid and reproducible, but they do not establish that the reader encounters
the scientific argument in the intended order.

For Figure 1, the existing area test treats the compound central card as the
primary object. That card occupies 60.25% of registered card area and passes.
The intended primary object is narrower: the thermodynamic landscape
$F(\Delta,q)$. It occupies 24.68% of the central card, whereas the topology
region occupies 40.35%.

## What did the region-level saliency proxy find?

After grayscale conversion and Gaussian blur, integrated luminance mass ranks
the topology region first, Physical observation second, Exact BdG oracle third,
and the thermodynamic landscape fourth. This does not measure eye tracking or
scientific truth, but it exposes a mismatch between the declared container
hierarchy and the intended scientific-subobject hierarchy.

## What is the new rule-layer principle?

The visual argument should move from physical object, to physical evidence, to
computational instrument. For AI-for-science figures, the physical result must
remain visually prior to the ML workflow. Unexpected pixel-level saliency is a
warning and human-review input, not a physics criterion or automatic scientific
failure.

## What remains deliberately undone?

The staged rule is not yet wired into the rule loader, contract schema,
validators, or tests. Figure 1 was not modified, and no v3 proposal was
generated. That integration requires explicit approval and negative/control
fixtures before any new figure build.
