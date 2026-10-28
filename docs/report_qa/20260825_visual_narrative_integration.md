# Visual Narrative Layer integration

Date: 2026-08-25

## Question

How can Figure Engine enforce a physics-first visual argument without claiming
that automatic saliency measurements prove aesthetic success?

## Answer

The engine now uses a versioned narrative contract. The primary object must be
a registered scientific subobject, not an enclosing card, and its area and
scientific-content fill ratio are hard constraints. Declared reading order is
also machine checked. Region-level blurred saliency, contrast hierarchy, and
equal-weight diagnostics are warnings because they are only proxies. Whether
the landscape is actually the first fixation and whether the composition makes
a persuasive scientific argument remain explicit human-review items.

For Fig.1, this separation exposes a real regression: the frozen engine-v1
landscape surface fills only 0.20295 of its declared landscape bbox, below the
0.65 Fig.1 threshold, even though the enclosing central card is large. The
engine therefore prevents a parent-card/large-bbox false pass without treating
its saliency proxy as an automatic aesthetic approval.

The scientific message remains unchanged: representative normal, uniform-SC,
cFFLO, and tFFLO minima occur at different parameter points; topology is
resolved by trusted gap/Pfaffian evidence rather than free-energy depth; and
machine learning schedules exact BdG-oracle calls rather than replacing the
physics solver.

## Gate status

```text
GATE_VISUAL_NARRATIVE_INTEGRATION = APPROVED_AND_IMPLEMENTED
GATE_FIG1_V3_PROPOSAL = HUMAN_WAIT
```

The v3 package is proposal-only and contains no final artwork.
