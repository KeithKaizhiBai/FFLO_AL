# Fig.1 engine-v3 review candidate

Date: 2026-08-26

## What changed?

After approval of `GATE_FIG1_V3_PROPOSAL`, Figure 1 was regenerated as a new
review candidate without overwriting v1 or v2. The card architecture and all
non-central scientific assets were preserved. The central thermodynamic
surface was rendered from the existing schematic free-energy generator,
tightly cropped, and embedded in a declared scientific-content region.

## Why is this scientifically safer than the previous area metric?

The older aggregate metric could pass because the central card was large even
when the actual surface was small. The new hard audit measures the embedded
surface itself relative to the thermodynamic-landscape bbox. The frozen v1
ratio is 0.202950; the v3 candidate reaches 0.716878 against a 0.65 minimum.
The primary object is registered as the thermodynamic scientific subobject,
not the enclosing card.

## What is and is not automated?

Geometry, fill ratio, typography, required/forbidden language, topology
semantics, output dimensions, provenance, and frozen-source hashes are hard
checks. Blurred saliency and contrast hierarchy are warnings only. Whether the
landscape is actually the first fixation and whether the figure reads as a
persuasive physical argument remain human judgments.

## Scientific interpretation retained

Normal, uniform-SC, cFFLO, and tFFLO markers denote representative minima at
different parameter points. Topology is resolved by the trusted bulk-gap and
Pfaffian audit, not by free-energy depth. The red tFFLO curves are in-gap
boundary states. The surrogate schedules exact BdG calls and is not the physics
solver.

```text
GATE_FIG1_ENGINE_V3_FINAL_APPROVAL = HUMAN_WAIT
```
