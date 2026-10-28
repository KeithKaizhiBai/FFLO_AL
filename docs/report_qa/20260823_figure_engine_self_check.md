# Figure-engine self-check: report Q&A

## Did the rules actually get read?

Yes. The health runner loads and hashes all 13 canonical files. It also replaces
the arrow file with a temporary conflicting copy during a unit test; constant
ribbon widths cause preflight to raise `RuleError`. This demonstrates active
parsing and cross-file validation rather than file-presence checking.

## Do the rules constrain generation?

For engine-native builders, yes. The SVG builder refuses a 5-pt label, a text
object without a positive measured bbox, and an unauthorized color. The
Matplotlib rc dictionary is generated from the loaded font, typography, SVG,
and palette tokens. The legacy Figure 1 v5 builder still contains local values,
so its preflight must be supplemented by postflight and figure-specific lint;
this remains known adapter debt.

## Can the validator find deliberate violations?

Yes. A synthetic invalid fixture triggers all 15 expected failure names, and a
matched corrected control passes. The failures cover typography, geometry,
hierarchy, scientific-subobject area, palette, arrows, leaders, and prohibited
scientific text.

## Did the engine improve an existing failed case?

The matched synthetic control demonstrates the repair loop without touching
manuscript artwork. More importantly, the stronger engine found a real failure
in the existing Figure 1 v5 that older checks missed: the first reliability row
has 3.25 px right clearance, below the 4-px rule. This is recorded as a
regression failure and deliberately remains unfixed until an authorized Figure
1 pass.

## What cannot be automated honestly?

The engine cannot infer a topological invariant, boundary support, or
perceptual first fixation from pixels alone. It can require source/evidence and
feature declarations, reject prohibited claims, and run figure-specific
audits. Scientific and perceptual approval remains human-controlled.
