# Figure engine architecture self-audit

**Date:** 2026-08-23  
**Scope:** non-rendering framework health check  
**Artwork policy:** Figure 1 v5 is read-only

## Verdict

The engine is now an executable specification, not only a YAML collection. A
build preflight loads and hashes 13 canonical rule/style files; the enforcement
registry names 36 machine-enforced rules and separates six rules that still
require honest human review. The negative validator fixture fails on all 15
expected violations, while the corrected control passes under the same engine.

The verdict is **pass with a known Figure 1 regression failure and legacy
adapter debt**. It is not authorization to redraw or approve Figure 1.

## Layer audit

| Layer | Canonical responsibility | Executable mechanism | Result |
|---|---|---|---|
| `constitution/` | invariant scientific, layout, typography, color, arrow, asset, and validation policy | complete rule loader, cross-file consistency checks, SHA-256 hashes | pass |
| `templates/` | figure-type defaults | template resolution during contract validation | pass |
| `styles/` | palette, fonts, SVG, Matplotlib tokens | SVG builder and Matplotlib rc translation consume the loaded rule bundle | pass for engine-native builders |
| `builders/` | preflight, registered text/assets/layout, editable SVG | below-floor fonts, missing bboxes, and unauthorized colors are rejected before output | pass |
| `validators/` | output, scientific text, assets, geometry, typography, color, arrows | hard-fail JSON/Markdown lint with nonzero exit | pass |
| `self_tests/` | negative controls, corrected controls, Fig.1 regression | repeatable health-check runner | pass |

## Conflicts checked automatically

The rule loader now rejects:

- non-monotonic or constant main-workflow ribbon widths;
- non-monotonic ribbon opacity;
- card clearance below the internal-padding hard minimum;
- font-stack disagreement between `fonts.yaml` and typography rules;
- label/leader-clearance disagreement between annotation and spacing rules;
- semantic color roles that reference absent palette families;
- disagreement between the schematic template and the canonical primary-area
  threshold;
- incomplete entries in the enforcement registry.

Postflight also compares the preflight record against the current contract path
and every current rule hash. A stale record is a hard failure; a dedicated unit
test mutates one recorded arrow hash and confirms rejection.

The self-test changes a copied arrow rule to a constant-width profile and
confirms that preflight raises `RuleError`.

## Previously non-executable rules repaired in this check

| Former gap | Remediation |
|---|---|
| Missing primary bbox silently passed hierarchy | primary registration is now mandatory |
| Contract could not impose a stricter primary fraction | contract threshold is combined with the template floor |
| A tiny scientific subobject could hide inside a large primary container | contract-declared internal area requirements are now linted |
| Text-to-border and text-to-plot rules existed only in YAML | both are hard geometry checks |
| Leader crossings were not generic | segment intersection and leader/leader checks added |
| Segment crossing a text box could evade corner-distance logic | exact segment/rectangle intersection added |
| Arrow taper, opacity, samples, and route/card crossing were not generic | `route_checker.py` hard checks added |
| Asset scientific requirements were descriptive only | typed assets must declare the required feature set |
| Matplotlib rc values duplicated YAML | rc parameters are now translated from the loaded rule bundle |
| SVG builder accepted 5-pt text, missing bboxes, or arbitrary colors | generation-time rejection added |
| New SVG metadata was only a warning | it is a hard rule for new output; legacy exception must be explicit |

## Natural-language rule audit

No canonical rule uses an unbounded instruction such as “make the figure
beautiful” as a hard rule. Non-numeric statements are assigned one of three
honest enforcement modes:

1. **Converted to measurable constraints:** area fractions, font floors,
   clearances, contrast, taper profiles, registered features, required labels.
2. **Contract plus evidence:** topology semantics, uncertainty separation,
   convergence features, data/schematic identity. The engine can require and
   audit declarations but cannot infer physical truth from pixels.
3. **Human review only:** first perceptual fixation, overall coherence,
   whether a scientifically correct topology cartoon is still visually
   misleading, and whether 3D lighting preserves interpretation.

These human-only rules are explicitly listed in
`constitution/enforcement_registry.yaml` and may not be reported as automated
passes.

## Remaining limitations

### Legacy Figure 1 builder adapter

The v5 builder now performs engine preflight and is postflight-audited, but it
still contains local hard-coded palette and geometry values inherited from the
approved v5 implementation. Preflight proves that rules were read; it does not
prove that every legacy constant was obtained from the engine. A future Figure
1 engine benchmark must use `SVGBuilder.from_context()` and dynamic Matplotlib
tokens, or provide an explicit adapter mapping every legacy token to a rule.

### Pixel-level scientific inference

Generic code cannot honestly prove from SVG pixels alone that a plotted tFFLO
bulk is reopened and gapped, that a boundary line has valid support, or that a
3D surface lighting choice preserves physical meaning. The engine enforces the
contract, asset feature/evidence declarations, prohibited claims, and
figure-specific numerical audits; scientific review remains required.

### Continuous scientific colormaps

Strict discrete-palette checking is inappropriate for dense programmatic
surfaces. Such assets require a declared continuous palette/raster exception.
The current legacy Fig.1 surface therefore produces a non-strict palette
warning rather than pretending that 259 interpolated colors are categorical
tokens.

## Phase decision

Phase 1 framework health checking is complete. Phase 2 has established the
unchanged Fig.1 v5 as a regression benchmark and exposed one hard failure. Do
not begin Phase 3 regeneration until the user authorizes a new Figure 1 engine
benchmark and decides how to resolve the known clearance defect.
