# Figure engine audit

Date: 2026-08-23

Scope: read-only audit of the current figure system before rule consolidation.
No figure, physical definition, numerical rule, or frozen result was changed
to produce this report.

## Resources inspected

### Existing `figure_engine`

| File | Current role | Assessment |
|---|---|---|
| `typography_rules.yaml` | Partial font and text-box rules | Useful seed, but no schema, role hierarchy, measurement units, exceptions, or executable validation mapping. |
| `layout_rules.yaml` | Partial Fig.1 card hierarchy | Fig.1-specific; its fixed `central_physics` identity and area threshold do not generalize to phase maps or slice atlases. |
| `style_rules.yaml` | Partial ribbon-arrow rules | Misnamed and incomplete; duplicates v5 arrow settings but uses different numerical values. |

There is no loader, contract schema, builder API, validator package, template,
example, test, workflow document, or automatic invocation mechanism.

### Figure 1 design history

The audit covered:

- `figures/fig1_framework/`: direct-SVG physics-first prototype and validation;
- `figures/fig1_framework_v3/style/`: approved semantic palette, typography,
  card material, layout, and style-board validation;
- `figures/fig1_framework_v3/assets/`: nine independently generated and
  contracted scientific assets;
- `refinement_v2/`: compact composition and saliency review;
- `refinement_v3/`: true-font geometry, 14-px padding, 20-px gutters, and
  hard-fail layout lint;
- `refinement_v4/`: metric-bounded targeted correction and print QA;
- `refinement_v5/`: branch semantics, DOM geometry, tapered ribbons,
  right-column typography, and 12-category hard lint;
- all current Fig.1 builders, layouts, style files, contracts, validation
  reports, debug geometry, and regeneration records.

### Future manuscript figures

The existing contracts and mockup builders for the following were inspected:

- Fig.2: phase maps, acquisition evolution, raw convergence diagnostics;
- Fig.3: trusted topology maps, boundary support, convergence and neighbor
  sensitivity;
- Fig.4: one 3D overview, matched fixed-parameter slice atlas, physical
  evolution, and scaling evidence.

These contracts already contain important scientific required/forbidden
language, but their schemas differ from the requested reusable contract.

### Existing validators and utilities

The following reusable patterns were found:

- editable-SVG text and prohibited-term checks in `validate_style_gate.py`;
- independent asset contracts and asset-first assembly;
- true-font bbox registration in refinement v3;
- card, label, leader, mini-plot, and route geometry tests in v3-v5;
- DOM-computed typography and transform checks in v5;
- semantic palette families in `style/fig1_style.yaml`;
- exact print-size, grayscale, saliency, crop, and debug review artifacts;
- build manifests and SHA-256 provenance records.

## General principles extracted from Figure 1 iterations

1. **Physics first.** The main scientific object must dominate before the
   computational workflow is read.
2. **Unequal visual weight.** Primary, secondary, and tertiary roles must be
   declared in the contract; equal-sized cards are not a neutral default.
3. **Asset-first construction.** Scientific assets should be generated and
   validated independently before master assembly.
4. **Contracts before drawing.** Required meanings, forbidden claims, data
   sources, and visual roles must be frozen before layout work.
5. **Measure instead of eyeballing.** Text, labels, plots, leaders, and routes
   require registered or DOM-measured bounding boxes.
6. **Remove text before shrinking it.** Print readability cannot be traded for
   local crowding relief.
7. **Semantic color is project-wide.** Exact computation, physical evidence,
   learning, decision, selected calls, topology, transition, and boundary
   states must retain stable color families.
8. **Arrows encode information flow.** Main workflow routes are tapered,
   directional ribbons; annotation connectors are weak and visually distinct.
9. **Scientific caveats belong in contracts/captions.** Audit metadata and
   provenance must remain available without dominating the artwork.
10. **Hard lint is necessary but not human approval.** Geometry and semantics
    can pass automatically; scientific/visual freeze remains an explicit gate.

## Duplicated rules

| Rule family | Duplicated locations | Consolidation decision |
|---|---|---|
| Fonts and minimum sizes | existing engine YAML, v3 style YAML, v3-v5 builders/lints | Canonicalize in constitution + fonts style; contracts may only tighten. |
| Card gaps and padding | existing engine layout, v2 reports, v3 layout engine, v4/v5 YAML | Canonicalize 8-px rhythm and hard minima; allow declared schematic exceptions only with justification. |
| Semantic palette | v3 style YAML, v4/v5 local styles, builders | Canonical palette file; figure-local aliases must resolve to approved tokens. |
| Arrow widths/opacities | existing `style_rules.yaml`, v3 arrow material, v4 route styles, v5 ribbons | Canonical role profiles; Fig.1 v5 profile becomes the manuscript-flow default. |
| Required/forbidden text | figure contracts, style gate validator, v5 semantic lint | Move to each figure contract and validate generically. |
| SVG editability | style gate validator, asset validators, v5 DOM lint | One SVG validator with configurable raster exceptions. |
| Print and visual QA | v2-v5 review builders | One validation policy plus figure-specific review builders. |

## Conflicts requiring resolution

1. `style_rules.yaml` specifies ribbon widths `5/18/24`, while the accepted v5
   build uses `6/17/23`; the latter becomes the default manuscript profile.
2. The old layout file declares a universal primary-area fraction of `0.55`.
   This is appropriate for a central schematic but not for multi-panel phase
   maps, topology audits, or 3D slice atlases. The rule becomes
   template-dependent.
3. Earlier v2 composition allowed intentional card docking/overlap; v3-v5
   adopted independent cards with measured gutters. Independent cards become
   the default, with explicit contract exceptions only for connected
   scientific composites.
4. The requested typography constitution forbids transformed text, while
   Matplotlib-generated SVG commonly uses group transforms for glyph
   placement. The executable rule therefore forbids non-uniform scaling,
   skewing, and condensed transforms; uniform placement/rotation is allowed
   only when the contract declares it.
5. `STIX Two Math` is the desired math face, but existing outputs may use
   STIXGeneral/Times fallback. New figures require the preferred stack;
   legacy compliance reports may classify a declared fallback as a warning,
   not silently as a pass.
6. The existing Fig.1 contracts use `panels`, `source_datasets`, and
   `allowed/forbidden`; the requested template uses `assets`, `data_sources`,
   and `forbidden_elements`. A versioned canonical schema will preserve both
   scientific content and machine-readable migration aliases.
7. Full object-overlap prohibition is incorrect for plots, intentional
   overlays, raster-under-vector assets, and nested card content. The engine
   must validate overlaps by object role and declared containment, not by all
   pairs indiscriminately.

## Missing capabilities

- versioned canonical rule schemas and cross-file reference validation;
- mandatory rule loader used by every builder;
- reusable figure contract template and contract validator;
- generic layout primitives and semantic asset registry;
- geometry manifest format for cards, plots, text, leaders, and routes;
- generic typography, bbox, SVG, color, and scientific-semantics validators;
- hard/soft severity model and machine-readable exit status;
- deterministic JSON/Markdown validation reports;
- templates for manuscript, presentation, multi-panel, and schematic figures;
- examples covering Fig.1 workflow, topology, phase maps, and 3D/slice use;
- tests proving Fig.2-Fig.4 contracts load without Fig.1-specific assumptions;
- project-level workflow documentation and automatic preflight command.

## Recommended final architecture

```text
figures/figure_engine/
  README.md
  figure_engine_audit.md
  figure_engine_architecture.md
  constitution/
    scientific_figure_principles.yaml
    typography_rules.yaml
    layout_rules.yaml
    spacing_rules.yaml
    color_rules.yaml
    arrow_rules.yaml
    annotation_rules.yaml
    asset_rules.yaml
    validation_rules.yaml
  templates/
    figure_contract.yaml
    manuscript_figure_template.yaml
    presentation_figure_template.yaml
    multi_panel_template.yaml
    schematic_template.yaml
  styles/
    matplotlib_rc.py
    svg_style.yaml
    fonts.yaml
    palette.yaml
  builders/
    rule_loader.py
    contract_loader.py
    layout_engine.py
    svg_builder.py
    asset_manager.py
    figure_build.py
  validators/
    text_collision_checker.py
    bbox_checker.py
    font_checker.py
    svg_checker.py
    color_checker.py
    scientific_checker.py
    figure_lint.py
  examples/
    fig1_example/
    topology_example/
    phase_map_example/
    high_dimensional_example/
  tests/
```

The engine will load constitution files first, merge only permitted
template/contract overrides, require a figure contract and geometry manifest,
run hard lint, and emit SVG/PDF/PNG plus JSON/Markdown validation artifacts.
It will remain figure-type agnostic: primary-object roles and scientific asset
types are supplied by the contract rather than hard-coded as Fig.1 cards.

## 2026-08-25 integration addendum

The Visual Narrative Layer is now active and executable rather than advisory.
The rule loader includes it in the 14-file preflight hash bundle; the contract
loader supports backward-compatible schema v1 and narrative-required schema
v2; and the enforcement registry separates hard failures, warnings, and human
review. The narrative validator checks scientific-subobject identity, primary
area, declared reading order, and scientific-content fill ratio. Blurred
saliency/contrast/equal-weight proxies are warnings only, while first fixation
remains explicit human review.

The invalid fixture proves that a parent card cannot stand in for the primary
scientific object and that a large landscape bbox cannot conceal a tiny surface.
The corrected control passes. The full health check passes with the expected
frozen Fig.1 regression detections, and no engine-v1 artwork hash changed.
