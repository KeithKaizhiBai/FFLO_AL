# Figure engine architecture

## Purpose

The engine converts a manuscript claim and its evidence contract into a
reproducible figure build with project-wide visual rules and hard validation.
It separates five concerns that were previously mixed inside Fig.1 scripts:

```text
scientific principles -> visual narrative -> style tokens -> layout/assets -> generation -> validation
```

## Data flow

1. `load_rule_bundle()` loads every canonical constitution/style file,
   including `visual_narrative_rules.yaml`, and
   validates cross-file consistency.
2. `load_contract()` validates the figure-specific scientific message, visual
   roles, data sources, assets, text budget, forbidden claims, and gate.
   Schema-v2 or explicitly opted-in schema-v1 contracts also require a
   non-card primary scientific subobject and ordered narrative regions.
3. `preflight()` binds the exact rule hashes to the build and writes an engine
   record before any drawing occurs.
4. Asset builders consume canonical fonts/colors and emit independent assets
   plus registered geometry.
5. The master builder assembles assets using declared bounding boxes and
   connectors; it does not invent scientific semantics.
6. `figure_lint` validates contract, output presence, geometry, typography,
   SVG editability, palette, and scientific text. It emits deterministic JSON
   and Markdown and returns nonzero on any hard failure. Enabled narrative
   contracts additionally produce a visual-narrative JSON/Markdown report;
   saliency mismatches are warnings and first fixation remains human review.
7. PDF/PNG visual QA checks print width, grayscale, saliency, and debug views.
   Human gates remain separate.

## Layer responsibilities

### Constitution

Defines invariant project rules. Figure contracts may tighten these rules but
may not silently weaken them. Exceptions require an explicit contract entry
and must not change a physical definition.

### Styles

Implements semantic tokens. All figures use one palette and normal-width font
stack. Continuous colormaps are registered separately from categorical phase
identity.

### Templates

Provide journal, presentation, schematic, and multi-panel defaults. Primary
area thresholds are template-dependent so a phase map, topology audit, or 3D
slice atlas is not forced into the Fig.1 card architecture.

### Builders

- `rule_loader.py`: canonical rule discovery and consistency checks;
- `contract_loader.py`: versioned contract schema and legacy migration view;
- `layout_engine.py`: bbox, gaps, containment, rhythm, and visual-area helpers;
- `asset_manager.py`: asset registry and SHA-256 manifest;
- `svg_builder.py`: editable SVG primitives using canonical styles;
- `figure_build.py`: mandatory preflight API and CLI.

### Validators

- `text_collision_checker.py`: text/text and text/leader geometry;
- `bbox_checker.py`: clipping, containment, card gaps, and hierarchy;
- `font_checker.py`: minimum sizes, font width, transforms, and math stack;
- `svg_checker.py`: parse/editability/raster/transform/metadata checks;
- `color_checker.py`: semantic family and palette checks;
- `scientific_checker.py`: required labels and prohibited claims;
- `visual_narrative_checker.py`: primary scientific-subobject resolution,
  area/fill hard rules, warning-only saliency/contrast proxies, and named human
  first-fixation review;
- `figure_lint.py`: orchestration, reports, and exit status.

### Enforcement registry and self-tests

- `constitution/enforcement_registry.yaml` maps each executable rule to its
  builder, contract, generic lint, figure-specific audit, or human-review
  enforcement level;
- `self_tests/invalid_figure/` is an intentionally non-compliant negative
  fixture;
- `self_tests/corrected_figure/` is the matched passing control;
- `run_validator_self_test.py` asserts the exact failure signature;
- `run_figure_engine_health_check.py` loads/hashes all rules, validates all
  example contracts, runs unit tests and negative controls, and checks the
  read-only Fig.1 regression expectation.

Human-review-only principles are named explicitly. They are not counted as
automatic passes.

The Fig.1-specific fill audit also inspects the actual rendered free-energy
surface inside the declared landscape bbox. The frozen v1 ratio is 0.2029
against the 0.65 proposal floor, demonstrating why container and bbox area are
insufficient hierarchy proxies.

## Generic geometry manifest

```json
{
  "schema_version": 1,
  "canvas": {"x": 0, "y": 0, "width": 1600, "height": 1080},
  "objects": [
    {
      "id": "phase_map",
      "type": "plot",
      "role": "primary",
      "parent": "canvas",
      "bbox": {"x": 100, "y": 100, "width": 900, "height": 700}
    }
  ],
  "routes": []
}
```

Text objects add font family, size, stretch, transform, measured bbox, and
semantic role. Cards declare whether they are independent. Intentional
overlays and raster scientific surfaces require contract declarations.

## Figure-type support

- Fig.1: central physical-state schematic plus a subordinate scheduling cycle;
- Fig.2: dominant phase map, common-domain acquisition trajectories, and raw
  convergence plots;
- Fig.3: topology map, support-only boundary extraction, bulk-gap/Pfaffian
  applicability, convergence, and neighbor sensitivity;
- Fig.4: one 3D overview linked to a shared-scale slice atlas and scaling
  diagnostics, with explicit non-closure language when required.

No module hard-codes the Fig.1 object name. Contracts select asset types and
template profiles.
