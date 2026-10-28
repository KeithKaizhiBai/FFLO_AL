# Mandatory scientific figure workflow

This workflow applies to every figure task under `figures/`.

## 1. Read the constitution

Read `figure_engine/README.md`, the complete `constitution/`, and the relevant
physical/numerical project specifications. Do not change scientific definitions
through a style file.

Read the relevant `figure_engine/lessons_learned/` memory. If modifying the
engine itself, run the complete health check before and after the change. New
hard rules require a failing negative fixture and a passing corrected control.

## 2. Create the figure contract

Start from `figure_engine/templates/figure_contract.yaml` for a backward-
compatible schema-v1 contract, or `figure_contract_v2.yaml` when an explicit
visual narrative is required. Declare:

- scientific message;
- primary, secondary, and tertiary visual roles;
- frozen data sources or schematic generators;
- assets and their scientific type;
- layout profile and bounding boxes;
- semantic color mappings;
- text budget;
- required labels/elements;
- forbidden elements and claims;
- validation requirements and human gate.

Schema v2, or schema v1 with `visual_narrative.enabled=true`, must additionally
declare the visual message, a real non-card primary scientific subobject,
reading order, narrative regions, saliency expectation, and forbidden visual
priorities. A parent card may not substitute for the primary scientific object.

Run engine preflight before drawing and save its record.

## 3. Create the asset list

Register each asset, source, parameters, role, bbox, and whether it is
data-backed or schematic. Quantitative assets must point to frozen sources.
Schematics cannot be used as quantitative evidence.

## 4. Generate and validate assets

Generate assets independently when practical. Use editable vector text, axes,
lines, markers, and annotations. A dense programmatic raster may be embedded
only under the declared asset policy and must retain sufficient effective dpi.

## 5. Assemble the figure

Use declared layout anchors and the 8-px spacing rhythm. Preserve primary,
secondary, and tertiary hierarchy. Connectors encode information flow and must
not cross cards, text, or unrelated plots.

## 6. Run validation

Run the generic engine lint plus any figure-specific scientific audit. Hard
failures stop the build. Warnings require a written disposition. Automated
validation never approves a human gate.

Visual-narrative saliency and contrast checks are warning-oriented. Actual
first fixation and aesthetic persuasiveness remain named human-review checks.
They must never be reported as automatic passes or used as physical criteria.

Do not make a regression pass by weakening a rule or editing its expectation.
Resolve the source defect or obtain an explicit, scientifically justified
contract exception.

## 7. Generate deliverables

Every final figure task must produce:

```text
<figure>.svg
<figure>.pdf
<figure>.png
figure_validation_report.json
figure_validation_report.md
asset_manifest.json
regeneration_log.md
```

Complex figures additionally require native-width, grayscale, and relevant
debug/saliency reviews. Record parameters and commit hash when available.

## 8. Preserve scientific and version history

Never overwrite an approved/frozen figure version. Never replace physics-based
criteria with visual smoothing or manual boundaries. Update project summary,
decisions when applicable, report QA, and handoff notes after meaningful work.
