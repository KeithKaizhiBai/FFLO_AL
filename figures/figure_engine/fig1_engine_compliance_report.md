# Figure 1 engine compliance report

**Audit date:** 2026-08-23  
**Figure:** `FIG-01-PIPELINE-V5-ENGINE`  
**Audited artwork:** `figures/fig1_framework_v3/fig1_master_manuscript_v5.{svg,pdf,png}`  
**Audit phase:** baseline, before any figure-engine integration or Figure 1 refinement

## Decision

The approved-layout Figure 1 v5 is scientifically and geometrically compatible
with the reusable figure engine, but its generation path was not yet compliant at
the time of this baseline audit. The artwork therefore **must not be redesigned**
to address the baseline failure. The correct remediation is to add the mandatory
engine preflight to the existing build path and preserve the current physical
content and composition.

The automated baseline result is:

- status: **FAIL**
- hard failures: **1**
- warnings: **2**
- passed checks: **30/33**

The detailed machine-readable and Markdown reports are:

- `engine_compliance_pre_refactor/figure_validation_report.json`
- `engine_compliance_pre_refactor/figure_validation_report.md`

## Rules that pass

### Scientific content

- All contract-required labels are present.
- No prohibited scientific claim appears in the artwork.
- The frozen v5-specific layout and semantic audit passes.
- The central physical-state manifold remains the primary visual object.
- The declared raster exception is limited to the central programmatic
  free-energy landscape.

### Output and reproducibility evidence

- Editable SVG, one-page PDF, and PNG preview are present.
- The SVG contains 76 editable text nodes.
- The PNG is 2103 x 1420 px with a white background.
- The PDF page size is recorded as 504.567 x 340.583 pt.
- The geometry manifest records the audited boxes, text, leaders, and measured
  v5 evidence.

### Layout and spacing

- All registered objects are within the canvas.
- Child objects remain inside their parent panels.
- Independent cards retain at least 20 px clearance.
- The primary registered visual occupies 60.25% of the registered visual area,
  above the 50% schematic-template minimum.
- No undeclared text-text overlap is detected.
- Phase labels and annotation leaders meet their required clearances.

### Typography and vector integrity

- All registered text is at least 7 pt.
- No condensed font, non-uniform text scale, or skewed text is used.
- The SVG has no non-uniform group scale or skew transform.
- The normal-width sans-serif stack is compliant.

### Color

- Every contract color role maps to a canonical semantic family.
- The dense Matplotlib landscape contains 259 interpolated colors; these are
  accepted because the contract explicitly disables strict discrete-palette
  enforcement for this programmatic scientific asset.

## Rules that fail

### Hard failure: engine preflight

The legacy Figure 1 v5 build did not create
`refinement_v5/fig1_engine_preflight.json`. Consequently, there was no recorded
proof that the canonical constitution and the Figure 1 contract had been loaded
before generation. This is a build-governance failure, not a scientific or
visual defect.

Required remediation:

1. add canonical contract/rule preflight to the existing Figure 1 v5 build
   entry point;
2. write a preflight record containing contract and rule hashes;
3. re-run the same validation suite;
4. do not alter the approved visual output during this integration pass.

## Warnings

### Missing engine metadata in the legacy SVG

The current SVG predates the engine and therefore does not embed the figure id,
contract path, and engine schema. This is retained as a warning because adding
metadata would modify an already approved artwork file. Newly regenerated SVGs
must include the metadata.

### Legacy mathematical font fallback

The current figure uses a Times-family mathematical fallback rather than the new
preferred STIX stack. The figure remains readable and compliant as legacy output.
STIX is mandatory for new or materially regenerated figure output, but no font
substitution is authorized in this non-visual integration pass.

## Human gate

This audit does not pass `GATE_FIG1_V5_APPROVAL` or any manuscript figure gate.
Automated compliance and human scientific approval remain separate.

## Post-baseline remediation record

The baseline report above was written before any builder was changed. The
following narrowly scoped integration was then applied:

- `build_v5_assets.py` now loads the canonical engine rules and Figure 1
  contract before generating scientific assets;
- `build_master_v5.py` now performs the same preflight and will embed engine
  provenance attributes in newly regenerated SVG output;
- `build_fig1_with_engine.py` provides one engine-governed regeneration entry
  point and a non-destructive `--preflight-only` mode;
- `fig1_engine_preflight.json` now records the contract, approval gate, and
  hashes of the loaded canonical rule files.

No SVG, PDF, PNG, scientific asset, layout coordinate, physical definition, or
visual style was regenerated during this remediation pass.

The identical validator suite was then re-run against the unchanged artwork:

- status: **PASS**
- hard failures: **0**
- warnings: **2**
- passed checks: **32/34**

The post-integration validator also audits five registered foreground/background
contrast pairs; all five meet their role-specific WCAG-style thresholds.

The remaining warnings are the two documented legacy-output warnings above.
They will clear or be reassessed only when a material regeneration is explicitly
authorized. The post-integration reports are:

- `engine_compliance_post_refactor/figure_validation_report.json`
- `engine_compliance_post_refactor/figure_validation_report.md`

This integration establishes engine compliance but does not pass the human
approval gate.

## 2026-08-23 strengthened-engine regression supersession

The earlier 32/34 post-integration pass above is retained as chronology, but it
is superseded by the framework health check. The expanded generic validator now
adds typed-asset features, scientific-subobject area requirements,
text-to-border and text-to-plot clearance, leader/leader crossing, and generic
workflow-route checks.

Against the unchanged Figure 1 v5 artwork, the expanded result is:

- status: **FAIL**
- hard failures: **1**
- warnings: **2**
- passed checks: **42/45**

The hard failure is `reliability-row-1`: its registered right clearance inside
the reliability card is 3.25 px, below the canonical 4-px minimum. This defect
was not detected by the previous validator. It is recorded as the expected
Figure 1 regression signature and is not repaired in the engine self-check.

See `self_tests/output/fig1_regression/fig1_engine_regression_report.md` and
`self_tests/fig1_regression_expectation.yaml`. `GATE_FIG1_V5_APPROVAL` remains
`HUMAN_WAIT`.
