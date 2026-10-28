# FFLO Active Learning figure engine

## MANDATORY BEFORE FIGURE GENERATION

Every script, Codex task, or manual workflow that creates or modifies a file
under `figures/` must do all of the following before drawing:

1. Read this file and `figures/FIGURE_WORKFLOW.md`.
2. Load the complete constitution through
   `figure_engine.builders.rule_loader.load_rule_bundle()`.
3. Create or validate a canonical `figure_contract.yaml` through
   `figure_engine.builders.contract_loader.load_contract()`.
4. Run `figure_engine.builders.figure_build.preflight()` and retain the
   machine-readable preflight record.
5. Register scientific assets and geometry before master assembly.
6. Run `validators.figure_lint` after generation. A hard failure must stop the
   build; an automated pass must not be interpreted as human approval.

Builders that bypass these steps are non-compliant even if their final image
looks acceptable.

## Canonical sources of truth

- `constitution/`: scientific, visual-narrative, typography, layout, spacing,
  color, arrow, annotation, asset, and validation rules;
- `styles/`: exact font, palette, Matplotlib, and SVG implementation tokens;
- `templates/`: versioned contract and layout profiles;
- `builders/`: mandatory loading, layout, SVG, and asset APIs;
- `validators/`: hard-fail geometry, typography, color, SVG, and scientific
  semantics checks plus warning-oriented visual-narrative evaluation;
- `examples/`: Fig.1 schematic, phase-map, topology, and higher-dimensional
  contracts demonstrating that the engine is not Fig.1-specific.

The three YAML files at the engine root are deprecated compatibility pointers.
They are not rule sources. Canonical rule files live only in `constitution/`.

## Commands

Run preflight before drawing:

```bash
python3 -m figures.figure_engine.builders.figure_build \
  figures/path/to/figure_contract.yaml \
  --record figures/path/to/engine_preflight.json
```

Run validation after SVG/PDF/PNG generation:

```bash
python3 -m figures.figure_engine.validators.figure_lint \
  --contract figures/path/to/figure_contract.yaml \
  --svg figure.svg \
  --pdf figure.pdf \
  --png figure.png \
  --geometry geometry_manifest.json \
  --output-dir figures/path/to/validation
```

Run engine tests:

```bash
python3 -m unittest discover figures/figure_engine/tests
```

Run the complete non-rendering framework health check:

```bash
python3 figures/figure_engine/self_tests/run_figure_engine_health_check.py
```

The health check must demonstrate all of the following:

- every canonical rule file is loaded and hashed;
- a conflicting copied rule is rejected;
- the intentionally invalid fixture fails for every expected reason;
- the corrected control passes;
- the unchanged Figure 1 regression matches its explicit expectation file.
- a parent card cannot substitute for a primary scientific subobject;
- scientific-content fill ratio fails/passes in the negative/control fixtures;
- the frozen Figure 1 artwork hashes remain unchanged.

## Versioned visual-narrative contracts

Schema-v1 contracts remain valid by default. Visual narrative fields become
mandatory only for schema v2 or schema v1 with
`visual_narrative.enabled=true`. The primary visual object must resolve to a
registered non-card scientific subobject. Area and scientific-content fill
ratio are hard rules; unexpected blurred saliency and contrast hierarchy are
warnings; actual first fixation remains human review.

Enabled builds additionally emit `visual_narrative_report.json` and `.md`.
These reports never constitute aesthetic or figure approval.

Do not “fix” a regression by weakening a rule or editing the expectation. Rule
changes, artwork changes, and expectation changes require separate written
justification.

## Figure engine memory

Reusable failure modes and successful patterns live under `lessons_learned/`.
Read the relevant memory files before starting a new figure contract. Memory
entries do not override physical definitions or numerical specifications.

## Scientific boundary

The engine governs representation, hierarchy, reproducibility, and validation.
It never changes the Hamiltonian, phase criteria, topology definition,
convergence rule, or frozen data. Those remain governed by `MODEL_SPEC.md`,
`docs/NUMERICS_SPEC.md`, and `docs/DECISIONS.md`.
