# Strict-data figure architecture audit

Date: 2026-08-27

## Scope and scientific boundary

This audit covers Figures 2-4. Figure 1 engine-v4 is registered for scientific
freeze and its artwork is read-only. The scientific evidence archive is also
read-only. No BdG, active-learning, or HPC calculation is authorized.

Exact BdG observations remain the authority for thermodynamic, spectral, and
topological labels. Active learning records where exact calls were allocated;
it is not a source of final physical labels. Interpolation, smoothing,
boundary extraction, gridding, resampling, and component filtering are disabled
unless a panel contract identifies an archived method, parameters, evidence,
and a validator.

## Existing architecture map

```text
figures/figure_engine/
  constitution/     canonical scientific, narrative, style, and validation rules
  templates/        versioned figure contracts and layout profiles
  styles/           palette, fonts, SVG, and Matplotlib tokens
  builders/         rule loader, contract loader, preflight, layout, SVG, assets
  validators/       output, scientific, geometry, typography, color, narrative lint
  self_tests/       invalid/control fixtures and frozen Fig.1 regressions
  examples/         schematic, phase-map, topology, and high-dimensional contracts
```

The mandatory rule-loading path is
`rule_loader.load_rule_bundle -> contract_loader.load_contract ->
figure_build.preflight`. Postflight uses `validators.figure_lint`. Preflight
records exact rule hashes, and postflight rejects stale rule bundles.

## Directly reusable components

- canonical semantic palette and print typography;
- schema-v1/v2 contract normalization and validation;
- preflight rule-hash record;
- asset registration and scientific/schematic distinction;
- visual-narrative hierarchy and warning-only saliency checks;
- output, SVG, font, color, geometry, collision, and claim lint;
- enforcement registry with hard-fail, warning, and human-review levels;
- negative/control self-test harness and immutable Figure 1 regression.

## Missing components

- strict panel-type classification before visual narrative;
- source hash/version/row/field/unit contracts;
- transformation authorization and immutable exact-layer rules;
- data-specific gates and readiness states;
- phase/topology/3D/convergence semantic validators;
- strict-data negative fixtures;
- canonical intermediate-data manifests;
- explicit linkage of Figure 1 freeze and Figures 2-4 readiness to the
  manuscript registries.

## Evidence discovery

- Stage I random-grid cold start: 512 actual selected coordinates at
  `selection_history/iter000`; full Stage I history through iter030 is present.
- Stage II continuation: iter026-034 is present; overlap iter026-030 is
  coordinate-identical to Stage I. Endpoint `dataset_iter035.npz` contains
  7434 exact observations and full trust/risk metadata. Raw stop metrics cover
  iterations 0-34.
- Stage III endpoint and convergence tables are present, including trusted
  gapped masks and k=8/k=6 sensitivity.
- Stage IV endpoint, exact slice evidence, return audit, and explicit strict
  non-convergence decisions are present.

The Stage II dense monitor-grid NPZ used by the archived report is not present.
Therefore the new Figure 2 does not reconstruct or fit a new smooth boundary.
Its main phase map uses exact samples only; archived boundary-shift metrics are
shown as raw convergence evidence. Phase adjacency is visible, but no new
claim-bearing curve is inferred.

## Files to modify or add

- extend the canonical rule loader and contract loader with schema-v3
  strict-data support;
- add `constitution/strict_data_rules.yaml` and enforcement entries;
- add a panel router, strict-data validator, tests, and negative/control
  fixtures;
- add `figures/fig2_strict_data/` build, contracts, canonical tables, reports,
  and review artifacts;
- add Fig.3/Fig.4 manifests, contracts, and readiness reports only;
- update manuscript figure/evidence/artifact/permission registries and project
  handoff documentation.

## Files not modified

- all files under `figures/fig1_engine_v4/`;
- all files under `manuscript_langgraph/recovered_artifacts/` and
  `downloads/active_learning_scientific_evidence_archive_20260813/`;
- model, numerical, phase, topology, and convergence definitions;
- manuscript LaTeX, Introduction retrieval, and citation sources.

## Compatibility risks and mitigation

- Existing schema-v1/v2 contracts remain valid; strict fields are mandatory
  only for schema v3 or explicit `strict_data.enabled=true`.
- New canonical rule hashes invalidate stale preflight records by design;
  frozen artwork is not regenerated, and its regression is executed through a
  fresh temporary preflight record.
- Generic validators cannot infer scientific truth from pixels. Hashes,
  counts, mappings, transformations, and caveats are machine-enforced; final
  scientific interpretation remains a named human gate.

## Implementation plan

1. Add canonical strict-data rules, router, schema validation, gates, and lint.
2. Add stable negative/control fixtures and integrate them into health check.
3. Build immutable Figure 2 tables from frozen Stage I/II evidence.
4. Reproduce Stage II counts and raw convergence endpoints.
5. Generate exact-sample-only low-fidelity and styled Figure 2 artifacts.
6. Run scientific, visual-narrative, print/export, and provenance validation.
7. Produce Fig.3/Fig.4 pre-contract readiness packages.
8. Update manuscript registries and stop at human science review.
