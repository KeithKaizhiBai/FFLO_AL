# Strict data figure workflow

```text
claim freeze
-> evidence inventory
-> panel-type router
-> source/provenance contract
-> semantic/transformation contract
-> canonical immutable inputs
-> reproducible low-fidelity reconstruction
-> scientific invariants
-> visual narrative
-> styled rendering
-> export integrity
-> human science review
-> human visual review
-> figure freeze
```

Every strict-data panel must declare a panel type, source hashes, fields,
units, selection/exclusion reconciliation, label semantics, transformations,
uncertainty and hard-risk layers, expected invariants, claims, caveats,
forbidden operations, validators, and human-review items.

Transformations default to disabled. A non-empty transformation requires an
archived authority, implementation, parameters, justification, immutable exact
input, and a matching validator. Missing evidence blocks the panel; no
synthetic substitute is permitted.

Panel types and routes:

| Panel type | Required route |
|---|---|
| `EXACT_SAMPLE_DATA` | provenance, semantics, reconstruction, invariants |
| `DERIVED_NUMERICAL_DATA` | above plus derivation authorization and tolerance |
| `DERIVED_BOUNDARY_GEOMETRY` | above plus boundary algorithm/support/sensitivity |
| `INTERPOLATED_FIELD` | above plus explicit interpolation authorization, masks, overlays, sensitivity |
| `SCHEMATIC_ANNOTATION` | quantitative claim must be false |
| `HYBRID_DATA_SCHEMATIC` | immutable data layer and visually distinct annotation layer |

All routes then pass visual narrative, style/layout, export integrity, and the
separate human gates. Automated success never freezes a figure.
