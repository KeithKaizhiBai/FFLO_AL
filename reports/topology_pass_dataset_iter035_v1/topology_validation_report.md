# Stage III Topology Pass: `topology_pass_dataset_iter035_v1`

## Scope

This report is an offline topology classification and compute-cost audit on frozen `dataset_iter035`.
It does not modify thermodynamic labels and does not run a new active-learning loop or Delta-q exact search.

## Current Status

- mode executed: `full`
- Pfaffian validation pass: `True`
- full-pass summary available: `True`

## Important Caveats

- Sparse topology scatter is not a final topological phase boundary.
- Pfaffian signs are used only together with the full-Brillouin-zone gap status.
- Hard-risk/provisional exact points are diagnostics only unless explicitly audited.
- Parquet output depends on a local parquet backend (`pyarrow` or `fastparquet`).

## Full-Pass Summary

- total SC points: 5567
- trusted topology points: 5357
- provisional topology points: 210
- uniform-SC trivial/topological/gapless/unresolved: 715 / 0 / 0 / 0
- FFLO trivial/topological/gapless/unresolved: 3127 / 1515 / 195 / 15
- backend: gpu
- Nk: 2048
- actual runtime seconds: 5.342
- bulk-gap min/median/p95: 2.44062e-13 / 0.215371 / 0.517202
- Pfaffian-margin min/median/p95: 9.55634e-05 / 0.340419 / 0.48351
- parquet status: not_written: ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.
A suitable version of pyarrow or fastparquet is required for parquet support.
Trying to import the above resulted in these errors:
 - `Import pyarrow` failed. pyarrow is required for parquet support. Use pip or conda to install the pyarrow package.
 - `Import fastparquet` failed. fastparquet is required for parquet support. Use pip or conda to install the fastparquet package.

## Pfaffian Convention Check

- The analytic formula is cross-checked against the project BdG Hamiltonian builder in the current Nambu convention.
- Verified convention: `P0 = (mu - t cos(q/2))^2 + Delta^2 - alpha_y^2 sin^2(q/2) - (J_A cos(q/2) + alpha_z sin(q/2))^2`.
- Verified convention: `Ppi = (mu + t cos(q/2))^2 + Delta^2 - alpha_y^2 sin^2(q/2) - (J_A cos(q/2) + alpha_z sin(q/2))^2`.
- Product-sign agreement is required on all non-boundary validation points before full pass.

## Coverage Diagnostics

- trusted FFLO gapped points used for Delaunay diagnostics: 4642
- Z2-change candidate edges: 182
- P0 sign-change candidate edges: 182
- Ppi sign-change candidate edges: 0
- large circumradius coverage-hole triangles: 447
- nearest-neighbor p95/max distance in normalized parameter space: 0.011762518145388454 / 0.054962868592079335

These are candidate topology-boundary seeds and coverage diagnostics only. They are not final topology contours.

## Resource Decision

- Pilot projection was below the 6-hour local threshold.
- GPU and CPU agreed to double-precision tolerance at Nk=2048 in the pilot comparison.
- The selected full-pass backend was chosen from measured pilot runtime, not assumed a priori.

## Output Dataset

- CSV output: `tables/dataset_iter035_topology_ground_v1.csv`.
- NPZ output: `tables/dataset_iter035_topology_ground_v1.npz`.
- Parquet output available: `False`.

## Decision

- Recommended next case: `Case B`.
- Current evidence contains both trusted FFLO trivial and topological points, so topology-aware follow-up is justified.
- Because this is sparse inherited AL sampling, the next step should target Pfaffian-margin, bulk-gap, Z2-change edges, and coverage holes rather than treating the scatter boundary as final.
