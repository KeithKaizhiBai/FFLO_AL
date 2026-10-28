# Publication-readiness audit

## Verdict

The publication-critical Stage II--IV evidence gaps are closed by the verified
scientific-evidence archive. The project is data-ready for reproducible
Figures 2--4 redesign and manuscript expansion, but the workflow remains at
`GATE_FIGURE_APPROVAL`; it is not yet an approved PRL submission package.

## Recovered and verified

- Stage II `dataset_iter035.npz`, full Phase II report, reproduction manifest,
  tables, figure data, plotting source, and original generated assets.
- Stage III `dataset_iter018.npz`, topology-pass ground dataset, formal
  convergence report/decision/tables, figure data, plotting source, and
  original generated assets.
- Stage IV `dataset_iter025.npz`, full return report, reproduction manifest,
  convergence and slice audits, tables, plotting source, and original 3D
  figures.

The archive ZIP SHA-256 is
`8f3402bb7dce090eac3504adff3cf3cfd55ab24564a309c6042d94bf3311027d`.
All 2,396 internal checksum entries pass. Endpoint hashes are:

- Stage II: `f2d751d0cfae8a3c63da786109903ffd1969a3f90c20a17d92665d8f252e2700`;
- Stage III: `6583230495ad9e3eb45b0cff034d5525fa443bd43b1b31a754defeaabdfb5ab4`;
- Stage IV: `7acab805115bcfcae56e94e2541bfdaa727de7767f2d7dd34330f163e2695842`.

The three endpoint NPZ/CSV pairs agree for every stored array field. Sample and
class counts reproduce the signed-off report records.

## Remaining evidence limitations

- The archive intentionally excludes redundant intermediate cumulative NPZ
  checkpoints. Stage III retains the derived per-iteration convergence tables,
  boundary points, report, and original figures needed for manuscript figure
  regeneration, but a ground-up rerun of the entire convergence audit from all
  raw cumulative checkpoints is not locally possible.
- In the frozen Stage III decision JSON, `source_dataset` names the NPZ while
  `source_dataset_sha256` is the correct hash of `dataset_iter018.csv`. The
  per-iteration provenance table correctly names that CSV, and every archived
  NPZ array equals its CSV counterpart. The frozen JSON is not edited; the
  metadata-label discrepancy must be documented.
- A multi-seed benchmark, matched random/Sobol baseline, and independent hidden
  ground truth remain optional improvements and are not required by the
  approved physics-first claim scope.

## Required scientific caveats

The Stage III configured $k=8$ audit passes coverage, while $k=6$ gives
$p_{95}=0.006269$, slightly above the $0.00625$ threshold. Report both values
and retain diagnostic-only bracket semantics.

The Stage IV 3D topology surface is not formally closed by the strict
convergence criteria used for the 2D result. Figure 4 may demonstrate
higher-dimensional exploration that reveals thermodynamic and topology-aware
structures, never a complete or fully converged 3D atlas.

## Current gate

`GATE_SCOPE_APPROVAL` and `GATE_CLAIM_APPROVAL` are approved.
`PUBLICATION_READINESS_GATE` now passes at the evidence level. The active human
stop remains `GATE_FIGURE_APPROVAL`, covering the Stage III vertical slice and
Figures 1--4 contracts. External Introduction retrieval remains unauthorized.
