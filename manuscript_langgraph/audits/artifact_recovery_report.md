# Frozen artifact recovery report

Date: 2026-08-13  
Mode: recovery from the supplied scientific-evidence archive; no physics
recomputation.

## Outcome

`RECOVERY_COMPLETE_FOR_REQUESTED STAGE II--IV PUBLICATION ARTIFACT GROUPS`

The user-supplied `active_learning_scientific_evidence_archive.zip` closes the
six previously recorded recovery targets:

1. Stage II final `dataset_iter035.npz`;
2. Stage II reproduction manifest, tables, report, figure data, and generated
   assets;
3. Stage III final `dataset_iter018.npz`;
4. Stage III convergence decision, report, tables, figure data, and generated
   assets;
5. Stage IV final `dataset_iter025.npz` and full return/audit bundles;
6. Stage IV source figures, including `phase_3d_jakt_view.png`.

Provenance-preserving copies are under `recovered_artifacts/`. The full archive
is unpacked under
`downloads/active_learning_scientific_evidence_archive_20260813/`.

## Integrity and identity checks

- ZIP SHA-256:
  `8f3402bb7dce090eac3504adff3cf3cfd55ab24564a309c6042d94bf3311027d`;
- ZIP member test: pass;
- path traversal, duplicate-member, and symlink checks: pass;
- internal archive checksums: 2,396/2,396 pass;
- Stage II endpoint SHA-256:
  `f2d751d0cfae8a3c63da786109903ffd1969a3f90c20a17d92665d8f252e2700`,
  matching its reproduction manifest;
- Stage III endpoint SHA-256:
  `6583230495ad9e3eb45b0cff034d5525fa443bd43b1b31a754defeaabdfb5ab4`;
- Stage IV endpoint SHA-256:
  `7acab805115bcfcae56e94e2541bfdaa727de7767f2d7dd34330f163e2695842`,
  matching its reproduction manifest.

Every stored NPZ array agrees exactly with the paired CSV for all 33 Stage II,
51 Stage III, and 52 Stage IV array fields. Sample counts are 7,434, 4,345,
and 7,081, respectively, and class counts agree with the signed-off reports.

## Stage III metadata note

The frozen convergence decision JSON names an NPZ in `source_dataset` but puts
the SHA-256 of `dataset_iter018.csv` in `source_dataset_sha256`. This is a field
labeling inconsistency, not a data mismatch: the per-iteration provenance table
correctly names the CSV and gives the same hash, and all NPZ arrays equal their
CSV columns. The frozen JSON was not edited.

## Reproducibility boundary

The archive deliberately excludes redundant intermediate cumulative NPZ
checkpoints. It retains the final endpoints, derived convergence/figure tables,
plotting source, reports, and original figures needed for manuscript figure
regeneration. It does not support a ground-up rerun of every convergence audit
from all historical raw cumulative endpoints without restoring those optional
checkpoints.

No physical calculation, interpolation, archive mutation, dataset merge, or
cross-stage substitution was performed.
