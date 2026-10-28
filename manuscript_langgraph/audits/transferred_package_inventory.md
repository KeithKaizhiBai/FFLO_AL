# Transferred-package inventory and data-presence decision

Updated: 2026-08-13  
Mode: evidence extraction and integrity audit; no physics recomputation.

## Short answer

The earlier workspace transfer was a minimal code/HPC handoff and did not
contain the frozen Stages II--IV publication evidence. The newly supplied
`active_learning_scientific_evidence_archive.zip` closes those critical gaps.

The necessary source evidence for manuscript Figures 2--4 is now local:

- final Stage II, Stage III, and Stage IV endpoints;
- paired CSVs and run identities;
- formal reports, decisions, tables, and reproduction manifests where defined;
- figure data, plotting source, and original generated PNG/PDF assets.

Therefore the paper is now **data-ready for reproducible figure redesign**, but
not figure-approved: `GATE_FIGURE_APPROVAL` remains a separate human decision.

## Archive integrity

The 163 MB ZIP was unpacked into
`downloads/active_learning_scientific_evidence_archive_20260813/`.

- ZIP SHA-256:
  `8f3402bb7dce090eac3504adff3cf3cfd55ab24564a309c6042d94bf3311027d`;
- 2,397 archive members; no duplicates, unsafe paths, or symlinks;
- ZIP member test passed;
- 2,396/2,396 internal SHA-256 entries passed;
- 2,397 files occupy approximately 289 MB after extraction.

Publication-critical copies were placed under
`manuscript_langgraph/recovered_artifacts/` without changing file bytes.

## Stage-by-stage decision

| Stage | Material now present | Remaining limitation | Manuscript consequence |
|---|---|---|---|
| I | final `dataset_iter031`, run sidecars, reports, source | no standalone Stage-I-only final report | sufficient proof-of-concept support |
| II | `dataset_iter035`, reproduction manifest, report, tables, figures, figure data/source | redundant intermediate checkpoints omitted | Figure 2 source gap closed |
| III | `dataset_iter018`, topology-pass ground dataset, convergence decision/tables/report/figures, figure data/source | redundant intermediate NPZ checkpoints omitted; one hash-field label note | Figure 3 source gap closed |
| IV | `dataset_iter025`, reproduction manifest, return/convergence/slice audits, tables, 3D figures/data/source | strict 3D convergence and hidden fixed-$\mu$ reference remain unresolved | Figure 4 source gap closed, scientific caveat retained |
| V | existing local V3/V4 matched evidence plus archive V1--V3 history | archive itself lacks final V4 production result | adequate for Discussion-only conclusion |
| VI | existing workspace Stage VI offline reports | archive intentionally contains no independent Stage VI experiment; spectral descriptor remains absent | Outlook only |

## Numerical identity checks

The endpoint sample counts and phase classes reproduce the signed-off records:

- Stage II: 7,434 = 1,867 normal + 715 uniform-SC + 4,852 FFLO;
- Stage III: 4,345 = 746 normal + 386 uniform-SC + 3,213 FFLO;
- Stage IV: 7,081 = 2,350 normal + 141 uniform-SC + 4,590 FFLO.

Stage III contains 2,290 cFFLO and 923 tFFLO points within the 3,213 trusted
gapped FFLO set. Stage IV contains 3,265 trivial and 1,466 topological trusted
points. NPZ/CSV pairs agree exactly for all stored fields.

## Stage III metadata note

The convergence decision JSON's `source_dataset` string names the NPZ, but its
recorded SHA-256 is the correct hash of the paired CSV. The per-iteration
provenance table names the CSV correctly, and the NPZ/CSV values agree exactly.
This is retained as a frozen metadata-label discrepancy, not treated as a
scientific mismatch.

## Publication consequence

The three raw-data blockers in the publication-gap matrix are closed. The
evidence-level readiness gate can pass. Remaining work is editorial and
figure-production work under human governance:

- approve or revise the Figures 1--4 contracts;
- redraw manuscript-ready panels from the recovered evidence;
- preserve the Stage III $k=6$ sensitivity and bracket caveats;
- preserve the Stage IV strict non-convergence caveat;
- keep external Introduction retrieval disabled until separately authorized.

No physics was recomputed and no Stage V data were substituted.
