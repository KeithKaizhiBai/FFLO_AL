# Scientific-evidence archive and manuscript progress

Date: 2026-08-13

## Question

Does `active_learning_scientific_evidence_archive.zip` contain the missing
Stage II--IV evidence needed for the approved manuscript, and what is the
current writing/workflow status?

## Answer

Yes. The archive closes the raw-data gaps for Figures 2--4. It passed ZIP
safety and decompression checks, and all 2,396 files named by its internal
checksum list verified. Its SHA-256 is
`8f3402bb7dce090eac3504adff3cf3cfd55ab24564a309c6042d94bf3311027d`.

Recovered evidence includes:

- Stage II final `dataset_iter035.npz`, full Phase II report, reproduction
  manifest, tables, figures, and figure data;
- Stage III final `dataset_iter018.npz`, topology-pass ground dataset, and the
  full 37-file convergence/figure bundle;
- Stage IV final `dataset_iter025.npz`, return report, reproduction manifest,
  convergence and slice audits, tables, source figures, and
  `phase_3d_jakt_view.png`.

Endpoint validation:

```text
Stage II: 7,434 samples; endpoint SHA-256 f2d751d0...e2700
Stage III: 4,345 samples; endpoint SHA-256 65832304...fab4
Stage IV: 7,081 samples; endpoint SHA-256 7acab805...5842
```

Every NPZ array matches the paired CSV. Stage II and IV endpoint hashes match
their reproduction manifests. The Stage III frozen decision JSON labels its
source as NPZ but stores the paired CSV hash; the input-provenance table is
correct and the arrays agree, so the discrepancy is documented without
altering the frozen record.

The archive intentionally excludes redundant intermediate cumulative NPZ
checkpoints. It nonetheless contains the derived convergence tables, boundary
data, figures, decision record, report, and source required for manuscript
redraw and bounded claim review. It does not support a ground-up rerun of every
historical convergence step solely from raw cumulative checkpoints.

## Manuscript status

The scope and all six bounded claims are approved. The rebuilt 51-row evidence
manifest has 51 present and zero missing rows. Artifact/provenance readiness and
the figure evidence contract pass. The workflow correctly stops at
`GATE_FIGURE_APPROVAL = HUMAN_WAIT`.

Therefore the paper is **data-ready for Figures 2--4**, but not yet
figure-approved or submission-ready. No final manuscript-ready redraw has been
made. External Introduction retrieval remains unauthorized. After figure
approval, the next task is to redraw Figures 1--4 and expand the Results text
figure by figure, while retaining the Stage III sensitivity/provenance notes
and Stage IV strict-convergence caveat.
