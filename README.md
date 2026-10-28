# Reliability-aware active learning for FFLO phase spaces

Computational BdG/FFLO research code, retained numerical evidence, Figure Engine
assets and a manuscript draft. The active-learning model schedules exact calls;
it does not replace the physical solver.

## Current manuscript

Author: **Kai-Zhi Bai**  
Email: **kzbai@connect.hku.hk**  
Department of Physics, The University of Hong Kong, Pokfulam Road, Hong Kong, China

The 2026-10-28 candidate is in
[`output/pdf/manuscript_author_revision_v9/`](output/pdf/manuscript_author_revision_v9/):

- [Double-column main text](output/pdf/manuscript_author_revision_v9/manuscript_double_column.pdf)
- [Single-column review text](output/pdf/manuscript_author_revision_v9/manuscript_single_column_review.pdf)
- [Supplemental Material](output/pdf/manuscript_author_revision_v9/supplement.pdf)
- [Author review / unresolved choices](output/pdf/manuscript_author_revision_v9/AUTHOR_REVIEW.md)

This version adds the author identity and project-repository information to v8.
Scientific content, bibliography, numerical data and figure artwork remain
unchanged. It is an initial draft, **not a submission-ready or fully approved
manuscript**. In particular, the higher-dimensional result does not establish a
formally closed continuous 3D topology surface. Scientific and figure-review
gates remain pending.

The clean LaTeX source is under the candidate's `source/`. To rebuild all three
PDFs together, install Python with `pypdf`, pdfLaTeX, BibTeX and REVTeX 4.2, then:

```sh
python3 output/pdf/manuscript_author_revision_v9/source/build.py
```

This command builds documents only. It does not run BdG or retrain a model.
Markdown companions, numerical CSV tables and figure PNGs are provided beside
the PDFs. The original v7/v8 review bundles are retained for provenance checks.

## Program and evidence layout

| Location | Purpose |
| --- | --- |
| `eta_phase_diagram_cuda.py`, `tfflo_1d_cuda.py` | BdG scientific kernels |
| `ml_phase/`, `scripts/`, `hpc_packages/` | Current local active-learning/oracle and launch code |
| `docs/`, `MODEL_SPEC.md` | Model, numerical rules, decisions and project history |
| `figures/figure_engine/` | Figure contracts, style rules, builders and validators |
| `figures/fig*_strict_data*/canonical/` | Saved data used in quantitative manuscript figures |
| `manuscript_langgraph/`, `manuscript_variants/` | Evidence-aware manuscript workflow and variants |
| `manuscript_langgraph/recovered_artifacts/` | Recovered endpoint and convergence evidence |
| `downloads/active_learning_scientific_evidence_archive_20260813/active_learning_scientific_evidence_archive/` | Historical `code_archive/`, `data_archive/` and manifests |
| `world_model_descriptor_dev/`, `stage_vi0_*/` | Exploratory world-model validation code and recorded diagnostics |
| `release_metadata/20261028/` | Exact upload selection, byte hashes and exclusions |

Current source and historical snapshots are intentionally distinct. Retained
configuration files are evidence of saved settings, not proof that every
historical executable invocation has been recovered. GPU programs require a
compatible PyTorch/CUDA environment; launch paths and scheduler settings must
be adapted to the user's own cluster. Nothing in the manuscript build installs
or invokes the physical solver.

## Reproduction limits

- Stage I is thermodynamic discovery; Stage II supplies oracle-reliability
  evidence; Stage III adds audited topology-aware sampling; Stage IV is a
  higher-dimensional exploration with explicit nonclosure caveats.
- Stage V learned acquisition is exploratory and does not establish superiority.
  Stage VI world-model/EIG work is an outlook, not validated EIG acquisition.
- Separate V3/V4 endpoint exports are included at their recorded relative paths.
  Their complete sequence of intermediate cumulative datasets is not part of
  this upload selection, so full Stage VI sequential replay is not promised.
- Some Figure Engine contracts contain absolute historical source paths and
  older comparison-asset dependencies. They need path rebinding before
  cross-machine regeneration; the embedded manuscript figure PDFs are portable.
- Historical archive status manifests may predate the newer V4/VI exports.
  The release inventory, not an old status file alone, identifies included data.
- Initial-draft figure readability and mechanism/Ansatz questions remain listed
  in the author-review document. Passing software tests is not scientific approval.
- No archival DOI or new license grant is implied by this repository update.

Virtual environments, credentials, cluster-login documents, caches and the
duplicated 163-MiB archive ZIP are excluded. Existing repository files outside
the update selection are preserved rather than silently removed.
