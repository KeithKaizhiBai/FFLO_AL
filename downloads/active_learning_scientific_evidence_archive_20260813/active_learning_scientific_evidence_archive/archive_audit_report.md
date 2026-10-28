# Active Learning Scientific Evidence Archive Audit

Generated: 2026-08-12T04:47:52.788899+00:00

## Scope Decision

This archive covers Active Learning scientific computation, exact-oracle development, active-learning loops, HPC execution provenance, audits, reports, and figure-generation source data. Manuscript LangGraph, claim registry, AI writing workflow, QAHE3D files, caches, and temporary development artifacts are excluded.

No original project file was deleted or modified by the archive builder.

## Inventory Summary

| Category | Meaning | Files | Bytes |
| --- | --- | ---: | ---: |
| A | MUST_KEEP | 879 | 226975147 |
| B | KEEP_IF_SMALL | 5101 | 286631001 |
| C | OPTIONAL_ARCHIVE | 52087 | 15073767246 |
| D | EXCLUDE | 5316 | 49161487451 |

The per-file classification, relevance statement, action, and archive destination are recorded in `archive_inventory.csv`.

## Stage Status

- Stage I: 2D thermodynamic baseline retained; final convergence semantics are superseded by Stage II.
- Stage II: frozen `dataset_iter035` and publication-grade thermodynamic/hard-risk audits retained.
- Stage III: final topology dataset, Pfaffian/bulk-gap evidence, and the formally converged 2D topology-boundary audit retained.
- Stage IV: final 3D dataset, fixed-mu slices, phase/topology maps, selected-point history, and audit reports retained; formal 3D convergence remains unpassed.
- Stage V-v1/v2/v2-tail/v3: final locally available datasets, reward/proposal histories, formal audits, and matched comparisons retained.
- Stage V-v4: source, config, smoke validation, and package provenance retained; a local final production dataset/report was not found.
- Stage VI: no independent formal world-model experiment was found.

Full run IDs, artifact paths, status, and missing items are in `stage_archive_status.yaml`.

## Archive Construction

- Copied source files: 2385
- Copied source bytes: 278165113
- Source-to-destination SHA256 checks: pass
- Final data formats were preserved byte-for-byte; already-compressed NPZ files were not recomputed.
- CSV-to-Parquet conversion was not applied because it would create a second representation and is unnecessary at the retained data sizes.
- Redundant ZIP/TAR bundles, candidate-score clouds, historical cumulative checkpoints, Slurm/nohup logs, LaTeX build junk, and caches were not copied.
- Transfer package: `active_learning_scientific_evidence_archive.zip`

## Figure Reproducibility

`figures_archive/FIGURE_MANIFEST.yaml` maps each selected logical figure to its purpose, source dataset, plotting script, required inputs, original source, and retained generated files. Source data and scripts are retained separately from final PNG/PDF/SVG assets.

## Reports and Code

Formal scientific reports are under `reports_archive/`. Reproduction-relevant exact-oracle, topology, acquisition, plotting, data-processing, validation, and HPC code is under `code_archive/`. Each top-level retained area has a README describing purpose, inputs, outputs, and dependencies.

## Missing or Limited Evidence

- `finite_T_phase_diagram.m`

## Scientific Caveats

- Provisional or hard-risk labels remain provisional; the archive does not promote them to definitive phases.
- Smooth 3D surfaces and report-only contours retain the caveats of their source audits.
- Stage IV and Stage V-v1/v2/v3 are not relabeled as converged by archival inclusion.
- Stage V-v4 production completion cannot be inferred from package/smoke artifacts alone.
