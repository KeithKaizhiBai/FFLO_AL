# Repository audit

Date: 2026-08-13 (updated after scientific-evidence archive recovery)

## Stage I — 2D thermodynamic discovery

The scientific-evidence archive supplies the final Stage I `dataset_iter031`,
run sidecars, reports, and supporting figures/source. The comprehensive Stage
I--V report and project memory describe cold-start/random or Sobol
thermodynamic discovery. A standalone Stage-I-only final report is absent.

Maturity: `supported_history`; proposed destination: Methods/background and
Supplement.

## Stage II — exact-oracle reliability and 2D thermodynamic convergence

The Phase-II decision log and Q&A record Decision A and a passed publication
boundary audit. The frozen endpoint is `dataset_iter035`, with 7434 samples:
1867 normal, 715 uniform-SC, and 4852 FFLO. The endpoint, paired CSV, run
sidecars, report, reproduction manifest, tables, figure data/source, and
original figures are recovered.

Maturity: `frozen_supported`; approved destination: main text and Supplement.

## Stage III — 2D topology-aware cold-start full loop

The topology convergence Q&A records Decision A for
`active_phase_topology_from_scratch_full_loop_v2`: 4345 final unique samples,
3213 trusted gapped FFLO points, 2290 cFFLO points, and 923 tFFLO points. The
configured audit passes all main gates, but the (k=6) sensitivity coverage
value is slightly above the nominal threshold. The final dataset,
per-iteration convergence CSVs, figure data/source, report, and original
figures are recovered. Redundant intermediate cumulative NPZ checkpoints were
omitted. The decision JSON stores the paired CSV hash under an NPZ-labeled
source field; NPZ/CSV values agree.

Maturity: `frozen_supported_with_caveat`; approved destination: central main
result, pending figure approval/redraw.

## Stage IV — 3D extension

Documentation and Q&A are present. The returned run discovered three-dimensional
thermodynamic/topological structure, but topology coverage, trusted surprise,
and component-stability gates did not pass. Hidden Stage III fixed-μ validation
was also unavailable.

The signed-off return records support higher-dimensional exploration and the
presence of thermodynamic/topology-aware structures. The final endpoint,
return-report/audit bundles, figure data/source, and original figures are
recovered.

Maturity: `supported_with_caveat`; approved destination: main-text 3D extension
and candidate Figure 4. Strict topology non-closure must be stated visibly.

## Stage V — acquisition redesign

V3/V4 frozen datasets, reports, figures, tables, and matched offline audit are
available locally. The matched audit gives Decision D: V3 and V4 trade off
coverage and sequential surprise, and neither is established as superior or
publication-ready.

Maturity: `method_development_not_main_claim`; proposed destination:
Discussion/Outlook.

## Stage VI — world-model validation

The validation packages are locally complete, including Markdown/PDF reports,
tables, figures, decisions, and tests. Stage VI-0 is Decision C and Stage
VI-0.2 is Decision D because the frozen packages lack spectral fingerprints.
No EIG acquisition is authorized.

Maturity: `negative_or_incomplete_method_result`; proposed destination:
Discussion/Outlook only.

## Repository-level conclusion

The scientific narrative and bounded claims are frozen, and the evidence
needed to redraw Figures 2--4 is local. Figure approval and versioned redraws
remain outstanding; no result is inferred from prose alone.

## Requested artifact-class coverage

The manifest includes representative reports and audit PDFs, frozen datasets,
convergence records, phase/topology maps, report/plotting sources, LaTeX report
sources, the AI-assisted workflow note, the recorded task prompt, and the local
LangGraph source. No pre-existing paper manuscript draft was found in the
exported package; `manuscript_langgraph/manuscript/` remains intentionally
empty except for its README. Recovered Stage II--IV publication bundles are
stored byte-for-byte under `manuscript_langgraph/recovered_artifacts/`.
