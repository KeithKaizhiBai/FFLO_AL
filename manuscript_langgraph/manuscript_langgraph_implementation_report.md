# Evidence-first PRL LangGraph implementation report

Date: 2026-08-07

> Status note (2026-08-12): this document records the original architecture
> baseline. Its Stage I--III recommendation and pending scope/claim status are
> superseded by `audits/MANUSCRIPT_SCOPE.md` and
> `audits/claim_approval_record_20260812.yaml`. The current approved scope spans
> Stages I--IV and the workflow stops at `GATE_FIGURE_APPROVAL`.

## Executive conclusion

An isolated evidence-first manuscript workflow has been built and revalidated
under `manuscript_langgraph/`. It preserves the supplied prompt, audits the
available evidence, proposes a scope and four-figure plan, defines
claim/evidence/figure registries, registers a 15-node LangGraph, and retains a
Stage III Results-by-figure vertical slice as an approval package. It does not
write a full PRL or modify any scientific code, physical definition, frozen
data, or QAHE3D file. The default run stops at `GATE_SCOPE_APPROVAL`.

## 1. Repository audit

The manifest contains 37 curated artifacts: 34 are present and 3 are explicitly
missing. It covers representative reports/audit PDFs, frozen datasets,
convergence records, phase/topology maps, plotting/report sources, LaTeX report
sources, AI-workflow documentation, the recorded prompt, and LangGraph source.
No pre-existing paper manuscript draft was found. The three missing records are
the Stage II endpoint, Stage III endpoint, and Stage III convergence-report
bundle; none was reconstructed from prose.

## 2. Scientific scope assessment

The strongest current paper is:

> Cold-start active learning reconstructs thermodynamic and topological FFLO
> phase diagrams using a numerically audited BdG oracle.

The central result should be the formally audited two-dimensional Stage III
cFFLO/tFFLO reconstruction. Active learning is presented as a scheduler for
exact BdG calls, not a replacement for the physics oracle.

Stages I–III can support the main narrative. Stage I supplies the discovery
problem, Stage II supplies oracle reliability and thermodynamic boundary
closure, and Stage III supplies the central topology-aware result. Stage IV is
conditional and non-converged; Stage V belongs in Discussion with no V3/V4
superiority claim; Stage VI belongs in Outlook because EIG readiness and the
requested spectral descriptors are absent.

## 3. Manuscript workflow design

The graph generates in evidence order rather than reading order:

```text
artifact/provenance -> readiness -> scope/claim freeze -> figure contracts
-> Results by figure -> Methods -> Supplement -> Discussion
-> Introduction/citations -> Abstract/Title -> audits -> LaTeX -> final approval
```

The staged human boundaries are `GATE_SCOPE_APPROVAL`,
`GATE_CLAIM_APPROVAL`, `GATE_FIGURE_APPROVAL`, permission-gated Introduction
retrieval, and `GATE_FINAL_MANUSCRIPT_APPROVAL`. Every Results paragraph stores
observation, interpretation, evidence source, figure reference, caveats, and
unsupported-sentence flags.

## 4. Main-text stages

Stages I–III can support the main narrative. Stage I supplies the discovery
problem, Stage II supplies oracle reliability and thermodynamic boundary
closure, and Stage III supplies the central topology-aware result. Stage I
should remain compact because its original raw bundle is not local.

## 5. Supplement, Discussion, and Outlook

Stage IV is conditional: it may appear as a caveated 3D demonstration, but its
topology surface is not formally converged. Stage V belongs in Discussion as a
method-development trade-off; neither V3 nor V4 is established as superior.
Stage VI belongs in Outlook because the world model is not EIG-ready and the
required spectral descriptors are absent.

## 6. Publication-critical evidence gaps

The local checkout lacks the Stage II/III final raw endpoints and the Stage III
convergence/figure bundle. It also lacks a matched-budget random/Sobol baseline,
a multi-seed benchmark, and an independent hidden-ground-truth evaluator. The
raw-bundle gap blocks final figure reproduction. The benchmark gaps block a
strong sample-efficiency claim but can be avoided by choosing a physics-first
scope.

## 7. Four-figure organization and regeneration needs

1. Physical model, exact oracle, label hierarchy, and AL loop.
2. Two-dimensional thermodynamic discovery and convergence with hard-risk
   uncertainty visible.
3. Stage III cFFLO/tFFLO map and formal convergence audit, including the
   (k=6) coverage sensitivity caveat.
4. Preferably a future matched-budget/multi-seed benchmark; currently, only a
   visibly caveated Stage IV 3D demonstration is defensible.

Figures 2 and 3 remain blocked pending their frozen raw bundles. Figure 1 is
ready for a reproducible vector redraw. Figure 4 requires a scope decision and
must be omitted if neither a caveated Stage IV demonstration nor an authorized
matched benchmark is available. No final scientific figure may be manually
edited without its source script.

## 8. QAHE3D assets that are genuinely reusable

The reusable elements are the Introduction claim/citation contract,
permission-gated retrieval stages, placement audit, paragraph blueprint,
agent ownership/work-order concepts, and editorial style rules. Their upstream
paths and SHA-256 hashes are recorded. QAHE3D code is not imported at runtime,
because its state, paths, claims, and domain policies are project-specific.

## 9. Components that had to be created

New components include artifact/readiness audits, scope and scientific claim
freeze, evidence and figure registries, Results-by-figure paragraph metadata,
cross-section consistency, PRL length/claim gates, LaTeX merge approval, and
the final human-approval boundary.

## 10. Stage III vertical slice

The slice runs end-to-end to `GATE_FIGURE_APPROVAL`. Its three paragraphs have
machine-readable observation, interpretation, claim, evidence, panel, caveat,
and unsupported-sentence fields; all unsupported-sentence lists are empty. The
fragment compiles successfully. It remains a QA artifact because final figure
sources are missing.

## 11. Readiness and next execution steps

The architecture is ready for scientific review, but full manuscript expansion
should not begin yet. First, approve or revise the physics-first Stages I–III
scope. Then review the claim registry and Stage III vertical slice. In parallel,
recover the frozen Stage II/III raw bundles without recomputation. Only after
those gates pass should Figures 1–3, Methods, Supplement, Discussion, and the
permission-gated citation loop be expanded. A methods-efficiency positioning
would additionally require separately authorized matched baselines, multiple
seeds, and hidden ground truth.

## Validation

- Artifact manifest: 37 curated rows, 34 present and 3 missing.
- Graph: 15 registered nodes and 16 gate records on a fully traversed test
  invocation; the unapproved invocation stops at scope.
- Automated tests: 14 passed.
- Prompt integrity: reverified against the SHA-256 recorded in
  `prompts/README.md`.
- Stage III LaTeX QA: compiled successfully to two pages and visually checked.
- No new exact calculation, external retrieval, or production manuscript merge
  was performed.
