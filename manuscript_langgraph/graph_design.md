# Graph design

## Scientific invariant

The graph may organize evidence and draft prose, but it cannot promote a
scientific claim. Claim status changes require an explicit human decision.
Frozen numerical outputs remain the authority.

## Generation order

```text
artifact_provenance_audit
  -> publication_readiness_gate
  -> claim_freeze
  -> figure_evidence_contract
  -> results_by_figure_writer
  -> methods_writer
  -> supplement_builder
  -> discussion_scope_guard
  -> intro_citation_loop
  -> abstract_title_generator
  -> cross_section_consistency_audit
  -> citation_audit
  -> prl_length_claim_gate
  -> latex_integration
  -> final_human_approval
```

This differs from reading order because Results must be grounded before the
Introduction positions the contribution, and the Abstract/Title must be
derived from already-frozen claims.

## Gates

- `GATE_SCOPE_APPROVAL`: blocks claim and manuscript generation.
- `GATE_CLAIM_APPROVAL`: blocks final scientific wording.
- `GATE_FIGURE_APPROVAL`: blocks expansion from the Stage III vertical slice
  into the approved Figures 1--4 workflow and all downstream LaTeX integration.
- `INTRO_CITATION_LOOP_GATE`: blocks live citation retrieval until separately
  authorized.
- `GATE_FINAL_MANUSCRIPT_APPROVAL`: never automated.

## Results-by-figure contract

For each figure, the subgraph loads a figure contract, reads registered
evidence, separates observation from interpretation, maps sentences to frozen
claims, drafts a caption and Results paragraphs, and emits paragraph metadata.
Any sentence without evidence is flagged instead of silently softened or
invented.

## Current controlled stopping point

The package implements the architecture and a Stage III vertical slice. The
default unapproved invocation stops at `GATE_SCOPE_APPROVAL`; the persisted
approved invocation passes scope and claims and stops at
`GATE_FIGURE_APPROVAL`. Test invocations stop independently at all human gates.
Full manuscript drafting remains blocked by figure approval and incomplete
publication-critical Figure 2--4 source bundles.
