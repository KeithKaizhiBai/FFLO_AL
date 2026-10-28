# Active Learning PRL figure-gate execution report

## Outcome

The scientific archive is fully integrated and the manuscript workflow has
advanced from data recovery to a complete, evidence-traced Figures 1–4
approval package. The actual 15-node LangGraph was invoked with persistent
SQLite checkpointing and stopped at `GATE_FIGURE_APPROVAL` as required.

## Scientific candidates

- Figure 1: audited BdG oracle, reliability hierarchy, and scheduler loop.
- Figure 2: 512-point random-grid cold start to a 7,434-point thermodynamic endpoint.
- Figure 3: 4,345 samples; 3,213 trusted FFLO; configured k=8 Decision A;
  k=6 sensitivity retained.
- Figure 4: 7,081-point higher-dimensional exploration with all four strict
  3D topology ratios visibly above threshold.
- Supplementary Figure S1: support and sensitivity audit summary.

## Validation

- Archive: 2,396/2,396 internal checksums pass.
- Curated artifact manifest: 65 present / 0 missing.
- Main-figure scientific/format validation: 29/29 checks pass.
- Approval PDF: 6/6 pages rendered and visually inspected.
- Graph validator: pass; automated tests: 16 passed.

## Gate state

- `GATE_SCOPE_APPROVAL = approved`
- `GATE_CLAIM_APPROVAL = approved_with_modifications`
- `GATE_FIGURE_APPROVAL = pending_human_approval`
- `INTRO_CITATION_LOOP_GATE = permission_not_granted`

The next action is scientific review of the figure package. Validation is not
approval. No production manuscript LaTeX, external Introduction retrieval, new
exact physics, HPC submission, frozen-data change, or QAHE3D modification was
performed.
