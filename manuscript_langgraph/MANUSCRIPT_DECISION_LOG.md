# Manuscript decision log

## M-001 - Canonical endpoints

Use Stage II `dataset_iter035`, Stage III cold-start `dataset_iter018`, and
Stage IV `dataset_iter025`. Treat the Stage III topology-pass and Stage IV
curve-fix artifacts as validation/derived layers, not competing endpoints.

## M-002 - Figure 2 initialization wording

Use “512-point cold-start random-grid design,” because that is the frozen
`run_config` value. Do not rewrite it as scrambled Sobol.

## M-003 - Figure 3 boundary semantics

Use the audited support-restricted contour. Local opposite-Z2 brackets are
support diagnostics only. Display k=8 closure and k=6 sensitivity together.

## M-004 - Figure 4 scientific position

Use exact 3D point clouds and narrow fixed-mu support. Exclude unsupported
broad-mu smooth curves. Display strict topology non-closure inside the figure
and caption; do not claim a complete or converged 3D topology atlas.

## M-005 - Candidate validation is not approval

Scientific, format, and rendered-page validation may mark a candidate as
technically valid, but only an explicit human record may pass
`GATE_FIGURE_APPROVAL`. Do not merge the candidates into production LaTeX.

## M-006 - Persistent LangGraph checkpoint

Invoke the registered graph with the actual `langgraph` runtime and a
`SQLiteSaver` checkpoint under `runs/active_learning_prl_v1/`. Resume from the
same thread only after an explicit figure-approval record. The current offline
compatibility layer remains a fallback for constrained tests, not the runtime
used for this manuscript run.

## M-007 - Supplementary candidate scope

Supplementary Figure S1 summarizes reliability and support audits only. It does
not add a new physical claim or promote Stage V/VI to Results.
