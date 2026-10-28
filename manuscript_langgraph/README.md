# Evidence-first PRL manuscript workflow

This directory is an isolated manuscript-planning and orchestration package for
the Active Learning FFLO project. It does not modify the BdG solver, active
learning code, frozen datasets, or any QAHE3D file.

The workflow is deliberately evidence-first. Its generation order is:

```text
artifact/provenance audit
-> publication-readiness gate
-> scope and claim freeze
-> figure/evidence contracts
-> Results by figure
-> Methods
-> Supplement
-> Discussion
-> Introduction and citations
-> Abstract/title
-> consistency, citation, and PRL-length gates
-> LaTeX integration
-> final human approval
```

The graph stops at human gates. A raw unapproved test invocation stops at
`GATE_SCOPE_APPROVAL`. With scope, claims, and figures approved, the persisted
current-state runner now stops at `GATE_RESULTS_HUMAN_REVIEW`; later staged
invocations stop independently at the Introduction retrieval gate and
`GATE_FINAL_MANUSCRIPT_APPROVAL`.

## Current status

All Figures 1--4 are scientifically frozen and the overall Figure gate is
approved with the permanent Figure 3 sensitivity and Figure 4 nonclosure
caveats. The existing SQLite thread was resumed through the real Results node,
which generated the evidence-traced Results/caption package under
`output/pdf/manuscript_results_review_v1/`. A minimal compatible
`results_human_review_gate` now prevents Methods from running until explicit
scientific approval. Current stop: `GATE_RESULTS_HUMAN_REVIEW=HUMAN_WAIT`;
Results are not frozen and external Introduction retrieval remains forbidden.

## Entry points

- `langgraph.json`: LangGraph registration draft.
- `review_agent/graphs/prl_manuscript_graph.py`: graph object.
- `review_agent/state_schema.py`: shared state and paragraph metadata schema.
- `review_agent/results_package.py`: graph-invoked Results/caption/audit builder.
- `node_contracts/`: node inputs, outputs, gates, and prohibitions.
- `audits/`: repository, provenance, scope, and readiness outputs.
- `registries/`: claims, evidence, and figure contracts.
- `vertical_slice_stageIII/`: first Results-by-figure approval package.
- `runs/active_learning_prl_v1/checkpoints/checkpoint_results_review.json`:
  current resumable human-gate checkpoint export.

Run validation from the project root:

```text
world_model_descriptor_dev/.venv/bin/python -m pytest manuscript_langgraph/tests -q
python3 manuscript_langgraph/scripts/build_artifact_manifest.py
python3 manuscript_langgraph/scripts/run_current_workflow.py
python3 manuscript_langgraph/scripts/validate_workflow.py
```
