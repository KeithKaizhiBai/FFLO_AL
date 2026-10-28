# LangGraph invocation audit

## Resolved graph

- Graph ID: `evidence_first_prl_workflow`
- Registry: `manuscript_langgraph/langgraph.json`
- Entrypoint: `review_agent/graphs/prl_manuscript_graph.py:graph`
- Node count: 15
- Runtime used for this run: installed `langgraph` package, not the offline
  compatibility runner
- Persistent checkpoint backend:
  `langgraph.checkpoint.sqlite.SqliteSaver`

## Environment

- Python: `world_model_descriptor_dev/.venv/bin/python`
- Declared dependencies: `langgraph>=0.2` and
  `langgraph-checkpoint-sqlite>=3.0`
- No model/API key is required because the current nodes are deterministic
  evidence and gate functions.

## State and checkpoint

- Initial state:
  `manuscript_langgraph/runs/active_learning_prl_v1/initial_state.json`
- SQLite checkpoint:
  `manuscript_langgraph/runs/active_learning_prl_v1/checkpoints/manuscript.sqlite`
- Human-readable checkpoint export:
  `manuscript_langgraph/runs/active_learning_prl_v1/checkpoints/checkpoint_figure_gate.json`
- Thread ID: `active_learning_prl_v1`

The graph is compiled with an interrupt after `results_by_figure_writer` and
must report `GATE_FIGURE_APPROVAL=HUMAN_WAIT`. Scope and claims are read as
approved/approved-with-modifications; figures are not auto-approved.

## Resume semantics

After an explicit figure approval record exists, the resume helper opens the
same SQLite database and thread, updates the state as the
`results_by_figure_writer` node, and invokes with `input=None`. This schedules
the downstream Methods/Supplement/Discussion nodes without rebuilding the
pre-gate evidence package. The helper refuses to run while approval is absent.

## Human interrupt semantics

- Current stop: `GATE_FIGURE_APPROVAL`
- Next planned stop after approval: `INTRO_CITATION_LOOP_GATE`
- External retrieval remains false.
- Final manuscript approval remains human-only.
