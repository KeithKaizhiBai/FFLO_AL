# Phase 5 plan — actual LangGraph invocation

## Audit

Resolve the graph ID, entrypoint, runtime dependency, state schema, runner,
checkpoint backend, and gate routing from the existing manuscript package.

## Plan

1. Use the existing 15-node graph without replacing its evidence-first order.
2. Invoke the actual `langgraph` runtime with persistent SQLite checkpoints.
3. Load all four figure metadata records in `results_by_figure_writer`.
4. Stop with `GATE_FIGURE_APPROVAL=HUMAN_WAIT`.

## Critique

The offline compatibility runner is useful for tests but is insufficient for
the requested persistent resume semantics. Figure approval must not be inferred
from the approved claims.

## Non-changes

The graph node order and scientific gate ownership remain unchanged.
