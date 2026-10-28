# Exact LangGraph commands

From the project root:

```bash
world_model_descriptor_dev/.venv/bin/python manuscript_langgraph/scripts/run_active_learning_prl_v1.py
```

Validate the graph and tests:

```bash
world_model_descriptor_dev/.venv/bin/python manuscript_langgraph/scripts/validate_workflow.py
world_model_descriptor_dev/.venv/bin/python -m pytest manuscript_langgraph/tests
```

After, and only after, a human figure-approval record has been written:

```bash
world_model_descriptor_dev/.venv/bin/python manuscript_langgraph/scripts/resume_after_figure_approval.py
```

The resume command is prepared but intentionally not run in this phase.
