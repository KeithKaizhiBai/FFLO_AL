# Changed-files manifest

New variant/configuration files are confined to:

- `manuscript_langgraph/publication_profiles/prl_integrated_main_text.yaml`
- `manuscript_langgraph/review_agent/prl_integrated_variant.py`
- `manuscript_langgraph/review_agent/graphs/prl_integrated_manuscript_graph.py`
- `manuscript_langgraph/scripts/build_prl_integrated_variant.py`
- `manuscript_langgraph/review_agent/state_schema.py` (optional variant fields only)
- `manuscript_langgraph/tests/test_prl_integrated_variant.py`
- `manuscript_langgraph/runs/active_learning_prl_integrated_v1/`
- `manuscript_variants/prl_integrated/`
- `output/pdf/manuscript_prl_integrated_v1/`
- project summary/decision/worklog/report-QA/handoff documentation for this new branch

The optional state schema gained variant fields. The parent run, intermediate review, frozen Results/captions, Supplement scope, and frozen figure assets were not edited.
