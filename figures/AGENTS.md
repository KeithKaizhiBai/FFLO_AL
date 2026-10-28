# Figure-work instructions

Before creating or modifying any artifact under this directory, read:

1. `figure_engine/README.md`
2. `FIGURE_WORKFLOW.md`
3. the complete canonical rule files under `figure_engine/constitution/`
4. the target figure contract

Every figure builder must call the figure-engine preflight before drawing and
the figure-engine lint after generation. Do not treat an automated pass as a
human gate approval. Do not change physical or numerical definitions through
figure code.

Before modifying the engine or adopting a new hard rule, run
`figure_engine/self_tests/run_figure_engine_health_check.py`. Every practical
hard rule requires a negative fixture that fails for the intended reason and a
corrected control that passes. Review `figure_engine/lessons_learned/` before
creating a new figure contract.
