# Strict Data Figure Workflow reproduction commands

Run from the project root:

```bash
PYTHONPATH=. /opt/homebrew/bin/python3 figures/fig2_strict_data/reproduce_review_package.py
```

That single command rebuilds canonical Figure 2 tables from frozen artifacts,
regenerates SVG/PDF/PNG and the contact sheet twice, verifies deterministic PNG
rendering, runs the full Figure Engine lint, and runs the Fig.2-specific
scientific/provenance validator.

Rebuild the Fig.3/Fig.4 manifests and precontracts without rendering artwork:

```bash
PYTHONPATH=. /opt/homebrew/bin/python3 figures/strict_data_precontracts/prepare_fig3_fig4_contracts.py
```

Run the complete Figure Engine health check and unit tests:

```bash
PYTHONPATH=. /opt/homebrew/bin/python3 figures/figure_engine/self_tests/run_figure_engine_health_check.py
PYTHONPATH=. /opt/homebrew/bin/python3 -m unittest discover -s figures/figure_engine/tests -v
```

No command above runs a BdG oracle, submits HPC work, advances active learning,
retrieves external literature, or modifies manuscript LaTeX.
