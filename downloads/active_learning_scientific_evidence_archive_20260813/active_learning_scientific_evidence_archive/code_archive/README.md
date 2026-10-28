# Scientific Code Archive

This directory contains source code required to reproduce the retained active-learning scientific evidence. It excludes caches, debug scripts, one-time migrations, model caches, and full historical run outputs.

- `canonical/`: current exact-oracle, phase-label, acquisition, plotting, validation, and report code.
- `stage_iii/`: topology-aware 2D implementation snapshot and package support.
- `stage_iv/`: 3D topology-aware implementation snapshot and package support.
- `stage_v_v3/`: sparse-arc acquisition implementation and production configuration.
- `stage_v_v4/`: phase-structure graph acquisition implementation and validated package configuration.

Inputs are the frozen datasets under `../data_archive/`. Outputs are reports, tables, and figures under `../reports_archive/` and `../figures_archive/`. Python/CUDA/Slurm dependencies are documented in the retained package READMEs and manifests.
