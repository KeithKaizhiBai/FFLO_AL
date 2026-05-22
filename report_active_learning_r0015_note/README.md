# Active-Learning r=0.015 Note

This folder contains an English LaTeX research note summarizing the exact BdG
warm-up phase diagram and the completed active-learning refinement stage that
used normalized dense-grid radius `r = 0.015`.

The note uses five figure files copied into `figures/`:

- `fig01_original_exact_phase_diagram.png`
- `fig02_active_learning_main_boundaries.png`
- `fig03_combined_eta_phase_diagram.png`
- `fig04_active_learning_workflow.png`
- `fig05_ml_training_architecture.png`

Build on Windows with:

```powershell
.\build_note.ps1
```

or directly:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error active_learning_r0015_note.tex
pdflatex -interaction=nonstopmode -halt-on-error active_learning_r0015_note.tex
```

The note is a project report draft, not a journal-template manuscript.
