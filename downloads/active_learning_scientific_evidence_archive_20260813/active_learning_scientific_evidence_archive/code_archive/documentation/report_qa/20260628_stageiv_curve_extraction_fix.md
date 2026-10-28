# Stage IV-A Curve-Extraction Fix

Date: 2026-06-28

## Question

After the mu-slice thermodynamic-boundary audit classified the suspicious
high-mu wide-bin normal/SC behavior as a broad-bin projection artifact, the
next task was to repair the figure and report pipeline.  The wide-bin atlas
should be shown as a projection diagnostic, narrow fixed-mu atlases should be
generated, local bracket support should be visible, and unsupported smooth
curve segments should be removed.

## Inputs

- Frozen Stage IV-A returned dataset: `dataset_iter025`
- Source prompt: `active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/docs/Report_Re_prompt.md`
- Previous audit output: `stageiv_mu_slice_boundary_audit_local`
- Normal-state single-band preflight scan

No exact calculation, active-learning continuation, or label modification was
performed.

## Output Location

```text
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_curve_extraction_fix_local/
```

Key files:

```text
stageiv_curve_extraction_fix_report.md
stageiv_curve_extraction_fix_report.tex
stageiv_curve_extraction_fix_report.pdf
stageiv_curve_extraction_fix_summary.json
stageiv_curve_extraction_fix_config.yaml
decision_log.md
tables/*.csv
figures/*.png
figures/*.pdf
```

## Main Result

```text
baseline_reproduced = true
broad_bin_projection_artifact = partial
smooth_fit_artifact = medium
sampling_support_limited = false
need_new_exact_calculation = false
recommended_next_action = curve_extraction_fix_only
old_curve_segments_removed = 144 / 1080
unsupported_arc_fraction = 0.133333
optional_traceability_candidate_count = 24
```

The optional candidate list remains traceability-only.  It is not a
recommended exact-task list under the current decision.

## Interpretation

The revised report treats wide-mu panels as projection diagnostics and uses
local mutual-kNN brackets to restrict normal/SC curve display.  Unsupported
smooth-curve artifacts, including the high-mu vertical spike, are filtered
from the revised support-restricted display.

This fixes the presentation and curve-extraction issue but does not change
the underlying Stage IV-A scientific state.  Stage IV-A remains useful for
tFFLO discovery and the single-band diagnostic, but formal 3D topology
convergence is still not passed.

## Caveats

- Smooth curves and surfaces are diagnostic only.
- Wide-bin panels are not fixed-mu physical phase boundaries.
- No new exact calculation was run.
- No active-learning continuation was launched.
- No thermodynamic or topology label was changed.
- The next scientific decision still requires the missing Stage III reference
  before choosing between a same-window tail and a lower-mu extension.

## Validation

```text
python -m py_compile active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\scripts\build_stageiv_curve_extraction_fix_report.py
python active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\scripts\build_stageiv_curve_extraction_fix_report.py
```

The report was written in LaTeX and compiled successfully with `pdflatex`.
Selected PDF pages were rendered with PyMuPDF and visually checked.
