# Stage IV-A Mu-Slice Boundary Audit

Date: 2026-06-28

## Question

The Stage IV-A 3D return showed suspicious normal/SC boundary behavior in wide
mu panels, especially around `0.83 <= mu/t < 1.17` and possibly
`0.50 <= mu/t < 0.83`.  The goal was to determine whether this is a real
thermodynamic feature or a visualization/projection artifact, using only the
existing returned data.

## Inputs

- Frozen returned dataset: `dataset_iter025`
- Source prompt: `active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/docs/Slice_Audit_prompt.md`
- Existing Stage IV-A return artifacts and selected-point metadata
- Normal-state single-band preflight scan

No exact calculation, active-learning continuation, or label modification was
performed.

## Output Location

```text
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_mu_slice_boundary_audit_local/
```

Key files:

```text
stageiv_mu_slice_boundary_audit_report.md
stageiv_mu_slice_boundary_audit_report.tex
stageiv_mu_slice_boundary_audit_report.pdf
stageiv_mu_slice_boundary_audit_decision.json
stageiv_mu_slice_boundary_audit_summary.json
tables/*.csv
figures/*.png
figures/*.pdf
```

## Main Result

The final classification is:

```text
Class A - broad-bin projection artifact
```

The recommended next action is:

```text
curve_extraction_fix_only
```

The audit does not recommend new exact calculation for this specific issue:

```text
need_new_exact_calculation = false
recommended_tail_candidate_count = 0
```

The optional candidate list contains 24 traceability candidates only, in case
a later decision changes after the plotting/curve extraction is fixed.

## Interpretation

The suspicious high-mu wide panels are best explained by projecting a
three-dimensional boundary through a broad mu bin onto a two-dimensional
`kBT/t`-`JA/t` plot.  Narrow fixed-mu slices and local mutual-KNN normal/SC
brackets show that the wide-bin panels can visually exaggerate mixed-mu
geometry.  Smooth curve fitting contributes a medium artifact risk, so smooth
diagnostic boundaries should remain report-only and support-filtered.

The audit found partial correlations with hard-risk and thermodynamic-margin
diagnostics, so this should not be described as a purely cosmetic plotting
issue.  However, the evidence does not support classifying the suspicious
wide-bin shape as a new real thermodynamic feature.

## Caveats

- Smooth curves and surfaces are diagnostic only.
- The single-band corridor is not a topology label.
- Candidate points were not submitted and are not recommended under the
  current decision.
- This audit does not change the broader Stage IV-A conclusion: tFFLO is
  clearly discovered, but formal 3D topology convergence has not yet passed.

## Validation

```text
python active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\scripts\build_stageiv_mu_slice_boundary_audit.py
python -m py_compile active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\scripts\build_stageiv_mu_slice_boundary_audit.py
```

The report was written in LaTeX and compiled successfully with `pdflatex`.
Selected PDF pages were rendered with PyMuPDF and visually checked.
