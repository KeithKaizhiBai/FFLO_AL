# Stage IV JA-kBT Comparison View Update

Date: 2026-06-30

## Context

The Stage IV 3D run is sampled in \((k_B T/t, J_A/t, \mu/t)\).  The original
return report already showed transparent 3D phase and topology point clouds, but
the first thermodynamic map was harder to compare directly with the earlier 2D
\(k_B T/t\)-\(J_A/t\) phase diagrams.

## Update

The local Stage IV return report now includes
`figures/phase_3d_jakt_view.png` and `figures/phase_3d_jakt_view.pdf`.

The figure has two panels:

- A \(k_B T/t\)-\(J_A/t\)-primary 3D view with \(\mu/t\) retained as depth.
- A collapsed \(k_B T/t\)-\(J_A/t\) projection where transparency increases with
  \(\mu/t\), so the reader can compare the 3D run with the older 2D phase-map
  intuition.

Both panels retain the report-only smooth diagnostic boundary surfaces or local
crossing markers for the normal/SC and uniform-SC/FFLO boundaries.

## Caveat

This view is a comparison projection of the 3D sampled cloud.  It is not a
single fixed-\(\mu\) phase diagram and must not be used as an independent
thermodynamic boundary definition.  The canonical data remain the Stage IV exact
points and the 3D report-only diagnostic boundary fits.

## Validation

The report was regenerated with:

```text
python -m py_compile scripts/build_stageiv_3d_return_report.py
python scripts/build_stageiv_3d_return_report.py --package-root active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624
```

The PDF compiled successfully with `pdflatex` and the new figure appears on page
2 of `stageiv_3d_return_report.pdf`.
