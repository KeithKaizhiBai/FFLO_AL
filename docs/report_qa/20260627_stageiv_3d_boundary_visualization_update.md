# Stage IV 3D Boundary Visualization Update

Date: 2026-06-27

Context:

The returned Stage IV 3D topology-aware run is complete through
`dataset_iter025`, but the official report-only convergence audit remains
`not_converged`.  The first returned-result report contained transparent 3D
point clouds and fixed-\(\mu\) slice atlases, but the phase and topology
boundaries were still visually difficult to read from point clouds alone.

What was added:

- The 3D thermodynamic figure now overlays diagnostic semi-transparent smooth
  normal/SC and uniform-SC/FFLO boundary surfaces.
- The 3D topology figure overlays diagnostic smooth normal/SC,
  uniform-SC/FFLO, and cFFLO/tFFLO boundary surfaces.
- The fixed-\(\mu\) thermodynamic atlas overlays smoothed normal/SC and
  uniform-SC/FFLO boundary curves.
- The fixed-\(\mu\) topology atlas overlays smoothed normal/SC,
  uniform-SC/FFLO, and cFFLO/tFFLO boundary curves.
- New companion tables record the visualization support:
  `boundary_surface_diagnostics.csv` and
  `slice_boundary_curve_diagnostics.csv`.

Technical interpretation:

The added boundaries are diagnostic visual aids only.  The code first extracts
locally supported boundary-crossing points from final exact labels using
Delaunay neighborhoods with long-edge filtering.  The 3D surfaces are then
fit as smooth RBF thin-plate-spline surfaces
\(J_A/t=f(k_B T/t,\mu/t)\).  The 2D fixed-\(\mu\) curves use binned-median
smoothing splines \(J_A/t=f(k_B T/t)\).  These overlays improve readability of
the sampled 3D structure, but they are not the formal convergence criterion
and should not be cited as proof of Stage IV closure.

Current scientific status:

The added visualization does not change the Stage IV decision:

```text
file_set_status = complete
stageiv_convergence_status = not_converged
hidden_slice_status = inconclusive
mu_domain_complete = false
mu_range_limited = true
```

The next useful action is still to inspect failed surface regions and provide
the frozen Stage III fixed-\(\mu\) reference for hidden-slice validation before
deciding whether targeted exact/spectral tail batches are needed.
