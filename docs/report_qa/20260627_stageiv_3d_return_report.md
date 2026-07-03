# 2026-06-27 Stage IV 3D Return Report

## Returned Data Status

The returned Stage IV 3D cold-start topology-aware run is complete through
`dataset_iter025`.

```text
final samples = 7081
normal = 2350
uniform_SC = 141
FFLO = 4590
trivial = 3265
topological = 1466
gapless_SC = 0
topology_unresolved = 0
trusted_exact = 6767
topology_trusted = 4731
needs_rerun_exact = 0
q_unresolved = 0
delta_unresolved = 0
```

The returned result passed the file-set check: no dataset, merge, trusted, or
shard iterations were missing.

## Report Outputs

The local report is:

```text
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/
  ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_report_local/
```

It contains Markdown, PDF, CSV tables, transparent 3D PNG/PDF figures, mu-slice
atlas figures, and a reproduction manifest.

## Main Scientific Interpretation

The returned run is data-complete and scientifically useful, but it is not a
formal Stage IV convergence result.

The report-only convergence audit found:

```text
stageiv_convergence_status = not_converged
mu_domain_complete = false
mu_range_limited = true
hidden_slice_status = inconclusive
```

Main limiting topology diagnostics:

```text
topology_volume_map_change_last3 = [0.001412, 0.001923, 0.002561]
topology_surface_shift_p95_last3 = [0.005584, 0.005907, 0.005996]
topology_surface_coverage_p95_final = 0.015318
trusted_topology_surprise_last3 = [0.065, 0.1123, 0.140625]
topology_surface_component_count_last3 = [3, 4, 3]
```

## Do Not Claim

- Do not claim formal Stage IV 3D convergence.
- Do not treat the transparent 3D point cloud as a continuous interpolated
  surface.
- Do not interpret missing hidden-slice validation as a physics failure; the
  Stage III frozen reference dataset was not supplied.
- Do not restart from scratch without first inspecting failed surface regions.

## Recommended Next Step

Inspect the failed topology/thermodynamic surface regions and supply the frozen
Stage III fixed-mu reference dataset for hidden-slice validation before planning
additional targeted exact calculations.
