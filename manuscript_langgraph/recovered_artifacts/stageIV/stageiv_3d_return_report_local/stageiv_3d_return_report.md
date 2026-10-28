# Stage IV 3D Topology-Aware Return Report

## Executive Summary

- Final dataset: `D:\PhD Study\4th\SC_GBU\Fu_FFLO\active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\ML_Phase_StageIV_Topology3D\active_runs\active_phase_topology_3d_t_ja_mu_from_scratch_v1\dataset_iter025.npz`.

- Final unique samples: **7081**.

- Thermodynamic counts: normal=2350, uniform_SC=141, FFLO=4590.

- Topology counts: trivial=3265, topological=1466, gapless_SC=0, unresolved=0, not_applicable=2350.

- Official file-set status: complete; `dataset_iter025` exists and no dataset/merge/trusted files are missing.

- Official convergence audit status: **not_converged**, decision class **Decision C**.

- Hidden fixed-mu validation status: **inconclusive** because no Stage III reference dataset was supplied.

- Interpretation: the returned run is data-complete and scientifically useful, but should not be claimed as a formally converged Stage IV 3D phase/topology map yet.


## Key Convergence Diagnostics

- Topology volume-map change last three transitions: `[0.001412248042110669, 0.001922583952832607, 0.0025614754098360654]`.

- Topology surface p95 shift last three transitions: `[0.00558353725352583, 0.005906681186021189, 0.005996195716502644]`.

- Final topology surface coverage p95: `0.015317527016765012`.

- Trusted topology surprise last three transitions: `[0.065, 0.11229946524064172, 0.140625]`.

- Topology component count last three iterations: `[3, 4, 3]`.

- Mu-domain complete: `False`; mu-range limited: `True`.


## Visualizations

### Transparent 3D thermodynamic point cloud with diagnostic normal/SC and uniform_SC/FFLO boundary surfaces.

![Transparent 3D thermodynamic point cloud with diagnostic normal/SC and uniform_SC/FFLO boundary surfaces.](figures/phase_3d_transparent.png)


### Thermodynamic phase map viewed primarily in the kBT/t-J_A/t plane, with mu/t retained as depth/opacity for comparison to earlier 2D phase maps.

![Thermodynamic phase map viewed primarily in the kBT/t-J_A/t plane, with mu/t retained as depth/opacity for comparison to earlier 2D phase maps.](figures/phase_3d_jakt_view.png)


### Transparent 3D topology-aware point cloud with diagnostic phase and cFFLO/tFFLO boundary surfaces.

![Transparent 3D topology-aware point cloud with diagnostic phase and cFFLO/tFFLO boundary surfaces.](figures/topology_3d_transparent.png)


### Thermodynamic phase atlas by mu/t slice with diagnostic boundary curves.

![Thermodynamic phase atlas by mu/t slice with diagnostic boundary curves.](figures/phase_mu_slice_atlas.png)


### Topology atlas by mu/t slice with diagnostic phase and cFFLO/tFFLO boundary curves.

![Topology atlas by mu/t slice with diagnostic phase and cFFLO/tFFLO boundary curves.](figures/topology_mu_slice_atlas.png)


### Convergence metrics from the report-only audit.

![Convergence metrics from the report-only audit.](figures/convergence_metrics_summary.png)


### Pfaffian margin and bulk-gap distributions.

![Pfaffian margin and bulk-gap distributions.](figures/pfaffian_margin_bulk_gap_histograms.png)


### Dataset and topology-count growth.

![Dataset and topology-count growth.](figures/dataset_growth_phase_topology.png)


## Result Tables

- `tables/final_phase_counts.csv`

- `tables/final_topology_counts.csv`

- `tables/final_reliability_counts.csv`

- `tables/dataset_iteration_summary.csv`

- `tables/slurm_rank_runtime_summary.csv`

- `tables/slurm_iteration_runtime_summary.csv`

- `tables/boundary_surface_diagnostics.csv`

- `tables/slice_boundary_curve_diagnostics.csv`

- `tables/return_report_key_metrics.csv`


## Boundary Visualization Notes

- The added 3D surfaces are diagnostic, report-only smooth RBF fits of locally supported boundary crossing points extracted from the final exact labels after filtering long local edges.

- The added 2D curves are binned-median smoothing-spline fits of locally supported boundary crossing points within each fixed-mu slice.

- The JA-kBT view collapses the sampled 3D cloud toward the historical 2D plotting plane while retaining mu/t as depth and opacity; it should be read as a comparison view, not as a single fixed-mu phase diagram.

- These overlays improve readability of the sampled 3D phase structure, but they do not replace the formal Stage IV convergence metrics.

- In particular, topology-surface convergence remains not_converged because map change, p95 shift, coverage, trusted topology surprise, and component stability still fail the current audit.


## Caveats

- The 3D scatter and boundary overlays show sampled exact points plus diagnostic interpolants, not a publication-grade final phase surface.

- The official convergence audit uses report-only KNN/nearest-neighbor proxy surfaces; visual review and targeted follow-up are still needed.

- Hidden fixed-mu validation is not complete without the frozen Stage III reference dataset.

- The current audit recommends inspecting failed surfaces before more exact work; it does not justify a from-scratch restart.
