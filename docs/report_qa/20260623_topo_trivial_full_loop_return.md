# 2026-06-23 Topology-Aware Full-Loop Return

## Question

The topology-aware active-learning run returned under:

```text
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop
```

What did it complete, what do the online topology diagnostics show, and what
should be claimed in the report?

## Answer

The completed run is:

```text
run_id = active_phase_topology_from_scratch_full_loop_v2
final dataset = dataset_iter018.npz
completed iterations = 18
final samples = 4345
```

The older `active_phase_topology_from_scratch_full_loop_v1` attempt failed at
iteration 0 because one shard on `gpuh14` reported CUDA devices unavailable.
The returned v2 run is therefore the valid run for this report.

Final thermodynamic phase counts:

```text
normal = 746
uniform_SC = 386
FFLO = 3213
```

Final online topology diagnostic counts:

```text
not_applicable = 746
trivial = 2676
topological = 923
gapless_SC = 0
unresolved = 0
topology_trusted = 3599
```

Final StopController status:

```text
stop_reason = converged_main_phase_boundaries
phase_map_change = 0.001745 < 0.002
normal/SC boundary shift = 0.004167 <= 0.004167
uniform/FFLO boundary shift = 0
trusted surprise = 0/200
all-selected surprise = 0.066406
hard-risk surprise = 0.303571
boundary_coverage_p95 = 0.006988 > 0.00625
```

Runtime summary:

```text
estimated exact-array walltime = 20.10 h
mean exact-array walltime = 67.0 min/iteration
median exact-array walltime = 64.6 min/iteration
max exact-array walltime = 115.9 min
rank-summed point runtime = 153.96 h
rank-summed local-refinement runtime = 121.51 h
online topology diagnostic runtime = 16.32 s
```

The walltime estimate uses the maximum shard `elapsed_sec` in each iteration
and sums those maxima over the 18 exact arrays.  It excludes login-node
training, merge, append, and report-generation overhead.  The rank-summed
runtime is a compute-cost diagnostic, not elapsed wall clock.

The correct interpretation is:

```text
This is a successful main-boundary convergence run with online topology
diagnostics.  It is not a publication-grade topology-boundary result.
```

The report should explicitly say:

1. The run is a new cold-start Stage III loop, not a continuation of
   `dataset_iter035`.
2. Online topology labels are acquisition/runtime diagnostics.
3. The absence of online gapless/unresolved labels in the appended final
   dataset does not prove that the full topology frontier is resolved.
4. The next required step is to freeze `dataset_iter018.npz` and run the
   publication-grade offline topology pass/audit on that dataset.

Generated report artifacts:

```text
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/active_learning_phase_boundary_report.tex
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/active_learning_phase_boundary_report.pdf
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/topo_trivial_full_loop_summary.md
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/tables/
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/figures/
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/figures/eta_topology_phase_map.png
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/figures/selected_boundary_concentration.png
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/tables/runtime_summary.csv
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/tables/response_region_diagnostics.csv
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/tables/selected_boundary_concentration_summary.csv
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/tables/selected_boundary_concentration_by_iteration.csv
```

The added red-blue `eta_topology_phase_map.png` follows the original eta phase
diagram style.  Color encodes eta sign and magnitude.  The solid black line is
the normal/SC visual contour, dashed gray is the uniform-SC/FFLO visual
contour, and purple short local edges mark online Z2 changes between cFFLO and
tFFLO samples.  These purple edges are diagnostic topology-boundary seeds, not
a final publication-grade topology boundary.

## Follow-up Question

The red-blue eta figure still looks rough in some regions.  Are those phase or
topology problems, and is there evidence that the new acquisition concentrated
near the intended boundaries?

## Follow-up Answer

Two rough-looking regions were added to the report as response-side diagnostics:

```text
low-T edge:
    region = kT <= 0.03 and 0.75 <= JA <= 1.25
    points = 42
    phase/topology = FFLO/topological for all points
    trusted_exact = 42/42
    q_unresolved = 0
    delta_unresolved = 0
    eta signs = 33 negative, 9 positive
    p95(|eta|) = 0.7377

diagonal high-J band:
    segment = (kT,JA) = (0.05,1.3) to (0,2)
    distance band = 0.03 in raw parameter units
    points = 74
    phase split = FFLO 39, normal 35
    q_expanded = 74/74
    q_unresolved = 0
    delta_unresolved = 0
    median eta = -0.004294
    p95(|eta|) = 0.01714
```

Interpretation:

```text
The low-T edge is not a phase/topology failure; it is a response-side eta
stability/sign issue on otherwise trusted FFLO/topological points.

The diagonal high-J band is a normal/SC boundary and q-window-sensitive
response band.  Eta is close to zero there, so it should not be presented as a
settled eta-sign boundary.
```

Selected-point concentration was quantified using normalized Euclidean
distance to final diagnostic Delaunay boundary segments.  The normal/SC
segments and local cFFLO/tFFLO Z2-change segments both used a maximum
normalized edge length of 0.03.

```text
initial seed iter000:
    selected points = 512
    within 0.03 of normal/SC = 0.1016
    within 0.03 of cFFLO/tFFLO = 0.04883
    within 0.03 of either = 0.1289
    median distance to either = 0.1733

all acquisition iterations iter001-017:
    selected points = 4352
    within 0.03 of normal/SC = 0.4065
    within 0.03 of cFFLO/tFFLO = 0.3408
    within 0.03 of either = 0.5949
    median distance to either = 0.01921

last five acquisition iterations iter013-017:
    selected points = 1280
    within 0.03 of normal/SC = 0.4031
    within 0.03 of cFFLO/tFFLO = 0.3578
    within 0.03 of either = 0.6016
    median distance to either = 0.01876
```

These metrics support the visual impression that the topology-aware
acquisition concentrated strongly near the thermodynamic normal/SC boundary
and the diagnostic cFFLO/tFFLO frontier.  They do not turn the diagnostic
cFFLO/tFFLO edges into final publication topology boundaries; that still
requires the offline topology pass/audit.
