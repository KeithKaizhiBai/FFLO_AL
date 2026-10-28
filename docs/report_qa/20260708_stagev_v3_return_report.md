# Stage V-v3 Return Report Notes

## Question

Did the sparse-arc-aware Stage V-v3 acquisition solve the topology-surface
coverage problem observed in the Stage V-v2 coverage tail?

## Short Answer

No.  Stage V-v3 substantially improved topology-surface coverage relative to
the Stage V-v2 coverage tail, but it did not pass the formal convergence gates.
It should be treated as a successful acquisition-direction test, not a
converged production result.

## Key Numbers

```text
final_dataset = dataset_iter097
final_exact_iteration = 96
total_samples = 7069
trusted_exact = 6715
normal = 3252
uniform_SC = 58
FFLO = 3759
cFFLO/trivial = 2720
tFFLO/topological = 1038

Stage V-v2 tail topology_surface_coverage_p95 = 0.03591463649972722
Stage V-v3 topology_surface_coverage_p95 = 0.01877224394307404
relative improvement = approximately 47.7 percent
coverage_gate = 0.00625
coverage_gate_ratio = approximately 3.0

topology_map_change_last3 = [0.0010978, 0.0004287, 0.0008745]
trusted_topology_surprise_last3 = [0.1212121, 0.0666667, 0.125]
topology_component_count_last3 = [4, 4, 4]
decision_class = C
```

## Interpretation

The sparse-arc acquisition mechanism appears to move the run in the correct
direction because the coverage p95 is much lower than the Stage V-v2 tail
baseline.  The result is still not publication-ready: coverage remains above
the threshold and trusted topology surprise is too high.

The increase in tFFLO/topological sample count should not be used alone as
evidence of convergence.  The meaningful evidence is the coverage p95
reduction, stable component count, and map-change behavior, balanced against
the failed coverage and surprise gates.

## Caveats

The source-attribution audit found 1019 `unknown_explicit` selected sources
out of 7168 selected metadata records.  Blank/NaN source failures were not
present, but the high unknown-explicit fraction is a logging/attribution issue
that should be inspected before another full-budget run.

The visual 3D surfaces and point-cloud maps are diagnostic figures only.  They
are not final topology boundaries.

## Report Use

Use this note to explain that Stage V-v3 validated the sparse-support idea but
did not close the convergence problem.  The next physics/computation step is a
targeted diagnostic of sparse-patch source efficiency and high-surprise points,
not a blind continuation of the same full-loop configuration.
