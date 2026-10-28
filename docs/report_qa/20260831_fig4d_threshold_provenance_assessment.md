# Figure 4(d) threshold-provenance assessment

## Question

Do the Stage IV thresholds shown in Fig. 4(d) have physical or computational
meaning strong enough for a main-text panel?

## Finding

The thresholds are operational convergence tolerances for a normalized,
surrogate-based audit. They are not fundamental physical scales of the BdG
model. The exact bulk-gap and Pfaffian tests remain the physical topology
authority; Fig. 4(d) instead measures whether a derived topology
reconstruction is stable and sufficiently covered.

- `volume_map_change < 0.002` limits the fraction of a fixed 20,000-point
  audit cloud whose predicted topology changes between iterations. It has a
  clear computational interpretation (about 40 changed audit points), but no
  direct thermodynamic meaning.
- `surface_shift_p95 < 0.0041667` limits the normalized bidirectional p95
  displacement of derived proxy surfaces. The value equals `1/240`, consistent
  with one cell of the earlier 241-point monitor-grid convention. No archived
  decision text was found that explicitly derives the threshold from a
  physical resolution.
- `coverage_p95 < 0.00625` limits normalized proxy-surface-to-exact-sample
  distance. The value equals `1.5/240`, consistent with a sampling-resolution
  tolerance rather than a physical length.
- `trusted_surprise < 0.02` limits disagreement between previous surrogate
  predictions and newly returned trusted exact topology labels to two percent.
  This is a statistical/generalization tolerance inherited from the Stage III
  topology audit, not a physical phase criterion.
- component stability requires a constant significant-component count over
  the final iterations, but the count depends on `component_radius=0.035` and
  `component_min_size=12`. It is a useful geometric warning, not a physical
  topological invariant.

The final Stage IV coverage and trusted-surprise failures are large relative
to their thresholds (approximately 2.45 and 7.03 times the gates), whereas map
change and surface shift are closer (approximately 1.28 and 1.44 times). The
nonclosure warning is therefore not supported equally by every metric.

## Recommendation

Do not present the present numeric threshold scoreboard as a main physical
result without a threshold-calibration and sensitivity subsection. Prefer to
replace Fig. 4(d) in the main text with a scope/evidence-limit panel stating:

- exact samples only; no fitted continuous topology surface;
- finite domain `mu/t in [-0.5, 1.5]` with lower-edge topology contact;
- fixed-mu slices are explicit bins, not inter-slice interpolation;
- strict 3D topology closure is not established.

Move the metric definitions, normalization, thresholds, proxy-surface method,
and sensitivity to grid size, neighbor count, support radius, component
radius, and tolerance choices to the Supplemental Material. No figure or
manuscript change is authorized by this assessment alone.
