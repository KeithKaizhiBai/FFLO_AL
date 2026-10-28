# Strict-data workflow notes for manuscript Figures 2-4

The publication figure system now distinguishes exact samples, derived
numerical tables, derived boundary geometry, interpolated fields, schematic
annotations, and hybrid data/schematic panels. Every data panel carries source
paths, hashes, versions, rows, fields, units, filters, missing-data policy,
transform authorization, trust masks, permitted claims, caveats, and forbidden
operations before rendering.

For Figure 2, frozen Stage II evidence independently reproduces 7434 exact
samples with phase counts 1867 Normal, 715 uniform SC, and 4852 FFLO. The true
cold start is the archived 512-point random-grid selection; 506 coordinates
remain in the final trusted endpoint and six are left explicitly unmatched.
The full selection history covers iterations 0-34 and 9216 calls including the
cold start. Raw phase-map change, Normal-SC shift, uniform-SC-FFLO shift, and
trusted surprise are shown without smoothing or normalized gate-ratio collapse.

The hard-risk audit is not a fourth phase. Its 129 archived exact calls appear
as open rings in a separate legend layer; 19 coordinate matches remain in the
final trusted endpoint. The evidence archive does not contain final dense-map
values or final boundary coordinates, so the hero panel is exact-sample-only.
No continuous field or new boundary curve is inferred.

Figure 3 evidence supports 3213 trusted gapped FFLO samples: 2290 cFFLO and
923 tFFLO. The configured k=8 audit has coverage p95 0.006185, whereas the k=6
sensitivity value 0.006269 slightly exceeds 0.00625. Gap closing remains a
transition diagnostic and local brackets remain support diagnostics.

Figure 4 evidence contains 7081 exact 3D observations. It supports the wording
"higher-dimensional exploration" and "3D extension," but strict topology
closure is not achieved. Exact point clouds and explicit fixed-$\mu$ slices are
eligible for a later figure; continuous RBF/KNN diagnostic surfaces are not
authorized as formal phase surfaces.

Current state: `GATE_FIG2_HUMAN_SCIENCE_REVIEW = HUMAN_WAIT`.
