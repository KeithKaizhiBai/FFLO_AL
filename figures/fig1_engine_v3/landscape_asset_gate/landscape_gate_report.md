# Compact thermodynamic landscape asset gate

Date: 2026-08-26

No full Figure 1 composition was modified. No exact BdG calculation was run.

## Candidate status

| Asset | Hard status | Aspect | Fill | Whitespace | Triangle | Relief | Warnings |
|---|---:|---:|---:|---:|---:|---:|---|
| current_v3 | fail | 2.077 | 0.200 | 0.800 | 0.027 | 0.146 | aspect_in_preferred_range, fill_preferred, whitespace_preferred, vertical_relief_preferred |
| candidate_A | pass | 1.776 | 0.703 | 0.297 | 0.095 | 0.319 | whitespace_preferred |
| candidate_B | pass | 1.795 | 0.705 | 0.295 | 0.095 | 0.314 | whitespace_preferred |

Candidate A is the compact 3D hero without a contour floor. Candidate B uses the same analytic three-basin surface and minima with a restrained five-level contour floor. Both use orthographic projection and one parent FFLO-sector label. Neither candidate assigns topology from terrain position or depth.

Validator self-test: **PASS**. The elongated negative fixture is rejected and the compact control passes.

## Metric definitions

- Aspect is the width/height of the tight alpha-derived colored-surface bbox.
- Fill is the count of actual colored-surface pixels with alpha > 8 divided by that tight bbox area; labels and axes are excluded.
- Internal whitespace is the complementary blank fraction inside the same declared surface box.
- Triangle is the projected Normal/Uniform-SC/FFLO marker-triangle area divided by the colored-surface bbox area.
- Relief is the projected peak-to-floor vertical displacement divided by the colored-surface bbox height.

The rounded parameter-domain edge is a compact illustrative terrain patch, not a computed physical phase boundary. No outer Figure-1 card bbox is used in candidate mask metrics.

## Camera sweep

| Camera | elev | azim | box aspect | hard status | aspect | fill | triangle | relief |
|---|---:|---:|---|---|---:|---:|---:|---:|
| camera_1 | 27 | -58 | [1.45, 1.0, 0.82] | fail | 2.464 | 0.672 | 0.094 | 0.414 |
| camera_2 | 30 | -50 | [1.4, 1.0, 0.86] | fail | 2.053 | 0.684 | 0.095 | 0.377 |
| camera_3 | 33 | -44 | [1.35, 1.0, 0.82] | pass | 1.776 | 0.703 | 0.095 | 0.319 |
| camera_4 | 29 | -54 | [1.42, 1.0, 0.9] | fail | 2.191 | 0.677 | 0.095 | 0.413 |
| camera_5 | 32 | -42 | [1.34, 1.0, 0.78] | pass | 1.795 | 0.705 | 0.095 | 0.314 |
| camera_6 | 34 | -40 | [1.3, 1.0, 0.74] | pass | 1.688 | 0.718 | 0.095 | 0.281 |

## Human review questions

1. Which candidate reads most immediately as a physical terrain?
2. Can Normal, Uniform SC, and FFLO be identified without reading the caption?
3. Does the surface appear compact rather than ribbon-like?
4. Are both $\Delta$ and $q$ directions visually meaningful?
5. Is free-energy relief strong enough?
6. Does the FFLO sector remain thermodynamic rather than prematurely topological?
7. Does the candidate overpower the topology row appropriately but not excessively?
8. Does the asset remain readable at manuscript scale?
9. Is Candidate A or Candidate B preferred?
10. Is a third candidate genuinely necessary?

```text
GATE_COMPACT_THERMODYNAMIC_LANDSCAPE_ASSET = HUMAN_WAIT
selected_candidate = none
master_fig1_modified = false
```

After explicit candidate approval only, the selected asset may be integrated as a new Figure 1 version. Automated validation does not select a candidate.
