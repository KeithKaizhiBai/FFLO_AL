# Figure 1 v3 full asset package

## Question

Are all independent visual components ready for Figure 1 master assembly?

## Answer

Yes, subject to the human asset gate. The package contains nine independent
assets: parameter space, exact BdG oracle, thermodynamic state manifold,
topology triptych, physical observation, surrogate, acquisition, next exact
calls, and reliability gates. All pass structural validation and were checked
in color, grayscale, and approximate 178-mm print context.

The visual hierarchy answers the physical question first: the thermodynamic
$F(\Delta,q)$ manifold and the cFFLO--gap-closing--tFFLO sequence dominate.
The exact BdG oracle supplies all labels. The surrogate and acquisition remain
subordinate scheduling components and do not replace the physics solver.

Asset C uses a 600-dpi raster only for the smooth 3D surface; its axes, labels,
contours, minima markers, and annotations remain vector. The other eight assets
are vector. This is a schematic framework figure and does not recompute or
claim new scientific results.

## Decision state

```text
GATE_FIG1_ASSET_APPROVAL = HUMAN_WAIT
GATE_FIG1_FINAL_APPROVAL = not reached
```

Master assembly must wait for explicit user approval.
