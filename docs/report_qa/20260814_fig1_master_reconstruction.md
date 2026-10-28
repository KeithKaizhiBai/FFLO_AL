# Figure 1 Master Reconstruction Review

Date: 2026-08-14

## Purpose

Replace the Matplotlib whole-figure prototype with a publication-quality,
directly editable vector schematic that makes the audited BdG physical state
manifold visually dominant over the active-learning scheduler.

## Implemented scientific hierarchy

The outer loop is a clockwise six-stage cycle:

1. parameter space;
2. exact BdG oracle;
3. trusted physical observation;
4. probabilistic surrogate model;
5. physics-aware acquisition;
6. next exact calls.

The center is a single audited physical-state object. Its upper half contains
one unified schematic three-dimensional vector landscape $F(\Delta,q)$ with
representative normal, uniform-SC, and symmetry-related finite-$q$ FFLO minima.
Its lower half contains three visually separate zones: a gapped cFFLO spectrum
with $\nu=0$, an isolated critical gap closing, and a final reopened, gapped
tFFLO spectrum with $\nu=1$.

The invariant is shown as

$$
(-1)^\nu=\operatorname{sgn}[P(0)P(\pi)],\qquad \nu\in\{0,1\},
$$

which preserves the project's binary-label convention. The surrogate is
explicitly described only as a guide for where to run the next exact BdG calls.

## Technical construction

The master is authored directly as SVG using vector paths and editable text.
Matplotlib is not used for the whole figure, and the SVG contains no raster
`image` element. The PDF is a one-page 7.2-inch by 4.44-inch vector export with
15 embedded font resources and no raster image XObjects. The PNG is rendered
from the PDF as a preview.

The PDF was visually inspected at 96, 180, and 300 dpi. Font sizes were raised
after the first print-scale review; the decision arrow was rerouted away from
the acquisition-factor labels; and the small oracle spectrum was made visibly
gapped to avoid an unintended zero-energy implication.

The subsequent v2 refinement retained the same vector architecture but changed
the reading order. The central shell was enlarged, the free-energy object was
given explicit $F$, $\Delta$, and $q$ axes and vector surface cells, and the
oracle was enlarged to show $H_{\mathrm{BdG}}(k)$, spectrum/eigensolve, and
free-energy generation. The six stage labels and loop arrows were deliberately
downweighted, with surrogate/acquisition carrying the lowest visual contrast.

## Validation and gate status

- Figure 1 master checks: 29/29 passed.
- Full style-package checks: 80/80 passed.
- No scientific data read or physics recomputed.
- `GATE_FIGURE_STYLE_APPROVAL = HUMAN_WAIT`.
- Manuscript LaTeX, Figures 2--4, and Introduction retrieval remain unchanged.
