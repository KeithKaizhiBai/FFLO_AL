# Figure 1 final layout and typography pass

## Question

Can the approved physics-first Figure 1 be made publication-ready at 178-mm
width without changing its scientific content?

## Answer

Yes. The final pass freezes the exact-BdG-generated physical content and
changes only card geometry, text placement, internal spacing, arrow routing,
and page normalization. The central state manifold remains the dominant
object, while exact oracle/observation are secondary and surrogate/acquisition
are tertiary.

The hard-fail geometry audit reports no card, text, plot, clipping, glyph, or
route-clearance failure. Declared neighboring cards have a minimum 20 px
gutter, non-exempt registered text and plots have at least 14 px padding, and
cycle arrows stop 8 px from card boundaries. At native print size the formulas
and one-line reliability/observation labels remain legible, including

\[
(-1)^\nu=\operatorname{sgn}[P(0)P(\pi)]
\]

and the acquisition score containing the boundary-distance term
$d_{\mathrm{boundary}}$.

## Caveat and gate

Automated geometry correctness is not human figure approval. The figure is
ready to freeze only after explicit `GATE_FIG1_LAYOUT_APPROVAL`. This pass does
not authorize manuscript-LaTeX integration or work on Figures 2--4.
