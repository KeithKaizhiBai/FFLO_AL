# Phase-Boundary-Focused Discovery Acquisition

## Question

Why did the previous discovery-mode active-learning run keep selecting many
predicted normal-interior points, and what changed in the sharpened acquisition
logic?

## Short Answer

The previous score treated

\[
B_{\Delta,\mathrm{raw}}(x)
=
\exp\!\left[
-\frac{|\Delta_{\mathrm{pred}}(x)-\Delta_\epsilon|}
{\Delta_{\mathrm{scale}}}
\right]
\]

as direct evidence of normal/SC boundary relevance.  In a high-confidence
normal region, however, \(\Delta_{\mathrm{pred}}\approx 0\), and
\(\Delta_\epsilon\) is small, so this raw score can be large even away from
the actual thermodynamic boundary.

The sharpened logic multiplies this raw score by the classifier competition
between normal and superconducting phases:

\[
U_{\mathrm{NS}}(x)=4P_{\mathrm{normal}}(x)P_{\mathrm{SC}}(x),
\qquad
B_{\Delta,\mathrm{gated}}(x)
=B_{\Delta,\mathrm{raw}}(x)U_{\mathrm{NS}}(x).
\]

Deep normal and deep SC regions have \(U_{\mathrm{NS}}\approx 0\), while the
normal/SC boundary has \(U_{\mathrm{NS}}\approx 1\).  The acquisition phase
score now uses \(B_{\Delta,\mathrm{gated}}\), while both raw and gated values
are saved for diagnostics.

## Technical Notes

The active pool is no longer built from a broad OR rule.  New discovery runs
use

\[
A_{0,\mathrm{pool}}(x)
=
A_{0,\mathrm{main}}(x)P_{\mathrm{interior}}(x)
\]

and admit candidates only when

\[
A_{0,\mathrm{pool}}(x)
\ge
\max\!\left[
Q_{q_{\mathrm{pool}}}(A_{0,\mathrm{pool}}),
\alpha_{95}Q_{0.95}(A_{0,\mathrm{pool}})
\right].
\]

The active-pool quantile, sampling power, and exploration weight are scheduled
piecewise so late iterations become more focused than early exploration.  The
stochastic sampler remains stochastic, but randomness now acts inside the
score-defined high-information pool:

\[
p_i\propto
\left[
A_{0,\mathrm{pool}}(x_i)
R_{\mathrm{obs}}(x_i)
R_{\mathrm{batch}}(x_i)
\right]^\gamma .
\]

The exact BdG oracle, phase-label definitions, neural-network architecture,
and StopController main convergence logic were not changed.

## Report Use

Use this note to explain why the active-learning selector needs phase-probability
competition gates in addition to regression proximity scores.  The key physical
point is that \(\Delta_{\mathrm{pred}}\approx 0\) can mean either "normal/SC
boundary vicinity" or "deep normal"; the classifier probabilities distinguish
these cases.
