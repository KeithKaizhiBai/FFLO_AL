# PRL Figure Redesign: Visual-Concept Freeze

Date: 2026-08-14

## Question

How should the current evidence-traced Figures 1--4 be redesigned into a
coherent PRL visual narrative without changing scientific claims or prematurely
rebuilding all quantitative panels?

## Answer

The redesigned hierarchy is physics-first. Figure 1 is a single elliptical
closed learning cycle around the exact BdG oracle, with a central bridge from
thermodynamic minimization to trusted topology. Figure 2 gives most of its area
to the final thermodynamic map and relegates initialization/acquisition history
to compact insets. Figure 3 gives most of its area to the trusted cFFLO/tFFLO
map, shows only support-localized points in the boundary diagnostic, and keeps
raw convergence and $k$ sensitivity distinct. Figure 4 uses one 3D overview,
then emphasizes matched fixed-$\mu$ slices, physical evolution with $\mu$, and
active-learning scaling evidence.

The common phase palette is soft gray (normal), deep blue (uniform SC), teal
(FFLO/cFFLO), magenta-purple (tFFLO), and translucent charcoal (uncertainty).
Red warning boxes, dashboard cards, provenance hashes, archive paths, and audit
metadata are excluded from the artwork. Stage IV non-closure remains a caption
and Supplement limitation, not a red main-panel banner.

For Figure 1, the binary topology label is displayed consistently as

$$
\chi=\operatorname{sgn}[P(0)P(\pi)]=(-1)^\nu,
$$

so the Pfaffian sign and $\nu\in\{0,1\}$ convention are both explicit.

Four scripted SVG mockups and one high-fidelity Figure 1 SVG/PDF/PNG prototype
were generated under `figures/`. Figure 2--4 mockups use synthetic geometry for
layout only; no frozen physics data were read and no exact calculation was
performed. Automated package/style checks pass 72/72; the pre-existing
scientific candidate validation remains 29/29 and manuscript LangGraph tests
remain 16/16.

## Decision status

`GATE_FIGURE_STYLE_APPROVAL` is awaiting human review. It was not passed
automatically. `GATE_FIGURE_APPROVAL`, the production manuscript LaTeX, the
claim registry, all frozen data, and all scientific caveats remain unchanged.

After style approval, Figures 2--4 may be reconstructed from recovered frozen
artifacts using raw, non-normalized audit metrics. External Introduction
retrieval remains unauthorized.
