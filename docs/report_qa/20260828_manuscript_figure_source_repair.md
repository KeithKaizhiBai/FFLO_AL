# Manuscript figure-source repair

## Question

Why did the manuscript not display the previously approved Figure Engine
Figures 1--4?

## Answer

The figure-generation framework was working, but the manuscript compiler did
not load its authoritative registry. It copied a separate legacy candidate
bundle from `manuscript_figures/final/`, so validation of the source figures
and compilation of the manuscript were disconnected.

The repair makes the scientific freezes the sole manuscript authority:

- Fig. 1: `figures/fig1_engine_v4/fig1_engine_v4.pdf`
- Fig. 2: `figures/fig2_strict_data/fig2_strict_data.pdf`
- Fig. 3: `figures/fig3_strict_data/fig3_strict_data.pdf`
- Fig. 4: `figures/fig4_strict_data/fig4_strict_data.pdf`

Each source hash is checked before copying, each staged hash must be identical,
and the LaTeX recorder must show that all four staged PDFs were read during
compilation. This closes the gap between Figure Engine approval and manuscript
assembly.

The restored Fig. 4 has four explicit fixed-$\mu$ bins centered at
$\mu/t=-0.45,0.15,0.85,1.45$, each with half-width 0.025. These are exact
slice samples, not a continuous surface. The strict three-dimensional topology
convergence gates remain unclosed, so the figure supports a caveated
higher-dimensional exploration, not a complete 3D phase diagram.
