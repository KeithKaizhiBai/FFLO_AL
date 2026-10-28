# Figure 1 v3 asset-first reconstruction - Phase 1 Q&A

## Why was the existing vector figure not simply edited again?

Its scientific ordering was already mostly correct, but its mini-panels were
not mature assets. The central $F(\Delta,q)$ object read as a pale wireframe,
the semantic colors were flat, and the tFFLO panel did not yet separate a red
in-gap boundary/surface branch from the gapped purple bulk bands. Further
whole-figure editing would preserve those local weaknesses.

## What is frozen before formal drawing begins?

- exact/oracle: orange dark/base/highlight/tint;
- trusted physics: green dark/base/highlight/tint;
- learning: blue dark/base/highlight/tint;
- decision: purple dark/base/highlight/tint;
- selected exact calls: red dark/base/highlight/tint;
- cFFLO: green gapped bulk, $\nu=0$;
- transition: gray bulk gap closes at the critical point;
- tFFLO: reopened purple bulk gap, $\nu=1$, with a visually separate red
  boundary/surface-state crossing;
- topology rule:
  $$
  (-1)^\nu=\operatorname{sgn}[P(0)P(\pi)].
  $$

The exact BdG oracle remains the source of every physical label. The surrogate
only schedules the next exact calculation locations.

## What was generated?

`figures/fig1_framework_v3/style/color_style_board.{svg,pdf,png}` plus frozen
YAML configuration, a grayscale review, an output/hash manifest, and a 16/16
validation report. All text, cards, arrows, markers, and topology curves are
vector. The smooth thermodynamic color/material swatch is the only raster
layer and is embedded at 600 dpi.

## What remains blocked?

`GATE_FIG1_COLOR_STYLE_APPROVAL` is still `HUMAN_WAIT`. Assets B/C/D, the full
asset contact sheet, and master SVG assembly must not start until explicit
approval. This automated validation is not a scientific or visual approval.

