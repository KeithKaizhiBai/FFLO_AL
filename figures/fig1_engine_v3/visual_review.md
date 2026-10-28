# Figure 1 engine-v3 visual review

Date: 2026-08-26

The SVG preview and the final 300-dpi PDF-derived color and grayscale previews
were inspected at full page and central-panel scale.

Observed:

- the tightly cropped thermodynamic surface is substantially larger than the
  frozen engine-v1 surface and remains legible in grayscale;
- Normal, Uniform SC, cFFLO, and tFFLO markers are locally attached to the
  landscape and are described as representative basins across parameter space;
- topology remains a separate second layer: gapped cFFLO, isolated gap closing,
  and reopened gapped tFFLO plus a separately labelled in-gap boundary state;
- the BdG oracle retains the required Hamiltonian-to-spectrum-to-free-energy
  chain, while the surrogate remains a scheduler;
- no clipping, missing glyphs, black raster boxes, card collision, or obvious
  text overlap was observed in color or grayscale.

The warning-oriented saliency proxy does not reproduce the declared full
reading order. This is retained as a warning rather than converted into an
automatic aesthetic failure or pass. Human review must decide whether the
thermodynamic landscape is the actual first fixation and whether the outer
workflow is sufficiently subordinate.

```text
GATE_FIG1_ENGINE_V3_FINAL_APPROVAL = HUMAN_WAIT
```
