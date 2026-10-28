# PRL Figure Style-Gate Package

This directory contains the visual-concept freeze requested before rebuilding
the quantitative manuscript figures.

```text
GATE_FIGURE_STYLE_APPROVAL = HUMAN_WAIT
GATE_FIGURE_APPROVAL       = unchanged
```

Review objects:

- `fig1_framework/fig1_master.svg`
- `fig1_framework/fig1_master.pdf`
- `fig1_framework/validation_report.md`
- `fig2_thermodynamic/fig2_visual_mockup.svg`
- `fig3_topology/fig3_visual_mockup.svg`
- `fig4_3d_extension/fig4_visual_mockup.svg`
- `style_gate_validation.md`

The Figure 1 master is a direct editable SVG with no raster image element and
does not use Matplotlib for the whole composition. The Figure 2--4 mockups
contain deterministic synthetic geometry solely to
test hierarchy and spacing. They are visibly marked as layout mockups and must
not be interpreted as numerical results. Their manifests record that no
quantitative dataset was read and no physics was recomputed.

Human approval should address the layout, visual hierarchy, palette, density,
and Figure 1 physical narrative. Approval will authorize scripted
reconstruction of the quantitative Figure 2--4 panels from registered frozen
artifacts; it will not itself pass the later scientific figure gate.
