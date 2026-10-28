# Figure 1 engine v4 - Candidate-A landscape integration

This independent review candidate is authorized by the approved
`GATE_COMPACT_THERMODYNAMIC_LANDSCAPE_ASSET`. It preserves the v3 canvas,
eight-card architecture, topology triptych, surrounding workflow, scientific
claims, and canonical Figure Engine rules.

Only the upper thermodynamic landscape and its immediate local markers and
parameter axes are rebuilt. The selected source is
`candidate_A_compact_hero`; its surface remains a deterministic schematic and
does not contain recomputed BdG physics. The landscape shows one parent FFLO
sector. cFFLO/tFFLO interpretation remains exclusively in the unchanged
topology row.

Automated validation cannot approve:

```text
GATE_FIG1_ENGINE_V4_APPROVAL = HUMAN_WAIT
```

Manuscript LaTeX and Figures 2-4 remain outside this gate.

## Rebuild order

```bash
MPLCONFIGDIR=/tmp/al_fflo_mpl python3 build_assets_v4.py
python3 build_fig1_engine_v4.py
node measure_rendered_geometry.mjs
node export_fig1_engine_v4.mjs
python3 normalize_pdf_page.py
python3 render_print_previews.py
python3 build_saliency_preview.py
python3 build_v3_vs_v4_comparison.py
python3 validate_fig1_engine_v4.py
python3 -m figures.figure_engine.validators.figure_lint ...
python3 finalize_provenance.py
```
