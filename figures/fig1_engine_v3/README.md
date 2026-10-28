# Figure 1 engine v3 — visual-narrative candidate

This independent review candidate is authorized by the approved
`GATE_FIG1_V3_PROPOSAL`. It preserves the v2 canvas, eight-card architecture,
scientific claims, physical/topological definitions, and Figure Engine rules.
No physics is recomputed.

The only scientific-asset rebuild is the central panel: its 3D schematic
surface is tightly cropped before placement so the real surface, not merely its
parent card or declared plot bbox, satisfies the scientific-content fill-ratio
contract. BdG, observation, and peripheral assets remain frozen inputs.

The build produces an SVG/PDF/PNG review candidate, validation reports, and
provenance evidence. Automated validation cannot pass:

```text
GATE_FIG1_ENGINE_V3_FINAL_APPROVAL = HUMAN_WAIT
```

Figures 2--4 and manuscript LaTeX are outside this gate.

## Rebuild order

```bash
MPLCONFIGDIR=/tmp/mplconfig python3 build_assets_v3.py
python3 build_fig1_engine_v3.py
node measure_rendered_geometry.mjs
node export_fig1_engine_v3.mjs
python3 normalize_pdf_page.py
python3 render_print_previews.py
python3 build_saliency_preview.py
python3 validate_fig1_engine_v3.py
python3 -m figures.figure_engine.validators.figure_lint ...
python3 finalize_provenance.py
```

Use the bundled PDF Python runtime for `normalize_pdf_page.py` when the system
Python does not provide `pypdf`.
