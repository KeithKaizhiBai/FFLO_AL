# Float layout comparison

## Candidate A: preferred interleaving

Source order places both scientific paragraphs before each full-width figure.

```yaml
page_count: 8
reference_pages:
- 2
- 2
- 2
- 2
figure_pages:
- 3
- 4
- 5
- 6
page_distances:
- 1
- 2
- 3
- 4
max_consecutive_figure_pages: 4
all_figures_after_first_discussion: true
float_warnings: 4
overfull_boxes: 0
rendering_mode: full-width figure*
minimum_artwork_scale: 0.96
readability_status: PASS
pdf: /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/tmp/pdfs/scientific_depth_v5/layouts/A/layout_A.pdf
```

## Candidate B: REVTeX-safe fallback

Source order places Fig. 1 full width and tests Figs. 2--4 as ordinary
single-column floats after the first substantive paragraph.  This improves
placement freedom but is acceptable only if the frozen artwork remains legible.

```yaml
page_count: 5
reference_pages:
- 2
- 2
- 2
- 2
figure_pages:
- 3
- 3
- 3
- 4
page_distances:
- 1
- 1
- 1
- 2
max_consecutive_figure_pages: 2
all_figures_after_first_discussion: true
float_warnings: 3
overfull_boxes: 0
rendering_mode: Fig. 1 full width; Figs. 2--4 single column
minimum_artwork_scale: 0.47
readability_status: 'FAIL: frozen 178-mm assets are below their registered print width'
pdf: /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/tmp/pdfs/scientific_depth_v5/layouts/B/layout_B.pdf
```

## Selection

Selected candidate: **A**.  Selection minimizes LaTeX float/box
warnings while treating registered print readability as a hard constraint,
then compares reference distance, consecutive figure pages, and total pages.
Candidate A uses legal `figure*` placement; Candidate B is a legal mixed
`figure*`/`figure` trial.  Neither uses `[H]`, negative spacing, cropping, or
forced page breaks.  The single-column trial scales the registered 178-mm artwork to roughly 47 percent
and therefore fails the figure-engine print-width/readability contract.  The
selected full-width candidate preserves readability but REVTeX queues the four
large floats on consecutive pages.  This is an explicit remaining limitation,
not an automatic interleaving pass; resolving it requires a human-authorized
single-column figure derivative or a different journal layout.
